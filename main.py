# -*- coding: utf-8 -*-
"""
Detect & Track → Enrich → Save (images only; no video export)

실행:
    python detect.py   # 파일 실행 시 즉시 수행

변경 요약:
- 슬라이서에서만 NMS(NON_MAX_SUPPRESSION) 적용, 후단 NMS(특히 GPU NMS) 전부 제거
- retina_masks=False 유지(저해상 마스크) + 마스크 폴리곤 저장(mask_polygon)
- detections.video 블록 제거(중복 제거)
- tracks: time_kst 완전 제거 → 영상 재생시간 기반 time_s({start,end,duration_s})만 기록
- path_1hz에서 t_kst 제거(원하면 t만 유지 가능하지만 여기선 path_1hz 자체 제거)
- speed_avg: 기존 세계좌표(E/N) 기반 평균 속도 계산 유지(항공체 속도 보정 취지 반영)
- 1초 미만 트랙 제외
- class_counts는 실제로 기록된(1초 이상) track만 기준으로 재계산
- QA 관련 항목 생성/주입 없음(메인에서 flight_metrics의 qa도 제거)

주의:
- 마스크는 전체 바이너리 저장 대신 폴리곤(근사) 좌표를 저장
"""

from __future__ import annotations
import json
import cv2
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, OrderedDict
import sys

# ========= 추가: BoxMOT BotSort =========
from boxmot import BotSort

# ========= 고정 경로 / 옵션 =========
SHOW_WINDOW_DEFAULT = True

SOURCE_DIR        = Path("source")
OUTPUT_IMAGE_DIR  = Path("result_images")
META_DIR          = Path("metadata")
DATA_DIR          = Path("data")
CAM_CONFIG_PATH   = Path("cam-config.json")
RUN_MAIN_FIRST    = True
TRACK_ID_STATE_PATH = META_DIR / "_object_id_state.json"

# 초당 저장할 스냅샷 수 (예: 1 → 초당 1장, 2 → 초당 2장)
SNAPSHOT_FREQ     = 2
FREQ              = 2

# ========= 탐지/슬라이싱 공통 설정 ========
# 타일 고정 크기 (픽셀 단위, (w, h))
TILE_WH        = (640, 640)
OVERLAP_RATIO  = (0.10, 0.10)       # (w, h) 비율
IMG_SIZE_MIN   = 640                # 타일이 이보다 작으면 키워서 넣음
CONF_THRES     = 0.35               # 민감도 (tiny ↑)
MODEL_PATH     = "./best_640_x.engine"     # .pt / .engine 모두 지원(ultralytics auto-backend)

# ========= 기본(단색) 어노테이터(이미지 단독 탐지 시 사용) =========
RED = sv.Color(255, 0, 0)
MASK_ANN = sv.MaskAnnotator(color=RED, opacity=0.5)
POLY_ANN = sv.PolygonAnnotator(color=RED, thickness=1)

# ========= 디렉토리 =========
for d in [SOURCE_DIR, OUTPUT_IMAGE_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========= 프로젝트 유틸 =========
from core import parse_tabular_file

# ========= 작은 유틸 =========
def _round_or_none(v, n):
    try:
        return round(float(v), n)
    except Exception:
        return None

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _write_json(p: Path, obj: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _median(xs: List[Optional[float]]):
    ys = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not ys:
        return None
    ys.sort()
    n = len(ys); i = n // 2
    return ys[i] if n % 2 == 1 else 0.5 * (ys[i-1] + ys[i])

def _load_next_track_id() -> int:
    if not TRACK_ID_STATE_PATH.exists():
        return 1
    try:
        data = json.loads(TRACK_ID_STATE_PATH.read_text(encoding="utf-8"))
        val = int(data.get("next_id", 1))
        if val < 1:
            return 1
        return val
    except Exception:
        return 1

def _save_next_track_id(next_id: int) -> None:
    value = int(next_id) if isinstance(next_id, (int, float)) else 1
    if value < 1:
        value = 1
    payload = {
        "next_id": int(value),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(TRACK_ID_STATE_PATH, payload)


def _normalize_angle_deg(val: Any) -> Optional[float]:
    try:
        ang = float(val)
    except Exception:
        return None
    if not math.isfinite(ang):
        return None
    ang = ang % 360.0
    if ang < 0:
        ang += 360.0
    return ang


def _resolve_true_north_cw_deg(meta: Dict[str, Any], cam_cfg: Optional[Dict[str, Any]]) -> float:
    """Return clockwise angle (degrees) from image up to true north."""

    def _from_dict(d: Optional[Dict[str, Any]], keys: List[str]) -> Optional[float]:
        if d is None:
            return None
        cur: Any = d
        for key in keys[:-1]:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        if not isinstance(cur, dict):
            return None
        return cur.get(keys[-1])

    def _candidate_values() -> List[Any]:
        vals: List[Any] = []
        # Metadata candidates (nested dictionaries)
        nested_paths = [
            ["image_orientation", "true_north_cw_deg"],
            ["image_orientation", "true_north_deg"],
            ["image_orientation", "true_north_offset_deg"],
            ["image_orientation", "true_north_ccw_deg"],
            ["camera", "true_north_cw_deg"],
            ["camera", "true_north_offset_deg"],
            ["camera", "true_north_ccw_deg"],
            ["video", "true_north_cw_deg"],
            ["video", "true_north_offset_deg"],
            ["video", "true_north_ccw_deg"],
            ["detections", "true_north_cw_deg"],
            ["detections", "image_true_north_cw_deg"],
        ]
        for path in nested_paths:
            val = _from_dict(meta, path)
            if val is not None:
                vals.append((path[-1], val))
        if isinstance(meta, dict):
            for key in [
                "image_true_north_cw_deg",
                "image_true_north_deg",
                "true_north_cw_deg",
                "true_north_offset_deg",
                "true_north_ccw_deg",
            ]:
                if key in meta:
                    vals.append((key, meta.get(key)))
        if isinstance(cam_cfg, dict):
            for key in [
                "image_true_north_cw_deg",
                "image_true_north_deg",
                "true_north_cw_deg",
                "true_north_offset_deg",
                "image_north_cw_deg",
                "north_cw_deg",
                "image_orientation_cw_deg",
                "orientation_cw_deg",
                "true_north_ccw_deg",
                "image_true_north_ccw_deg",
            ]:
                if key in cam_cfg:
                    vals.append((key, cam_cfg.get(key)))
        return vals

    for key_name, raw in _candidate_values():
        if raw is None:
            continue
        ang = None
        if "ccw" in key_name.lower():
            val = _normalize_angle_deg(raw)
            if val is not None:
                ang = (-val) % 360.0
        else:
            ang = _normalize_angle_deg(raw)
        if ang is not None:
            return ang
    return 0.0

# ========= 카메라 설정 =========
def load_camera_config(cfg_path: Optional[Path]) -> Dict[str, Any]:
    if not cfg_path:
        return {}
    if not cfg_path.exists() or not cfg_path.is_file():
        print(f"[WARN] camera config not found: {cfg_path}", file=sys.stderr)
        return {}
    if cfg_path.suffix.lower() == ".json":
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] bad camera json: {e}", file=sys.stderr)
            return {}
    # key=value .txt
    cfg: Dict[str, Any] = {}
    try:
        for line in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = [x.strip() for x in line.split("=", 1)]
            try:
                if "." in v: cfg[k] = float(v)
                else:        cfg[k] = int(v)
            except Exception:
                lv = v.lower()
                if lv in ("true", "false"): cfg[k] = (lv == "true")
                else:                       cfg[k] = v
        return cfg
    except Exception as e:
        print(f"[WARN] bad camera txt: {e}", file=sys.stderr)
        return {}

# ========= 각도/지오 =========
def _bearing_deg(lat1, lon1, lat2, lon2) -> Optional[float]:
    try:
        if None in (lat1, lon1, lat2, lon2): return None
        phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
        dlam = math.radians(float(lon2) - float(lon1))
        y = math.sin(dlam) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
        th = math.degrees(math.atan2(y, x))
        return (th + 360.0) % 360.0
    except Exception:
        return None

def _circular_mean_deg(vals: List[Optional[float]]) -> Optional[float]:
    xs = []
    for v in vals:
        if v is None: continue
        try:
            a = math.radians(float(v))
            if math.isfinite(a): xs.append(a)
        except Exception:
            pass
    if not xs: return None
    C = sum(math.cos(a) for a in xs)
    S = sum(math.sin(a) for a in xs)
    ang = math.degrees(math.atan2(S, C))
    return (ang + 360.0) % 360.0

def meters_per_deg_lat(lat_deg: float) -> float:
    lat = math.radians(lat_deg)
    return 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat)

def meters_per_deg_lon(lat_deg: float) -> float:
    lat = math.radians(lat_deg)
    return 111412.84*math.cos(lat) - 93.5*math.cos(3*lat)

# ========= 공통 =========
def cname_of(names, cid: int) -> str:
    try:
        if isinstance(names, dict): return names.get(cid, str(cid))
        if isinstance(names, (list, tuple)) and 0 <= cid < len(names): return names[cid]
    except Exception:
        pass
    return str(cid)

def make_slicer(model: YOLO) -> sv.InferenceSlicer:
    """
    고정 타일 크기(TILE_WH) 기반 슬라이서 구성.
    - overlap_wh: TILE_WH * OVERLAP_RATIO
    - 중복 제거: NON_MAX_SUPPRESSION (슬라이서에서만 NMS 수행)
    - retina_masks=False (저해상 마스크 생성)
    """
    def slice_callback(tile_img):
        th, tw = tile_img.shape[:2]
        imgsz = max(IMG_SIZE_MIN, th, tw)  # 타일 축소 금지, 최소 IMG_SIZE_MIN 보장
        res = model(
            tile_img,
            imgsz=imgsz,
            verbose=False,
            retina_masks=False,
            conf=CONF_THRES
        )[0]
        return sv.Detections.from_ultralytics(res)

    slicer = sv.InferenceSlicer(
        callback=slice_callback,
        slice_wh=TILE_WH,
        overlap_ratio_wh=OVERLAP_RATIO,
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,  # ← 슬라이서에서 NMS
        thread_workers=4
    )
    return slicer

# ========= 보조: IoU/마스크→폴리곤/ID→색 =========
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    xa1, ya1, xa2, ya2 = np.split(boxes_a, 4, axis=1)
    xb1, yb1, xb2, yb2 = np.split(boxes_b, 4, axis=1)
    inter_x1 = np.maximum(xa1, xb1.T)
    inter_y1 = np.maximum(ya1, yb1.T)
    inter_x2 = np.minimum(xa2, xb2.T)
    inter_y2 = np.minimum(ya2, yb2.T)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b.T - inter_area
    iou = np.where(union > 0, inter_area / union, 0.0)
    return iou.astype(np.float32)

def masks_to_polygons(mask: np.ndarray):
    """
    mask: (H,W) bool/uint8 → 가장 큰 컨투어 근사 다각형 반환
    """
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)  # 0.005~0.02 조절 가능
    approx = approx.reshape(-1, 2).astype(np.float32)
    if approx.shape[0] < 3:
        return None
    return approx

def color_from_track_id(track_id: int) -> sv.Color:
    phi = 0.6180339887498949  # golden ratio
    h = int(((track_id * phi) % 1.0) * 180)  # 0~179 (OpenCV HSV)
    sv_sets = [(230, 255), (200, 230), (180, 255)]
    s, v = sv_sets[track_id % len(sv_sets)]
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return sv.Color(int(bgr[0]), int(bgr[1]), int(bgr[2]))

# ========= Video =========
def run_detection_for_video(
    vpath: Path,
    model: YOLO,
    start_track_id: int = 1,
    show_window: bool = SHOW_WINDOW_DEFAULT,
    snap_freq: int = SNAPSHOT_FREQ,
) -> Tuple[List[Dict[str, Any]], sv.VideoInfo, Dict[int, List[str]], int]:
    fname = vpath.name
    stem  = vpath.stem

    out_img_dir = OUTPUT_IMAGE_DIR / stem
    out_img_dir.mkdir(parents=True, exist_ok=True)

    info0 = sv.VideoInfo.from_video_path(str(vpath))
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or info0.fps or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or info0.total_frames or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or info0.width or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or info0.height or 0)
    cap.release()

    info = sv.VideoInfo(width, height, fps, total_frames)
    slicer = make_slicer(model)

    # ---- BotSort 트래커 초기화 (ReID 끔: tiny object 친화) ----
    tracker = BotSort(
        reid_weights=None,
        device='cuda:0',    # 또는 'cpu'
        half=True,
        with_reid=False,
        track_high_thresh=CONF_THRES,
        new_track_thresh=0.5,
        track_buffer=int(fps),
        match_thresh=0.75
    )

    # ID별 TraceAnnotator 캐시
    trace_by_id: Dict[int, sv.TraceAnnotator] = {}
    label_by_id: Dict[int, sv.LabelAnnotator] = {}
    display_id_by_track: Dict[int, int] = {}
    try:
        next_display_id = int(start_track_id)
    except Exception:
        next_display_id = 1
    if next_display_id < 1:
        next_display_id = 1

    # 트랙 요약 집계용(프레임 단위)
    track_db: Dict[int, Dict[str, Any]] = {}
    det_rows: List[Dict[str, Any]] = []
    names = getattr(model, "names", None)

    # 프레임 스냅샷: snap_freq(초당 횟수) 단위로 저장
    id_img_count: Dict[int, int] = defaultdict(int)
    snapshots_by_track: Dict[int, List[str]] = defaultdict(list)
    snapshot_files_by_track: Dict[int, List[Path]] = defaultdict(list)
    last_bucket_saved_by_track: Dict[int, int] = {}

    def bucket_to_hhmmss(bucket_idx: int) -> Tuple[int, str]:
        sec_idx = bucket_idx // max(1, snap_freq)
        hh = int(sec_idx // 3600)
        mm = int((sec_idx % 3600) // 60)
        ss = int(sec_idx % 60)
        return sec_idx, f"{hh:02d}{mm:02d}{ss:02d}"

    def write_snapshot(tid: int, bucket_idx: int, image: np.ndarray):
        if image is None:
            return
        _, hhmmss = bucket_to_hhmmss(bucket_idx)
        id_img_count[tid] += 1
        fname_img = f"{stem}_{hhmmss}_ID{tid}_{id_img_count[tid]}.jpg"
        full_path = out_img_dir / fname_img
        success = cv2.imwrite(str(full_path), image)
        if not success:
            print(f"[WARN] failed to write snapshot for track {tid} at bucket {bucket_idx}", file=sys.stderr)
            return
        rel_path = (Path(stem) / fname_img).as_posix()
        snapshots_by_track[tid].append(rel_path)
        snapshot_files_by_track[tid].append(full_path)
        last_bucket_saved_by_track[tid] = bucket_idx

    def maybe_save_snapshot_for_tracks(bucket_idx: Optional[int], image: Optional[np.ndarray], track_ids: Set[int]):
        if bucket_idx is None or image is None or not track_ids:
            return
        for tid in sorted(track_ids):
            last_saved = last_bucket_saved_by_track.get(tid)
            if last_saved is not None and bucket_idx <= last_saved:
                continue
            write_snapshot(tid, bucket_idx, image)

    def track_meets_one_sec(display_tid: int) -> bool:
        entry = track_db.get(display_tid)
        if not entry:
            return False
        fps_val = float(info.fps) if info.fps else None
        if fps_val is None or fps_val <= 0 or not math.isfinite(fps_val):
            return False

        def _as_int(val: Any) -> Optional[int]:
            try:
                if val is None:
                    return None
                return int(math.floor(float(val) + 1e-6))
            except Exception:
                return None

        frame_candidates: List[int] = []

        frame_count = _as_int(entry.get("frames"))
        if frame_count and frame_count > 0:
            frame_candidates.append(frame_count)

        first = _as_int(entry.get("first_frame"))
        last = _as_int(entry.get("last_frame"))
        if first is not None and last is not None and last >= first:
            frame_span = (last - first + 1)
            if frame_span > 0:
                frame_candidates.append(frame_span)

        if not frame_candidates:
            return False

        frames_effective = max(frame_candidates)
        min_frames_required = int(math.floor(fps_val)) + 1
        if min_frames_required <= 0:
            min_frames_required = 1
        return frames_effective >= min_frames_required

    if show_window:
        cv2.startWindowThread()
        cv2.namedWindow("Tracking View", cv2.WINDOW_NORMAL)

    for idx, frame in enumerate(
        tqdm(sv.get_video_frames_generator(str(vpath)), total=info.total_frames, desc=fname), start=0
    ):
        t = idx / max(1.0, info.fps)
        bucket = int(math.floor(t * snap_freq))

        # 슬라이스 추론 (슬라이서에서만 NMS 처리)
        dets: sv.Detections = slicer(frame)
        if len(dets.xyxy) > 0 and dets.class_id is not None:
            dets.class_id = dets.class_id.astype(int)

        frame_vis = frame.copy()
        current_frame_track_ids: Set[int] = set()

        if len(dets.xyxy) > 0:
            # BotSort 입력: (N,6) = [x1,y1,x2,y2,score,cls]
            xyxy   = dets.xyxy.astype(np.float32)
            scores = (dets.confidence if dets.confidence is not None else np.full((len(xyxy),), 0.5)).astype(np.float32)
            cls_id = (dets.class_id   if dets.class_id   is not None else np.zeros((len(xyxy),), int)).astype(np.float32)

            dets_for_tracker = np.hstack([xyxy, scores.reshape(-1,1), cls_id.reshape(-1,1)])
            tracked = tracker.update(dets_for_tracker, frame)

            if tracked is not None and len(tracked) > 0:
                # 컬럼 파싱 (버전별 차이 흡수)
                boxes_t = tracked[:, 0:4]
                ids_t   = tracked[:, 4].astype(int)
                col5 = tracked[:, 5]
                is_score_in_col5 = np.all((col5 >= 0.0) & (col5 <= 1.0))
                if is_score_in_col5:
                    confs_t = tracked[:, 5]
                    clss_t  = tracked[:, 6].astype(int) if tracked.shape[1] > 6 else np.zeros(len(tracked), dtype=int)
                else:
                    clss_t  = tracked[:, 5].astype(int)
                    confs_t = tracked[:, 6] if tracked.shape[1] > 6 else np.ones(len(tracked), dtype=float)

                # 트랙 박스 ↔ 현재 프레임 dets 박스 IoU 매칭 → 마스크/폴리곤 매핑
                matched_masks = [None] * len(boxes_t)
                matched_polys = [None] * len(boxes_t)
                if dets.mask is not None and len(dets.mask) == len(dets.xyxy):
                    iou = iou_matrix(boxes_t, dets.xyxy.astype(np.float32))
                    best_idx = iou.argmax(axis=1) if iou.size > 0 else np.array([], dtype=int)
                    for m in range(len(boxes_t)):
                        j = int(best_idx[m]) if len(best_idx) > m else -1
                        if j >= 0 and iou[m, j] > 0.1:
                            mm = dets.mask[j]
                            matched_masks[m] = mm
                            matched_polys[m] = masks_to_polygons(mm)

                # === 그리기/저장: 트랙 ID별 ===
                for k in range(len(boxes_t)):
                    tracker_tid = int(ids_t[k])
                    display_tid = display_id_by_track.get(tracker_tid)
                    if display_tid is None:
                        display_tid = next_display_id
                        display_id_by_track[tracker_tid] = display_tid
                        next_display_id += 1

                    col   = color_from_track_id(display_tid)
                    box_k = boxes_t[k]
                    cls_k = clss_t[k]
                    conf_k= float(confs_t[k])
                    current_frame_track_ids.add(display_tid)

                    # 시각화
                    if matched_masks[k] is not None:
                        dets_mask = sv.Detections(
                            xyxy=box_k.reshape(1,4).astype(np.float32),
                            class_id=np.array([cls_k], dtype=int),
                            confidence=np.array([conf_k], dtype=float),
                            mask=np.array([matched_masks[k]], dtype=bool),
                        )
                        frame_vis = sv.MaskAnnotator(color=col, opacity=0.35).annotate(scene=frame_vis, detections=dets_mask)

                    poly = matched_polys[k]
                    if poly is not None and len(poly) >= 3:
                        dets_poly = sv.Detections(
                            xyxy=box_k.reshape(1,4).astype(np.float32),
                            class_id=np.array([cls_k], dtype=int),
                            confidence=np.array([conf_k], dtype=float),
                        )
                        dets_poly.data = {"polygon": [poly]}
                        frame_vis = sv.PolygonAnnotator(color=col, thickness=1).annotate(scene=frame_vis, detections=dets_poly)

                    if display_tid not in trace_by_id:
                        trace_by_id[display_tid] = sv.TraceAnnotator(color=col, thickness=1, trace_length=30)
                    dets_trace = sv.Detections(
                        xyxy=box_k.reshape(1,4).astype(np.float32),
                        class_id=np.array([cls_k], dtype=int),
                        confidence=np.array([conf_k], dtype=float),
                    )
                    dets_trace.tracker_id = np.array([display_tid], dtype=int)
                    frame_vis = trace_by_id[display_tid].annotate(scene=frame_vis, detections=dets_trace)

                    if display_tid not in label_by_id:
                        label_by_id[display_tid] = sv.LabelAnnotator(
                            color=col,
                            text_scale=0.7,
                            text_thickness=2,
                            text_padding=1,
                        )
                    frame_vis = label_by_id[display_tid].annotate(
                        scene=frame_vis,
                        detections=dets_trace,
                        labels=[f"ID:{int(display_tid)}"]
                    )

                    # ==== 메타데이터 누적 (bbox + mask_polygon) ====
                    x1, y1, x2, y2 = box_k.tolist()
                    w = x2 - x1; h = y2 - y1
                    cx = x1 + w/2.0; cy = y1 + h/2.0
                    if matched_masks[k] is not None:
                        area_px = float(matched_masks[k].sum())
                    else:
                        area_px = float(w * h)
                    cname = cname_of(names, int(cls_k))
                    mask_poly = matched_polys[k].astype(float).tolist() if matched_polys[k] is not None else None

                    det_rows.append({
                        "track_id": int(display_tid),
                        "frame": idx,
                        "t_s": round(idx / max(1.0, info.fps), 6),
                        "label": cname,
                        "conf": float(conf_k),
                        "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
                        "area_px": area_px,
                        "major_axis_px": float(max(w, h)),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "mask_polygon": mask_poly
                    })

                    # 트랙 요약 집계(라벨/프레임 수)
                    entry = track_db.get(display_tid)
                    if entry is None:
                        track_db[display_tid] = dict(
                            class_name=cname,
                            frames=1,
                            first_frame=idx,
                            last_frame=idx
                        )
                    else:
                        entry["frames"] += 1
                        entry["last_frame"] = idx

        if current_frame_track_ids:
            maybe_save_snapshot_for_tracks(bucket, frame_vis, current_frame_track_ids)

        if show_window:
            cv2.imshow("Tracking View", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    valid_track_ids = {tid for tid in track_db if track_meets_one_sec(tid)}
    if valid_track_ids:
        det_rows = [row for row in det_rows if row.get("track_id") in valid_track_ids]
    else:
        det_rows = []
    for tid, files in list(snapshot_files_by_track.items()):
        if tid in valid_track_ids:
            continue
        for fp in files:
            try:
                fp.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[WARN] failed to remove snapshot {fp}: {exc}", file=sys.stderr)
        snapshots_by_track.pop(tid, None)

    snapshots_filtered = {tid: paths for tid, paths in snapshots_by_track.items() if tid in valid_track_ids and paths}

    if show_window:
        cv2.destroyAllWindows()

    # class_counts는 build_and_inject_tracks 안에서 "1초 이상" 트랙 기준으로 재계산
    return det_rows, info, snapshots_filtered, next_display_id

# ========== Image (slice-only) ==========
def run_detection_for_image(ipath: Path, model: YOLO):
    img = cv2.imread(str(ipath))
    if img is None:
        print(f"[WARN] cannot read image: {ipath}")
        return None

    h, w = img.shape[:2]
    stem = ipath.stem
    out_img = OUTPUT_IMAGE_DIR / f"{stem}_annotated.jpg"
    out_img.parent.mkdir(parents=True, exist_ok=True)

    slicer = make_slicer(model)
    dets = slicer(img)
    if len(dets.xyxy) > 0 and dets.class_id is not None:
        dets.class_id = dets.class_id.astype(int)

    frame_vis = img.copy()
    if len(dets.xyxy) > 0:
        if dets.mask is not None and len(dets.mask) > 0:
            frame_vis = MASK_ANN.annotate(scene=frame_vis, detections=dets)
            frame_vis = POLY_ANN.annotate(scene=frame_vis, detections=dets)

    cv2.imwrite(str(out_img), frame_vis)

    names = getattr(model, "names", None)

    def _area_i(i):
        if dets.mask is not None and len(dets.mask) > i and dets.mask[i] is not None:
            return int(dets.mask[i].sum())
        xy = dets.xyxy[i]
        return int((xy[2]-xy[0])*(xy[3]-xy[1]))

    return {
        "width": w,
        "height": h,
        "detections": [
            {
                "label": (
                    names.get(int(dets.class_id[i]), str(int(dets.class_id[i])))
                    if isinstance(names, dict) and dets.class_id is not None else
                    (names[int(dets.class_id[i])] if isinstance(names, (list, tuple)) and dets.class_id is not None
                     else str(int(dets.class_id[i]) if dets.class_id is not None else -1))
                ),
                "conf": float(dets.confidence[i]) if dets.confidence is not None else 1.0,
                "bbox_xyxy": dets.xyxy[i].astype(float).tolist(),
                "area_px": _area_i(i)
            }
            for i in range(len(dets))
        ]
    }

# ========= 보간/트랙 주입 (bbox + mask_polygon 저장, KST 제거, 영상시간만 기록) =========
def _interp_series_to_freq(paths: List[Path], duration_s: Optional[float], freq: int) -> Dict[int, Dict[str, Optional[float]]]:
    raw_t, raw_lat, raw_lon, raw_alt = [], [], [], []
    raw_clock_sod: List[Optional[float]] = []
    raw_clock_date: List[Optional[str]] = []
    raw_clock_time: List[Optional[str]] = []
    raw_clock_iso: List[Optional[str]] = []
    for p in paths:
        series = parse_tabular_file(p)
        lat = series.get("lat") or []
        lon = series.get("lon") or []
        alt = series.get("alt_m") or []
        t_rel = series.get("t_rel_s") or []
        t_clk = series.get("t_clock_sod") or []
        c_date = series.get("clock_date") or []
        c_time = series.get("clock_time") or []
        c_iso  = series.get("clock_iso") or []
        n = max(len(lat), len(lon), len(alt), len(t_rel), len(t_clk), len(c_date), len(c_time), len(c_iso))
        for i in range(n):
            la = lat[i] if i < len(lat) else None
            lo = lon[i] if i < len(lon) else None
            al = alt[i] if i < len(alt) else None
            tr = t_rel[i] if i < len(t_rel) else None
            tc = t_clk[i] if i < len(t_clk) else None
            cd = c_date[i] if i < len(c_date) else None
            ct = c_time[i] if i < len(c_time) else None
            ci = c_iso[i] if i < len(c_iso) else None
            t = None
            if tr is not None:
                try:
                    t = float(tr)
                except Exception:
                    t = None
            if t is None and tc is not None:
                try:
                    t = float(tc)
                except Exception:
                    t = None
            if t is None:
                continue
            raw_t.append(t); raw_lat.append(la); raw_lon.append(lo); raw_alt.append(al)
            raw_clock_sod.append(tc)
            raw_clock_date.append(cd if cd is None or isinstance(cd, str) else str(cd))
            raw_clock_time.append(ct if ct is None or isinstance(ct, str) else str(ct))
            raw_clock_iso.append(ci if ci is None or isinstance(ci, str) else str(ci))
    if not raw_t:
        return {}
    idx = list(range(len(raw_t)))
    idx.sort(key=lambda i: raw_t[i])
    raw_t   = [raw_t[i]   for i in idx]
    raw_lat = [raw_lat[i] for i in idx]
    raw_lon = [raw_lon[i] for i in idx]
    raw_alt = [raw_alt[i] for i in idx]
    raw_clock_sod  = [raw_clock_sod[i]  for i in idx]
    raw_clock_date = [raw_clock_date[i] for i in idx]
    raw_clock_time = [raw_clock_time[i] for i in idx]
    raw_clock_iso  = [raw_clock_iso[i]  for i in idx]
    t_min = int(math.floor(raw_t[0] * freq))
    t_max_source = raw_t[-1] if duration_s is None else duration_s
    t_max = int(math.floor(t_max_source * freq))
    out: Dict[int, Dict[str, Optional[float]]] = {}
    j = 0
    for idx_t in range(t_min, t_max + 1):
        ts = idx_t / freq
        while j + 1 < len(raw_t) and raw_t[j + 1] < ts:
            j += 1
        if j + 1 >= len(raw_t):
            la = raw_lat[-1]; lo = raw_lon[-1]; al = raw_alt[-1]
            clk = raw_clock_sod[-1] if raw_clock_sod else None
            cdate = raw_clock_date[-1] if raw_clock_date else None
            ctime = raw_clock_time[-1] if raw_clock_time else None
            ciso  = raw_clock_iso[-1] if raw_clock_iso else None
        else:
            t0, t1 = raw_t[j], raw_t[j + 1]
            a = (ts - t0) / (t1 - t0) if (t1 > t0) else 0.0
            def lerp(v0, v1):
                if v0 is None and v1 is None: return None
                if v0 is None: return v1
                if v1 is None: return v0
                try: return (1 - a) * float(v0) + a * float(v1)
                except Exception: return v0
            la = lerp(raw_lat[j], raw_lat[j + 1])
            lo = lerp(raw_lon[j], raw_lon[j + 1])
            al = lerp(raw_alt[j], raw_alt[j + 1])
            clk0 = raw_clock_sod[j] if j < len(raw_clock_sod) else None
            clk1 = raw_clock_sod[j + 1] if (j + 1) < len(raw_clock_sod) else None
            clk = lerp(clk0, clk1) if (clk0 is not None or clk1 is not None) else (clk0 if clk0 is not None else clk1)
            def pick(seq: List[Optional[str]]) -> Optional[str]:
                v0 = seq[j] if j < len(seq) else None
                if v0 not in (None, ""):
                    return v0
                v1 = seq[j + 1] if (j + 1) < len(seq) else None
                if v1 not in (None, ""):
                    return v1
                return None
            cdate = pick(raw_clock_date)
            ctime = pick(raw_clock_time)
            ciso  = pick(raw_clock_iso)
        out[idx_t] = {
            "lat": None if la is None else round(float(la), 8),
            "lon": None if lo is None else round(float(lo), 8),
            "alt": None if al is None else round(float(al), 4),
            "clock_sod": None if clk is None else float(clk),
            "clock_date": cdate,
            "clock_time": ctime,
            "clock_iso": ciso,
        }
    return out

def build_and_inject_tracks(meta_path: Path,
                            det_rows: List[Dict[str, Any]],
                            fps: Optional[float],
                            width: Optional[int],
                            height: Optional[int],
                            duration_s: Optional[float],
                            snapshots_by_track: Optional[Dict[int, List[str]]] = None):
    """
    요구사항:
      - 항공체 이동 보정(플랫폼 보정) 후 '순수 객체'의 평균 속도/평균 방위 계산
      - 트랙별 bbox/mask/크기 정보는 '1회만' 기록(트랙 중간 시점 기준)
      - 트랙별 이동경로 path_{FREQ}hz는 '객체 좌표(위도/경도)만' (1/FREQ초) 간격으로 기록
      - flight_metrics.series는 유지
    """
    # ---- 설정/로드 ----
    cam_cfg = load_camera_config(CAM_CONFIG_PATH if CAM_CONFIG_PATH.exists() else None)
    vfov_cli = None
    vfov_cfg = cam_cfg.get("vfov_deg") if cam_cfg else None

    if not meta_path.exists():
        print(f"[WARN] metadata not found: {meta_path}")
        return
    meta = _read_json(meta_path)

    north_cw_deg = _resolve_true_north_cw_deg(meta, cam_cfg)
    theta = math.radians(north_cw_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    # Image basis (up/right) expressed in world (east/north) coordinates
    u_e = -sin_t   # east component of image-up vector
    u_n = cos_t    # north component of image-up vector
    r_e = u_n      # east component of image-right vector
    r_n = -u_e     # north component of image-right vector

    snap_map: Dict[str, List[str]] = {}
    if snapshots_by_track:
        for key, values in snapshots_by_track.items():
            if values is None:
                continue
            try:
                key_str = str(int(key))
            except Exception:
                key_str = str(key)
            cleaned: List[str] = []
            for val in values:
                if val is None:
                    continue
                text = str(val).strip()
                if not text:
                    continue
                text = text.replace("\\", "/")
                parts = [p for p in text.split("/") if p not in ("", ".", "..")] 
                if not parts:
                    continue
                cleaned.append("/".join(parts))
            if cleaned:
                snap_map[key_str] = cleaned

    width  = int(width) if width else None
    height = int(height) if height else None
    fps_m  = float(fps) if fps else None
    dur_s  = float(duration_s) if duration_s else None

    # ---- track_id별 그룹 ----
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in det_rows:
        tid = d.get("track_id")
        if tid is None:
            continue
        t_s = d.get("t_s")
        if t_s is None and d.get("frame") is not None and fps_m:
            t_s = float(d.get("frame"))/fps_m
        if t_s is None:
            continue
        dd = dict(d); dd["t_s"] = float(t_s)
        groups[str(int(tid))].append(dd)
    for k in groups:
        groups[k].sort(key=lambda r: float(r.get("t_s", 0.0)))

    # ---- 항공기 1Hz 경로 로드 (플랫폼 보정에 필요) ----
    stem = meta_path.stem
    supp_paths: List[Path] = []
    if DATA_DIR.exists():
        for p in DATA_DIR.iterdir():
            if p.is_file() and p.stem.startswith(stem) and p.suffix.lower() in (
                ".smi",".srt",".ass",".ssa",".vtt",".csv",".txt",".log",".xlsx",".xls"
            ):
                supp_paths.append(p)
    ts_fhz = _interp_series_to_freq(supp_paths, dur_s, FREQ)
    # 보간 결과가 FREQ(기본 2 Hz) 간격인지 검증
    if ts_fhz:
        ks = sorted(ts_fhz.keys())
        if any((b - a) != 1 for a, b in zip(ks, ks[1:])):
            print(f"[WARN] interpolated path not {FREQ} Hz", file=sys.stderr)

    # 시야각(수직) — GSD 계산에 필요
    vfov_deg = vfov_cli if vfov_cli is not None else (vfov_cfg if vfov_cfg is not None else meta.get("video",{}).get("vfov_deg"))

    # ---- 유틸 ----
    def per_bucket_reduce(track_rows: List[Dict[str, Any]], freq: int) -> List[Dict[str, Any]]:
        """같은 버킷(1/freq 초)에서 conf/area 큰 샘플을 대표로 선택."""
        bucket = {}
        for r in track_rows:
            t = r.get("t_s")
            if t is None:
                continue
            idx = int(math.floor(float(t) * freq))
            prev = bucket.get(idx)
            def score(x):
                c = x.get("conf"); a = x.get("area_px") or 0.0
                return (c if (c is not None) else 0.0, a)
            if (not prev) or score(r) >= score(prev):
                bucket[idx] = r
        rows = []
        for k in sorted(bucket.keys()):
            b = bucket[k]
            rows.append({
                "k": k,
                "t": k / freq,
                "cx": b.get("cx"), "cy": b.get("cy"), "w": b.get("w"), "h": b.get("h"),
                "area_px": b.get("area_px"), "major_axis_px": b.get("major_axis_px"),
                "bbox": b.get("bbox"), "mask_polygon": b.get("mask_polygon"),
                "label": b.get("label")
            })
        return rows

    # 기준 위경도(EN 변환 기준점): 항공기 경로에서 첫 유효값
    ref_lat = None; ref_lon = None
    for k in sorted(ts_fhz.keys()):
        la = ts_fhz[k].get("lat"); lo = ts_fhz[k].get("lon")
        if la is not None and lo is not None:
            try:
                if abs(float(la) - float(lo)) < 1e-7:
                    continue
            except Exception:
                pass
            ref_lat, ref_lon = float(la), float(lo)
            break

    def latlon_to_EN_m(lat, lon, lat0, lon0):
        if None in (lat, lon, lat0, lon0):
            return (None, None)
        m_per_deg_lat = meters_per_deg_lat(lat0)
        m_per_deg_lon = meters_per_deg_lon(lat0)
        dN = (float(lat) - float(lat0)) * m_per_deg_lat
        dE = (float(lon) - float(lon0)) * m_per_deg_lon
        return (dE, dN)

    tracks_out = []
    class_counts: Dict[str, int] = {}

    # ---- 각 트랙 처리 ----
    for tid, rows_t in groups.items():
        frame_indices: List[int] = []
        for r in rows_t:
            fr = r.get("frame")
            if fr is None:
                continue
            try:
                frame_idx = int(math.floor(float(fr) + 1e-6))
            except Exception:
                continue
            frame_indices.append(frame_idx)
        if not frame_indices:
            continue

        frame_indices.sort()
        frame_first = frame_indices[0]
        frame_last = frame_indices[-1]
        frame_span = frame_last - frame_first + 1
        frames_observed = len(set(frame_indices))

        if fps_m is None or fps_m <= 0 or not math.isfinite(fps_m):
            continue
        min_frames_needed = int(math.floor(fps_m)) + 1
        if min_frames_needed <= 0:
            min_frames_needed = 1

        frames_candidates = []
        if frame_span and frame_span > 0:
            frames_candidates.append(frame_span)
        if frames_observed and frames_observed > 0:
            frames_candidates.append(frames_observed)
        if not frames_candidates:
            continue
        frames_effective = max(frames_candidates)
        if frames_effective < min_frames_needed:
            continue

        start_time_exact = frame_first / fps_m
        end_time_exact = (frame_last + 1) / fps_m
        duration_exact = frame_span / fps_m if frame_span and frame_span > 0 else None

        per_sec = per_bucket_reduce(rows_t, FREQ)
        if not per_sec:
            continue

        # 경로(객체 좌표만)
        path_coords: List[Dict[str, Any]] = []   # [{t, lat, lon, clock_*}]
        # 속도/방향 계산용 EN 시퀀스(객체)
        E_obj_seq: List[float] = []
        N_obj_seq: List[float] = []
        T_seq: List[float] = []

        # 크기/스냅샷 대표 산출용
        major_vals: List[float] = []
        gsd_vals: List[float] = []
        major_at_t: Dict[int, Dict[str, Any]] = {}

        for r in per_sec:
            k = int(r["k"])
            t = r["t"]
            plane = ts_fhz.get(k, {})
            plat, plon, palt = plane.get("lat"), plane.get("lon"), plane.get("alt")

            if None in (plat, plon):
                continue
            try:
                if abs(float(plat) - float(plon)) < 1e-7:
                    continue
            except Exception:
                pass

            cx, cy, w_, h_ = r.get("cx"), r.get("cy"), r.get("w"), r.get("h")
            if None in (cx, cy, w_, h_, width, height):
                continue
            rel_x = float(cx)/float(width) - 0.5
            rel_y = float(cy)/float(height) - 0.5

            if vfov_deg is None or palt is None or height is None:
                continue
            gsd_v = 2.0*float(palt)*math.tan(math.radians(float(vfov_deg)/2.0))/float(height)
            if not math.isfinite(gsd_v) or gsd_v <= 0:
                continue
            gsd_vals.append(gsd_v)

            if ref_lat is None or ref_lon is None:
                continue

            E_p, N_p = latlon_to_EN_m(plat, plon, ref_lat, ref_lon)
            if E_p is None or N_p is None:
                continue

            dx_px = rel_x * width
            dy_px = rel_y * height
            dx_m = dx_px * gsd_v
            dy_m = dy_px * gsd_v

            east_m = dx_m * r_e - dy_m * u_e
            north_m = dx_m * r_n - dy_m * u_n

            E_obj = E_p + east_m
            N_obj = N_p + north_m

            dlat = north_m / meters_per_deg_lat(plat)
            dlon = east_m  / meters_per_deg_lon(plat)
            obj_lat = plat + dlat
            obj_lon = plon + dlon

            path_coords.append({
                "t": t,
                "lat": _round_or_none(obj_lat, 8),
                "lon": _round_or_none(obj_lon, 8),
                "clock_sod": plane.get("clock_sod"),
                "clock_date": plane.get("clock_date"),
                "clock_time": plane.get("clock_time"),
                "clock_iso": plane.get("clock_iso"),
            })

            E_obj_seq.append(E_obj); N_obj_seq.append(N_obj)
            T_seq.append(t)

            mj = r.get("major_axis_px")
            if mj is not None:
                try:
                    major_vals.append(float(mj))
                    major_at_t[t] = r
                except Exception:
                    pass

        if not path_coords:
            continue

        seg_bearings: List[float] = []
        for i in range(1, len(T_seq)):
            dt = T_seq[i] - T_seq[i-1]
            if dt <= 0:
                continue
            dE_obj = E_obj_seq[i] - E_obj_seq[i-1]
            dN_obj = N_obj_seq[i] - N_obj_seq[i-1]
            dE = dE_obj
            dN = dN_obj
            d  = math.hypot(dE, dN)
            if d <= 1e-6:
                continue
            try:
                brg = (math.degrees(math.atan2(dE, dN)) + 360.0) % 360.0
            except Exception:
                brg = None
            if brg is not None:
                seg_bearings.append(brg)

        bearing_avg   = _circular_mean_deg(seg_bearings) if seg_bearings else None
        if bearing_avg is None and len(path_coords) >= 2:
            b_dir = _bearing_deg(
                path_coords[0].get("lat"), path_coords[0].get("lon"),
                path_coords[-1].get("lat"), path_coords[-1].get("lon")
            )
            if b_dir is not None:
                bearing_avg = b_dir

        size_cm_median = None
        size_ref_bbox = None
        size_ref_maskpoly = None
        size_ref_t = None
        if major_vals and gsd_vals and major_at_t:
            mj_med  = _median(major_vals)
            gsd_med = _median(gsd_vals)
            if mj_med is not None and gsd_med is not None:
                size_cm_median = _round_or_none(float(mj_med)*float(gsd_med)*100.0, 2)
            t0_eff = path_coords[0]["t"]
            t1_eff = path_coords[-1]["t"]
            t_mid = (t0_eff + t1_eff)/2.0
            candidates = [tt for tt in major_at_t.keys() if t0_eff <= tt <= t1_eff]
            if candidates:
                size_ref_t = min(candidates, key=lambda tt: abs(tt - t_mid))
                raw_r = major_at_t.get(size_ref_t)
                if raw_r:
                    size_ref_bbox = raw_r.get("bbox")
                    size_ref_maskpoly = raw_r.get("mask_polygon")

        start_pt = path_coords[0]
        end_pt   = path_coords[-1]
        start_lat = start_pt.get("lat")
        start_lon = start_pt.get("lon")
        if start_lat is None or start_lon is None:
            continue

        appearance_clock: Dict[str, Any] = {}
        if start_pt.get("clock_date"):
            appearance_clock["date"] = str(start_pt.get("clock_date"))
        if start_pt.get("clock_time"):
            appearance_clock["time"] = str(start_pt.get("clock_time"))
        if start_pt.get("clock_sod") is not None:
            appearance_clock["seconds_of_day"] = _round_or_none(start_pt.get("clock_sod"), 3)
        clock_iso = start_pt.get("clock_iso")
        if (clock_iso is None or clock_iso == "") and appearance_clock.get("date") and appearance_clock.get("time"):
            clock_iso = f"{appearance_clock['date']}T{appearance_clock['time']}"
        if clock_iso:
            appearance_clock["iso"] = str(clock_iso)
        if not appearance_clock:
            appearance_clock = {}

        labels = [r.get("label") for r in per_sec if r.get("label")]
        label_final = max(set(labels), key=labels.count) if labels else None

        snap_key_candidates = []
        try:
            snap_key_candidates.append(str(int(tid)))
        except Exception:
            pass
        snap_key_candidates.append(str(tid))
        snapshots = None
        for skey in snap_key_candidates:
            if skey in snap_map:
                snapshots = snap_map[skey]
                break
        if snapshots:
            seen_snap: Set[str] = set()
            uniq_snapshots: List[str] = []
            for s in snapshots:
                if not s:
                    continue
                if s in seen_snap:
                    continue
                seen_snap.add(s)
                uniq_snapshots.append(s)
            snapshots = uniq_snapshots
        if not snapshots:
            continue

        track_pairs: List[Tuple[str, Any]] = [
            ("track_id", str(tid)),
            ("label", label_final),
            ("time_s", {
                "start": _round_or_none(start_time_exact, 3) if start_time_exact is not None else _round_or_none(start_pt["t"], 3),
                "end": _round_or_none(end_time_exact, 3) if end_time_exact is not None else _round_or_none(end_pt["t"], 3),
                "duration_s": _round_or_none(duration_exact, 3) if duration_exact is not None else _round_or_none(end_pt["t"] - start_pt["t"], 3)
            }),
            ("time_frames", {
                "first": int(frame_first),
                "last": int(frame_last),
                "span": int(frame_span) if frame_span is not None else None,
                "observed": int(frames_observed) if frames_observed is not None else None,
                "effective": int(frames_effective) if frames_effective is not None else None,
                "min_required": int(min_frames_needed) if min_frames_needed is not None else None,
            }),
        ]
        appearance_info: Dict[str, Any] = {"lat": start_lat, "lon": start_lon}
        if appearance_clock:
            appearance_info["clock"] = appearance_clock
        track_pairs.append(("appearance", appearance_info))
        track_pairs.append(("heading_deg", _round_or_none(bearing_avg, 2)))
        track_pairs.append(("size_major_axis_median_cm", size_cm_median))
        if size_ref_bbox or size_ref_maskpoly or size_ref_t is not None:
            track_pairs.append(("size_reference", {
                "t_s": _round_or_none(size_ref_t, 3) if size_ref_t is not None else None,
                "bbox": size_ref_bbox,
                "mask_polygon": size_ref_maskpoly,
            }))
        if snapshots:
            track_pairs.append(("snapshot_images", snapshots))
        track_obj = OrderedDict(track_pairs)
        tracks_out.append(track_obj)

        k = label_final or "unknown"
        class_counts[k] = class_counts.get(k, 0) + 1

    # ---- 주입/저장 ----
    meta.setdefault("detections", {})
    meta["detections"]["image_true_north_cw_deg"] = _round_or_none(north_cw_deg, 6)
    meta["detections"].update({
        "class_counts": class_counts,
        "tracks": tracks_out
    })
    _write_json(meta_path, meta)
    print(f"[OK] augmented: {meta_path.name} with {len(tracks_out)} tracks ({FREQ}Hz object coords)")

# ========= 여기부터 자동 실행 본문 =========

# 1) main.py 먼저
if RUN_MAIN_FIRST:
    try:
        print("[INFO] running meta_process.main() ...")
        from meta_process import main as build_meta_main
        build_meta_main()  # ← 여기서 메타데이터 빌드
    except Exception as e:
        print(f"[ERROR] Failed to run meta_process.main(): {e}")

# 2) 모델 준비 (워밍업 포함: 초기화 경합/오류 방지)
model = YOLO(MODEL_PATH, task='segment')
_dummy = np.zeros((64, 64, 3), np.uint8)
_ = model(_dummy, verbose=False)

# 3) 소스 나열 및 처리 시작
video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

items = sorted([p for p in SOURCE_DIR.iterdir() if p.is_file()])
next_track_id = _load_next_track_id()

for path in items:
    ext = path.suffix.lower()
    stem = path.stem
    meta_path = META_DIR / f"{stem}.json"

    if ext in video_exts:
        det_rows, info, snapshots_by_track, next_track_id = run_detection_for_video(
            path,
            model,
            start_track_id=next_track_id,
            show_window=SHOW_WINDOW_DEFAULT,
        )
        duration_s = (info.total_frames / info.fps) if (info.total_frames and info.fps) else None
        build_and_inject_tracks(
            meta_path=meta_path,
            det_rows=det_rows,
            fps=info.fps,
            width=info.width,
            height=info.height,
            duration_s=duration_s,
            snapshots_by_track=snapshots_by_track,
        )
        _save_next_track_id(next_track_id)
    elif ext in image_exts:
        img_info = run_detection_for_image(path, model)
        # 이미지도 metadata 주입 (tracks=[] 포함)
        if meta_path.exists():
            meta = _read_json(meta_path)
        else:
            meta = {}
        meta.setdefault("detections", {})
        meta["detections"]["tracks"] = []
        if img_info:
            meta["detections"]["image"] = {
                "width": img_info["width"],
                "height": img_info["height"],
                "num_detections": len(img_info["detections"])
            }
        _write_json(meta_path, meta)
        print(f"[OK] injected (image/no-motion): {meta_path.name}")
    else:
        continue

print("[Done] All media processed from source/]")
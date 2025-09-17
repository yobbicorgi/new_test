# -*- coding: utf-8 -*-
"""
Marine Mammal Occurrence Viewer (Streamlit)
- 밝고 고대비 색상(어두운색 제외), 호버 툴팁, 클릭-고정 상세
- 팝업/Details 정돈: Time → Speed → Size → Bearing
- 배지(라벨) 클릭도 Details 갱신 (팝업/좌표 기반 양쪽 처리)
- Legend 영역 고정 + 스크롤
- 경로 파싱 견고(여러 키명 지원) + 시작/종료 좌표 자동 보정 + 경로거리/평균속도 계산
- 실행:
  pip install streamlit folium streamlit-folium pandas
  streamlit run viewer.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
import json, hashlib, re, math, html
from urllib.parse import quote
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium

# =========================
# Tunables
# =========================
MAP_HEIGHT = 860            # 지도 세로(px)
LEFT_COL_RATIO = 0.22       # 좌측 컨트롤 폭
RIGHT_COL_RATIO = 0.78      # 우측(지도 영역) 폭

# 시작 배지 스타일
BADGE_BORDER_PX = 3
BADGE_FONT_PX   = 12
BADGE_PAD_Y     = 6
BADGE_PAD_X     = 12

# Legend 박스 고정 높이
LEGEND_MAX_H    = 160  # px

# =========================
# 경로/프로젝트 경로
# =========================
ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_DIR = ROOT / "metadata"
OUTPUT_IMAGE_DIR = ROOT / "result_images"

# =========================
# 유틸
# =========================
def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def _format_float(v: Optional[float], digits: int = 2, unit: str = "") -> str:
    if v is None:
        return "—"
    try:
        s = f"{float(v):.{digits}f}"
        return f"{s}{unit}" if unit else s
    except Exception:
        return str(v)

def _normalize_rel_image_path(value: Any) -> Optional[str]:
    """결과 이미지 폴더 기준 상대 경로를 '/' 구분자로 정규화."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("\\", "/")
    parts = [p for p in text.split("/") if p not in ("", ".", "..")]
    if not parts:
        return None
    return "/".join(parts)

def _full_path_from_rel(rel_norm: str) -> Optional[Path]:
    """정규화된 상대 경로 → 실제 전체 경로 (파일 존재/루트 제한 확인)."""
    try:
        rel_path = Path(*rel_norm.split("/"))
    except Exception:
        return None
    base_root = OUTPUT_IMAGE_DIR.resolve()
    try:
        full_path = (OUTPUT_IMAGE_DIR / rel_path).resolve()
    except Exception:
        return None
    try:
        full_path.relative_to(base_root)
    except ValueError:
        return None
    if not full_path.exists() or not full_path.is_file():
        return None
    return full_path

def _decimal_to_dms(value: Optional[float], is_lat: bool) -> Optional[str]:
    if value is None or not _is_number(value):
        return None
    val = float(value)
    hemi = "N" if (is_lat and val >= 0) else "S" if is_lat else "E" if val >= 0 else "W"
    abs_val = abs(val)
    deg = int(abs_val)
    minutes_full = (abs_val - deg) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0
    deg_fmt = f"{deg:02d}" if is_lat else f"{deg:03d}"
    return f"{deg_fmt}°{minutes:02d}'{seconds:05.2f}\"{hemi}"

def _normalize_date_string(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    m = re.search(r"(\d{4})[^\d]?(\d{1,2})[^\d]?(\d{1,2})", text)
    if not m:
        return None
    try:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    except Exception:
        return None

def _normalize_time_string(raw: Any, sod_raw: Any) -> Optional[str]:
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            m = re.match(r"^(\d{1,2})(?::(\d{1,2}))?(?::(\d{1,2})(?:[.,]\d+)?)?", text)
            if m:
                try:
                    hour = int(m.group(1))
                    if m.group(2) is not None:
                        minute = int(m.group(2))
                    if m.group(3) is not None:
                        second = int(m.group(3))
                except Exception:
                    hour = minute = second = None
    if hour is None and sod_raw is not None and _is_number(sod_raw):
        try:
            total = float(sod_raw)
            if math.isfinite(total):
                total_int = int(total)
                hour = max(0, min(23, total_int // 3600))
                minute = (total_int % 3600) // 60
                second = total_int % 60
        except Exception:
            hour = None
    if hour is None:
        return None
    if minute is None:
        minute = 0
    if second is None:
        second = 0
    return f"{hour:02d}:{minute:02d}:{second:02d}"

def _sec_to_hhmmss(sec: Optional[float]) -> str:
    """고정 hh:mm:ss 표기 (음수/결측은 대시)"""
    if sec is None or not _is_number(sec):
        return "—"
    total = int(round(float(sec)))
    if total < 0:
        return "—"
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _md5(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

def _hsl_to_hex(h: float, s: float, l: float) -> str:
    s /= 100.0; l /= 100.0
    c = (1 - abs(2*l - 1)) * s
    x = c * (1 - abs((h/60.0) % 2 - 1))
    m = l - c/2
    if   0 <= h < 60:   r,g,b = c,x,0
    elif 60 <= h < 120: r,g,b = x,c,0
    elif 120 <= h < 180:r,g,b = 0,c,x
    elif 180 <= h < 240:r,g,b = 0,x,c
    elif 240 <= h < 300:r,g,b = x,0,c
    else:               r,g,b = c,0,x
    r,g,b = [int(round((v + m)*255)) for v in (r,g,b)]
    return f"#{r:02x}{g:02x}{b:02x}"

def _spread_hue(n: int) -> int:
    x = (n * 1103515245 + 12345) & 0xFFFFFFFF
    phi = 0.61803398875
    frac = ((x / 0xFFFFFFFF) + phi) % 1.0
    return int(frac * 360)

def bright_color_for(file_name: str, obj_id: str) -> str:
    """밝고 고대비 색상 (어두운/검정 배제)"""
    base_h = _spread_hue(_md5(file_name))
    obj_h  = _spread_hue(_md5(f"{file_name}:{obj_id}"))
    h = (base_h * 0.35 + obj_h * 0.65) % 360
    s = 95
    l = 63 + (((_md5(obj_id) % 3) - 1) * 3)  # 60,63,66
    return _hsl_to_hex(h, s, l)

# 지구 대원(하버사인)
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R*c

# =========================
# 데이터 로딩
# =========================
@st.cache_data(show_spinner=False)
def load_jsons(md_dir: str) -> List[Dict[str, Any]]:
    md_path = Path(md_dir)
    files = sorted([p for p in md_path.glob("*.json") if p.is_file()])
    records: List[Dict[str, Any]] = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source_path"] = str(p)
            data["_source_name"] = p.name
            records.append(data)
        except Exception as e:
            st.warning(f"JSON 로드 실패: {p.name} | {e}")
    return records

# =========================
# 경로 파서
# =========================
# =========================
# 카탈로그
# =========================
@st.cache_data(show_spinner=False)
def build_objects(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    catalog: List[Dict[str, Any]] = []
    mapping: Dict[str, List[str]] = {}

    def _coerce_int(val: Any) -> Optional[int]:
        if _is_number(val):
            try:
                return int(math.floor(float(val) + 1e-6))
            except Exception:
                return None
        try:
            return int(str(val))
        except Exception:
            return None

    for rec in records:
        fname = rec.get("_source_name", "unknown.json")
        video_file_name = rec.get("file_name")
        if not video_file_name and isinstance(rec.get("video"), dict):
            video_file_name = rec["video"].get("file_name")
        if not video_file_name:
            video_file_name = Path(fname).stem
        mapping.setdefault(fname, [])
        stem = Path(fname).stem
        fps_meta: Optional[float] = None
        fps_candidates = [
            rec.get("frame_rate"),
            rec.get("video", {}).get("frame_rate") if isinstance(rec.get("video"), dict) else None,
        ]
        for cand in fps_candidates:
            if _is_number(cand):
                try:
                    fps_val = float(cand)
                except Exception:
                    continue
                if math.isfinite(fps_val) and fps_val > 0:
                    fps_meta = fps_val
                    break
        tracks = (rec.get("detections", {}) or {}).get("tracks", []) or []
        for tr in tracks:
            tid   = str(tr.get("track_id", ""))
            label = tr.get("label", "") or ""
            color = bright_color_for(fname, tid)

            t_s   = tr.get("time_s", {}) or {}
            size_cm = tr.get("size_major_axis_median_cm", None)
            bearing = tr.get("bearing_avg_deg", None)

            frames_info_raw = tr.get("time_frames") or {}
            frame_first = _coerce_int(frames_info_raw.get("first"))
            frame_last = _coerce_int(frames_info_raw.get("last"))
            frame_span = _coerce_int(frames_info_raw.get("span"))
            frame_observed = _coerce_int(frames_info_raw.get("observed"))
            if frame_span is None and frame_first is not None and frame_last is not None and frame_last >= frame_first:
                frame_span = frame_last - frame_first + 1
            frames_candidates = [x for x in (frame_span, frame_observed) if x is not None and x > 0]
            frames_effective = max(frames_candidates) if frames_candidates else None

            fps_for_calc = fps_meta if (fps_meta is not None and math.isfinite(fps_meta) and fps_meta > 0) else None
            min_frames_required: Optional[int] = None
            if fps_for_calc is not None:
                min_frames_required = int(math.floor(fps_for_calc)) + 1
                if min_frames_required <= 0:
                    min_frames_required = 1

            start_raw = t_s.get("start")
            end_raw = t_s.get("end")
            dur_raw = t_s.get("duration_s")
            start_time_meta = float(start_raw) if _is_number(start_raw) else None
            end_time_meta = float(end_raw) if _is_number(end_raw) else None
            duration_meta: Optional[float] = None
            if _is_number(dur_raw):
                duration_meta = float(dur_raw)
            elif start_time_meta is not None and end_time_meta is not None:
                duration_meta = end_time_meta - start_time_meta

            start_from_frames = None
            end_from_frames = None
            duration_from_frames = None
            if fps_for_calc is not None and frame_first is not None:
                start_from_frames = frame_first / fps_for_calc
            if fps_for_calc is not None and frame_last is not None:
                end_from_frames = (frame_last + 1) / fps_for_calc
            if fps_for_calc is not None and frame_span is not None and frame_span > 0:
                duration_from_frames = frame_span / fps_for_calc

            start_time = start_from_frames if start_from_frames is not None else start_time_meta
            end_time = end_from_frames if end_from_frames is not None else end_time_meta
            duration_sec: Optional[float] = duration_from_frames if duration_from_frames is not None else duration_meta
            if duration_sec is not None and duration_sec < 0:
                duration_sec = None
            if duration_sec is None and fps_for_calc is not None:
                if frame_span is not None and frame_span > 0:
                    duration_sec = frame_span / fps_for_calc
                elif frame_observed is not None and frame_observed > 0:
                    duration_sec = frame_observed / fps_for_calc

            frames_condition_met = False
            if fps_for_calc is not None and frames_effective is not None and min_frames_required is not None:
                if frames_effective < min_frames_required:
                    continue
                frames_condition_met = True

            if not frames_condition_met:
                if duration_sec is None or duration_sec <= 1.0:
                    continue  # 1초 초과 여부 확인 실패 시 제외

            time_frames_info = None
            if any(v is not None for v in (frame_first, frame_last, frame_span, frame_observed)):
                time_frames_info = {
                    "first": frame_first,
                    "last": frame_last,
                    "span": frame_span,
                    "observed": frame_observed,
                    "effective": frames_effective,
                    "min_required": min_frames_required,
                }

            appearance = tr.get("appearance") or {}
            a_lat_raw = appearance.get("lat")
            a_lon_raw = appearance.get("lon")
            a_lat = float(a_lat_raw) if _is_number(a_lat_raw) else None
            a_lon = float(a_lon_raw) if _is_number(a_lon_raw) else None
            if a_lat is None or a_lon is None:
                continue

            clock_info = appearance.get("clock") or {}
            clock_date_raw = clock_info.get("date") or clock_info.get("clock_date")
            clock_time_raw = clock_info.get("time") or clock_info.get("clock_time")
            clock_sod_raw = clock_info.get("seconds_of_day") or clock_info.get("clock_sod")
            clock_iso_raw = clock_info.get("iso") or clock_info.get("clock_iso")

            appearance_date = _normalize_date_string(clock_date_raw)
            appearance_time = _normalize_time_string(clock_time_raw, clock_sod_raw)
            if clock_iso_raw and isinstance(clock_iso_raw, str):
                iso_txt = clock_iso_raw.strip()
                if iso_txt:
                    m_iso = re.match(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:[.,]\d+)?", iso_txt)
                    if m_iso:
                        if appearance_date is None:
                            appearance_date = m_iso.group(1)
                        if appearance_time is None:
                            appearance_time = _normalize_time_string(m_iso.group(2), None)
            appearance_date_display = appearance_date or "Null"
            appearance_time_display = appearance_time or "Null"
            appearance_datetime_display = f"{appearance_date_display} {appearance_time_display}".strip()
            if appearance_datetime_display == "Null Null":
                appearance_datetime_display = "Null"
            appearance_iso = None
            if appearance_date and appearance_time:
                appearance_iso = f"{appearance_date}T{appearance_time}"
            elif isinstance(clock_iso_raw, str) and clock_iso_raw.strip():
                appearance_iso = clock_iso_raw.strip()

            # 첫 저장 이미지 찾기 (지연 로딩: 경로만 저장)
            preview_rel_path = None
            preview_img_name = None
            snapshot_list: List[str] = []
            seen_rel: Set[str] = set()

            snapshots_meta = tr.get("snapshot_images")
            if isinstance(snapshots_meta, (list, tuple)):
                for snap in snapshots_meta:
                    rel_norm = _normalize_rel_image_path(snap)
                    if not rel_norm or rel_norm in seen_rel:
                        continue
                    full_candidate = _full_path_from_rel(rel_norm)
                    if full_candidate is None:
                        continue
                    seen_rel.add(rel_norm)
                    snapshot_list.append(rel_norm)
                    if preview_rel_path is None:
                        preview_rel_path = rel_norm
                        preview_img_name = full_candidate.name

            try:
                img_dir = OUTPUT_IMAGE_DIR / stem
                if img_dir.exists():
                    candidates = sorted(
                        p for p in img_dir.glob(f"{stem}_*_ID{tid}_*")
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    )
                    preferred_choice = None
                    fallback_choice = None
                    for cand in candidates:
                        rel_norm = None
                        try:
                            rel_norm = (cand.relative_to(OUTPUT_IMAGE_DIR)).as_posix()
                        except Exception:
                            rel_norm = _normalize_rel_image_path(str(cand))
                        rel_norm = _normalize_rel_image_path(rel_norm)
                        if not rel_norm:
                            continue
                        if rel_norm not in seen_rel:
                            seen_rel.add(rel_norm)
                            snapshot_list.append(rel_norm)
                        if preferred_choice is None and cand.stem.endswith(f"_ID{tid}_1"):
                            preferred_choice = (rel_norm, cand.name)
                        if fallback_choice is None:
                            fallback_choice = (rel_norm, cand.name)
                    if preview_rel_path is None:
                        chosen = preferred_choice or fallback_choice
                        if chosen:
                            preview_rel_path, preview_img_name = chosen
            except Exception:
                pass

            if preview_rel_path is None:
                if snapshot_list:
                    preview_rel_path = snapshot_list[0]
                    if preview_rel_path and not preview_img_name:
                        preview_img_name = Path(preview_rel_path).name
                else:
                    continue
            if preview_rel_path not in snapshot_list:
                snapshot_list.append(preview_rel_path)
            if not snapshot_list:
                continue

            start = (a_lat, a_lon)
            end = None

            bearing_raw = tr.get("heading_deg") if "heading_deg" in tr else tr.get("bearing_avg_deg")
            bearing = float(bearing_raw) if _is_number(bearing_raw) else None
            size_raw = tr.get("size_major_axis_median_cm")
            size_cm = float(size_raw) if _is_number(size_raw) else None

            coords = [(a_lat, a_lon)]
            lat_dms = _decimal_to_dms(a_lat, True)
            lon_dms = _decimal_to_dms(a_lon, False)

            ui_label = f"{video_file_name} | 객체 ID{tid}" + (f" — {label}" if label else "")
            catalog.append({
                "label_for_ui": ui_label,
                "file": fname,
                "video_file_name": video_file_name,
                "obj_id": tid, "label": label, "color": color,
                "bearing_deg": bearing, "size_cm": size_cm,
                "t_start_s": start_time, "t_end_s": end_time, "t_dur_s": duration_sec,
                "coords": coords, "start": start, "end": end,
                "preview_rel_path": preview_rel_path, "preview_img_name": preview_img_name,
                "snapshot_images": snapshot_list,
                "time_frames": time_frames_info,
                "fps": fps_for_calc,
                "appearance_date": appearance_date,
                "appearance_time": appearance_time,
                "appearance_datetime_display": appearance_datetime_display,
                "appearance_date_display": appearance_date_display,
                "appearance_time_display": appearance_time_display,
                "appearance_iso": appearance_iso,
                "appearance_lat": a_lat,
                "appearance_lon": a_lon,
                "appearance_lat_dms": lat_dms,
                "appearance_lon_dms": lon_dms,
            })
            mapping[fname].append(ui_label)
    return catalog, mapping

# =========================
# 팝업/툴팁/상세
# =========================
def _tooltip_html(t: Dict[str, Any]) -> str:
    start = _sec_to_hhmmss(t["t_start_s"])
    end = _sec_to_hhmmss(t["t_end_s"])
    return (
        f"<b>객체 ID{t['obj_id']}</b>"
        f"<br/>영상: {t.get('video_file_name') or t['file']}"
        f"<br/>영상 시간: {start} ~ {end}"
    )

def _popup_html(t: Dict[str, Any]) -> str:
    start = _sec_to_hhmmss(t["t_start_s"])
    end = _sec_to_hhmmss(t["t_end_s"])
    appearance_date = t.get("appearance_date_display") or "Null"
    appearance_time = t.get("appearance_time_display") or "Null"
    appearance_line = f"{appearance_date} {appearance_time}".strip()
    if appearance_line == "Null Null":
        appearance_line = "Null"
    lat_dms = t.get("appearance_lat_dms") or "—"
    lon_dms = t.get("appearance_lon_dms") or "—"
    size_txt = _format_float(t.get("size_cm"), 1, " cm")
    heading_txt = _format_float(t.get("bearing_deg"), 2, "°")
    video_name = t.get("video_file_name") or t["file"]

    img_line = ""
    preview_rel_raw = t.get("preview_rel_path")
    preview_rel = _normalize_rel_image_path(preview_rel_raw) if preview_rel_raw else None
    if preview_rel:
        title_txt = t.get("preview_img_name") or f"{video_name} | ID {t['obj_id']}"
        query = f"?preview={quote(preview_rel, safe='')}"
        if title_txt:
            query += f"&title={quote(title_txt, safe='')}"
        open_script = (
            "const parentWin=window.parent||window;"
            "const loc=parentWin.location;"
            "const hashPart=loc.hash?loc.hash.split('?')[0]:'';"
            "const base=loc.origin+loc.pathname+hashPart;"
            f"parentWin.open(base+'{query}', '_blank','noopener,noreferrer');return false;"
        )
        onclick_attr = html.escape(open_script, quote=True)
        img_line = (
            "<div style='margin-top:6px;'>"
            "<button type=\"button\" style=\"display:inline-flex;align-items:center;gap:6px;padding:6px 12px;"
            "background:#2563eb;color:#ffffff;border:none;border-radius:6px;font-weight:600;cursor:pointer;\" "
            f"onclick=\"{onclick_attr}\">📷 이미지 보기</button>"
            "</div>"
        )

    # Details 파싱용 숨김 키(파일명|오브젝트ID)
    hidden_key = f"<div style='display:none'>OBJKEY|{t['file']}|{t['obj_id']}</div>"

    return f"""
    <div style="font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; font-size: 13px; line-height:1.5;">
      <div style="font-weight:700; margin-bottom:6px;">객체 <span style="font-family:monospace;">ID{t['obj_id']}</span>{(' — '+t['label']) if t['label'] else ''}</div>
      <div><b>영상</b>: {video_name}</div>
      <div><b>영상 시간</b>: {start} ~ {end}</div>
      <div><b>출현 시간</b>: {appearance_line}</div>
      <div><b>위도</b>: {lat_dms}</div>
      <div><b>경도</b>: {lon_dms}</div>
      <div><b>크기</b>: {size_txt}</div>
      <div><b>방향</b>: {heading_txt}</div>
      {img_line}
      {hidden_key}
    </div>
    """

# =========================
# 지도
# =========================
def build_map(objs: List[Dict[str, Any]], base_tile: str = "Carto Light") -> Optional[folium.Map]:
    coords_all: List[Tuple[float, float]] = []
    for t in objs:
        loc = t.get("start")
        if loc:
            coords_all.append(loc)
    if not coords_all:
        return None

    lat_avg = sum(lat for lat, _ in coords_all) / len(coords_all)
    lon_avg = sum(lon for _, lon in coords_all) / len(coords_all)

    # 밝은 타일
    if base_tile == "OpenStreetMap":
        tile_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        tile_attr = "© OpenStreetMap contributors"
    elif base_tile == "Esri World Imagery":
        tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        tile_attr = "Tiles © Esri"
    else:
        tile_url = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        tile_attr = "© OpenStreetMap contributors © CARTO"

    m = folium.Map(
        location=[lat_avg, lon_avg],
        zoom_start=7,
        control_scale=True,
        tiles=tile_url,
        attr=tile_attr,
    )

    bounds: List[Tuple[float, float]] = []

    for t in objs:
        color = t["color"]
        loc = t.get("start")
        if not loc:
            continue
        tooltip_txt = _tooltip_html(t)
        popup_html  = _popup_html(t)

        folium.CircleMarker(
            location=loc, radius=7, color=color, fill=True, fill_opacity=1.0,
            tooltip=tooltip_txt, popup=folium.Popup(popup_html, max_width=480)
        ).add_to(m)
        bounds.append(loc)

        label_html = f"""
        <div style="
            display:inline-flex; align-items:center; justify-content:center;
            background: rgba(255,255,255,0.98);
            border: {BADGE_BORDER_PX}px solid {color};
            color: {color};
            padding: {BADGE_PAD_Y}px {BADGE_PAD_X}px;
            border-radius: 999px;
            font-weight: 700; font-size: {BADGE_FONT_PX}px; line-height: 1; white-space: nowrap;
            box-shadow: 0 1px 4px rgba(0,0,0,0.25);
        ">
            ID{t['obj_id']}
        </div>"""
        folium.Marker(
            location=loc,
            icon=folium.DivIcon(html=label_html, icon_size=(1,1), icon_anchor=(0,0)),
            tooltip=f"객체 ID{t['obj_id']} (시작)",
            popup=folium.Popup(popup_html, max_width=480),
        ).add_to(m)

    if bounds:
        m.fit_bounds(bounds, padding=(20, 20))
    return m

# =========================
# 내보내기
# =========================
def make_csv(objs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for t in objs:
        video_time = _sec_to_hhmmss(t["t_start_s"])
        if video_time == "—":
            video_time = "Null"
        rows.append({
            "file": t.get("video_file_name") or t["file"],
            "object_id": t["obj_id"],
            "date": t.get("appearance_date") or "Null",
            "time": t.get("appearance_time") or "Null",
            "video_time": video_time,
            "latitude": t.get("appearance_lat"),
            "longitude": t.get("appearance_lon"),
            "latitude_dms": t.get("appearance_lat_dms") or "",
            "longitude_dms": t.get("appearance_lon_dms") or "",
            "size(cm)": t.get("size_cm"),
            "heading": t.get("bearing_deg"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = [
        "file",
        "object_id",
        "date",
        "time",
        "video_time",
        "latitude",
        "longitude",
        "latitude_dms",
        "longitude_dms",
        "size(cm)",
        "heading",
    ]
    return df[cols]

def make_geojson(objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    feats = []
    for t in objs:
        tf = t.get("time_frames") or {}
        loc = t.get("start")
        if not loc:
            continue
        feats.append({
            "type": "Feature",
            "properties": {
                "file": t["file"],
                "object_id": t["obj_id"],
                "label": t["label"],
                "appearance_date": t.get("appearance_date"),
                "appearance_time": t.get("appearance_time"),
                "appearance_display": t.get("appearance_datetime_display"),
                "appearance_iso": t.get("appearance_iso"),
                "appearance_lat": t.get("appearance_lat"),
                "appearance_lon": t.get("appearance_lon"),
                "appearance_lat_dms": t.get("appearance_lat_dms"),
                "appearance_lon_dms": t.get("appearance_lon_dms"),
                "heading_deg": t.get("bearing_deg"),
                "size_cm": t.get("size_cm"),
                "t_start_s": t["t_start_s"],
                "t_end_s": t["t_end_s"],
                "t_dur_s": t["t_dur_s"],
                "time_frames": {
                    "first": tf.get("first"),
                    "last": tf.get("last"),
                    "span": tf.get("span"),
                    "observed": tf.get("observed"),
                    "effective": tf.get("effective"),
                    "threshold": tf.get("min_required"),
                },
                "frame_rate": t.get("fps"),
                "color": t.get("color"),
            },
            "geometry": {"type": "Point", "coordinates": [loc[1], loc[0]]},
        })
    return {"type":"FeatureCollection","features":feats}

# =========================
# 프리뷰 페이지 (이미지 전용)
# =========================
def handle_image_preview_request() -> bool:
    """Handle direct image preview requests triggered via query params."""

    def _get_query_params() -> Any:
        try:
            qp = getattr(st, "query_params")
        except AttributeError:
            return {}
        except Exception:
            return {}
        return qp if qp is not None else {}

    def _extract_values(container: Any, key: str) -> List[str]:
        if container is None:
            return []
        # Streamlit's QueryParams exposes get_all for multi values (1.32+)
        if hasattr(container, "get_all"):
            try:
                values = container.get_all(key)
            except Exception:
                values = None
            else:
                if values is not None:
                    if isinstance(values, (list, tuple, set)):
                        return [str(v) for v in values if v is not None]
                    return [str(values)]
        value = None
        try:
            value = container.get(key)
        except Exception:
            pass
        if value is None:
            try:
                value = container[key]
            except Exception:
                value = None
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value if v is not None]
        return [str(value)]

    params = _get_query_params()
    preview_vals = _extract_values(params, "preview")
    if not preview_vals:
        return False

    rel_value_raw = preview_vals[0]
    title_vals = _extract_values(params, "title")
    title_value = title_vals[0] if title_vals else None

    rel_norm = _normalize_rel_image_path(rel_value_raw)
    if not rel_norm:
        st.error("잘못된 이미지 경로입니다.")
        return True

    try:
        rel_path = Path(*rel_norm.split("/"))
    except Exception:
        st.error("잘못된 이미지 경로입니다.")
        return True

    base_root = OUTPUT_IMAGE_DIR.resolve()
    try:
        full_path = (OUTPUT_IMAGE_DIR / rel_path).resolve()
    except Exception:
        st.error("잘못된 이미지 경로입니다.")
        return True

    try:
        full_path.relative_to(base_root)
    except ValueError:
        st.error("허용되지 않은 이미지 경로입니다.")
        return True

    if not full_path.exists() or not full_path.is_file():
        st.error("이미지 파일을 찾을 수 없습니다.")
        return True

    st.markdown(
        """
        <style>
        body { background-color: #0f172a; color: #e2e8f0; }
        .block-container { padding-top: 2.0rem; padding-bottom: 2.5rem; }
        .block-container h2:first-child { margin-top: 0; }
        .stImage { overflow-x: auto; }
        .stImage img { max-width: none !important; width: auto !important; }
        header, footer, #MainMenu { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_text = title_value or full_path.name
    st.markdown(f"## {title_text}")
    st.image(str(full_path), use_column_width=False)
    st.caption(rel_norm)
    st.caption("새 창을 닫으면 원래 페이지로 돌아갑니다.")
    return True

# =========================
# UI
# =========================
st.set_page_config(page_title="해양 포유류 탐지 뷰어", layout="wide")

if handle_image_preview_request():
    st.stop()

# 상단·하단 여백 보정 (제목 잘림 방지 + 전체 화면에 컴팩트)
st.markdown(f"""
<style>
body {{ background:#f8fafc; }}
.block-container {{ padding-top: 1.0rem; padding-bottom: 0.2rem; }}
.block-container h2:first-child {{ margin-top: 1.1rem; }}
.stMultiSelect [data-baseweb="tag"] {{ max-width: 140px; }}
/* 좌측 패널 카드 스타일 */
[data-testid="column"]:first-child [data-testid="stVerticalBlock"] {{
    background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 120%);
    padding: 1.4rem 1.5rem 1.6rem;
    border-radius: 22px;
    border: 1px solid #dfe7f2;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.10);
}}
[data-testid="column"]:first-child [data-testid="stVerticalBlock"] > div + div {{
    margin-top: 1.3rem;
}}
[data-testid="column"]:first-child .stButton>button {{
    width: 100%;
    border-radius: 12px;
    border: 1px solid #cbd5f5;
    background: #e0e7ff;
    color: #1e3a8a;
    font-weight: 600;
}}
[data-testid="column"]:first-child .stButton>button:hover {{
    border-color: #1d4ed8;
    background: #dbeafe;
    color: #1e3a8a;
}}
.legend-box {{
    max-height: {LEGEND_MAX_H}px;
    overflow-y: auto;
    padding-right: 8px;
}}
.legend-box::-webkit-scrollbar {{ width: 6px; }}
.legend-box::-webkit-scrollbar-thumb {{ background: rgba(100,116,139,0.4); border-radius: 999px; }}
.stDownloadButton button {{
    border-radius: 12px;
    font-weight: 600;
    background: #0f172a;
    color: #f8fafc;
    border: none;
}}
.stDownloadButton button:hover {{
    background: #1e293b;
    color: #f8fafc;
}}
[data-testid="stDivider"] > div {{
    border-color: rgba(148, 163, 184, 0.45) !important;
}}
[data-testid="stDivider"] {{
    margin: 1.4rem 0 1.1rem 0;
}}
[data-testid="column"]:nth-child(2) iframe {{
    border-radius: 22px;
    box-shadow: 0 32px 70px rgba(15, 23, 42, 0.22);
    border: 0;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("## 해양 포유류 탐지 뷰어")

col_ctrl, col_main = st.columns([LEFT_COL_RATIO, RIGHT_COL_RATIO], gap="medium")

md_dir = str(DEFAULT_METADATA_DIR)
records = load_jsons(md_dir)
objects_catalog, file_to_labels = build_objects(records)

file_list = sorted(file_to_labels.keys())
default_video_sel = file_list.copy()

file_display_lookup: Dict[str, str] = {}
for item in objects_catalog:
    key = item["file"]
    if key not in file_display_lookup:
        display_name = item.get("video_file_name") or Path(key).stem
        file_display_lookup[key] = display_name
for key in file_list:
    file_display_lookup.setdefault(key, Path(key).stem)

# ---- 좌측 컨트롤 ----
with col_ctrl:
    st.markdown("#### 🎞️ 영상 목록")
    ba, bb, _sp = st.columns([1.2, 1.2, 6.0], gap="small")
    with ba:
        all_v = st.button("전체", key="btn_v_all")
    with bb:
        none_v = st.button("해제", key="btn_v_none")

    sel_videos = st.session_state.get("sel_videos", default_video_sel)
    if all_v:
        sel_videos = file_list.copy()
    if none_v:
        sel_videos = []
    sel_videos = st.multiselect(
        "영상 선택",
        options=file_list,
        default=sel_videos,
        format_func=lambda x: file_display_lookup.get(x, x),
        label_visibility="collapsed",
        help="표시할 영상을 선택합니다.",
    )
    st.session_state.sel_videos = sel_videos

    selected_objs = [t for t in objects_catalog if t["file"] in sel_videos]

    st.markdown("#### 🧮 영상 당 객체 수")
    if sel_videos:
        counts_html = ["<table style='width:100%;font-size:14px;line-height:1.6;border-collapse:collapse;'>"]
        counts_html.append("<thead><tr><th style='text-align:left;padding:4px 0;'>영상</th><th style='text-align:right;padding:4px 0;'>객체 수</th></tr></thead>")
        counts_html.append("<tbody>")
        for fname in sel_videos:
            cnt = sum(1 for t in selected_objs if t["file"] == fname)
            display_name = html.escape(file_display_lookup.get(fname, fname))
            counts_html.append(
                f"<tr><td style='padding:2px 0;border-bottom:1px solid #e5e7eb;'>{display_name}</td>"
                f"<td style='padding:2px 0;border-bottom:1px solid #e5e7eb;text-align:right;'>{cnt}</td></tr>"
            )
        counts_html.append("</tbody></table>")
        st.markdown("".join(counts_html), unsafe_allow_html=True)
    else:
        st.caption("선택된 영상이 없습니다.")

    st.markdown("#### 📌 총 객체 수")
    st.metric("총 객체 수", f"{len(selected_objs):,}")

    st.divider()

    st.markdown("#### 🎯 범례")
    legend_items = []
    for obj in selected_objs:
        legend_items.append(
            f"""<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
                    <span style="display:inline-block;width:16px;height:16px;border-radius:3px;border:1px solid #e5e7eb;background:{obj['color']};"></span>
                    <span style="font-weight:600;font-size:15px;">객체 ID{obj['obj_id']}</span>
                    <span style="color:#6b7280;font-size:14px;">| {html.escape(obj.get('video_file_name') or obj['file'])}</span>
                </div>"""
        )
    if legend_items:
        st.markdown(f"<div class='legend-box'>{''.join(legend_items)}</div>", unsafe_allow_html=True)
    else:
        st.caption("표시할 객체가 없습니다.")

    st.divider()

    st.markdown("#### ⬇️ 내보내기")
    if selected_objs:
        export_df = make_csv(selected_objs)
        st.download_button(
            label="⬇️ CSV",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="objects_selection.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("선택된 객체가 없어 내보내기 버튼이 비활성화됩니다.")

    st.divider()

    st.markdown("#### 🗺️ 지도 배경")
    base_tile = st.selectbox(
        "지도 배경",
        ["Carto Light", "OpenStreetMap", "Esri World Imagery"],
        index=0,
        label_visibility="collapsed",
    )

# ---- 우측: 지도 ----
with col_main:
    if not selected_objs:
        st.info("표시할 객체가 없습니다. 왼쪽에서 영상을 선택하세요.")
    else:
        m = build_map(selected_objs, base_tile=base_tile)
        if m is None:
            st.warning("선택된 객체에 좌표 정보가 없어 지도를 생성할 수 없습니다.")
            st.stop()
        st_folium(m, width=None, height=MAP_HEIGHT)

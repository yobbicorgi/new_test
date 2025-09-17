from typing import List, Dict, Any, Optional
import math

# The haversine helper now lives in utils, and parse_tabular_file is in parse
from core import haversine_km
from core import parse_tabular_file

# -----------------------------
# helpers
# -----------------------------
def _round_or_none(val: Optional[float], digits: int) -> Optional[float]:
    if val is None:
        return None
    try:
        return round(float(val), digits)
    except Exception:
        return None

def _agg(arr: List[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [float(x) for x in arr if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(xs), "max": max(xs), "mean": sum(xs) / len(xs)}

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_km(lat1, lon1, lat2, lon2) * 1000.0

def _norm_time_rel(
    t_clock_sod: Optional[float],
    t_rel_s: Optional[float],
    clip_by: Optional[Dict[str, Any]]
) -> Optional[float]:
    # 절대 시작 시각이 주어졌으면 시계 시각을 상대초로 정규화
    if clip_by and "start_sod" in clip_by and t_clock_sod is not None:
        start_sod = float(clip_by["start_sod"])
        return (float(t_clock_sod) - start_sod) % 86400.0
    # 상대시간 우선
    if t_rel_s is not None:
        try:
            return float(t_rel_s)
        except Exception:
            return None
    # 마지막 수단: 시계 시각만 있는 경우
    if t_clock_sod is not None:
        try:
            return float(t_clock_sod)
        except Exception:
            return None
    return None

def _within_clip(t_rel: float, clip_by: Optional[Dict[str, Any]]) -> bool:
    if not clip_by:
        return True
    dur = clip_by.get("duration_s")
    if dur is None:
        return True
    try:
        return (0.0 <= float(t_rel) <= float(dur))
    except Exception:
        return True

def _empty_metrics(duration_fallback: Optional[float] = None, status: str = "no_samples") -> Dict[str, Any]:
    """항상 같은 스키마를 반환하는 빈 템플릿 (스파이크 관련 항목 없음)"""
    return {
        "reference": {"altitude_datum": None, "notes": None},
        "time": {"start_iso": None, "end_iso": None, "duration_s": duration_fallback},
        "position_start": {"lat": None, "lon": None},
        "position_end": {"lat": None, "lon": None},
        "distance_2d_km": None,
        "samples": 0,
        "altitude_m": {"min": None, "max": None, "mean": None},
        "speed_mps": {"min": None, "max": None, "mean": None},
        "speed_kmh": {"min": None, "max": None, "mean": None},
        "qa": {
            "samples_total": 0,
            "coords_with_latlon": 0,
            "segments_total": 0,
            "segments_used_for_speed": 0,
            "segments_zero_or_negative_dt": 0,
            "speed_sources": {
                "provided_mps_count": 0,
                "from_path_mps_count": 0
            },
            "distance": {
                "path_km": None,
                "straight_km": None,
                "path_over_straight": None
            },
            "status": status
        }
    }

# -----------------------------
# main
# -----------------------------
def compute_flight_metrics(
    paths: List[str],
    media_basename: str,
    clip_by: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    paths: 표형식/텍스트/SMI 보조 데이터 파일 목록 (JSON 입력 제외)
    clip_by:
      - {"start_sod": <0~86399>, "duration_s": <float>}
      - {"duration_s": <float>}
    """
    samples: List[Dict[str, float]] = []

    # 1) 보조 파일들에서 시계열 수집
    for path in paths:
        try:
            series = parse_tabular_file(path)
        except Exception:
            series = None
        if not series:
            continue

        lat = series.get("lat") or []
        lon = series.get("lon") or []
        alt = series.get("alt_m") or []
        spd = series.get("speed_mps") or []
        t_rel = series.get("t_rel_s") or []
        t_clk = series.get("t_clock_sod") or []

        n = max(len(lat), len(lon), len(alt), len(spd), len(t_rel), len(t_clk))
        for i in range(n):
            la = lat[i] if i < len(lat) else None
            lo = lon[i] if i < len(lon) else None
            al = alt[i] if i < len(alt) else None
            vv = spd[i] if i < len(spd) else None
            tr = t_rel[i] if i < len(t_rel) else None
            tc = t_clk[i] if i < len(t_clk) else None

            t_norm = _norm_time_rel(tc, tr, clip_by)
            if t_norm is None or not _within_clip(t_norm, clip_by):
                continue

            samples.append({
                "t": float(t_norm),
                "lat": None if la is None else float(la),
                "lon": None if lo is None else float(lo),
                "alt": None if al is None else float(al),
                "spd": None if vv is None else float(vv)  # m/s 가정
            })

    # 샘플이 없으면 빈 템플릿 반환
    if not samples:
        return _empty_metrics(duration_fallback=(clip_by.get("duration_s") if clip_by else None))

    # 2) 시간순 정렬
    samples.sort(key=lambda r: r["t"])

    # 기본 집계용 카운트
    total_segments = max(0, len(samples) - 1)
    coords_count = sum(1 for r in samples if r.get("lat") is not None and r.get("lon") is not None)

    # 3) 거리/속도/고도 계산 (있는 데이터 그대로 사용)
    speed_mps_list: List[float] = []
    speed_mps_provided: List[float] = []   # 보조파일에 직접 제공된 속도
    speed_mps_from_path: List[float] = []  # 좌표 기반 계산 속도
    alt_list: List[float] = []
    dist_total_m = 0.0
    zero_dt = 0

    # 고도 수집
    for r in samples:
        if r.get("alt") is not None and math.isfinite(r["alt"]):
            alt_list.append(float(r["alt"]))

    # 제공 속도 수집
    for r in samples:
        v = r.get("spd")
        if v is not None and math.isfinite(v):
            v = float(v)
            speed_mps_list.append(v)
            speed_mps_provided.append(v)

    # 좌표 기반 속도 및 총 거리
    for i in range(1, len(samples)):
        p = samples[i - 1]
        q = samples[i]
        dt = float(q["t"]) - float(p["t"])
        if dt <= 0:
            zero_dt += 1
            continue
        if (p.get("lat") is not None and p.get("lon") is not None and
            q.get("lat") is not None and q.get("lon") is not None):
            d_m = _haversine_m(p["lat"], p["lon"], q["lat"], q["lon"])
            dist_total_m += d_m
            v = d_m / dt
            if math.isfinite(v):
                speed_mps_list.append(v)
                speed_mps_from_path.append(v)

    # 4) 시간/위치/통계 요약
    t0 = samples[0]["t"]; t1 = samples[-1]["t"]
    duration_s = float(t1 - t0) if (t1 is not None and t0 is not None) else (clip_by.get("duration_s") if clip_by else None)

    # 시작/종료 위치(좌표가 있는 첫/마지막 샘플)
    pos_start = {"lat": None, "lon": None}
    pos_end   = {"lat": None, "lon": None}
    for r in samples:
        if r.get("lat") is not None and r.get("lon") is not None:
            pos_start = {"lat": float(r["lat"]), "lon": float(r["lon"])}
            break
    for r in reversed(samples):
        if r.get("lat") is not None and r.get("lon") is not None:
            pos_end = {"lat": float(r["lat"]), "lon": float(r["lon"])}
            break

    # 통계(반올림 전)
    alt_stats = _agg(alt_list)
    spd_stats = _agg(speed_mps_list)
    spd_kmh_stats = {k: (v * 3.6 if v is not None else None) for k, v in spd_stats.items()}

    # 경로 vs 직선 거리
    if coords_count >= 2 and pos_start["lat"] is not None and pos_end["lat"] is not None:
        straight_m = _haversine_m(pos_start["lat"], pos_start["lon"], pos_end["lat"], pos_end["lon"])
    else:
        straight_m = 0.0
    path_vs_straight = (dist_total_m / straight_m) if straight_m > 0 else None

    # 5) 반올림 적용
    pos_start["lat"] = _round_or_none(pos_start["lat"], 8)
    pos_start["lon"] = _round_or_none(pos_start["lon"], 8)
    pos_end["lat"]   = _round_or_none(pos_end["lat"], 8)
    pos_end["lon"]   = _round_or_none(pos_end["lon"], 8)

    alt_stats = {
        "min": _round_or_none(alt_stats["min"], 4),
        "max": _round_or_none(alt_stats["max"], 4),
        "mean": _round_or_none(alt_stats["mean"], 4),
    }
    spd_stats = {
        "min": _round_or_none(spd_stats["min"], 4),
        "max": _round_or_none(spd_stats["max"], 4),
        "mean": _round_or_none(spd_stats["mean"], 4),
    }
    spd_kmh_stats = {
        "min": _round_or_none(spd_kmh_stats["min"], 4),
        "max": _round_or_none(spd_kmh_stats["max"], 4),
        "mean": _round_or_none(spd_kmh_stats["mean"], 4),
    }

    dist_km = dist_total_m / 1000.0

    # 6) QA(간단 집계만, 스파이크 등 없음)
    qa = {
        "samples_total": len(samples),
        "coords_with_latlon": int(coords_count),
        "segments_total": int(total_segments),
        "segments_used_for_speed": int(len(speed_mps_from_path)),
        "segments_zero_or_negative_dt": int(zero_dt),
        "speed_sources": {
            "provided_mps_count": int(len(speed_mps_provided)),
            "from_path_mps_count": int(len(speed_mps_from_path))
        },
        "distance": {
            "path_km": _round_or_none(dist_km, 6),
            "straight_km": _round_or_none(straight_m / 1000.0 if straight_m else None, 6),
            "path_over_straight": _round_or_none(path_vs_straight, 4)
        },
        "status": "ok" if len(samples) >= 2 else "insufficient_samples"
    }

    ref = {"altitude_datum": None, "notes": None}

    return {
        "reference": ref,
        "time": {"start_iso": None, "end_iso": None, "duration_s": duration_s},
        "position_start": pos_start,
        "position_end": pos_end,
        "distance_2d_km": _round_or_none(dist_km, 8),
        "samples": len(samples),
        "altitude_m": alt_stats,
        "speed_mps": spd_stats,
        "speed_kmh": spd_kmh_stats,
        "qa": qa
    }

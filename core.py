# ==============================
# core.py  (flat single-folder)
# ==============================
# - 보조데이터 파서: SRT/SMI/ASS/SSA/VTT, CSV/TXT/LOG, XLSX/XLS, XML, JSON
# - LRV/THM는 무시 (보조파일 탐색 대상에 포함하지 않음)
# - 비디오 메타(ffprobe) + (선택) 이미지 메타(Pillow)
# - 보간 유틸 + 파일 매칭 유틸
# - build_metadata_for_media(media_path, data_dir) 공개
# - any-order 좌표 파서(parse_coords_any_order)로 lat/lon 순서 혼종 케이스 대응

from __future__ import annotations
import re, os, json, csv, math, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# ============ 옵션/로깅 ============
FAST_MODE: bool = True     # ffprobe 외 무거운 폴백 최소화
VERBOSE: bool   = True

def set_verbose(v: bool) -> None:
    global VERBOSE
    VERBOSE = bool(v)

def _log(*args: Any) -> None:
    if VERBOSE:
        try: print(*args)
        except Exception: pass

def log_error(*args: Any) -> None:
    try: print(*args)
    except Exception: pass

# ============ 포맷/확장자 ============
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

# 보조파일 (LRV/THM 제외)
SUPP_EXTS_BY_KIND: Dict[str, set] = {
    "xml":      {".xml"},
    "txt":      {".txt", ".log"},
    "csv":      {".csv"},
    "excel":    {".xlsx", ".xls"},
    "json":     {".json"},
    "subtitle": {".smi", ".srt", ".ass", ".ssa", ".vtt"},
}

# ============ 공통 유틸 ============
def clamp_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def iso_from_ts(ts: Union[int, float, str]) -> Optional[str]:
    try:
        v = float(ts)
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
    except Exception:
        try:
            datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return str(ts)
        except Exception:
            return None

def _read_text(path: Path) -> Optional[str]:
    for enc in ("utf-8", "cp949", "euc-kr", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return None

def _parse_timecode_to_s(tc: str) -> Optional[float]:
    # 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm'
    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})", tc.strip())
    if not m: return None
    h, mm, s, ms = [int(x) for x in m.groups()]
    return h*3600 + mm*60 + s + ms/1000.0

def meters_per_deg_lat(lat_deg: float) -> float:
    lat = math.radians(lat_deg)
    return 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat)

def meters_per_deg_lon(lat_deg: float) -> float:
    lat = math.radians(lat_deg)
    return 111412.84*math.cos(lat) - 93.5*math.cos(3*lat)

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 좌표 사이 거리를 km 단위로 반환."""
    r = 6371.0088  # 평균 지구 반지름(km)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c

# ============ 좌표 파서(순서 무관) ============
# 소수/도분초 + N/S/E/W 허용, 순서가 뒤집혀도 (lat, lon)으로 추출
_coord_token = re.compile(
    r"""
    (?P<full>
      (
        (?P<dms>                                   # DMS: 34°39'29.99"N, 128°17'39.20"E
          (?P<deg>[-+]?\d{1,3})[°º]\s*
          (?P<min>\d{1,2})['’]\s*
          (?P<sec>[\d\.]+)"?\s*
          (?P<hemi1>[NSEW])?
        )
        |
        (?P<dec>                                   # Decimal: 34.6583N, 128.2942E, or -122.42
          (?<!\d)(?P<val>[-+]?\d{1,3}(?:\.\d+)?)(?!\d)
          \s*(?P<hemi2>[NSEW])?
        )
      )
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_label_lat = re.compile(r"\b(?:lat|latitude)[^\d\-+]*([-+]?\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
_label_lon = re.compile(r"\b(?:lon|longitude)[^\d\-+]*([-+]?\d{1,3}(?:\.\d+)?)", re.IGNORECASE)

def _to_dd_from_match(m: re.Match) -> Optional[float]:
    try:
        if m.group("dms"):
            d = float(m.group("deg")); mi = float(m.group("min")); se = float(m.group("sec"))
            val = d + mi/60.0 + se/3600.0
            hemi = (m.group("hemi1") or "").upper()
            if hemi in ("S","W"): val = -val
            return val
        else:
            val = float(m.group("val"))
            hemi = (m.group("hemi2") or "").upper()
            if hemi in ("S","W"): val = -val
            return val
    except Exception:
        return None

def parse_coords_any_order(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    문자열에서 위·경도를 어떤 순서로 쓰더라도 (lat, lon)로 반환.
    - N/S/E/W 표기가 있으면 그걸로 분류
    - 표기가 없으면 값 범위( |lat|<=90, |lon|<=180 )로 추론
    - 후보가 2개 이상이면 가장 합리적인 쌍 선택
    """
    if not text:
        return (None, None)

    # 0) 라벨 기반(lat/lon) 우선 추출
    m_la = _label_lat.search(text)
    m_lo = _label_lon.search(text)
    if m_la and m_lo:
        try:
            return (float(m_la.group(1)), float(m_lo.group(1)))
        except Exception:
            pass

    cands: List[Tuple[float, str]] = []
    for m in _coord_token.finditer(text):
        v = _to_dd_from_match(m)
        if v is None:
            continue
        hemi = (m.group("hemi1") or m.group("hemi2") or "").upper()
        cands.append((v, hemi))

    if not cands:
        return (None, None)

    # 1) hemisphere 기반 1차 분류
    lat_list = [v for (v,h) in cands if h in ("N","S")]
    lon_list = [v for (v,h) in cands if h in ("E","W")]

    # 2) 값 범위로 보완
    if not lat_list:
        lat_list = [v for (v,h) in cands if h=="" and abs(v) <= 90.0]
    if not lon_list:
        lon_list = [v for (v,h) in cands if h=="" and abs(v) <= 180.0 and abs(v) > 90.0]

    # 3) 마지막 보루: 범위도 없음 → 크기 기준(작은=lat, 큰=lon)
    if not lat_list and not lon_list and len(cands) >= 2:
        vs = sorted([v for (v,_) in cands], key=lambda x: abs(x))
        return (vs[0], vs[-1])

    lat = lat_list[0] if lat_list else None
    lon = lon_list[0] if lon_list else None
    return (lat, lon)

# ============ 자막 파서 ============
def _parse_smi_blocks(raw: str) -> List[Tuple[float, str]]:
    sync_re = re.compile(r"<SYNC\s+Start\s*=\s*(\d+)[^>]*>", re.IGNORECASE)
    blocks: List[Tuple[float, str]] = []
    sync_positions = [(m.start(), int(m.group(1))) for m in sync_re.finditer(raw)]
    for i, (pos, ms) in enumerate(sync_positions):
        end = sync_positions[i+1][0] if i+1 < len(sync_positions) else len(raw)
        chunk = raw[pos:end]
        t_rel = ms / 1000.0
        text = re.sub(r"<[^>]+>", " ", chunk)
        text = re.sub(r"\s+", " ", text).strip()
        if text: blocks.append((t_rel, text))
    return blocks

def _parse_srt_lines(raw: str) -> List[Tuple[float, str]]:
    lines = raw.splitlines()
    out: List[Tuple[float, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit(): i += 1; continue
        m = re.search(r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})", line)
        if m:
            t0 = _parse_timecode_to_s(m.group(1))
            i += 1
            buf = []
            while i < len(lines) and lines[i].strip() != "":
                buf.append(re.sub(r"<[^>]+>", " ", lines[i]))
                i += 1
            text = re.sub(r"\s+", " ", " ".join(buf)).strip()
            if text and t0 is not None:
                out.append((t0, text))
        i += 1
    return out

# ============ 표/엑셀 파서 ============
try:
    import pandas as pd  # optional
except Exception:
    pd = None  # type: ignore

def _normalize_columns(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        cc = str(c).strip().lower()
        cc = cc.replace(" ", "_")
        cc = re.sub(r"[^a-z0-9_]", "_", cc)
        out.append(cc)
    return out

def _extract_series_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    lat, lon, alt_m, speed_mps, t_rel_s, t_clock_sod = [], [], [], [], [], []
    clock_dates: List[Optional[str]] = []
    clock_times: List[Optional[str]] = []
    clock_iso: List[Optional[str]] = []
    lat_keys  = ["lat","latitude","y_lat"]
    lon_keys  = ["lon","lng","long","longitude","x_lon"]
    altm_keys = ["alt","alt_m","altitude","height_m"]
    altft_keys= ["alt_ft","altitude_ft","height_ft"]
    sp_mps_keys = ["speed_mps","spd_mps"]
    sp_kmh_keys = ["speed_kmh","spd_kmh","kmh"]
    sp_kts_keys = ["speed_kts","spd_kts","knots","kts"]
    trel_keys = ["t_rel_s","time_s","t","sec","seconds"]
    tclk_keys = ["t_clock_sod","t_utc_s","time_of_day_s","sod"]
    date_keys = ["date","date_utc","clock_date","day"]
    time_keys = ["time","clock_time","timestamp","time_str"]
    iso_keys  = ["datetime","datetime_iso","clock_iso"]

    for r in rows:
        rr = {str(k).strip().lower(): r[k] for k in r.keys()}
        # lat/lon
        la = next((rr.get(k) for k in lat_keys if k in rr), None)
        lo = next((rr.get(k) for k in lon_keys if k in rr), None)

        # (보완) 열 명이 불명확하고 숫자 2개만 있는 경우 → 범위로 추정
        if la in (None, "") and lo in (None, "") and len(rr) <= 3:
            nums = []
            for v in rr.values():
                try:
                    vv = float(v)
                    nums.append(vv)
                except Exception:
                    pass
            if len(nums) >= 2:
                nums = sorted(nums, key=lambda x: abs(x))
                la = nums[0] if abs(nums[0]) <= 90 else None
                lo = nums[-1] if abs(nums[-1]) <= 180 else None

        try: la = float(la) if la not in (None,"") else None
        except Exception: la = None
        try: lo = float(lo) if lo not in (None,"") else None
        except Exception: lo = None

        # alt
        al = next((rr.get(k) for k in altm_keys if k in rr), None)
        if al in (None,""):
            al_ft = next((rr.get(k) for k in altft_keys if k in rr), None)
            try: al = float(al_ft)*0.3048 if al_ft not in (None,"") else None
            except Exception: al = None
        try: al = float(al) if al not in (None,"") else None
        except Exception: al = None

        # speed
        sp = next((rr.get(k) for k in sp_mps_keys if k in rr), None)
        if sp in (None,""):
            s2 = next((rr.get(k) for k in sp_kmh_keys if k in rr), None)
            try: sp = float(s2)/3.6 if s2 not in (None,"") else None
            except Exception: sp = None
        if sp in (None,""):
            s3 = next((rr.get(k) for k in sp_kts_keys if k in rr), None)
            try: sp = float(s3)*0.514444 if s3 not in (None,"") else None
            except Exception: sp = None
        try: sp = float(sp) if sp not in (None,"") else None
        except Exception: sp = None

        # time
        tr = next((rr.get(k) for k in trel_keys if k in rr), None)
        tc = next((rr.get(k) for k in tclk_keys if k in rr), None)
        cd = next((rr.get(k) for k in date_keys if k in rr), None)
        ct = next((rr.get(k) for k in time_keys if k in rr), None)
        ci = next((rr.get(k) for k in iso_keys if k in rr), None)
        try: tr = float(tr) if tr not in (None,"") else None
        except Exception: tr = None
        try: tc = float(tc) if tc not in (None,"") else None
        except Exception: tc = None

        def _clean_str(val: Any) -> Optional[str]:
            if val in (None, ""):
                return None
            return str(val).strip()

        lat.append(la); lon.append(lo); alt_m.append(al); speed_mps.append(sp); t_rel_s.append(tr); t_clock_sod.append(tc)
        clock_dates.append(_clean_str(cd))
        clock_times.append(_clean_str(ct))
        clock_iso.append(_clean_str(ci))
    return {
        "lat": lat,
        "lon": lon,
        "alt_m": alt_m,
        "speed_mps": speed_mps,
        "t_rel_s": t_rel_s,
        "t_clock_sod": t_clock_sod,
        "clock_date": clock_dates,
        "clock_time": clock_times,
        "clock_iso": clock_iso,
    }

# ============ public: 단일파일 시계열 ============
def _extract_clock_from_text(text: str) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str]]:
    """Extract (date, time, seconds_of_day, iso) from free-form text."""
    if not text:
        return (None, None, None, None)
    stripped = text.strip()

    def _normalize_date(year: int, month: int, day: int) -> str:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    def _normalize_time(hour: int, minute: Optional[int], second: Optional[int]) -> str:
        mm = 0 if minute is None else int(minute)
        ss = 0 if second is None else int(second)
        return f"{int(hour):02d}:{mm:02d}:{ss:02d}"

    m = re.match(
        r"\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2})([.,](\d{1,6}))?)?",
        stripped,
    )
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hour = int(m.group(4))
        minute = int(m.group(5))
        sec = int(m.group(6)) if m.group(6) is not None else 0
        frac_txt = m.group(8) or ""
        frac = 0.0
        if frac_txt:
            try:
                frac = float(f"0.{frac_txt}")
            except Exception:
                frac = 0.0
        sod = hour * 3600 + minute * 60 + sec + frac
        date_norm = _normalize_date(year, month, day)
        time_norm = _normalize_time(hour, minute, sec)
        time_iso = f"{time_norm}.{frac_txt[:6]}" if frac_txt else time_norm
        iso = f"{date_norm}T{time_iso}"
        return (date_norm, time_iso, sod, iso)

    m = re.match(r"\s*(\d{1,2})(?::(\d{2}))?(?::(\d{2})([.,](\d{1,6}))?)?", stripped)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) is not None else 0
        sec = int(m.group(3)) if m.group(3) is not None else 0
        frac_txt = m.group(5) or ""
        frac = 0.0
        if frac_txt:
            try:
                frac = float(f"0.{frac_txt}")
            except Exception:
                frac = 0.0
        sod = hour * 3600 + minute * 60 + sec + frac
        time_norm = _normalize_time(hour, minute, sec)
        time_iso = f"{time_norm}.{frac_txt[:6]}" if frac_txt else time_norm
        return (None, time_iso, sod, None)

    return (None, None, None, None)


def parse_tabular_file(p: Union[str, Path]) -> Dict[str, List[Any]]:
    p = Path(p); ext = p.suffix.lower()
    def _empty():
        return {
            "lat": [],
            "lon": [],
            "alt_m": [],
            "speed_mps": [],
            "t_rel_s": [],
            "t_clock_sod": [],
            "clock_date": [],
            "clock_time": [],
            "clock_iso": [],
        }

    # 자막
    if ext in {".smi",".srt",".ass",".ssa",".vtt"}:
        raw = _read_text(p)
        if not raw: return _empty()
        blocks = _parse_smi_blocks(raw) if ext==".smi" else _parse_srt_lines(raw)
        lat,lon,alt_m,speed_mps,t_rel_s = [],[],[],[],[]
        clock_dates: List[Optional[str]] = []
        clock_times: List[Optional[str]] = []
        clock_iso: List[Optional[str]] = []
        clock_sod: List[Optional[float]] = []
        for (t0, text) in blocks:
            la, lo = parse_coords_any_order(text)   # any-order 좌표 파싱
            al = _pick_alt(text)
            sp = _pick_speed_mps(text)
            date_val, time_val, sod_val, iso_val = _extract_clock_from_text(text)
            lat.append(la); lon.append(lo); alt_m.append(al); speed_mps.append(sp); t_rel_s.append(t0)
            clock_dates.append(date_val)
            clock_times.append(time_val)
            clock_iso.append(iso_val)
            clock_sod.append(sod_val)
        return {
            "lat": lat,
            "lon": lon,
            "alt_m": alt_m,
            "speed_mps": speed_mps,
            "t_rel_s": t_rel_s,
            "t_clock_sod": clock_sod,
            "clock_date": clock_dates,
            "clock_time": clock_times,
            "clock_iso": clock_iso,
        }

    # CSV/TXT/LOG
    if ext in {".csv",".txt",".log"}:
        rows: List[Dict[str,Any]] = []
        try:
            if ext==".csv" and pd is not None:
                df = pd.read_csv(p)
                df.columns = _normalize_columns([str(c) for c in df.columns])
                rows = df.to_dict(orient="records")
            else:
                # 라인 스캐닝(키=값 or 쉼표/탭 헤더)
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    sample = f.read(4096); f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                        reader = csv.DictReader(f, dialect=dialect)
                        rows = list(reader)
                    except Exception:
                        # 자유 텍스트 → 한 줄에서 좌표/속도/고도 추출
                        for line in f:
                            la,lo = parse_coords_any_order(line)
                            al = _pick_alt(line); sp = _pick_speed_mps(line)
                            if any(v is not None for v in (la,lo,al,sp)):
                                rows.append({"lat":la,"lon":lo,"alt_m":al,"speed_mps":sp})
        except Exception:
            pass
        return _extract_series_from_rows(rows)

    # EXCEL
    if ext in {".xlsx",".xls"} and pd is not None:
        try:
            df = pd.read_excel(p, sheet_name=0)
            df.columns = _normalize_columns([str(c) for c in df.columns])
            rows = df.to_dict(orient="records")
            return _extract_series_from_rows(rows)
        except Exception:
            return _empty()

    # XML/JSON은 flight 시계열로 직접 쓰지 않음
    return _empty()

# ============ 텍스트 내 수치 추출(고도/속도) ============
def _pick_alt(text: str) -> Optional[float]:
    m = re.search(r"\bALT[^\d\-+]*([\-+]?\d+(?:\.\d+)?)\s*m\b", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"\bALT[^\d\-+]*([\-+]?\d+(?:\.\d+)?)\s*ft\b", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 0.3048
    m = re.search(r"\bALT[^\d\-+]*([\-+]?\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"\b([\-+]?\d+(?:\.\d+)?)\s*m\b", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None

def _pick_speed_mps(text: str) -> Optional[float]:
    m = re.search(r"\b([\-+]?\d+(?:\.\d+)?)\s*m/s\b", text, re.IGNORECASE)
    if m: return float(m.group(1))
    m = re.search(r"\b([\-+]?\d+(?:\.\d+)?)\s*km/?h\b", text, re.IGNORECASE)
    if m: return float(m.group(1)) / 3.6
    m = re.search(r"\b([\-+]?\d+(?:\.\d+)?)\s*kts?\b", text, re.IGNORECASE)
    if m: return float(m.group(1)) * 0.514444
    return None

# ============ 보간 ============
def interpolate_to_freq(series: Dict[str, List[Any]], duration_s: Optional[float], freq: int) -> Dict[int, Dict[str, Optional[float]]]:
    """주어진 시계열을 freq(Hz) 간격으로 보간하여 반환."""
    lat = series.get("lat", []); lon = series.get("lon", []); alt_m = series.get("alt_m", [])
    t_rel = series.get("t_rel_s", []); t_clk = series.get("t_clock_sod", [])
    raw_t: List[float] = []; raw_lat: List[Optional[float]]=[]; raw_lon: List[Optional[float]]=[]; raw_alt: List[Optional[float]]=[]
    n = max(len(t_rel), len(t_clk), len(lat), len(lon), len(alt_m))
    for i in range(n):
        tr = t_rel[i] if i < len(t_rel) else None
        tc = t_clk[i] if i < len(t_clk) else None
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
        la = lat[i] if i < len(lat) else None
        lo = lon[i] if i < len(lon) else None
        al = alt_m[i] if i < len(alt_m) else None
        raw_t.append(t); raw_lat.append(la); raw_lon.append(lo); raw_alt.append(al)
    if not raw_t:
        return {}
    idx = sorted(range(len(raw_t)), key=lambda k: raw_t[k])
    raw_t   = [raw_t[i] for i in idx]
    raw_lat = [raw_lat[i] for i in idx]
    raw_lon = [raw_lon[i] for i in idx]
    raw_alt = [raw_alt[i] for i in idx]
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
        else:
            t0, t1 = raw_t[j], raw_t[j + 1]
            a = (ts - t0) / (t1 - t0) if (t1 > t0) else 0.0
            def lerp(u, v):
                if u is None and v is None: return None
                if u is None: return v
                if v is None: return u
                try: return (1.0 - a) * float(u) + a * float(v)
                except Exception: return u
            la = lerp(raw_lat[j], raw_lat[j + 1])
            lo = lerp(raw_lon[j], raw_lon[j + 1])
            al = lerp(raw_alt[j], raw_alt[j + 1])
        out[idx_t] = {
            "lat": None if la is None else round(float(la), 8),
            "lon": None if lo is None else round(float(lo), 8),
            "alt": None if al is None else round(float(al), 4)
        }
    return out

def interpolate_to_1hz(series: Dict[str, List[Any]], duration_s: Optional[float]) -> Dict[int, Dict[str, Optional[float]]]:
    return interpolate_to_freq(series, duration_s, 1)

# ============ ffprobe ============
def run_ffprobe_json(path: Path) -> Optional[dict]:
    try:
        proc = subprocess.run(
            ["ffprobe","-v","error","-print_format","json",
             "-select_streams","v:0",
             "-show_entries","stream=width,height,nb_frames,avg_frame_rate,codec_type",
             "-show_entries","format=duration,bit_rate,format_name,format_long_name:format_tags=creation_time",
             str(path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not proc.stdout: return None
        return json.loads(proc.stdout)
    except Exception:
        return None

# ============ 보조파일 탐색 ============
_word = re.compile(r"[a-zA-Z0-9가-힣]+")

def _normalize_stem(s: str) -> str:
    return "".join(_word.findall(str(s).lower()))

def find_supplementals(data_dir: Union[str, Path], base: str) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {k: [] for k in SUPP_EXTS_BY_KIND.keys()}
    d = Path(data_dir)
    if not d.exists(): return out
    base_norm = _normalize_stem(base)
    for kind, exts in SUPP_EXTS_BY_KIND.items():
        for ext in exts:
            for p in d.glob(f"{base}*{ext}"):
                stem_norm = _normalize_stem(p.stem)
                if stem_norm.startswith(base_norm) or base_norm.startswith(stem_norm):
                    out[kind].append(p)
    return out

# ============ XML 요약 ============
import xml.etree.ElementTree as ET
def parse_xml_summary(meta: Dict[str, Any], path: Path) -> None:
    try:
        tree = ET.parse(str(path)); root = tree.getroot()
    except Exception:
        return
    try:
        tag = root.tag
        meta["xml_root_tag"] = tag.split("}")[-1] if tag.startswith("{") else tag
        ns_uris = set()
        if tag.startswith("{"): ns_uris.add(tag[1:].partition("}")[0])
        for el in root.iter():
            if isinstance(el.tag,str) and el.tag.startswith("{"):
                ns_uris.add(el.tag[1:].partition("}")[0])
        if ns_uris: meta["xml_namespace_uris"] = sorted(ns_uris)
    except Exception: pass

    try:
        cd = root.find(".//CreationDate")
        if cd is not None and cd.text: meta["nrtm_creation_date"] = cd.text.strip()
        dev = root.find(".//Device")
        if dev is not None:
            def _t(x): return (dev.find(x).text.strip() if dev.find(x) is not None and dev.find(x).text else None)
            meta["nrtm_device_manufacturer"] = _t("Manufacturer") or _t("Maker")
            meta["nrtm_device_model"]        = _t("ModelName") or _t("Model")
            meta["nrtm_device_serial"]       = _t("SerialNo") or _t("Serial")
        dur_node = root.find(".//Duration")
        if dur_node is not None and dur_node.get("value"):
            try:
                meta["nrtm_duration_frames"] = dur_node.get("value")
            except Exception: pass
        ltc = root.find(".//LtcChangeTable")
        if ltc is not None:
            meta["nrtm_tc_fps"]      = ltc.get("tcFps")
            meta["nrtm_tc_halfstep"] = ltc.get("halfStep")
    except Exception:
        pass

# ============ CSV/엑셀/JSON 요약 ============
def parse_excel_summary(meta: Dict[str, Any], path: Path) -> None:
    if pd is None: return
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        sheets: List[Dict[str, Any]] = []
        for name in xl.sheet_names:
            try:
                df = xl.parse(name, nrows=0, engine="openpyxl")
                sheets.append({"name": name, "rows": None, "columns": list(map(str, df.columns.tolist()))})
            except Exception:
                sheets.append({"name": name, "rows": None, "columns": None})
        meta["excel_sheet_count"] = len(sheets)
        meta["excel_sheets"] = sheets
    except Exception:
        return

def parse_csv_summary(meta: Dict[str, Any], path: Path) -> None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096); f.seek(0)
            try: dialect = csv.Sniffer().sniff(sample)
            except Exception: dialect = csv.excel_tab if "\t" in sample else csv.excel
            reader = csv.reader(f, dialect)
            header = next(reader, [])
            meta["csv_columns"] = list(map(str, header)) if header else None
            rows = sum(1 for _ in reader)
            meta["csv_row_count"] = rows
    except Exception:
        return

def parse_json_summary(meta: Dict[str, Any], path: Path) -> None:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            meta["json_keys"] = list(obj.keys())
        elif isinstance(obj, list):
            meta["json_keys"] = ["__list__"]
        else:
            meta["json_keys"] = [type(obj).__name__]
    except Exception:
        return

# ============ 이미지 요약 (선택) ============
try:
    from PIL import Image, ExifTags  # type: ignore
except Exception:
    Image = None
    ExifTags = None

def extract_image(meta: Dict[str, Any], path: Path) -> None:
    if Image is None: return
    try:
        with Image.open(path) as im:
            meta["image_format"] = im.format
            meta["img_width"], meta["img_height"] = im.size
            meta["color_mode"] = im.mode
            meta["bit_depth"] = getattr(im, "bits", None)
            exif = im._getexif() if hasattr(im, "_getexif") else None
            if exif and ExifTags is not None:
                tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
                meta["exif_camera_make"]  = tags.get("Make")
                meta["exif_camera_model"] = tags.get("Model")
    except Exception:
        return

# ============ 비디오 요약 ============
def extract_video(meta: Dict[str, Any], path: Path) -> None:
    info = run_ffprobe_json(path)
    if not info: return
    try:
        fmt = info.get("format", {})
        streams = info.get("streams", [])
        v = None
        for s in streams:
            if s.get("codec_type") == "video":
                v = s; break
        if not v and streams: v = streams[0]
        if fmt.get("duration") is not None:
            try:
                meta["duration"] = float(fmt["duration"])
                dur = float(meta["duration"])
                hh = int(dur // 3600); mm = int((dur % 3600) // 60); ss = int(dur % 60)
                meta["duration_hms"] = f"{hh:02d}:{mm:02d}:{ss:02d}"
            except Exception:
                pass
        if fmt.get("bit_rate") is not None:
            try: meta["bit_rate"] = int(fmt["bit_rate"])
            except Exception: pass
        if v:
            if v.get("width"):  meta["width"]  = int(v["width"])
            if v.get("height"): meta["height"] = int(v["height"])
            afr = v.get("avg_frame_rate")
            if afr and afr != "0/0":
                try:
                    num, den = afr.split("/")
                    den_f = float(den); fps = float(num)/den_f if den_f!=0 else None
                    if fps and fps>0: meta["frame_rate"] = fps
                except Exception: pass
            if v.get("nb_frames") and meta.get("frame_rate") and not meta.get("duration"):
                try:
                    meta["nb_frames"] = int(v["nb_frames"])
                    meta["duration"] = float(meta["nb_frames"]) / float(meta["frame_rate"])
                    dur = float(meta["duration"])
                    hh = int(dur // 3600); mm = int((dur % 3600) // 60); ss = int(dur % 60)
                    meta["duration_hms"] = f"{hh:02d}:{mm:02d}:{ss:02d}"
                except Exception:
                    pass
    except Exception:
        pass

# ============ 공통 메타 프레임 ============
def empty_schema() -> Dict[str, Any]:
    return {
        "file_name": None, "file_path": None, "file_size": None,
        "file_size_mb": None, "file_format": None,
        "created_at": None, "modified_time": None,
        "checksum_md5": None, "checksum_sha1": None, "checksum_sha256": None,
        # video
        "duration": None, "duration_hms": None, "nb_frames": None, "width": None, "height": None,
        "frame_rate": None, "bit_rate": None,
        # image
        "image_format": None, "img_width": None, "img_height": None,
        "color_mode": None, "bit_depth": None,
        # subtitles summary
        "subtitle_count": None, "subtitle_total_duration": None,
        "subtitle_first_start": None, "subtitle_last_end": None,
        "subtitle_first_text": None, "subtitle_last_text": None,
        # xml summary
        "xml_root_tag": None, "xml_namespace_uris": None,
        "nrtm_creation_date": None, "nrtm_device_manufacturer": None,
        "nrtm_device_model": None, "nrtm_device_serial": None,
        "nrtm_duration_frames": None, "nrtm_tc_fps": None, "nrtm_tc_halfstep": None,
        # excel/csv/json summaries
        "excel_sheet_count": None, "excel_sheets": [],
        "csv_row_count": None, "csv_columns": None,
        "json_keys": None,
        # flight metrics (series/path_{FREQ}hz는 meta_process 단계에서 확장)
        "flight_metrics": {}
    }

def fill_common(meta: Dict[str, Any], p: Path) -> None:
    try:
        meta["file_name"] = p.name
        meta["file_path"] = str(p)
        size = int(p.stat().st_size)
        meta["file_size"] = size
        meta["file_size_mb"] = round(size / (1024 * 1024), 6)
        meta["file_format"] = p.suffix.lstrip(".").lower()
        meta["modified_time"] = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception as e:
        log_error(f"[FILL_COMMON] {p}: {e}")

# ============ 메타 빌더(보조파일 요약 + 시계열 축적) ============
def build_metadata_for_media(media_path: Path, data_dir: Path) -> Dict[str, Any]:
    meta = empty_schema()
    fill_common(meta, media_path)

    # 1) 비디오/이미지 메타
    if is_video(media_path):
        extract_video(meta, media_path)
    elif is_image(media_path):
        extract_image(meta, media_path)

    # 2) 보조파일 찾기 (LRV/THM 제외)
    supp = find_supplementals(data_dir, media_path.stem)

    # 3) 보조파일 요약 + flight 시계열 누적
    series_accum = {"lat": [], "lon": [], "alt_m": [], "speed_mps": [], "t_rel_s": [], "t_clock_sod": []}

    # 자막: 요약 + 시계열 추출
    for p in supp.get("subtitle", []):
        raw = _read_text(p)
        if not raw: continue
        blocks = _parse_smi_blocks(raw) if p.suffix.lower()==".smi" else _parse_srt_lines(raw)
        if blocks:
            meta["subtitle_count"] = (meta.get("subtitle_count") or 0) + len(blocks)
            meta["subtitle_first_text"] = meta.get("subtitle_first_text") or blocks[0][1][:200]
            meta["subtitle_last_text"]  = blocks[-1][1][:200]
        s = parse_tabular_file(p)
        for k in series_accum.keys(): series_accum[k].extend(s.get(k, []))

    # XML 요약
    for p in supp.get("xml", []):
        parse_xml_summary(meta, p)

    # CSV 요약 + 시계열
    for p in supp.get("csv", []):
        parse_csv_summary(meta, p)
        s = parse_tabular_file(p)
        for k in series_accum.keys(): series_accum[k].extend(s.get(k, []))

    # EXCEL 요약 + 시계열
    for p in supp.get("excel", []):
        parse_excel_summary(meta, p)
        s = parse_tabular_file(p)
        for k in series_accum.keys(): series_accum[k].extend(s.get(k, []))

    # 자유 텍스트도 스캔(라인별 좌표 추출)
    for p in supp.get("txt", []):
        s = parse_tabular_file(p)
        for k in series_accum.keys(): series_accum[k].extend(s.get(k, []))

    # JSON 키 요약
    for p in supp.get("json", []):
        parse_json_summary(meta, p)

    meta["flight_metrics"] = {
        "source_files": sum([[str(x) for x in supp[k]] for k in supp], []),
        "series": series_accum
    }
    return meta

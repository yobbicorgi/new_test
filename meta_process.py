# ==============================
# meta_process.py (flat layout)
# ==============================
# - source/ 안의 미디어 파일마다 metadata/<stem>.json 생성
# - core.build_metadata_for_media()로 기본 메타 + 보조파일 요약/시계열 수집
# - 보간 경로(path_{FREQ}hz) 생성 후 저장
# - flight_metrics.series 원본 시계열 유지

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

from core import (
    build_metadata_for_media,
    interpolate_to_freq,
    is_video, is_image, set_verbose
)
from flight import compute_flight_metrics

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
DATA_DIR   = ROOT / "data"
META_DIR   = ROOT / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)
# 경로 보간 주파수(Hz). 1 Hz 로그도 이 값(기본 2 Hz)으로 보간됨.
FREQ = 2

def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    # 보기 좋은 들여쓰기 + UTF-8
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    set_verbose(True)

    if not SOURCE_DIR.exists():
        print("[WARN] source/ not found")
        return

    for media in sorted(SOURCE_DIR.iterdir()):
        if not media.is_file():
            continue
        # 비디오/이미지에만 메타 생성
        if not (is_video(media) or is_image(media)):
            continue

        # 1) 기본 메타 + 보조파일 요약 + 원시 시계열(series)
        meta = build_metadata_for_media(media, DATA_DIR)

        # 2) 비행 통계 계산
        fm = meta.get("flight_metrics") or {}
        supp_paths = fm.get("source_files") or []
        clip = {"duration_s": meta.get("duration")} if meta.get("duration") is not None else None
        if supp_paths:
            stats = compute_flight_metrics(supp_paths, media.stem, clip_by=clip)
            fm["stats"] = stats

        # 시계열 값 반올림
        series = fm.get("series") or {}
        def _round_list(xs, n):
            return [round(float(x), n) if x is not None else None for x in xs]
        if "lat" in series: series["lat"] = _round_list(series["lat"], 8)
        if "lon" in series: series["lon"] = _round_list(series["lon"], 8)
        if "alt_m" in series: series["alt_m"] = _round_list(series["alt_m"], 4)
        if "speed_mps" in series: series["speed_mps"] = _round_list(series["speed_mps"], 4)
        fm["series"] = series

        meta["flight_metrics"] = fm

        # 3) 보간 경로(path_{FREQ}hz)
        duration_s = meta.get("duration")
        # 항상 FREQ(기본 2 Hz)로 보간하여 균일한 좌표 시퀀스를 확보
        path_fhz = interpolate_to_freq(series, duration_s, freq=FREQ)

        # 4) 경로 및 주파수 기록
        fm[f"path_{FREQ}hz"] = path_fhz
        fm["path_freq_hz"] = FREQ

        # 5) 저장
        out_path = META_DIR / f"{media.stem}.json"
        _write_json(out_path, meta)
        print(f"[OK] metadata → {out_path.name}")

if __name__ == "__main__":
    main()

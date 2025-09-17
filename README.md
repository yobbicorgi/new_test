# Marine Mammal Detection and Viewer

해양 포유류를 탐지하고 결과를 시각화하는 파이프라인과 스트림릿(Viewer) 대시보드를 제공합니다.

## 주요 변경 사항
- `source/`에는 원본 영상(또는 이미지)을 그대로 두고, `data/`에는 같은 파일명 접두사를 공유하는 보조 데이터(JSON, CSV, SMI 등)를 배치합니다. 각 파일에 포함된 시각·날짜 정보가 존재하는 경우 자동으로 추출하고, 찾을 수 없을 때만 `Null`로 남깁니다.
- 1초를 초과해 검출된 객체만 메타데이터와 뷰어에 노출되며, 각 객체는 최소 1장의 스냅샷 이미지를 반드시 포함합니다.
- 뷰어 팝업에는 영상 재생 시간(Video time), 추출된 출현 일시, 위도·경도(DMS), 추정 크기(cm), 헤딩(°)이 한글로 표기되며 "이미지 보기" 버튼을 누르면 새 창에서 스냅샷을 확인할 수 있습니다.
- 왼쪽 컨트롤 패널은 전면 한글화되었고, 선택한 객체의 평균 지속 시간·크기·헤딩과 스냅샷 개수를 통계 카드로 제공합니다.

## 사용 방법
1. **환경 설정**
   ```bash
   pip install streamlit streamlit-folium folium pandas supervision ultralytics opencv-python
   ```
2. **데이터 배치**
   - 원본 영상·이미지는 `source/` 루트에 그대로 배치합니다. (예: `source/sample.mp4`)
   - 같은 대상에 대한 센서 로그나 자막 등 보조 데이터는 `data/`에 두며, 파일명 접두사가 영상과 동일하면 자동으로 연동됩니다. (예: `data/sample.smi`, `data/sample_track.csv`)
   - `main.py` 실행 시 결과 이미지는 `result_images/`, 메타데이터는 `metadata/` 아래에 생성됩니다.
3. **탐지 실행**
   ```bash
   python main.py
   ```
4. **뷰어 실행**
   ```bash
   streamlit run viewer.py
   ```
   - 좌측에서 영상과 객체를 선택하고, 팝업의 "📷 이미지 보기" 버튼으로 새 창에서 스냅샷을 확인할 수 있습니다.
   - 하단 "내보내기" 영역에서 선택된 객체 정보를 CSV 또는 GeoJSON으로 내려받을 수 있습니다.

## License
MIT
"# new_test" 
"# new_test" 

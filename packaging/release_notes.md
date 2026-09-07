<!--
Body prepended to the GitHub Release that .github/workflows/build.yml
creates when a v* tag is pushed; gh appends its auto-generated commit list
below it. This file is release-specific -- update it before tagging.
-->

## ⚠️ 이 버전부터 결과 수치가 달라집니다 / Numerical results change in this release

v0.1.1 이하에서 처리한 결과와 **직접 비교하지 마세요.** MATLAB 버전과 대조하는 과정에서
발견된 계산 오류들이 수정되어, **같은 원본 데이터에서 다른 값이 나옵니다.** 진행 중인
정량 분석이 있다면 이 버전으로 다시 처리하시기 바랍니다.

Do not compare output from this version against v0.1.1 or earlier. Correctness
fixes found by cross-checking against the MATLAB version change the numbers
produced from the same raw data; reprocess any in-progress quantitative work.

### 수치에 영향을 주는 수정 / Fixes that change values

- **검출기 마스크가 실제로 적용됩니다.** 이전에는 마스크 파일이 배포판에 포함되지 않아
  항상 무효과였습니다. 이제 Eiger4M은 약 6.2%, MarCCD/SX165는 약 21.6%의 불량 픽셀이
  평균 계산에서 제외됩니다.
- **캡톤(Kapton) 창 투과율 보정과 Eiger4M 실리콘 센서 흡수 보정이 모든 q-Cal 경로에
  적용됩니다.** 이전에는 GIXS-series의 "Change ai" 모드에서만 적용되어, 일반적인
  워크플로우 전체에서 q에 따라 세기가 어긋나 있었습니다.
- **Order / Temp.Sort 정렬이 데이터에도 반영됩니다.** 이전에는 X축만 정렬되고 프레임
  데이터는 파일 순서 그대로여서, 온도가 단조롭지 않은 시리즈(승온-냉각 등)에서 각
  프레임의 강도가 **다른 프레임의 온도와 짝지어져** 플롯·저장됐습니다.
- **오프셋 라인 컷(Pos move ≠ 0)의 X축**이 |Q| 대신 컷을 따라 실제로 변하는 성분
  (수평 컷은 Qxy, 수직 컷은 Qz)으로 바뀌었습니다. 피크 위치와 d-spacing이 달라집니다.
- **SDD Calibration의 Smart Auto Run** 피크 탐색 임계값이 빔스탑 영역을 제외한 구간
  기준으로 바뀌어, 이전에 링을 찾지 못하던 데이터에서도 정상 동작합니다.
- **SDD 캘리브레이션 저장 시 빔 중심**이 `.mat`/`.std`에서 한 픽셀 어긋나던 문제가
  수정됐습니다(MATLAB과 상호 호환).

### 동작이 바뀐 부분 / Behavior changes

- **Grid Remove**를 q-Map Cal. 이후에 실행해도 조용히 무시되던 문제가 수정됐습니다.
  단일 프레임 콘솔에서는 q-Map Cal.을 다시 실행해야 하고, 시리즈 탭에서는 q-Cal 이전에만
  실행할 수 있습니다.
- **q-Map 입력값(Energy/SDD/Center/FP/입사각)을 수정하면** q-Map Cal. 버튼의 완료
  표시가 해제되어, 다시 계산해야 함을 알려줍니다.
- **Auto Refresh** 직후 Save 1D/Azi/Integ.가 실패하던 문제가 수정됐습니다.
- GIXS-series/TrXS-series의 Play·Heatmap·ROI·Mask 창을 닫을 때 발생하던 오류가
  수정됐습니다.

자세한 내용은 저장소의 `docs/desktop_manual_ko.md`(또는 `.docx`) 매뉴얼을 참고하세요.

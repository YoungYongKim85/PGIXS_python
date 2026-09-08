<!--
Body prepended to the GitHub Release that .github/workflows/build.yml
creates when a v* tag is pushed; gh appends its auto-generated commit list
below it. This file is release-specific -- update it before tagging.
-->

## ⚠️ v0.2.2 — 결과 수치가 또 달라집니다 / Numerical results change again

v0.2.1 이하에서 처리한 결과와 **직접 비교하지 마세요.** MATLAB 버전과의 남은 차이를
맞추는 과정에서 계산이 바뀌어, **같은 원본 데이터에서 다른 값이 나옵니다.**

Do not compare output against v0.2.1 or earlier. Aligning the remaining differences
with the MATLAB version changes the numbers produced from the same raw data.

### 수치에 영향을 주는 수정 / Fixes that change values

- **모듈 갭 경계 3픽셀 제외** — 검출기 모듈 갭에 바로 붙은 픽셀은 전하 수집이 불완전하거나
  이웃 픽셀과 합산되어 값이 치우칠 수 있습니다. MATLAB은 원래 이 3픽셀 띠를 평균 계산에서
  제외해 왔는데, 이 포팅에는 없어서 그대로 커브에 들어가고 있었습니다. 이제
  **Circular Average / 1D·Cakecut·Accumulation 컷 / Azimuthal / missing wedge 리그리딩**에서
  제외합니다(Eiger4M 기준 제외 픽셀 6.2% → 약 7.9%).
  **2D 이미지 표시와 ROI 합계는 그대로입니다** — MATLAB도 여기에는 적용하지 않습니다.
  제외된 픽셀은 기하에는 남아 coverage를 떨어뜨리므로, 갭을 많이 지나는 q-bin은
  **Min coverage에 걸려 빠질 수 있습니다**(TrXS Circular Process).

- **SDD Smart Auto Run의 차수 판정** — 링들이 서로 같은 SDD를 주는지(일관성)를 우선하도록
  바꿨습니다. SDD가 짧아 1차 링이 빔스탑에 가려진 패턴에서, 이전에는 보이는 링을
  1·2·3차로 읽어 SDD가 6.7%나 벌어지는 해석을 골랐습니다. 이제 2·3차로 올바르게 읽습니다
  (같은 데이터에서 편차 0.04%). **이전 버전으로 구한 SDD 값을 다시 확인해 보시길 권합니다.**

### MATLAB 연동 / MATLAB interoperability

- MATLAB 버전(PGIXS_matlab)도 함께 수정됐습니다: TrXS의 `Min cov` 임계값이 실제로
  동작하게 되었고, SDD 캘리브레이션의 `.json` 빔 중심을 `.mat`/`.std`와 같은 **1-based로
  저장**합니다. 그 수정 이전의 MATLAB이 만든 `.json`을 Load Info로 불러오면 두 축 모두
  한 픽셀씩 어긋납니다 — 해당 파일은 MATLAB에서 다시 저장해 주세요.

---

## v0.2.1 (성능) / Performance

계산 결과는 v0.2.0과 동일했습니다. Circular Average 약 3.2배 속도 향상(100프레임 기준
약 231초 → 약 70초), 시리즈 메모리 약 50% 절감(Eiger4M 300프레임 기준 약 10.8GB → 약 5.4GB).

---

## ⚠️ v0.1.1 이하에서 올라오신다면 / Upgrading from v0.1.1 or earlier

v0.2.0에서 MATLAB 버전과 대조해 발견된 계산 오류들이 수정됐습니다. 위 항목에 더해 다음이
결과를 바꿉니다.

- **검출기 마스크가 실제로 적용됩니다.** 이전에는 마스크 파일이 배포판에 포함되지 않아
  항상 무효과였습니다(Eiger4M 약 6.2%, MarCCD/SX165 약 21.6% 제외).
- **캡톤(Kapton) 창 투과율 보정과 Eiger4M 실리콘 센서 흡수 보정이 모든 q-Cal 경로에
  적용됩니다.** 이전에는 GIXS-series의 "Change ai" 모드에서만 적용됐습니다.
- **Order / Temp.Sort 정렬이 데이터에도 반영됩니다.** 이전에는 X축만 정렬되어, 온도가
  단조롭지 않은 시리즈에서 각 프레임의 강도가 다른 프레임의 온도와 짝지어졌습니다.
- **오프셋 라인 컷(Pos move ≠ 0)의 X축**이 |Q| 대신 컷을 따라 변하는 성분(수평 컷은 Qxy,
  수직 컷은 Qz)으로 바뀌었습니다.
- **SDD 캘리브레이션 저장 시 빔 중심**이 `.mat`/`.std`에서 한 픽셀 어긋나던 문제 수정.

동작이 바뀐 부분: Grid Remove를 q-Map Cal. 이후 실행하는 경우의 처리, q-Map 입력값 변경 시
완료 표시 해제, Auto Refresh 직후 저장 실패 수정, Play·Heatmap·ROI·Mask 창 종료 시 오류 수정.

자세한 내용은 저장소의 `docs/desktop_manual_ko.md`(또는 `.docx`) 매뉴얼을 참고하세요.

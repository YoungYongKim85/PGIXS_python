<!--
Body prepended to the GitHub Release that .github/workflows/build.yml
creates when a v* tag is pushed; gh appends its auto-generated commit list
below it. This file is release-specific -- update it before tagging.
-->

## v0.3.14 — Mean이 실제 산술평균으로 / "Mean" is now the arithmetic mean

- 1D 처리에서 **Mean**을 고르면 이제 실제 **산술평균**을 냅니다. **결과 숫자가 바뀌는 수정입니다.**
  - 지금까지 Mean은 Huber 강건 평균이었습니다. 원형 평균, 라인컷, Azimuthal, 시리즈 처리, H5 1D export가 모두 해당됩니다.
  - Huber는 링의 일부 방위각에만 있는 신호를 이상치로 보고 깎습니다.
    그래서 배향된 박막이나 단결정 패턴을 원형 평균하면 Bragg 피크가 대부분 사라졌습니다.
    예를 들어 PTCDI-C13 GIWAXS의 0.25 Å⁻¹ 피크가 pyFAI 대비 약 1/250로 줄었습니다.
  - 등방적인 데이터는 결과가 0.1% 수준으로만 달라지고, 배향 시료는 사라졌던 피크가 돌아옵니다.
  - TrXS 원형 평균이 MATLAB과 같은 방식이 되어, 두 버전의 차이가 중앙값 0.47%에서 0.07%로 줄었습니다.
  - Std 오차막대는 표본 표준편차(N−1)입니다. 핫 픽셀을 걸러야 하면 **Median**을 고르세요.

"Mean" in every 1D reduction is now the plain arithmetic mean. It used to be a
Huber M-estimate, which removed most of the Bragg intensity from full-circle
averages of textured patterns. Mean-based results change: by ~0.1% for
isotropic data, and textured data recover their peaks.

---

## v0.3.13 — 입사각 메타데이터 IAngle 읽기 / Incidence angle stored as IAngle

- H5 파일의 입사각을 `IncAngle`뿐 아니라 **`IAngle`**(`/entry/instrument/user/params`)에서도 읽습니다.
  - 2026년 10월부터 Eiger 파일은 입사각을 `IAngle`로 기록해서, 지금까지는 이 파일들에서
    GIXS-series의 **Angle X축**이 값을 찾지 못했습니다.
  - 두 이름이 모두 있으면 `IncAngle`을 씁니다. 둘 다 없는 파일은 전처럼 값 없음(NaN)으로
    처리되어, 프레임 번호로 대신 표시됩니다.

The H5 metadata reader now takes the incidence angle from `IAngle` as well
as `IncAngle`, so angle scans written by the 2026-10 Eiger files get their
Angle X axis. `IncAngle` wins when both are present.

---

## v0.3.12 — Smart Auto Run 링 차수 인식 개선 / Smart Auto Run ring orders

- **SDD Calibration의 Smart Auto Run**이 링 차수를 더 정확히 붙입니다.
  - 전에는 링 후보를 순서대로 1, 2, 3차로 붙여서, 빔스톱 가장자리나 검출기 이음매 같은
    엉뚱한 피크가 하나 끼면 그 뒤 링이 모두 한 차수씩 밀려 SDD가 1.5~2.5배 틀렸습니다.
  - 이제 각 차수의 링이 있어야 할 반지름을 예측해 실제로 맞는 후보에만 차수를 붙이고,
    나머지 피크는 계산에서 뺍니다(후보 표에 "Not in the ring series"로 표시).
  - 테스트 데이터 표준시료 40개 중 39개가 저장된 SDD와 1% 이내로 맞습니다(전에는 28개).
- MATLAB 버전(v2.5.1)도 같은 방식으로 바뀌었습니다.

Smart Auto Run now matches ring candidates to the radii each order should
have, so a stray peak no longer shifts every later ring by one order.

---

## v0.3.11 — tif 방향·Angle X축 수정 / TIF orientation and Angle X axis fixes

- **tif 이미지 방향**: tif·mat·텍스트 이미지에 H5 전용 축 뒤집기가 잘못 적용되던 문제를 고쳤습니다.
  - Eiger4M tif가 열리지 않던 문제, MX170HS tif의 y·z가 바뀌어 중심이 어긋나던 문제가 해결됩니다.
  - SDD Calibration, GIXS·TrXS 콘솔, 배경 불러오기에 모두 적용됩니다.
  - 같은 표준시료의 h5와 tif가 똑같이 열립니다.
- **Angle 모드 X축**: H5에 각도가 기록되지 않은 시리즈에서 X축이 비어 있던 문제를 고쳤습니다.
  MATLAB처럼 프레임 번호(1, 2, 3, …)를 X축으로 씁니다.

TIF/MAT/text images now load in their own orientation (Eiger4M TIFs open,
MX170HS TIFs no longer swap y and z). Series X axes fall back to frame numbers
when the chosen value was never logged.

---

## v0.3.10 — 1D 플롯 Norm(Pt) / Norm(Pt) on the 1D plot

- **GIXS-series·TrXS-series 1D 뷰어에 Norm(Pt) 버튼**을 추가했습니다(MATLAB과 같은 기능).
  버튼을 누르고 1D 그래프에서 기준 Q를 클릭하면, 각 프레임을 그 Q에서의 값으로 나눠
  모든 곡선이 그 점에서 1이 됩니다. 버튼을 다시 누르면 원래대로 돌아갑니다.
  - Save 1D·Save Integ.·Integration·Heatmap에도 같은 값이 쓰입니다.
  - TrXS-series는 배경도 그 Q에서 1로 맞춘 뒤 뺍니다(MATLAB과 같음).
  - 고른 Q에 값이 없는 프레임(검출기 갭·마스크)은 그대로 두고, 상태줄에 몇 개인지 표시합니다.
  - 새로 Exec를 하면 꺼지고, Auto Refresh에서는 유지됩니다.

New Norm(Pt) button on the series 1D viewers: pick a Q and every frame is
scaled to 1 there (as in MATLAB). Press again to undo.

---

## v0.3.9 — GIXS-series X축 옵션 즉시 반영 / X-axis options apply at once

- **GIXS-series**의 X-Axis / Auto 패널에서 Sub-mode(LowTemp1/LowTemp2,
  Temperature1/2/3 등)를 바꾸거나 **Order**를 켜고 끄면, 1D·Azimuthal·Integration
  플롯이 바로 새 X축 순서와 색으로 다시 그려집니다. 전에는 **Set X-Axis**를 눌러야
  반영됐습니다. TrXS-series와 MATLAB은 원래 이렇게 동작했습니다.
- 계산 결과는 v0.3.8과 같습니다.

In GIXS-series, changing the Sub-mode or toggling Order now redraws the plots
immediately instead of waiting for Set X-Axis. Results are unchanged from v0.3.8.

---

## ⚠️ v0.3.8 — MATLAB과 결과 맞추기 / Results aligned with MATLAB

MATLAB 원본과 결과를 하나하나 비교해서, 숫자가 달라지던 곳을 고쳤습니다.
**IC/BC 정규화 값과 시간 X축 값의 크기가 이전 버전과 달라집니다.**

Fixes from a line-by-line comparison with the MATLAB original. Normalized
values and time-axis values change scale compared with v0.3.7.

### 고친 문제 / Bugs fixed
- **Heatmap (GIXS-series·TrXS-series).** Order나 Temp.Sort를 켜면 각 곡선이 다른
  프레임의 온도·각도 위치에 그려졌습니다. 이제 곡선 플롯과 같은 순서를 따릅니다.
- **Accumulation cut의 q축 (콘솔·GIXS-series).** y/z-pos로 띠를 옮기면 q가 그
  이동량만큼 커져서 피크가 높은 q에 찍혔습니다(예: z +200 px에서 약 12%).
  이제 MATLAB처럼 빔이 지나는 줄에서 q를 읽습니다.
- **Integration 범위.** 적분 그래프와 Save Integ.가 **Q Min-Max / Deg Min-Max**
  범위 안만 합합니다(MATLAB과 같음). GIXS-series에 두 칸이 새로 생겼고, 두 탭 모두
  Exec를 할 때마다 데이터 범위로 다시 채워집니다. 피크 하나만 적분하려면 범위를 좁히면 됩니다.

### MATLAB 방식으로 맞춘 것 / Aligned with MATLAB
- **IC/BC 정규화**는 첫 프레임의 카운터를 1로 놓은 상대값으로 나눕니다. 정규화한
  곡선도 첫 프레임 raw 값과 같은 크기로 나옵니다.
- **시간 X축.** GIXS-series에 **Exp. time**(초, 기본 1), TrXS-series에 **Time offset**
  (기본 1) 칸과 second/min/hour 단위 선택이 생겼습니다.
  - GIXS: `(t − t₁ + Exp. time) / 단위` (10초 간격이면 1, 11, 21 …)
  - TrXS: `(t − t₁) / 단위 + offset`
- **콘솔 입사각.** 빔 아래쪽을 클릭해도 같은 거리의 양수 각도가 나옵니다.

---

## v0.3.7 — SDD Send to Main을 MATLAB과 같게 / SDD Send to Main matches MATLAB

**계산 결과는 v0.3.6과 같습니다.** 축약 경로는 바뀌지 않았습니다.

- **SDD Calibration Send to Main:** Y_Plot/Z_Plot을 자를 때의 빔 중심을 기억해, Send to
  Main이 "그 중심 + 피팅 결과"를 보냅니다(MATLAB `OneDplotpixel_gui`와 같음). 피팅 뒤에
  중심을 옮겨도 올바른 피크 위치가 나옵니다. v0.3.6에서는 이 경우 "Fit a peak first."로
  거부했습니다.

SDD Send to Main now adds the fitted offset to the centre the Y/Z_Plot was cut at,
as MATLAB does, so moving the centre after the fit no longer matters.

---

## ⚠️ v0.3.6 — Auto Refresh 파일 선택·저장 이름·SDD 중심 수정 / Auto Refresh, save names, SDD centre

전체 로직 점검에서 찾은, **결과를 조용히 틀리게 만들던 문제**들을 고쳤습니다.

Fixes from a full logic review — mostly things that silently produced wrong results.

### GIXS-series·TrXS-series
- **Auto Refresh·Watch가 엉뚱한 파일을 붙이던 문제.** 새 파일을 "불러온 개수 다음
  순서"로만 골라서, 다음 경우에 틀렸습니다.
  - Set/Table에서 중간 파일 체크를 빼고 불러오면 같은 파일을 한 번 더 읽어, 이후 모든
    프레임이 X축과 한 칸씩 어긋났습니다.
  - 다른 이름으로 Set/Table 목록만 열어 보고 닫아도 그 시리즈 파일이 붙었습니다.
  - Directory 칸을 다른 폴더로 바꾸면 그 폴더의 같은 이름 파일이 붙었습니다.

  이제 불러온 시리즈의 폴더와 이름 기준으로, 마지막으로 불러온 파일 **다음** 파일만
  새 파일로 봅니다.
- **저장 파일 이름.** LOAD DATA로 연 파일 이름으로 저장되어, 다른 시리즈를 불러온 뒤에도
  예전 시리즈 폴더에 저장(덮어쓰기)되던 문제를 고쳤습니다.
- 시리즈를 불러올 때 Detector를 바꾸면, 이전 검출기의 Load Info는 다시 하도록 합니다.
- TrXS-series: 크기가 다른 background는 새 시리즈를 불러올 때 버립니다(Exec 오류 방지).
- 시간 오프셋 등에 잘못된 값을 넣으면 오류 대신 안내 메시지를 보여줍니다.

### SDD Calibration
- **Send to Main.** 피팅 후 중심을 옮기고 Send to Main을 누르면, 예전 중심 기준 피팅
  결과가 새 중심에 더해져 빔 중심이 크게 어긋났습니다. 이제 중심을 옮기면 예전 피팅을
  버립니다.
- Smart Auto Run의 링 분산을 MATLAB과 같이 표본 표준편차(N−1)로 계산합니다.
- `.json` 표준 파일에 빈 값(null)이 있으면 오류 대신 안내 메시지를 보여줍니다.

### 화면
- z-plot에서 피크를 고르지 않고 다른 곡선을 그린 뒤 클릭하면 입사각이 바뀌던 문제를
  고쳤습니다(콘솔·GIXS-series).
- 콘솔 상태줄의 coverage 값을 소수 셋째 자리로 표시하고, 성공한 뒤에도 남던 오류 문구를
  지웁니다.
- Detector가 파일과 맞지 않으면 알아보기 쉬운 메시지를 보여줍니다.

**영향 범위:** v0.3.5에서 중간 파일을 빼고 시리즈를 불러온 뒤 Auto Refresh/Watch를
썼거나, 여러 시리즈를 바꿔 가며 저장했다면 그 결과를 다시 확인하는 것을 권합니다.

---

## ⚠️ v0.3.5 — 시리즈 검출기 마스크 수정 / Series detector mask fixed

GIXS-series·TrXS-series 탭에서 **LOAD DATA 없이 바로 Set/Table → Load Series로**
시리즈를 불러오면, 검출기 마스크가 적용되지 않았습니다. 핫픽셀이 그대로 남고 모듈 갭
옆 3픽셀 버퍼도 빠져, 1D·Azi 결과가 같은 프레임을 콘솔에서 처리한 값과 달랐습니다
(Cooling 데이터에서 중앙값 약 0.6%, 갭 부분 NaN 구간 누락). LOAD DATA 뒤에 Detector
선택을 바꾼 경우에도 이전 검출기의 마스크가 쓰였습니다.

In the GIXS-series and TrXS-series tabs, a series imported straight from
Set/Table without LOAD DATA first was read with no detector mask: hot pixels
were kept and the 3-pixel gap buffer skipped, so curves differed from the
console's for the same frames (~0.6% median on the Cooling data, missing gap
NaNs). Changing the Detector box after LOAD DATA kept the old detector's mask.

- 이제 Load Series가 Detector 칸의 검출기로 마스크를 직접 만듭니다.
- **영향 범위:** LOAD DATA를 먼저 하고 Detector를 바꾸지 않았다면 결과는 v0.3.4와
  같습니다. 그렇지 않았다면 다시 처리하는 것을 권합니다.
- 수정 후 TrXS-series Circular 결과는 같은 프레임의 콘솔 Circular Average와 정확히
  일치합니다.

---

## v0.3.4 — in-situ 자동 갱신(Watch) / Automatic Auto Refresh for in-situ runs

**계산 결과는 v0.3.3과 같습니다.** 축약 경로는 바뀌지 않았습니다.

Numerically identical to v0.3.3 — the reduction path did not change.

### 새 기능: Watch / New: Watch

GIXS-series·TrXS-series 탭의 Auto Refresh 아래에 **"Watch every _N_ s"** 체크박스가
생겼습니다. 켜 두면 _N_초(기본 10초)마다 폴더를 확인해, **새 파일이 생겼을 때만**
Auto Refresh를 자동으로 실행합니다. in-situ 측정 중에 버튼을 반복해서 누를 필요가
없습니다.

A "Watch every N s" toggle under Auto Refresh polls the folder (default 10 s)
and runs Auto Refresh by itself whenever new frames have landed.

- **쓰는 중인 파일은 건너뜁니다.** 마지막 수정 후 3초가 지나지 않은 파일은 다음 확인
  때로 미룹니다. 검출기가 아직 쓰고 있는 파일을 읽어 빈(NaN) 프레임이 되는 것을 막습니다.
- 로딩·Exec·ROI 그리기 같은 다른 작업 중에는 그 회차를 건너뜁니다.
- 확장할 시리즈가 없으면 스스로 꺼지고 이유를 상태줄에 표시합니다.
- 수동 Auto Refresh 버튼은 이전과 같이 동작합니다.
- 웹 버전(PGIXS_web)에는 아직 없습니다.

---

## v0.3.3 — 입력값 유지, 배치 덮어쓰기 방지 / Typed values kept, batch overwrites refused

**계산 결과는 v0.3.2와 같습니다.** 축약 경로는 바뀌지 않았습니다.

Numerically identical to v0.3.2 — the reduction path did not change.

- **GIXS·TrXS 탭: LOAD DATA가 직접 입력한 값을 지우지 않습니다.** 이전에는 새 이미지를
  불러올 때 에너지·SDD·빔 중심·입사각 칸이 마지막 Load Info/q-Map 값으로 되돌아가,
  그 사이에 고친 값이 사라졌습니다. MATLAB 버전과 같이 이제 그대로 유지됩니다.
- **배치: 앞 job의 결과를 덮어쓰게 되는 job은 실행하지 않습니다.** 같은 입력과 같은
  `--out`으로 설정만 다른 GIXS/TrXS job(예: 입사각 두 가지)이 같은 파일 이름으로
  저장되어 나중 결과가 앞 결과를 덮어썼습니다. 이제 나중 job은 실패로 표시되고
  "별도의 `--out`을 지정하라"는 메시지가 나옵니다(`--dry-run`에서도 걸러집니다).
- **매뉴얼:** 갭 경계 버퍼 설명을 MATLAB v2.2.0 이후 동작(MATLAB도 재보간 전에 제외)에
  맞게 고쳤습니다.

---

## ⚠️ v0.3.2 — Q-space 1D 컷 결과가 달라집니다 / Q-space 1D cuts change

GIXS·TrXS 탭에서 **Map format을 Qxy/Qz 또는 Qy/Qz로 두고** 1D 컷(Line, Cakecut,
Accumulation, Azimuthal)을 하면, 이제 **검출기 모듈 갭 옆 3픽셀**(갭 경계 버퍼)을 빼고
Q 격자로 재보간합니다. 이전에는 이 픽셀들이 빠지지 않아, 값이 치우칠 수 있는 갭 옆
픽셀이 보간을 거쳐 커브에 섞였습니다.

In the GIXS and TrXS tabs, 1D cuts made with Map format set to Qxy/Qz or Qy/Qz
now drop the 3-pixel module-gap shoulder before regridding to the Q grid. Before,
those possibly biased pixels were interpolated into the curve.

- **영향 범위:** 갭을 지나는 bin만 값이 바뀝니다. Pixel 모드의 컷과 GIXS-series 결과는
  그대로입니다.
- **2D 이미지 표시는 바뀌지 않습니다.** Qxy/Qz 이미지와 Polar 이미지는 지금처럼 모든
  픽셀을 보여줍니다.
- 이제 Pixel 모드, Q-space 모드, GIXS-series의 Reflect missing wedge 모드가 모두 같은
  기준으로 갭 옆 픽셀을 제외합니다.

**MATLAB 버전과의 차이:** MATLAB은 Q 재보간 이미지에서는 이 버퍼를 적용하지 않습니다.
그래서 Q-space 컷에서 갭을 지나는 bin은 두 버전의 값이 다를 수 있습니다. 이 차이는
의도한 것입니다.

---

## ⚠️ v0.3.1 — 일부 결과가 달라집니다 / Some results change

v0.3.0 이후 남아 있던 작은 오류들을 정리했습니다. 대부분은 저장·불러오기·설정 관련이지만,
**아래 세 가지는 같은 원본 데이터에서 v0.3.0과 다른 값이 나올 수 있습니다.**

Small fixes left over after v0.3.0. Most concern saving, loading and settings,
but the three items below can change numbers produced from the same raw data.

### 수치에 영향을 주는 수정 / Fixes that change values

- **SDD Calibration의 coverage** — 검출기 갭 픽셀이 coverage 계산에서 빠지지 않아, 갭을
  지나는 링이 실제보다 덜 덮인 것으로 계산됐습니다. 그래서 Smart Auto Run의 coverage
  0.80 기준에 걸려 **좋은 링이 후보에서 탈락**할 수 있었습니다. 이제 갭은 계산에서
  제외되므로 **Smart Auto Run이 고르는 링(과 SDD)이 달라질 수 있습니다.**
- **시리즈 시간 축** — 타임스탬프가 하나라도 없으면 실제 시간값을 모두 버리고 일정한
  간격으로 바꿔 그렸습니다. 이제 빠진 프레임만 앞뒤 값으로 채웁니다. 타임스탬프가 아예
  없으면 0만 나오던 축 대신 **프레임 번호**를 씁니다.
- **반지름 프로파일** — 가장 바깥 픽셀이 정확히 정수 반지름에 있을 때 그 픽셀이 빠지던
  문제를 고쳤습니다(마지막 bin 하나에만 영향).

### 저장·불러오기 / Saving and loading

- **XTrans / Penetration Depth 저장 파일 이름에 화학식이 들어갑니다**(Penetration은
  에너지도). 예: `Transmittance_SiO2_5_to_15.dat`, `PenetrationDepth_Si_11.3keV_0_to_1.dat`.
  다른 물질을 저장해도 이전 파일을 덮어쓰지 않습니다.
- CSV 헤더에 쉼표가 들어간 라벨이 열을 한 칸씩 밀던 문제, 저장 값이 8번째 자리부터
  미세하게 바뀌던 문제(float32 정밀도)를 고쳤습니다.
- 프레임 번호가 9999를 넘는 긴 시리즈에서 `_10000`이 `_9999`보다 앞에 정렬되던 문제를
  고쳤습니다. Auto Refresh가 새 파일을 잘못 고를 수 있었습니다.
- v7.3 형식 `.mat` 이미지가 이전 형식 `.mat`과 전치된 방향으로 읽히던 문제를 고쳤습니다.
- H5 시리즈의 프레임을 하나도 읽지 못하면(데이터셋 경로가 틀린 경우 등) 빈 결과 대신
  오류를 표시합니다.
- 2D-info 저장이 일부 진단 데이터 형식 때문에 통째로 실패하던 문제를 고쳤습니다.

### 그 밖에 / Also

- GIXS-series의 입사각 변환(픽셀 ↔ 각도)은 Load Info를 먼저 해야 동작합니다.
- 워크플로 설정에서 `dataset`·`detector`·`metadata_mode`에 `null`을 쓰면 오류가 납니다.
- `export-1d`에서 `--out` 파일 확장자와 `--export-format`이 다르면 오류가 납니다.
- 배치에서 job마다 지정한 manifest가 공용 manifest에 덮이지 않습니다.

---

## ⚠️ v0.3.0 — 결과 수치가 달라집니다 / Numerical results change

전체 로직을 검토하다가 **결과값을 틀리게 만드는 오류**를 여러 개 찾아 고쳤습니다. 아래
기능을 쓰셨다면 **같은 원본 데이터에서 v0.2.4와 다른 값이 나옵니다.** 해당 결과는 다시
처리하시길 권합니다.

A full logic review found several bugs that produced wrong numbers. If you used
any of the features below, the same raw data now gives different results than
v0.2.4. Please reprocess those results.

### 수치에 영향을 주는 수정 / Fixes that change values

- **Grid Remove (Eiger4M)** — 보정할 grid line 위치가 **가로·세로가 뒤바뀐 채로**
  적용되고 있었습니다. 그래서 "적용됨"이라고 표시하면서도 실제 칩 경계선은 그대로
  두고, 엉뚱한 픽셀 행·열에 보정 계수를 곱했습니다. **Grid Remove를 켜고 처리한 모든
  결과**(GIXS·TrXS·두 series 탭·배치 워크플로)가 해당됩니다.
- **Accumulation 컷 / 오프셋 라인 컷** — 평균을 내는 픽셀 띠가 빔 중심에서 **1픽셀
  어긋나** 있었습니다(빔 행 25, 기본 ±1이면 원래 24–26행이어야 하는데 25–27행을
  사용). 이제 MATLAB `Accumpix_gi`와 같은 행을 씁니다.
- **IC/BC 정규화 실패 프레임** — 카운터가 없거나 0인 프레임이 정규화되지 않은 raw 값
  그대로 평균에 섞이고(예: 1.0이어야 할 값이 500.5), 워크플로는 성공으로 끝났습니다.
  이제 해당 프레임은 **NaN**으로 처리되어 평균에서 빠지고, 모든 프레임이 실패하면
  오류를 냅니다. GIXS-/TrXS-series에서도 카운터가 없는 프레임은 NaN이 됩니다.
- **GIXS-series Save Integ.** — 1D/Azimuthal 모드에서 Normalize 설정을 무시하고 raw
  합계를 저장했습니다. 이제 그래프와 같은 값이 저장됩니다.
- **TrXS-series 배경**
  - 시리즈를 다시 불러온 뒤 q-Cal을 하면 배경에 검출기 보정이 **두 번** 적용되었습니다.
  - 새 배경을 불러와도 다음 Exec 전까지 **이전 배경**을 계속 빼고 있었습니다.
  - IC/BC 정규화 중 한쪽에만 카운터가 있으면 정규화된 값에서 raw 값을 빼고 있었습니다.
    이제 차감을 건너뛰고 상태줄에 이유를 표시합니다.
- **에너지 정보가 없는 캘리브레이션** — 감쇠 길이를 아무 표시 없이 8 keV 기준으로
  계산했습니다. 이제 11.2 keV 기준입니다.
- **Change ai 시리즈** — 앞쪽에 픽셀을 하나도 못 고른 프레임이 있으면 행이 빠져, 이후
  모든 커브가 각도/온도와 **한 칸씩 어긋났습니다.**

### 동작이 바뀐 부분 / Behaviour changes

- **Load Info를 q-Cal 이후에 다시 하면** 이전 q-map과 결과가 지워지고 q-Cal을 다시
  하라는 안내가 나옵니다. 이전에는 새 빔 중심과 옛 q-map이 섞여 계산됐습니다.
- **Auto Refresh**는 Set / Table로 시리즈를 불러온 뒤에만 동작합니다. 새 프레임에도
  Grid Remove가 적용됩니다.
- **SDD Calibration**: Send to Main은 오프셋을 한 번만 적용합니다(두 번 누르면 두 배로
  이동하던 문제). 중심을 옮기면 예전 중심으로 구한 SDD 결과는 지워지므로, Save 전에
  Calc SDD를 다시 실행하세요.
- 가우시안 피팅이 **음수 baseline**(배경을 뺀 프로파일)에서도 동작합니다.
- 프레임이 아주 많은 H5 스택(예: 한 파일에 수천 프레임)을 이제 올바르게 읽습니다.

### 화면 / Desktop

- 로딩 중에는 창 전체가 잠겨, 로딩 도중 다른 버튼을 눌러 **프로그램이 강제 종료되던
  문제**가 없어졌습니다.
- 탭을 바꾼 뒤 GIXS에서 Line·Draw·클릭 위치가 TrXS의 q-map으로 계산되던 문제 수정.
- 새 이미지를 열면 항상 자동 콘트라스트로 시작합니다(이전 이미지 값 때문에 빈 화면처럼
  보이던 문제).
- 새 시리즈를 불러오면 Play 창이 닫힙니다.

### PGIXS_web

- 리포트·캘리브레이션 JSON이 `NaN` 대신 `null`을 씁니다(브라우저에서 읽을 수 있음).
  다시 불러오면 이전과 같이 동작합니다.
- 워크플로를 동시에 여러 개 실행해도 출력이 서로 섞이지 않습니다.

---

## v0.2.4 — 아이콘, 문서, 예제 데이터 / Icons, documentation, example data

**계산 결과는 v0.2.3과 완전히 동일합니다.** 축약 경로(q 매핑, 검출기 보정, 마스킹,
비닝, 평균) 코드는 한 줄도 바뀌지 않았습니다. v0.2.3에서 처리한 결과를 그대로
비교하셔도 됩니다.

Numerically identical to v0.2.3 — not one line of the reduction path changed.
Output from v0.2.3 remains directly comparable.

### 프로그램 아이콘 / Application icon

기존 아이콘이 흐릿했던 건 해상도만의 문제가 아니었습니다. 512픽셀 마스터 하나에서
모든 크기를 만들었는데, **그림이 캔버스의 6.3%**만 차지하고 배경이 투명했습니다.
그래서 실제로 표시되는 16·32픽셀에서 글자 높이가 3~7픽셀에 불과했고, 어두운
작업표시줄에서는 검은 글자가 배경에 묻혔습니다.

- **마스터를 1024픽셀로** 다시 그렸습니다. macOS `.icns`도 512를 늘린 게 아니라
  진짜 1024 항목을 담습니다.
- **불투명 배경**을 넣어 밝은 바탕과 어두운 바탕에서 똑같이 보입니다.
- **크기에 따라 다른 그림**을 씁니다. 48픽셀 이하에서는 산란 패턴 마크만
  (horizon 선·Bragg rod·in-plane 반사) 나와 16픽셀에서도 형태가 읽히고, 64픽셀
  이상에서 `p-GIXS` 워드마크가 함께 나옵니다.
- **실행 중인 창·작업표시줄 아이콘**도 고쳤습니다. 이전에는 399×150 가로 배너를
  정사각 아이콘 자리에 넣고 있어서 실행 파일 아이콘보다 오히려 나빴습니다.

### 새 문서 / New documentation

- **[영어 quickstart](../docs/quickstart.md)** — 캘리브레이션부터 q 매핑, 1D 컷,
  in-situ 시리즈까지 전체 흐름을 따라가는 안내서. **각 단계에서 나와야 하는 값**을
  같이 적어두어, 실제 측정에 쓰기 전에 프로그램이 제대로 도는지 확인할 수 있습니다.
- **예제 데이터 생성기** — `python examples/make_example_data.py` 한 줄이면 약 15MB의
  합성 프레임(silver behenate 캘리브런트 + 온도 램프 GIXS 3프레임)이 만들어집니다.
  실제 Eiger4M 기하와 빔라인 HDF5 레이아웃을 그대로 쓰므로 실제 데이터와 같은 경로로
  처리됩니다. **알려진 값으로 만들었기 때문에** 프로그램이 되돌려주는 SDD·피크 위치를
  검증할 수 있습니다(SDD 220.0 mm → 220.02 mm로 복원).
- 기여·문의 안내(`CONTRIBUTING.md`)와 **MIT 라이선스**를 추가했습니다.

### 그 밖에 / Also

- 테스트가 이제 모든 커밋에서 CI로 실행됩니다(632개). 이전에는 릴리즈 태그를 밀 때만
  패키징이 돌고 테스트는 돌지 않았습니다.
- 오래된 테스트 실패 하나를 정리했습니다. 코드 문제가 아니라 기댓값이 낡은 것이었고,
  그 과정에서 사라져 있던 `.std` 구형 레이아웃 파싱 검증을 되살렸습니다.
- 학술지(JOSS) 투고용 원고 초안이 저장소에 들어왔습니다(`paper.md`).

---

## ⚠️ v0.2.3 — Azimuthal / 배경 뺀 결과가 달라집니다 / Azimuthal and background-subtracted results change

검출기 모듈 갭 경계 3픽셀 제외가 **네 곳에서 빠져 있었습니다.** 영향 범위는 갭을
지나는 bin으로 좁지만, **같은 원본 데이터에서 이전과 다른 값이 나옵니다.**

The 3-pixel module-gap shoulder exclusion was missing from four reductions.
The affected range is narrow — only bins that cross a gap — but the same raw
data now produces different numbers.

### 수치에 영향을 주는 수정 / Fixes that change values

- **GIXS-series Azimuthal Process** — Exec Azi가 갭 경계 버퍼를 적용하지 않던 유일한
  축약 경로였습니다. 같은 탭의 1D 컷도, **Auto Refresh가 프레임을 덧붙일 때 도는 재실행
  경로도** 적용하고 있었으므로, Exec으로 만든 프레임과 Auto Refresh로 덧붙인 프레임이
  **서로 다른 방식으로 계산된 커브가 한 행렬에 섞여** 있었습니다. 실측(Eiger4M,
  r = 202 ± 1) 기준 180개 각도 bin 중 **4개**(모듈 갭 가장자리, 118.5–124.5°)가
  최대 21.8% 바뀌고, 2개 bin은 NaN이 되어야 했습니다.

- **TrXS-series Background Subtraction** — 배경 컷 세 개(Circular / 1D Line / Azimuthal)가
  모두 버퍼 없이 계산되고 있었습니다. 시료 커브는 갭 어깨 픽셀을 버린 값인데 배경은
  포함한 값이라, **Subtraction을 켜면 갭을 지나는 bin에서만 어긋난 뺄셈**이 되었습니다.
  아티팩트로 알아차리기 가장 어려운 자리입니다.

### 참고: "Reflect missing wedge"는 원래 값이 다릅니다 / Expected, not a bug

이번 수정은 GIXS-series Azimuthal의 **Reflect missing wedge 켜고/끈 차이를 없애지
않습니다.** 두 모드는 같은 값을 다르게 계산하는 게 아니라 **다른 질문에 답합니다.**

- 리그리딩한 이미지는 약 30%가 NaN입니다(픽셀 공간은 약 8%). 이 공백이 missing wedge
  그 자체이고, 그걸 보여주는 게 이 옵션의 목적입니다.
- Qxy ≈ 0, 즉 **90° 부근 bin은 통째로 비어 NaN이 될 수 있습니다** — 실측 데이터에서는
  커브 전체의 최대 피크가 여기에 있어서 그 피크가 사라집니다.
- 링의 물리적 위치도 약 0.4% 다르고(균일 q 격자 기준 반경 vs 픽셀 반경),
  bilinear 재샘플링 자체가 몇 % 산포를 만듭니다.

그러므로 **두 모드의 절대 강도를 직접 비교하지 마십시오.** 정량적인 강도가 필요하면
기본(픽셀) 모드를, 커버리지 공백을 확인하려면 wedge 모드를 쓰십시오.

### 개발자용 / For developers

`tools/matlab_crosscheck/`에 MATLAB 버전과 같은 데이터로 결과를 대조하는 검증 도구가
추가됐습니다. 매뉴얼 `docs/desktop_manual_ko.md`의 **10장 부록**에 사용법과, 두 버전
사이에서 어디까지가 정상 차이인지 정리돼 있습니다.

---

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

자세한 내용은 매뉴얼을 참고하세요 — 저장소의 `docs/desktop_manual_ko.md`, 또는 이 릴리즈에 첨부된 `desktop_manual_ko.docx`.

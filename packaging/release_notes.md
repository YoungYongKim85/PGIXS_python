<!--
Body prepended to the GitHub Release that .github/workflows/build.yml
creates when a v* tag is pushed; gh appends its auto-generated commit list
below it. This file is release-specific -- update it before tagging.
-->

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

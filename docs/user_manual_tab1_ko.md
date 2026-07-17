# PGIXS Python Tab1 사용자 매뉴얼

이 문서는 PGIXS Python Tab1 local web UI의 사용자용 매뉴얼이다. 현재 UI는
초기 scaffold 단계이며, 기능이 구현될 때마다 이 문서를 함께 갱신한다.

## 1. 프로그램 목적

PGIXS Python Tab1은 raw detector data를 불러와서 GIXS 또는 TrXS 방식으로
가공하고, Python-native `*_2Dinfo.h5` 파일과 1D export 결과를 만드는
프로그램이다.

주요 목표:

- MATLAB Tab1에서 사용하던 핵심 용어와 workflow를 유지한다.
- 화면 구조는 더 정리된 새 workflow로 재설계한다.
- Windows와 Linux에서 자연스럽게 동작하도록 한다.
- 나중에 hosted web 서비스로 확장할 수 있도록 local web UI 형태로 시작한다.

## 2. 기본 실행

Python 3.10 환경에서 실행한다.

```powershell
python -m pgixs_python.ui_backend.local_server
```

브라우저에서 아래 주소를 연다.

```text
http://127.0.0.1:8765/
```

포트를 바꾸고 싶으면 다음처럼 실행한다.

```powershell
python -m pgixs_python.ui_backend.local_server --port 8766
```

## 3. 화면 구성

초기 UI는 다음 영역으로 구성된다.

| 영역 | 목적 |
| --- | --- |
| Left sidebar | Batch config, shared manifest/log path, single job default 설정 |
| Main toolbar | Validate Batch, Run Batch, Add Job, Save Batch Config 실행 |
| Main tabs | Batch Queue, Run Summary, 2D Image, 1D Curve, Coverage, Report 확인 |
| Inspector | 선택한 workflow와 config field, metadata, warning 정보 확인 |
| Log drawer | 검증 메시지, 실행 로그, 오류 원인 확인 |

## 4. 기본 Workflow

1. 첫 화면은 `Batch Queue`로 시작한다.
2. Batch config JSON을 불러오거나 새 batch를 만든다.
3. `Add Job`으로 GIXS, TrXS, Export 1D job을 추가한다.
4. 각 job에 raw H5 file/folder, standard/calibration file, output folder를 지정한다.
5. 필요한 경우 normalization, background, q export option을 설정한다.
6. `Validate Batch`로 전체 batch config와 입력 옵션을 먼저 확인한다.
   이 단계는 실제 processing을 실행하지 않고 batch dry-run validation을 수행한다.
7. `Run Batch`로 processing을 실행한다.
   이 단계는 batch JSON에 들어 있는 job들을 순서대로 실행하고, log/summary path가
   지정되어 있으면 실행 로그와 요약 파일을 저장한다.
8. `Run Summary`에서 job별 성공/실패, exit code, message, log path를 확인한다.
9. `2D-info H5 preview`에 `*_2Dinfo.h5` 경로를 넣고 `Load 2D Preview`로
   작은 heatmap, shape, intensity range, q range, NaN fraction, calibration
   요약을 확인한다. Detector gap 또는 NaN 영역은 heatmap에서 어둡게 표시된다.
10. `1D export preview`에 `*_1D.csv` 또는 `*_1D.dat` 경로를 넣고
    `Load 1D Preview`로 q-intensity curve, q/intensity range, std/coverage
    column 여부를 확인한다. Coverage column이 있으면 low-coverage bin과
    abrupt coverage drop warning도 함께 표시된다.
11. `2D Image`, `1D Curve`, `Coverage`, `Report` tab에서 선택한 job의 결과를 확인한다.
12. 반복 사용이 필요한 batch 설정은 config/preset으로 저장한다.

## 5. MATLAB 용어와 Python UI 용어

화면 구조는 새로 설계하지만, 사용자가 익숙한 MATLAB 용어는 유지한다.

| MATLAB/기존 용어 | Python UI에서의 의미 |
| --- | --- |
| GIXS | Grazing-incidence geometry 기반 Tab1 processing mode |
| TrXS | Transmission geometry 기반 Tab1 processing mode |
| Standard | beam center, wavelength, SDD, pixel size 등의 calibration 정보 |
| Normalization | IC/BC/None normalization |
| Background | background image subtraction |
| Grid Remove | Eiger4M detector grid correction |
| Detector Correction | solid-angle/polarization 계열 correction |
| 2Dinfo | 처리된 2D image, q-map, metadata를 담은 H5 output |
| 1D Export | q-binned curve를 `.csv` 또는 `.dat`로 저장 |
| Coverage | detector gap/NaN 때문에 q-bin에 유효 pixel이 얼마나 있는지 나타내는 진단값 |

## 6. 출력 파일

주요 출력은 다음과 같다.

| 파일 | 설명 |
| --- | --- |
| `*_2Dinfo.h5` | Python-native 2D result. image, q-map, metadata, diagnostics 저장 |
| `*_1D.csv` 또는 `*_1D.dat` | q, intensity, optional standard deviation 저장 |
| `*_report.json` | 입력, 출력, option, calibration, warning, coverage diagnostics 저장 |
| `manifest.jsonl` | 여러 실행 결과를 UI/batch에서 추적하기 위한 compact log |
| `batch_summary.json` | batch workflow 실행 요약 |

## 7. Detector Gap과 Coverage Warning

Detector gap 또는 masked/NaN 영역 때문에 특정 q-bin에서 유효 pixel 수가 급격히
줄어들 수 있다. 이 경우 1D curve에서 kink 또는 움푹 파이는 현상이 생길 수 있다.

Python UI에서는 다음 정보를 보여주는 방향으로 설계한다.

- low coverage q-bin
- abrupt coverage drop 위치
- 1D curve와 coverage plot의 동시 확인
- JSON report에 warning 기록

Warning은 popup으로 사라지지 않고 `Run Summary`, `Coverage`, `Inspector`에
남겨서 사용자가 원인을 추적할 수 있게 한다.

## 8. Config와 Preset

Python UI는 내부적으로 JSON config를 사용한다. 이 config는 CLI와 UI가 공유하는
계약이다.

예상 사용 방식:

- 자주 쓰는 detector/calibration 조건을 preset으로 저장한다.
- q export option을 preset으로 저장한다.
- GIXS/TrXS workflow별 config를 불러와 반복 분석에 사용한다.

## 9. MATLAB-Python 비교

MATLAB 결과와 Python 결과를 비교할 때는 1D export 파일을 만든 뒤 다음 도구를
사용한다.

```powershell
python tools\compare_1d_exports.py `
  --reference validation_outputs\matlab_1d.csv `
  --candidate validation_outputs\python_1d.csv `
  --json-out validation_outputs\comparison_summary.json `
  --csv-out validation_outputs\comparison_points.csv
```

q-grid가 다르면 `--interpolate` 옵션을 사용할 수 있다.

## 10. 오류 확인

오류가 발생하면 우선 다음 순서로 확인한다.

1. `Log drawer`의 마지막 메시지
2. `Run Summary`의 warning/error
3. 생성된 `*_report.json`
4. 입력 file path와 standard file path
5. Python environment check

환경 확인:

```powershell
python tools\check_environment.py
```

빠른 검증:

```powershell
python tools\check_environment.py --run-tests
```

## 11. 현재 구현 상태

현재 구현된 UI 기반:

- local web server scaffold
- `/api/status`
- `/api/workflows`
- `/api/startup`
- workflow schema preview
- 기본 workbench shell
- Batch-first screen layout
- Validate Batch API 연결
- Run Batch API 연결
- Batch summary JSON 기반 결과 table
- 2D-info H5 compact preview API와 요약 table
- 2D-info H5 compact heatmap rendering
- 1D export CSV/DAT preview API와 SVG curve rendering
- 1D coverage diagnostics warning 표시

아직 구현 전인 기능:

- 실제 file picker
- full-resolution 2D image interaction
- interactive 1D curve tools
- dedicated coverage plot tab
- preset save/load
- batch result table

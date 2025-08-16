MLflow 로컬 템플릿 (CTA 세그멘테이션 예시)
이 저장소는 **MLflow(file 모드)**를 기본으로 하여,

재현 가능한 실험 관리

설정 기반(코드≠설정) 확장

나중에 공용 MLflow 서버로 손쉽게 이전
이 가능하도록 설계된 템플릿입니다.

🗂 폴더 구조
.
├─ README.md
├─ .gitignore
├─ .env.example                 # MLFLOW_TRACKING_URI / EXPERIMENT 등
├─ requirements.txt
├─ Makefile                     # (선택) macOS/Linux용 실행 단축키
├─ scripts/
│  ├─ run_train.sh              # macOS/Linux
│  ├─ run_eval.sh
│  └─ run_train.ps1             # Windows PowerShell
├─ configs/
│  ├─ tracking.yaml             # MLflow 추적/태그 설정
│  ├─ experiment.yaml           # 공통 시드/로그 레벨
│  ├─ data/cta.yaml             # 데이터 경로/전처리/분할
│  ├─ model/unet.yaml           # 모델 하이퍼파라미터
│  ├─ train.yaml                # 학습 하이퍼파라미터
│  └─ eval.yaml                 # 평가 설정
├─ data/                        # (gitignore) raw/interim/processed
├─ artifacts/                   # (gitignore) 모델/그림 등 산출물
├─ mlruns/                      # (gitignore) MLflow 로컬 저장소
├─ src/
│  ├─ pipelines/                # 얇고 안정적인 엔트리포인트
│  │  ├─ train.py
│  │  └─ eval.py
│  ├─ tracking/mlflow_utils.py  # set_experiment, log helpers
│  ├─ ml/
│  │  ├─ data/dataset.py        # Dataset/Dataloader
│  │  ├─ models/unet.py         # Mini U-Net (예시)
│  │  ├─ losses/dice.py
│  │  ├─ metrics/segmentation.py
│  │  └─ utils/{seed.py,saver.py,viz.py}
└─ tests/
   └─ test_smoke.py             # (선택) 1-epoch 스모크
원칙: 코드는 src/, 설정은 configs/. 실험=설정 조합으로 정의.

🚀 빠른 시작
1) 가상환경
macOS/Linux:

python3 -m venv .venv && source .venv/bin/activate
Windows(PowerShell):

powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

2) 패키지 설치
pip install -r requirements.txt
PyTorch는 OS/CUDA에 맞춰 설치가 필요할 수 있음. 설치가 안 될 땐 [PyTorch 공식 안내]의 커맨드로 설치 후 pip install mlflow numpy pyyaml tqdm 진행.

3) MLflow UI
mlflow ui --port 5000 --backend-store-uri ./mlruns
# 브라우저 → http://127.0.0.1:5000
4) 학습 실행
macOS/Linux:

bash
chmod +x scripts/run_train.sh
./scripts/run_train.sh
Windows:

powershell
.\scripts\run_train.ps1
UI에서 metrics(val_dice), **artifacts(models/unet_best.pt)**가 기록되는지 확인하세요.

⚙️ configs/ 파일 설명 & 사용법
모든 설정은 YAML로 관리되고 코드와 분리됩니다.

configs/tracking.yaml
yaml
복사
편집
# MLflow 추적/레지스트리 관련 설정 + 공통 태그
mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI:-file:./mlruns}
  experiment: ${MLFLOW_EXPERIMENT_NAME:-cta_vessel_seg}
tags:
  project: "cta-vessel"
  owner: "solo"
tracking_uri: 로컬은 file:./mlruns, 서버 전환 시 http://mlflow.internal:5000

experiment: 연구(프로젝트) 이름

tags: 모든 Run에 공통 태그 부착(project, owner 등)

configs/experiment.yaml
yaml
복사
편집
# 공통 실험 환경(재현성)
seed: 42
log_level: "INFO"
configs/data/cta.yaml
yaml
복사
편집
# 데이터 경로/전처리/분할 비율. 연구/버전별로 파일을 늘려가면 됨
root: "./data"
img_size: [256, 256]
train_split: 0.75
val_split: 0.15
test_split: 0.10
configs/model/unet.yaml
yaml
복사
편집
# 모델 하이퍼파라미터(이름만 바꿔도 새 모델로 교체 가능)
name: "unet"
in_channels: 1
out_channels: 1
base_channels: 16
dropout: 0.0
configs/train.yaml
yaml
복사
편집
# 학습 하이퍼파라미터
epochs: 3
batch_size: 4
lr: 1e-3
optimizer: "adam"
save_top_k: 1
configs/eval.yaml
yaml
복사
편집
# 평가 설정(배치/메트릭/데이터 버전 등 확장 예정)
batch_size: 4
변경 방법: 설정만 바꿔도 코드 수정 없이 실험 재현/변형 가능.

🧪 실험 실행/재현 규칙
Run 이름: unet_256_v1__YYYYMMDD_HHMM

필수 태그(자동/수동 혼용):

git.commit(자동)

project, owner, data.version, model.name, issue(선택)

src/tracking/mlflow_utils.py가 tracking_uri/experiment 설정 및
딕셔너리 파라미터 일괄 로깅을 도와줍니다.

➕ 새 모델 추가하는 법 (5분 컷)
파일 추가: src/ml/models/my_model.py 생성

python
복사
편집
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, **kw):
        super().__init__()
        # ... 네트워크 정의 ...
    def forward(self, x):
        # ... forward ...
        return x
설정 추가: configs/model/my_model.yaml

yaml
복사
편집
name: "my_model"
in_channels: 1
out_channels: 1
base_channels: 32
파이프라인에서 연결: (간단 버전) train.py에서 name 분기

python
복사
편집
mcfg = cfg["model"]
if mcfg["name"] == "unet":
    model = MiniUNet(...)
elif mcfg["name"] == "my_model":
    from src.ml.models.my_model import MyModel
    model = MyModel(in_channels=..., out_channels=...)
else:
    raise ValueError(f"Unknown model name: {mcfg['name']}")
(권장) models/factory.py에 레지스트리 딕셔너리로 모듈화하면 더 깔끔합니다.

학습 실행: configs/model/unet.yaml → my_model.yaml로 바꿔서 실행

기록된 params에 cfg.model.name=my_model이 남고,

metrics/artifacts는 동일한 규칙으로 저장됩니다.

📝 MLflow에 무엇을 기록해야 하나?
1) 파라미터 (Parameters)
모든 configs/(data/model/train/…)를 flatten하여 log_param

예: cfg.model.base_channels=32, cfg.train.lr=1e-3

2) 메트릭 (Metrics)
train_loss, val_loss, val_dice, val_iou 등 epoch 단위로 log_metric(step=epoch)

3) 아티팩트 (Artifacts)
모델 가중치(artifacts/models/*.pt)

시각화(학습 곡선, 예측-정답 비교 이미지)

로그/리포트(CSV, Markdown)

예시는 src/pipelines/train.py에 구현되어 있음.

👀 자주 하는 작업 시나리오
A. 해상도/배치만 바꿔 재학습
configs/data/cta.yaml → img_size 변경

configs/train.yaml → batch_size, epochs 조정

학습 실행 → UI에서 Run 비교

B. 데이터 버전 바뀔 때
configs/data/cta_v2.yaml 새로 만들고 root/분할 변경

tracking.yaml의 태그나 Run 이름에 data.version=v2 반영

Run 비교 시 필터링이 쉬워짐

C. 실험 메모 남기기
Run 상세 페이지 → Tags/Description에 적거나

코드에서 mlflow.set_tags({"note":"…"})로 기록

🧱 Git 사용 원칙
Git에 올리지 않을 것:

mlruns/, artifacts/, data/ (대용량/개인 환경)

커밋 메시지 규칙(예):

feat(model): add swin_unet, exp(train): lr=3e-4, img=256

PR 템플릿(협업 시): 재현 커맨드 + MLflow Run 링크 필수

☁️ 나중에 공용 MLflow 서버로 이전하기
지금: MLFLOW_TRACKING_URI=file:./mlruns (로컬)

나중: MLFLOW_TRACKING_URI=http://mlflow.internal:5000 로 환경변수만 교체

Artifact는 S3/MinIO/NAS로 설정 후, 버킷/경로를 프로젝트 prefix로 구분

기존 로컬 기록 이관이 필요하면 mlflow export-import 도구 사용

🧯 트러블슈팅
Torch 설치 에러: OS/파이썬/CUDA 매트릭스 확인 후 공식 커맨드 사용

아티팩트가 Git에 올라감: .gitignore 확인 (mlruns/, artifacts/, data/)

한글/공백 경로 문제: 프로젝트 경로는 영문·공백 없음 권장

UI에 실행이 안 보임: tracking_uri가 file:./mlruns인지 확인, 다른 경로면 UI 실행 시 같은 URI로 열기

✅ 체크리스트
 .venv 만들고 pip install -r requirements.txt

 mlflow ui로 대시보드 확인

 ./scripts/run_train.sh 또는 .\scripts\run_train.ps1 실행

 Run 페이지에서 params/metrics/artifacts 확인

 configs/만 바꿔 두 번째 실험을 만들어 비교

 (선택) models/factory.py를 도입해 모델 추가 더 쉽게 만들기

📚 부록: 최소 실행 커맨드 모음
bash
# UI 띄우기
mlflow ui --port 5000 --backend-store-uri ./mlruns
python -m mlflow ui --port 5000 --backend-store-uri ./mlruns
# 학습
python -m src.pipelines.train

# (예시) 환경변수로 experiment 바꾸고 실행
MLFLOW_EXPERIMENT_NAME=cta_vessel_seg_v2 python -m src.pipelines.train
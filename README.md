# MLflow Default Project

MLFlow 기본 환경

## 📦 설치 방법

### 1. 저장소 클론 (원하는 경로 이동 후)
```bash
git clone https://github.com/1wi11/mlflow-default.git
cd mlflow-default
```

### 2. 가상환경 생성 및 활성화
가상환경 생성
```bash
python3 -m venv 1wi11
```

가상환경 활성화
```bash
source 1wi11/bin/activate
.1wi11/bin/activate # 혹은
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. MLflow UI 실행
```bash
python -m mlflow ui --port 5000 --backend-store-uri ./mlruns
```

### 5. 학습 및 평가 실행
학습
```bash
python -m src.pipelines.train
```
- iris_demo Experiment가 생성되고, 모델이 학습됩니다.
- Accuracy, F1 Score 등이 MLflow에 기록됩니다.

평가
```bash
python -m src.pipelines.eval
```
- 가장 최근 Run을 불러와 평가합니다.
- 새로운 Run(eval_of_<run_id>)이 추가되고 Metric이 기록됩니다.

## 🗂 폴더 구조

```bash
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
   └─ test_smoke.py
```




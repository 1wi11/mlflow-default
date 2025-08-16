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

  





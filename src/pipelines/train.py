import os, io, yaml, mlflow, time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def yload(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_model(name: str, params: dict):
    if name == "logreg":
        return LogisticRegression(**params)
    if name == "rf":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unknown model: {name}")

def plot_confusion_matrix(cm, classes, out_png):
    plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    # 1) 설정 로드
    cfg_track = yload("configs/tracking.yaml")
    cfg_train = yload("configs/train.yaml")

    # 2) MLflow 준비
    mlflow.set_tracking_uri(cfg_track["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg_track["mlflow"]["experiment"])
    run_name = f'iris_{cfg_train["model"]["name"]}_{int(time.time())}'
    with mlflow.start_run(run_name=run_name):
        # 태그
        if "tags" in cfg_track:
            mlflow.set_tags(cfg_track["tags"])

        # 3) 데이터 로드 & 분할
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target
        classes = iris.target_names.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg_train.get("test_size", 0.2),
            random_state=cfg_train.get("seed", 42),
            stratify=y,
        )

        # 4) 모델 구성
        name = cfg_train["model"]["name"]
        params = cfg_train["model"].get("params", {})
        model = build_model(name, params)

        # 5) 학습
        model.fit(X_train, y_train)

        # 6) 예측/지표
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)

        # 7) 로깅: 파라미터/메트릭/아티팩트/모델
        # 파라미터
        mlflow.log_param("model.name", name)
        for k, v in params.items():
            mlflow.log_param(f"model.{k}", v)
        mlflow.log_param("test_size", cfg_train.get("test_size", 0.2))
        mlflow.log_param("seed", cfg_train.get("seed", 42))

        # 메트릭
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("f1_macro", f1m)

        # 혼동행렬 이미지
        os.makedirs("artifacts/figures", exist_ok=True)
        cm_png = "artifacts/figures/confusion_matrix.png"
        plot_confusion_matrix(cm, classes, cm_png)
        mlflow.log_artifact(cm_png, artifact_path="figures")

        # 테스트셋 저장 (eval에서 재사용)
        os.makedirs("artifacts/data", exist_ok=True)
        test_csv = "artifacts/data/test.csv"
        pd.DataFrame(
            np.column_stack([X_test.values, y_test.values]),
            columns=list(X.columns) + ["target"]
        ).to_csv(test_csv, index=False)
        mlflow.log_artifact(test_csv, artifact_path="data")

        # 모델 저장 (sklearn flavor)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[OK] acc={acc:.4f}, f1_macro={f1m:.4f}")

if __name__ == "__main__":
    main()

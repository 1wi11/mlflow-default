import os, yaml, mlflow, pandas as pd

def yload(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_target_experiment_id(experiment_name):
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {experiment_name}")
    return exp.experiment_id

def find_latest_run_id(experiment_id: str):
    df = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)
    if df.empty:
        raise RuntimeError("No runs found")
    return df.iloc[0]["run_id"]

def main():
    cfg_track = yload("configs/tracking.yaml")
    cfg_eval  = yload("configs/eval.yaml")

    mlflow.set_tracking_uri(cfg_track["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg_track["mlflow"]["experiment"])
    exp_name = cfg_track["mlflow"]["experiment"]
    exp_id = get_target_experiment_id(exp_name)

    # 1) 대상 Run 선택
    if cfg_eval.get("which_run", "latest") == "latest":
        run_id = find_latest_run_id(exp_id)
    else:
        run_id = cfg_eval["which_run"]

    print(f"[Eval] target run: {run_id}")

    # 2) 모델/테스트셋 로드
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # train에서 기록한 test.csv 사용
    client = mlflow.tracking.MlflowClient()
    local_test = client.download_artifacts(run_id, "data/test.csv", dst_path=".")
    df = pd.read_csv(local_test)
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    # 3) 평가지표
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    # 4) 평가 결과를 새 Run으로 남기고 싶다면:
    with mlflow.start_run(run_name=f"eval_of_{run_id}"):
        mlflow.set_tags({"project": "iris", "phase": "eval", "source_run": run_id})
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("f1_macro", f1m)

    print(f"[EVAL OK] acc={acc:.4f}, f1_macro={f1m:.4f}")

if __name__ == "__main__":
    main()

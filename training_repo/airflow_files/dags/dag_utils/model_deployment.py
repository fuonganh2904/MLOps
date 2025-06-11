
from mlflow.tracking import MlflowClient
import mlflow
import os
import json
from datetime import datetime

def register_model(model_uri: str, model_name: str, tags: dict = None, archive_existing: bool = True):
    """Đăng ký model, thêm tag (nếu có), và đưa lên Production.""" 
    client = MlflowClient()
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    if tags: 
        for key, value in tags.items():
            client.set_model_version_tag(name=model_name, version=result.version, key=key, value=value)
    client.transition_model_version_stage(
        name=model_name, version=result.version, stage="Production", archive_existing_versions=archive_existing
    )
    return result.version


def compare_models(new_run_id: str, experiment_name: str, metric_key: str = "accuracy") -> bool:
    """
    So sánh model mới với model tốt nhất trong cùng experiment.

    Args:
        new_run_id (str): Run ID của model mới vừa huấn luyện.
        experiment_name (str): Tên experiment.
        metric_key (str): Tên metric để so sánh (ví dụ: "accuracy").

    Returns:
        bool: True nếu model mới tốt hơn, False nếu không.
    """
    client = mlflow.tracking.MlflowClient()

    # Lấy thông tin model mới
    new_metrics = client.get_run(new_run_id).data.metrics
    new_metric_value = new_metrics.get(metric_key, None)
    if new_metric_value is None:
        raise ValueError(f"Metric '{metric_key}' không tồn tại trong run {new_run_id}")

    # Lấy top model trước đó trong cùng experiment
    experiment = client.get_experiment_by_name(experiment_name)
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric_key} DESC"],
        max_results=5,
    )

    all_runs = [run for run in all_runs if metric_key in run.data.metrics and run.info.run_id != new_run_id]

    if not all_runs:
        print("Không có model trước đó để so sánh — dùng model mới.")
        return True

    best_old_run = all_runs[0]
    best_old_value = best_old_run.data.metrics[metric_key]

    print(f"[Model cũ] {metric_key}: {best_old_value:.4f}")
    print(f"[Model mới] {metric_key}: {new_metric_value:.4f}")

    return new_metric_value > best_old_value


def create_model_documentation(run_id: str, model_name: str, doc_dir: str):
    """
    Tạo file JSON lưu thông tin tài liệu hóa cho model đã được đăng ký.

    Args:
        run_id (str): MLflow run ID của model.
        model_name (str): Tên model.
        doc_dir (str): Đường dẫn thư mục lưu document.
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # 1. Thu thập thông tin từ run
    params = run.data.params
    metrics = run.data.metrics
    tags = run.data.tags
    artifact_uri = run.info.artifact_uri
    start_time = datetime.fromtimestamp(run.info.start_time / 1000.0).isoformat()

    # 2. Tạo cấu trúc dữ liệu documentation
    doc = {
        "model_name": model_name,
        "run_id": run_id,
        "start_time": start_time,
        "artifact_uri": artifact_uri,
        "params": params,
        "metrics": metrics,
        "tags": tags,
        "description": tags.get("description", ""),
        "version": tags.get("version", ""),
        "source": tags.get("source", ""),
    }

    # 3. Tạo thư mục nếu chưa có
    os.makedirs(doc_dir, exist_ok=True)

    # 4. Lưu file
    doc_path = os.path.join(doc_dir, f"{run_id}_model_doc.json")
    with open(doc_path, "w") as f:
        json.dump(doc, f, indent=4)

    print(f"Model documentation saved to {doc_path}")
    return doc_path







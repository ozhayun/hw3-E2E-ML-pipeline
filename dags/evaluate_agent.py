"""
Airflow DAG: evaluate_agent

Runs mini-swe-agent on a SWE-bench subset and evaluates the results.
Pipeline: prepare_run -> run_agent -> run_eval -> summarize_and_log
"""

import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
from airflow.decorators import dag, task
from airflow.models.param import Param

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_run_config(params: dict) -> dict:
    raw_run_id = params.get("run_id", "")
    run_id = raw_run_id.strip() if raw_run_id else ""
    if not run_id:
        run_id = f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    model = (params.get("model") or "").strip() or "nebius/moonshotai/Kimi-K2.6"
    task_slice = (params.get("task_slice") or "").strip() or None
    cost_limit = (params.get("cost_limit") or "").strip() or None

    return {
        "run_id": run_id,
        "split": params["split"],
        "subset": params["subset"],
        "workers": int(params["workers"]),
        "model": model,
        "task_slice": task_slice,
        "cost_limit": cost_limit,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def prepare_run_dir(run_config: dict) -> Path:
    run_dir = RUNS_DIR / run_config["run_id"]
    (run_dir / "run-agent" / "trajectories").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    return run_dir


def run_agent_batch(run_config: dict, run_dir: Path) -> Path:
    agent_dir = run_dir / "run-agent"
    traj_dir = agent_dir / "trajectories"

    cmd = [
        "uv", "run", "mini-extra", "swebench",
        "--subset", run_config["subset"],
        "--split", run_config["split"],
        "--model", run_config["model"],
        "--workers", str(run_config["workers"]),
        "-o", str(traj_dir),
    ]

    # Optional: point at a local mini-swe-agent config if the clone is present
    local_config = PROJECT_ROOT / "mini-swe-agent" / "src" / "minisweagent" / "config" / "benchmarks" / "swebench.yaml"
    if local_config.exists():
        cmd += ["--config", str(local_config)]

    if run_config.get("task_slice"):
        cmd += ["--slice", run_config["task_slice"]]

    if run_config.get("cost_limit") is not None:
        cmd += ["--cost-limit", str(run_config["cost_limit"])]

    env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"}

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)

    # preds.json is written inside the trajectories dir by mini-extra swebench
    traj_preds = traj_dir / "preds.json"
    preds_path = agent_dir / "preds.json"
    if traj_preds.exists():
        shutil.copy2(traj_preds, preds_path)
    elif not preds_path.exists():
        raise FileNotFoundError(f"preds.json not found in {traj_dir} or {agent_dir}")

    return preds_path


def _dataset_name(subset: str) -> str:
    mapping = {
        "verified": "princeton-nlp/SWE-bench_Verified",
        "lite": "princeton-nlp/SWE-bench_Lite",
        "full": "princeton-nlp/SWE-bench",
    }
    return mapping.get(subset.lower(), f"princeton-nlp/SWE-bench_{subset.capitalize()}")


def run_swebench_eval(run_config: dict, preds_path: Path, run_dir: Path) -> Path:
    eval_dir = run_dir / "run-eval"

    cmd = [
        "uv", "run", "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", _dataset_name(run_config["subset"]),
        "--predictions_path", str(preds_path),
        "--max_workers", str(run_config["workers"]),
        "--run_id", run_config["run_id"],
    ]

    # Run in eval_dir so SWE-bench writes its output files there
    subprocess.run(cmd, cwd=str(eval_dir), check=True)

    return eval_dir


def collect_metrics(eval_dir: Path) -> dict:
    report_files = sorted(eval_dir.glob("*.json"))
    if not report_files:
        return {
            "resolved_instances": 0,
            "submitted_instances": 0,
            "total_instances": 0,
            "resolve_rate": 0.0,
        }

    report = json.loads(report_files[0].read_text())
    submitted = int(report.get("submitted_instances", 0))
    resolved = int(report.get("resolved_instances", 0))
    resolve_rate = resolved / submitted if submitted > 0 else 0.0

    return {
        "resolved_instances": resolved,
        "submitted_instances": submitted,
        "total_instances": int(report.get("total_instances", 0)),
        "resolve_rate": round(resolve_rate, 4),
        "error_instances": int(report.get("error_instances", 0)),
        "empty_patch_instances": int(report.get("empty_patch_instances", 0)),
    }


def log_mlflow_run(run_config: dict, metrics: dict, run_dir: Path) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("evaluate-agent")

    with mlflow.start_run(run_name=run_config["run_id"]) as active_run:
        mlflow.log_params({
            "run_id": run_config["run_id"],
            "split": run_config["split"],
            "subset": run_config["subset"],
            "workers": run_config["workers"],
            "model": run_config["model"],
            "task_slice": run_config.get("task_slice") or "all",
            "cost_limit": run_config.get("cost_limit") or "none",
        })
        mlflow.log_metrics(metrics)
        mlflow.log_param("artifact_local_path", str(run_dir))

        # Log config.json as an artifact for full provenance
        mlflow.log_artifact(str(run_dir / "config.json"), artifact_path="run")
        mlflow.log_artifact(str(run_dir / "metrics.json"), artifact_path="run")

        return active_run.info.run_id


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    params={
        "split": Param(
            "test",
            type="string",
            description="SWE-bench split to use: 'test' or 'dev'",
        ),
        "subset": Param(
            "verified",
            type="string",
            description="SWE-bench subset: 'verified', 'lite', or 'full'",
        ),
        "workers": Param(
            5,
            type="integer",
            description="Number of parallel workers for agent and evaluation",
        ),
        "model": Param(
            "nebius/moonshotai/Kimi-K2.6",
            type="string",
            description="Model identifier (e.g. nebius/moonshotai/Kimi-K2.6)",
        ),
        "task_slice": Param(
            "0:3",
            type="string",
            description="Python-style slice of tasks to run, e.g. '0:10'. Empty = all.",
        ),
        "run_id": Param(
            "",
            type=["string", "null"],
            description="Custom run ID. Auto-generated (run-YYYYMMDD-HHMMSS-xxxxxx) if empty.",
        ),
        "cost_limit": Param(
            "",
            type=["string", "null"],
            description="Max cost per task in USD. Empty = no limit.",
        ),
    },
)
def evaluate_agent():

    @task(task_id="prepare_run")
    def prepare_run(**context) -> dict:
        """Read Airflow params, create runs/<run-id>/config.json."""
        params = context["params"]
        run_config = build_run_config(params)
        run_dir = prepare_run_dir(run_config)
        print(f"[prepare_run] Run directory: {run_dir}")
        return {"run_config": run_config, "run_dir": str(run_dir)}

    @task(task_id="run_agent", execution_timeout=timedelta(hours=4))
    def run_agent(run_info: dict) -> dict:
        """Execute mini-swe-agent batch, write trajectories and preds.json."""
        run_config = run_info["run_config"]
        run_dir = Path(run_info["run_dir"])
        print(f"[run_agent] Starting agent run: {run_config['run_id']}")
        preds_path = run_agent_batch(run_config, run_dir)
        print(f"[run_agent] Predictions written to: {preds_path}")
        return {**run_info, "preds_path": str(preds_path)}

    @task(task_id="run_eval", execution_timeout=timedelta(hours=4))
    def run_eval(run_info: dict) -> dict:
        """Run SWE-bench evaluation on preds.json, write logs and reports."""
        run_config = run_info["run_config"]
        run_dir = Path(run_info["run_dir"])
        preds_path = Path(run_info["preds_path"])
        print(f"[run_eval] Evaluating predictions: {preds_path}")
        eval_dir = run_swebench_eval(run_config, preds_path, run_dir)
        print(f"[run_eval] Evaluation output: {eval_dir}")
        return {**run_info, "eval_dir": str(eval_dir)}

    @task(task_id="summarize_and_log")
    def summarize_and_log(run_info: dict) -> None:
        """Parse eval reports, write metrics.json + manifest.json, log to MLflow."""
        run_config = run_info["run_config"]
        run_dir = Path(run_info["run_dir"])
        eval_dir = Path(run_info["eval_dir"])

        metrics = collect_metrics(eval_dir)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        manifest = {
            "run_id": run_config["run_id"],
            "created_at": run_config["created_at"],
            "files": {
                "config": "config.json",
                "predictions": "run-agent/preds.json",
                "trajectories": "run-agent/trajectories/",
                "eval_logs": "run-eval/",
                "metrics": "metrics.json",
                "manifest": "manifest.json",
            },
            "mlflow_experiment": "evaluate-agent",
        }
        mlflow_run_id = log_mlflow_run(run_config, metrics, run_dir)
        manifest["mlflow_run_id"] = mlflow_run_id
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        print(f"[summarize_and_log] Run complete. Metrics: {metrics}")
        print(f"[summarize_and_log] MLflow run ID: {mlflow_run_id}")
        print(f"[summarize_and_log] Artifacts: {run_dir}")

    prepare = prepare_run()
    agent = run_agent(prepare)
    evaluation = run_eval(agent)
    summarize_and_log(evaluation)


evaluate_agent()

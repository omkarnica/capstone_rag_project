from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

_RUN_ID_RE = re.compile(r"^[\w.\-T:]+$")

_RESULTS_DIR = Path(__file__).parent.parent / "evals" / "results"

router = APIRouter(prefix="/eval", tags=["evaluation"])


def _results_dir() -> Path:
    return _RESULTS_DIR


def _load_run(run_id: str) -> dict[str, Any]:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    results = _results_dir()
    path = (results / f"{run_id}.json").resolve()
    if not str(path).startswith(str(results.resolve())):
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/runs")
def list_runs() -> dict[str, list[str]]:
    d = _results_dir()
    if not d.exists():
        return {"run_ids": []}
    run_ids = sorted(
        p.stem for p in d.glob("*.json") if p.name != ".gitkeep"
    )
    return {"run_ids": run_ids}


@router.get("/runs/latest")
def get_latest_run() -> dict[str, Any]:
    d = _results_dir()
    if not d.exists():
        raise HTTPException(status_code=404, detail="No runs found")
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    files = [f for f in files if f.name != ".gitkeep"]
    if not files:
        raise HTTPException(status_code=404, detail="No runs found")
    return json.loads(files[0].read_text(encoding="utf-8"))


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    return _load_run(run_id)


@router.get("/runs/{run_id}/summary")
def get_run_summary(run_id: str) -> dict[str, Any]:
    data = _load_run(run_id)
    summary: dict[str, Any] = {"run_id": run_id, "configs": {}}
    for config_name, tiers in data.get("configs", {}).items():
        all_metrics: dict[str, list[float]] = {}
        for tier_metrics in tiers.values():
            for metric, val in tier_metrics.items():
                all_metrics.setdefault(metric, []).append(val)
        summary["configs"][config_name] = {
            m: round(sum(vals) / len(vals), 4) for m, vals in all_metrics.items()
        }
    return summary


@router.get("/runs/{run_id}/ablation")
def get_run_ablation(run_id: str) -> dict[str, Any]:
    data = _load_run(run_id)
    return {
        "run_id": run_id,
        "configs": data.get("configs", {}),
        "baseline_delta": data.get("baseline_delta", {}),
    }


_DATASETS = {
    "default": "golden_queries.json",
    "aapl_retrieval": "aapl_retrieval_eval.json",
    "aapl_xbrl": "aapl_xbrl_eval.json",
}


@router.get("/datasets")
def list_datasets() -> dict[str, list[str]]:
    return {"datasets": list(_DATASETS.keys())}


@router.get("/datasets/{dataset_name}")
def get_dataset(dataset_name: str) -> dict:
    if dataset_name not in _DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_name}'. Available: {list(_DATASETS.keys())}")
    path = Path(__file__).parent.parent / "evals" / "dataset" / _DATASETS[dataset_name]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")
    queries = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset_name,
        "total": len(queries),
        "routes": {r: sum(1 for q in queries if q.get("route") == r) for r in sorted({q.get("route") for q in queries})},
        "queries": queries,
    }


@router.post("/trigger")
def trigger_eval(
    background_tasks: BackgroundTasks,
    skip_llm_metrics: bool = False,
    max_queries: int | None = None,
    routes: str | None = None,
    dataset: str = "default",
) -> dict[str, str]:
    from datetime import datetime
    from evals.configs.ablation_configs import ABLATION_CONFIGS
    from evals.runner import EvalRunner

    if dataset not in _DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{dataset}'. Available: {list(_DATASETS.keys())}")

    dataset_path = Path(__file__).parent.parent / "evals" / "dataset" / _DATASETS[dataset]
    if not dataset_path.exists():
        raise HTTPException(status_code=500, detail=f"Dataset file not found: {_DATASETS[dataset]}")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if max_queries is not None:
        dataset = dataset[:max_queries]
    if routes is not None:
        allowed = {r.strip() for r in routes.split(",")}
        dataset = [q for q in dataset if q.get("route") in allowed]
    if not dataset:
        raise HTTPException(status_code=400, detail="No queries match the requested filters")
    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")

    # Write skeleton immediately so the first poll doesn't 404
    results_dir = _results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    skeleton = {
        "run_id": run_id,
        "completed_at": None,
        "status": "running",
        "configs": {},
        "baseline_delta": {},
    }
    (results_dir / f"{run_id}.json").write_text(
        json.dumps(skeleton), encoding="utf-8"
    )

    def _run(rid: str) -> None:
        try:
            runner = EvalRunner()
            runner.run(configs=ABLATION_CONFIGS, dataset=dataset, run_id=rid, skip_llm_metrics=skip_llm_metrics, inter_query_delay=8.0)
        except Exception as exc:
            # Write error so the frontend stops polling instead of looping forever
            error_path = results_dir / f"{rid}.json"
            error_result = json.loads(error_path.read_text(encoding="utf-8")) if error_path.exists() else skeleton.copy()
            error_result["completed_at"] = datetime.utcnow().isoformat()
            error_result["status"] = "error"
            error_result["error"] = str(exc)
            error_path.write_text(json.dumps(error_result), encoding="utf-8")

    background_tasks.add_task(_run, run_id)
    return {"run_id": run_id, "status": "started"}

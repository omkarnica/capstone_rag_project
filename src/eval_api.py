from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

_RESULTS_DIR = Path(__file__).parent.parent / "evals" / "results"

router = APIRouter(prefix="/eval", tags=["evaluation"])


def _results_dir() -> Path:
    return _RESULTS_DIR


def _load_run(run_id: str) -> dict[str, Any]:
    path = _results_dir() / f"{run_id}.json"
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


@router.post("/trigger")
def trigger_eval(background_tasks: BackgroundTasks) -> dict[str, str]:
    from datetime import datetime
    from evals.configs.ablation_configs import ABLATION_CONFIGS
    from evals.runner import EvalRunner

    dataset_path = Path(__file__).parent.parent / "evals" / "dataset" / "golden_queries.json"
    if not dataset_path.exists():
        raise HTTPException(status_code=500, detail="Golden queries dataset not found")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")

    def _run():
        runner = EvalRunner()
        runner.run(configs=ABLATION_CONFIGS, dataset=dataset)

    background_tasks.add_task(_run)
    return {"run_id": run_id, "status": "started"}

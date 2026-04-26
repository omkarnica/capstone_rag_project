from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.graph import build_graph
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"


class EvalRunner:
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        configs: dict[str, dict],
        dataset: list[dict],
        skip_llm_metrics: bool = False,
        run_id: str | None = None,
    ) -> str:
        if run_id is None:
            run_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        result: dict[str, Any] = {
            "run_id": run_id,
            "completed_at": None,
            "configs": {},
            "baseline_delta": {},
        }

        for config_name, eval_config in configs.items():
            logger.info("Running ablation config: %s", config_name)
            config_results = self._run_config(
                config_name=config_name,
                eval_config=eval_config,
                dataset=dataset,
                skip_llm_metrics=skip_llm_metrics,
            )
            result["configs"][config_name] = config_results
            # Incremental write after each config completes
            self._write(run_id, result)

        result["completed_at"] = datetime.utcnow().isoformat()
        result["baseline_delta"] = self._compute_deltas(result["configs"])
        self._write(run_id, result)

        logger.info("Eval run %s complete. Results at %s/%s.json", run_id, self.output_dir, run_id)
        return run_id

    def _run_config(
        self,
        config_name: str,
        eval_config: dict,
        dataset: list[dict],
        skip_llm_metrics: bool,
    ) -> dict[str, dict]:
        graph = build_graph(eval_config=eval_config)
        tier_scores: dict[str, list[dict]] = defaultdict(list)

        for item in dataset:
            tier_key = f"tier_{item['tier']}"
            try:
                scores = self._evaluate_item(graph, item, skip_llm_metrics)
                tier_scores[tier_key].append(scores)
            except Exception as exc:
                logger.warning("Item %s failed in config %s: %s", item.get("id", "<unknown>"), config_name, exc)
                tier_scores[tier_key].append({})

        return {
            tier: _average_scores(score_list)
            for tier, score_list in tier_scores.items()
        }

    def _evaluate_item(self, graph, item: dict, skip_llm_metrics: bool) -> dict[str, float]:
        response = graph.invoke({
            "question": item["query"],
            "company": _extract_company(item["query"]),
        })

        answer = response.get("answer", "")
        citations = response.get("citations", [])
        retrieved_docs = response.get("retrieved_docs", [])
        retrieved_sources = [
            d.get("metadata", {}).get("source", "") for d in retrieved_docs
        ]

        scores: dict[str, float] = {}

        # --- Retrieval ranking metrics (no LLM) ---
        from evals.metrics.retrieval import MRRMetric, NDCGMetric
        scores["mrr"] = MRRMetric().compute(retrieved_sources, item.get("expected_sources", []))
        scores["ndcg"] = NDCGMetric().compute(retrieved_sources, item.get("expected_sources", []))

        # --- Domain metrics (no LLM) ---
        from evals.metrics.due_diligence import (
            NumericalAccuracyMetric,
            EntityMatchPrecisionMetric,
            CitationAccuracyMetric,
        )
        scores["numerical_accuracy"] = NumericalAccuracyMetric().compute(
            answer, item.get("expected_numbers", [])
        )
        scores["entity_match_precision"] = EntityMatchPrecisionMetric().compute(
            answer, item.get("expected_entities", [])
        )
        scores["citation_accuracy"] = CitationAccuracyMetric().compute(
            citations, item.get("expected_sources", [])
        )

        if not skip_llm_metrics:
            scores.update(self._run_llm_metrics(answer, item, retrieved_sources, scores))

        return scores

    def _run_llm_metrics(
        self,
        answer: str,
        item: dict,
        retrieved_sources: list[str],
        prior_scores: dict[str, float],
    ) -> dict[str, float]:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            HallucinationMetric,
        )
        from evals.metrics.gemini_judge import GeminiJudge
        from evals.metrics.due_diligence import (
            CompletenessMetric,
            ContradictionDetectionRate,
            DueDiligenceConfidenceScore,
        )

        judge = GeminiJudge()
        test_case = LLMTestCase(
            input=item["query"],
            actual_output=answer,
            expected_output=item.get("expected_answer", ""),
            retrieval_context=retrieved_sources,
            context=retrieved_sources,
        )

        scores: dict[str, float] = {}
        tier = item.get("tier", 1)

        faithfulness_metric = FaithfulnessMetric(model=judge, threshold=0.7)
        faithfulness_metric.measure(test_case)
        scores["faithfulness"] = faithfulness_metric.score

        relevancy_metric = AnswerRelevancyMetric(model=judge, threshold=0.7)
        relevancy_metric.measure(test_case)
        scores["answer_relevancy"] = relevancy_metric.score

        hallucination_metric = HallucinationMetric(model=judge, threshold=0.3)
        hallucination_metric.measure(test_case)
        scores["hallucination_rate"] = hallucination_metric.score

        if tier in (1, 2, 3):
            from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
            cp = ContextualPrecisionMetric(model=judge, threshold=0.7)
            cp.measure(test_case)
            scores["contextual_precision"] = cp.score

            cr = ContextualRecallMetric(model=judge, threshold=0.7)
            cr.measure(test_case)
            scores["contextual_recall"] = cr.score

        completeness = CompletenessMetric(model=judge)
        scores["completeness"] = completeness.compute(answer, item.get("expected_answer", ""))

        if tier == 4:
            cdr = ContradictionDetectionRate(model=judge)
            scores["contradiction_detection_rate"] = cdr.compute(
                answer, item.get("expected_contradictions", [])
            )

        scores["due_diligence_confidence"] = DueDiligenceConfidenceScore.weighted_score(
            faithfulness=scores.get("faithfulness", 0.0),
            completeness=scores.get("completeness", 0.0),
            numerical_accuracy=prior_scores.get("numerical_accuracy", 0.0),
            citation_accuracy=prior_scores.get("citation_accuracy", 0.0),
        )

        return scores

    def _compute_deltas(self, configs: dict) -> dict:
        deltas: dict = {}
        if "naive_rag" not in configs:
            return deltas
        baseline = configs["naive_rag"]
        for config_name, config_data in configs.items():
            if config_name == "naive_rag":
                continue
            delta: dict = {}
            for tier, tier_metrics in config_data.items():
                tier_baseline = baseline.get(tier, {})
                delta[tier] = {
                    metric: round(val - tier_baseline.get(metric, 0.0), 4)
                    for metric, val in tier_metrics.items()
                }
            deltas[f"{config_name}_vs_naive"] = delta
        return deltas

    def _write(self, run_id: str, result: dict) -> None:
        path = self.output_dir / f"{run_id}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _average_scores(score_list: list[dict]) -> dict[str, float]:
    if not score_list:
        return {}
    keys = {k for d in score_list for k in d}
    averages: dict[str, float] = {}
    for key in keys:
        vals = [d[key] for d in score_list if key in d]
        averages[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return averages


def _extract_company(query: str) -> str | None:
    query_lower = query.lower()
    if "apple" in query_lower or "aapl" in query_lower:
        return "Apple"
    if "microsoft" in query_lower or "msft" in query_lower:
        return "Microsoft"
    if "nvidia" in query_lower or "nvda" in query_lower:
        return "Nvidia"
    return None

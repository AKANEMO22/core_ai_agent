"""ID-labelled retrieval evaluation; unreviewed synthetic labels are never called gold."""

import json
import math
from pathlib import Path

import numpy as np

from .indexing import digest
from .schemas import GoldCase, RAGInterviewRequest


def retrieval_metrics(ids: list[str], relevance: dict[str, int], k=5):
    ids = list(dict.fromkeys(ids))[:k]
    relevant = {qid for qid, grade in relevance.items() if grade > 0}
    if not relevant:
        return {
            "recall_at_k": None,
            "precision_at_k": None,
            "ndcg_at_k": None,
            "mrr": None,
            "abstention_correct": float(not ids),
        }
    hits = sum(qid in relevant for qid in ids)
    dcg = sum(
        (2 ** relevance.get(qid, 0) - 1) / math.log2(i + 2) for i, qid in enumerate(ids)
    )
    ideal = sum(
        (2**grade - 1) / math.log2(i + 2)
        for i, grade in enumerate(sorted(relevance.values(), reverse=True)[:k])
    )
    mrr = next((1 / (i + 1) for i, qid in enumerate(ids) if qid in relevant), 0)
    return {
        "recall_at_k": hits / len(relevant),
        "precision_at_k": hits / k,
        "ndcg_at_k": dcg / ideal if ideal else 0,
        "mrr": mrr,
        "abstention_correct": None,
    }


def generate_cases(payload, count=150):
    roles = [
        r for r in ("IF_AI_SWE", "IF_BE_JAVA", "IF_FE") if r in payload["frameworks"]
    ]
    if not roles:
        raise ValueError("No PoC roles available")
    cases = []
    for role in roles:
        qs = [
            q
            for q in payload["questions"].values()
            if q["role_id"] == role and q["question_type"] == "technical"
        ]
        for i in range(math.ceil(count / len(roles))):
            question = qs[i % len(qs)]
            group = f"{role}:{question['question_id']}"
            # Entire question/paraphrase family stays in one split.
            split = "dev" if int(digest(group)[:8], 16) % 5 < 2 else "test"
            kind = (
                "source_question",
                "concept",
                "project_template",
                "multi_hop",
                "unanswerable",
            )[i % 5]
            concepts = [payload["concepts"][c] for c in question["concepts"]]
            query = question["question"]
            if kind == "concept" and concepts:
                query = " ".join(concepts)
            elif kind == "project_template":
                query = "Explain implementation tradeoffs in my project: " + (
                    " ".join(concepts) if concepts else question["question"]
                )
            relevance = {question["question_id"]: 2}
            if kind == "multi_hop":
                links = [
                    e
                    for e in payload["edges"]
                    if e["target"] in question["concepts"]
                    and e["source"] in payload["concepts"]
                ]
                if links:
                    query = payload["concepts"][links[0]["source"]]
                else:
                    kind = "source_question"
            if kind == "unanswerable":
                query = "zz_unavailable_domain_" + str(i)
                relevance = {}
            cases.append(
                GoldCase(
                    id=f"SYN_{role}_{i:03}",
                    split=split,
                    group_id=group,
                    query_type=kind,
                    request=RAGInterviewRequest(
                        role_id=role,
                        target_level=question["level"],
                        query=query,
                        top_k=5,
                        token_budget=24000,
                    ),
                    relevance=relevance,
                    answerable=bool(relevance),
                )
            )
    return cases[:count]


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [GoldCase.model_validate(row) for row in data]
    splits = {}
    for case in cases:
        if case.group_id in splits and splits[case.group_id] != case.split:
            raise ValueError("Group leakage across dev/test splits")
        splits[case.group_id] = case.split
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case IDs")
    return cases


def evaluate(engine, cases, split="test", require_reviewed=False):
    selected = [c for c in cases if c.split == split]
    if not selected:
        raise ValueError("No evaluation cases in requested split")
    if require_reviewed and any(c.label_origin != "human_reviewed" for c in selected):
        raise ValueError("Acceptance requires human-reviewed labels")
    records = []
    strategies = [
        "bm25",
        "dense",
        "text",
        "graph",
        "hybrid",
        "hybrid_no_expansion",
        "hybrid_no_planner",
    ]
    for case in selected:
        for strategy in strategies:
            request = case.request.model_copy(
                update={
                    "mode": "hybrid" if strategy.startswith("hybrid_") else strategy,
                    "include_prerequisites": strategy != "hybrid_no_expansion",
                    "graph_max_hops": 1 if strategy == "hybrid_no_expansion" else 3,
                    "plan_coverage": strategy != "hybrid_no_planner",
                    "rerank": False,
                    "session_id": None,
                }
            )
            result = engine.retrieve(request, persist=False)
            ids = [q.question_id for q in result.selected_questions]
            metrics = retrieval_metrics(ids, case.relevance, request.top_k)
            violations = sum(
                q.role_id != result.role_id or q.level != result.target_level
                for q in result.selected_questions
            )
            rubric_valid = all(
                q.rubric
                and q.sources
                and (
                    q.question_type != "technical"
                    or all(q.rubric.get(k) for k in ("poor", "acceptable", "excellent"))
                )
                for q in result.selected_questions
            )
            records.append(
                {
                    "case_id": case.id,
                    "group_id": case.group_id,
                    "role_id": request.role_id,
                    "query_type": case.query_type,
                    "strategy": strategy,
                    "retrieved_ids": ids,
                    "label_origin": case.label_origin,
                    **metrics,
                    "latency_ms": result.trace.get("total_ms", 0),
                    "guard_violations": violations,
                    "rubric_integrity": float(rubric_valid),
                    "faithfulness": None,
                }
            )
    summaries = {}
    metric_names = (
        "recall_at_k",
        "precision_at_k",
        "ndcg_at_k",
        "mrr",
        "abstention_correct",
        "rubric_integrity",
    )
    for strategy in strategies:
        rows = [r for r in records if r["strategy"] == strategy]
        summary = {
            m: float(np.mean([r[m] for r in rows if r[m] is not None]))
            if any(r[m] is not None for r in rows)
            else None
            for m in metric_names
        }
        summary.update(
            {
                "guard_violations": sum(r["guard_violations"] for r in rows),
                "p50_ms": float(np.percentile([r["latency_ms"] for r in rows], 50)),
                "p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
                "faithfulness": None,
                "judge_completion_rate": 0,
            }
        )
        summaries[strategy] = summary
    # Paired, group-bootstrap estimate: variants of the same question are not independent.
    differences = {}
    for row in records:
        if row["strategy"] == "hybrid" and row["ndcg_at_k"] is not None:
            baseline = next(
                r
                for r in records
                if r["case_id"] == row["case_id"] and r["strategy"] == "text"
            )
            differences.setdefault(row["group_id"], []).append(
                row["ndcg_at_k"] - baseline["ndcg_at_k"]
            )
    values = np.array([np.mean(v) for v in differences.values()])
    interval = None
    if len(values):
        rng = np.random.default_rng(42)
        samples = [
            float(np.mean(rng.choice(values, len(values), replace=True)))
            for _ in range(1000)
        ]
        interval = {
            "mean_delta_ndcg": float(np.mean(values)),
            "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
            "groups": len(values),
        }
    by_role = {}
    by_type = {}
    for dimension, output in (("role_id", by_role), ("query_type", by_type)):
        for group in {r[dimension] for r in records}:
            output[group] = {
                s: float(
                    np.mean(
                        [
                            r["ndcg_at_k"]
                            for r in records
                            if r[dimension] == group
                            and r["strategy"] == s
                            and r["ndcg_at_k"] is not None
                        ]
                    )
                )
                for s in strategies
                if any(
                    r[dimension] == group
                    and r["strategy"] == s
                    and r["ndcg_at_k"] is not None
                    for r in records
                )
            }
    return {
        "index_version": engine.version,
        "provider": engine.provider.signature,
        "split": split,
        "dense_min_score": engine.dense_min_score,
        "cases": len(selected),
        "acceptance_ready": all(c.label_origin == "human_reviewed" for c in selected),
        "summaries": summaries,
        "paired_graph_gain": interval,
        "by_role_ndcg": by_role,
        "by_query_type_ndcg": by_type,
        "records": records,
        "limitations": [
            "Synthetic labels are provisional and may omit other relevant questions",
            "Faithfulness requires generated responses and independent evidence checking",
            "No expert sign-off is inferred from these results",
        ],
    }


def markdown_report(report):
    rows = [
        "# Retrieval benchmark",
        "",
        f"Index: `{report['index_version']}`; provider: `{report['provider']}`.",
        f"Split: {report['split']}; cases: {report['cases']}; acceptance-ready labels: {report['acceptance_ready']}.",
        f"Dense minimum score: {report['dense_min_score']}.",
        "",
        "| Strategy | Recall@5 | Precision@5 | nDCG@5 | MRR | Abstention | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, metrics in report["summaries"].items():
        values = [
            "NA" if metrics[m] is None else f"{metrics[m]:.3f}"
            for m in (
                "recall_at_k",
                "precision_at_k",
                "ndcg_at_k",
                "mrr",
                "abstention_correct",
                "p95_ms",
            )
        ]
        rows.append("| " + strategy + " | " + " | ".join(values) + " |")
    rows += [
        "",
        "Faithfulness: NA (no generated-response grounding experiment).",
        "",
        "Paired graph gain: " + json.dumps(report["paired_graph_gain"]),
        "",
    ]
    rows += ["- " + limitation for limitation in report["limitations"]]
    return "\n".join(rows) + "\n"


def calibrate_dense(engine, cases):
    """Calibrate on development only; never use held-out queries to choose a threshold."""
    scores = {"answerable": [], "unanswerable": []}
    for case in cases:
        if case.split != "dev":
            continue
        request = case.request
        eligible = [
            i
            for i, qid in enumerate(engine.ids)
            if engine.questions[qid]["role_id"] == request.role_id
            and engine.questions[qid]["level"] == request.target_level
        ]
        if not eligible:
            continue
        vector = engine.provider.encode([request.query], query=True)[0]
        score = float(np.max(engine.vectors[eligible] @ vector))
        scores["answerable" if case.answerable else "unanswerable"].append(score)
    if not all(scores.values()):
        raise ValueError("Development set needs both answerable and unanswerable cases")
    threshold = min(1.0, max(scores["unanswerable"]) + 0.01)
    return {
        "split": "dev",
        "provider": engine.provider.signature,
        "index_version": engine.version,
        "recommended_min_score": threshold,
        "distribution": {
            kind: {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "median": float(np.median(values)),
            }
            for kind, values in scores.items()
        },
        "answerable_query_retention": sum(s >= threshold for s in scores["answerable"])
        / len(scores["answerable"]),
        "limitations": [
            "This threshold rejects synthetic OOD negatives, not all real-world unknown topics",
            "Use independent reviewed development queries before production calibration",
        ],
    }

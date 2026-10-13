import time
from collections import defaultdict, deque

import numpy as np
from rank_bm25 import BM25Okapi

from .ontology import mentions, tokens
from .schemas import EvidencePath, RetrievalHit, SourceRef


def bm25_retrieve(query, questions, texts, eligible, limit):
    ids = [qid for qid in questions if qid in eligible]
    if not ids or not tokens(query):
        return []
    corpus = [tokens(texts[qid]) for qid in ids]
    index = BM25Okapi(corpus)
    scores = index.get_scores(tokens(query))
    # BM25 idf can be zero on a tiny per-role pool. Exact overlap is a transparent tie-break.
    query_terms = set(tokens(query))
    ranked = sorted(
        (
            (qid, float(score), len(query_terms & set(corpus[i])))
            for i, (qid, score) in enumerate(zip(ids, scores))
        ),
        key=lambda x: (-x[1], -x[2], x[0]),
    )
    return [
        RetrievalHit(question_id=qid, channel="bm25", rank=i + 1, score=score)
        for i, (qid, score, overlap) in enumerate(
            [x for x in ranked if x[2] > 0][:limit]
        )
    ]


def dense_retrieve(query, provider, vectors, ids, eligible, limit, min_score=0.05):
    qvec = provider.encode([query], query=True)[0]
    if not np.isfinite(qvec).all() or np.linalg.norm(qvec) == 0:
        return []
    scores = vectors @ qvec
    ranked = sorted(
        (
            (qid, float(scores[i]))
            for i, qid in enumerate(ids)
            if qid in eligible and scores[i] >= min_score
        ),
        key=lambda x: (-x[1], x[0]),
    )[:limit]
    return [
        RetrievalHit(question_id=qid, channel="dense", rank=i + 1, score=score)
        for i, (qid, score) in enumerate(ranked)
    ]


class GraphRetriever:
    def __init__(self, payload):
        self.concepts = payload["concepts"]
        self.adjacency = defaultdict(list)
        for edge in payload["edges"]:
            self.adjacency[edge["source"]].append(edge)

    def retrieve(
        self,
        query,
        eligible,
        limit,
        *,
        include_prerequisites=True,
        max_hops=3,
        max_expansions=1000,
        timeout=0.25,
    ):
        seeds = [cid for cid, label in self.concepts.items() if mentions(query, label)]
        queue = deque((seed, [seed], [], [], 1.0) for seed in seeds)
        results = defaultdict(list)
        started, expansions = time.perf_counter(), 0
        truncated = False
        while queue:
            if expansions >= max_expansions or time.perf_counter() - started > timeout:
                truncated = True
                break
            node, nodes, relations, refs, confidence = queue.popleft()
            if len(relations) >= max_hops:
                continue
            for edge in self.adjacency[node][:40]:
                expansions += 1
                if edge["relation"] == "HAS_PREREQUISITE" and not include_prerequisites:
                    continue
                target = edge["target"]
                if target in nodes:
                    continue
                new_nodes = nodes + [target]
                new_relations = relations + [edge["relation"]]
                new_refs = refs + [SourceRef.model_validate(edge["ref"])]
                new_confidence = confidence * edge["confidence"]
                if edge["relation"] == "TESTED_BY":
                    if target in eligible and len(results[target]) < 3:
                        results[target].append(
                            EvidencePath(
                                nodes=new_nodes,
                                relations=new_relations,
                                sources=new_refs,
                                confidence=new_confidence,
                            )
                        )
                elif target in self.concepts:
                    queue.append(
                        (target, new_nodes, new_relations, new_refs, new_confidence)
                    )
        scored = [
            (qid, max(p.confidence / len(p.relations) for p in paths), paths)
            for qid, paths in results.items()
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [
            RetrievalHit(
                question_id=qid, channel="graph", rank=i + 1, score=score, paths=paths
            )
            for i, (qid, score, paths) in enumerate(scored[:limit])
        ], {"seeds": seeds, "expansions": expansions, "truncated": truncated}


def fuse(channels: dict[str, list[RetrievalHit]], weights=None, k=60):
    if k <= 0:
        raise ValueError("RRF smoothing must be positive")
    weights = weights or {}
    scores = defaultdict(float)
    for channel, hits in channels.items():
        seen = set()
        for hit in hits:
            if hit.question_id not in seen:
                scores[hit.question_id] += weights.get(channel, 1.0) / (k + hit.rank)
                seen.add(hit.question_id)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

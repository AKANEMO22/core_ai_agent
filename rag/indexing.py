import hashlib
import json
import re
from pathlib import Path

from .corpus import InterviewFramework, validate_frameworks
from .embedding import EmbeddingProvider, LSAEmbedding, NeuralEmbedding
from .gemini import GeminiEmbedding
from .ontology import (
    ONTOLOGY_VERSION,
    RELATIONS,
    SKILL_CONCEPTS,
    canonical,
    entity_id,
    mentions,
)
from .schemas import QuestionBundle, SourceRef
from .storage import RAGStore


def digest(data) -> str:
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


EXTRA_CONCEPTS = [
    "retrieval",
    "embedding",
    "recall",
    "precision",
    "faithfulness",
    "hallucination",
    "chunking",
    "reranking",
    "fine-tuning",
    "dependency injection",
    "inversion of control",
    "garbage collection",
    "polymorphism",
    "encapsulation",
    "inheritance",
    "abstraction",
    "transaction",
    "acid",
    "index",
    "b-tree",
    "deadlock",
    "concurrency",
    "thread",
    "jvm",
    "virtual dom",
    "hooks",
    "event loop",
    "promise",
    "closure",
    "accessibility",
    "seo",
    "http",
    "grpc",
    "rest",
    "authentication",
    "authorization",
    "container",
    "microservices",
    "system design",
    "normalization",
    "star schema",
    "partition",
    "sharding",
    "cache",
    "attention",
    "transformer",
    "backpropagation",
    "gradient",
    "overfitting",
    "cross validation",
    "unit testing",
    "integration testing",
    "latency",
    "throughput",
]


def build_index(
    store: RAGStore,
    data_dir: str | Path,
    provider: EmbeddingProvider | None = None,
    *,
    provider_kind="lsa",
    model=None,
    revision=None,
    dimensions=768,
) -> dict:
    data_dir = Path(data_dir)
    raw = json.loads(
        (data_dir / "interview_frameworks.json").read_text(encoding="utf-8")
    )
    frameworks = validate_frameworks(raw)
    if provider is None:
        if provider_kind == "lsa":
            provider = LSAEmbedding()
        elif provider_kind == "gemini":
            provider = GeminiEmbedding(
                model or "gemini-embedding-2",
                dimensions,
                cache_path=store.path.parent / "embedding_cache.db",
            )
        elif provider_kind == "neural":
            provider = NeuralEmbedding(model or "Qwen/Qwen3-Embedding-0.6B", revision)
        else:
            raise ValueError("Unsupported embedding provider")
    model = getattr(provider, "name", model)
    version = digest(
        {
            "data": raw,
            "ontology": ONTOLOGY_VERSION,
            "relations": RELATIONS,
            "provider": provider.signature,
            "builder": 5,
        }
    )[:24]
    try:
        _, existing = store.load(version)
        store.activate(version)
        return {**existing["audit"], "index_version": version, "cached": True}
    except KeyError:
        pass
    labels = {canonical(s) for s in set(EXTRA_CONCEPTS) | SKILL_CONCEPTS}
    labels.update(canonical(s) for a, _, b in RELATIONS for s in (a, b))
    labels.update(
        canonical(s)
        for fw in frameworks
        for s in fw.target_competencies
    )
    labels.update(
        canonical(s)
        for fw in frameworks
        for q in fw.technical_questions
        for s in q.key_concepts
    )
    concepts = {entity_id(label): label for label in sorted(labels) if label}
    questions, edges, ids, texts = {}, [], set(), []
    fw_map = {}
    for fw in frameworks:
        fw_map[fw.position_id] = fw.model_dump()
        behavioral_stages = [
            s.stage
            for s in fw.interview_stages
            if re.search(
                r"behavior|culture|soft|star|research mindset",
                s.stage_name,
                re.IGNORECASE,
            )
        ]
        technical_stages = [
            s.stage for s in fw.interview_stages if s.stage not in behavioral_stages
        ]
        for q in fw.technical_questions:
            qid = f"{fw.position_id}_{q.id}"
            if qid in ids:
                raise ValueError(f"Duplicate question ID: {qid}")
            ids.add(qid)
            if q.level not in ("Intern", "Fresher", "Junior"):
                raise ValueError(f"Unsupported question level: {qid}")
            if not q.expected_answer.strip() or not all(
                q.scoring_rubric.get(k, "").strip()
                for k in ("poor", "acceptable", "excellent")
            ):
                raise ValueError(f"Incomplete atomic question bundle: {qid}")
            # Extract only from question / declared concepts, not incidental rubric words.
            inferred = [
                cid
                for cid, label in concepts.items()
                if mentions(q.question + " " + " ".join(q.key_concepts), label)
            ]
            ref = SourceRef(
                locator=f"interview_frameworks.json#{fw.position_id}/technical_questions/{q.id}",
                content_hash=digest(q.model_dump()),
                url=q.citation_url,
            )
            bundle = QuestionBundle(
                question_id=qid,
                original_id=q.id,
                role_id=fw.position_id,
                level=q.level,
                applicable_levels=[q.level],
                stages=technical_stages,
                question=q.question,
                expected_answer=q.expected_answer,
                rubric=q.scoring_rubric,
                concepts=inferred,
                sources=[ref],
            )
            questions[qid] = bundle.model_dump()
            texts.append(
                f"{q.question}\nConcepts: {' '.join(concepts[c] for c in inferred)}\n{q.expected_answer}"
            )
            for cid in inferred:
                edges.append(
                    {
                        "source": cid,
                        "target": qid,
                        "relation": "TESTED_BY",
                        "confidence": 0.85,
                        "ref": ref.model_dump(),
                        "review_status": "rule_defined",
                    }
                )
            edges.extend(
                [
                    {
                        "source": fw.position_id,
                        "target": qid,
                        "relation": "HAS_QUESTION",
                        "confidence": 1.0,
                        "ref": ref.model_dump(),
                        "review_status": "source_mapping",
                    },
                    {
                        "source": qid,
                        "target": "rubric:" + qid,
                        "relation": "HAS_RUBRIC",
                        "confidence": 1.0,
                        "ref": ref.model_dump(),
                        "review_status": "source_mapping",
                    },
                ]
            )
        for q in fw.behavioral_questions:
            qid = f"{fw.position_id}_{q.id}"
            if qid in ids or not q.star_criteria:
                raise ValueError(f"Invalid behavioral question: {qid}")
            ids.add(qid)
            ref = SourceRef(
                locator=f"interview_frameworks.json#{fw.position_id}/behavioral_questions/{q.id}",
                content_hash=digest(q.model_dump()),
            )
            inferred = [
                cid
                for cid, label in concepts.items()
                if mentions(q.question + " " + q.evaluation_focus, label)
            ]
            bundle = QuestionBundle(
                question_id=qid,
                original_id=q.id,
                role_id=fw.position_id,
                level="Fresher",
                question_type="behavioral",
                applicable_levels=["Intern", "Fresher", "Junior"],
                stages=behavioral_stages or [1],
                question=q.question,
                expected_answer=q.evaluation_focus,
                rubric=q.star_criteria,
                concepts=inferred,
                sources=[ref],
            )
            questions[qid] = bundle.model_dump()
            texts.append(
                f"{q.question}\n{q.evaluation_focus}\n{' '.join(q.star_criteria.values())}"
            )
            for cid in inferred:
                edges.append(
                    {
                        "source": cid,
                        "target": qid,
                        "relation": "TESTED_BY",
                        "confidence": 0.85,
                        "ref": ref.model_dump(),
                        "review_status": "rule_defined",
                    }
                )
            edges.append(
                {
                    "source": qid,
                    "target": "rubric:" + qid,
                    "relation": "HAS_RUBRIC",
                    "confidence": 1,
                    "ref": ref.model_dump(),
                    "review_status": "source_mapping",
                }
            )
    for left, relation, right in RELATIONS:
        ref = SourceRef(
            locator=f"src/rag/ontology.py#{ONTOLOGY_VERSION}/{left}/{relation}/{right}",
            content_hash=digest([left, relation, right]),
            review_status="rule_defined",
        )
        edges.append(
            {
                "source": entity_id(left),
                "target": entity_id(right),
                "relation": relation,
                "confidence": 0.8,
                "ref": ref.model_dump(),
                "review_status": "rule_defined",
            }
        )
    # Explicit graph node inventory, including framework-level competencies/stages.
    # Candidate/project nodes are only attached in the private request overlay.
    nodes = {
        cid: {"id": cid, "kind": "Concept", "label": label}
        for cid, label in concepts.items()
    }
    for role_id, framework in fw_map.items():
        nodes[role_id] = {
            "id": role_id,
            "kind": "Role",
            "label": framework["role_title"],
        }
        for stage in framework["interview_stages"]:
            sid = f"stage:{role_id}:{stage['stage']}"
            nodes[sid] = {
                "id": sid,
                "kind": "InterviewStage",
                "label": stage["stage_name"],
            }
        for pillar in framework["evaluation_matrix"]:
            cid = "competency:" + digest([role_id, pillar["pillar"]])[:20]
            nodes[cid] = {
                "id": cid,
                "kind": "Competency",
                "label": pillar["pillar"],
                "weight_percent": pillar["weight_percent"],
            }
            ref = SourceRef(
                locator=f"interview_frameworks.json#{role_id}/evaluation_matrix/{pillar['pillar']}",
                content_hash=digest(pillar),
            )
            edges.append(
                {
                    "source": role_id,
                    "target": cid,
                    "relation": "REQUIRES",
                    "confidence": 1.0,
                    "ref": ref.model_dump(),
                    "review_status": "source_mapping",
                }
            )
    for qid, question in questions.items():
        nodes[qid] = {"id": qid, "kind": "Question", "label": question["question"]}
        rid = "rubric:" + qid
        nodes[rid] = {"id": rid, "kind": "Rubric", "label": rid}
        for level in question["applicable_levels"]:
            lid = "level:" + level
            nodes[lid] = {"id": lid, "kind": "Level", "label": level}
            edges.append(
                {
                    "source": qid,
                    "target": lid,
                    "relation": "APPLICABLE_TO",
                    "confidence": 1.0,
                    "ref": question["sources"][0],
                    "review_status": "source_mapping",
                }
            )
        for stage in question["stages"]:
            edges.append(
                {
                    "source": qid,
                    "target": f"stage:{question['role_id']}:{stage}",
                    "relation": "APPLICABLE_TO",
                    "confidence": 1.0,
                    "ref": question["sources"][0],
                    "review_status": "rule_defined",
                }
            )
    if any(e["source"] not in nodes or e["target"] not in nodes for e in edges):
        raise ValueError("Graph contains dangling node references")
    if isinstance(provider, LSAEmbedding):
        vectors = provider.fit(texts)
    else:
        vectors = provider.encode(texts)
    if vectors.shape[0] != len(questions):
        raise ValueError("Embedding row count mismatch")
    import numpy as np

    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite embeddings")
    audit = {
        "frameworks": len(frameworks),
        "technical_questions": sum(
            q["question_type"] == "technical" for q in questions.values()
        ),
        "behavioral_questions": sum(len(f.behavioral_questions) for f in frameworks),
        "indexed_bundles": len(questions),
        "concepts": len(concepts),
        "edges": len(edges),
        "nodes": len(nodes),
        "missing_external_question_citations": sum(
            not q["sources"][0]["url"] for q in questions.values()
        ),
        "human_reviewed_relations": 0,
        "provider": provider.signature,
    }
    store.publish(
        version,
        {
            "frameworks": fw_map,
            "questions": questions,
            "texts": texts,
            "concepts": concepts,
            "edges": edges,
            "nodes": nodes,
            "vectors": vectors.tolist(),
            "embedding_dimensions": vectors.shape[1],
            "embedding_state": provider.export()
            if isinstance(provider, LSAEmbedding)
            else None,
            "provider": provider.signature,
            "model": model if not isinstance(provider, LSAEmbedding) else None,
            "model_revision": getattr(provider, "resolved_revision", revision),
            "audit": audit,
        },
    )
    return {**audit, "index_version": version, "cached": False}

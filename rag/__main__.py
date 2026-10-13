import argparse
import json
from pathlib import Path

from .engine import RAGEngine
from .indexing import build_index
from .schemas import RAGInterviewRequest
from .storage import RAGStore

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="RoleUp RAG / GraphRAG / HybridRAG")
    parser.add_argument("--db", default=str(ROOT / "data/rag/index.db"))
    parser.add_argument("--data", default=str(ROOT / "data"))
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index")
    index.add_argument(
        "--provider", choices=["gemini", "neural", "lsa"], default="lsa"
    )
    index.add_argument("--model", default=None)
    index.add_argument("--dimensions", type=int, default=768)
    index.add_argument("--revision")
    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("--request", required=True, help="UTF-8 JSON file")
    commands.add_parser("audit")
    activate = commands.add_parser("activate")
    activate.add_argument("version")
    gold = commands.add_parser("make-eval")
    gold.add_argument("--output", default=str(ROOT / "data/rag_eval_cases.json"))
    gold.add_argument("--count", type=int, default=150)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--cases", default=str(ROOT / "data/rag_eval_cases.json"))
    evaluate.add_argument("--split", choices=["dev", "test"], default="test")
    evaluate.add_argument("--require-reviewed", action="store_true")
    evaluate.add_argument("--output", default=str(ROOT / "output/rag_benchmark"))
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--cases", default=str(ROOT / "data/rag_eval_cases.json"))
    parse = commands.add_parser("parse")
    parse.add_argument("path")
    parse.add_argument("--kind", choices=["cv", "jd"], required=True)
    grounding = commands.add_parser("evaluate-grounding")
    grounding.add_argument("--samples", required=True)
    args = parser.parse_args()
    if args.command == "evaluate-grounding":
        from .grounding import GroundingSample, evaluate_grounding

        result = evaluate_grounding(
            [
                GroundingSample.model_validate(s)
                for s in json.loads(Path(args.samples).read_text(encoding="utf-8"))
            ]
        )
    elif args.command == "parse":
        from .parsing import parse_document

        result = parse_document(args.path, args.kind)
    else:
        store = RAGStore(args.db)
        if args.command == "index":
            result = build_index(
                store,
                args.data,
                provider_kind=args.provider,
                model=args.model,
                revision=args.revision,
                dimensions=args.dimensions,
            )
        elif args.command == "activate":
            store.activate(args.version)
            result = {"activated": args.version}
        elif args.command == "audit":
            version, payload = store.load()
            result = {"index_version": version, **payload["audit"]}
        elif args.command == "make-eval":
            from .evaluation import generate_cases

            _, payload = store.load()
            cases = generate_cases(payload, args.count)
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    [c.model_dump(mode="json") for c in cases],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            result = {
                "output": str(path),
                "cases": len(cases),
                "label_origin": "synthetic_unreviewed",
            }
        elif args.command == "calibrate":
            from .evaluation import calibrate_dense, load_cases

            result = calibrate_dense(
                RAGEngine(store, args.data), load_cases(args.cases)
            )
        elif args.command == "evaluate":
            from .evaluation import evaluate, load_cases, markdown_report

            engine = RAGEngine(store, args.data)
            report = evaluate(
                engine, load_cases(args.cases), args.split, args.require_reviewed
            )
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.with_suffix(".json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            path.with_suffix(".md").write_text(
                markdown_report(report), encoding="utf-8"
            )
            result = {
                k: v
                for k, v in report.items()
                if k not in {"records", "by_role_ndcg", "by_query_type_ndcg"}
            }
        else:
            engine = RAGEngine(store, args.data)
            request = RAGInterviewRequest.model_validate_json(
                Path(args.request).read_text(encoding="utf-8")
            )
            result = engine.retrieve(request).model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

"""Run the question set in LangSmith, through the same graphs as the app.

The script ingests eval/corpus plus the prompt-injection fixture, sends each
question through the answer graph, and records scores on a LangSmith experiment.

Needs a running Postgres (pgvector), plus these environment variables:
DATABASE_URL, JWT_SECRET, GEMINI_API_KEY, LANGSMITH_API_KEY.

    pip install -r requirements.txt -r eval/requirements.txt
    python eval/run_eval.py
    python eval/run_eval.py --chunk-size 400

The second command is the one-variable experiment: same questions, same model,
same number of passages, only the chunk window changes. Compare the two
experiments in the LangSmith project.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
CORPUS_DIR = ROOT / "eval" / "corpus"
INJECTION_PATH = ROOT / "eval" / "fixtures" / "prompt-injection.md"
DATASET_NAME = "documind-questions"
EVAL_USERNAME = "documind-eval"
EVAL_EMAIL = "documind-eval@example.com"
# Local fixture account only. It is not a deployed password.
EVAL_PASSWORD = "eval-only-local"


def load_env_file(path: Path) -> None:
    """Copy KEY=VALUE lines into the process when they are not already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_questions() -> list[dict]:
    rows = []
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _files_to_ingest() -> list[Path]:
    files = sorted(CORPUS_DIR.glob("*.md"))
    files.append(INJECTION_PATH)
    return files


def ensure_eval_user():
    """Create the local eval user if this database does not have one yet."""
    from sqlalchemy import select

    from app.auth import hash_password
    from app.db import SessionLocal, init_db
    from app.models import User

    init_db()
    with SessionLocal() as session:
        user = session.scalar(select(User).where(User.username == EVAL_USERNAME))
        if user is None:
            user = User(
                username=EVAL_USERNAME,
                email=EVAL_EMAIL,
                password_hash=hash_password(EVAL_PASSWORD),
            )
            session.add(user)
            session.commit()
            session.refresh(user)
        return user.id


def ingest_corpus(user_id: uuid.UUID, chunk_size: int) -> None:
    """Replace this user's eval files and run the ingest graph on each one."""
    import app.services.chunk as chunk_module
    from sqlalchemy import delete, select

    from app.db import SessionLocal
    from app.graphs.ingest import graph as ingest_graph
    from app.models import Document
    from app.services.storage import document_file_path

    if chunk_size <= chunk_module.OVERLAP_SIZE:
        raise SystemExit("Chunk size must be larger than the overlap.")
    chunk_module.WINDOW_SIZE = chunk_size

    with SessionLocal() as session:
        old_ids = session.scalars(select(Document.id).where(Document.user_id == user_id)).all()
        for old_id in old_ids:
            old_file = document_file_path(str(old_id))
            if old_file.is_file():
                old_file.unlink()
        session.execute(delete(Document).where(Document.user_id == user_id))
        session.commit()

    for path in _files_to_ingest():
        file_bytes = path.read_bytes()
        document_id = uuid.uuid4()
        with SessionLocal() as session:
            session.add(
                Document(
                    id=document_id,
                    user_id=user_id,
                    filename=path.name,
                    media_type="text/markdown",
                    storage_key=str(document_id),
                    status="queued",
                    error=None,
                    byte_size=len(file_bytes),
                )
            )
            session.commit()
        document_file_path(str(document_id)).write_bytes(file_bytes)
        ingest_graph.invoke({"document_id": str(document_id)})
        with SessionLocal() as session:
            document = session.scalar(select(Document).where(Document.id == document_id))
            status = document.status if document is not None else "missing"
            reason = document.error if document is not None else None
        if status != "ready":
            raise SystemExit(f"Ingest failed for {path.name}: {status} {reason or ''}".strip())
        print(f"ready {path.name}")


def ready_document_ids(user_id: uuid.UUID) -> list[str]:
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Document

    with SessionLocal() as session:
        rows = session.scalars(
            select(Document.id).where(Document.user_id == user_id, Document.status == "ready")
        ).all()
    return [str(document_id) for document_id in rows]


def sync_dataset(client, questions: list[dict], refresh: bool) -> None:
    """Create the LangSmith dataset from questions.jsonl when it is missing."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        if not refresh:
            return
        existing = client.read_dataset(dataset_name=DATASET_NAME)
        client.delete_dataset(dataset_id=existing.id)
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="DocuMind public eval questions. Passages are data, not instructions.",
    )
    client.create_examples(
        inputs=[{"id": row["id"], "question": row["question"]} for row in questions],
        outputs=[
            {"expect": row["expect"], "gold_substring": row.get("gold_substring") or ""}
            for row in questions
        ],
        dataset_id=dataset.id,
    )


def _score(key: str, value: float | None) -> dict:
    return {"key": key, "score": value}


def build_evaluators():
    from eval.score import (
        answer_correct,
        correct_refusal,
        false_refusal,
        injection_held,
        retrieval_hit,
    )

    def retrieval_hit_eval(outputs: dict, reference_outputs: dict) -> dict:
        return _score(
            "retrieval_hit",
            retrieval_hit(
                outputs.get("retrieved_text") or "",
                reference_outputs.get("expect") or "",
                reference_outputs.get("gold_substring") or "",
            ),
        )

    def answer_correct_eval(outputs: dict, reference_outputs: dict) -> dict:
        return _score(
            "answer_correct",
            answer_correct(
                outputs.get("answer") or "",
                reference_outputs.get("expect") or "",
                reference_outputs.get("gold_substring") or "",
            ),
        )

    def correct_refusal_eval(outputs: dict, reference_outputs: dict) -> dict:
        return _score(
            "correct_refusal",
            correct_refusal(bool(outputs.get("refused")), reference_outputs.get("expect") or ""),
        )

    def false_refusal_eval(outputs: dict, reference_outputs: dict) -> dict:
        return _score(
            "false_refusal",
            false_refusal(bool(outputs.get("refused")), reference_outputs.get("expect") or ""),
        )

    def injection_held_eval(outputs: dict, reference_outputs: dict) -> dict:
        return _score(
            "injection_held",
            injection_held(outputs.get("answer") or "", reference_outputs.get("expect") or ""),
        )

    def latency_eval(outputs: dict, reference_outputs: dict) -> dict:
        return _score("latency_ms", float(outputs.get("latency_ms") or 0))

    return [
        retrieval_hit_eval,
        answer_correct_eval,
        correct_refusal_eval,
        false_refusal_eval,
        injection_held_eval,
        latency_eval,
    ]


def ask_question(user_id: uuid.UUID, document_ids: list[str], inputs: dict) -> dict:
    """One answer-graph call. Gemini runs only inside that graph."""
    from app.graphs.answer import graph as answer_graph

    started_at = time.perf_counter()
    state = answer_graph.invoke(
        {
            "user_id": str(user_id),
            "question": inputs["question"],
            "document_ids": document_ids,
            "started_at": started_at,
        }
    )
    retrieved = "\n".join(
        passage.get("content") or "" for passage in state.get("retrieved_chunks") or []
    )
    return {
        "answer": state.get("answer") or "",
        "refused": bool(state.get("refused")),
        "retrieved_text": retrieved,
        "latency_ms": int(state.get("latency_ms") or 0),
    }


def _result_scores(item) -> list[tuple[str, float | None]]:
    evaluation = item["evaluation_results"] if isinstance(item, dict) else item.evaluation_results
    results = evaluation["results"] if isinstance(evaluation, dict) else evaluation.results
    scored = []
    for result in results:
        if isinstance(result, dict):
            scored.append((result.get("key"), result.get("score")))
        else:
            scored.append((result.key, result.score))
    return scored


def print_summary(results) -> None:
    from eval.score import mean

    buckets: dict[str, list[float]] = {}
    for item in results:
        for key, score in _result_scores(item):
            if key is None or score is None:
                continue
            buckets.setdefault(key, []).append(float(score))

    labels = [
        ("retrieval_hit", "retrieval hit rate"),
        ("answer_correct", "answer correctness"),
        ("correct_refusal", "correct refusal rate"),
        ("false_refusal", "false refusal rate"),
        ("injection_held", "injection held"),
        ("latency_ms", "average latency ms"),
    ]
    print("")
    for key, label in labels:
        value = mean(buckets.get(key, []))
        if value is None:
            print(f"{label}: n/a")
        elif key == "latency_ms":
            print(f"{label}: {value:.0f}")
        else:
            print(f"{label}: {value:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocuMind in LangSmith.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Ask against documents already ingested for this chunk size.",
    )
    parser.add_argument(
        "--sync-dataset",
        action="store_true",
        help="Replace the LangSmith dataset with the current questions.jsonl.",
    )
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "documind-eval")
    if not os.environ.get("LANGSMITH_API_KEY", "").strip():
        raise SystemExit("Set LANGSMITH_API_KEY before running the eval.")

    from langsmith import Client
    from langsmith.evaluation import evaluate

    questions = load_questions()
    user_id = ensure_eval_user()
    if not args.skip_ingest:
        ingest_corpus(user_id, args.chunk_size)
    document_ids = ready_document_ids(user_id)
    if not document_ids:
        raise SystemExit("No ready documents. Run without --skip-ingest.")

    client = Client()
    sync_dataset(client, questions, args.sync_dataset)

    def target(inputs: dict) -> dict:
        return ask_question(user_id, document_ids, inputs)

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=build_evaluators(),
        experiment_prefix=f"documind-chunk-{args.chunk_size}",
        max_concurrency=1,
        metadata={"chunk_size": args.chunk_size},
        client=client,
    )
    print_summary(results)
    experiment_name = getattr(results, "experiment_name", None)
    if experiment_name:
        print(f"LangSmith experiment: {experiment_name}")


if __name__ == "__main__":
    main()

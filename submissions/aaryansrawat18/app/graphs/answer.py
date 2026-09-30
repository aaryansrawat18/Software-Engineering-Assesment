"""Answer graph: retrieve, then refuse or generate, then save the question.

The router calls `graph.invoke(...)` once after auth and the ownership check.
There is no checkpointer. Gemini is called only from the `generate` node.

    START --> retrieve --below threshold or no chunks--> refuse --> persist --> END
                      --similarity passes--> generate --> filterCitations --> persist --> END

`generate` raises AnswerFailed on a provider or JSON failure. That skips
`persist`, and the router turns it into HTTP 503.
"""

import time
import uuid
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.db import SessionLocal
from app.graphs.state import AnswerState
from app.models import Question
from app.services.generate import AnswerFailed, estimate_cost_usd, generate_answer
from app.services.retrieve import embed_question, search_ready_chunks

# Used when retrieval itself says the documents do not contain the answer.
# Gemini is not called on that path, so this sentence is fixed.
NOT_FOUND_ANSWER = "I could not find this in your documents."


def retrieve_node(state: AnswerState) -> dict:
    """Embed the question and load the closest ready passages for this user."""
    user_id = uuid.UUID(state["user_id"])
    document_ids = [uuid.UUID(document_id) for document_id in state.get("document_ids") or []]
    query_vector = embed_question(state["question"])
    with SessionLocal() as session:
        passages = search_ready_chunks(session, user_id, query_vector, document_ids)
    return {"retrieved_chunks": passages}


def route_after_retrieve(state: AnswerState) -> Literal["refuse", "generate"]:
    """Gate 1. Skip Gemini when nothing was found or the best match is too weak."""
    passages = state.get("retrieved_chunks") or []
    if not passages:
        return "refuse"
    best_similarity = max(passage["similarity"] for passage in passages)
    if best_similarity < get_settings().similarity_threshold:
        return "refuse"
    return "generate"


def refuse_node(state: AnswerState) -> dict:
    """Say the documents do not contain the answer. Do not call Gemini."""
    return {
        "answer": NOT_FOUND_ANSWER,
        "refused": True,
        "model_chunk_ids": [],
        "citations": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }


def generate_node(state: AnswerState) -> dict:
    """Ask Gemini to answer from the retrieved passages only."""
    model_answer = generate_answer(state["question"], state.get("retrieved_chunks") or [])
    answer = model_answer["answer"]
    if model_answer["refused"] and not answer:
        answer = NOT_FOUND_ANSWER
    if not answer:
        raise AnswerFailed()
    return {
        "answer": answer,
        "refused": model_answer["refused"],
        "model_chunk_ids": model_answer["chunk_ids"],
        "input_tokens": model_answer["input_tokens"],
        "output_tokens": model_answer["output_tokens"],
    }


def filter_citations_node(state: AnswerState) -> dict:
    """Keep a citation only when its chunk id was retrieved for this question.

    A refusal cites nothing. An id the model invented is dropped.
    """
    if state.get("refused"):
        return {"citations": []}

    passages_by_id = {
        passage["chunk_id"]: passage for passage in state.get("retrieved_chunks") or []
    }
    citations = []
    seen_ids: set[str] = set()
    for chunk_id in state.get("model_chunk_ids") or []:
        if chunk_id in seen_ids:
            continue
        passage = passages_by_id.get(chunk_id)
        if passage is None:
            continue
        seen_ids.add(chunk_id)
        citations.append(
            {
                "document_name": passage["document_name"],
                "passage": passage["content"],
                "page": passage["page"],
            }
        )
    return {"citations": citations}


def persist_node(state: AnswerState) -> dict:
    """Insert the questions row. This row is the record that remains after invoke."""
    input_tokens = int(state.get("input_tokens") or 0)
    output_tokens = int(state.get("output_tokens") or 0)
    tokens = input_tokens + output_tokens
    cost = estimate_cost_usd(input_tokens, output_tokens)
    latency_ms = _latency_ms(state.get("started_at"))
    citations = list(state.get("citations") or [])
    document_ids = [uuid.UUID(document_id) for document_id in state.get("document_ids") or []]

    with SessionLocal() as session:
        question_row = Question(
            user_id=uuid.UUID(state["user_id"]),
            question=state["question"],
            answer=state.get("answer") or "",
            refused=bool(state.get("refused")),
            citations=citations,
            document_ids=document_ids,
            tokens=tokens,
            latency_ms=latency_ms,
            estimated_cost=cost,
        )
        session.add(question_row)
        session.commit()
        question_id = question_row.id

    return {
        "question_id": str(question_id),
        "tokens": tokens,
        "latency_ms": latency_ms,
        "estimated_cost_usd": float(cost),
        "citations": citations,
    }


def _latency_ms(started_at: float | None) -> int:
    """Milliseconds since the router started this question. Zero if the clock is missing."""
    if started_at is None:
        return 0
    return max(0, int((time.perf_counter() - started_at) * 1000))


def build_answer_graph(
    retrieve_node_fn=None,
    generate_node_fn=None,
    refuse_node_fn=None,
    filter_citations_node_fn=None,
    persist_node_fn=None,
):
    """Compile the answer graph. Tests can pass stub nodes. No checkpointer."""
    builder = StateGraph(AnswerState)
    builder.add_node("retrieve", retrieve_node_fn or retrieve_node)
    builder.add_node("refuse", refuse_node_fn or refuse_node)
    builder.add_node("generate", generate_node_fn or generate_node)
    builder.add_node("filterCitations", filter_citations_node_fn or filter_citations_node)
    builder.add_node("persist", persist_node_fn or persist_node)

    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"refuse": "refuse", "generate": "generate"},
    )
    builder.add_edge("refuse", "persist")
    builder.add_edge("generate", "filterCitations")
    builder.add_edge("filterCitations", "persist")
    builder.add_edge("persist", END)
    return builder.compile()


# Compiled once at import. The router calls this object. No checkpointer.
graph = build_answer_graph()

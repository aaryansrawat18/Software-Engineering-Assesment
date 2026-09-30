"""The answer graph's similarity gate, using stub nodes so Gemini is never called."""

from app.graphs.answer import build_answer_graph


def _passage(similarity: float) -> dict:
    return {
        "chunk_id": "chunk-1",
        "content": "Employees receive 15 days of annual leave.",
        "document_name": "handbook.txt",
        "page": 1,
        "chunk_index": 0,
        "similarity": similarity,
    }


def _record(visited: list[str], node_name: str, update: dict):
    """A stub node that records its name and returns a fixed update."""

    def node(state):
        visited.append(node_name)
        return update

    return node


def test_low_similarity_never_reaches_generate():
    visited: list[str] = []
    graph = build_answer_graph(
        retrieve_node_fn=_record(visited, "retrieve", {"retrieved_chunks": [_passage(0.1)]}),
        generate_node_fn=_record(visited, "generate", {"answer": "should not run"}),
        refuse_node_fn=_record(visited, "refuse", {"refused": True, "answer": "not found", "citations": []}),
        filter_citations_node_fn=_record(visited, "filterCitations", {"citations": []}),
        persist_node_fn=_record(visited, "persist", {"question_id": "saved"}),
    )
    graph.invoke({"question": "What is the wifi password?", "user_id": "user-1", "document_ids": []})
    assert visited == ["retrieve", "refuse", "persist"]


def test_passing_similarity_reaches_generate_then_filters_citations():
    visited: list[str] = []
    graph = build_answer_graph(
        retrieve_node_fn=_record(visited, "retrieve", {"retrieved_chunks": [_passage(0.9)]}),
        generate_node_fn=_record(
            visited,
            "generate",
            {"answer": "15 days.", "refused": False, "model_chunk_ids": ["chunk-1"]},
        ),
        refuse_node_fn=_record(visited, "refuse", {"refused": True}),
        filter_citations_node_fn=_record(visited, "filterCitations", {"citations": []}),
        persist_node_fn=_record(visited, "persist", {"question_id": "saved"}),
    )
    graph.invoke({"question": "How many days of leave?", "user_id": "user-1", "document_ids": []})
    assert visited == ["retrieve", "generate", "filterCitations", "persist"]

# backend/app/utils.py
import json
import re
from typing import List, Dict, Any
import numpy as np

# In reality, the variables/functions in this file would be replaced by a proper RAG
# pipeline (i.e., using Langchain)
DOCUMENTS = [
    {
        "text": "The latest version of Nextflow is version 25.04.7",
        "url": "https://github.com/nextflow-io/nextflow/releases/tag/v25.04.7",
        "id": "doc1",
        "title": "Nextflow release v25.04.7",
    },
    {
        "text": "Nextflow supports a range of different executors, including the moab exector",
        "url": "https://www.nextflow.io/docs/latest/executor.html",
        "id": "doc2",
        "title": "Nextflow executors (moab)",
    },
    {
        "text": "Nextflow supports a range of different executors, including the pbs exector",
        "url": "https://www.nextflow.io/docs/latest/executor.html",
        "id": "doc3",
        "title": "Nextflow executors (pbs)",
    },
    {
        "text": "Nextflow supports a range of different executors, including the oar exector",
        "url": "https://www.nextflow.io/docs/latest/executor.html",
        "id": "doc4",
        "title": "Nextflow executors (oar)",
    },
    {
        "text": (
        "In the new DSL2 syntax, the keywords from and into are deprecated. "
        "In DSL2, processes are invoked directly within the workflow block, "
        "using function calls and channel arguments."
        ),
        "url": "https://seqera.io/blog/dsl2-is-here/",
        "id": "doc5",
        "title": "Keywords from and into",
    }
]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def top_k_by_embedding(query_vec: np.ndarray, docs: List[Dict[str, Any]], k: int = 3, threshold: float = 0.0) -> List[Dict[str, Any]]:
    """Return top-k docs by cosine similarity if embeddings exist, filtered by threshold."""
    scored = []
    for d in docs:
        emb = d.get("embedding")
        if emb is not None:
            score = cosine_similarity(query_vec, emb)
        else:
            score = 0.0
        scored.append((score, d))

    # Sort by score (highest first)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Filter by threshold and limit to k
    filtered = [(score, d) for score, d in scored if score >= threshold]

    # Add similarity scores to documents for debugging
    result_docs = []
    for score, d in filtered[:k]:
        doc_with_score = d.copy()
        doc_with_score["similarity_score"] = round(score, 3)
        result_docs.append(doc_with_score)

    return result_docs

def simple_overlap_rank(query: str, docs: List[Dict[str, Any]], k: int = 3, threshold: int = 1) -> List[Dict[str, Any]]:
    """
    Lightweight fallback ranking: count common words (case-insensitive).
    Useful when embeddings are not available.
    threshold: minimum number of overlapping words required
    """
    q_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for d in docs:
        dtokens = set(re.findall(r"\w+", d["text"].lower()))
        score = len(q_tokens & dtokens)
        scored.append((score, d))

    # Sort by score (highest first)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Filter by threshold and limit to k
    filtered = [(score, d) for score, d in scored if score >= threshold]

    # Add overlap scores to documents for debugging
    result_docs = []
    for score, d in filtered[:k]:
        doc_with_score = d.copy()
        doc_with_score["overlap_score"] = score
        result_docs.append(doc_with_score)

    return result_docs

def safe_parse_metadata_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON metadata wrapped between <|metadata|> ... <|endmetadata|>
    Returns an empty dict if not present or parse fails.
    """
    start = "<|metadata|>"
    end = "<|endmetadata|>"
    if start in text and end in text:
        try:
            raw = text.split(start, 1)[1].split(end, 1)[0].strip()
            return json.loads(raw)
        except Exception:
            return {}
    return {}

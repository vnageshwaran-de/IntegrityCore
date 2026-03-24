"""
Gold Query store utilities — find similar problem descriptions for few-shot LLM prompting.
"""
import re
from typing import List, Optional, Tuple

from integritycore.db.engine import get_db
from integritycore.db.models import GoldQuery


def _tokenize(text: str) -> set:
    """Extract lowercase word tokens (len > 2) for similarity."""
    if not text:
        return set()
    stop = {"the", "and", "from", "into", "with", "for", "that", "this", "data", "table", "pull", "copy", "load"}
    words = set(w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 2 and w.lower() not in stop)
    return words


def find_similar_gold_queries(
    problem_description: str,
    dialect: Optional[str] = None,
    top_k: int = 3,
) -> List[Tuple[str, str, str]]:
    """
    Find top-k most similar GoldQuery entries by keyword overlap on problem_description.
    Returns list of (problem_description, sql_query, dialect) tuples.
    """
    if not problem_description or not problem_description.strip():
        return []

    query_words = _tokenize(problem_description)
    if not query_words:
        return []

    with get_db() as db:
        rows = db.query(GoldQuery).all()
        if not rows:
            return []

        scored: List[Tuple[float, GoldQuery]] = []
        for gq in rows:
            if dialect and gq.dialect and gq.dialect.upper() != dialect.upper():
                continue
            desc_words = _tokenize(gq.problem_description or "")
            overlap = len(query_words & desc_words) if query_words else 0
            if overlap > 0:
                scored.append((overlap, gq))

        scored.sort(key=lambda x: (-x[0], x[1].created_at or ""))
        top = scored[:top_k]

        return [
            (gq.problem_description, gq.sql_query, gq.dialect or "")
            for _, gq in top
        ]


def add_gold_query(problem_description: str, sql_query: str, dialect: str) -> Optional[dict]:
    """Add a new gold query to the store. Returns the created record as dict."""
    if not problem_description or not sql_query or not dialect:
        return None
    with get_db() as db:
        gq = GoldQuery(
            problem_description=problem_description.strip(),
            sql_query=sql_query.strip(),
            dialect=dialect.strip().upper() or "SNOWFLAKE",
        )
        db.add(gq)
        db.flush()
        db.refresh(gq)
        return gq.to_dict()

"""
Enterprise Data Catalog: technical metadata in DuckDB, semantic cards in LanceDB.
Introspection, Semantic Profiler (LLM Metadata Cards), and async harvesting.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from integritycore.metadata.models import (
    DistinctValueProfile,
    MetadataCard,
    TableMetadata,
)
from integritycore.metadata.introspectors import BaseIntrospector, SnowflakeIntrospector

log = logging.getLogger("integritycore.metadata.manager")

# Default paths under ~/.integritycore/
_DEFAULT_BASE = Path.home() / ".integritycore"
DEFAULT_DUCKDB_PATH = _DEFAULT_BASE / "catalog.duckdb"
DEFAULT_LANCEDB_PATH = _DEFAULT_BASE / "catalog_lance"
METADATA_STALE_HOURS = 24


def _ensure_base():
    _DEFAULT_BASE.mkdir(parents=True, exist_ok=True)


# Google Gemini embedding model (via LiteLLM). Set GOOGLE_API_KEY or GEMINI_API_KEY.
GOOGLE_EMBED_MODEL = "gemini/gemini-embedding-001"
GOOGLE_EMBED_DIM = 768  # gemini-embedding-001 output dimension


def _get_embedding_function():
    """Return a function that takes a list of strings and returns list of vectors (for LanceDB). Uses Google Gemini embedding."""
    try:
        import litellm
        def embed(texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            try:
                out = litellm.embed(model=GOOGLE_EMBED_MODEL, input=texts)
                if hasattr(out, "data"):
                    return [e["embedding"] for e in out.data]
                if isinstance(out, list):
                    return [e["embedding"] for e in out]
                return []
            except Exception as e:
                log.warning("Google embedding failed, using zero vectors: %s", e)
                return [[0.0] * GOOGLE_EMBED_DIM for _ in texts]
        return embed
    except Exception:
        def embed(texts: List[str]) -> List[List[float]]:
            return [[0.0] * GOOGLE_EMBED_DIM for _ in texts]
        return embed


class MetadataManager:
    """
    Enterprise Data Catalog: DuckDB for technical metadata, LanceDB for semantic card embeddings.
    Semantic Profiler runs in background (LLM-generated Metadata Cards).
    """

    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        lancedb_path: Optional[str] = None,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        _ensure_base()
        self.duckdb_path = duckdb_path or str(DEFAULT_DUCKDB_PATH)
        self.lancedb_path = lancedb_path or str(DEFAULT_LANCEDB_PATH)
        self._embed_fn = embedding_fn or _get_embedding_function()
        self._duck = None
        self._lance_table = None
        self._lock = threading.Lock()
        self._init_storage()

    def _init_storage(self):
        import duckdb
        self._duck = duckdb.connect(self.duckdb_path)
        self._duck.execute("""
            CREATE TABLE IF NOT EXISTS table_metadata (
                conn_id VARCHAR,
                database_name VARCHAR,
                schema_name VARCHAR,
                table_name VARCHAR,
                table_comment VARCHAR,
                columns_json CLOB,
                constraints_json CLOB,
                last_crawled_at TIMESTAMP,
                PRIMARY KEY (conn_id, schema_name, table_name)
            )
        """)
        self._duck.execute("""
            CREATE TABLE IF NOT EXISTS distinct_value_profiles (
                conn_id VARCHAR,
                schema_name VARCHAR,
                table_name VARCHAR,
                column_name VARCHAR,
                sample_values_json CLOB,
                distinct_count INTEGER,
                is_categorical BOOLEAN,
                PRIMARY KEY (conn_id, schema_name, table_name, column_name)
            )
        """)
        self._duck.execute("""
            CREATE TABLE IF NOT EXISTS metadata_cards (
                conn_id VARCHAR,
                schema_name VARCHAR,
                table_name VARCHAR,
                business_domain VARCHAR,
                logical_description VARCHAR,
                column_synonyms_json CLOB,
                keywords_json CLOB,
                raw_text CLOB,
                generated_at TIMESTAMP,
                PRIMARY KEY (conn_id, schema_name, table_name)
            )
        """)
        self._duck.close()
        self._duck = duckdb.connect(self.duckdb_path)

        # LanceDB for vector search (table created on first store_metadata_card)
        try:
            import lancedb
            self._lance_db = lancedb.connect(self.lancedb_path)
            self._lance_table = (
                self._lance_db.open_table("metadata_cards")
                if "metadata_cards" in self._lance_db.table_names()
                else None
            )
        except Exception as e:
            log.warning("LanceDB init failed: %s. Vector search disabled.", e)
            self._lance_db = None
            self._lance_table = None

    def _get_conn(self):
        import duckdb
        if self._duck is None:
            self._duck = duckdb.connect(self.duckdb_path)
        return self._duck

    def store_table_metadata(self, meta: TableMetadata) -> None:
        """Persist technical metadata to DuckDB."""
        conn = self._get_conn()
        cols_json = json.dumps([c.model_dump() for c in meta.columns])
        constraints_json = json.dumps([c.model_dump() for c in meta.constraints])
        conn.execute("""
            INSERT INTO table_metadata
            (conn_id, database_name, schema_name, table_name, table_comment, columns_json, constraints_json, last_crawled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (conn_id, schema_name, table_name) DO UPDATE SET
            database_name=excluded.database_name, table_comment=excluded.table_comment,
            columns_json=excluded.columns_json, constraints_json=excluded.constraints_json,
            last_crawled_at=excluded.last_crawled_at
        """, [
            meta.conn_id,
            meta.database or "",
            meta.schema_name,
            meta.table_name,
            meta.table_comment or "",
            cols_json,
            constraints_json,
            meta.last_crawled_at or datetime.utcnow(),
        ])

    def store_metadata_card(self, card: MetadataCard) -> None:
        """Persist Metadata Card to DuckDB and optionally to LanceDB for vector search."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO metadata_cards
            (conn_id, schema_name, table_name, business_domain, logical_description, column_synonyms_json, keywords_json, raw_text, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (conn_id, schema_name, table_name) DO UPDATE SET
            business_domain=excluded.business_domain, logical_description=excluded.logical_description,
            column_synonyms_json=excluded.column_synonyms_json, keywords_json=excluded.keywords_json,
            raw_text=excluded.raw_text, generated_at=excluded.generated_at
        """, [
            card.conn_id,
            card.schema_name,
            card.table_name,
            card.business_domain,
            card.logical_description,
            json.dumps(card.column_synonyms),
            json.dumps(card.keywords),
            card.raw_text or card.to_embedding_text(),
            card.generated_at or datetime.utcnow(),
        ])
        if self._lance_db is not None:
            text = card.raw_text or card.to_embedding_text()
            vec = self._embed_fn([text])
            if vec:
                import pyarrow as pa
                data = pa.table({
                    "conn_id": [card.conn_id],
                    "schema_name": [card.schema_name],
                    "table_name": [card.table_name],
                    "text": [text],
                    "vector": vec,
                })
                if self._lance_table is None:
                    self._lance_table = self._lance_db.create_table("metadata_cards", data)
                else:
                    self._lance_table.add(data)

    def get_table_metadata(
        self,
        conn_id: str,
        schema_name: str,
        table_name: str,
    ) -> Optional[TableMetadata]:
        """Load technical metadata from DuckDB."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT conn_id, database_name, schema_name, table_name, table_comment, columns_json, constraints_json, last_crawled_at
            FROM table_metadata WHERE conn_id = ? AND schema_name = ? AND table_name = ?
        """, (conn_id, schema_name, table_name)).fetchone()
        if not row:
            return None
        from integritycore.metadata.models import ColumnMetadata, ConstraintMetadata
        cols = [ColumnMetadata(**c) for c in json.loads(row[5] or "[]")]
        constraints = [ConstraintMetadata(**c) for c in json.loads(row[6] or "[]")]
        return TableMetadata(
            conn_id=row[0],
            database=row[1] or None,
            schema_name=row[2],
            table_name=row[3],
            table_comment=row[4] or None,
            columns=cols,
            constraints=constraints,
            last_crawled_at=row[7],
        )

    def get_last_crawled_at(self, conn_id: str, schema_name: str, table_name: str) -> Optional[datetime]:
        """Return last_crawled_at for schema versioning / stale check."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT last_crawled_at FROM table_metadata WHERE conn_id = ? AND schema_name = ? AND table_name = ?
        """, (conn_id, schema_name, table_name)).fetchone()
        return row[0] if row else None

    def is_metadata_stale(self, conn_id: str, schema_name: str, table_name: str, max_age_hours: int = METADATA_STALE_HOURS) -> bool:
        """True if metadata is older than max_age_hours (e.g. 24)."""
        t = self.get_last_crawled_at(conn_id, schema_name, table_name)
        if not t:
            return True
        return (datetime.utcnow() - t.replace(tzinfo=None) if t.tzinfo else t) > timedelta(hours=max_age_hours)

    def list_tables(self, conn_id: str, schema_name: Optional[str] = None) -> List[Tuple[str, str]]:
        """List (schema_name, table_name) for conn_id."""
        conn = self._get_conn()
        if schema_name:
            rows = conn.execute(
                "SELECT schema_name, table_name FROM table_metadata WHERE conn_id = ? AND schema_name = ?",
                (conn_id, schema_name),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT schema_name, table_name FROM table_metadata WHERE conn_id = ?",
                (conn_id,),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def search_by_semantics(self, query: str, conn_id: Optional[str] = None, top_k: int = 10) -> List[Tuple[str, str, str, float]]:
        """Vector search: return (conn_id, schema_name, table_name, score) for tables matching business terms."""
        if self._lance_table is None:
            return []
        try:
            qvec = self._embed_fn([query])
            if not qvec:
                return []
            results = self._lance_table.search(qvec[0]).limit(top_k).to_list()
            out = []
            for r in results:
                if conn_id and r.get("conn_id") != conn_id:
                    continue
                out.append((
                    r.get("conn_id", ""),
                    r.get("schema_name", ""),
                    r.get("table_name", ""),
                    float(r.get("_distance", 0)),
                ))
            return out
        except Exception as e:
            log.warning("Vector search failed: %s", e)
            return []

    # ---------- Semantic Profiler (background) ----------

    def generate_metadata_card(
        self,
        table_meta: TableMetadata,
        distinct_profiles: List[DistinctValueProfile],
        model_name: str = "gemini/gemini-2.5-flash",
    ) -> MetadataCard:
        """Use LLM to generate a Metadata Card (business_domain, logical_description, synonyms)."""
        import litellm
        columns_desc = "\n".join(
            f"- {c.name}: {c.data_type}, nullable={c.is_nullable}, comment={c.comment or 'N/A'}"
            for c in table_meta.columns
        )
        profiles_desc = ""
        for p in distinct_profiles:
            profiles_desc += f"\n- {p.column_name}: sample values {p.sample_values[:10]!r}"
        prompt = f"""You are a data steward. For this table, produce a JSON object with:
- business_domain: one short phrase (e.g. "Sales", "Finance")
- logical_description: 1-2 sentence description of what the table represents
- column_synonyms: object mapping column name to list of business-friendly synonyms (e.g. "rev" -> ["revenue", "sales amount"])
- keywords: list of 5-10 searchable terms (business concepts) for this table

Table: {table_meta.schema_name}.{table_meta.table_name}
Comment: {table_meta.table_comment or 'N/A'}

Columns:
{columns_desc}

Distinct value samples (categorical): {profiles_desc or 'None'}

Return ONLY valid JSON: {{ "business_domain": "...", "logical_description": "...", "column_synonyms": {{}}, "keywords": [] }}"""

        try:
            resp = litellm.completion(model=model_name, messages=[{"role": "user", "content": prompt}])
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(content)
        except Exception as e:
            log.warning("LLM Metadata Card failed: %s", e)
            data = {
                "business_domain": "Unknown",
                "logical_description": table_meta.table_comment or f"Table {table_meta.table_name}",
                "column_synonyms": {},
                "keywords": [table_meta.table_name],
            }
        card = MetadataCard(
            conn_id=table_meta.conn_id,
            schema_name=table_meta.schema_name,
            table_name=table_meta.table_name,
            business_domain=data.get("business_domain", ""),
            logical_description=data.get("logical_description", ""),
            column_synonyms=data.get("column_synonyms", {}),
            keywords=data.get("keywords", []),
            raw_text="",
            generated_at=datetime.utcnow(),
        )
        card.raw_text = card.to_embedding_text()
        return card

    def harvest_connection(
        self,
        conn: Any,
        conn_id: str,
        schema: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        run_profiler: bool = True,
        model_name: str = "gemini/gemini-2.5-flash",
    ) -> int:
        """
        Introspect connection, store technical metadata, optionally run Semantic Profiler (LLM cards).
        Returns number of tables processed. Run in background thread to avoid blocking UI.
        """
        dialect = (getattr(conn, "dialect", "") or "").upper()
        if dialect == "SNOWFLAKE":
            introspector = SnowflakeIntrospector(conn, conn_id)
        else:
            if progress_cb:
                progress_cb(f"Unsupported dialect for introspection: {dialect}")
            return 0
        tables = introspector.get_tables(None, schema)
        if not tables:
            if progress_cb:
                progress_cb("No tables found")
            return 0
        count = 0
        for t in tables:
            if len(t) == 3:
                db, sch, tbl = t[0], t[1], t[2]
            else:
                sch, tbl = t[0], t[1]
                db = None
            try:
                meta = introspector.introspect_table(db, sch, tbl)
                if meta:
                    meta.last_crawled_at = datetime.utcnow()
                    self.store_table_metadata(meta)
                    count += 1
                    if progress_cb:
                        progress_cb(f"Crawled {sch}.{tbl}")
                if run_profiler and meta:
                    profiles = self._sample_distinct_values(introspector, meta)
                    card = self.generate_metadata_card(meta, profiles, model_name)
                    self.store_metadata_card(card)
            except Exception as e:
                log.warning("Harvest error for %s.%s: %s", sch, tbl, e)
                if progress_cb:
                    progress_cb(f"Error {sch}.{tbl}: {e}")
        return count

    def _sample_distinct_values(self, introspector: BaseIntrospector, meta: TableMetadata) -> List[DistinctValueProfile]:
        """Extract distinct value profile for categorical-looking columns (e.g. VARCHAR with low cardinality)."""
        profiles = []
        # Simple heuristic: sample first few string columns
        for col in meta.columns:
            if col.data_type and "CHAR" in col.data_type.upper():
                profiles.append(
                    DistinctValueProfile(
                        column_name=col.name,
                        table_schema=meta.schema_name,
                        table_name=meta.table_name,
                        sample_values=[],
                        is_categorical=True,
                    )
                )
        return profiles

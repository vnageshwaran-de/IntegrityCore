import json
import os
import re
from typing import List, Optional, Tuple
from pydantic import BaseModel
import uuid


def get_relevant_schema(user_request: str, connection_id: str, metadata_manager=None, top_k: int = 10, parsed_table: Optional[str] = None) -> str:
    """
    Dynamic context fetcher: returns DDL for tables.
    When parsed_table is provided (e.g. SCHEMA.TABLE from first LLM call), fetch ONLY that table.
    Otherwise uses keyword-matching over table names, comments, and metadata cards.
    """
    if not connection_id:
        return ""

    try:
        from integritycore.metadata.manager import MetadataManager
        mgr = metadata_manager or MetadataManager()
    except Exception:
        return ""

    # When parsed_table is provided (from first LLM call), fetch ONLY that table's DDL
    if parsed_table and parsed_table.strip():
        parts = parsed_table.strip().split(".")
        if len(parts) == 2:
            schema_name, table_name = parts[0], parts[1]
        else:
            schema_name, table_name = "PUBLIC", parts[-1] if parts else ""
        if table_name:
            meta = mgr.get_table_metadata(connection_id, schema_name, table_name)
            if meta:
                from integritycore.core.grounding import GroundingEngine
                engine = GroundingEngine(mgr)
                return engine._table_to_clean_ddl(meta)
        return ""

    if not user_request or not user_request.strip():
        return ""

    # Extract searchable keywords from user request (words > 2 chars, lowercased)
    stop = {"the", "and", "from", "into", "with", "for", "that", "this", "data", "table", "pull", "copy", "load"}
    words = set(
        w.lower().strip(".,;:!?")
        for w in re.findall(r"\b\w+\b", user_request)
        if len(w) > 2 and w.lower() not in stop
    )
    if not words:
        return ""

    tables = mgr.list_tables(connection_id)
    if not tables:
        return ""

    # Build searchable text and score for each table
    scored: List[Tuple[float, str, str]] = []  # (score, schema, table)
    conn = mgr._get_conn()

    for schema_name, table_name in tables:
        searchable = [schema_name.lower(), table_name.lower()]
        # Table metadata: comment
        row_tm = conn.execute("""
            SELECT table_comment, columns_json FROM table_metadata
            WHERE conn_id = ? AND schema_name = ? AND table_name = ?
        """, (connection_id, schema_name, table_name)).fetchone()
        if row_tm:
            if row_tm[0]:
                searchable.append(row_tm[0].lower())
            try:
                cols = json.loads(row_tm[1] or "[]")
                for c in cols:
                    searchable.append(c.get("name", "").lower())
                    if c.get("comment"):
                        searchable.append(c.get("comment", "").lower())
            except Exception:
                pass

        # Metadata card: business_domain, logical_description, keywords
        row_card = conn.execute("""
            SELECT business_domain, logical_description, keywords_json FROM metadata_cards
            WHERE conn_id = ? AND schema_name = ? AND table_name = ?
        """, (connection_id, schema_name, table_name)).fetchone()
        if row_card:
            if row_card[0]:
                searchable.append(row_card[0].lower())
            if row_card[1]:
                searchable.append(row_card[1].lower())
            try:
                kws = json.loads(row_card[2] or "[]")
                searchable.extend(k.lower() for k in kws)
            except Exception:
                pass

        text = " ".join(s for s in searchable if s)
        score = sum(1 for w in words if w in text)
        if score > 0:
            scored.append((score, schema_name, table_name))

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    top = scored[:top_k]

    if not top:
        return ""

    # Build DDL for top tables
    from integritycore.core.grounding import GroundingEngine
    engine = GroundingEngine(mgr)
    parts = []
    for _, sch, tbl in top:
        meta = mgr.get_table_metadata(connection_id, sch, tbl)
        if meta:
            parts.append(engine._table_to_clean_ddl(meta))

    return "\n\n".join(parts) if parts else ""

class DBConnection(BaseModel):
    id: str
    name: str
    dialect: str
    host: Optional[str] = None
    port: Optional[str] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    account: Optional[str] = None
    warehouse: Optional[str] = None
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    service_account_json: Optional[str] = None

class ConnectionManager:
    """Manages local storage of database connections."""
    def __init__(self, filepath: str = os.path.expanduser("~/.integritycore/connections.json")):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump([], f)

    def load_connections(self) -> List[DBConnection]:
        with open(self.filepath, 'r') as f:
            data = json.load(f)
            return [DBConnection(**conn) for conn in data]

    def add_connection(self, name: str, dialect: str, **kwargs) -> DBConnection:
        conns = [c.model_dump() for c in self.load_connections()]
        new_conn = {
            "id": str(uuid.uuid4()),
            "name": name,
            "dialect": dialect,
            **kwargs
        }
        conns.append(new_conn)
        
        with open(self.filepath, 'w') as f:
            json.dump(conns, f, indent=4)
            
        return DBConnection(**new_conn)

    def delete_connection(self, conn_id: str) -> bool:
        conns = self.load_connections()
        initial_count = len(conns)
        conns = [c for c in conns if c.id != conn_id]

        if len(conns) == initial_count:
            return False

        with open(self.filepath, 'w') as f:
            json.dump([c.model_dump() for c in conns], f, indent=4)

        return True

    def update_connection(self, conn_id: str, **fields) -> Optional[DBConnection]:
        conns = self.load_connections()
        updated = None
        for c in conns:
            if c.id == conn_id:
                data = c.model_dump()
                for k, v in fields.items():
                    if k in data and v is not None:
                        data[k] = v
                updated = DBConnection(**data)
                break

        if updated is None:
            return None

        with open(self.filepath, 'w') as f:
            json.dump([c.model_dump() if c.id != conn_id else updated.model_dump()
                       for c in conns], f, indent=4)

        return updated

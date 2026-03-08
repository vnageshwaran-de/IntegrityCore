"""
Grounding Engine: vector search + graph expansion + Clean DDL for metadata-grounded SQL generation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from integritycore.metadata.models import (
    ConstraintType,
    GroundingResult,
    MetadataCard,
    TableMetadata,
)

log = logging.getLogger("integritycore.core.grounding")

CONFIDENCE_THRESHOLD_LOW = 0.6  # Below this: return closeness_matches for user disambiguation


class GroundingEngine:
    """
    Two-stage retrieval: (1) Vector search for tables matching business terms,
    (2) Graph expansion via FK to pull related table DDL. Produces Verified Schema Fragment + Clean DDL.
    """

    def __init__(self, metadata_manager: Any):
        self.manager = metadata_manager

    def retrieve(
        self,
        user_prompt: str,
        conn_id: Optional[str] = None,
        source_schema: Optional[str] = None,
        expand_fk: bool = True,
        top_k: int = 5,
        confidence_threshold: float = CONFIDENCE_THRESHOLD_LOW,
    ) -> GroundingResult:
        """
        Vector search for tables matching prompt terms; optionally expand to FK-related tables;
        build Clean DDL and semantic mappings. If confidence low, fill closeness_matches.
        """
        # 1) Vector search
        hits = self.manager.search_by_semantics(user_prompt, conn_id=conn_id, top_k=top_k)
        if not hits:
            return GroundingResult(
                verified_schema_fragment="",
                semantic_mappings={},
                related_tables=[],
                confidence=0.0,
                closeness_matches=[],
            )
        # Score: lower distance = better; normalize to 0-1 confidence
        best_distance = hits[0][3] if hits else 1.0
        confidence = max(0, 1.0 - best_distance) if isinstance(best_distance, (int, float)) else 0.8
        tables_to_include: List[Tuple[str, str, str]] = [(h[0], h[1], h[2]) for h in hits]

        # 2) Graph expansion: add FK-related tables
        if expand_fk and conn_id:
            seen = {(t[1], t[2]) for t in tables_to_include}
            for cid, sch, tbl in list(tables_to_include):
                meta = self.manager.get_table_metadata(cid, sch, tbl)
                if not meta:
                    continue
                for c in meta.constraints:
                    if c.constraint_type == ConstraintType.FOREIGN_KEY and c.ref_table_schema and c.ref_table_name:
                        key = (c.ref_table_schema, c.ref_table_name)
                        if key not in seen:
                            seen.add(key)
                            tables_to_include.append((cid, c.ref_table_schema, c.ref_table_name))

        # 3) Load full metadata and build DDL + semantic mappings
        cards: List[MetadataCard] = []
        fragment_parts: List[str] = []
        semantic_mappings: Dict[str, str] = {}
        for cid, sch, tbl in tables_to_include:
            meta = self.manager.get_table_metadata(cid, sch, tbl)
            if meta:
                fragment_parts.append(self._table_to_clean_ddl(meta))
            # Card for synonyms mapping
            card = self._get_card(cid, sch, tbl)
            if card:
                cards.append(card)
                for col, syns in card.column_synonyms.items():
                    for s in syns:
                        semantic_mappings[s.lower()] = f"{sch}.{tbl}.{col}"
                semantic_mappings[card.business_domain.lower()] = f"{sch}.{tbl}"
                for kw in card.keywords:
                    semantic_mappings[kw.lower()] = f"{sch}.{tbl}"

        verified_schema_fragment = "\n\n".join(fragment_parts)
        related_tables = [f"{t[1]}.{t[2]}" for t in tables_to_include]

        # 4) If confidence low, get alternative concepts for "Which did you mean?"
        closeness_matches: List[str] = []
        if confidence < confidence_threshold and cards:
            closeness_matches = list({c.business_domain for c in cards if c.business_domain})[:5]
            if not closeness_matches and cards:
                closeness_matches = [c.table_name for c in cards[:5]]

        return GroundingResult(
            verified_schema_fragment=verified_schema_fragment,
            semantic_mappings=semantic_mappings,
            related_tables=related_tables,
            confidence=confidence,
            closeness_matches=closeness_matches,
            metadata_cards=cards,
        )

    def _get_card(self, conn_id: str, schema_name: str, table_name: str) -> Optional[MetadataCard]:
        """Load MetadataCard from DuckDB (manager has no get_metadata_card; we need to add or read from cards in retrieve)."""
        conn = self.manager._get_conn()
        row = conn.execute("""
            SELECT conn_id, schema_name, table_name, business_domain, logical_description,
                   column_synonyms_json, keywords_json, raw_text, generated_at
            FROM metadata_cards WHERE conn_id = ? AND schema_name = ? AND table_name = ?
        """, (conn_id, schema_name, table_name)).fetchone()
        if not row:
            return None
        import json
        return MetadataCard(
            conn_id=row[0],
            schema_name=row[1],
            table_name=row[2],
            business_domain=row[3] or "",
            logical_description=row[4] or "",
            column_synonyms=json.loads(row[5] or "{}"),
            keywords=json.loads(row[6] or "[]"),
            raw_text=row[7] or "",
            generated_at=row[8],
        )

    def _table_to_clean_ddl(self, meta: TableMetadata, include_all_columns: bool = True) -> str:
        """Convert TableMetadata to a Clean DDL string for the LLM (exclude irrelevant columns if needed)."""
        lines = [f"-- Table: {meta.schema_name}.{meta.table_name}"]
        if meta.table_comment:
            lines.append(f"-- Description: {meta.table_comment}")
        cols = meta.columns
        if not include_all_columns:
            cols = [c for c in cols if c.comment or c.name in ("id", "created_at", "updated_at")][:20]
        for c in cols:
            nullable = "NULL" if c.is_nullable else "NOT NULL"
            comment = f"  -- {c.comment}" if c.comment else ""
            lines.append(f"  {c.name} {c.data_type} {nullable},{comment}")
        if lines[-1].endswith(","):
            lines[-1] = lines[-1].rstrip(",")
        return "\n".join(lines)

    def build_grounded_prompt(
        self,
        user_objective: str,
        grounding_result: GroundingResult,
        source_dialect: str,
        target_dialect: str,
    ) -> str:
        """Strict Grounding Template for generate_sql_node."""
        mapping_rules = "\n".join(f"  - {k} -> {v}" for k, v in grounding_result.semantic_mappings.items())
        return (
            "You are a Senior Data Engineer. You MUST ONLY use the tables and columns defined in the <VERIFIED_SCHEMA> block below.\n"
            f"Mapping Rules (business term -> table.column):\n{mapping_rules or '  (none)'}\n\n"
            f"<VERIFIED_SCHEMA>\n{grounding_result.verified_schema_fragment}\n</VERIFIED_SCHEMA>\n\n"
            f"Source dialect: {source_dialect}. Target dialect: {target_dialect}.\n\n"
            f"User objective: {user_objective}\n\n"
            "Return ONLY valid SQL wrapped in a ```sql code block. No explanation."
        )

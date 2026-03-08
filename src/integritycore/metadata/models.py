"""Pydantic models for the semantic metadata catalog."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConstraintType(str, Enum):
    PRIMARY_KEY = "PRIMARY KEY"
    FOREIGN_KEY = "FOREIGN KEY"
    UNIQUE = "UNIQUE"


class ColumnMetadata(BaseModel):
    """Column-level technical metadata from DB introspection."""
    name: str
    ordinal_position: int
    data_type: str
    is_nullable: bool = True
    comment: Optional[str] = None
    table_schema: str = ""
    table_name: str = ""

    def full_name(self) -> str:
        if self.table_schema:
            return f"{self.table_schema}.{self.table_name}.{self.name}"
        return f"{self.table_name}.{self.name}"


class ConstraintMetadata(BaseModel):
    """Primary/Foreign key constraint for relational graph."""
    constraint_name: str
    constraint_type: ConstraintType
    table_schema: str
    table_name: str
    column_names: List[str]
    # For FK: referenced table/columns
    ref_table_schema: Optional[str] = None
    ref_table_name: Optional[str] = None
    ref_column_names: Optional[List[str]] = None


class TableMetadata(BaseModel):
    """Table-level technical metadata."""
    conn_id: str
    database: Optional[str] = None
    schema_name: str
    table_name: str
    columns: List[ColumnMetadata] = Field(default_factory=list)
    constraints: List[ConstraintMetadata] = Field(default_factory=list)
    table_comment: Optional[str] = None
    last_crawled_at: Optional[datetime] = None

    def full_table_name(self) -> str:
        parts = [self.schema_name, self.table_name]
        if self.database:
            parts.insert(0, self.database)
        return ".".join(parts)


class DistinctValueProfile(BaseModel):
    """Distinct value profile for a categorical column (semantic profiler)."""
    column_name: str
    table_schema: str
    table_name: str
    sample_values: List[str] = Field(default_factory=list)
    distinct_count: Optional[int] = None
    is_categorical: bool = True


class MetadataCard(BaseModel):
    """LLM-generated semantic card for a table (business context)."""
    conn_id: str
    schema_name: str
    table_name: str
    business_domain: str = ""
    logical_description: str = ""
    column_synonyms: Dict[str, List[str]] = Field(default_factory=dict)  # column_name -> [synonyms]
    keywords: List[str] = Field(default_factory=list)  # for vector search
    raw_text: str = ""  # concatenated for embedding
    generated_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Text representation for vector embedding."""
        parts = [
            self.business_domain,
            self.logical_description,
            self.table_name,
            " ".join(self.keywords),
        ]
        for col, syns in self.column_synonyms.items():
            parts.append(f"{col}: {', '.join(syns)}")
        return " ".join(p for p in parts if p).strip()


class GroundingResult(BaseModel):
    """Result of grounding engine retrieval."""
    verified_schema_fragment: str  # Clean DDL string
    semantic_mappings: Dict[str, str] = Field(default_factory=dict)  # business term -> table.column
    related_tables: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    closeness_matches: List[str] = Field(default_factory=list)  # when confidence low: "Sales", "Returns", etc.
    metadata_cards: List[MetadataCard] = Field(default_factory=list)

"""Enterprise Semantic Metadata Layer for IntegrityCore."""
from integritycore.metadata.models import (
    ColumnMetadata,
    TableMetadata,
    ConstraintMetadata,
    MetadataCard,
    DistinctValueProfile,
    GroundingResult,
)

try:
    from integritycore.metadata.manager import MetadataManager
except Exception:
    MetadataManager = None  # type: ignore

__all__ = [
    "ColumnMetadata",
    "TableMetadata",
    "ConstraintMetadata",
    "MetadataCard",
    "DistinctValueProfile",
    "GroundingResult",
    "MetadataManager",
]

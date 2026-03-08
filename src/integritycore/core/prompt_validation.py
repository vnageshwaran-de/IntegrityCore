"""
Structured, interactive prompt-to-SQL validation for IntegrityCore.

Validates ETL prompts against completeness, unambiguity, dialect consistency,
and technical feasibility. Returns machine-readable results so the UI can
show specific blockers, warnings, and clarification questions.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import litellm

log = logging.getLogger("integritycore.prompt_validation")


class ValidationSeverity(str, Enum):
    BLOCKER = "blocker"   # Must fix before proceeding
    WARNING = "warning"   # Should fix or acknowledge
    INFO = "info"         # Suggestion only


@dataclass
class ValidationItem:
    """A single validation finding (blocker, warning, or info)."""
    severity: ValidationSeverity
    code: str
    message: str
    suggestion_question: Optional[str] = None  # Interactive: question to ask the user
    suggested_value: Optional[str] = None       # Optional pre-fill or example

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "suggestion_question": self.suggestion_question,
            "suggested_value": self.suggested_value,
        }


@dataclass
class ExtractedHints:
    """Optional hints the validator inferred from the prompt (for UI pre-fill or confirm)."""
    source_schema: Optional[str] = None
    source_table: Optional[str] = None
    target_schema: Optional[str] = None
    target_table: Optional[str] = None
    incremental_column: Optional[str] = None
    filter_description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "source_schema": self.source_schema,
            "source_table": self.source_table,
            "target_schema": self.target_schema,
            "target_table": self.target_table,
            "incremental_column": self.incremental_column,
            "filter_description": self.filter_description,
        }.items() if v is not None}


@dataclass
class PromptValidationResult:
    """Full result of prompt validation: blockers, warnings, suggestions, and hints."""
    valid: bool
    blockers: List[ValidationItem] = field(default_factory=list)
    warnings: List[ValidationItem] = field(default_factory=list)
    suggestions: List[ValidationItem] = field(default_factory=list)
    extracted_hints: Optional[ExtractedHints] = None
    raw_llm_response: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.valid and len(self.blockers) == 0

    def summary_message(self) -> str:
        if self.is_valid and not self.warnings:
            return "Validation passed."
        if self.blockers:
            return "; ".join(b.message for b in self.blockers)
        if self.warnings:
            return "; ".join(w.message for w in self.warnings)
        return "Validation passed with suggestions."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.is_valid,
            "summary_message": self.summary_message(),
            "blockers": [b.to_dict() for b in self.blockers],
            "warnings": [w.to_dict() for w in self.warnings],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "extracted_hints": self.extracted_hints.to_dict() if self.extracted_hints else None,
        }


# ─── Validation criteria (for LLM prompt and for rule-based fallback) ─────────

VALIDATION_SYSTEM_PROMPT = """You are a strict data engineering ETL requirements validator. Your job is to validate whether a user's natural-language ETL instruction has enough context to generate correct SQL.

**Context**
- Source dialect: {source_dialect}
- Target dialect: {target_dialect}

**Validation criteria — check ALL of these:**

1. **Completeness**
   - Is the SOURCE clearly identified? (database/schema and table or dataset.table)
   - Is the TARGET clearly identified? (schema and table where data should land)
   - Is it clear WHAT to move? (full table, specific columns, or a filtered subset)
   - If incremental/CDC is implied (e.g. "only new rows", "since last run", "updated_at"), is the incremental column or watermark logic clear?

2. **Unambiguity**
   - No vague references: avoid "the table", "that report", "the india table" without saying which schema/database.
   - No ambiguous table or column names that could refer to multiple objects.
   - Timezone or environment (dev/staging/prod) specified if it could matter.

3. **Consistency**
   - Source and target dialect compatibility (e.g. data types, syntax) is assumed; flag only if the user mixes incompatible concepts.
   - If the user says "incremental" or "delta", ensure a timestamp/watermark column or condition is mentioned or can be inferred.

4. **Technical feasibility**
   - Table/object names should be valid identifiers (no spaces, no reserved keywords as bare names unless quoted).
   - If the user mentions columns, they should be plausible (e.g. "updated_at" for incremental).

5. **Security / sensitivity (warnings only)**
   - If PII or sensitive data is mentioned (e.g. "customer emails", "SSN"), add a WARNING suggesting confirmation of access and masking.

**Output format**
You MUST respond with a single JSON object (no markdown, no code fence), with this exact structure:

{{
  "valid": true or false,
  "blockers": [
    {{ "code": "MISSING_SOURCE_TABLE", "message": "Clear one-line description.", "suggestion_question": "Which schema and table should we read from?", "suggested_value": null or "e.g. PUBLIC.orders" }}
  ],
  "warnings": [
    {{ "code": "AMBIGUOUS_TARGET", "message": "Description.", "suggestion_question": "Optional question for user.", "suggested_value": null }}
  ],
  "suggestions": [
    {{ "code": "SPECIFY_INCREMENTAL_COLUMN", "message": "Description.", "suggestion_question": null, "suggested_value": "updated_at" }}
  ],
  "extracted_hints": {{
    "source_schema": null or "e.g. PUBLIC",
    "source_table": null or "e.g. orders",
    "target_schema": null or "e.g. ANALYTICS",
    "target_table": null or "e.g. orders_snapshot",
    "incremental_column": null or "e.g. updated_at",
    "filter_description": null or short description of filters
  }}
}}

Rules:
- "valid" is false if and only if there is at least one BLOCKER. Warnings and suggestions do not make valid=false.
- blockers: issues that MUST be fixed before SQL can be generated (e.g. missing source table, missing target table, completely vague prompt).
- warnings: issues that should be fixed or acknowledged (e.g. ambiguous name, possible PII).
- suggestions: nice-to-have improvements (e.g. "Consider specifying timezone for timestamps").
- For each item use code: a short UPPER_SNAKE_CASE identifier (e.g. MISSING_SOURCE_TABLE, MISSING_TARGET_TABLE, AMBIGUOUS_TARGET, PII_WARNING).
- suggestion_question: if the UI can ask the user a clarifying question, put it here; otherwise null.
- suggested_value: example or default the UI could pre-fill; otherwise null.
- extracted_hints: infer from the prompt whatever you can (schema.table, incremental column); use null for unknown.
- Return ONLY the JSON object, no other text before or after.

**Critical for "pull data from X" prompts:** When the user says things like "pull data from zillow.city.india" or "get data from schema.table" without specifying WHERE to load the data:
- Add exactly one blocker with code MISSING_TARGET_TABLE, message "The target schema and table where the data should be loaded are not specified."
- ALWAYS set suggestion_question to: "Where should the data be loaded? Reply with target schema and table (e.g. STAGING.my_table or ANALYTICS.my_table)."
- ALWAYS set suggested_value to a plausible default derived from the source: e.g. if source is zillow.city.india use "STAGING.india" or "STAGING.zillow_city_india"; if source is PUBLIC.orders use "STAGING.orders". So the user can reply with that exact value or edit it. Use the rightmost table-like part of the source as the table name and a generic schema like STAGING or ANALYTICS."""


def _parse_validation_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, with optional markdown code block stripping."""
    raw = text.strip()
    # Strip ```json ... ``` if present
    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Failed to parse validation JSON: %s", raw[:200])
        return None


def _parse_severity(s: Optional[str]) -> ValidationSeverity:
    if s == "blocker":
        return ValidationSeverity.BLOCKER
    if s == "warning":
        return ValidationSeverity.WARNING
    return ValidationSeverity.INFO


def _item_from_dict(d: Dict[str, Any], default_severity: ValidationSeverity) -> ValidationItem:
    return ValidationItem(
        severity=_parse_severity(d.get("severity")) or default_severity,
        code=str(d.get("code", "UNKNOWN")),
        message=str(d.get("message", "")).strip() or "No description",
        suggestion_question=d.get("suggestion_question"),
        suggested_value=d.get("suggested_value"),
    )


def _derive_target_table_from_prompt(prompt: str) -> Optional[str]:
    """Derive a plausible target table name from phrases like 'from zillow.city.india' or 'from schema.table'."""
    # Match "from X" or "pull data from X" where X is dotted identifiers
    m = re.search(r"\b(?:from|pull\s+data\s+from|get\s+data\s+from)\s+([A-Za-z0-9_.]+)", prompt, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).strip()
    parts = [p for p in re.split(r"[.\s]+", raw) if p]
    if not parts:
        return None
    # Use last part as table (e.g. india), or join with underscore (zillow_city_india)
    if len(parts) == 1:
        return parts[0]
    return parts[-1]  # e.g. "india" from "zillow.city.india"


def _ensure_target_suggestion(
    blockers: List[ValidationItem],
    extracted_hints: Optional[ExtractedHints],
    prompt: str,
) -> None:
    """Ensure any missing-target blocker has suggestion_question and suggested_value so user can reply."""
    target_question = "Where should the data be loaded? Reply with target schema and table (e.g. STAGING.my_table or ANALYTICS.my_table)."
    for b in blockers:
        is_target_blocker = (
            b.code == "MISSING_TARGET_TABLE"
            or "target" in b.message.lower() and "not specified" in b.message.lower()
        )
        if not is_target_blocker:
            continue
        if not b.suggestion_question:
            b.suggestion_question = target_question
        if not b.suggested_value:
            table_name = None
            if extracted_hints and extracted_hints.source_table:
                table_name = extracted_hints.source_table
            if not table_name:
                table_name = _derive_target_table_from_prompt(prompt)
            b.suggested_value = f"STAGING.{table_name}" if table_name else "STAGING.target_table"
    return None


def _hints_from_dict(d: Optional[Dict[str, Any]]) -> Optional[ExtractedHints]:
    if not d or not isinstance(d, dict):
        return None
    return ExtractedHints(
        source_schema=d.get("source_schema"),
        source_table=d.get("source_table"),
        target_schema=d.get("target_schema"),
        target_table=d.get("target_table"),
        incremental_column=d.get("incremental_column"),
        filter_description=d.get("filter_description"),
    )


def validate_etl_prompt(
    prompt: str,
    source_dialect: str,
    target_dialect: str,
    model_name: str = "gemini/gemini-2.5-flash",
) -> PromptValidationResult:
    """
    Run full interactive validation on an ETL prompt.
    Returns structured blockers, warnings, suggestions, and extracted hints.
    """
    system_content = VALIDATION_SYSTEM_PROMPT.format(
        source_dialect=source_dialect,
        target_dialect=target_dialect,
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Validate this ETL instruction:\n\n{prompt}"},
    ]
    try:
        response = litellm.completion(model=model_name, messages=messages)
        content = (response.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("LLM validation request failed: %s", e)
        return PromptValidationResult(
            valid=True,
            raw_llm_response=None,
            warnings=[ValidationItem(
                severity=ValidationSeverity.WARNING,
                code="VALIDATOR_ERROR",
                message=f"Validation check could not be completed: {e}. Proceeding with caution.",
                suggestion_question=None,
                suggested_value=None,
            )],
        )

    data = _parse_validation_response(content)
    if not data:
        return PromptValidationResult(
            valid=True,
            raw_llm_response=content,
            warnings=[ValidationItem(
                severity=ValidationSeverity.WARNING,
                code="PARSE_ERROR",
                message="Could not parse validation result. Proceeding with caution.",
                suggestion_question=None,
                suggested_value=None,
            )],
        )

    blockers = [_item_from_dict(b, ValidationSeverity.BLOCKER) for b in (data.get("blockers") or []) if isinstance(b, dict)]
    warnings = [_item_from_dict(w, ValidationSeverity.WARNING) for w in (data.get("warnings") or []) if isinstance(w, dict)]
    suggestions = [_item_from_dict(s, ValidationSeverity.INFO) for s in (data.get("suggestions") or []) if isinstance(s, dict)]

    extracted_hints = _hints_from_dict(data.get("extracted_hints"))
    # Ensure MISSING_TARGET_TABLE blocker has a concrete suggested_value so user can reply easily
    _ensure_target_suggestion(blockers, extracted_hints, prompt)

    valid = bool(data.get("valid", True)) and len(blockers) == 0

    return PromptValidationResult(
        valid=valid,
        blockers=blockers,
        warnings=warnings,
        suggestions=suggestions,
        extracted_hints=extracted_hints,
        raw_llm_response=content,
    )

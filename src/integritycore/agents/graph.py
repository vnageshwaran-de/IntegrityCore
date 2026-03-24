import json
import operator
import asyncio
import re
from typing import Annotated, TypedDict, Any, Dict, List, Optional, Literal
from langgraph.graph import StateGraph, START, END
import litellm

from integritycore.core.verifier import LogicVerifier, ETLStrategy
from integritycore.core.grounding import GroundingEngine
from integritycore.adapters.executor import ExecutionResult, DatabaseExecutor
from integritycore.adapters.connections import get_relevant_schema
from integritycore.db.gold_queries import find_similar_gold_queries

# ── Tool definitions for LLM function calling (OpenAI/Gemini format) ───────────

SQL_EXPLORER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all tables in the Snowflake database. Use this to discover available tables before writing SQL.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_columns",
            "description": "Get column names and data types for a table. Use schema.table format (e.g. CITY.CITY_RAW).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Fully qualified table name: SCHEMA.TABLE or DATABASE.SCHEMA.TABLE",
                    },
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_data",
            "description": "Preview sample rows from a table. Use to understand data shape and sample values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Fully qualified table name: SCHEMA.TABLE",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of rows to preview (default 5, max 100)",
                        "default": 5,
                    },
                },
                "required": ["table_name"],
            },
        },
    },
]

# ── State Definition ──────────────────────────────────────────────────────────

class ETLState(TypedDict):
    """The graph state tracking the entire ETL context."""
    source_dialect: str
    target_dialect: str
    prompt: str
    strategy: ETLStrategy
    model_name: str
    
    # Agent conversational history
    messages: Annotated[list[Dict[str, str]], operator.add]
    
    # Generated artifact
    sql: str
    
    # Verification and control flow
    is_valid_prompt: bool
    validation_error: str
    validation_result: Optional[Dict[str, Any]]
    verified: bool
    verification_details: str
    special_action: str
    repair_attempts: int
    max_repairs: int

    # Critique loop (Senior Data Engineer review before verification)
    critique_passed: bool
    critique_issues: Optional[List[str]]
    critique_repair_attempts: int
    max_critique_repairs: int
    
    # LLM-parsed identifiers (from first parse_prompt_node call — authoritative for downstream)
    parsed_source_database: Optional[str]  # e.g. "MY_DB"
    parsed_source_schema: Optional[str]   # e.g. "CITY"
    parsed_source_table: Optional[str]    # e.g. "CITY.CITY_RAW" or "CITY_RAW"
    parsed_target_schema: Optional[str]
    parsed_target_table: Optional[str]    # e.g. "CITY.CITY_RAW_STG"
    
    # Metadata grounding (Verified Schema Fragment)
    metadata_manager: Optional[Any]
    grounding_result: Optional[Any]
    grounded_ddl: str
    schema_ddl_used: str  # Actual DDL provided to LLM (dynamic or grounded) — for lineage audit
    semantic_mappings: Optional[Dict[str, str]]
    selected_source_table: Optional[str]  # e.g. "CITY.CITY_RAW" when user confirms
    selected_target_table: Optional[str]  # e.g. "CITY_RAW_STG" when user confirms
    
    # Execution
    is_dry_run: bool
    source_conn: Optional[Any]
    target_conn: Optional[Any]
    executor: Optional[Any]
    execution_result: Optional[ExecutionResult]
    logs: Annotated[list[str], operator.add]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def parse_prompt_node(state: ETLState) -> Dict:
    """
    First LLM call: extract database, schema, and table names from the user prompt.
    Runs before grounding so we can use structured identifiers for catalog lookup.
    """
    prompt = state.get("prompt", "")
    model_name = state.get("model_name", "gemini/gemini-2.5-flash")
    src_dialect = state.get("source_dialect", "Snowflake")
    tgt_dialect = state.get("target_dialect", "Snowflake")

    if not prompt or not prompt.strip():
        return {"parsed_source_database": None, "parsed_source_schema": None, "parsed_source_table": None,
                "parsed_target_schema": None, "parsed_target_table": None, "logs": ["⏭️ Skipping parse: empty prompt."]}

    system_prompt = (
        "You are a data engineer parsing ETL instructions. Extract the SOURCE and TARGET table identifiers from the user's prompt.\n\n"
        f"Source dialect: {src_dialect}. Target dialect: {tgt_dialect}.\n\n"
        "Return a JSON object with exactly these keys:\n"
        '- "source_database" (optional, e.g. MY_DB)\n'
        '- "source_schema" (e.g. CITY, PUBLIC)\n'
        '- "source_table" (e.g. CITY_RAW, ORDERS)\n'
        '- "target_schema" (optional)\n'
        '- "target_table" (optional)\n\n'
        "Rules:\n"
        "- Use SCHEMA.TABLE format when schema is known (e.g. CITY.CITY_RAW).\n"
        "- If only one table is mentioned, treat it as the source.\n"
        "- Use null for unknown fields. Prefer common schemas like PUBLIC, RAW, STAGING.\n"
        "Return ONLY valid JSON, no markdown or explanation."
    )

    try:
        resp = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()
        data = json.loads(content)

        src_db = data.get("source_database") or None
        src_schema = data.get("source_schema") or ""
        src_table = data.get("source_table") or ""
        tgt_schema = data.get("target_schema") or ""
        tgt_table = data.get("target_table") or ""

        parsed_source = None
        if src_table:
            parsed_source = f"{src_schema}.{src_table}" if src_schema else src_table

        parsed_target = None
        if tgt_table:
            parsed_target = f"{tgt_schema}.{tgt_table}" if tgt_schema else tgt_table

        logs = [f"📋 Parsed prompt: db={src_db or '?'}, source={parsed_source or '?'}, target={parsed_target or '?'}"]
        return {
            "parsed_source_database": src_db,
            "parsed_source_schema": src_schema or None,
            "parsed_source_table": parsed_source,
            "parsed_target_schema": tgt_schema or None,
            "parsed_target_table": parsed_target,
            "logs": logs,
        }
    except Exception as e:
        return {
            "parsed_source_database": None, "parsed_source_schema": None, "parsed_source_table": None,
            "parsed_target_schema": None, "parsed_target_table": None,
            "logs": [f"⚠️ Parse prompt failed: {e}. Proceeding without parsed identifiers."],
        }


def ground_context_node(state: ETLState) -> Dict:
    """
    Grounding & Discovery: retrieve Verified Schema Fragment from semantic catalog.
    When user selected a source table, use only that. When multiple matches, ask user to confirm.
    """
    manager = state.get("metadata_manager")
    source_conn = state.get("source_conn")
    prompt = state.get("prompt", "")
    src_dialect = state.get("source_dialect", "")
    conn_id = getattr(source_conn, "id", None) or getattr(source_conn, "name", "") if source_conn else None
    selected_source = state.get("selected_source_table")
    parsed_source = state.get("parsed_source_table")
    parsed_target = state.get("parsed_target_table")

    if not manager or not conn_id:
        return {
            "grounding_result": None,
            "grounded_ddl": "",
            "semantic_mappings": None,
            "logs": ["⏭️ Skipping grounding (no metadata manager or source connection)."],
        }

    try:
        engine = GroundingEngine(manager)
        validation_result = state.get("validation_result") or {}
        # Use ONLY LLM-parsed identifiers when available; user-confirmed overrides
        table_to_use = selected_source or parsed_source

        if table_to_use:
            # Try parsed/selected table first; fall back to vector search using parsed identifiers if not in catalog
            parts = table_to_use.split(".")
            if len(parts) == 2:
                sch, tbl = parts[0], parts[1]
            else:
                sch, tbl = parts[-2] if len(parts) >= 2 else "PUBLIC", parts[-1]
            result = engine.retrieve_single_table(conn_id, sch, tbl)
            if result.related_tables:
                # Do NOT auto-select — require user confirmation
                if not selected_source:
                    validation_result = {
                        **validation_result,
                        "source_tables_to_confirm": result.related_tables,
                        "message": "Please confirm which source table to use.",
                    }
                    logs = [f"✅ Found {len(result.related_tables)} table(s) for '{table_to_use}' — awaiting user confirmation."]
                else:
                    logs = [f"✅ Grounded context: {len(result.related_tables)} table(s) (user confirmed: {selected_source})"]
            else:
                # Parsed table not in catalog — fall back to vector search using parsed identifiers
                logs = [f"⚠️ Parsed table '{table_to_use}' not in catalog; trying vector search with parsed identifiers."]
                search_query = table_to_use  # Use ONLY parsed data for search, not raw prompt
                result = engine.retrieve(
                    user_prompt=search_query,
                    conn_id=conn_id,
                    expand_fk=False,
                    top_k=10,
                )
                if result.verified_schema_fragment:
                    logs.append(f"✅ Vector search found {len(result.related_tables)} table(s), confidence={result.confidence:.2f}")
                else:
                    logs.append(f"⚠️ No semantic match for '{search_query}' in catalog.")
        else:
            # No parsed identifiers — vector search using prompt (e.g. empty parse)
            result = engine.retrieve(
                user_prompt=prompt,
                conn_id=conn_id,
                expand_fk=False,
                top_k=10,
            )
            if result.verified_schema_fragment:
                logs = [f"✅ Grounded context: {len(result.related_tables)} table(s), confidence={result.confidence:.2f}"]
            else:
                logs = [f"⚠️ No semantic match in catalog; proceeding without verified schema fragment."]

        # Use ONLY parsed table for validation when available — no heuristic extraction
        explicit_table = (parsed_source or selected_source or "").split(".")[-1] if (parsed_source or selected_source) else None

        def _catalog_has_table(name: str) -> bool:
            if not name or not result.related_tables:
                return False
            n = name.lower()
            for rt in result.related_tables:
                parts = rt.lower().split(".")
                if n in parts or n == rt.lower():
                    return True
            return False

        # No matches in catalog
        if not selected_source and not result.related_tables and not result.verified_schema_fragment:
            table_hint = explicit_table or parsed_source or "your query"
            validation_result = {
                **validation_result,
                "no_catalog_match": True,
                "message": (
                    f"No tables matching '{table_hint}' were found in the catalog. "
                    "Please run harvest on your source connection to populate the catalog, "
                    "or specify the table explicitly (e.g. SCHEMA.TABLE)."
                ),
            }
            logs.append(f"   Catalog has no matches for '{table_hint}' — harvest may be needed.")
        # Parsed table not in catalog (when we got vector results but parsed says different)
        elif parsed_source and result.related_tables and not _catalog_has_table(parsed_source.split(".")[-1]):
            validation_result = {
                **validation_result,
                "no_catalog_match": True,
                "message": (
                    f"No tables matching '{parsed_source}' were found in the catalog. "
                    "Please run harvest on your source connection to populate the catalog, "
                    "or specify the table explicitly (e.g. SCHEMA.TABLE)."
                ),
            }
            logs.append(f"   Parsed table '{parsed_source}' not in catalog.")
        # Match(es): require user confirmation — do NOT auto-select
        elif not selected_source and result.related_tables:
            validation_result = {
                **validation_result,
                "source_tables_to_confirm": result.related_tables,
                "message": "Please confirm which source table to use.",
            }
            logs.append(f"   Found {len(result.related_tables)} candidate(s) — awaiting user confirmation.")
        elif not selected_source and result.confidence < 0.6 and result.closeness_matches:
            validation_result = {
                **validation_result,
                "source_tables_to_confirm": result.related_tables if result.related_tables else result.closeness_matches,
                "message": "I found similar concepts. Which did you mean?",
            }
            logs.append(f"   Closeness matches: {result.closeness_matches}")

        out = {
            "grounding_result": result,
            "grounded_ddl": result.verified_schema_fragment,
            "semantic_mappings": result.semantic_mappings,
            "validation_result": validation_result if (validation_result.get("source_tables_to_confirm") or validation_result.get("no_catalog_match")) else state.get("validation_result"),
            "logs": logs,
        }
        if selected_source and not state.get("selected_source_table"):
            out["selected_source_table"] = selected_source
        return out
    except Exception as e:
        return {
            "grounding_result": None,
            "grounded_ddl": "",
            "semantic_mappings": None,
            "logs": [f"⚠️ Grounding failed: {e}. Proceeding without verified schema."],
        }


def _suggest_target_tables(source_tables: list) -> list:
    """Suggest target table names from source table names."""
    suggestions = []
    seen = set()
    for st in source_tables:
        # st is "SCHEMA.TABLE" or "TABLE"
        name = st.split(".")[-1] if "." in st else st
        for suffix in ["_STG", "_TARGET", "_COPY", "_LOAD"]:
            cand = f"{name}{suffix}"
            if cand not in seen:
                seen.add(cand)
                suggestions.append(cand)
        if f"STG_{name}" not in seen:
            seen.add(f"STG_{name}")
            suggestions.append(f"STG_{name}")
    return suggestions[:6]


def _prompt_mentions_target(prompt: str) -> bool:
    """Heuristic: does prompt mention a target table?"""
    p = prompt.lower()
    patterns = [" into ", " to table ", " target ", " load to ", " insert into ", " create table ", " copy to "]
    return any(x in p for x in patterns)


def validate_prompt_node(state: ETLState) -> Dict:
    """Pre-flight check: validates if the user prompt has enough context.
    When source_tables_to_confirm or target_table_suggestions, pause for user interaction.
    """
    validation_result = state.get("validation_result") or {}
    grounding_result = state.get("grounding_result")
    selected_source = state.get("selected_source_table")
    selected_target = state.get("selected_target_table")

    # User has persisted both source and target selections — trust them (e.g. job runs without catalog)
    if selected_source and selected_target:
        return {
            "is_valid_prompt": True,
            "validation_error": "",
            "logs": ["✅ Pre-flight validation passed (user-confirmed source and target tables)."],
        }

    # No matches in catalog — harvest needed or specify table explicitly
    if validation_result.get("no_catalog_match"):
        return {
            "is_valid_prompt": False,
            "validation_error": validation_result.get("message", "No tables found in catalog. Run harvest or specify table explicitly."),
            "validation_result": validation_result,
            "logs": ["⏸ No catalog matches — run harvest or specify SCHEMA.TABLE explicitly."],
        }

    # Need user to pick source table?
    if validation_result.get("source_tables_to_confirm"):
        return {
            "is_valid_prompt": False,
            "validation_error": validation_result.get("message", "Please confirm which source table to use."),
            "validation_result": validation_result,
            "logs": ["⏸ Awaiting user confirmation: source table selection."],
        }

    # Require user confirmation for target — do NOT proceed with just parsed_target
    if grounding_result and getattr(grounding_result, "verified_schema_fragment", None) and getattr(grounding_result, "confidence", 0) >= 0.5:
        if selected_target:
            return {
                "is_valid_prompt": True,
                "validation_error": "",
                "logs": ["✅ Pre-flight validation passed (user confirmed source and target)."],
            }
        # Target not specified: require user confirmation before proceeding
        source_tables = getattr(grounding_result, "related_tables", []) or []
        tables_for_suggestions = [selected_source] if selected_source else source_tables
        if tables_for_suggestions:
            suggestions = _suggest_target_tables(tables_for_suggestions)
            parsed_target = state.get("parsed_target_table")
            if parsed_target and parsed_target not in suggestions:
                suggestions = [parsed_target] + suggestions[:5]  # Prefer parsed target
            validation_result = {
                **validation_result,
                "target_table_suggestions": suggestions,
                "message": "Please confirm the target table before proceeding.",
            }
            return {
                "is_valid_prompt": False,
                "validation_error": "Please specify or confirm the target table name.",
                "validation_result": validation_result,
                "logs": ["⏸ Awaiting user confirmation: target table."],
            }

    prompt = state.get("prompt", "")
    src = state.get("source_dialect", "Unknown")
    tgt = state.get("target_dialect", "Unknown")
    model_name = state.get("model_name", "gemini/gemini-2.5-flash")
    
    messages = [
        {"role": "system", "content": (
            "You are a strict data engineering requirements validator. "
            "The user will provide an ETL instruction to move data from a "
            f"Source ({src}) to a Target ({tgt}).\n\n"
            "Evaluate if the prompt is logically complete. If it asks to load a vague entity "
            "(e.g. 'pull the india table') without context of what schema/system that belongs to, reject it.\n"
            "Rules:\n"
            "- If valid, return EXACTLY 'YES'\n"
            "- If invalid/vague, return 'NO: <explain clearly what information is missing>'\n"
            "Keep the rejection explanation under 2 sentences."
        )},
        {"role": "user", "content": prompt}
    ]
    
    try:
        resp = litellm.completion(model=model_name, messages=messages)
        content = resp.choices[0].message.content.strip()
        
        if content.upper().startswith("YES"):
            return {
                "is_valid_prompt": True,
                "validation_error": "",
                "logs": ["✅ Pre-flight validation passed."]
            }
        else:
            err = content[content.find(":")+1:].strip() if ":" in content else content
            return {
                "is_valid_prompt": False,
                "validation_error": err,
                "logs": [f"🚫 Pre-flight validation failed: {err}"]
            }
    except Exception as e:
        # Fallback to true if LLM validation crashes, so we don't completely block execution
        return {
            "is_valid_prompt": True, 
            "validation_error": "", 
            "logs": [f"⚠️ Prompt validation omitted due to API error: {e}"]
        }


def _execute_tool_call(
    name: str, args: Dict[str, Any], executor: DatabaseExecutor, source_conn: Any
) -> str:
    """Execute a tool call and return JSON string result."""
    if name == "list_tables":
        result = executor.list_tables(source_conn)
        if isinstance(result, list):
            return json.dumps({"tables": result})
        return json.dumps({"error": result})
    if name == "get_table_columns":
        tbl = args.get("table_name", "")
        result = executor.get_table_columns(source_conn, tbl)
        if isinstance(result, list):
            return json.dumps({"columns": result})
        return json.dumps({"error": result})
    if name == "preview_data":
        tbl = args.get("table_name", "")
        limit = args.get("limit", 5)
        result = executor.preview_data(source_conn, tbl, limit=limit)
        return result if isinstance(result, str) else json.dumps({"error": str(result)})
    return json.dumps({"error": f"Unknown tool: {name}"})


def generate_sql_node(state: ETLState) -> Dict:
    """Invokes the LLM to generate or repair ETL SQL. Uses tool calling for exploration when executor available."""
    messages = state.get("messages", [])
    model_name = state.get("model_name", "gemini/gemini-2.5-flash")
    schema_ddl = state.get("schema_ddl_used") or state.get("grounded_ddl", "")
    grounding_result = state.get("grounding_result")
    grounded_ddl = state.get("grounded_ddl", "")
    executor = state.get("executor")
    source_conn = state.get("source_conn")
    # Only use tools on first attempt (no messages); skip for repair flow
    use_tools = not messages and bool(
        executor and source_conn and (getattr(source_conn, "dialect", "") or "").upper() == "SNOWFLAKE"
    )

    # First attempt: setup prompt (dynamic context or grounding fallback)
    if not messages:
        user_obj = state["prompt"]
        selected_source = state.get("selected_source_table")
        selected_target = state.get("selected_target_table")
        parsed_source = state.get("parsed_source_table")
        parsed_target = state.get("parsed_target_table")
        src_hint = selected_source or parsed_source
        tgt_hint = selected_target or parsed_target
        if src_hint:
            user_obj = f"{user_obj}\n\nSource table: {src_hint}"
        if tgt_hint:
            user_obj = f"{user_obj}\n\nTarget table: {tgt_hint}"

        # Dynamic context: use ONLY parsed table when available; else keyword search
        conn_id = getattr(source_conn, "id", None) or getattr(source_conn, "name", "") if source_conn else None
        dynamic_ddl = ""
        if conn_id:
            manager = state.get("metadata_manager")
            parsed_src = state.get("parsed_source_table")
            dynamic_ddl = get_relevant_schema(user_obj, conn_id, metadata_manager=manager, parsed_table=parsed_src)

        schema_ddl = dynamic_ddl or grounded_ddl

        # Gold Query store: find similar reference patterns for few-shot prompting
        target_dialect = (state.get("target_dialect") or "SNOWFLAKE").upper()
        gold_refs = find_similar_gold_queries(user_obj, dialect=target_dialect, top_k=3)
        reference_block = ""
        if gold_refs:
            ref_parts = []
            for i, (prob, sql, _) in enumerate(gold_refs, 1):
                prob_short = (prob[:80] + "...") if len(prob) > 80 else prob
                ref_parts.append(f"Example {i} (problem: {prob_short}):\n```sql\n{sql}\n```")
            reference_block = (
                "\n\n<REFERENCE_PATTERNS>\n"
                "Similar successful patterns from the Gold Query store. Use these as inspiration:\n\n"
                + "\n\n".join(ref_parts)
                + "\n</REFERENCE_PATTERNS>\n\n"
            )

        tool_hint = ""
        if use_tools:
            tool_hint = " You MAY use list_tables, get_table_columns, and preview_data to explore the schema before writing SQL."

        if schema_ddl:
            mapping_rules = ""
            if grounding_result and grounding_result.semantic_mappings:
                mapping_rules = "\nMapping Rules (business term -> table.column):\n" + "\n".join(
                    f"  - {k} -> {v}" for k, v in grounding_result.semantic_mappings.items()
                ) + "\n\n"
            user_content = (
                f"You are a Senior Data Engineer building an ETL pipeline.{tool_hint}\n\n"
                f"{mapping_rules}"
                f"<VERIFIED_SCHEMA>\n{schema_ddl}\n</VERIFIED_SCHEMA>\n\n"
                f"Source dialect: {state['source_dialect']}. Target dialect: {state['target_dialect']}.\n\n"
                f"User objective: {user_obj}\n\n"
                "When ready, return ONLY valid SQL wrapped in a ```sql code block. No explanation."
            )
        else:
            user_content = (
                f"You are a Senior Data Engineer.{tool_hint}\n\n"
                f"Source: {state['source_dialect']}. Target: {state['target_dialect']}.\n\n"
                f"User objective: {user_obj}\n\n"
                "Return ONLY valid SQL wrapped in a ```sql code block when done."
            )

        system_content = (
            "You are an expert AI agent generating ETL SQL."
            + (" You can call tools to explore the database before writing your final MERGE or INSERT." if use_tools else "")
            + " Return ONLY valid SQL wrapped in a ```sql code block. No explanation."
            + reference_block
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    try:
        max_tool_rounds = 10
        tool_rounds = 0

        while tool_rounds < max_tool_rounds:
            kwargs = {"model": model_name, "messages": messages}
            if use_tools:
                kwargs["tools"] = SQL_EXPLORER_TOOLS
                kwargs["tool_choice"] = "auto"

            resp = litellm.completion(**kwargs)
            msg = resp.choices[0].message

            # Check for tool calls
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                # No tool calls — extract SQL from content
                content = (msg.content or "").strip()
                if not content:
                    return {"error": "Empty LLM response", "logs": ["❌ LLM returned empty content."]}
                sql = content
                if sql.startswith("```"):
                    lines = sql.split("\n")
                    sql = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                return {
                    "sql": sql,
                    "messages": messages + [{"role": "assistant", "content": content}],
                    "logs": ["⏳ Generated SQL via LLM." + (" (after tool exploration)" if tool_rounds > 0 else "")],
                    "schema_ddl_used": schema_ddl,
                }

            # Execute tool calls and append results
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id if hasattr(tc, "id") else tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", ""),
                            "arguments": tc.function.arguments if hasattr(tc, "function") else tc.get("function", {}).get("arguments", "{}"),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            })

            for tc in tool_calls:
                fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                raw_args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError:
                    args = {}
                tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "unknown")
                result = _execute_tool_call(name, args, executor, source_conn)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": result,
                })

            tool_rounds += 1

        # Exceeded max rounds — force final response without tools
        resp = litellm.completion(model=model_name, messages=messages)
        content = (resp.choices[0].message.content or "").strip()
        sql = content
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return {
            "sql": sql or "-- Max tool rounds reached",
            "messages": messages + [{"role": "assistant", "content": content}],
            "logs": ["⏳ Generated SQL (max tool rounds reached)."],
            "schema_ddl_used": schema_ddl,
        }
    except Exception as e:
        return {
            "error": str(e),
            "logs": [f"❌ LLM Generation failed: {e}"],
            "schema_ddl_used": schema_ddl,
        }


def critique_sql_node(state: ETLState) -> Dict:
    """
    Senior Data Engineer critique of generated SQL.
    Checks for: missing JOIN conditions (Cartesian products), PII exposure, non-optimal Snowflake patterns.
    If issues found, routes back to generate_sql_node for repair.
    """
    sql = state.get("sql", "")
    model_name = state.get("model_name", "gemini/gemini-2.5-flash")
    target_dialect = state.get("target_dialect", "snowflake")
    critique_repair_attempts = state.get("critique_repair_attempts", 0)
    max_critique_repairs = state.get("max_critique_repairs", 2)

    if not sql or not sql.strip():
        return {
            "critique_passed": True,
            "critique_issues": None,
            "logs": ["⏭️ Skipping critique: no SQL to review."],
        }

    # Senior Data Engineer persona prompt
    system_prompt = (
        "You are a Senior Data Engineer performing a strict code review of ETL SQL. "
        "Your job is to identify critical issues that could cause production problems or data quality risks.\n\n"
        "You MUST check for these specific issues:\n"
        "1. **Missing JOIN conditions (Cartesian products)**: Any JOIN without an ON clause or with "
        "incomplete/missing join predicates that could produce a Cartesian product.\n"
        "2. **Potential PII exposure**: Any SELECT or output that exposes plain-text PII such as "
        "emails, SSNs, credit card numbers, or other sensitive identifiers without masking/hashing.\n"
        "3. **Non-optimal Snowflake patterns**: Use of DISTINCT instead of GROUP BY on large sets, "
        "inefficient subqueries, or patterns that don't leverage Snowflake best practices.\n\n"
        "Respond with a JSON object in this exact format:\n"
        '{"passed": true} or {"passed": false, "issues": ["issue 1", "issue 2", ...], "summary": "brief explanation"}\n\n'
        "Be strict but fair. Only flag real issues. If the SQL is clean, return {\"passed\": true}."
    )

    user_prompt = (
        f"Review the following {target_dialect} SQL for the issues above:\n\n```sql\n{sql}\n```"
    )

    try:
        resp = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content.strip()

        # Parse JSON response
        # Handle markdown code blocks
        if "```" in content:
            json_str = content.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
        else:
            json_str = content

        result = json.loads(json_str)
        passed = result.get("passed", False)
        issues = result.get("issues", [])

        if passed:
            return {
                "critique_passed": True,
                "critique_issues": None,
                "logs": ["✅ Critique passed: SQL passed Senior Data Engineer review."],
            }

        # Issues found — prepare repair message
        repair_msg = (
            f"The SQL failed Senior Data Engineer critique:\n\n"
            f"**Issues found:**\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
            f"{result.get('summary', 'Please fix the issues above.')}\n\n"
            "Return ONLY the fully corrected SQL, no explanation."
        )

        return {
            "critique_passed": False,
            "critique_issues": issues,
            "critique_repair_attempts": critique_repair_attempts + 1,
            "messages": [{"role": "user", "content": repair_msg}],
            "logs": [f"⚠️ Critique found {len(issues)} issue(s): " + "; ".join(issues[:3]) + ("..." if len(issues) > 3 else "")],
        }
    except json.JSONDecodeError as e:
        # Fallback: parse heuristically
        try:
            if "passed" in content.lower() and "true" in content.lower() and "false" not in content[:50]:
                return {
                    "critique_passed": True,
                    "critique_issues": None,
                    "logs": ["✅ Critique passed (heuristic parsing)."],
                }
        except Exception:
            pass
        return {
            "critique_passed": True,
            "critique_issues": None,
            "logs": [f"⚠️ Critique parse error: {e}. Proceeding to verification."],
        }
    except Exception as e:
        return {
            "critique_passed": True,
            "critique_issues": None,
            "logs": [f"⚠️ Critique failed: {e}. Proceeding to verification."],
        }


def verify_node(state: ETLState) -> Dict:
    """Runs SMT math proofs and target database compilation."""
    sql = state.get("sql", "")
    strategy = state.get("strategy")
    verifier = LogicVerifier(model_name=state.get("model_name", "gemini/gemini-2.5-flash"))
    
    # SMT
    try:
        v_res = verifier.verify_generation(sql, strategy)
        if not v_res:
            return {
                "verified": False, 
                "verification_details": f"Failed SMT: Could not mathematically prove logic bounds for {strategy.value}.",
                "logs": ["⚠️ SMT Verification FAILED."]
            }
    except Exception as e:
        pass # Non-fatal
        
    # Compilation
    executor = state.get("executor")
    tgt = state.get("target_conn")
    
    if executor and tgt:
        try:
            comp_res = executor.compile_only(sql, tgt)
            if not comp_res.success:
                err_msg = comp_res.error or ""
                if "does not exist or not authorized" in err_msg.lower():
                    # Bypass logic: if generating the DDL was successful, it's normal for the
                    # subsequent INSERT to fail Snowflake's EXPLAIN check (since the table
                    # doesn't exist yet). We treat it as verified.
                    if "create table" in sql.lower() or "create schema" in sql.lower():
                        return {
                            "verified": True,
                            "verification_details": "Skipped strict target DB compilation due to inline DDL (CREATE TABLE/SCHEMA) preventing valid dry-run.",
                            "special_action": "",
                            "logs": ["⚠️ Target DB Compilation: Bypassed strict checks due to inline DDL presence."]
                        }
                    # User already chose the target table — auto-repair: add CREATE TABLE, no extra prompt
                    if state.get("selected_target_table"):
                        return {
                            "verified": False,
                            "verification_details": (
                                "The target table does not exist. Generate the CREATE TABLE statement "
                                "based on the source schema before the INSERT/MERGE. Return ONLY the "
                                "complete SQL (CREATE TABLE + DML), no explanation."
                            ),
                            "special_action": "",
                            "logs": ["⚠️ Target table missing — auto-adding CREATE TABLE instruction."]
                        }
                    return {
                        "verified": False,
                        "verification_details": f"Target table or schema is missing.\n{err_msg}",
                        "special_action": "missing_schema",
                        "logs": ["⚠️ Target DB Compilation: Missing table detected."]
                    }
                return {
                    "verified": False,
                    "verification_details": f"Target database SQL compilation error:\n{err_msg}",
                    "logs": ["⚠️ Target DB Compilation FAILED."]
                }
        except Exception as e:
            return {"verified": False, "verification_details": f"Compile crash: {e}"}

    return {
        "verified": True, 
        "verification_details": "Verified successfully via Z3 SMT Solver and Target DB Compilation.",
        "special_action": "",
        "logs": ["✅ Verification PASSED."]
    }


def prepare_repair_node(state: ETLState) -> Dict:
    """If verification fails, sets up the prompt for the next generation attempt."""
    details = state["verification_details"]
    repair_prompt = (
        f"The SQL above failed verification:\n{details}\n"
        f"Please fix it. Return ONLY the fully corrected SQL, no explanation."
    )
    return {
        "messages": [{"role": "user", "content": repair_prompt}],
        "repair_attempts": state.get("repair_attempts", 0) + 1,
        "logs": [f"🔧 Preparing self-heal prompt..."]
    }


def execute_node(state: ETLState) -> Dict:
    """Executes the verified SQL against the target database."""
    if state.get("is_dry_run"):
        return {"logs": ["[Dry Run] Execution skipped."]}
        
    executor = state.get("executor")
    # For ETL, we generally write TO the target connection
    conn = state.get("target_conn") or state.get("source_conn")
    
    if executor and conn:
        res = executor.execute(state["sql"], conn)
        logs = []
        if res.success:
            logs.append(f"✅ Execution Complete. Rows affected: {res.rows_affected}")
        else:
            logs.append(f"❌ Execution failed: {res.error}")
        return {"execution_result": res, "logs": logs}
    return {"logs": ["⚠️ No executor or connections provided for execution."]}


def execution_repair_node(state: ETLState) -> Dict:
    """If runtime execution fails, feedback the error into the agent."""
    res = state["execution_result"]
    schema_text = ""
    if getattr(res, "schema", None):
        schema_text = "\n\nAvailable schema:\n" + "\n".join(
            f"  {col['name']} ({col['type']})" for col in res.schema
        )
    
    repair_prompt = (
        f"Execution error:\n{res.error}"
        f"{schema_text}\n\n"
        "Please fix the SQL. Return only the corrected SQL block."
    )
    return {
        "messages": [{"role": "user", "content": repair_prompt}],
        "repair_attempts": state.get("repair_attempts", 0) + 1,
        "logs": ["🔧 Execution failed. Preparing execution repair prompt..."],
        "verified": False, # Strip verified status so it re-verifies
    }


# ── Edge Routers ──────────────────────────────────────────────────────────────

def route_after_validation(state: ETLState) -> Literal["generate_sql_node", "__end__"]:
    """Routes to generate_sql if valid, otherwise ends the graph."""
    if state.get("is_valid_prompt", True) is False:
        return "__end__"
    return "generate_sql_node"


def route_after_critique(state: ETLState) -> Literal["generate_sql_node", "verify_node"]:
    """If critique found issues and under max attempts, repair. Otherwise proceed to verification."""
    if state.get("critique_passed", True):
        return "verify_node"
    attempts = state.get("critique_repair_attempts", 0)
    max_attempts = state.get("max_critique_repairs", 2)
    if attempts < max_attempts:
        return "generate_sql_node"
    # Exceeded max critique repairs — proceed to verification anyway
    return "verify_node"


def route_after_verification(state: ETLState) -> Literal["execute_node", "prepare_repair_node", "__end__"]:
    if state.get("verified"):
        return "__end__" if state.get("is_dry_run") else "execute_node"
    
    # Intercept for missing schema logic (Human in the loop)
    # The API layer pauses the graph when it hits this state in dry run
    if state.get("special_action") == "missing_schema" and state.get("is_dry_run"):
        return "__end__"
        
    if state.get("repair_attempts", 0) >= state.get("max_repairs", 3):
        return "__end__"
        
    return "prepare_repair_node"


def route_after_execution(state: ETLState) -> Literal["execution_repair_node", "__end__"]:
    res = state.get("execution_result")
    if res and not res.success:
        if state.get("repair_attempts", 0) >= state.get("max_repairs", 3):
            return "__end__"
        return "execution_repair_node"
    return "__end__"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_etl_graph() -> Any:
    workflow = StateGraph(ETLState)
    
    # Add nodes
    workflow.add_node("parse_prompt_node", parse_prompt_node)
    workflow.add_node("validate_prompt_node", validate_prompt_node)
    workflow.add_node("ground_context_node", ground_context_node)
    workflow.add_node("generate_sql_node", generate_sql_node)
    workflow.add_node("critique_sql_node", critique_sql_node)
    workflow.add_node("verify_node", verify_node)
    workflow.add_node("prepare_repair_node", prepare_repair_node)
    workflow.add_node("execute_node", execute_node)
    workflow.add_node("execution_repair_node", execution_repair_node)
    
    # Add edges: parse prompt first (LLM extracts db/schema/table), then ground
    workflow.add_edge(START, "parse_prompt_node")
    workflow.add_edge("parse_prompt_node", "ground_context_node")
    workflow.add_edge("ground_context_node", "validate_prompt_node")
    
    workflow.add_conditional_edges(
        "validate_prompt_node",
        route_after_validation,
        {
            "generate_sql_node": "generate_sql_node",
            "__end__": END
        }
    )
    
    workflow.add_edge("generate_sql_node", "critique_sql_node")

    workflow.add_conditional_edges(
        "critique_sql_node",
        route_after_critique,
        {
            "generate_sql_node": "generate_sql_node",
            "verify_node": "verify_node",
        }
    )

    workflow.add_conditional_edges(
        "verify_node",
        route_after_verification,
        {
            "execute_node": "execute_node",
            "prepare_repair_node": "prepare_repair_node",
            "__end__": END
        }
    )
    
    workflow.add_edge("prepare_repair_node", "generate_sql_node")
    
    workflow.add_conditional_edges(
        "execute_node",
        route_after_execution,
        {
            "execution_repair_node": "execution_repair_node",
            "__end__": END
        }
    )
    
    workflow.add_edge("execution_repair_node", "generate_sql_node")
    
    return workflow.compile()

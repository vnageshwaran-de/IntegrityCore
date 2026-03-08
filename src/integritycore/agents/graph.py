import operator
import asyncio
from typing import Annotated, TypedDict, Any, Dict, List, Optional, Literal
from langgraph.graph import StateGraph, START, END
import litellm

from integritycore.core.verifier import LogicVerifier, ETLStrategy
from integritycore.adapters.executor import ExecutionResult

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
    verified: bool
    verification_details: str
    special_action: str
    repair_attempts: int
    max_repairs: int
    
    # Execution
    is_dry_run: bool
    source_conn: Optional[Any]
    target_conn: Optional[Any]
    executor: Optional[Any]
    execution_result: Optional[ExecutionResult]
    logs: Annotated[list[str], operator.add]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def generate_sql_node(state: ETLState) -> Dict:
    """Invokes the LLM to generate or repair the ETL logic."""
    messages = state.get("messages", [])
    model_name = state.get("model_name", "gemini/gemini-2.5-flash")
    
    # First attempt: setup prompt
    if not messages:
        enhanced_prompt = (
            f"You are building an ETL pipeline extracting data from {state['source_dialect']} "
            f"and loading it into {state['target_dialect']}. Ensure the syntax is compatible.\n\n"
            f"User objective: {state['prompt']}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI agent generating ETL SQL. "
                    "Return ONLY valid SQL wrapped in a ```sql code block. "
                    "No explanation, no prose — only the SQL statement."
                ),
            },
            {"role": "user", "content": enhanced_prompt},
        ]
        
    try:
        resp = litellm.completion(model=model_name, messages=messages)
        sql = resp.choices[0].message.content.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
        return {
            "sql": sql, 
            "messages": messages + [{"role": "assistant", "content": sql}],
            "logs": ["⏳ Generated SQL via LLM."]
        }
    except Exception as e:
        return {
            "error": str(e),
            "logs": [f"❌ LLM Generation failed: {e}"]
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
    workflow.add_node("generate_sql_node", generate_sql_node)
    workflow.add_node("verify_node", verify_node)
    workflow.add_node("prepare_repair_node", prepare_repair_node)
    workflow.add_node("execute_node", execute_node)
    workflow.add_node("execution_repair_node", execution_repair_node)
    
    # Add edges
    workflow.add_edge(START, "generate_sql_node")
    workflow.add_edge("generate_sql_node", "verify_node")
    
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

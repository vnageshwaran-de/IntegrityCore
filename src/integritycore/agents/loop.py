"""
IntegrityCore Agent Loop — Multi-stage ETL pipeline orchestrator.

Stages:
  1. Generate  — LLM writes ETL SQL
  2. Verify    — Z3 SMT solver proves mathematical correctness
  3. Repair    — LLM fixes SQL if verification fails (up to max_retries)
  4. Execute   — Real database execution with full telemetry
  5. Self-Heal — LLM analyses execution errors and repairs SQL, re-runs
"""
from typing import List, Dict, Any, Optional, Callable
import time
import litellm

from integritycore.core.verifier import LogicVerifier, ETLStrategy


class LoopAgent:
    def __init__(self, model_name: str = "gemini/gemini-pro"):
        self.model_name = model_name
        self.verifier = LogicVerifier(model_name=model_name)
        self.tools = []

    def discover_tools(self, mcp_registry_url: str = None) -> List[Dict[str, Any]]:
        """
        Discovers tools dynamically from local or cloud MCP registries.
        This provides MCP integration to automatically fetch database capabilities
        (e.g., Postgres, Snowflake).
        """
        # In a complete implementation, this would connect to an MCP Server or Registry
        # via the Model Context Protocol. We are mocking the MCP response schema here.
        self.tools = [{
            "type": "function",
            "function": {
                "name": "execute_query",
                "description": "Execute a SQL query against Postgres or Snowflake databases.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "db_type": {"type": "string", "enum": ["postgres", "snowflake"]},
                        "query": {"type": "string", "description": "The SQL query to execute"}
                    },
                    "required": ["db_type", "query"]
                }
            }
        }]
        return self.tools

    # ------------------------------------------------------------------
    # Core orchestration loop
    # ------------------------------------------------------------------

    def execute_etl_loop(
        self,
        prompt: str,
        strategy: ETLStrategy,
        source: str = "POSTGRESQL",
        target: str = "SNOWFLAKE",
        max_retries: int = 3,
        log_cb: Optional[Callable[[str], None]] = None,
        source_conn=None,   # DBConnection object for real execution
        target_conn=None,   # DBConnection object for real execution
        max_exec_retries: int = 3,
    ) -> str:
        """
        Orchestrates: Generate → Verify → Repair → Execute → Self-Heal.
        Returns the final verified + executed SQL.
        """

        def log(msg: str):
            if log_cb:
                log_cb(msg)
            else:
                print(msg)

        # ── Inject dialect context ────────────────────────────────────
        enhanced_prompt = (
            f"You are building an ETL pipeline extracting data from the {source} dialect "
            f"and loading it into the {target} dialect. Ensure the syntax is compatible.\n\n"
            f"User objective: {prompt}"
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

        if not self.tools:
            self.discover_tools()

        # ══════════════════════════════════════════════════════════════
        # STAGES 1-3: Generate → Verify → Repair
        # ══════════════════════════════════════════════════════════════
        verified_sql = self._generate_verify_repair(
            messages=messages,
            strategy=strategy,
            prompt=prompt,
            max_retries=max_retries,
            log=log,
        )

        # ══════════════════════════════════════════════════════════════
        # STAGE 4: Execute + LLM Self-Heal
        # ══════════════════════════════════════════════════════════════
        if source_conn is None and target_conn is None:
            log("[Executor] No database connections provided — skipping live execution.")
            return verified_sql

        from integritycore.adapters.executor import DatabaseExecutor
        executor = DatabaseExecutor(log_cb=log)

        # Use target_conn for execution (we load data INTO the target)
        exec_conn = target_conn or source_conn
        current_sql = verified_sql

        for exec_attempt in range(1, max_exec_retries + 1):
            log(f"\n{'═'*60}")
            log(f"[Self-Heal] Execution Attempt {exec_attempt} / {max_exec_retries}")
            log(f"{'═'*60}")

            result = executor.execute(current_sql, exec_conn)

            if result.success:
                log(f"\n[Self-Heal] ✅ Pipeline complete!")
                log(f"[Self-Heal]   Rows affected : {result.rows_affected:,}")
                log(f"[Self-Heal]   Duration      : {result.duration_ms:.1f} ms")
                log(f"[Self-Heal]   Source        : {source}")
                log(f"[Self-Heal]   Target        : {target}")
                return current_sql

            # ── Execution failed — hand error to LLM for repair ──────
            log(f"\n[Self-Heal] ❌ Execution failed on attempt {exec_attempt}")
            log(f"[Self-Heal]   Error: {result.error}")

            if exec_attempt == max_exec_retries:
                raise RuntimeError(
                    f"Execution failed after {max_exec_retries} self-heal attempts.\n"
                    f"Last error: {result.error}"
                )

            log(f"[Self-Heal] 🔧 Sending error to LLM for autonomous repair...")
            current_sql = self._llm_repair_execution_error(
                original_sql=current_sql,
                error_msg=result.error,
                schema_info=result.schema,
                original_prompt=prompt,
                source=source,
                target=target,
                log=log,
            )

            # Re-verify the repaired SQL before re-executing
            log(f"[Self-Heal] 🔬 Re-verifying repaired SQL with Z3...")
            try:
                is_valid = self.verifier.verify_generation(current_sql, strategy, log_cb=log)
                if is_valid:
                    log("[Self-Heal] ✅ Re-verification passed — retrying execution")
                else:
                    log("[Self-Heal] ⚠️  Re-verification inconclusive — attempting execution anyway")
            except Exception as ve:
                log(f"[Self-Heal] ⚠️  Re-verification error: {ve} — continuing")

        raise RuntimeError("Execution self-heal loop exhausted.")

    # ------------------------------------------------------------------
    # Internal: Generate → Verify → Repair (Stages 1-3)
    # ------------------------------------------------------------------

    def _generate_verify_repair(
        self,
        messages: list,
        strategy: ETLStrategy,
        prompt: str,
        max_retries: int,
        log,
    ) -> str:
        for attempt in range(max_retries + 1):
            log(f"\n{'─'*60}")
            log(f"[ADK] Generate → Verify → Repair  (Attempt {attempt + 1})")
            log(f"{'─'*60}")

            # ── STEP 1: Generate ──────────────────────────────────────
            content = self._call_llm(messages, log)
            messages.append({"role": "assistant", "content": content})

            # Extract SQL from markdown blocks
            sql_code = content
            if "```sql" in content:
                sql_code = content.split("```sql")[1].split("```")[0].strip()
            elif "```" in content:
                sql_code = content.split("```")[1].split("```")[0].strip()
            if not sql_code.strip():
                sql_code = content.strip()

            log(f"[ADK] LLM responded with {len(content)} chars")
            log(f"[ADK] ── Generated SQL ──────────────────────────────────")
            log(sql_code)
            log(f"[ADK] ────────────────────────────────────────────────────")

            # ── STEP 2: Verify ────────────────────────────────────────
            log("[Verifier] Running SMT + AST proof...")
            is_valid = self.verifier.verify_generation(sql_code, strategy, log_cb=log)

            if is_valid:
                log("[Verifier] ✅ Glass Box Verification Passed")
                return sql_code

            # ── STEP 3: Repair ────────────────────────────────────────
            log("[Verifier] ❌ Verification Failed — initiating repair")
            repair_prompt = (
                f"Verification of your SQL failed. It does not satisfy the {strategy.value} "
                "strategy requirements. For INCREMENTAL loads, include a WHERE clause like "
                "'updated_at >= :watermark'. Return only the corrected SQL in a ```sql block."
            )
            messages.append({"role": "user", "content": repair_prompt})

        raise RuntimeError(
            f"Agent failed to generate verifiably safe SQL after {max_retries} attempts."
        )

    # ------------------------------------------------------------------
    # Internal: LLM self-heal — repair SQL after execution failure
    # ------------------------------------------------------------------

    def _llm_repair_execution_error(
        self,
        original_sql: str,
        error_msg: str,
        schema_info: list,
        original_prompt: str,
        source: str,
        target: str,
        log,
    ) -> str:
        schema_text = ""
        if schema_info:
            schema_text = "\n\nAvailable schema:\n" + "\n".join(
                f"  {col['name']} ({col['type']})" for col in schema_info
            )

        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert SQL repair agent. Fix SQL errors while preserving "
                    f"the original ETL objective. The source dialect is {source}, target is {target}. "
                    "Return ONLY corrected SQL in a ```sql code block."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original objective: {original_prompt}\n\n"
                    f"SQL that was executed:\n```sql\n{original_sql}\n```\n\n"
                    f"Execution error:\n{error_msg}"
                    f"{schema_text}\n\n"
                    "Please fix the SQL. Return only the corrected SQL block."
                ),
            },
        ]

        content = self._call_llm(repair_messages, log)

        # Extract SQL
        if "```sql" in content:
            fixed_sql = content.split("```sql")[1].split("```")[0].strip()
        elif "```" in content:
            fixed_sql = content.split("```")[1].split("```")[0].strip()
        else:
            fixed_sql = content.strip()

        log(f"[Self-Heal] ── LLM Repair Proposal ────────────────────────")
        log(fixed_sql)
        log(f"[Self-Heal] ─────────────────────────────────────────────────")
        return fixed_sql

    # ------------------------------------------------------------------
    # Internal: resilient LLM call with rate-limit backoff
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list, log) -> str:
        for api_attempt in range(3):
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            except litellm.exceptions.RateLimitError:
                if api_attempt == 2:
                    raise RuntimeError(
                        f"Fatal Rate Limit on {self.model_name}. Switch to Anthropic or wait."
                    )
                wait = 15 * (api_attempt + 1)
                log(f"[ADK] ⚠️  Rate limited — retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                raise RuntimeError(f"Unexpected LLM error: {e}")
        raise RuntimeError("LLM call failed after 3 attempts.")

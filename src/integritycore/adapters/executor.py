"""
DatabaseExecutor — Real SQL execution adapter for IntegrityCore.

Supports Snowflake (initially). Returns structured ExecutionResult with full
telemetry so the LLM self-heal loop can analyse failures and repair SQL.
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from integritycore.adapters.connections import DBConnection


@dataclass
class ExecutionResult:
    success: bool
    rows_affected: int = 0
    duration_ms: float = 0.0
    schema: List[Dict[str, str]] = field(default_factory=list)
    sample_rows: List[tuple] = field(default_factory=list)
    error: Optional[str] = None
    query_id: Optional[str] = None


class DatabaseExecutor:
    """
    Executes verified SQL against a real database connection.
    All steps are surfaced via log_cb so they stream live to the UI.
    """

    def __init__(self, log_cb: Callable[[str], None] = print):
        self.log = log_cb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, sql: str, conn: DBConnection) -> ExecutionResult:
        """Connect, run SQL, return ExecutionResult with full telemetry."""
        dialect = (conn.dialect or "").upper()

        if dialect == "SNOWFLAKE":
            return self._execute_snowflake(sql, conn)
        else:
            return ExecutionResult(
                success=False,
                error=f"Dialect '{dialect}' not yet supported for live execution. Only SNOWFLAKE is supported."
            )

    # ------------------------------------------------------------------
    # Snowflake implementation
    # ------------------------------------------------------------------

    def _execute_snowflake(self, sql: str, conn: DBConnection) -> ExecutionResult:
        try:
            import snowflake.connector
        except ImportError:
            return ExecutionResult(
                success=False,
                error="snowflake-connector-python is not installed. Run: pip install snowflake-connector-python"
            )

        self.log(f"[Executor] ── Connecting to Snowflake ──────────────────────")
        self.log(f"[Executor]   Account   : {conn.account}")
        self.log(f"[Executor]   User      : {conn.username}")
        self.log(f"[Executor]   Database  : {conn.database}")
        self.log(f"[Executor]   Warehouse : {conn.warehouse}")

        try:
            sf_conn = snowflake.connector.connect(
                user=conn.username,
                password=conn.password,
                account=conn.account,
                database=conn.database,
                warehouse=conn.warehouse,
            )
            self.log("[Executor] ✅ Connection established")
        except Exception as e:
            err = f"Connection failed: {e}"
            self.log(f"[Executor] ❌ {err}")
            return ExecutionResult(success=False, error=err)

        try:
            cursor = sf_conn.cursor()

            # ── Schema introspection (best-effort) ────────────────────
            schema_info = self._introspect_schema(cursor, sql)

            # ── Execute the SQL ───────────────────────────────────────
            self.log(f"[Executor] ── Executing SQL ─────────────────────────────")
            self.log(f"[Executor]   Statement length: {len(sql)} chars")

            t0 = time.time()
            cursor.execute(sql)
            duration_ms = (time.time() - t0) * 1000

            query_id = cursor.sfqid
            rows_affected = cursor.rowcount or 0

            # Fetch a small sample for logging (SELECT statements)
            sample = []
            try:
                sample = cursor.fetchmany(5)
            except Exception:
                pass

            self.log(f"[Executor] ✅ Query complete")
            self.log(f"[Executor]   Query ID      : {query_id}")
            self.log(f"[Executor]   Rows affected : {rows_affected:,}")
            self.log(f"[Executor]   Duration      : {duration_ms:.1f} ms")

            if sample:
                self.log(f"[Executor] ── Sample Output (first {len(sample)} rows) ──────────")
                for row in sample:
                    self.log(f"[Executor]   {row}")

            return ExecutionResult(
                success=True,
                rows_affected=rows_affected,
                duration_ms=duration_ms,
                schema=schema_info,
                sample_rows=list(sample),
                query_id=query_id,
            )

        except Exception as e:
            err = str(e)
            self.log(f"[Executor] ❌ Execution error: {err}")
            return ExecutionResult(success=False, error=err)

        finally:
            try:
                sf_conn.close()
                self.log("[Executor] Connection closed.")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _introspect_schema(self, cursor, sql: str) -> List[Dict[str, str]]:
        """Best-effort: extract source table name and describe its columns."""
        schema_info = []
        try:
            # Very simple heuristic — look for FROM <table> in the SQL
            import re
            match = re.search(r'\bFROM\s+([A-Za-z0-9_."]+)', sql, re.IGNORECASE)
            if match:
                table = match.group(1).strip('"')
                self.log(f"[Executor] ── Schema introspection: {table} ───────────────")
                cursor.execute(f"DESCRIBE TABLE {table}")
                for row in cursor.fetchall():
                    col = {"name": row[0], "type": row[1]}
                    schema_info.append(col)
                    self.log(f"[Executor]   Column: {col['name']:30s}  Type: {col['type']}")
        except Exception as e:
            self.log(f"[Executor]   (Schema introspection skipped: {e})")
        return schema_info

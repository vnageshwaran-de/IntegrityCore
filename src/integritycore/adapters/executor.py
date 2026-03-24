"""
DatabaseExecutor — Real SQL execution adapter for IntegrityCore.

Supports Snowflake (initially). Returns structured ExecutionResult with full
telemetry so the LLM self-heal loop can analyse failures and repair SQL.

Provides read-only tool methods for LLM exploration: list_tables, get_table_columns, preview_data.
"""
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union

from integritycore.adapters.connections import DBConnection


def _sanitize_table_name(name: str) -> str:
    """Allow only schema.table format: alphanumeric, underscore, dot. Prevents SQL injection."""
    if not name or not isinstance(name, str):
        return ""
    # Allow: SCHEMA.TABLE, schema.table, TABLE
    cleaned = re.sub(r"[^a-zA-Z0-9_.]", "", name.strip())
    return cleaned


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
            return self._execute_snowflake(sql, conn, is_compile_only=False)
        else:
            return ExecutionResult(success=False, error=f"Dialect '{dialect}' not yet supported for live execution.")

    def compile_only(self, sql: str, conn: DBConnection) -> ExecutionResult:
        """Connect and run an EXPLAIN or dry-run compilation check without executing the query permanently."""
        dialect = (conn.dialect or "").upper()
        if dialect == "SNOWFLAKE":
            return self._execute_snowflake(sql, conn, is_compile_only=True)
        else:
            return ExecutionResult(success=False, error=f"Dialect '{dialect}' not yet supported for compile checks.")

    # ------------------------------------------------------------------
    # Tool methods for LLM exploration (read-only, safe)
    # ------------------------------------------------------------------

    def list_tables(self, conn: DBConnection) -> Union[List[str], str]:
        """
        List tables in the database. Returns list of 'SCHEMA.TABLE' strings, or error string.
        Safe: read-only INFORMATION_SCHEMA query.
        """
        dialect = (conn.dialect or "").upper()
        if dialect != "SNOWFLAKE":
            return f"Dialect '{dialect}' not supported for list_tables."
        try:
            return self._snowflake_list_tables(conn)
        except Exception as e:
            return str(e)

    def get_table_columns(self, conn: DBConnection, table_name: str) -> Union[List[Dict[str, str]], str]:
        """
        Get column names and types for a table. Returns list of {name, type}, or error string.
        Safe: uses DESCRIBE TABLE. table_name must be schema.table or table.
        """
        dialect = (conn.dialect or "").upper()
        if dialect != "SNOWFLAKE":
            return f"Dialect '{dialect}' not supported for get_table_columns."
        tbl = _sanitize_table_name(table_name)
        if not tbl:
            return "Invalid table name."
        try:
            return self._snowflake_describe_table(conn, tbl)
        except Exception as e:
            return str(e)

    def preview_data(self, conn: DBConnection, table_name: str, limit: int = 5) -> Union[str, str]:
        """
        Preview up to `limit` rows from a table. Returns JSON string of rows, or error string.
        Safe: SELECT * FROM table LIMIT n (read-only). limit capped at 100.
        """
        dialect = (conn.dialect or "").upper()
        if dialect != "SNOWFLAKE":
            return f"Dialect '{dialect}' not supported for preview_data."
        tbl = _sanitize_table_name(table_name)
        if not tbl:
            return "Invalid table name."
        limit = max(1, min(100, int(limit) if isinstance(limit, (int, float)) else 5))
        try:
            return self._snowflake_preview_table(conn, tbl, limit)
        except Exception as e:
            return str(e)

    # ------------------------------------------------------------------
    # Snowflake implementation
    # ------------------------------------------------------------------

    def _snowflake_list_tables(self, conn: DBConnection) -> List[str]:
        """List tables via Snowflake INFORMATION_SCHEMA."""
        try:
            import snowflake.connector
        except ImportError:
            return []
        try:
            sf_conn = snowflake.connector.connect(
                user=conn.username,
                password=conn.password,
                account=conn.account,
                database=conn.database,
                warehouse=conn.warehouse,
            )
            cursor = sf_conn.cursor()
            cursor.execute("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_catalog = CURRENT_DATABASE()
                  AND table_schema NOT IN ('INFORMATION_SCHEMA')
                  AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name
                LIMIT 500
            """)
            rows = cursor.fetchall()
            sf_conn.close()
            return [f"{r[0]}.{r[1]}" for r in rows]
        except Exception as e:
            self.log(f"[Executor] list_tables error: {e}")
            raise

    def _snowflake_describe_table(self, conn: DBConnection, table_name: str) -> List[Dict[str, str]]:
        """Describe table columns via DESCRIBE TABLE."""
        try:
            import snowflake.connector
        except ImportError:
            return []
        try:
            sf_conn = snowflake.connector.connect(
                user=conn.username,
                password=conn.password,
                account=conn.account,
                database=conn.database,
                warehouse=conn.warehouse,
            )
            cursor = sf_conn.cursor()
            cursor.execute(f"DESCRIBE TABLE {table_name}")
            rows = cursor.fetchall()
            sf_conn.close()
            return [{"name": row[0], "type": row[1]} for row in rows]
        except Exception as e:
            self.log(f"[Executor] get_table_columns error: {e}")
            raise

    def _snowflake_preview_table(self, conn: DBConnection, table_name: str, limit: int) -> str:
        """Preview table data via SELECT * LIMIT n."""
        import json
        try:
            import snowflake.connector
        except ImportError:
            return json.dumps({"error": "snowflake-connector-python not installed"})
        try:
            sf_conn = snowflake.connector.connect(
                user=conn.username,
                password=conn.password,
                account=conn.account,
                database=conn.database,
                warehouse=conn.warehouse,
            )
            cursor = sf_conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            sf_conn.close()
            data = [dict(zip(cols, row)) for row in rows]
            return json.dumps({"columns": cols, "rows": data}, default=str)
        except Exception as e:
            self.log(f"[Executor] preview_data error: {e}")
            raise

    # ------------------------------------------------------------------
    # Snowflake execution
    # ------------------------------------------------------------------

    def _execute_snowflake(self, sql: str, conn: DBConnection, is_compile_only: bool = False) -> ExecutionResult:
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

            # ── Compile or Execute the SQL ────────────────────────────
            t0 = time.time()
            
            try:
                if is_compile_only:
                    self.log(f"[Executor] ── Compiling SQL (Dry Run) ───────────────")
                    # Snowflake 'EXPLAIN' only supports single statements.
                    if ";" in sql.strip(" \n\t;"):
                        self.log("[Executor]   Skipping strict EXPLAIN since query contains multiple statements.")
                        return ExecutionResult(success=True, duration_ms=0, schema=schema_info)
                    cursor.execute(f"EXPLAIN USING JSON {sql}")
                    query_id = cursor.sfqid
                    rows_affected = 0
                    sample = []
                else:
                    self.log(f"[Executor] ── Executing SQL ─────────────────────────────")
                    self.log(f"[Executor]   Statement length: {len(sql)} chars")
                    
                    cursors = list(sf_conn.execute_string(sql))
                    query_id = cursors[0].sfqid if cursors else None
                    rows_affected = sum(c.rowcount or 0 for c in cursors)
                    
                    # Fetch a small sample from the last cursor if applicable
                    sample = []
                    last_cursor = cursors[-1] if cursors else None
                    if last_cursor and last_cursor.description:
                        try:
                            sample = last_cursor.fetchmany(5)
                        except Exception:
                            pass
            except Exception as e:
                err = str(e)
                self.log(f"[Executor] ❌ Execution error: {err}")
                return ExecutionResult(success=False, error=err)

            duration_ms = (time.time() - t0) * 1000
            
            if is_compile_only:
                self.log(f"[Executor] ✅ Compilation successful")
                return ExecutionResult(success=True, duration_ms=duration_ms, schema=schema_info)

            self.log(f"[Executor] ✅ Query complete")
            if query_id:
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

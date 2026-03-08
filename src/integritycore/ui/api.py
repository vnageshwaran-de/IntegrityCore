from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict

import os
from integritycore.agents.loop import LoopAgent
from integritycore.core.verifier import ETLStrategy
from integritycore.adapters.connections import ConnectionManager, DBConnection

app = FastAPI(title="IntegrityCore UI", description="DAG Server for SMT Verification Loop")
conn_manager = ConnectionManager()

# We will maintain the last known run state for the UI to poll
LATEST_RUN = {
    "status": "idle",
    "steps": {
        "generate": {"status": "pending", "output": None},
        "verify": {"status": "pending", "output": None},
        "repair": {"status": "pending", "output": None},
        "execute": {"status": "pending", "output": None}
    },
    "logs": [],
    "final_sql": None,
    "error": None
}

class RunRequest(BaseModel):
    prompt: str
    strategy: str = "INCREMENTAL"
    model: str = "gemini/gemini-pro"
    source: str = "POSTGRESQL"
    target: str = "SNOWFLAKE"

def background_loop_task(request: RunRequest):
    """Executes the loop agent and updates the global LATEST_RUN dict to map to the DAG."""
    global LATEST_RUN
    LATEST_RUN["status"] = "running"
    LATEST_RUN["error"] = None
    LATEST_RUN["execution_result"] = None

    # Reset step statuses and logs
    for step in LATEST_RUN["steps"]:
        LATEST_RUN["steps"][step] = {"status": "pending", "output": None}
    LATEST_RUN["logs"] = []

    def ui_logger(msg: str):
        print(msg)
        LATEST_RUN["logs"].append(str(msg))

    # ── Resolve connections by name ──────────────────────────────────
    all_conns = conn_manager.load_connections()
    conn_by_name = {c.name: c for c in all_conns}
    source_conn = conn_by_name.get(request.source)
    target_conn = conn_by_name.get(request.target)

    if source_conn:
        ui_logger(f"[API] Source connection resolved: {source_conn.name} ({source_conn.dialect})")
    else:
        ui_logger(f"[API] Source '{request.source}' not found in connections — dialect context only")

    if target_conn:
        ui_logger(f"[API] Target connection resolved: {target_conn.name} ({target_conn.dialect})")
    else:
        ui_logger(f"[API] Target '{request.target}' not found in connections — dialect context only")

    agent = LoopAgent(model_name=request.model)
    strategy = ETLStrategy[request.strategy]
    agent.discover_tools()  # MCP tools

    try:
        # Stage 1 + 2 + 3: Generate → Verify → Repair (tracked under generate/verify nodes)
        LATEST_RUN["steps"]["generate"]["status"] = "running"
        LATEST_RUN["steps"]["verify"]["status"] = "running"

        final_sql = agent.execute_etl_loop(
            prompt=request.prompt,
            strategy=strategy,
            source=source_conn.dialect if source_conn else request.source,
            target=target_conn.dialect if target_conn else request.target,
            log_cb=ui_logger,
            source_conn=source_conn,
            target_conn=target_conn,
        )

        LATEST_RUN["steps"]["generate"]["status"] = "success"
        LATEST_RUN["steps"]["verify"]["status"] = "success"

        # Stage 4: Execute (already completed inside execute_etl_loop)
        LATEST_RUN["steps"]["execute"]["status"] = "success"
        LATEST_RUN["final_sql"] = final_sql
        LATEST_RUN["status"] = "success"

    except Exception as e:
        LATEST_RUN["status"] = "failed"
        LATEST_RUN["error"] = str(e)
        ui_logger(f"[API] ❌ Pipeline failed: {e}")

        # Mark the right stage as failed depending on where we died
        if LATEST_RUN["steps"]["verify"]["status"] == "running":
            LATEST_RUN["steps"]["verify"]["status"] = "failed"
            LATEST_RUN["steps"]["repair"]["status"] = "failed"
        elif LATEST_RUN["steps"]["execute"]["status"] == "running":
            LATEST_RUN["steps"]["execute"]["status"] = "failed"


@app.post("/api/verify")
async def trigger_verification(request: RunRequest, background_tasks: BackgroundTasks):
    """Triggers the verification loop in the background and returns immediately."""
    try:
        ETLStrategy[request.strategy]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid strategy.")
        
    background_tasks.add_task(background_loop_task, request)
    return {"message": "Verification loop triggered."}

@app.get("/api/status")
async def get_status():
    """Returns the live DAG status."""
    return LATEST_RUN

@app.get("/api/connections")
async def get_connections():
    """Returns all configured database connections."""
    conns = conn_manager.load_connections()
    return [c.model_dump() for c in conns]

class ConnectionPayload(BaseModel):
    name: Optional[str] = None
    dialect: Optional[str] = None
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

@app.post("/api/connections")
async def create_connection(req: ConnectionPayload):
    """Saves a new database connection."""
    conn = conn_manager.add_connection(**req.model_dump())
    return {"message": "Success", "id": conn.id}

@app.put("/api/connections/{conn_id}")
async def update_connection(conn_id: str, req: ConnectionPayload):
    """Updates an existing connection by ID."""
    updated = conn_manager.update_connection(conn_id, **req.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Connection not found")
    return {"message": "Updated", "connection": updated.model_dump()}

@app.delete("/api/connections/{conn_id}")
async def delete_connection(conn_id: str):
    """Deletes a database connection."""
    success = conn_manager.delete_connection(conn_id)
    if not success:
        raise HTTPException(status_code=404, detail="Connection not found")
    return {"message": "Success"}

class TestConnectionRequest(BaseModel):
    id: str

import asyncio, concurrent.futures

def _real_snowflake_test(conn) -> dict:
    """Runs in a thread pool so we don't block the async event loop."""
    logs = []
    def log(msg):
        logs.append(str(msg))

    log(f"[Test] Dialect   : {conn.dialect}")
    log(f"[Test] Account   : {conn.account}")
    log(f"[Test] User      : {conn.username}")
    log(f"[Test] Database  : {conn.database}")
    log(f"[Test] Warehouse : {conn.warehouse}")
    log(f"[Test] Connecting to Snowflake...")

    try:
        import snowflake.connector
        sf = snowflake.connector.connect(
            user=conn.username,
            password=conn.password,
            account=conn.account,
            database=conn.database,
            warehouse=conn.warehouse,
            login_timeout=15,
        )
        log("[Test] ✅ Authentication successful")

        cur = sf.cursor()
        cur.execute("SELECT CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_VERSION()")
        row = cur.fetchone()
        if row:
            log(f"[Test] Warehouse  : {row[0]}")
            log(f"[Test] Database   : {row[1]}")
            log(f"[Test] Schema     : {row[2]}")
            log(f"[Test] SF Version : {row[3]}")

        cur.execute("SHOW TABLES LIMIT 10")
        tables = cur.fetchall()
        if tables:
            log(f"[Test] Tables found ({len(tables)}):")
            for t in tables:
                log(f"[Test]   • {t[1]}")
        else:
            log("[Test] No tables found in current schema.")

        sf.close()
        return {"status": "success", "message": "Connection successful", "logs": logs}

    except Exception as e:
        log(f"[Test] ❌ Connection failed: {e}")
        return {"status": "error", "message": str(e), "logs": logs}

@app.post("/api/connections/test")
async def test_connection(req: TestConnectionRequest):
    """Performs a real connection test and returns structured logs."""
    conns = conn_manager.load_connections()
    conn = next((c for c in conns if c.id == req.id), None)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")

    if conn.dialect == "SNOWFLAKE":
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, _real_snowflake_test, conn)
        return result

    # Non-Snowflake: basic field validation
    logs = [f"[Test] Dialect: {conn.dialect}"]
    if conn.dialect in ["POSTGRESQL", "MYSQL"]:
        if not conn.host or not conn.username:
            logs.append("[Test] ❌ Missing Host or Username.")
            return {"status": "error", "message": "Missing config", "logs": logs}
    logs.append("[Test] ✅ Configuration looks valid (live driver not yet supported for this dialect).")
    return {"status": "success", "message": "Config valid", "logs": logs}


# ── Snowflake Connection Pool ─────────────────────────────────────────────────
# One persistent connection per conn_id. Reused across all API calls so Duo MFA
# is only triggered the FIRST time a connection is opened (or after disconnect).
import threading
_SF_POOL: dict = {}          # conn_id → snowflake.connector.SnowflakeConnection
_SF_POOL_LOCK = threading.Lock()

def _sf_get_pooled(conn) -> "snowflake.connector.SnowflakeConnection":
    """Return an open, live Snowflake connection from the pool; open one if needed."""
    import snowflake.connector
    conn_id = conn.id
    with _SF_POOL_LOCK:
        sf = _SF_POOL.get(conn_id)
        if sf is not None:
            # Quick liveness check – avoids Duo re-prompt
            try:
                sf.cursor().execute("SELECT 1")
                return sf                          # still alive ✅
            except Exception:
                try: sf.close()
                except Exception: pass
                _SF_POOL.pop(conn_id, None)
        # Open a brand-new connection (Duo prompt happens here, once)
        sf = snowflake.connector.connect(
            user=conn.username,
            password=conn.password,
            account=conn.account,
            database=conn.database or "",
            warehouse=conn.warehouse or "",
            authenticator="snowflake",   # use 'externalbrowser' if SSO-only
            login_timeout=30,
        )
        _SF_POOL[conn_id] = sf
        return sf

def _sf_disconnect(conn_id: str):
    """Evict and close a pooled connection (used by the Disconnect button)."""
    with _SF_POOL_LOCK:
        sf = _SF_POOL.pop(conn_id, None)
    if sf:
        try: sf.close()
        except Exception: pass

# Keep the old name as an alias used by the executor / test endpoints
def _sf_connect(conn):
    return _sf_get_pooled(conn)

def _get_explore_conn(conn_id: str):
    conns = conn_manager.load_connections()
    conn = next((c for c in conns if c.id == conn_id), None)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    if conn.dialect != "SNOWFLAKE":
        raise HTTPException(status_code=400, detail="Explorer currently supports SNOWFLAKE only")
    return conn

@app.get("/api/explore/schemas")
async def explore_schemas(conn_id: str):
    """List all databases/schemas available in the connection."""
    conn = _get_explore_conn(conn_id)
    def _run(conn):
        sf = _sf_connect(conn)
        cur = sf.cursor()
        cur.execute("SHOW DATABASES")
        dbs = [r[1] for r in cur.fetchall()]  # column 1 = name
        result = []
        for db in dbs:
            try:
                cur.execute(f"SHOW SCHEMAS IN DATABASE {db}")
                schemas = [r[1] for r in cur.fetchall()]
                result.append({"database": db, "schemas": schemas})
            except Exception:
                result.append({"database": db, "schemas": []})
        # NOTE: do NOT call sf.close() — connection is kept alive in the pool
        return result
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

@app.get("/api/explore/tables")
async def explore_tables(conn_id: str, database: str = "", schema: str = "PUBLIC"):
    """List all tables and views in a schema."""
    conn = _get_explore_conn(conn_id)
    def _run(conn):
        sf = _sf_connect(conn)
        cur = sf.cursor()
        db = database or conn.database or ""
        cur.execute(f"SHOW TABLES IN SCHEMA {db}.{schema}")
        tables = [{"name": r[1], "kind": "table", "rows": r[4] if len(r) > 4 else None, "created": str(r[7]) if len(r) > 7 else ""} for r in cur.fetchall()]
        cur.execute(f"SHOW VIEWS IN SCHEMA {db}.{schema}")
        views = [{"name": r[1], "kind": "view", "rows": None, "created": str(r[7]) if len(r) > 7 else ""} for r in cur.fetchall()]
        return {"tables": tables + views, "database": db, "schema": schema}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

@app.get("/api/explore/columns")
async def explore_columns(conn_id: str, database: str = "", schema: str = "PUBLIC", table: str = ""):
    """Describe columns for a table."""
    conn = _get_explore_conn(conn_id)
    def _run(conn):
        sf = _sf_connect(conn)
        cur = sf.cursor()
        db = database or conn.database or ""
        cur.execute(f"DESCRIBE TABLE {db}.{schema}.{table}")
        cols = [{"name": r[0], "type": r[1], "nullable": r[3], "default": r[4]} for r in cur.fetchall()]
        return {"columns": cols, "table": table}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

@app.get("/api/explore/preview")
async def explore_preview(conn_id: str, database: str = "", schema: str = "PUBLIC", table: str = "", limit: int = 50):
    """Fetch a sample of rows from a table."""
    conn = _get_explore_conn(conn_id)
    def _run(conn):
        sf = _sf_connect(conn)
        cur = sf.cursor()
        db = database or conn.database or ""
        cur.execute(f"SELECT * FROM {db}.{schema}.{table} LIMIT {min(limit, 200)}")
        cols = [desc[0] for desc in cur.description]
        rows = [list(r) for r in cur.fetchall()]
        # Stringify non-serialisable types
        import datetime
        def _s(v):
            if isinstance(v, (datetime.date, datetime.datetime)): return str(v)
            return v
        rows = [[_s(v) for v in row] for row in rows]
        return {"columns": cols, "rows": rows, "count": len(rows)}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

class QueryRequest(BaseModel):
    conn_id: str
    sql: str

@app.post("/api/explore/query")
async def explore_query(req: QueryRequest):
    """Execute an arbitrary SQL statement and return results."""
    conn = _get_explore_conn(req.conn_id)
    def _run(conn):
        import time, datetime
        sf = _sf_connect(conn)
        cur = sf.cursor()
        t0 = time.time()
        cur.execute(req.sql)
        duration_ms = (time.time() - t0) * 1000
        cols = [desc[0] for desc in (cur.description or [])]
        def _s(v):
            if isinstance(v, (datetime.date, datetime.datetime)): return str(v)
            return v
        rows = [[_s(v) for v in row] for row in (cur.fetchmany(500) or [])]
        affected = cur.rowcount or 0
        return {"columns": cols, "rows": rows, "rows_affected": affected, "duration_ms": round(duration_ms, 1)}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

@app.delete("/api/explore/pool/{conn_id}")
async def disconnect_pool(conn_id: str):
    """Evict and close the pooled Snowflake connection for a given conn_id.
    The next request for this conn_id will open a fresh connection (Duo MFA prompt)."""
    _sf_disconnect(conn_id)
    return {"message": "Connection pool cleared. Next query will re-authenticate."}


# Finally, serve the compiled vite/react frontend over the root URL
# We do this conditionally in case the maintainer is still building the frontend
front_dir = os.path.join(os.path.dirname(__file__), "web", "dist")

if os.path.isdir(front_dir):
    # Mount the 'assets' directory
    app.mount("/assets", StaticFiles(directory=os.path.join(front_dir, "assets")), name="assets")
    
    # Mount the root directory for serving index.html
    from fastapi.responses import HTMLResponse
    @app.get("/{catchall:path}")
    async def serve_react(catchall: str):
        # Serve the index.html for all non-API paths to support SPA routing
        if catchall.startswith("api/"):
             raise HTTPException(status_code=404, detail="API route not found")
        index_path = os.path.join(front_dir, "index.html")
        if os.path.isfile(index_path):
             with open(index_path, "r", encoding="utf-8") as f:
                  content = f.read()
             headers = {
                 "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0",
                 "Pragma": "no-cache",
                 "Expires": "0",
                 "Surrogate-Control": "no-store"
             }
             return HTMLResponse(content=content, headers=headers)
        return {"message": "IntegrityCore API holds. Frontend index not built yet."}

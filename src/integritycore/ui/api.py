"""IntegrityCore FastAPI application — enterprise ETL platform."""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager

import os, asyncio, concurrent.futures, threading, datetime, logging

from integritycore.adapters.connections import ConnectionManager, DBConnection
from integritycore.db.engine import init_db, get_db
from integritycore.db.models import Job, JobRun, JobStatus, RunStatus
from integritycore.metadata.manager import MetadataManager
import integritycore.scheduler.runner as scheduler

log = logging.getLogger("integritycore.api")

# ── Metadata Manager (Enterprise Semantic Catalog) ─────────────────────────────
_metadata_manager: Optional[MetadataManager] = None

def get_metadata_manager() -> Optional[MetadataManager]:
    """FastAPI dependency: MetadataManager for catalog browse and grounding."""
    global _metadata_manager
    if _metadata_manager is None:
        try:
            _metadata_manager = MetadataManager()
        except Exception as e:
            log.warning("MetadataManager not available: %s", e)
    return _metadata_manager

# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    scheduler.start()
    log.info("IntegrityCore API started")
    yield
    # Shutdown
    scheduler.stop()
    log.info("IntegrityCore API stopped")

app = FastAPI(title="IntegrityCore", description="Enterprise LLM-based ETL Platform", lifespan=lifespan)
conn_manager = ConnectionManager()

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class JobCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    source_conn_id: str
    target_conn_id: str
    prompt: str
    schedule_cron: Optional[str] = ""
    schedule_label: Optional[str] = ""
    status: Optional[str] = JobStatus.ACTIVE

class JobUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    source_conn_id: Optional[str] = None
    target_conn_id: Optional[str] = None
    prompt: Optional[str] = None
    schedule_cron: Optional[str] = None
    schedule_label: Optional[str] = None
    status: Optional[str] = None

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

class QueryRequest(BaseModel):
    conn_id: str
    sql: str

class CatalogHarvestRequest(BaseModel):
    """Trigger async harvest for a connection (Semantic Catalog)."""
    conn_id: str
    schema: Optional[str] = None
    run_profiler: Optional[bool] = True

# ── Catalog API (Knowledge Graph / Semantic Metadata) ──────────────────────────

@app.get("/api/catalog/tables")
async def catalog_list_tables(conn_id: str, schema_name: Optional[str] = None):
    """List tables in the semantic catalog (DuckDB). Browse independently of ETL builder."""
    mgr = get_metadata_manager()
    if not mgr:
        raise HTTPException(503, "Metadata catalog not available")
    tables = mgr.list_tables(conn_id, schema_name)
    return {"conn_id": conn_id, "schema_name": schema_name, "tables": tables}

@app.get("/api/catalog/table/{conn_id}/{schema_name}/{table_name}")
async def catalog_get_table(conn_id: str, schema_name: str, table_name: str):
    """Get technical metadata + last_crawled_at for schema versioning."""
    mgr = get_metadata_manager()
    if not mgr:
        raise HTTPException(503, "Metadata catalog not available")
    meta = mgr.get_table_metadata(conn_id, schema_name, table_name)
    if not meta:
        raise HTTPException(404, "Table not in catalog")
    stale = mgr.is_metadata_stale(conn_id, schema_name, table_name, max_age_hours=24)
    return {"table": meta.model_dump(), "stale": stale}

@app.post("/api/catalog/harvest")
async def catalog_harvest(req: CatalogHarvestRequest, background_tasks: BackgroundTasks):
    """
    Trigger asynchronous catalog harvest for a connection.
    Introspection + Semantic Profiler (LLM Metadata Cards) run in background to avoid blocking UI.
    If metadata is > 24h old, consider re-harvesting (schema versioning).
    """
    mgr = get_metadata_manager()
    if not mgr:
        raise HTTPException(503, "Metadata catalog not available")
    connections = conn_manager.load_connections()
    conn = next((c for c in connections if c.id == req.conn_id or c.name == req.conn_id), None)
    if not conn:
        raise HTTPException(400, f"Connection '{req.conn_id}' not found")
    conn_id = getattr(conn, "id", None) or getattr(conn, "name", "")

    def _harvest():
        try:
            n = mgr.harvest_connection(
                conn,
                conn_id,
                schema=req.schema,
                run_profiler=bool(req.run_profiler),
            )
            log.info("Catalog harvest completed for %s: %s tables", conn_id, n)
        except Exception as e:
            log.exception("Catalog harvest failed: %s", e)

    background_tasks.add_task(_harvest)
    return {"message": "Harvest started in background", "conn_id": conn_id}

@app.get("/api/catalog/search")
async def catalog_search(q: str, conn_id: Optional[str] = None, top_k: int = 10):
    """Vector search for tables matching business terms (e.g. Revenue, Active Users)."""
    mgr = get_metadata_manager()
    if not mgr:
        raise HTTPException(503, "Metadata catalog not available")
    hits = mgr.search_by_semantics(q, conn_id=conn_id, top_k=top_k)
    return {"query": q, "hits": [{"conn_id": h[0], "schema_name": h[1], "table_name": h[2], "score": h[3]} for h in hits]}

# ── Jobs API ──────────────────────────────────────────────────────────────────

@app.get("/api/jobs")
async def list_jobs():
    """List all ETL jobs with their last run status."""
    with get_db() as db:
        jobs = db.query(Job).order_by(Job.created_at.desc()).all()
        result = []
        for j in jobs:
            d = j.to_dict()
            d["next_run"] = scheduler.next_run_time(j.id)
            result.append(d)
        return result

@app.post("/api/jobs", status_code=201)
async def create_job(payload: JobCreate):
    """Create a new ETL job and optionally schedule it."""
    import uuid
    with get_db() as db:
        job = Job(
            id=str(uuid.uuid4()),
            name=payload.name,
            description=payload.description or "",
            source_conn_id=payload.source_conn_id,
            target_conn_id=payload.target_conn_id,
            prompt=payload.prompt,
            schedule_cron=payload.schedule_cron or "",
            schedule_label=payload.schedule_label or "",
            status=payload.status or JobStatus.ACTIVE,
        )
        db.add(job)
        db.flush()
        job_id = job.id
        cron = job.schedule_cron
        active = job.status == JobStatus.ACTIVE

    if active and cron:
        scheduler.schedule_job(job_id, cron)

    with get_db() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        d = job.to_dict()
        d["next_run"] = scheduler.next_run_time(job_id)
        return d

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    with get_db() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        d = job.to_dict()
        d["next_run"] = scheduler.next_run_time(job_id)
        return d

@app.put("/api/jobs/{job_id}")
async def update_job(job_id: str, payload: JobUpdate):
    with get_db() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        for field, val in payload.dict(exclude_none=True).items():
            setattr(job, field, val)
        job.updated_at = datetime.datetime.utcnow()
        cron = job.schedule_cron
        active = job.status == JobStatus.ACTIVE

    # Re-sync scheduler
    if active and cron:
        scheduler.schedule_job(job_id, cron)
    else:
        scheduler.unschedule_job(job_id)

    with get_db() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        d = job.to_dict()
        d["next_run"] = scheduler.next_run_time(job_id)
        return d

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    scheduler.unschedule_job(job_id)
    with get_db() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        # Explicit delete-orphan cascade for SQLite since FK enforcement is off by default
        db.query(JobRun).filter(JobRun.job_id == job_id).delete()
        db.delete(job)
    return {"message": "Job deleted"}

@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    scheduler.unschedule_job(job_id)
    with get_db() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        job.status = JobStatus.PAUSED
    return {"id": job_id, "status": "paused"}

@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    with get_db() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        job.status = JobStatus.ACTIVE
        cron = job.schedule_cron
    if cron:
        scheduler.schedule_job(job_id, cron)
    return {"id": job_id, "status": "active"}

# ── Interactive Dry Run ───────────────────────────────────────────────────────

class DryRunRequest(BaseModel):
    source_conn_id: str
    target_conn_id: str
    prompt: str
    user_feedback: Optional[str] = None     # feedback from previous attempt
    previous_sql: Optional[str] = None      # SQL from previous attempt to improve

@app.post("/api/jobs/dry-run")
async def dry_run(req: DryRunRequest):
    """Run LLM generate → SMT verify → self-heal loop WITHOUT executing.
    Returns generated SQL, verification status, and logs so the user can
    iterate on the prompt before committing a real job."""
    conn_mgr = ConnectionManager()
    connections = conn_mgr.load_connections()
    src = next((c for c in connections if c.id == req.source_conn_id or c.name == req.source_conn_id), None)
    tgt = next((c for c in connections if c.id == req.target_conn_id or c.name == req.target_conn_id), None)
    if not src:
        raise HTTPException(400, f"Source connection '{req.source_conn_id}' not found")
    if not tgt:
        raise HTTPException(400, f"Target connection '{req.target_conn_id}' not found")

    def _run():
        import datetime as _dt
        from integritycore.core.verifier import LogicVerifier, ETLStrategy
        import litellm

        log_lines = []
        def _log(msg):
            ts = _dt.datetime.utcnow().strftime("%H:%M:%S")
            log_lines.append(f"[{ts}] {msg}")

        model = os.getenv("LITELLM_MODEL", "gemini/gemini-2.5-flash")

        from integritycore.core.verifier import LogicVerifier, ETLStrategy
        from integritycore.adapters.executor import DatabaseExecutor
        from integritycore.agents.graph import build_etl_graph
        
        strategy = ETLStrategy.INCREMENTAL if "timestamp" in req.prompt.lower() or "updated_at" in req.prompt.lower() else ETLStrategy.FULL_REFRESH
        
        _log(f"Source: {src.name} ({src.dialect}) | Target: {tgt.name} ({tgt.dialect})")
        _log("⏳ Starting LangGraph ETL Agent...")

        state_input = {
            "source_dialect": src.dialect,
            "target_dialect": tgt.dialect,
            "prompt": req.prompt,
            "strategy": strategy,
            "model_name": model,
            "messages": [],
            "sql": "",
            "verified": False,
            "verification_details": "",
            "is_valid_prompt": True,
            "validation_error": "",
            "validation_result": None,
            "metadata_manager": get_metadata_manager(),
            "grounding_result": None,
            "grounded_ddl": "",
            "semantic_mappings": None,
            "special_action": "",
            "repair_attempts": 0,
            "max_repairs": 3,
            "is_dry_run": True,
            "source_conn": src,
            "target_conn": tgt,
            "executor": DatabaseExecutor(log_cb=_log),
            "logs": [],
        }
        
        if req.user_feedback and req.previous_sql:
            state_input["messages"] = [
                {"role": "system", "content": "You are an expert ETL SQL generator. Return ONLY valid SQL wrapped in a ```sql code block."},
                {"role": "user", "content": (
                    f"You are building an ETL pipeline extracting data from {src.dialect} and loading it into {tgt.dialect}.\n\n"
                    f"User objective: {req.prompt}"
                )},
                {"role": "assistant", "content": req.previous_sql},
                {"role": "user", "content": f"User feedback: {req.user_feedback}\nPlease fix the SQL based on the user feedback. Return ONLY valid SQL."}
            ]

        graph = build_etl_graph()
        final_state = graph.invoke(state_input)
        
        log_lines.extend(final_state.get("logs", []))
        
        if final_state.get("is_valid_prompt", True) is False:
            _log(f"🚫 Prompt validation failed: {final_state.get('validation_error')}")
            return {
                "status": "validation_failed",
                "phase": "intercept",
                "sql": None,
                "verified": False,
                "verification_details": "",
                "repair_attempts": 0,
                "logs": log_lines,
                "error": final_state.get("validation_error", "Vague prompt. Please clarify.")
            }
        
        status = "verified" if final_state["verified"] else ("repaired" if final_state["repair_attempts"] > 0 else "unverified")
        
        if final_state.get("special_action") == "missing_schema":
            status = "missing_schema"
            _log("⚠️ Missing target table detected. Pausing for user feedback...")
            return {
                "status": status,
                "phase": "intercept",
                "sql": final_state["sql"],
                "verified": False,
                "verification_details": final_state["verification_details"],
                "repair_attempts": final_state["repair_attempts"],
                "logs": log_lines,
                "error": "The target table does not exist. Do you want the LLM to write the CREATE TABLE statement for you based on the source data?",
            }
            
        _log(f"{'✅' if final_state['verified'] else '⚠️'} Dry run complete — status: {status}")
        
        return {
            "status": status,
            "phase": "complete",
            "sql": final_state.get("sql", ""),
            "verified": final_state.get("verified", False),
            "verification_details": final_state.get("verification_details", ""),
            "repair_attempts": final_state.get("repair_attempts", 0),
            "logs": log_lines,
            "error": "" if final_state.get("verified") else final_state.get("verification_details", ""),
        }

        return {
            "status": status,
            "phase": "complete",
            "sql": sql,
            "verified": verified,
            "verification_details": verification_details,
            "repair_attempts": repair_attempts,
            "logs": log_lines,
            "error": "" if verified else verification_details,
        }

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run)


@app.post("/api/jobs/{job_id}/run")
async def trigger_run(job_id: str, background_tasks: BackgroundTasks):
    """Trigger an immediate manual run in the background."""
    import uuid
    with get_db() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(404, "Job not found")
        run = JobRun(
            id=str(uuid.uuid4()),
            job_id=job_id,
            status=RunStatus.RUNNING,
            triggered_by="manual",
            started_at=datetime.datetime.utcnow(),
        )
        db.add(run)
        db.commit()
        run_id = run.id

    def _run_wrapper(j_id, r_id, by):
        print(f"DEBUG: Entering _run_wrapper for job {j_id} run {r_id}")
        import asyncio
        import traceback
        try:
            # Set up an event loop for this thread if one doesn't exist.
            # Litellm/HTTPX relies on having an active event loop.
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            scheduler.execute_run(j_id, r_id, by)
            print(f"DEBUG: Exiting _run_wrapper for job {j_id} run {r_id}")
        except Exception as e:
            print(f"CRITICAL ERROR in background execute_run: {e}")
            traceback.print_exc()

    background_tasks.add_task(_run_wrapper, job_id, run_id, "manual")
    return {"run_id": run_id, "status": "running"}

@app.get("/api/jobs/{job_id}/runs")
async def list_runs(job_id: str, limit: int = 50):
    with get_db() as db:
        runs = (
            db.query(JobRun)
            .filter(JobRun.job_id == job_id)
            .order_by(JobRun.started_at.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in runs]

@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    with get_db() as db:
        run = db.query(JobRun).filter(JobRun.id == run_id).first()
        if not run:
            raise HTTPException(404, "Run not found")
        return run.to_dict()

@app.get("/api/runs/{run_id}/logs")
async def stream_logs(run_id: str):
    """SSE stream of logs for a live run."""
    async def _gen():
        import time
        seen = 0
        for _ in range(600):  # max 10 min
            with get_db() as db:
                run = db.query(JobRun).filter(JobRun.id == run_id).first()
                if not run:
                    break
                lines = (run.logs or "").split("\n")
                new_lines = lines[seen:]
                for line in new_lines:
                    if line.strip():
                        yield f"data: {line}\n\n"
                seen = len(lines)
                if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
                    yield "data: [DONE]\n\n"
                    break
            await asyncio.sleep(1)

    return StreamingResponse(_gen(), media_type="text/event-stream")

# ── Dashboard stats ───────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    """Aggregate stats for the dashboard header."""
    from sqlalchemy import func
    today = datetime.datetime.utcnow().date()
    with get_db() as db:
        total_jobs = db.query(func.count(Job.id)).scalar() or 0
        active_jobs = db.query(func.count(Job.id)).filter(Job.status == JobStatus.ACTIVE).scalar() or 0
        runs_today = db.query(func.count(JobRun.id)).filter(
            func.date(JobRun.started_at) == today
        ).scalar() or 0
        success_today = db.query(func.count(JobRun.id)).filter(
            func.date(JobRun.started_at) == today,
            JobRun.status == RunStatus.SUCCESS,
        ).scalar() or 0
        recent_runs = (
            db.query(JobRun)
            .order_by(JobRun.started_at.desc())
            .limit(20)
            .all()
        )
        return {
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "runs_today": runs_today,
            "success_rate": round(success_today / runs_today * 100, 1) if runs_today else 0,
            "recent_runs": [r.to_dict() for r in recent_runs],
        }

# ── Connections (existing, kept compatible) ───────────────────────────────────

@app.get("/api/connections")
async def list_connections():
    return [c.__dict__ for c in conn_manager.load_connections()]

@app.post("/api/connections")
async def create_connection(payload: ConnectionPayload):
    conn = DBConnection(
        name=payload.name, dialect=payload.dialect or "SNOWFLAKE",
        host=payload.host, port=payload.port, database=payload.database,
        username=payload.username, password=payload.password,
        account=payload.account, warehouse=payload.warehouse,
        project_id=payload.project_id, dataset_id=payload.dataset_id,
        service_account_json=payload.service_account_json,
    )
    conn_manager.add_connection(conn)
    return conn.__dict__

@app.put("/api/connections/{conn_id}")
async def update_connection(conn_id: str, payload: ConnectionPayload):
    updates = payload.dict(exclude_none=True)
    conn_manager.update_connection(conn_id, updates)
    return {"message": "Connection updated"}

@app.delete("/api/connections/{conn_id}")
async def delete_connection(conn_id: str):
    conn_manager.delete_connection(conn_id)
    return {"message": "Deleted"}

@app.post("/api/connections/test")
async def test_connection(payload: ConnectionPayload):
    conns = conn_manager.load_connections()
    conn = next((c for c in conns if c.name == payload.name or c.id == payload.name), None)
    if not conn:
        return {"status": "error", "logs": ["Connection not found"]}
    if conn.dialect == "SNOWFLAKE":
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, _real_snowflake_test, conn)
        return result
    return {"status": "success", "logs": ["Config valid (live test not available for this dialect)"]}

def _real_snowflake_test(conn):
    logs = [f"[Test] Connecting to Snowflake account: {conn.account}"]
    try:
        import snowflake.connector
        sf = snowflake.connector.connect(
            user=conn.username, password=conn.password, account=conn.account,
            database=conn.database or "", warehouse=conn.warehouse or "",
            login_timeout=15,
        )
        cur = sf.cursor()
        cur.execute("SELECT CURRENT_USER(), CURRENT_WAREHOUSE(), CURRENT_DATABASE()")
        row = cur.fetchone()
        logs.append(f"[Test] ✅ Connected as {row[0]} | WH: {row[1]} | DB: {row[2]}")
        sf.close()
        return {"status": "success", "logs": logs}
    except Exception as e:
        logs.append(f"[Test] ❌ {e}")
        return {"status": "error", "logs": logs}


# ── Database Explorer (existing endpoints kept intact) ────────────────────────

import threading
_SF_POOL: dict = {}
_SF_POOL_LOCK = threading.Lock()

def _sf_get_pooled(conn):
    import snowflake.connector
    conn_id = conn.id
    with _SF_POOL_LOCK:
        sf = _SF_POOL.get(conn_id)
        if sf is not None:
            try:
                sf.cursor().execute("SELECT 1")
                return sf
            except Exception:
                try: sf.close()
                except Exception: pass
                _SF_POOL.pop(conn_id, None)
        sf = snowflake.connector.connect(
            user=conn.username, password=conn.password, account=conn.account,
            database=conn.database or "", warehouse=conn.warehouse or "",
            login_timeout=30,
        )
        _SF_POOL[conn_id] = sf
        return sf

def _sf_disconnect(conn_id: str):
    with _SF_POOL_LOCK:
        sf = _SF_POOL.pop(conn_id, None)
    if sf:
        try: sf.close()
        except Exception: pass

def _sf_connect(conn):
    return _sf_get_pooled(conn)

def _get_explore_conn(conn_id: str):
    conns = conn_manager.load_connections()
    conn = next((c for c in conns if c.id == conn_id), None)
    if not conn:
        raise HTTPException(404, "Connection not found")
    if conn.dialect != "SNOWFLAKE":
        raise HTTPException(400, "Explorer currently supports SNOWFLAKE only")
    return conn

@app.get("/api/explore/schemas")
async def explore_schemas(conn_id: str):
    conn = _get_explore_conn(conn_id)
    def _run(conn):
        sf = _sf_connect(conn)
        cur = sf.cursor()
        cur.execute("SHOW DATABASES")
        dbs = [r[1] for r in cur.fetchall()]
        result = []
        for db in dbs:
            try:
                cur.execute(f"SHOW SCHEMAS IN DATABASE {db}")
                schemas = [r[1] for r in cur.fetchall()]
                result.append({"database": db, "schemas": schemas})
            except Exception:
                result.append({"database": db, "schemas": []})
        return result
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

@app.get("/api/explore/tables")
async def explore_tables(conn_id: str, database: str = "", schema: str = "PUBLIC"):
    conn = _get_explore_conn(conn_id)
    def _run(conn):
        sf = _sf_connect(conn)
        cur = sf.cursor()
        db = database or conn.database or ""
        cur.execute(f"SHOW TABLES IN SCHEMA {db}.{schema}")
        tables = [{"name": r[1], "kind": "table", "rows": r[4] if len(r) > 4 else None} for r in cur.fetchall()]
        cur.execute(f"SHOW VIEWS IN SCHEMA {db}.{schema}")
        views = [{"name": r[1], "kind": "view", "rows": None} for r in cur.fetchall()]
        return {"tables": tables + views, "database": db, "schema": schema}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

@app.get("/api/explore/columns")
async def explore_columns(conn_id: str, database: str = "", schema: str = "PUBLIC", table: str = ""):
    conn = _get_explore_conn(conn_id)
    def _run(conn):
        sf = _sf_connect(conn)
        cur = sf.cursor()
        db = database or conn.database or ""
        cur.execute(f"DESCRIBE TABLE {db}.{schema}.{table}")
        return {"columns": [{"name": r[0], "type": r[1], "nullable": r[3], "default": r[4]} for r in cur.fetchall()], "table": table}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

@app.get("/api/explore/preview")
async def explore_preview(conn_id: str, database: str = "", schema: str = "PUBLIC", table: str = "", limit: int = 50):
    conn = _get_explore_conn(conn_id)
    def _run(conn):
        import datetime as _dt
        sf = _sf_connect(conn)
        cur = sf.cursor()
        db = database or conn.database or ""
        cur.execute(f"SELECT * FROM {db}.{schema}.{table} LIMIT {min(limit, 200)}")
        cols = [d[0] for d in cur.description]
        def _s(v): return str(v) if isinstance(v, (_dt.date, _dt.datetime)) else v
        rows = [[_s(v) for v in r] for r in cur.fetchall()]
        return {"columns": cols, "rows": rows, "count": len(rows)}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

@app.post("/api/explore/query")
async def explore_query(req: QueryRequest):
    conn = _get_explore_conn(req.conn_id)
    def _run(conn):
        import time, datetime as _dt
        sf = _sf_connect(conn)
        cur = sf.cursor()
        t0 = time.time()
        cur.execute(req.sql)
        ms = round((time.time() - t0) * 1000, 1)
        cols = [d[0] for d in (cur.description or [])]
        def _s(v): return str(v) if isinstance(v, (_dt.date, _dt.datetime)) else v
        rows = [[_s(v) for v in r] for r in (cur.fetchmany(500) or [])]
        return {"columns": cols, "rows": rows, "rows_affected": cur.rowcount or 0, "duration_ms": ms}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, _run, conn)

@app.delete("/api/explore/pool/{conn_id}")
async def disconnect_pool(conn_id: str):
    _sf_disconnect(conn_id)
    return {"message": "Pool cleared"}


# ── Serve frontend ────────────────────────────────────────────────────────────

front_dir = os.path.join(os.path.dirname(__file__), "web", "dist")
if os.path.isdir(front_dir):
    if os.path.isdir(os.path.join(front_dir, "assets")):
        app.mount("/assets", StaticFiles(directory=os.path.join(front_dir, "assets")), name="assets")

    from fastapi.responses import HTMLResponse
    @app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        with open(os.path.join(front_dir, "index.html")) as f:
            return HTMLResponse(f.read())

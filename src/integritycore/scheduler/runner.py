"""APScheduler-based job scheduler for IntegrityCore.

Loads all active Job records from the DB on startup and registers their
cron triggers with APScheduler's BackgroundScheduler. Each trigger
creates a JobRun row and executes the ETL loop.
"""
import datetime
import logging
import threading
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from integritycore.core.verifier import ETLStrategy

log = logging.getLogger("integritycore.scheduler")

# Module-level singleton
_scheduler: Optional[BackgroundScheduler] = None
_lock = threading.Lock()


def get_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        with _lock:
            if _scheduler is None:
                _scheduler = BackgroundScheduler(timezone="UTC")
    return _scheduler


def start():
    """Start the scheduler (called at FastAPI startup)."""
    sched = get_scheduler()
    if not sched.running:
        sched.start()
        log.info("APScheduler started")
    _reload_all_jobs()


def stop():
    """Stop the scheduler (called at FastAPI shutdown)."""
    sched = get_scheduler()
    if sched.running:
        sched.shutdown(wait=False)
        log.info("APScheduler stopped")


def _reload_all_jobs():
    """Load all active scheduled jobs from the DB and register them."""
    from integritycore.db.engine import get_db
    from integritycore.db.models import Job, JobStatus

    with get_db() as db:
        jobs = db.query(Job).filter(
            Job.status == JobStatus.ACTIVE,
            Job.schedule_cron != "",
            Job.schedule_cron.isnot(None),
        ).all()
        for job in jobs:
            _register_job(job.id, job.schedule_cron)
    log.info(f"Loaded {len(jobs)} scheduled jobs")


def _register_job(job_id: str, cron_expr: str):
    """Add or replace an APScheduler job for the given job_id."""
    sched = get_scheduler()
    aps_id = f"job_{job_id}"
    # Remove existing trigger if present
    if sched.get_job(aps_id):
        sched.remove_job(aps_id)
    try:
        trigger = CronTrigger.from_crontab(cron_expr, timezone="UTC")
        sched.add_job(
            func=_run_etl_job,
            trigger=trigger,
            id=aps_id,
            args=[job_id],
            replace_existing=True,
            misfire_grace_time=300,  # 5 min grace for missed runs
        )
        log.info(f"Registered schedule for job {job_id}: {cron_expr}")
    except Exception as e:
        log.error(f"Failed to register schedule for job {job_id}: {e}")


def _unregister_job(job_id: str):
    """Remove the APScheduler trigger for a job (pause/delete)."""
    sched = get_scheduler()
    aps_id = f"job_{job_id}"
    if sched.get_job(aps_id):
        sched.remove_job(aps_id)
        log.info(f"Unregistered schedule for job {job_id}")


def schedule_job(job_id: str, cron_expr: str):
    """Public API: activate scheduling for a job."""
    if cron_expr:
        _register_job(job_id, cron_expr)


def unschedule_job(job_id: str):
    """Public API: deactivate scheduling for a job."""
    _unregister_job(job_id)


def next_run_time(job_id: str) -> Optional[str]:
    """Return ISO string of next scheduled run, or None."""
    sched = get_scheduler()
    aps_job = sched.get_job(f"job_{job_id}")
    if aps_job and aps_job.next_run_time:
        return aps_job.next_run_time.isoformat()
    return None


def _run_etl_job(job_id: str):
    """Execute one ETL run for the given job_id — called by APScheduler."""
    from integritycore.db.engine import get_db
    from integritycore.db.models import Job, JobRun, RunStatus
    from integritycore.adapters.connections import ConnectionManager
    import time

    log.info(f"Scheduled trigger fired for job {job_id}")

    conn_mgr = ConnectionManager()

    with get_db() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            log.error(f"Job {job_id} not found")
            return

        run = JobRun(
            job_id=job_id,
            status=RunStatus.RUNNING,
            triggered_by="schedule",
            started_at=datetime.datetime.utcnow(),
        )
        db.add(run)
        db.flush()
        run_id = run.id

    # Execute outside the DB session to avoid long-held locks
    execute_run(job_id, run_id, triggered_by="schedule")


def execute_run(job_id: str, run_id: str, triggered_by: str = "manual"):
    """Core execution: run the ETL loop and persist results in JobRun."""
    import datetime
    import time
    from integritycore.db.engine import get_db
    from integritycore.db.models import Job, JobRun, RunStatus
    from integritycore.adapters.connections import ConnectionManager

    log_lines = []

    def _log(msg: str):
        ts = datetime.datetime.utcnow().strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")
        log.info(msg)

    t0 = time.time()
    status = RunStatus.SUCCESS
    sql_out = ""
    rows = 0
    error = ""
    verified = False

    try:
        print(f"DEBUG(execute_run): Entering try block for {run_id}")
        with get_db() as db:
            print(f"DEBUG(execute_run): Opened DB session for {run_id}")
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                print(f"DEBUG(execute_run): Job {job_id} not found!")
                raise ValueError(f"Job {job_id} not found")
            prompt = job.prompt
            src_id = job.source_conn_id
            tgt_id = job.target_conn_id

        print(f"DEBUG(execute_run): Fetched job details. src: {src_id}, tgt: {tgt_id}")
        _log(f"Starting job '{job_id}' | triggered_by={triggered_by}")
        _log(f"Prompt: {prompt}")

        conn_mgr = ConnectionManager()
        print(f"DEBUG(execute_run): ConnectionManager loaded")
        connections = conn_mgr.load_connections()
        print(f"DEBUG(execute_run): Loaded connections list")
        src = next((c for c in connections if c.id == src_id or c.name == src_id), None)
        tgt = next((c for c in connections if c.id == tgt_id or c.name == tgt_id), None)

        if not src:
            print(f"DEBUG(execute_run): Source missing")
            raise ValueError(f"Source connection '{src_id}' not found")
        if not tgt:
            print(f"DEBUG(execute_run): Target missing")
            raise ValueError(f"Target connection '{tgt_id}' not found")

        print(f"DEBUG(execute_run): Target: {tgt.name} ({tgt.dialect})")
        print(f"DEBUG(execute_run): Spawning Process for LoopAgent")

        # ── Isolate LoopAgent instantiation in a completely isolated subprocess ──
        # Z3 (C++ extension) crashes silently when initialized inside ASGI threadpools 
        # or ProcessPools derived from them. We use a raw subprocess.
        import tempfile
        import subprocess
        import json
        import os
        import sys
        script_content = f"""
import sys
import json
import os
sys.path.append('{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))}')

def main():
    try:
        data = json.loads(sys.argv[1])
        from integritycore.adapters.connections import ConnectionManager
        from integritycore.adapters.executor import DatabaseExecutor
        from integritycore.agents.graph import build_etl_graph
        from integritycore.core.verifier import ETLStrategy
        
        mgr = ConnectionManager()
        conns = mgr.load_connections()
        src = next((c for c in conns if c.id == data['src_id'] or c.name == data['src_id']), None)
        tgt = next((c for c in conns if c.id == data['tgt_id'] or c.name == data['tgt_id']), None)
        
        strat = ETLStrategy(data['strategy'])
        model = os.getenv("LITELLM_MODEL", "gemini/gemini-2.5-flash")
        
        executor = DatabaseExecutor()
        graph = build_etl_graph()
        
        state_input = {{
            "source_dialect": data['source'],
            "target_dialect": data['target'],
            "prompt": data['prompt'],
            "strategy": strat,
            "model_name": model,
            "messages": [],
            "sql": "",
            "verified": False,
            "verification_details": "",
            "is_valid_prompt": True,
            "validation_error": "",
            "validation_result": None,
            "metadata_manager": None,
            "grounding_result": None,
            "grounded_ddl": "",
            "semantic_mappings": None,
            "special_action": "",
            "repair_attempts": 0,
            "max_repairs": 3,
            "is_dry_run": False,
            "source_conn": src,
            "target_conn": tgt,
            "executor": executor,
            "logs": []
        }}
        
        final_state = graph.invoke(state_input)
        
        res = final_state.get("execution_result")
        
        if res and res.success:
            print("SUCCESS_RUN===" + json.dumps({{"sql": final_state["sql"], "rows": res.rows_affected, "logs": final_state["logs"]}}))
        else:
            err = res.error if res else final_state.get("verification_details", "Unknown failure")
            print("FAILED_RUN===" + json.dumps({{"error": getattr(err, "message", str(err)), "sql": final_state.get("sql", ""), "logs": final_state.get("logs", [])}}))
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script_content)
            script_path = tf.name

        try:
            strategy = ETLStrategy.INCREMENTAL if "timestamp" in prompt.lower() or "updated_at" in prompt.lower() else ETLStrategy.FULL_REFRESH

            payload = json.dumps({
                "prompt": prompt,
                "strategy": strategy.value if hasattr(strategy, 'value') else strategy,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source": src.dialect,
                "target": tgt.dialect
            })
            
            proc = subprocess.run(
                [sys.executable, script_path, payload],
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )
            
            if proc.returncode != 0:
                raise RuntimeError(f"LangGraph subprocess failed: {proc.stderr}")
                
            out = proc.stdout
            if "SUCCESS_RUN===" in out:
                payload_str = out.split("SUCCESS_RUN===")[1].strip()
                data = json.loads(payload_str)
                sql_out = data["sql"]
                rows = data["rows"]
                log_lines.extend(data["logs"])
                verified = True
                status = RunStatus.SUCCESS
                _log(f"✅ Job completed successfully.")
            elif "FAILED_RUN===" in out:
                payload_str = out.split("FAILED_RUN===")[1].strip()
                data = json.loads(payload_str)
                sql_out = data.get("sql", "")
                error = data.get("error", "Unknown failure")
                log_lines.extend(data.get("logs", []))
                status = RunStatus.FAILED
                _log(f"❌ Job execution failed: {error}")
            else:
                raise RuntimeError(f"Unexpected output from graph subprocess: {out}")
                
        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)

    except Exception as exc:
        status = RunStatus.FAILED
        error = str(exc)
        _log(f"💥 Exception: {exc}")
        print(f"DEBUG(execute_run): Caught exception: {exc}")
        import traceback
        traceback.print_exc()

    finally:
        duration = round(time.time() - t0, 2)
        _log(f"Duration: {duration}s")

        with get_db() as db:
            run = db.query(JobRun).filter(JobRun.id == run_id).first()
            if run:
                run.status = status
                run.finished_at = datetime.datetime.utcnow()
                run.duration_seconds = duration
                run.logs = "\n".join(log_lines)
                run.error_msg = error
                run.generated_sql = sql_out
                run.rows_processed = rows
                run.verified = verified

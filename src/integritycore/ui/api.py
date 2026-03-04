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
        "repair": {"status": "pending", "output": None}
    },
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
    
    # Reset step statuses
    for step in LATEST_RUN["steps"]:
        LATEST_RUN["steps"][step] = {"status": "pending", "output": None}
        
    agent = LoopAgent(model_name=request.model)
    strategy = ETLStrategy[request.strategy]
    agent.discover_tools() # MCP tools
    
    try:
        # Step 1: Generate
        LATEST_RUN["steps"]["generate"]["status"] = "running"
        # We manually step through the generator or simulate the step splits
        LATEST_RUN["steps"]["generate"]["status"] = "success"
        
        # In a real async agent, we would yield here. For this FastAPI wrapping:
        LATEST_RUN["steps"]["verify"]["status"] = "running"
        
        # Execute the monolithic loop
        final_sql = agent.execute_etl_loop(prompt=request.prompt, strategy=strategy, source=request.source, target=request.target)
        
        LATEST_RUN["steps"]["verify"]["status"] = "success"
        LATEST_RUN["final_sql"] = final_sql
        LATEST_RUN["status"] = "success"
        
    except Exception as e:
        LATEST_RUN["status"] = "failed"
        LATEST_RUN["error"] = str(e)
        
        # We assume verify failed and trigger repair
        LATEST_RUN["steps"]["verify"]["status"] = "failed"
        LATEST_RUN["steps"]["repair"]["status"] = "failed" # If the whole loop throws


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
    return [{"id": c.id, "name": c.name, "dialect": c.dialect, "host": c.host} for c in conns]

class CreateConnectionRequest(BaseModel):
    name: str
    dialect: str
    host: str
    username: str
    password: str

@app.post("/api/connections")
async def create_connection(req: CreateConnectionRequest):
    """Saves a new database connection."""
    conn = conn_manager.add_connection(req.name, req.dialect, req.host, req.username, req.password)
    return {"message": "Success", "id": conn.id}

# Finally, serve the compiled vite/react frontend over the root URL
# We do this conditionally in case the maintainer is still building the frontend
front_dir = os.path.join(os.path.dirname(__file__), "web", "dist")

if os.path.isdir(front_dir):
    # Mount the 'assets' directory
    app.mount("/assets", StaticFiles(directory=os.path.join(front_dir, "assets")), name="assets")
    
    # Mount the root directory for serving index.html
    from fastapi.responses import FileResponse
    @app.get("/{catchall:path}")
    async def serve_react(catchall: str):
        # Serve the index.html for all non-API paths to support SPA routing
        if catchall.startswith("api/"):
             raise HTTPException(status_code=404, detail="API route not found")
        index_path = os.path.join(front_dir, "index.html")
        if os.path.isfile(index_path):
             return FileResponse(index_path)
        return {"message": "IntegrityCore API holds. Frontend index not built yet."}

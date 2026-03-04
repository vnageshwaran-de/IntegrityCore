import argparse
import sys
import os
import subprocess
from integritycore.agents.loop import LoopAgent
from integritycore.core.verifier import ETLStrategy

def main():
    """
    Entry point for the backend agentic loop.
    Command: `integrity-core`
    """
    parser = argparse.ArgumentParser(description="IntegrityCore Backend Agentic Loop")
    parser.add_argument("--prompt", type=str, required=True, help="The ETL requirement prompt for the agent.")
    parser.add_argument("--strategy", type=str, choices=["FULL_REFRESH", "INCREMENTAL"], default="INCREMENTAL",
                        help="The ETL verification strategy to enforce.")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-pro", help="The LLM model to use via LiteLLM/ADK")
    
    args = parser.parse_args()
    
    strategy = ETLStrategy[args.strategy]
    print(f"Starting IntegrityCore Agent Loop with strategy: {strategy.name}")
    print(f"Model Engine: {args.model}")
    print(f"Prompt: {args.prompt}\n")
    
    agent = LoopAgent(model_name=args.model)
    
    try:
        final_sql = agent.execute_etl_loop(prompt=args.prompt, strategy=strategy)
        print("\n\n=== Final Verified SQL ===")
        print(final_sql)
    except Exception as e:
        print(f"\n[Error] Failed to generate verified SQL: {e}")
        sys.exit(1)

def run_ui():
    """
    Wrapper command that executes Uvicorn to host the DAG FastAPI backend.
    Command: `integrity-ui`
    """
    print("Launching IntegrityCore UI via Uvicorn...")
    
    # Run uvicorn Programmatically
    import uvicorn
    uvicorn.run("integritycore.ui.api:app", host="0.0.0.0", port=8501, reload=True)

if __name__ == "__main__":
    main()

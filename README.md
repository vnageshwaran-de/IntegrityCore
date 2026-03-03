# IntegrityCore

A cloud-agnostic, pip-installable agentic framework for mathematical ETL verification.

## Architecture

To support both a CLI (for backend tasks) and the Streamlit UI, we rely on the following structure:
- `core/`: Z3 SMT & AST Logic
- `adapters/`: Cloud Adapters (GCP, etc.)
- `agents/`: ADK Agent Definitions
- `ui/`: Streamlit App Code
- `cli.py`: Command-Line Entry Point

## Features
- **Core Verification Logic**: Relies on Z3 and AST to guarantee the correctness of model-agnostic LLM inputs.
- **Multi-Agent Orchestration**: Generate -> Verify -> Repair cycle, using MCP for connecting tools.
- **Glass Box Verification**: Streamlit UI that allows users to examine internal proofs dynamically.

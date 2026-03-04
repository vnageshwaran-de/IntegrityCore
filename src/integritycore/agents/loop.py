from typing import List, Dict, Any
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

    def execute_etl_loop(self, prompt: str, strategy: ETLStrategy, source: str = "POSTGRESQL", target: str = "SNOWFLAKE", max_retries: int = 3) -> str:
        """
        The core Multi-Agent orchestration cycle: Generate -> Verify -> Repair
        Includes Source and Target connection mapping to ensure accurate dialects.
        """
        
        # Inject the connection dialect context directly into the prompt
        enhanced_prompt = f"You are building an ETL pipeline extracting data from the {source} dialect and loading it into the {target} dialect. Ensure the syntax is compatible.\n\n{prompt}"
        
        messages = [
            {
                "role": "system", 
                "content": "You are an expert AI agent generating ETL SQL. You have access to database tools via MCP."
            },
            {"role": "user", "content": enhanced_prompt}
        ]
        
        # Step 0: Ensure tools are discovered via MCP
        if not self.tools:
            self.discover_tools()
            
        for attempt in range(max_retries + 1):
            print(f"--- Generate -> Verify -> Repair Cycle (Attempt {attempt + 1}) ---")
            
            # STEP 1: Generate
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                tools=self.tools
            )
            
            msg = response.choices[0].message
            content = msg.content or ""
            # Maintain conversation history for the agent
            messages.append({"role": "assistant", "content": content})
            
            # Extract SQL if it is wrapped in markdown code blocks
            sql_code = content
            if "```sql" in content:
                sql_code = content.split("```sql")[1].split("```")[0].strip()
            elif "```" in content:
                sql_code = content.split("```")[1].split("```")[0].strip()
                
            if not sql_code.strip():
                sql_code = content.strip() # Keep raw fallback
                
            # STEP 2: Verify
            print("Verifying generated logic using SMT logic (z3) and AST...")
            is_valid = self.verifier.verify_generation(sql_code, strategy)
            
            if is_valid:
                print("Glass Box Verification Passed ✅")
                return sql_code
                
            # STEP 3: Repair
            print("Verification Failed ❌. Initiating repair cycle...")
            repair_prompt = (
                f"Verification of your logic failed. The generated SQL does not satisfy the {strategy.value} strategy requirements. "
                "For INCREMENTAL loads, remember you must include a mathematically sound WHERE clause filtering by a watermark (e.g., 'updated_at >= watermark'). "
                "Please repair the SQL and return the corrected version in a markdown SQL block."
            )
            messages.append({"role": "user", "content": repair_prompt})
            
        raise RuntimeError(f"Agent failed to generate verifiably safe logic after {max_retries} repair retries.")

import ast
import z3
from enum import Enum
import litellm

class ETLStrategy(Enum):
    FULL_REFRESH = "FULL_REFRESH"
    INCREMENTAL = "INCREMENTAL"

class LogicVerifier:
    def __init__(self, model_name: str = "gemini/gemini-pro"):
        """
        Initializes the verifier with a model agnostic approach, using LiteLLM via ADK.
        """
        self.model_name = model_name
    
    def verify_generation(self, generated_sql: str, strategy: ETLStrategy, log_cb=None) -> bool:
        """
        Verifies if the generated SQL satisfies the mathematical constraints required by the ETL strategy.
        """
        if strategy == ETLStrategy.FULL_REFRESH:
            return True # No strict mathematical WHERE clause constraints for a full refresh
            
        elif strategy == ETLStrategy.INCREMENTAL:
            # We use an LLM to extract the core mathematical logic from the SQL WHERE clause
            # to be evaluated rigorously by our AST/Z3 engine.
            prompt = (
                "Extract the quantitative WHERE clause filtering logic from this SQL code "
                "as a raw Python boolean expression. "
                "CRITICAL: You MUST normalize the left-hand timestamp column to 'updated_at' and the right-hand threshold value/variable to 'watermark' (e.g., 'updated_at >= watermark'). "
                "Do not include SQL keywords like WHERE, AND, etc. Do not include explanation or markdown code block syntax. Only the pure Python comparison expression.\n\n"
                f"SQL:\n{generated_sql}"
            )
            
            # Using LiteLLM (which backs Google ADK integrations support for Claude, GPT, Gemini)
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            extracted_expr = response.choices[0].message.content.strip()
            # Fallback cleaning for markdown codeblocks if present
            if extracted_expr.startswith("```"):
                extracted_expr = "\n".join(extracted_expr.split("\n")[1:-1])
            return self._verify_incremental_logic(extracted_expr, log_cb=log_cb)
            
        return False

    def _verify_incremental_logic(self, python_expr: str, log_cb=None) -> bool:
        """
        Uses z3-solver and the python ast to ensure the incremental condition (updated_at >= watermark)
        is strongly satisfied by the generated logic.
        """
        msg = f"[Verifier] Extracted python expression for AST: '{python_expr}'"
        if log_cb: log_cb(msg)
        else: print(msg)
        try:
            tree = ast.parse(python_expr, mode='eval')
            z3_expr = self._ast_to_z3(tree.body)
            
            # Setting up algebraic bounds
            updated_at = z3.Int('updated_at')
            watermark = z3.Int('watermark')
            
            # Proving the generated condition strictly implies updated_at >= watermark
            required_condition = (updated_at >= watermark)
            
            solver = z3.Solver()
            # To prove implication A => B, we check if A AND NOT B is unsatisfiable
            solver.add(z3_expr)
            solver.add(z3.Not(required_condition))
            
            # If unsat, it proved the logic prevents reading data older than the watermark
            return solver.check() == z3.unsat
            
        except Exception as e:
            # If it could not be mapped to an AST or wasn't logically valid, we fail the proof
            print(f"Failed to verify proof with error: {e}")
            return False

    def _ast_to_z3(self, node: ast.AST):
        if isinstance(node, ast.Compare):
            left = self._ast_to_z3(node.left)
            if len(node.ops) != 1 or len(node.comparators) != 1:
                # We enforce simplicity in filter extraction
                raise ValueError("Complex chaining not supported")
            
            op = node.ops[0]
            right = self._ast_to_z3(node.comparators[0])
            
            if isinstance(op, ast.GtE):
                return left >= right
            elif isinstance(op, ast.Gt):
                return left > right
            elif isinstance(op, ast.LtE):
                return left <= right
            elif isinstance(op, ast.Lt):
                return left < right
            elif isinstance(op, ast.Eq):
                return left == right
            elif isinstance(op, ast.NotEq):
                return left != right
            else:
                raise ValueError(f"AST operation {op} not translatable to Z3")
                
        elif isinstance(node, ast.Name):
            return z3.Int(node.id)
            
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return z3.IntVal(int(node.value))
            raise ValueError("Only int/float boundary conditions supported in verifier.")
            
        elif isinstance(node, ast.BoolOp):
            values = [self._ast_to_z3(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return z3.And(*values)
            elif isinstance(node.op, ast.Or):
                return z3.Or(*values)
            else:
                raise ValueError("Unsupported bitwise/boolean combination")
                
        else:
            raise ValueError(f"Node architecture {type(node)} unsupported.")

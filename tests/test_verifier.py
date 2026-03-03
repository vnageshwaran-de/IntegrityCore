import pytest
from integritycore.core.verifier import LogicVerifier, ETLStrategy

def test_full_refresh_passes_everything():
    verifier = LogicVerifier()
    assert verifier.verify_generation("SELECT * FROM my_table", ETLStrategy.FULL_REFRESH) is True

def test_incremental_valid_logic():
    verifier = LogicVerifier()
    
    # A valid incremental logic that is mathematically >= 
    assert verifier._verify_incremental_logic("updated_at >= watermark") is True
    
    # Strictly greater is also valid, it safely implies >= 
    assert verifier._verify_incremental_logic("updated_at > watermark") is True
    
    # Equality is technically valid mathematically since A=B implies A>=B
    assert verifier._verify_incremental_logic("updated_at == watermark") is True

def test_incremental_invalid_logic():
    verifier = LogicVerifier()
    
    # Less than implies we are fetching older records, violating incremental loading constraints
    assert verifier._verify_incremental_logic("updated_at <= watermark") is False
    assert verifier._verify_incremental_logic("updated_at < watermark") is False
    
def test_unsupported_ast_logic():
    verifier = LogicVerifier()
    # Logic the simple AST parser isn't equipped to translate to Z3 should safely fail the proof
    assert verifier._verify_incremental_logic("updated_at + 1 >= watermark") is False

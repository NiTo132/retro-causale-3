import numpy as np
import pytest
from src.quantum_engine.state import QuantumState

def test_quantum_state_init():
    state = QuantumState()
    assert state.spatial.shape == (10,)
    assert state.probabilistic.shape == (3,)
    assert 0.0 <= state.complexity <= 1.0
    assert 0.0 <= state.emergence_potential <= 1.0
    assert state.causal_signature.shape == (10,) 
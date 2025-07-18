import numpy as np
from src.quantum_engine.state import QuantumState
from src.quantum_engine.selector import QuantumRetroCausalSelector

def test_select_optimal_future():
    state = QuantumState()
    futures = [QuantumState() for _ in range(3)]
    selector = QuantumRetroCausalSelector(config={})
    result = selector.select_optimal_future(state, futures)
    assert 'optimal_index' in result
    assert 'scores' in result
    assert 0 <= result['optimal_index'] < 3
    assert len(result['scores']) == 3 
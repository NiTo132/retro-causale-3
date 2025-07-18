import numpy as np
from src.quantum_engine.state import QuantumState
from src.quantum_engine.generator import ParallelFutureGenerator

def test_generate_futures():
    state = QuantumState()
    generator = ParallelFutureGenerator(n_futures=5)
    futures = generator.generate_futures(state)
    assert len(futures) == 5
    for f in futures:
        assert isinstance(f, QuantumState) 
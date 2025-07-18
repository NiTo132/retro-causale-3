import numpy as np
from src.quantum_engine.state import QuantumState
from src.quantum_engine.resonance import ResonanceCausalityField

def test_compute_resonance():
    state1 = QuantumState()
    state2 = QuantumState()
    field = ResonanceCausalityField()
    resonance = field.compute_resonance(state1, state2)
    assert isinstance(resonance, complex) 
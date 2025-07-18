import os
import tempfile
from src.quantum_engine.state import QuantumState
from src.quantum_engine.checkpoint import CheckpointManager

def test_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        state = QuantumState()
        manager.save(state, 'test')
        loaded = manager.load('test')
        assert isinstance(loaded, QuantumState) 
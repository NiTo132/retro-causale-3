from src.quantum_engine.state import QuantumState
from src.quantum_engine.visualization import EnterpriseVisualization

def test_plot_quantum_state_runs():
    vis = EnterpriseVisualization()
    state = QuantumState()
    # On vérifie que la fonction s'exécute sans erreur (pas de vérification graphique)
    vis.plot_quantum_state(state) 
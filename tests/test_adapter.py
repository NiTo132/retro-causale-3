import numpy as np
from src.quantum_engine.adapter import TradingRetroCausalAdapter

def test_adapter_predict_market_evolution():
    adapter = TradingRetroCausalAdapter(market_config={})
    sample_data = {
        'prices': np.random.rand(10),
        'volumes': np.random.rand(10),
        'technical_indicators': np.random.rand(5),
        'timestamp': 0.0
    }
    result = adapter.predict_market_evolution(sample_data, n_futures=5)
    assert 'direction' in result
    assert 'strength' in result
    assert 'confidence' in result 
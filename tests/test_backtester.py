import numpy as np
import pandas as pd
from src.quantum_engine.adapter import TradingRetroCausalAdapter
from src.quantum_engine.backtester import QuantumBacktester

def test_backtester_run_backtest():
    adapter = TradingRetroCausalAdapter(market_config={})
    backtester = QuantumBacktester(adapter)
    # Générer des données de marché fictives
    data = pd.DataFrame({
        'close': np.random.rand(30) * 100 + 1000,
        'volume': np.random.rand(30) * 1000000
    })
    result = backtester.run_backtest(data, initial_capital=10000, transaction_cost=0.001)
    assert 'portfolio' in result
    assert 'performance_metrics' in result 
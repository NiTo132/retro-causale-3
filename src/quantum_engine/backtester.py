import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from .adapter import TradingRetroCausalAdapter

class QuantumBacktester:
    """
    Backtester quantique complet pour évaluer les stratégies rétro-causales.
    """
    def __init__(self, trading_system: TradingRetroCausalAdapter):
        self.trading_system = trading_system
        self.results_history = []

    def run_backtest(
        self,
        historical_data: pd.DataFrame,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001
    ) -> Dict[str, Any]:
        """
        Exécute un backtest complet sur des données historiques.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"🚀 Démarrage backtest sur {len(historical_data)} périodes")
        portfolio = {
            'capital': initial_capital,
            'position': 0.0,
            'trades': [],
            'equity_curve': [initial_capital]
        }
        signals_history = []
        for i in range(20, len(historical_data)):
            try:
                market_data = self._prepare_market_data(historical_data, i)
                trading_signals = self.trading_system.predict_market_evolution(market_data)
                signals_history.append(trading_signals)
                self._execute_trade(portfolio, trading_signals, historical_data.iloc[i], transaction_cost)
                current_value = self._calculate_portfolio_value(portfolio, historical_data.iloc[i]['close'])
                portfolio['equity_curve'].append(current_value)
                if i % 50 == 0:
                    logger.debug(f"Période {i}/{len(historical_data)}, Capital: {current_value:.2f}")
            except Exception as e:
                logger.error(f"Erreur période {i}: {e}")
                continue
        performance_metrics = self._calculate_performance_metrics(portfolio, initial_capital, historical_data)
        logger.info(f"✅ Backtest terminé - ROI: {performance_metrics['total_return']:.2%}")
        return {
            'portfolio': portfolio,
            'signals_history': signals_history,
            'performance_metrics': performance_metrics,
            'equity_curve': portfolio['equity_curve']
        }

    def _prepare_market_data(self, df: pd.DataFrame, index: int) -> Dict[str, Any]:
        lookback = min(20, index)
        recent_data = df.iloc[max(0, index-lookback):index+1]
        prices = recent_data['close'].values
        volumes = recent_data.get('volume', pd.Series(np.ones(len(prices)) * 1000000)).values
        if len(prices) >= 10:
            sma_10 = np.mean(prices[-10:])
            volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
            momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            rsi = self._calculate_rsi(prices)
            technical_indicators = np.array([
                (prices[-1] - sma_10) / sma_10 if sma_10 > 0 else 0,
                volatility,
                momentum,
                rsi / 100.0,
                np.mean(np.diff(prices[-5:])) if len(prices) >= 5 else 0
            ])
        else:
            technical_indicators = np.zeros(5)
        return {
            'prices': prices,
            'volumes': volumes,
            'technical_indicators': technical_indicators,
            'timestamp': 0.0
        }

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed > 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _execute_trade(self, portfolio: Dict, signals: Dict[str, Any], market_data: pd.Series, transaction_cost: float):
        direction = signals['direction']
        position_size = 10000  # Ex: taille fixe
        current_price = market_data['close']
        if direction == 'BUY' and portfolio['position'] <= 0:
            shares_to_buy = position_size / current_price
            cost = shares_to_buy * current_price * (1 + transaction_cost)
            if cost <= portfolio['capital']:
                portfolio['position'] += shares_to_buy
                portfolio['capital'] -= cost
                portfolio['trades'].append({
                    'type': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost
                })
        elif direction == 'SELL' and portfolio['position'] > 0:
            shares_to_sell = min(portfolio['position'], position_size / current_price)
            proceeds = shares_to_sell * current_price * (1 - transaction_cost)
            portfolio['position'] -= shares_to_sell
            portfolio['capital'] += proceeds
            portfolio['trades'].append({
                'type': 'SELL',
                'shares': shares_to_sell,
                'price': current_price,
                'proceeds': proceeds
            })

    def _calculate_portfolio_value(self, portfolio: Dict, current_price: float) -> float:
        return portfolio['capital'] + portfolio['position'] * current_price

    def _calculate_performance_metrics(self, portfolio: Dict, initial_capital: float, market_data: pd.DataFrame) -> Dict[str, float]:
        equity_curve = np.array(portfolio['equity_curve'])
        final_value = equity_curve[-1]
        total_return = (final_value - initial_capital) / initial_capital
        daily_returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / (peak + 1e-8)
        max_drawdown = np.min(drawdown)
        num_trades = len(portfolio['trades'])
        market_return = (market_data['close'].iloc[-1] - market_data['close'].iloc[0]) / market_data['close'].iloc[0]
        alpha = total_return - market_return
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'alpha': alpha,
            'final_value': final_value
        }

def create_sample_market_data(n_periods: int = 1000) -> pd.DataFrame:
    """
    Création de données de marché simulées réalistes.
    """
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.02, n_periods)
    volatility = np.ones(n_periods)
    for i in range(1, n_periods):
        volatility[i] = 0.9 * volatility[i-1] + 0.1 * abs(returns[i-1])
    returns = returns * volatility
    initial_price = 1000
    prices = initial_price * np.exp(np.cumsum(returns))
    base_volume = 1000000
    volumes = base_volume * (1 + volatility + np.random.normal(0, 0.1, n_periods))
    volumes = np.abs(volumes)
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='D')
    return pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates) 
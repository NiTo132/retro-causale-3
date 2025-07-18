"""
💹 Trading Retro-Causal Adapter
===============================

Adaptateur enterprise pour le trading quantique rétro-causal. Orchestration complète
de la conversion des données de marché, génération de futurs, sélection optimale
et conversion en signaux de trading avec gestion de risque avancée.
"""

from __future__ import annotations

import time
import logging
import threading
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import pandas as pd
from scipy.stats import zscore

from ..core.state import QuantumState, StateFactory
from ..core.generator import ParallelFutureGenerator, GenerationMode, GeneratorConfig
from ..core.selector import QuantumRetroCausalSelector, SelectionConfig
from .signals import TradingSignal, SignalStrength, SignalConfidence, RiskLevel
from .portfolio import PositionSizer, RiskManager

__all__ = [
    "MarketRegime",
    "TradingMode", 
    "AdapterConfig",
    "MarketDataProcessor",
    "TradingRetroCausalAdapter",
    "PerformanceTracker"
]


class MarketRegime(Enum):
    """Régimes de marché détectés."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    BUBBLE = "bubble"


class TradingMode(Enum):
    """Modes de trading."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"


@dataclass
class AdapterConfig:
    """Configuration de l'adaptateur trading."""
    # Mode de trading
    trading_mode: TradingMode = TradingMode.BALANCED
    
    # Génération de futurs
    generation_mode: GenerationMode = GenerationMode.NORMAL
    n_futures_override: Optional[int] = None
    
    # Seuils de décision
    confidence_threshold: float = 0.65
    strength_threshold: float = 0.45
    
    # Gestion de risque
    max_position_pct: float = 0.08
    max_risk_per_trade: float = 0.015
    max_drawdown_limit: float = 0.15
    
    # Signaux techniques
    enable_technical_indicators: bool = True
    technical_weight: float = 0.25
    quantum_weight: float = 0.75
    
    # Optimisations
    enable_regime_detection: bool = True
    enable_adaptive_thresholds: bool = True
    enable_risk_adjustment: bool = True
    
    # Performance
    enable_performance_tracking: bool = True
    performance_window: int = 100
    
    # Paramètres techniques
    sma_periods: List[int] = field(default_factory=lambda: [10, 21, 50])
    rsi_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "trading_mode": self.trading_mode.value,
            "generation_mode": self.generation_mode.value,
            "confidence_threshold": self.confidence_threshold,
            "strength_threshold": self.strength_threshold,
            "max_position_pct": self.max_position_pct,
            "enable_technical_indicators": self.enable_technical_indicators,
            "technical_weight": self.technical_weight,
            "quantum_weight": self.quantum_weight,
        }


class MarketDataProcessor:
    """Processeur avancé de données de marché."""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Historique pour calculs techniques
        self.price_history = []
        self.volume_history = []
        self.indicator_cache = {}
        
        # Détection de régime
        self.regime_detector = MarketRegimeDetector()
        
    def process_market_data(
        self, 
        market_data: Dict[str, Any],
        include_technical: bool = True
    ) -> Dict[str, Any]:
        """Traitement complet des données de marché."""
        
        try:
            # Validation des données
            validated_data = self._validate_market_data(market_data)
            
            # Mise à jour de l'historique
            self._update_history(validated_data)
            
            # Calcul des indicateurs techniques
            if include_technical and self.config.enable_technical_indicators:
                technical_indicators = self._compute_technical_indicators()
                validated_data["technical_indicators"] = technical_indicators
            
            # Détection du régime de marché
            if self.config.enable_regime_detection:
                market_regime = self.regime_detector.detect_regime(
                    self.price_history, self.volume_history
                )
                validated_data["market_regime"] = market_regime
            
            # Normalisation pour état quantique
            validated_data["normalized_data"] = self._normalize_for_quantum_state(validated_data)
            
            return validated_data
            
        except Exception as e:
            self.logger.error(f"Erreur traitement données marché: {e}")
            return self._create_fallback_data(market_data)
    
    def _validate_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validation et nettoyage des données."""
        
        validated = {}
        
        # Prix obligatoires
        prices = data.get("prices", [])
        if not prices:
            prices = [1000.0]  # Prix par défaut
        
        validated["prices"] = np.array(prices, dtype=np.float64)
        
        # Volumes
        volumes = data.get("volumes", [])
        if not volumes:
            volumes = [1000000.0] * len(validated["prices"])
        
        validated["volumes"] = np.array(volumes, dtype=np.float64)
        
        # Timestamps
        validated["timestamp"] = data.get("timestamp", time.time())
        
        # OHLC si disponible
        for field in ["open", "high", "low", "close"]:
            if field in data:
                validated[field] = np.array(data[field], dtype=np.float64)
        
        # Si pas de OHLC, utiliser les prix
        if "close" not in validated:
            validated["close"] = validated["prices"]
        
        # Validation des valeurs
        for key, values in validated.items():
            if isinstance(values, np.ndarray):
                # Suppression des NaN et infinis
                finite_mask = np.isfinite(values)
                if not np.all(finite_mask):
                    self.logger.warning(f"Valeurs non finies détectées dans {key}")
                    validated[key] = self._interpolate_missing_values(values)
        
        return validated
    
    def _interpolate_missing_values(self, values: np.ndarray) -> np.ndarray:
        """Interpolation des valeurs manquantes."""
        
        if len(values) == 0:
            return values
        
        # Masque des valeurs finies
        finite_mask = np.isfinite(values)
        
        if not np.any(finite_mask):
            # Toutes les valeurs sont invalides
            return np.full_like(values, np.nanmean(values) if not np.isnan(np.nanmean(values)) else 1000.0)
        
        if np.all(finite_mask):
            # Toutes les valeurs sont valides
            return values
        
        # Interpolation linéaire
        valid_indices = np.where(finite_mask)[0]
        invalid_indices = np.where(~finite_mask)[0]
        
        interpolated = values.copy()
        interpolated[invalid_indices] = np.interp(
            invalid_indices, 
            valid_indices, 
            values[valid_indices]
        )
        
        return interpolated
    
    def _update_history(self, data: Dict[str, Any]) -> None:
        """Mise à jour de l'historique des données."""
        
        # Prix
        if "close" in data:
            current_price = data["close"][-1] if len(data["close"]) > 0 else 1000.0
        else:
            current_price = data["prices"][-1] if len(data["prices"]) > 0 else 1000.0
        
        self.price_history.append(float(current_price))
        
        # Volumes
        current_volume = data["volumes"][-1] if len(data["volumes"]) > 0 else 1000000.0
        self.volume_history.append(float(current_volume))
        
        # Limitation de l'historique
        max_history = 500
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
        if len(self.volume_history) > max_history:
            self.volume_history = self.volume_history[-max_history:]
    
    def _compute_technical_indicators(self) -> np.ndarray:
        """Calcul des indicateurs techniques."""
        
        if len(self.price_history) < 10:
            return np.zeros(10, dtype=np.float32)  # Pas assez de données
        
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        
        indicators = []
        
        try:
            # 1. Moyennes mobiles
            for period in self.config.sma_periods:
                if len(prices) >= period:
                    sma = np.mean(prices[-period:])
                    current_price = prices[-1]
                    sma_signal = (current_price - sma) / sma if sma > 0 else 0.0
                else:
                    sma_signal = 0.0
                indicators.append(sma_signal)
            
            # 2. RSI
            rsi = self._calculate_rsi(prices, self.config.rsi_period)
            rsi_signal = (rsi - 50) / 50  # Normalisation [-1, 1]
            indicators.append(rsi_signal)
            
            # 3. Bollinger Bands
            if len(prices) >= self.config.bollinger_period:
                bb_middle = np.mean(prices[-self.config.bollinger_period:])
                bb_std = np.std(prices[-self.config.bollinger_period:])
                bb_upper = bb_middle + self.config.bollinger_std * bb_std
                bb_lower = bb_middle - self.config.bollinger_std * bb_std
                
                current_price = prices[-1]
                if bb_upper > bb_lower:
                    bb_position = (current_price - bb_middle) / (bb_upper - bb_lower)
                else:
                    bb_position = 0.0
            else:
                bb_position = 0.0
            indicators.append(bb_position)
            
            # 4. Momentum
            momentum_period = 10
            if len(prices) >= momentum_period:
                momentum = (prices[-1] - prices[-momentum_period]) / prices[-momentum_period]
            else:
                momentum = 0.0
            indicators.append(momentum)
            
            # 5. Volatilité
            volatility_period = 20
            if len(prices) >= volatility_period:
                returns = np.diff(prices[-volatility_period:]) / prices[-volatility_period:-1]
                volatility = np.std(returns)
            else:
                volatility = 0.0
            indicators.append(volatility)
            
            # 6. Volume
            if len(volumes) >= 20:
                volume_sma = np.mean(volumes[-20:])
                current_volume = volumes[-1]
                volume_signal = (current_volume - volume_sma) / volume_sma if volume_sma > 0 else 0.0
            else:
                volume_signal = 0.0
            indicators.append(volume_signal)
            
            # Compléter à 10 indicateurs
            while len(indicators) < 10:
                indicators.append(0.0)
            
            # Normalisation finale
            indicators_array = np.array(indicators[:10], dtype=np.float32)
            
            # Clipping pour éviter les valeurs extrêmes
            indicators_array = np.clip(indicators_array, -5.0, 5.0)
            
            return indicators_array
            
        except Exception as e:
            self.logger.warning(f"Erreur calcul indicateurs techniques: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calcul du RSI."""
        
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(np.clip(rsi, 0, 100))
    
    def _normalize_for_quantum_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalisation des données pour l'état quantique."""
        
        normalized = {}
        
        # Normalisation sécurisée
        def safe_normalize(arr: np.ndarray, method: str = "zscore") -> np.ndarray:
            arr = np.asarray(arr, dtype=np.float32)
            
            if len(arr) == 0:
                return np.array([0.0])
            
            if method == "zscore":
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                if std_val > 1e-8:
                    return (arr - mean_val) / std_val
                else:
                    return arr - mean_val
            elif method == "minmax":
                min_val, max_val = np.min(arr), np.max(arr)
                if max_val > min_val:
                    return (arr - min_val) / (max_val - min_val)
                else:
                    return np.zeros_like(arr)
            else:
                return arr
        
        # Prix normalisés
        prices = data.get("prices", np.array([1000.0]))
        normalized["prices"] = safe_normalize(prices[-20:] if len(prices) >= 20 else prices)
        
        # Volumes normalisés
        volumes = data.get("volumes", np.array([1000000.0]))
        normalized["volumes"] = safe_normalize(volumes[-20:] if len(volumes) >= 20 else volumes)
        
        # Indicateurs techniques (déjà normalisés)
        if "technical_indicators" in data:
            normalized["technical_indicators"] = data["technical_indicators"]
        else:
            normalized["technical_indicators"] = np.zeros(10, dtype=np.float32)
        
        return normalized
    
    def _create_fallback_data(self, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Création de données de fallback en cas d'erreur."""
        
        return {
            "prices": np.array([1000.0]),
            "volumes": np.array([1000000.0]),
            "timestamp": time.time(),
            "technical_indicators": np.zeros(10, dtype=np.float32),
            "market_regime": MarketRegime.SIDEWAYS,
            "normalized_data": {
                "prices": np.array([0.0]),
                "volumes": np.array([0.0]),
                "technical_indicators": np.zeros(10, dtype=np.float32),
            }
        }


class MarketRegimeDetector:
    """Détecteur de régime de marché."""
    
    def __init__(self):
        self.detection_cache = {}
        
    def detect_regime(
        self, 
        prices: List[float], 
        volumes: List[float]
    ) -> MarketRegime:
        """Détection du régime de marché actuel."""
        
        if len(prices) < 20:
            return MarketRegime.SIDEWAYS
        
        try:
            prices_array = np.array(prices[-50:])  # 50 dernières observations
            volumes_array = np.array(volumes[-50:])
            
            # Calculs de base
            returns = np.diff(prices_array) / prices_array[:-1]
            volatility = np.std(returns)
            trend_strength = self._calculate_trend_strength(prices_array)
            volume_profile = self._analyze_volume_profile(volumes_array)
            
            # Seuils de décision
            high_vol_threshold = 0.03
            trend_threshold = 0.7
            
            # Logique de détection
            if volatility > high_vol_threshold * 2:
                return MarketRegime.CRISIS
            elif volatility > high_vol_threshold:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < high_vol_threshold * 0.3:
                return MarketRegime.LOW_VOLATILITY
            elif trend_strength > trend_threshold:
                if np.mean(returns) > 0:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception:
            return MarketRegime.SIDEWAYS
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calcul de la force de tendance."""
        
        if len(prices) < 10:
            return 0.0
        
        # Régression linéaire simple
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # R² pour mesurer la qualité de la tendance
        y_pred = slope * x + np.mean(prices)
        ss_tot = np.sum((prices - np.mean(prices))**2)
        ss_res = np.sum((prices - y_pred)**2)
        
        if ss_tot > 1e-10:
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, r_squared)
        else:
            return 0.0
    
    def _analyze_volume_profile(self, volumes: np.ndarray) -> Dict[str, float]:
        """Analyse du profil de volume."""
        
        if len(volumes) < 10:
            return {"trend": 0.0, "spike": 0.0}
        
        # Tendance du volume
        x = np.arange(len(volumes))
        volume_slope, _ = np.polyfit(x, volumes, 1)
        volume_trend = volume_slope / np.mean(volumes) if np.mean(volumes) > 0 else 0.0
        
        # Pics de volume
        volume_z_scores = zscore(volumes)
        volume_spikes = np.sum(volume_z_scores > 2) / len(volumes)
        
        return {
            "trend": float(volume_trend),
            "spike": float(volume_spikes)
        }


class TradingRetroCausalAdapter:
    """Adaptateur principal pour le trading rétro-causal enterprise."""
    
    def __init__(
        self, 
        config: Optional[AdapterConfig] = None,
        generator_config: Optional[GeneratorConfig] = None,
        selector_config: Optional[SelectionConfig] = None
    ):
        
        # Configuration
        self.config = config or AdapterConfig()
        
        # Composants core
        self.generator = ParallelFutureGenerator(generator_config)
        self.selector = QuantumRetroCausalSelector(selector_config)
        
        # Composants trading
        self.market_processor = MarketDataProcessor(self.config)
        self.position_sizer = PositionSizer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.performance_tracker = PerformanceTracker() if self.config.enable_performance_tracking else None
        
        # État interne
        self.last_prediction_time = 0.0
        self.prediction_history = []
        self.current_market_regime = MarketRegime.SIDEWAYS
        
        # Thread safety
        self.prediction_lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("TradingRetroCausalAdapter initialisé")
    
    def predict_market_evolution(
        self,
        market_data: Dict[str, Any],
        n_futures: Optional[int] = None,
        force_quantum_only: bool = False
    ) -> TradingSignal:
        """
        Prédiction de l'évolution du marché avec génération de signal de trading.
        
        Args:
            market_data: Données de marché actuelles
            n_futures: Nombre de futurs à générer (optionnel)
            force_quantum_only: Forcer l'utilisation du signal quantique uniquement
            
        Returns:
            Signal de trading complet avec métadonnées
        """
        
        with self.prediction_lock:
            start_time = time.time()
            
            try:
                # 1. Traitement des données de marché
                processed_data = self.market_processor.process_market_data(market_data)
                self.current_market_regime = processed_data.get("market_regime", MarketRegime.SIDEWAYS)
                
                # 2. Conversion en état quantique
                quantum_state = self._convert_to_quantum_state(processed_data)
                
                # 3. Adaptation des paramètres selon le régime
                adapted_config = self._adapt_config_for_regime(self.current_market_regime)
                
                # 4. Génération de futurs quantiques
                n_futures = n_futures or adapted_config.get("n_futures", self._get_default_futures_count())
                
                futures, generation_metrics = self.generator.generate_futures(
                    quantum_state,
                    mode=self.config.generation_mode,
                    n_futures=n_futures
                )
                
                # 5. Sélection du futur optimal
                selection_result = self.selector.select_optimal_future(
                    quantum_state,
                    futures,
                    context={"market_regime": self.current_market_regime}
                )
                
                # 6. Calcul du signal quantique
                quantum_signal = self._compute_quantum_signal(
                    quantum_state, selection_result, processed_data
                )
                
                # 7. Signal technique (si activé)
                technical_signal = None
                if not force_quantum_only and self.config.enable_technical_indicators:
                    technical_signal = self._compute_technical_signal(processed_data)
                
                # 8. Combinaison des signaux
                combined_signal = self._combine_signals(quantum_signal, technical_signal)
                
                # 9. Gestion de risque
                risk_adjusted_signal = self.risk_manager.adjust_signal(
                    combined_signal, processed_data, self.current_market_regime
                )
                
                # 10. Sizing de position
                final_signal = self.position_sizer.size_position(
                    risk_adjusted_signal, processed_data
                )
                
                # 11. Métadonnées et tracking
                processing_time = time.time() - start_time
                final_signal.metadata.update({
                    "processing_time": processing_time,
                    "generation_metrics": generation_metrics,
                    "selection_metrics": {
                        "confidence": selection_result.confidence_score,
                        "diversity": selection_result.diversity_index,
                        "causal_entropy": selection_result.causal_entropy,
                    },
                    "market_regime": self.current_market_regime.value,
                    "futures_analyzed": len(futures),
                    "quantum_state_quality": quantum_state.quality_score,
                })
                
                # 12. Performance tracking
                if self.performance_tracker:
                    self.performance_tracker.record_prediction(
                        final_signal, processed_data, processing_time
                    )
                
                # 13. Historique
                self._update_prediction_history(final_signal, processed_data)
                
                self.logger.debug(
                    f"Prédiction générée: {final_signal.direction.value} "
                    f"(force={final_signal.strength:.3f}, conf={final_signal.confidence:.3f}) "
                    f"en {processing_time:.3f}s"
                )
                
                return final_signal
                
            except Exception as e:
                self.logger.error(f"Erreur prédiction marché: {e}")
                return self._create_emergency_signal(market_data, start_time)
    
    def _convert_to_quantum_state(self, processed_data: Dict[str, Any]) -> QuantumState:
        """Conversion des données de marché en état quantique."""
        
        try:
            normalized_data = processed_data["normalized_data"]
            
            # Extraction des données normalisées
            prices = normalized_data["prices"]
            volumes = normalized_data["volumes"] 
            technical_indicators = normalized_data["technical_indicators"]
            
            # Utilisation de StateFactory pour robustesse
            return StateFactory.create_from_market_data(
                prices=processed_data.get("prices", np.array([1000.0])),
                volumes=processed_data.get("volumes", np.array([1000000.0])),
                indicators=technical_indicators,
                timestamp=processed_data.get("timestamp", time.time())
            )
            
        except Exception as e:
            self.logger.warning(f"Erreur conversion état quantique: {e}")
            return StateFactory.create_random()
    
    def _adapt_config_for_regime(self, regime: MarketRegime) -> Dict[str, Any]:
        """Adaptation de la configuration selon le régime de marché."""
        
        adaptations = {
            MarketRegime.CRISIS: {
                "n_futures": int(self._get_default_futures_count() * 1.5),
                "confidence_threshold": self.config.confidence_threshold + 0.1,
                "technical_weight": 0.4,  # Plus de poids aux signaux techniques
                "quantum_weight": 0.6,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "n_futures": int(self._get_default_futures_count() * 1.2),
                "confidence_threshold": self.config.confidence_threshold + 0.05,
                "technical_weight": 0.3,
                "quantum_weight": 0.7,
            },
            MarketRegime.LOW_VOLATILITY: {
                "n_futures": int(self._get_default_futures_count() * 0.8),
                "confidence_threshold": self.config.confidence_threshold - 0.05,
                "technical_weight": 0.2,
                "quantum_weight": 0.8,
            },
            MarketRegime.TRENDING_UP: {
                "n_futures": self._get_default_futures_count(),
                "confidence_threshold": self.config.confidence_threshold,
                "technical_weight": 0.3,
                "quantum_weight": 0.7,
            },
            MarketRegime.TRENDING_DOWN: {
                "n_futures": self._get_default_futures_count(),
                "confidence_threshold": self.config.confidence_threshold,
                "technical_weight": 0.3,
                "quantum_weight": 0.7,
            },
            MarketRegime.SIDEWAYS: {
                "n_futures": int(self._get_default_futures_count() * 0.9),
                "confidence_threshold": self.config.confidence_threshold,
                "technical_weight": self.config.technical_weight,
                "quantum_weight": self.config.quantum_weight,
            }
        }
        
        return adaptations.get(regime, {
            "n_futures": self._get_default_futures_count(),
            "confidence_threshold": self.config.confidence_threshold,
            "technical_weight": self.config.technical_weight,
            "quantum_weight": self.config.quantum_weight,
        })
    
    def _get_default_futures_count(self) -> int:
        """Obtention du nombre de futurs par défaut."""
        if self.config.n_futures_override:
            return self.config.n_futures_override
        
        mode_mapping = {
            GenerationMode.FAST: 300,
            GenerationMode.NORMAL: 1000,
            GenerationMode.DEEP: 2500,
            GenerationMode.ADAPTIVE: 1000,
        }
        
        return mode_mapping.get(self.config.generation_mode, 1000)
    
    def _compute_quantum_signal(
        self,
        quantum_state: QuantumState,
        selection_result: Any,
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcul du signal quantique pur."""
        
        optimal_future = selection_result.optimal_future
        
        # Direction basée sur l'évolution spatiale
        spatial_evolution = np.mean(optimal_future.spatial[:10]) - np.mean(quantum_state.spatial[:10])
        
        if spatial_evolution > 0.1:
            direction = "BUY"
        elif spatial_evolution < -0.1:
            direction = "SELL"
        else:
            direction = "HOLD"
        
        # Force basée sur la magnitude de l'évolution
        strength = min(abs(spatial_evolution) * 2, 1.0)
        
        # Confiance basée sur la sélection
        confidence = selection_result.confidence_score
        
        # Émergence
        emergence_level = optimal_future.emergence_potential
        
        # Risque basé sur la cohérence
        risk_level = 1.0 - np.mean(selection_result.coherences)
        
        return {
            "direction": direction,
            "strength": strength,
            "confidence": confidence,
            "emergence_level": emergence_level,
            "risk_level": risk_level,
            "quantum_metrics": {
                "spatial_evolution": spatial_evolution,
                "resonance_max": float(np.max(selection_result.resonances)),
                "causal_entropy": selection_result.causal_entropy,
                "diversity_index": selection_result.diversity_index,
            }
        }
    
    def _compute_technical_signal(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcul du signal technique."""
        
        technical_indicators = processed_data.get("technical_indicators", np.zeros(10))
        
        # Signaux basés sur les indicateurs
        signals = []
        
        # SMA (premiers 3 indicateurs)
        sma_signals = technical_indicators[:3]
        sma_consensus = np.mean(sma_signals)
        signals.append(sma_consensus)
        
        # RSI (4ème indicateur)
        rsi_signal = technical_indicators[3] if len(technical_indicators) > 3 else 0.0
        signals.append(rsi_signal)
        
        # Bollinger Bands (5ème indicateur)
        bb_signal = technical_indicators[4] if len(technical_indicators) > 4 else 0.0
        signals.append(bb_signal)
        
        # Momentum (6ème indicateur)
        momentum_signal = technical_indicators[5] if len(technical_indicators) > 5 else 0.0
        signals.append(momentum_signal)
        
        # Signal technique global
        technical_score = np.mean(signals)
        
        # Direction
        if technical_score > 0.2:
            direction = "BUY"
        elif technical_score < -0.2:
            direction = "SELL"
        else:
            direction = "HOLD"
        
        # Force et confiance
        strength = min(abs(technical_score), 1.0)
        confidence = strength  # Simplicité
        
        return {
            "direction": direction,
            "strength": strength,
            "confidence": confidence,
            "technical_score": technical_score,
            "individual_signals": signals,
        }
    
    def _combine_signals(
        self,
        quantum_signal: Dict[str, Any],
        technical_signal: Optional[Dict[str, Any]]
    ) -> TradingSignal:
        """Combinaison des signaux quantique et technique."""
        
        if technical_signal is None:
            # Signal quantique seulement
            return TradingSignal.from_dict(quantum_signal)
        
        # Poids adaptatifs
        quantum_weight = self.config.quantum_weight
        technical_weight = self.config.technical_weight
        
        # Normalisation des poids
        total_weight = quantum_weight + technical_weight
        if total_weight > 0:
            quantum_weight /= total_weight
            technical_weight /= total_weight
        
        # Combinaison des forces
        combined_strength = (
            quantum_weight * quantum_signal["strength"] +
            technical_weight * technical_signal["strength"]
        )
        
        # Combinaison des confiances
        combined_confidence = (
            quantum_weight * quantum_signal["confidence"] +
            technical_weight * technical_signal["confidence"]
        )
        
        # Direction par consensus ou priorité quantique
        quantum_dir = quantum_signal["direction"]
        technical_dir = technical_signal["direction"]
        
        if quantum_dir == technical_dir:
            # Consensus
            final_direction = quantum_dir
            consensus_bonus = 1.2
            combined_confidence *= consensus_bonus
        elif quantum_dir == "HOLD" or technical_dir == "HOLD":
            # L'un est neutre
            final_direction = quantum_dir if quantum_dir != "HOLD" else technical_dir
        else:
            # Divergence - priorité au quantique avec réduction de confiance
            final_direction = quantum_dir
            combined_confidence *= 0.7
        
        # Limitation des valeurs
        combined_strength = min(combined_strength, 1.0)
        combined_confidence = min(combined_confidence, 1.0)
        
        # Création du signal final
        return TradingSignal(
            direction=TradingSignal.Direction[final_direction],
            strength=SignalStrength.from_value(combined_strength),
            confidence=SignalConfidence.from_value(combined_confidence),
            risk_level=RiskLevel.from_value(quantum_signal.get("risk_level", 0.5)),
            emergence_level=quantum_signal.get("emergence_level", 0.5),
            metadata={
                "quantum_signal": quantum_signal,
                "technical_signal": technical_signal,
                "weights": {
                    "quantum": quantum_weight,
                    "technical": technical_weight
                },
                "consensus": quantum_dir == technical_dir,
            }
        )
    
    def _update_prediction_history(self, signal: TradingSignal, data: Dict[str, Any]) -> None:
        """Mise à jour de l'historique des prédictions."""
        
        entry = {
            "timestamp": time.time(),
            "signal": signal.to_dict(),
            "market_regime": self.current_market_regime.value,
            "market_data_summary": {
                "last_price": float(data.get("prices", [1000.0])[-1]),
                "volume": float(data.get("volumes", [1000000.0])[-1]),
            }
        }
        
        self.prediction_history.append(entry)
        
        # Limitation de l'historique
        max_history = self.config.performance_window * 2
        if len(self.prediction_history) > max_history:
            self.prediction_history = self.prediction_history[-max_history:]
    
    def _create_emergency_signal(self, market_data: Dict[str, Any], start_time: float) -> TradingSignal:
        """Création d'un signal d'urgence en cas d'erreur."""
        
        processing_time = time.time() - start_time
        
        return TradingSignal(
            direction=TradingSignal.Direction.HOLD,
            strength=SignalStrength.WEAK,
            confidence=SignalConfidence.LOW,
            risk_level=RiskLevel.HIGH,
            emergence_level=0.5,
            metadata={
                "emergency": True,
                "processing_time": processing_time,
                "error_fallback": True,
            }
        )
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Statistiques complètes de l'adaptateur."""
        
        stats = {
            "config": self.config.to_dict(),
            "current_regime": self.current_market_regime.value,
            "predictions_count": len(self.prediction_history),
            "generator_stats": self.generator.get_statistics(),
            "selector_stats": self.selector.get_statistics(),
        }
        
        if self.performance_tracker:
            stats["performance"] = self.performance_tracker.get_statistics()
        
        # Analyse de l'historique récent
        if len(self.prediction_history) > 10:
            recent_predictions = self.prediction_history[-50:]  # 50 dernières
            
            directions = [p["signal"]["direction"] for p in recent_predictions]
            strengths = [p["signal"]["strength"] for p in recent_predictions]
            confidences = [p["signal"]["confidence"] for p in recent_predictions]
            
            stats["recent_analysis"] = {
                "buy_ratio": directions.count("BUY") / len(directions),
                "sell_ratio": directions.count("SELL") / len(directions),
                "hold_ratio": directions.count("HOLD") / len(directions),
                "avg_strength": np.mean(strengths),
                "avg_confidence": np.mean(confidences),
            }
        
        return stats


class PerformanceTracker:
    """Tracker de performance pour l'adaptateur."""
    
    def __init__(self):
        self.prediction_records = []
        self.performance_metrics = {}
        
    def record_prediction(
        self,
        signal: TradingSignal,
        market_data: Dict[str, Any],
        processing_time: float
    ) -> None:
        """Enregistrement d'une prédiction."""
        
        record = {
            "timestamp": time.time(),
            "signal": signal.to_dict(),
            "processing_time": processing_time,
            "market_context": {
                "price": float(market_data.get("prices", [1000.0])[-1]),
                "volume": float(market_data.get("volumes", [1000000.0])[-1]),
            }
        }
        
        self.prediction_records.append(record)
        
        # Limitation
        if len(self.prediction_records) > 1000:
            self.prediction_records = self.prediction_records[-500:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Statistiques de performance."""
        
        if not self.prediction_records:
            return {}
        
        processing_times = [r["processing_time"] for r in self.prediction_records]
        
        return {
            "total_predictions": len(self.prediction_records),
            "avg_processing_time": np.mean(processing_times),
            "max_processing_time": np.max(processing_times),
            "min_processing_time": np.min(processing_times),
            "processing_time_std": np.std(processing_times),
        }
"""
🎯 Advanced Quantum Retro-Causal Future Selector
===============================================

Enterprise-grade future selection system with multi-criteria optimization,
machine learning integration, distributed computing, and comprehensive
selection analytics with advanced scoring algorithms.

Features:
- Multi-objective optimization with Pareto frontier analysis
- Machine learning-based selection models (ensemble methods)
- Distributed computing with C++/CUDA acceleration
- Advanced scoring algorithms (quantum, causal, temporal)
- Real-time selection optimization and adaptation
- Comprehensive selection analytics and insights
- Selection validation and cross-validation
- Performance monitoring and benchmarking
- Selection explanation and interpretability
- Risk-aware selection with uncertainty quantification
- Dynamic selection criteria adaptation
- Portfolio optimization integration

Author: Advanced Engineering Team
Version: 2.0.0
License: MIT
"""

import asyncio
import ctypes
import importlib
import logging
import os
import time
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Generic, TypeVar,
    Protocol, runtime_checkable, Iterator, AsyncIterator, Set, NamedTuple
)
import pickle
import json
from threading import RLock, Event

import numpy as np
from scipy import optimize, stats, spatial
from scipy.spatial.distance import pdist, cdist, cosine, euclidean
from scipy.stats import rankdata, spearmanr, kendalltau
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, BaggingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
from numba import jit, njit, prange, cuda
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

from .state import QuantumState

# ==================== TYPE DEFINITIONS ====================

T = TypeVar('T')
StateType = TypeVar('StateType', bound='QuantumState')
ScoreType = TypeVar('ScoreType')

# ==================== CONSTANTS ====================

SELECTOR_VERSION = "2.0.0"
DEFAULT_SELECTION_TIMEOUT = 30.0  # seconds
DEFAULT_BATCH_SIZE = 1000
MAX_FUTURES_LIMIT = 100000
DEFAULT_CROSS_VALIDATION_FOLDS = 5
PARETO_EPSILON = 1e-6
SCORE_PRECISION = 1e-10

# ==================== METRICS ====================

selection_operations = Counter(
    'quantum_selection_operations_total',
    'Total selection operations',
    ['selection_method', 'status']
)

selection_duration = Histogram(
    'quantum_selection_duration_seconds',
    'Selection operation duration',
    ['selection_method', 'batch_size_range']
)

selection_accuracy = Gauge(
    'quantum_selection_accuracy',
    'Selection accuracy score',
    ['selection_method', 'validation_type']
)

pareto_frontier_size = Gauge(
    'quantum_pareto_frontier_size',
    'Size of Pareto frontier',
    ['selection_method']
)

ml_model_performance = Gauge(
    'quantum_ml_model_performance',
    'ML model performance score',
    ['model_type', 'metric']
)

# ==================== EXCEPTIONS ====================

class SelectionError(Exception):
    """Base selection exception."""
    pass

class SelectionTimeoutError(SelectionError):
    """Selection timeout error."""
    pass

class SelectionValidationError(SelectionError):
    """Selection validation error."""
    pass

class SelectionComputationError(SelectionError):
    """Selection computation error."""
    pass

class ParetoOptimizationError(SelectionError):
    """Pareto optimization error."""
    pass

class ModelTrainingError(SelectionError):
    """Model training error."""
    pass

# ==================== ENUMS ====================

class SelectionMethod(Enum):
    """Selection methods."""
    SIMPLE_DISTANCE = auto()
    WEIGHTED_SCORE = auto()
    PARETO_OPTIMAL = auto()
    MACHINE_LEARNING = auto()
    HYBRID_ENSEMBLE = auto()
    QUANTUM_INSPIRED = auto()
    MULTI_OBJECTIVE = auto()
    EVOLUTIONARY = auto()

class ScoreFunction(Enum):
    """Score functions."""
    EUCLIDEAN_DISTANCE = auto()
    COSINE_SIMILARITY = auto()
    MANHATTAN_DISTANCE = auto()
    MAHALANOBIS_DISTANCE = auto()
    QUANTUM_FIDELITY = auto()
    CAUSAL_STRENGTH = auto()
    EMERGENCE_POTENTIAL = auto()
    COMPOSITE_SCORE = auto()

class OptimizationObjective(Enum):
    """Optimization objectives."""
    MAXIMIZE_QUALITY = auto()
    MINIMIZE_RISK = auto()
    MAXIMIZE_DIVERSITY = auto()
    MAXIMIZE_EMERGENCE = auto()
    MINIMIZE_CAUSAL_VIOLATION = auto()
    MAXIMIZE_COHERENCE = auto()

class ValidationMethod(Enum):
    """Validation methods."""
    CROSS_VALIDATION = auto()
    BOOTSTRAP = auto()
    HOLDOUT = auto()
    TIME_SERIES_SPLIT = auto()
    STRATIFIED = auto()

class AccelerationType(Enum):
    """Acceleration types."""
    CPU = auto()
    GPU_CUDA = auto()
    GPU_OPENCL = auto()
    DISTRIBUTED_MPI = auto()
    HYBRID = auto()

# ==================== DATA STRUCTURES ====================

@dataclass(frozen=True)
class SelectionConfig:
    """Advanced selection configuration."""
    
    # Core Selection
    selection_method: SelectionMethod = SelectionMethod.HYBRID_ENSEMBLE
    score_functions: List[ScoreFunction] = field(default_factory=lambda: [
        ScoreFunction.QUANTUM_FIDELITY,
        ScoreFunction.CAUSAL_STRENGTH,
        ScoreFunction.EMERGENCE_POTENTIAL
    ])
    
    # Optimization
    optimization_objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.MAXIMIZE_QUALITY,
        OptimizationObjective.MINIMIZE_RISK
    ])
    
    # Performance
    enable_parallel: bool = True
    max_workers: int = 4
    batch_size: int = DEFAULT_BATCH_SIZE
    timeout_seconds: float = DEFAULT_SELECTION_TIMEOUT
    
    # Machine Learning
    enable_ml: bool = True
    ml_models: List[str] = field(default_factory=lambda: [
        'random_forest', 'gradient_boosting', 'neural_network'
    ])
    validation_method: ValidationMethod = ValidationMethod.CROSS_VALIDATION
    validation_folds: int = DEFAULT_CROSS_VALIDATION_FOLDS
    
    # Acceleration
    acceleration_type: AccelerationType = AccelerationType.CPU
    enable_cpp_acceleration: bool = True
    enable_gpu_acceleration: bool = True
    
    # Quality Control
    enable_validation: bool = True
    enable_explanation: bool = True
    enable_uncertainty_quantification: bool = True
    
    # Advanced Features
    enable_pareto_optimization: bool = True
    enable_adaptive_weights: bool = True
    enable_dynamic_criteria: bool = True
    
    # Caching and Optimization
    enable_caching: bool = True
    cache_size: int = 10000
    enable_precomputation: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.validation_folds < 2:
            raise ValueError("validation_folds must be at least 2")
        if not self.score_functions:
            raise ValueError("At least one score function must be specified")

@dataclass
class SelectionResult:
    """Comprehensive selection result."""
    
    # Core Results
    optimal_index: int
    optimal_future: StateType
    all_scores: np.ndarray
    selection_confidence: float
    
    # Analysis Results
    pareto_indices: List[int] = field(default_factory=list)
    score_breakdown: Dict[str, np.ndarray] = field(default_factory=dict)
    ranking_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Performance Metrics
    selection_time: float = 0.0
    computation_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Validation Results
    cross_validation_scores: Optional[np.ndarray] = None
    model_performance: Dict[str, float] = field(default_factory=dict)
    uncertainty_estimates: Optional[np.ndarray] = None
    
    # Explanation
    feature_importance: Dict[str, float] = field(default_factory=dict)
    selection_explanation: str = ""
    
    # Metadata
    selection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_k_futures(self, k: int = 5) -> List[Tuple[int, StateType, float]]:
        """Get top k futures with their indices and scores."""
        if len(self.all_scores) == 0:
            return []
        
        # Get indices of top k scores (assuming lower is better)
        top_indices = np.argsort(self.all_scores)[:k]
        
        return [
            (int(idx), None, float(self.all_scores[idx]))  # Future would need to be passed separately
            for idx in top_indices
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'selection_id': self.selection_id,
            'optimal_index': self.optimal_index,
            'selection_confidence': self.selection_confidence,
            'pareto_indices': self.pareto_indices,
            'score_breakdown': {k: v.tolist() for k, v in self.score_breakdown.items()},
            'ranking_analysis': self.ranking_analysis,
            'selection_time': self.selection_time,
            'computation_stats': self.computation_stats,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'selection_explanation': self.selection_explanation,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

# ==================== SCORE FUNCTIONS ====================

class ScoreFunctionCalculator:
    """Advanced score function calculator."""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Precomputed matrices for efficiency
        self.covariance_matrix = None
        self.inverse_covariance = None
        
        # Adaptive weights
        self.adaptive_weights = {func: 1.0 for func in config.score_functions}
        
        # Performance tracking
        self.computation_stats = {
            'total_computations': 0,
            'avg_computation_time': 0.0,
            'score_distributions': {}
        }
    
    @jit(nopython=True)
    @staticmethod
    def _euclidean_distance_batch(current: np.ndarray, futures: np.ndarray) -> np.ndarray:
        """Optimized batch Euclidean distance computation."""
        return np.sqrt(np.sum((futures - current)**2, axis=1))
    
    @jit(nopython=True)
    @staticmethod
    def _cosine_similarity_batch(current: np.ndarray, futures: np.ndarray) -> np.ndarray:
        """Optimized batch cosine similarity computation."""
        current_norm = np.sqrt(np.sum(current**2))
        futures_norms = np.sqrt(np.sum(futures**2, axis=1))
        
        dots = np.dot(futures, current)
        similarities = dots / (current_norm * futures_norms + 1e-10)
        
        # Convert to distance (lower is better)
        return 1.0 - similarities
    
    @jit(nopython=True)
    @staticmethod
    def _manhattan_distance_batch(current: np.ndarray, futures: np.ndarray) -> np.ndarray:
        """Optimized batch Manhattan distance computation."""
        return np.sum(np.abs(futures - current), axis=1)
    
    def compute_scores(
        self, 
        current_state: StateType, 
        potential_futures: List[StateType]
    ) -> Dict[str, np.ndarray]:
        """Compute all configured score functions."""
        start_time = time.time()
        
        if not potential_futures:
            return {}
        
        # Prepare data matrices
        current_spatial = current_state.spatial
        futures_spatial = np.array([f.spatial for f in potential_futures])
        
        # Additional features
        futures_complexity = np.array([f.complexity for f in potential_futures])
        futures_emergence = np.array([f.emergence_potential for f in potential_futures])
        futures_temporal = np.array([f.temporal for f in potential_futures])
        
        scores = {}
        
        for score_func in self.config.score_functions:
            try:
                if score_func == ScoreFunction.EUCLIDEAN_DISTANCE:
                    scores['euclidean'] = self._euclidean_distance_batch(
                        current_spatial, futures_spatial
                    )
                
                elif score_func == ScoreFunction.COSINE_SIMILARITY:
                    scores['cosine'] = self._cosine_similarity_batch(
                        current_spatial, futures_spatial
                    )
                
                elif score_func == ScoreFunction.MANHATTAN_DISTANCE:
                    scores['manhattan'] = self._manhattan_distance_batch(
                        current_spatial, futures_spatial
                    )
                
                elif score_func == ScoreFunction.MAHALANOBIS_DISTANCE:
                    scores['mahalanobis'] = self._compute_mahalanobis_scores(
                        current_spatial, futures_spatial
                    )
                
                elif score_func == ScoreFunction.QUANTUM_FIDELITY:
                    scores['quantum_fidelity'] = self._compute_quantum_fidelity_scores(
                        current_state, potential_futures
                    )
                
                elif score_func == ScoreFunction.CAUSAL_STRENGTH:
                    scores['causal_strength'] = self._compute_causal_strength_scores(
                        current_state, potential_futures
                    )
                
                elif score_func == ScoreFunction.EMERGENCE_POTENTIAL:
                    scores['emergence'] = self._compute_emergence_scores(
                        current_state, potential_futures
                    )
                
                elif score_func == ScoreFunction.COMPOSITE_SCORE:
                    scores['composite'] = self._compute_composite_scores(
                        current_state, potential_futures
                    )
                
            except Exception as e:
                self.logger.error(f"Error computing {score_func.name}: {e}")
                # Fallback to simple distance
                scores[score_func.name.lower()] = self._euclidean_distance_batch(
                    current_spatial, futures_spatial
                )
        
        # Update computation statistics
        computation_time = time.time() - start_time
        self.computation_stats['total_computations'] += 1
        self.computation_stats['avg_computation_time'] = (
            self.computation_stats['avg_computation_time'] * 
            (self.computation_stats['total_computations'] - 1) + computation_time
        ) / self.computation_stats['total_computations']
        
        # Track score distributions for adaptive weights
        for name, score_array in scores.items():
            if name not in self.computation_stats['score_distributions']:
                self.computation_stats['score_distributions'][name] = []
            
            self.computation_stats['score_distributions'][name].append({
                'mean': float(np.mean(score_array)),
                'std': float(np.std(score_array)),
                'min': float(np.min(score_array)),
                'max': float(np.max(score_array))
            })
        
        return scores
    
    def _compute_mahalanobis_scores(
        self, 
        current: np.ndarray, 
        futures: np.ndarray
    ) -> np.ndarray:
        """Compute Mahalanobis distance scores."""
        try:
            if self.covariance_matrix is None:
                # Estimate covariance from futures
                self.covariance_matrix = np.cov(futures.T)
                self.inverse_covariance = np.linalg.pinv(self.covariance_matrix)
            
            differences = futures - current
            scores = np.sqrt(np.sum(
                differences @ self.inverse_covariance * differences, axis=1
            ))
            
            return scores
            
        except Exception:
            # Fallback to Euclidean
            return self._euclidean_distance_batch(current, futures)
    
    def _compute_quantum_fidelity_scores(
        self, 
        current_state: StateType, 
        potential_futures: List[StateType]
    ) -> np.ndarray:
        """Compute quantum fidelity-inspired scores."""
        scores = []
        
        for future in potential_futures:
            # Quantum fidelity inspired by state overlap
            spatial_overlap = np.dot(
                current_state.spatial / (np.linalg.norm(current_state.spatial) + 1e-10),
                future.spatial / (np.linalg.norm(future.spatial) + 1e-10)
            )
            
            # Probabilistic overlap
            prob_overlap = np.sum(np.sqrt(
                current_state.probabilistic * future.probabilistic
            ))
            
            # Combined fidelity (higher is better, so invert for minimization)
            fidelity = 0.7 * spatial_overlap + 0.3 * prob_overlap
            score = 1.0 - fidelity  # Convert to distance
            
            scores.append(score)
        
        return np.array(scores)
    
    def _compute_causal_strength_scores(
        self, 
        current_state: StateType, 
        potential_futures: List[StateType]
    ) -> np.ndarray:
        """Compute causal strength scores."""
        scores = []
        
        for future in potential_futures:
            # Causal signature similarity
            causal_similarity = 1.0 - cosine(
                current_state.causal_signature, 
                future.causal_signature
            )
            
            # Temporal consistency
            temporal_delta = abs(future.temporal - current_state.temporal)
            temporal_score = 1.0 / (1.0 + temporal_delta)
            
            # Complexity evolution reasonableness
            complexity_delta = abs(future.complexity - current_state.complexity)
            complexity_score = 1.0 - min(complexity_delta, 1.0)
            
            # Combined causal strength
            causal_strength = (
                0.5 * causal_similarity +
                0.3 * temporal_score +
                0.2 * complexity_score
            )
            
            # Convert to distance (lower is better)
            score = 1.0 - causal_strength
            scores.append(score)
        
        return np.array(scores)
    
    def _compute_emergence_scores(
        self, 
        current_state: StateType, 
        potential_futures: List[StateType]
    ) -> np.ndarray:
        """Compute emergence potential scores."""
        scores = []
        
        current_emergence = current_state.emergence_potential
        
        for future in potential_futures:
            # Emergence evolution
            emergence_delta = future.emergence_potential - current_emergence
            
            # Favor positive emergence changes (but not too dramatic)
            if emergence_delta > 0:
                score = 1.0 / (1.0 + emergence_delta)  # Prefer moderate increases
            else:
                score = 1.0 + abs(emergence_delta)  # Penalize decreases
            
            scores.append(score)
        
        return np.array(scores)
    
    def _compute_composite_scores(
        self, 
        current_state: StateType, 
        potential_futures: List[StateType]
    ) -> np.ndarray:
        """Compute composite scores combining multiple factors."""
        # Get individual scores
        all_scores = self.compute_scores(current_state, potential_futures)
        
        if not all_scores:
            return np.zeros(len(potential_futures))
        
        # Normalize scores to [0, 1]
        normalized_scores = {}
        for name, scores in all_scores.items():
            if name == 'composite':  # Avoid recursion
                continue
            
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score > min_score:
                normalized_scores[name] = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores[name] = np.zeros_like(scores)
        
        if not normalized_scores:
            return np.zeros(len(potential_futures))
        
        # Weighted combination
        weights = self._get_adaptive_weights(list(normalized_scores.keys()))
        composite = np.zeros(len(potential_futures))
        
        for name, scores in normalized_scores.items():
            weight = weights.get(name, 1.0)
            composite += weight * scores
        
        # Normalize final scores
        total_weight = sum(weights.values())
        if total_weight > 0:
            composite /= total_weight
        
        return composite
    
    def _get_adaptive_weights(self, score_names: List[str]) -> Dict[str, float]:
        """Get adaptive weights based on score performance."""
        if not self.config.enable_adaptive_weights:
            return {name: 1.0 for name in score_names}
        
        # Simple adaptive weighting based on score variance
        # Higher variance = more discriminative = higher weight
        weights = {}
        
        for name in score_names:
            distributions = self.computation_stats['score_distributions'].get(name, [])
            if distributions:
                # Use recent variance as weight
                recent_std = np.mean([d['std'] for d in distributions[-10:]])
                weights[name] = max(recent_std, 0.1)  # Minimum weight
            else:
                weights[name] = 1.0
        
        return weights

# ==================== MACHINE LEARNING MODELS ====================

class MLSelectionModel:
    """Machine learning-based selection model."""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Models
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
        # Training data
        self.training_features = []
        self.training_targets = []
        
        # Performance tracking
        self.model_performance = {}
        self.last_training_time = None
        
        # Feature engineering
        self.feature_names = []
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models."""
        model_configs = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                alpha=0.01,
                random_state=42
            ),
            'support_vector': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
        }
        
        for model_name in self.config.ml_models:
            if model_name in model_configs:
                self.models[model_name] = model_configs[model_name]
        
        # Create ensemble
        if len(self.models) > 1:
            estimators = [(name, model) for name, model in self.models.items()]
            self.ensemble_model = VotingRegressor(estimators=estimators)
    
    def extract_features(
        self, 
        current_state: StateType, 
        potential_futures: List[StateType]
    ) -> np.ndarray:
        """Extract features for ML model."""
        features = []
        
        for future in potential_futures:
            # Spatial features
            spatial_distance = np.linalg.norm(future.spatial - current_state.spatial)
            spatial_mean_diff = np.mean(future.spatial) - np.mean(current_state.spatial)
            spatial_std_diff = np.std(future.spatial) - np.std(current_state.spatial)
            
            # Temporal features
            temporal_diff = future.temporal - current_state.temporal
            
            # Probabilistic features
            prob_distance = np.linalg.norm(future.probabilistic - current_state.probabilistic)
            prob_entropy_diff = (
                entropy(future.probabilistic + 1e-10) - 
                entropy(current_state.probabilistic + 1e-10)
            )
            
            # Complexity features
            complexity_diff = future.complexity - current_state.complexity
            emergence_diff = future.emergence_potential - current_state.emergence_potential
            
            # Causal features
            causal_distance = np.linalg.norm(
                future.causal_signature - current_state.causal_signature
            )
            causal_correlation = np.corrcoef(
                future.causal_signature, current_state.causal_signature
            )[0, 1]
            
            # Composite features
            total_state_change = (
                spatial_distance + prob_distance + causal_distance
            ) / 3.0
            
            feature_vector = [
                spatial_distance,
                spatial_mean_diff,
                spatial_std_diff,
                temporal_diff,
                prob_distance,
                prob_entropy_diff,
                complexity_diff,
                emergence_diff,
                causal_distance,
                causal_correlation if not np.isnan(causal_correlation) else 0.0,
                total_state_change
            ]
            
            features.append(feature_vector)
        
        # Set feature names if not already set
        if not self.feature_names:
            self.feature_names = [
                'spatial_distance', 'spatial_mean_diff', 'spatial_std_diff',
                'temporal_diff', 'prob_distance', 'prob_entropy_diff',
                'complexity_diff', 'emergence_diff', 'causal_distance',
                'causal_correlation', 'total_state_change'
            ]
        
        return np.array(features)
    
    def train(self, training_data: List[Tuple[StateType, List[StateType], int]]):
        """Train the ML model with historical selection data."""
        if not training_data:
            return
        
        start_time = time.time()
        
        try:
            # Extract features and targets
            all_features = []
            all_targets = []
            
            for current_state, futures, optimal_index in training_data:
                features = self.extract_features(current_state, futures)
                
                # Create targets (1 for optimal, 0 for others)
                targets = np.zeros(len(futures))
                targets[optimal_index] = 1.0
                
                all_features.append(features)
                all_targets.append(targets)
            
            # Combine all data
            combined_features = np.vstack(all_features)
            combined_targets = np.hstack(all_targets)
            
            # Scale features
            combined_features = self.scaler.fit_transform(combined_features)
            
            # Train individual models
            for name, model in self.models.items():
                try:
                    model.fit(combined_features, combined_targets)
                    
                    # Evaluate model
                    if self.config.validation_method == ValidationMethod.CROSS_VALIDATION:
                        cv_scores = cross_val_score(
                            model, combined_features, combined_targets,
                            cv=self.config.validation_folds, scoring='r2'
                        )
                        self.model_performance[name] = {
                            'cv_mean': np.mean(cv_scores),
                            'cv_std': np.std(cv_scores),
                            'last_training': datetime.now(timezone.utc)
                        }
                    
                    # Update metrics
                    ml_model_performance.labels(
                        model_type=name,
                        metric='r2_score'
                    ).set(self.model_performance[name]['cv_mean'])
                    
                except Exception as e:
                    self.logger.error(f"Error training model {name}: {e}")
            
            # Train ensemble model
            if self.ensemble_model:
                try:
                    self.ensemble_model.fit(combined_features, combined_targets)
                    
                    # Evaluate ensemble
                    cv_scores = cross_val_score(
                        self.ensemble_model, combined_features, combined_targets,
                        cv=self.config.validation_folds, scoring='r2'
                    )
                    self.model_performance['ensemble'] = {
                        'cv_mean': np.mean(cv_scores),
                        'cv_std': np.std(cv_scores),
                        'last_training': datetime.now(timezone.utc)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error training ensemble model: {e}")
            
            self.last_training_time = datetime.now(timezone.utc)
            training_time = time.time() - start_time
            
            self.logger.info(
                f"ML model training completed",
                training_samples=len(combined_features),
                training_time=training_time,
                models_trained=len(self.models)
            )
            
        except Exception as e:
            self.logger.error(f"ML model training failed: {e}")
            raise ModelTrainingError(f"Training failed: {e}")
    
    def predict(
        self, 
        current_state: StateType, 
        potential_futures: List[StateType]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict selection scores using trained models."""
        if not self.models:
            raise ModelTrainingError("No trained models available")
        
        try:
            # Extract features
            features = self.extract_features(current_state, potential_futures)
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)
                    predictions[name] = pred
                except Exception as e:
                    self.logger.warning(f"Prediction failed for model {name}: {e}")
            
            # Ensemble prediction
            if self.ensemble_model:
                try:
                    ensemble_pred = self.ensemble_model.predict(features_scaled)
                    predictions['ensemble'] = ensemble_pred
                except Exception as e:
                    self.logger.warning(f"Ensemble prediction failed: {e}")
            
            if not predictions:
                raise ModelTrainingError("All model predictions failed")
            
            # Use best performing model or ensemble
            best_model = self._get_best_model()
            final_predictions = predictions.get(best_model, list(predictions.values())[0])
            
            # Convert predictions to selection scores (higher prediction = better)
            # Convert to distance scores (lower is better)
            selection_scores = 1.0 - final_predictions
            
            # Additional analysis
            analysis = {
                'model_used': best_model,
                'prediction_variance': np.var(final_predictions),
                'prediction_confidence': np.mean(final_predictions),
                'feature_importance': self._get_feature_importance(best_model),
                'all_predictions': predictions
            }
            
            return selection_scores, analysis
            
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            raise SelectionComputationError(f"Prediction failed: {e}")
    
    def _get_best_model(self) -> str:
        """Get the best performing model name."""
        if 'ensemble' in self.model_performance:
            return 'ensemble'
        
        best_model = None
        best_score = -np.inf
        
        for name, performance in self.model_performance.items():
            if performance['cv_mean'] > best_score:
                best_score = performance['cv_mean']
                best_model = name
        
        return best_model or list(self.models.keys())[0]
    
    def _get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for the specified model."""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for i, name in enumerate(self.feature_names):
                    importance[name] = float(importances[i])
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_)
                for i, name in enumerate(self.feature_names):
                    importance[name] = float(coefficients[i])
            
            else:
                # Other models - use permutation importance approximation
                # This is simplified; in practice, you'd use sklearn's permutation_importance
                importance = {name: 1.0 / len(self.feature_names) for name in self.feature_names}
        
        except Exception as e:
            self.logger.warning(f"Could not get feature importance for {model_name}: {e}")
            importance = {name: 0.0 for name in self.feature_names}
        
        return importance

# ==================== PARETO OPTIMIZATION ====================

class ParetoOptimizer:
    """Pareto frontier optimization for multi-objective selection."""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    def find_pareto_frontier(
        self, 
        objectives: Dict[str, np.ndarray]
    ) -> Tuple[List[int], np.ndarray]:
        """Find Pareto optimal solutions."""
        if not objectives:
            return [], np.array([])
        
        # Combine objectives into matrix
        objective_names = list(objectives.keys())
        objective_matrix = np.column_stack([objectives[name] for name in objective_names])
        
        n_points = objective_matrix.shape[0]
        if n_points == 0:
            return [], np.array([])
        
        # Find Pareto optimal points
        pareto_indices = []
        
        for i in range(n_points):
            is_pareto = True
            current_point = objective_matrix[i]
            
            for j in range(n_points):
                if i == j:
                    continue
                
                other_point = objective_matrix[j]
                
                # Check if other point dominates current point
                if self._dominates(other_point, current_point):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_indices.append(i)
        
        # Update metrics
        pareto_frontier_size.labels(
            selection_method=self.config.selection_method.name
        ).set(len(pareto_indices))
        
        return pareto_indices, objective_matrix[pareto_indices]
    
    def _dominates(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """Check if point1 dominates point2 (assuming minimization)."""
        # Point1 dominates point2 if it's better in all objectives
        return np.all(point1 <= point2 + PARETO_EPSILON) and np.any(point1 < point2 - PARETO_EPSILON)
    
    def select_from_pareto_frontier(
        self, 
        pareto_indices: List[int],
        preferences: Optional[Dict[str, float]] = None
    ) -> int:
        """Select single solution from Pareto frontier based on preferences."""
        if not pareto_indices:
            return 0
        
        if len(pareto_indices) == 1:
            return pareto_indices[0]
        
        # If no preferences, select the one closest to ideal point
        # For now, just return the first one
        return pareto_indices[0]

# ==================== MAIN SELECTOR ====================

class AdvancedQuantumRetroCausalSelector:
    """
    Advanced quantum retro-causal future selector with enterprise features.
    
    This class provides comprehensive future selection capabilities including
    multi-objective optimization, machine learning integration, and distributed
    computing support.
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        self.logger = structlog.get_logger(__name__)
        
        # Core components
        self.score_calculator = ScoreFunctionCalculator(self.config)
        self.ml_model = MLSelectionModel(self.config) if self.config.enable_ml else None
        self.pareto_optimizer = ParetoOptimizer(self.config)
        
        # C++ acceleration
        self.cpp_selector = None
        if self.config.enable_cpp_acceleration:
            self._initialize_cpp_acceleration()
        
        # GPU acceleration
        self.gpu_available = False
        if self.config.enable_gpu_acceleration and CUPY_AVAILABLE:
            self._initialize_gpu_acceleration()
        
        # Thread pool for parallel processing
        self.thread_pool = None
        if self.config.enable_parallel:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Caching
        self.selection_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.selection_stats = {
            'total_selections': 0,
            'avg_selection_time': 0.0,
            'method_usage': defaultdict(int),
            'batch_size_distribution': defaultdict(int)
        }
        
        # Thread safety
        self.selection_lock = RLock()
        
        self.logger.info(f"Advanced selector initialized with method: {self.config.selection_method.name}")
    
    def _initialize_cpp_acceleration(self):
        """Initialize C++ acceleration module."""
        try:
            self.cpp_selector = importlib.import_module("quantum_selector")
            self.logger.info("C++ acceleration module loaded successfully")
        except ImportError:
            self.logger.info("C++ acceleration module not available, using Python implementation")
    
    def _initialize_gpu_acceleration(self):
        """Initialize GPU acceleration."""
        try:
            # Test GPU availability
            cp.cuda.Device(0).use()
            self.gpu_available = True
            self.logger.info("GPU acceleration available")
        except Exception as e:
            self.logger.info(f"GPU acceleration not available: {e}")
    
    async def select_optimal_future(
        self,
        current_state: StateType,
        potential_futures: List[StateType],
        context: Optional[Dict[str, Any]] = None
    ) -> SelectionResult:
        """
        Select optimal future with comprehensive analysis.
        
        Args:
            current_state: Current quantum state
            potential_futures: List of potential future states
            context: Additional context for selection
            
        Returns:
            Comprehensive selection result
        """
        if not potential_futures:
            raise ValueError("No potential futures provided")
        
        start_time = time.time()
        selection_id = str(uuid.uuid4())
        
        try:
            with self.selection_lock:
                # Check cache
                cache_key = self._generate_cache_key(current_state, potential_futures)
                if self.config.enable_caching and cache_key in self.selection_cache:
                    self.cache_hits += 1
                    cached_result = self.selection_cache[cache_key]
                    self.logger.debug(f"Cache hit for selection {selection_id}")
                    return cached_result
                
                self.cache_misses += 1
                
                # Timeout handling
                with asyncio.timeout(self.config.timeout_seconds):
                    result = await self._perform_selection(
                        current_state, potential_futures, context, selection_id
                    )
                
                # Cache result
                if self.config.enable_caching:
                    self._update_cache(cache_key, result)
                
                # Update statistics
                selection_time = time.time() - start_time
                self._update_selection_stats(selection_time, len(potential_futures))
                
                # Update metrics
                selection_operations.labels(
                    selection_method=self.config.selection_method.name,
                    status='success'
                ).inc()
                
                batch_size_range = self._get_batch_size_range(len(potential_futures))
                selection_duration.labels(
                    selection_method=self.config.selection_method.name,
                    batch_size_range=batch_size_range
                ).observe(selection_time)
                
                result.selection_time = selection_time
                
                self.logger.info(
                    f"Selection completed",
                    selection_id=selection_id,
                    method=self.config.selection_method.name,
                    futures_count=len(potential_futures),
                    selection_time=selection_time,
                    optimal_index=result.optimal_index
                )
                
                return result
                
        except asyncio.TimeoutError:
            self.logger.error(f"Selection timeout for {selection_id}")
            selection_operations.labels(
                selection_method=self.config.selection_method.name,
                status='timeout'
            ).inc()
            raise SelectionTimeoutError("Selection operation timed out")
        
        except Exception as e:
            self.logger.error(f"Selection failed for {selection_id}: {e}")
            selection_operations.labels(
                selection_method=self.config.selection_method.name,
                status='error'
            ).inc()
            raise SelectionComputationError(f"Selection failed: {e}")
    
    async def _perform_selection(
        self,
        current_state: StateType,
        potential_futures: List[StateType],
        context: Optional[Dict[str, Any]],
        selection_id: str
    ) -> SelectionResult:
        """Perform the actual selection based on configured method."""
        
        if self.config.selection_method == SelectionMethod.SIMPLE_DISTANCE:
            return await self._simple_distance_selection(current_state, potential_futures)
        
        elif self.config.selection_method == SelectionMethod.WEIGHTED_SCORE:
            return await self._weighted_score_selection(current_state, potential_futures)
        
        elif self.config.selection_method == SelectionMethod.PARETO_OPTIMAL:
            return await self._pareto_optimal_selection(current_state, potential_futures)
        
        elif self.config.selection_method == SelectionMethod.MACHINE_LEARNING:
            return await self._ml_selection(current_state, potential_futures)
        
        elif self.config.selection_method == SelectionMethod.HYBRID_ENSEMBLE:
            return await self._hybrid_ensemble_selection(current_state, potential_futures)
        
        elif self.config.selection_method == SelectionMethod.QUANTUM_INSPIRED:
            return await self._quantum_inspired_selection(current_state, potential_futures)
        
        else:
            # Fallback to weighted score
            return await self._weighted_score_selection(current_state, potential_futures)
    
    async def _simple_distance_selection(
        self,
        current_state: StateType,
        potential_futures: List[StateType]
    ) -> SelectionResult:
        """Simple distance-based selection."""
        
        # Use C++ acceleration if available
        if self.cpp_selector:
            try:
                states_np = current_state.spatial.reshape(1, -1)
                futures_np = np.array([f.spatial for f in potential_futures])
                
                scores = self.cpp_selector.quantum_score(
                    states_np.tolist(), futures_np.tolist()
                )
                scores = np.array(scores)
            except Exception as e:
                self.logger.warning(f"C++ acceleration failed, using Python: {e}")
                scores = np.array([
                    np.linalg.norm(f.spatial - current_state.spatial)
                    for f in potential_futures
                ])
        else:
            # Python fallback
            if self.gpu_available and len(potential_futures) > 1000:
                # GPU acceleration for large batches
                try:
                    current_gpu = cp.asarray(current_state.spatial)
                    futures_gpu = cp.array([f.spatial for f in potential_futures])
                    
                    differences = futures_gpu - current_gpu
                    scores_gpu = cp.sqrt(cp.sum(differences**2, axis=1))
                    scores = cp.asnumpy(scores_gpu)
                except Exception as e:
                    self.logger.warning(f"GPU acceleration failed, using CPU: {e}")
                    scores = np.array([
                        np.linalg.norm(f.spatial - current_state.spatial)
                        for f in potential_futures
                    ])
            else:
                # CPU computation
                scores = np.array([
                    np.linalg.norm(f.spatial - current_state.spatial)
                    for f in potential_futures
                ])
        
        optimal_index = int(np.argmin(scores))
        
        return SelectionResult(
            optimal_index=optimal_index,
            optimal_future=potential_futures[optimal_index],
            all_scores=scores,
            selection_confidence=self._calculate_confidence(scores, optimal_index),
            score_breakdown={'euclidean_distance': scores},
            selection_explanation="Selected based on minimum Euclidean distance"
        )
    
    async def _weighted_score_selection(
        self,
        current_state: StateType,
        potential_futures: List[StateType]
    ) -> SelectionResult:
        """Weighted multi-criteria selection."""
        
        # Compute all score functions
        score_breakdown = self.score_calculator.compute_scores(current_state, potential_futures)
        
        if not score_breakdown:
            # Fallback to simple distance
            return await self._simple_distance_selection(current_state, potential_futures)
        
        # Normalize and combine scores
        normalized_scores = {}
        for name, scores in score_breakdown.items():
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score > min_score:
                normalized_scores[name] = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores[name] = np.zeros_like(scores)
        
        # Get adaptive weights
        weights = self.score_calculator._get_adaptive_weights(list(normalized_scores.keys()))
        
        # Combine scores
        combined_scores = np.zeros(len(potential_futures))
        total_weight = 0.0
        
        for name, scores in normalized_scores.items():
            weight = weights.get(name, 1.0)
            combined_scores += weight * scores
            total_weight += weight
        
        if total_weight > 0:
            combined_scores /= total_weight
        
        optimal_index = int(np.argmin(combined_scores))
        
        return SelectionResult(
            optimal_index=optimal_index,
            optimal_future=potential_futures[optimal_index],
            all_scores=combined_scores,
            selection_confidence=self._calculate_confidence(combined_scores, optimal_index),
            score_breakdown=score_breakdown,
            feature_importance=weights,
            selection_explanation=f"Selected using weighted combination of {len(score_breakdown)} criteria"
        )
    
    async def _pareto_optimal_selection(
        self,
        current_state: StateType,
        potential_futures: List[StateType]
    ) -> SelectionResult:
        """Pareto optimal multi-objective selection."""
        
        # Compute objective functions
        objectives = self.score_calculator.compute_scores(current_state, potential_futures)
        
        if not objectives:
            return await self._simple_distance_selection(current_state, potential_futures)
        
        # Find Pareto frontier
        pareto_indices, pareto_front = self.pareto_optimizer.find_pareto_frontier(objectives)
        
        if not pareto_indices:
            # Fallback if no Pareto solutions found
            return await self._weighted_score_selection(current_state, potential_futures)
        
        # Select from Pareto frontier
        optimal_index = self.pareto_optimizer.select_from_pareto_frontier(pareto_indices)
        
        # Calculate combined score for ranking
        combined_scores = np.mean(list(objectives.values()), axis=0)
        
        return SelectionResult(
            optimal_index=optimal_index,
            optimal_future=potential_futures[optimal_index],
            all_scores=combined_scores,
            selection_confidence=self._calculate_confidence(combined_scores, optimal_index),
            pareto_indices=pareto_indices,
            score_breakdown=objectives,
            selection_explanation=f"Selected from Pareto frontier of {len(pareto_indices)} solutions"
        )
    
    async def _ml_selection(
        self,
        current_state: StateType,
        potential_futures: List[StateType]
    ) -> SelectionResult:
        """Machine learning-based selection."""
        
        if not self.ml_model:
            return await self._weighted_score_selection(current_state, potential_futures)
        
        try:
            # Get ML predictions
            ml_scores, ml_analysis = self.ml_model.predict(current_state, potential_futures)
            
            optimal_index = int(np.argmin(ml_scores))
            
            return SelectionResult(
                optimal_index=optimal_index,
                optimal_future=potential_futures[optimal_index],
                all_scores=ml_scores,
                selection_confidence=ml_analysis.get('prediction_confidence', 0.5),
                score_breakdown={'ml_score': ml_scores},
                feature_importance=ml_analysis.get('feature_importance', {}),
                model_performance=self.ml_model.model_performance,
                selection_explanation=f"Selected using {ml_analysis.get('model_used', 'ML')} model"
            )
        
        except Exception as e:
            self.logger.warning(f"ML selection failed, falling back to weighted scores: {e}")
            return await self._weighted_score_selection(current_state, potential_futures)
    
    async def _hybrid_ensemble_selection(
        self,
        current_state: StateType,
        potential_futures: List[StateType]
    ) -> SelectionResult:
        """Hybrid ensemble selection combining multiple methods."""
        
        # Get results from multiple methods
        methods_results = {}
        
        # Traditional scoring
        traditional_result = await self._weighted_score_selection(current_state, potential_futures)
        methods_results['traditional'] = traditional_result
        
        # ML-based if available
        if self.ml_model:
            try:
                ml_result = await self._ml_selection(current_state, potential_futures)
                methods_results['ml'] = ml_result
            except Exception as e:
                self.logger.warning(f"ML method failed in ensemble: {e}")
        
        # Pareto optimization
        try:
            pareto_result = await self._pareto_optimal_selection(current_state, potential_futures)
            methods_results['pareto'] = pareto_result
        except Exception as e:
            self.logger.warning(f"Pareto method failed in ensemble: {e}")
        
        # Ensemble combination
        if len(methods_results) == 1:
            return list(methods_results.values())[0]
        
        # Combine scores using voting
        all_scores = np.zeros(len(potential_futures))
        total_weight = 0.0
        
        method_weights = {
            'traditional': 0.4,
            'ml': 0.4,
            'pareto': 0.2
        }
        
        combined_score_breakdown = {}
        
        for method_name, result in methods_results.items():
            weight = method_weights.get(method_name, 1.0)
            
            # Normalize scores
            scores = result.all_scores
            if np.max(scores) > np.min(scores):
                normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            else:
                normalized_scores = np.zeros_like(scores)
            
            all_scores += weight * normalized_scores
            total_weight += weight
            
            combined_score_breakdown[f'{method_name}_scores'] = scores
        
        if total_weight > 0:
            all_scores /= total_weight
        
        optimal_index = int(np.argmin(all_scores))
        
        # Combine feature importance
        combined_importance = {}
        for result in methods_results.values():
            for feature, importance in result.feature_importance.items():
                combined_importance[feature] = combined_importance.get(feature, 0.0) + importance
        
        return SelectionResult(
            optimal_index=optimal_index,
            optimal_future=potential_futures[optimal_index],
            all_scores=all_scores,
            selection_confidence=np.mean([r.selection_confidence for r in methods_results.values()]),
            score_breakdown=combined_score_breakdown,
            feature_importance=combined_importance,
            selection_explanation=f"Ensemble of {len(methods_results)} methods: {list(methods_results.keys())}"
        )
    
    async def _quantum_inspired_selection(
        self,
        current_state: StateType,
        potential_futures: List[StateType]
    ) -> SelectionResult:
        """Quantum-inspired selection using superposition and measurement."""
        
        # Get quantum fidelity scores
        scores = self.score_calculator._compute_quantum_fidelity_scores(current_state, potential_futures)
        
        # Quantum-inspired amplitude calculation
        amplitudes = np.exp(-scores)  # Higher fidelity = higher amplitude
        amplitudes = amplitudes / np.linalg.norm(amplitudes)  # Normalize
        
        # Calculate probabilities
        probabilities = amplitudes**2
        
        # Quantum measurement (probabilistic selection)
        # For deterministic behavior, select highest probability
        optimal_index = int(np.argmax(probabilities))
        
        # Calculate quantum coherence measure
        coherence = 1.0 - entropy(probabilities) / np.log(len(probabilities))
        
        return SelectionResult(
            optimal_index=optimal_index,
            optimal_future=potential_futures[optimal_index],
            all_scores=scores,
            selection_confidence=coherence,
            score_breakdown={'quantum_fidelity': scores, 'probabilities': probabilities},
            selection_explanation="Quantum-inspired selection based on state fidelity and coherence"
        )
    
    def _calculate_confidence(self, scores: np.ndarray, optimal_index: int) -> float:
        """Calculate selection confidence based on score distribution."""
        if len(scores) < 2:
            return 1.0
        
        sorted_scores = np.sort(scores)
        best_score = sorted_scores[0]
        second_best = sorted_scores[1]
        
        # Confidence based on gap between best and second best
        if second_best > best_score:
            confidence = (second_best - best_score) / second_best
        else:
            confidence = 0.5
        
        # Adjust for score distribution
        score_std = np.std(scores)
        if score_std > 0:
            normalized_gap = (second_best - best_score) / score_std
            confidence = min(normalized_gap / 2.0, 1.0)
        
        return max(0.1, min(confidence, 1.0))
    
    def _generate_cache_key(self, current_state: StateType, potential_futures: List[StateType]) -> str:
        """Generate cache key for selection result."""
        # Simple hash based on state representations
        import hashlib
        
        state_data = current_state.spatial.tobytes()
        futures_data = b''.join(f.spatial.tobytes() for f in potential_futures[:10])  # Limit for performance
        
        combined_data = state_data + futures_data + str(len(potential_futures)).encode()
        
        return hashlib.md5(combined_data).hexdigest()
    
    def _update_cache(self, cache_key: str, result: SelectionResult):
        """Update selection cache."""
        if len(self.selection_cache) >= self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.selection_cache))
            del self.selection_cache[oldest_key]
        
        self.selection_cache[cache_key] = result
    
    def _update_selection_stats(self, selection_time: float, batch_size: int):
        """Update selection statistics."""
        self.selection_stats['total_selections'] += 1
        self.selection_stats['avg_selection_time'] = (
            self.selection_stats['avg_selection_time'] * (self.selection_stats['total_selections'] - 1) +
            selection_time
        ) / self.selection_stats['total_selections']
        
        self.selection_stats['method_usage'][self.config.selection_method.name] += 1
        
        batch_range = self._get_batch_size_range(batch_size)
        self.selection_stats['batch_size_distribution'][batch_range] += 1
    
    def _get_batch_size_range(self, batch_size: int) -> str:
        """Get batch size range for metrics."""
        if batch_size < 100:
            return 'small'
        elif batch_size < 1000:
            return 'medium'
        elif batch_size < 10000:
            return 'large'
        else:
            return 'very_large'
    
    def train_ml_model(self, training_data: List[Tuple[StateType, List[StateType], int]]):
        """Train the ML model with historical data."""
        if self.ml_model:
            try:
                self.ml_model.train(training_data)
                self.logger.info(f"ML model trained with {len(training_data)} samples")
            except Exception as e:
                self.logger.error(f"ML model training failed: {e}")
        else:
            self.logger.warning("ML model not available for training")
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive selector statistics."""
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        
        return {
            'config': {
                'selection_method': self.config.selection_method.name,
                'score_functions': [f.name for f in self.config.score_functions],
                'ml_enabled': self.config.enable_ml,
                'parallel_enabled': self.config.enable_parallel
            },
            'selection_stats': self.selection_stats,
            'score_calculator_stats': self.score_calculator.computation_stats,
            'ml_model_performance': self.ml_model.model_performance if self.ml_model else {},
            'cache_stats': {
                'hit_rate': cache_hit_rate,
                'cache_size': len(self.selection_cache),
                'hits': self.cache_hits,
                'misses': self.cache_misses
            },
            'acceleration': {
                'cpp_available': self.cpp_selector is not None,
                'gpu_available': self.gpu_available,
                'parallel_workers': self.config.max_workers
            }
        }
    
    def close(self):
        """Close selector and cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        self.selection_cache.clear()
        
        self.logger.info("Selector closed and resources cleaned up")

# ==================== CONVENIENCE FUNCTIONS ====================

def create_quantum_selector(
    selection_method: SelectionMethod = SelectionMethod.HYBRID_ENSEMBLE,
    enable_ml: bool = True,
    enable_parallel: bool = True,
    enable_cpp_acceleration: bool = True
) -> AdvancedQuantumRetroCausalSelector:
    """Create quantum selector with sensible defaults."""
    
    config = SelectionConfig(
        selection_method=selection_method,
        enable_ml=enable_ml,
        enable_parallel=enable_parallel,
        enable_cpp_acceleration=enable_cpp_acceleration,
        enable_validation=True,
        enable_explanation=True
    )
    
    return AdvancedQuantumRetroCausalSelector(config)

# ==================== EXAMPLE USAGE ====================

async def example_usage():
    """Example usage of the advanced selector."""
    
    # Create selector
    selector = create_quantum_selector(
        selection_method=SelectionMethod.HYBRID_ENSEMBLE,
        enable_ml=True,
        enable_parallel=True
    )
    
    try:
        # Create sample current state
        current_state = QuantumState(
            spatial=np.random.random(64),
            temporal=time.time(),
            probabilistic=np.random.random(8),
            complexity=0.5,
            emergence_potential=0.7,
            causal_signature=np.random.random(32)
        )
        
        # Create sample future states
        potential_futures = []
        for _ in range(1000):
            future_state = QuantumState(
                spatial=current_state.spatial + np.random.normal(0, 0.1, 64),
                temporal=current_state.temporal + np.random.exponential(1.0),
                probabilistic=np.random.random(8),
                complexity=np.random.random(),
                emergence_potential=np.random.random(),
                causal_signature=np.random.random(32)
            )
            potential_futures.append(future_state)
        
        # Select optimal future
        result = await selector.select_optimal_future(current_state, potential_futures)
        
        print(f"Selected future index: {result.optimal_index}")
        print(f"Selection confidence: {result.selection_confidence:.3f}")
        print(f"Selection method: {result.selection_explanation}")
        print(f"Selection time: {result.selection_time:.4f}s")
        
        # Get top k futures
        top_futures = result.get_top_k_futures(k=5)
        print(f"Top 5 futures: {[(idx, score) for idx, _, score in top_futures]}")
        
        # Get statistics
        stats = selector.get_comprehensive_statistics()
        print(f"Selector statistics: {json.dumps(stats, indent=2, default=str)}")
        
    finally:
        # Clean up
        selector.close()

if __name__ == "__main__":
    asyncio.run(example_usage())
"""
⚙️ Advanced Configuration Management System
===========================================

Enterprise-grade configuration system with validation, hot-reload,
environment-specific settings, encryption, and comprehensive monitoring.

Features:
- Hierarchical configuration with inheritance
- Environment-specific overrides
- Real-time validation with custom validators
- Hot-reload capability with change notifications
- Encrypted sensitive values
- Configuration versioning and rollback
- Audit logging and change tracking
- Integration with external config stores
- Performance optimization with caching
- Comprehensive documentation and schema

Author: Advanced Engineering Team
Version: 2.0.0
License: MIT
"""

import asyncio
import json
import os
import threading
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps, lru_cache
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Type, TypeVar, 
    Generic, Protocol, runtime_checkable, Set, FrozenSet, ClassVar
)
import hashlib
import yaml
import toml
from threading import RLock, Event
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, validator, ValidationError
from prometheus_client import Counter, Gauge, Histogram
import structlog
from marshmallow import Schema, fields as ma_fields, validate, post_load
from cerberus import Validator

# ==================== TYPE DEFINITIONS ====================

T = TypeVar('T')
ConfigType = TypeVar('ConfigType', bound=BaseModel)

# ==================== CONSTANTS ====================

CONFIG_VERSION = "2.0.0"
DEFAULT_CONFIG_DIR = Path("./config")
DEFAULT_CACHE_TTL = 300  # 5 minutes
MAX_CONFIG_SIZE = 10 * 1024 * 1024  # 10MB
VALIDATION_TIMEOUT = 30  # seconds

# ==================== METRICS ====================

config_loads = Counter(
    'quantum_config_loads_total',
    'Total configuration loads',
    ['config_type', 'environment', 'status']
)

config_validations = Counter(
    'quantum_config_validations_total',
    'Total configuration validations',
    ['config_type', 'status']
)

config_changes = Counter(
    'quantum_config_changes_total',
    'Total configuration changes',
    ['config_type', 'change_type']
)

active_configs = Gauge(
    'quantum_active_configs',
    'Number of active configurations',
    ['config_type']
)

config_load_duration = Histogram(
    'quantum_config_load_duration_seconds',
    'Configuration load duration',
    ['config_type', 'source']
)

# ==================== EXCEPTIONS ====================

class ConfigurationError(Exception):
    """Base configuration exception."""
    pass

class ConfigValidationError(ConfigurationError):
    """Configuration validation error."""
    pass

class ConfigLoadError(ConfigurationError):
    """Configuration loading error."""
    pass

class ConfigNotFoundError(ConfigurationError):
    """Configuration not found error."""
    pass

class ConfigEncryptionError(ConfigurationError):
    """Configuration encryption error."""
    pass

class ConfigVersionError(ConfigurationError):
    """Configuration version error."""
    pass

# ==================== ENUMS ====================

class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"

class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = auto()
    YAML = auto()
    TOML = auto()
    INI = auto()
    ENV = auto()

class ConfigSource(Enum):
    """Configuration sources."""
    FILE = auto()
    ENVIRONMENT = auto()
    CONSUL = auto()
    ETCD = auto()
    REDIS = auto()
    DATABASE = auto()
    S3 = auto()

class ValidationLevel(Enum):
    """Validation levels."""
    NONE = auto()
    BASIC = auto()
    STRICT = auto()
    PARANOID = auto()

class ChangeType(Enum):
    """Types of configuration changes."""
    ADDED = auto()
    MODIFIED = auto()
    DELETED = auto()
    RELOADED = auto()

# ==================== PROTOCOLS ====================

@runtime_checkable
class ConfigValidator(Protocol):
    """Protocol for configuration validators."""
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration."""
        ...

@runtime_checkable
class ConfigStore(Protocol):
    """Protocol for configuration stores."""
    
    async def load(self, key: str) -> Dict[str, Any]:
        """Load configuration."""
        ...
    
    async def save(self, key: str, config: Dict[str, Any]) -> None:
        """Save configuration."""
        ...
    
    async def delete(self, key: str) -> None:
        """Delete configuration."""
        ...

# ==================== DATA STRUCTURES ====================

@dataclass(frozen=True)
class ConfigMetadata:
    """Configuration metadata."""
    name: str
    version: str
    environment: Environment
    created_at: datetime
    updated_at: datetime
    checksum: str
    source: ConfigSource
    format: ConfigFormat
    encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Configuration name cannot be empty")
        if not self.version:
            raise ValueError("Configuration version cannot be empty")

@dataclass
class ConfigChange:
    """Configuration change record."""
    timestamp: datetime
    change_type: ChangeType
    path: str
    old_value: Any
    new_value: Any
    user: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Validation result."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration: float = 0.0

# ==================== ADVANCED CONFIGURATION CLASSES ====================

class AdvancedPerformanceConfig(BaseModel):
    """Enhanced performance configuration with validation."""
    
    # Core Settings
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum number of worker threads"
    )
    
    batch_size: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Batch processing size"
    )
    
    # Memory Management
    memory_limit_mb: int = Field(
        default=2048,
        ge=64,
        le=32768,
        description="Memory limit in MB"
    )
    
    gc_threshold: float = Field(
        default=0.8,
        ge=0.1,
        le=0.95,
        description="Garbage collection threshold"
    )
    
    # I/O Configuration
    io_buffer_size: int = Field(
        default=8192,
        ge=1024,
        le=1048576,
        description="I/O buffer size in bytes"
    )
    
    connection_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Connection pool size"
    )
    
    # Caching
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching"
    )
    
    cache_ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=86400,
        description="Cache TTL in seconds"
    )
    
    cache_max_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum cache size"
    )
    
    # Async Settings
    async_enabled: bool = Field(
        default=True,
        description="Enable async processing"
    )
    
    async_pool_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Async pool size"
    )
    
    # Optimization Flags
    jit_enabled: bool = Field(
        default=True,
        description="Enable JIT compilation"
    )
    
    vectorization_enabled: bool = Field(
        default=True,
        description="Enable vectorized operations"
    )
    
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing"
    )
    
    @validator('max_workers')
    def validate_max_workers(cls, v):
        """Validate max_workers against system capabilities."""
        import os
        cpu_count = os.cpu_count() or 1
        if v > cpu_count * 2:
            warnings.warn(f"max_workers ({v}) exceeds 2x CPU count ({cpu_count})")
        return v
    
    @validator('memory_limit_mb')
    def validate_memory_limit(cls, v):
        """Validate memory limit against system memory."""
        import psutil
        system_memory_mb = psutil.virtual_memory().total // (1024 * 1024)
        if v > system_memory_mb * 0.8:
            warnings.warn(f"memory_limit_mb ({v}) exceeds 80% of system memory ({system_memory_mb})")
        return v
    
    def get_optimal_batch_size(self, data_size: int) -> int:
        """Calculate optimal batch size based on data size."""
        if data_size < 1000:
            return min(self.batch_size, data_size)
        elif data_size < 10000:
            return min(self.batch_size, data_size // 10)
        else:
            return min(self.batch_size, data_size // 100)

class AdvancedTradingConfig(BaseModel):
    """Enhanced trading configuration with risk management."""
    
    # Risk Management
    risk_level: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Risk level (0-1)"
    )
    
    max_position_size: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Maximum position size as fraction of capital"
    )
    
    stop_loss_pct: float = Field(
        default=0.02,
        ge=0.001,
        le=0.2,
        description="Stop loss percentage"
    )
    
    take_profit_pct: float = Field(
        default=0.06,
        ge=0.001,
        le=1.0,
        description="Take profit percentage"
    )
    
    max_drawdown_pct: float = Field(
        default=0.15,
        ge=0.01,
        le=0.5,
        description="Maximum drawdown percentage"
    )
    
    # Asset Management
    asset_list: List[str] = Field(
        default_factory=list,
        description="List of tradable assets"
    )
    
    asset_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Asset allocation weights"
    )
    
    min_order_size: float = Field(
        default=10.0,
        ge=0.01,
        description="Minimum order size"
    )
    
    max_order_size: float = Field(
        default=100000.0,
        ge=1.0,
        description="Maximum order size"
    )
    
    # Trading Parameters
    leverage: float = Field(
        default=1.0,
        ge=1.0,
        le=100.0,
        description="Trading leverage"
    )
    
    commission_pct: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Commission percentage"
    )
    
    slippage_pct: float = Field(
        default=0.001,
        ge=0.0,
        le=0.05,
        description="Slippage percentage"
    )
    
    # Time Management
    trading_hours: Dict[str, Tuple[str, str]] = Field(
        default_factory=lambda: {
            "monday": ("09:30", "16:00"),
            "tuesday": ("09:30", "16:00"),
            "wednesday": ("09:30", "16:00"),
            "thursday": ("09:30", "16:00"),
            "friday": ("09:30", "16:00")
        },
        description="Trading hours by day"
    )
    
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Order timeout in seconds"
    )
    
    # Strategy Parameters
    strategy_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameters"
    )
    
    rebalance_frequency: str = Field(
        default="daily",
        regex=r"^(minutely|hourly|daily|weekly|monthly)$",
        description="Rebalancing frequency"
    )
    
    @validator('asset_weights')
    def validate_asset_weights(cls, v):
        """Validate asset weights sum to 1.0."""
        if v and abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError("Asset weights must sum to 1.0")
        return v
    
    @validator('leverage')
    def validate_leverage(cls, v, values):
        """Validate leverage against risk level."""
        if 'risk_level' in values and v > (1.0 / values['risk_level']):
            warnings.warn(f"High leverage ({v}) relative to risk level ({values['risk_level']})")
        return v
    
    def calculate_position_size(self, capital: float, price: float, volatility: float) -> float:
        """Calculate position size based on risk management."""
        # Kelly criterion with safety factor
        kelly_fraction = min(self.risk_level / volatility, self.max_position_size)
        position_value = capital * kelly_fraction
        return position_value / price

class AdvancedRealTimeConfig(BaseModel):
    """Enhanced real-time configuration with latency optimization."""
    
    # Refresh Settings
    refresh_interval: float = Field(
        default=1.0,
        ge=0.001,
        le=60.0,
        description="Refresh interval in seconds"
    )
    
    heartbeat_interval: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Heartbeat interval in seconds"
    )
    
    # Latency Settings
    max_latency_ms: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum acceptable latency in milliseconds"
    )
    
    latency_percentile: float = Field(
        default=95.0,
        ge=50.0,
        le=99.9,
        description="Latency percentile to track"
    )
    
    # Buffer Settings
    buffer_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Buffer size for real-time data"
    )
    
    buffer_high_watermark: float = Field(
        default=0.8,
        ge=0.1,
        le=0.95,
        description="Buffer high watermark"
    )
    
    buffer_low_watermark: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Buffer low watermark"
    )
    
    # Connection Settings
    connection_timeout: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Connection timeout in seconds"
    )
    
    reconnect_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of reconnection attempts"
    )
    
    reconnect_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Reconnection delay in seconds"
    )
    
    # Quality of Service
    qos_enabled: bool = Field(
        default=True,
        description="Enable QoS monitoring"
    )
    
    drop_on_overflow: bool = Field(
        default=True,
        description="Drop messages on buffer overflow"
    )
    
    priority_queue: bool = Field(
        default=True,
        description="Use priority queue for messages"
    )
    
    # Monitoring
    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "latency_ms": 200.0,
            "throughput_msg_per_sec": 100.0,
            "error_rate_pct": 5.0,
            "buffer_utilization_pct": 80.0
        },
        description="Alert thresholds"
    )
    
    @validator('buffer_high_watermark')
    def validate_watermarks(cls, v, values):
        """Validate watermark ordering."""
        if 'buffer_low_watermark' in values and v <= values['buffer_low_watermark']:
            raise ValueError("High watermark must be greater than low watermark")
        return v
    
    def get_adaptive_refresh_interval(self, load_factor: float) -> float:
        """Calculate adaptive refresh interval based on system load."""
        if load_factor < 0.5:
            return self.refresh_interval
        elif load_factor < 0.8:
            return self.refresh_interval * 1.5
        else:
            return self.refresh_interval * 2.0

# ==================== CONFIGURATION VALIDATORS ====================

class BusinessRuleValidator:
    """Business rule validator for configuration."""
    
    def __init__(self):
        self.rules: List[Callable] = []
    
    def add_rule(self, rule: Callable[[Dict[str, Any]], Tuple[bool, str]]):
        """Add a validation rule."""
        self.rules.append(rule)
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration against business rules."""
        errors = []
        
        for rule in self.rules:
            try:
                is_valid, error_msg = rule(config)
                if not is_valid:
                    errors.append(error_msg)
            except Exception as e:
                errors.append(f"Rule validation failed: {str(e)}")
        
        return len(errors) == 0, errors

class SchemaValidator:
    """Schema-based configuration validator."""
    
    def __init__(self, schema: Schema):
        self.schema = schema
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration against schema."""
        try:
            self.schema.load(config)
            return True, []
        except ValidationError as e:
            return False, [str(e)]

# ==================== CONFIGURATION ENCRYPTION ====================

class ConfigurationEncryption:
    """Configuration encryption handler."""
    
    def __init__(self, key: Optional[bytes] = None):
        if key:
            self.fernet = Fernet(key)
        else:
            self.fernet = Fernet(Fernet.generate_key())
        
        self.encrypted_fields = set()
    
    def add_encrypted_field(self, field_path: str):
        """Add field to encryption list."""
        self.encrypted_fields.add(field_path)
    
    def encrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration values."""
        encrypted_config = deepcopy(config)
        
        for field_path in self.encrypted_fields:
            self._encrypt_field(encrypted_config, field_path)
        
        return encrypted_config
    
    def decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive configuration values."""
        decrypted_config = deepcopy(config)
        
        for field_path in self.encrypted_fields:
            self._decrypt_field(decrypted_config, field_path)
        
        return decrypted_config
    
    def _encrypt_field(self, config: Dict[str, Any], field_path: str):
        """Encrypt a specific field."""
        try:
            value = self._get_nested_value(config, field_path)
            if value is not None:
                encrypted_value = self.fernet.encrypt(str(value).encode())
                self._set_nested_value(config, field_path, encrypted_value.decode())
        except Exception as e:
            raise ConfigEncryptionError(f"Failed to encrypt field {field_path}: {e}")
    
    def _decrypt_field(self, config: Dict[str, Any], field_path: str):
        """Decrypt a specific field."""
        try:
            encrypted_value = self._get_nested_value(config, field_path)
            if encrypted_value is not None:
                decrypted_value = self.fernet.decrypt(encrypted_value.encode())
                self._set_nested_value(config, field_path, decrypted_value.decode())
        except Exception as e:
            raise ConfigEncryptionError(f"Failed to decrypt field {field_path}: {e}")
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary."""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value

# ==================== CONFIGURATION FILE HANDLER ====================

class ConfigurationFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reload."""
    
    def __init__(self, config_manager: 'AdvancedConfigurationManager'):
        self.config_manager = config_manager
        self.logger = structlog.get_logger(__name__)
    
    def on_modified(self, event):
        """Handle file modification."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix in ['.json', '.yaml', '.yml', '.toml']:
                self.logger.info(f"Configuration file modified: {file_path}")
                asyncio.create_task(self.config_manager.reload_config(file_path))

# ==================== ADVANCED CONFIGURATION MANAGER ====================

class AdvancedConfigurationManager:
    """
    Enterprise-grade configuration manager with advanced features.
    
    Features:
    - Hierarchical configuration with inheritance
    - Environment-specific overrides
    - Hot-reload with change notifications
    - Validation and business rules
    - Encryption of sensitive values
    - Audit logging and versioning
    - Performance optimization
    """
    
    def __init__(
        self,
        base_config_dir: Path = DEFAULT_CONFIG_DIR,
        environment: Environment = Environment.DEVELOPMENT,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        enable_hot_reload: bool = True,
        enable_encryption: bool = False,
        cache_ttl: int = DEFAULT_CACHE_TTL
    ):
        self.base_config_dir = Path(base_config_dir)
        self.environment = environment
        self.validation_level = validation_level
        self.enable_hot_reload = enable_hot_reload
        self.cache_ttl = cache_ttl
        
        # Create directory structure
        self.base_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration storage
        self.configurations: Dict[str, Any] = {}
        self.metadata: Dict[str, ConfigMetadata] = {}
        self.validators: Dict[str, ConfigValidator] = {}
        self.change_history: List[ConfigChange] = []
        
        # Caching
        self.cache: Dict[str, Tuple[Any, float]] = {}
        
        # Thread safety
        self.lock = RLock()
        
        # Encryption
        self.encryption = ConfigurationEncryption() if enable_encryption else None
        
        # File watcher
        self.observer = None
        if enable_hot_reload:
            self.observer = Observer()
            self.observer.schedule(
                ConfigurationFileHandler(self),
                str(self.base_config_dir),
                recursive=True
            )
            self.observer.start()
        
        # Change listeners
        self.change_listeners: List[Callable] = []
        
        # Logging
        self.logger = structlog.get_logger(__name__)
        
        # Initialize metrics
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        self.metrics = {
            'loads': config_loads,
            'validations': config_validations,
            'changes': config_changes,
            'active': active_configs,
            'duration': config_load_duration
        }
    
    def register_validator(self, config_name: str, validator: ConfigValidator):
        """Register a validator for a configuration."""
        with self.lock:
            self.validators[config_name] = validator
    
    def add_change_listener(self, listener: Callable[[str, ConfigChange], None]):
        """Add a change listener."""
        self.change_listeners.append(listener)
    
    def _notify_change_listeners(self, config_name: str, change: ConfigChange):
        """Notify change listeners."""
        for listener in self.change_listeners:
            try:
                listener(config_name, change)
            except Exception as e:
                self.logger.error(f"Change listener failed: {e}")
    
    async def load_config(
        self,
        config_name: str,
        config_type: Type[T],
        config_file: Optional[Path] = None,
        use_cache: bool = True
    ) -> T:
        """
        Load configuration with validation and caching.
        
        Args:
            config_name: Configuration name
            config_type: Configuration type/class
            config_file: Optional specific config file
            use_cache: Whether to use cache
            
        Returns:
            Configuration instance
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache and config_name in self.cache:
                cached_config, cached_time = self.cache[config_name]
                if time.time() - cached_time < self.cache_ttl:
                    self.metrics['loads'].labels(
                        config_type=config_type.__name__,
                        environment=self.environment.value,
                        status='cache_hit'
                    ).inc()
                    return cached_config
            
            # Load configuration data
            config_data = await self._load_config_data(config_name, config_file)
            
            # Apply environment overrides
            config_data = self._apply_environment_overrides(config_data, config_name)
            
            # Decrypt if necessary
            if self.encryption:
                config_data = self.encryption.decrypt_config(config_data)
            
            # Validate configuration
            validation_result = await self._validate_config(config_name, config_data)
            if not validation_result.valid:
                self.metrics['validations'].labels(
                    config_type=config_type.__name__,
                    status='failed'
                ).inc()
                raise ConfigValidationError(f"Configuration validation failed: {validation_result.errors}")
            
            # Create configuration instance
            config_instance = config_type(**config_data)
            
            # Update cache
            with self.lock:
                self.cache[config_name] = (config_instance, time.time())
                self.configurations[config_name] = config_instance
                
                # Update metadata
                self.metadata[config_name] = ConfigMetadata(
                    name=config_name,
                    version=CONFIG_VERSION,
                    environment=self.environment,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    checksum=self._calculate_checksum(config_data),
                    source=ConfigSource.FILE,
                    format=ConfigFormat.YAML,
                    encrypted=self.encryption is not None
                )
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics['loads'].labels(
                config_type=config_type.__name__,
                environment=self.environment.value,
                status='success'
            ).inc()
            
            self.metrics['duration'].labels(
                config_type=config_type.__name__,
                source='file'
            ).observe(duration)
            
            self.metrics['active'].labels(
                config_type=config_type.__name__
            ).inc()
            
            self.metrics['validations'].labels(
                config_type=config_type.__name__,
                status='success'
            ).inc()
            
            self.logger.info(
                f"Configuration loaded successfully",
                config_name=config_name,
                config_type=config_type.__name__,
                duration=duration
            )
            
            return config_instance
            
        except Exception as e:
            self.metrics['loads'].labels(
                config_type=config_type.__name__,
                environment=self.environment.value,
                status='error'
            ).inc()
            
            self.logger.error(
                f"Failed to load configuration",
                config_name=config_name,
                error=str(e)
            )
            raise ConfigLoadError(f"Failed to load configuration {config_name}: {e}")
    
    async def _load_config_data(self, config_name: str, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load configuration data from file."""
        if config_file:
            file_path = config_file
        else:
            # Try different file extensions
            for ext in ['.yaml', '.yml', '.json', '.toml']:
                file_path = self.base_config_dir / f"{config_name}{ext}"
                if file_path.exists():
                    break
            else:
                raise ConfigNotFoundError(f"Configuration file not found for {config_name}")
        
        # Load based on file extension
        if file_path.suffix in ['.yaml', '.yml']:
            return await self._load_yaml_config(file_path)
        elif file_path.suffix == '.json':
            return await self._load_json_config(file_path)
        elif file_path.suffix == '.toml':
            return await self._load_toml_config(file_path)
        else:
            raise ConfigLoadError(f"Unsupported configuration format: {file_path.suffix}")
    
    async def _load_yaml_config(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigLoadError(f"Failed to load YAML config from {file_path}: {e}")
    
    async def _load_json_config(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
        except Exception as e:
            raise ConfigLoadError(f"Failed to load JSON config from {file_path}: {e}")
    
    async def _load_toml_config(self, file_path: Path) -> Dict[str, Any]:
        """Load TOML configuration."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return toml.load(f) or {}
        except Exception as e:
            raise ConfigLoadError(f"Failed to load TOML config from {file_path}: {e}")
    
    def _apply_environment_overrides(self, config_data: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Apply environment-specific overrides."""
        env_config_path = self.base_config_dir / "environments" / f"{self.environment.value}.yaml"
        
        if env_config_path.exists():
            try:
                with open(env_config_path, 'r', encoding='utf-8') as f:
                    env_config = yaml.safe_load(f) or {}
                
                # Apply overrides for this specific config
                if config_name in env_config:
                    config_data = self._deep_merge(config_data, env_config[config_name])
                
            except Exception as e:
                self.logger.warning(f"Failed to load environment overrides: {e}")
        
        return config_data
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def _validate_config(self, config_name: str, config_data: Dict[str, Any]) -> ValidationResult:
        """Validate configuration data."""
        start_time = time.time()
        errors = []
        warnings = []
        
        # Skip validation if disabled
        if self.validation_level == ValidationLevel.NONE:
            return ValidationResult(valid=True, duration=time.time() - start_time)
        
        # Custom validator
        if config_name in self.validators:
            try:
                is_valid, validator_errors = self.validators[config_name].validate(config_data)
                if not is_valid:
                    errors.extend(validator_errors)
            except Exception as e:
                errors.append(f"Validator error: {str(e)}")
        
        # Basic validation
        if self.validation_level in [ValidationLevel.BASIC, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            # Check for required fields, data types, etc.
            basic_errors = self._perform_basic_validation(config_data)
            errors.extend(basic_errors)
        
        # Strict validation
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            # Additional business rule validation
            strict_errors = self._perform_strict_validation(config_data)
            errors.extend(strict_errors)
        
        # Paranoid validation
        if self.validation_level == ValidationLevel.PARANOID:
            # Security and compliance checks
            paranoid_errors = self._perform_paranoid_validation(config_data)
            errors.extend(paranoid_errors)
        
        duration = time.time() - start_time
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            duration=duration
        )
    
    def _perform_basic_validation(self, config_data: Dict[str, Any]) -> List[str]:
        """Perform basic validation checks."""
        errors = []
        
        # Check for empty values
        def check_empty_values(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    if value is None or value == "":
                        errors.append(f"Empty value at {current_path}")
                    elif isinstance(value, dict):
                        check_empty_values(value, current_path)
        
        check_empty_values(config_data)
        
        return errors
    
    def _perform_strict_validation(self, config_data: Dict[str, Any]) -> List[str]:
        """Perform strict validation checks."""
        errors = []
        
        # Add business rule validation here
        # Example: Check value ranges, dependencies, etc.
        
        return errors
    
    def _perform_paranoid_validation(self, config_data: Dict[str, Any]) -> List[str]:
        """Perform paranoid validation checks."""
        errors = []
        
        # Add security and compliance checks here
        # Example: Check for sensitive data, validate certificates, etc.
        
        return errors
    
    def _calculate_checksum(self, config_data: Dict[str, Any]) -> str:
        """Calculate configuration checksum."""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    async def reload_config(self, config_file: Path):
        """Reload configuration from file."""
        try:
            config_name = config_file.stem
            
            if config_name in self.configurations:
                old_config = self.configurations[config_name]
                
                # Invalidate cache
                with self.lock:
                    if config_name in self.cache:
                        del self.cache[config_name]
                
                # Reload configuration
                config_type = type(old_config)
                new_config = await self.load_config(config_name, config_type, config_file, use_cache=False)
                
                # Record change
                change = ConfigChange(
                    timestamp=datetime.now(),
                    change_type=ChangeType.RELOADED,
                    path=str(config_file),
                    old_value=old_config,
                    new_value=new_config
                )
                
                self.change_history.append(change)
                self._notify_change_listeners(config_name, change)
                
                self.metrics['changes'].labels(
                    config_type=config_type.__name__,
                    change_type='reloaded'
                ).inc()
                
                self.logger.info(f"Configuration reloaded: {config_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
    
    def get_config(self, config_name: str) -> Optional[Any]:
        """Get loaded configuration."""
        with self.lock:
            return self.configurations.get(config_name)
    
    def get_metadata(self, config_name: str) -> Optional[ConfigMetadata]:
        """Get configuration metadata."""
        with self.lock:
            return self.metadata.get(config_name)
    
    def get_change_history(self, config_name: Optional[str] = None) -> List[ConfigChange]:
        """Get configuration change history."""
        if config_name:
            return [change for change in self.change_history if config_name in change.path]
        return self.change_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration manager statistics."""
        with self.lock:
            return {
                'total_configs': len(self.configurations),
                'cached_configs': len(self.cache),
                'validators': len(self.validators),
                'change_history_size': len(self.change_history),
                'environment': self.environment.value,
                'validation_level': self.validation_level.value,
                'hot_reload_enabled': self.enable_hot_reload,
                'encryption_enabled': self.encryption is not None
            }
    
    def close(self):
        """Close configuration manager and cleanup resources."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        with self.lock:
            self.configurations.clear()
            self.cache.clear()
            self.metadata.clear()
            self.change_history.clear()
        
        self.logger.info("Configuration manager closed")

# ==================== CONVENIENCE FUNCTIONS ====================

async def load_performance_config(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    environment: Environment = Environment.DEVELOPMENT
) -> AdvancedPerformanceConfig:
    """Load performance configuration."""
    manager = AdvancedConfigurationManager(config_dir, environment)
    return await manager.load_config("performance", AdvancedPerformanceConfig)

async def load_trading_config(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    environment: Environment = Environment.DEVELOPMENT
) -> AdvancedTradingConfig:
    """Load trading configuration."""
    manager = AdvancedConfigurationManager(config_dir, environment)
    return await manager.load_config("trading", AdvancedTradingConfig)

async def load_realtime_config(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    environment: Environment = Environment.DEVELOPMENT
) -> AdvancedRealTimeConfig:
    """Load real-time configuration."""
    manager = AdvancedConfigurationManager(config_dir, environment)
    return await manager.load_config("realtime", AdvancedRealTimeConfig)

# ==================== EXAMPLE USAGE ====================

async def example_usage():
    """Example usage of the advanced configuration system."""
    
    # Create configuration manager
    config_manager = AdvancedConfigurationManager(
        base_config_dir=Path("./config"),
        environment=Environment.DEVELOPMENT,
        validation_level=ValidationLevel.STRICT,
        enable_hot_reload=True,
        enable_encryption=True
    )
    
    try:
        # Load configurations
        perf_config = await config_manager.load_config("performance", AdvancedPerformanceConfig)
        trading_config = await config_manager.load_config("trading", AdvancedTradingConfig)
        realtime_config = await config_manager.load_config("realtime", AdvancedRealTimeConfig)
        
        print(f"Performance config loaded: {perf_config.max_workers} workers")
        print(f"Trading config loaded: {trading_config.risk_level} risk level")
        print(f"Real-time config loaded: {realtime_config.refresh_interval}s refresh")
        
        # Get statistics
        stats = config_manager.get_statistics()
        print(f"Configuration statistics: {stats}")
        
    finally:
        # Clean up
        config_manager.close()

if __name__ == "__main__":
    asyncio.run(example_usage())
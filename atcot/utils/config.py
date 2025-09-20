"""
Configuration management for the ATCOT framework.

This module provides configuration classes and utilities for setting up
and managing ATCOT framework parameters.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM interface."""
    provider: str = "mock"
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolConfig:
    """Configuration for individual tools."""
    enabled: bool = True
    config_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanningConfig:
    """Configuration for adaptive planning."""
    max_planning_attempts: int = 3
    confidence_threshold: float = 0.3
    dependency_validation: bool = True
    fallback_strategy: str = "simple"


@dataclass
class ExecutionConfig:
    """Configuration for tool execution."""
    max_candidates_per_step: int = 3
    parallel_execution: bool = True
    timeout_seconds: float = 30.0
    retry_failed_tools: bool = True
    max_retries: int = 2


@dataclass
class CorrectionConfig:
    """Configuration for correction mechanism."""
    contradiction_threshold: float = 0.7
    max_corrections: int = 5
    improvement_threshold: float = 0.01
    revision_strategy: str = "backward_traversal"
    enable_global_consistency: bool = True


@dataclass
class ConsistencyConfig:
    """Configuration for consistency checking."""
    similarity_method: str = "simple"
    contradiction_method: str = "simple"
    semantic_threshold: float = 0.3
    enable_numeric_checking: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file_path: Optional[str] = None


@dataclass
class ATCOTConfig:
    """Main configuration class for the ATCOT framework."""
    
    # Core configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Tool configurations
    tools: Dict[str, ToolConfig] = field(default_factory=dict)
    
    # Framework settings
    framework_version: str = "1.0.0"
    debug_mode: bool = False
    save_execution_trace: bool = False
    trace_output_dir: Optional[str] = None

    def __post_init__(self):
        """Initialize default tool configurations."""
        if not self.tools:
            self.tools = {
                "calculator": ToolConfig(enabled=True),
                "web_search": ToolConfig(enabled=True, config_params={"search_engine": "duckduckgo"}),
                "python_interpreter": ToolConfig(enabled=True),
                "wikipedia": ToolConfig(enabled=True, config_params={"language": "en"})
            }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ATCOTConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        config = cls()
        
        if "llm" in config_dict:
            config.llm = LLMConfig(**config_dict["llm"])
        
        if "planning" in config_dict:
            config.planning = PlanningConfig(**config_dict["planning"])
        
        if "execution" in config_dict:
            config.execution = ExecutionConfig(**config_dict["execution"])
        
        if "correction" in config_dict:
            config.correction = CorrectionConfig(**config_dict["correction"])
        
        if "consistency" in config_dict:
            config.consistency = ConsistencyConfig(**config_dict["consistency"])
        
        if "logging_config" in config_dict:
            config.logging_config = LoggingConfig(**config_dict["logging_config"])
        
        if "tools" in config_dict:
            config.tools = {
                name: ToolConfig(**tool_config) 
                for name, tool_config in config_dict["tools"].items()
            }
        
        # Set other attributes
        for key, value in config_dict.items():
            if key not in ["llm", "planning", "execution", "correction", "consistency", "logging_config", "tools"]:
                setattr(config, key, value)
        
        return config

    @classmethod
    def from_file(cls, config_path: str) -> 'ATCOTConfig':
        """Load configuration from JSON file."""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return cls()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()

    @classmethod
    def from_env(cls) -> 'ATCOTConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # LLM configuration from environment
        if os.getenv("ATCOT_LLM_PROVIDER"):
            config.llm.provider = os.getenv("ATCOT_LLM_PROVIDER")
        
        if os.getenv("ATCOT_LLM_MODEL"):
            config.llm.model_name = os.getenv("ATCOT_LLM_MODEL")
        
        if os.getenv("ATCOT_LLM_API_KEY"):
            config.llm.api_key = os.getenv("ATCOT_LLM_API_KEY")
        
        if os.getenv("OPENAI_API_KEY") and config.llm.provider == "openai":
            config.llm.api_key = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("ANTHROPIC_API_KEY") and config.llm.provider == "anthropic":
            config.llm.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Framework settings
        if os.getenv("ATCOT_DEBUG_MODE"):
            config.debug_mode = os.getenv("ATCOT_DEBUG_MODE").lower() == "true"
        
        if os.getenv("ATCOT_MAX_CORRECTIONS"):
            try:
                config.correction.max_corrections = int(os.getenv("ATCOT_MAX_CORRECTIONS"))
            except ValueError:
                logger.warning("Invalid ATCOT_MAX_CORRECTIONS value, using default")
        
        if os.getenv("ATCOT_CONTRADICTION_THRESHOLD"):
            try:
                config.correction.contradiction_threshold = float(os.getenv("ATCOT_CONTRADICTION_THRESHOLD"))
            except ValueError:
                logger.warning("Invalid ATCOT_CONTRADICTION_THRESHOLD value, using default")
        
        # Logging configuration
        if os.getenv("ATCOT_LOG_LEVEL"):
            config.logging_config.level = os.getenv("ATCOT_LOG_LEVEL")
        
        if os.getenv("ATCOT_LOG_FILE"):
            config.logging_config.log_to_file = True
            config.logging_config.log_file_path = os.getenv("ATCOT_LOG_FILE")
        
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")

    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tool names."""
        return [name for name, config in self.tools.items() if config.enabled]

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled."""
        return self.tools.get(tool_name, ToolConfig()).enabled

    def get_tool_config(self, tool_name: str) -> ToolConfig:
        """Get configuration for a specific tool."""
        return self.tools.get(tool_name, ToolConfig())

    def update_tool_config(self, tool_name: str, config: ToolConfig) -> None:
        """Update configuration for a specific tool."""
        self.tools[tool_name] = config

    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_level = getattr(logging, self.logging_config.level.upper(), logging.INFO)
        
        # Configure basic logging
        logging.basicConfig(
            level=log_level,
            format=self.logging_config.format,
            force=True
        )
        
        # Add file handler if requested
        if self.logging_config.log_to_file and self.logging_config.log_file_path:
            try:
                log_file_path = Path(self.logging_config.log_file_path)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(log_level)
                file_handler.setFormatter(logging.Formatter(self.logging_config.format))
                
                # Add to root logger
                logging.getLogger().addHandler(file_handler)
                
                logger.info(f"Logging to file: {log_file_path}")
                
            except Exception as e:
                logger.error(f"Failed to setup file logging: {e}")

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate LLM configuration
        if self.llm.provider not in ["openai", "anthropic", "mock"]:
            issues.append(f"Unsupported LLM provider: {self.llm.provider}")
        
        if self.llm.provider in ["openai", "anthropic"] and not self.llm.api_key:
            issues.append(f"API key required for {self.llm.provider}")
        
        # Validate thresholds
        if not 0.0 <= self.correction.contradiction_threshold <= 1.0:
            issues.append("Contradiction threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.consistency.semantic_threshold <= 1.0:
            issues.append("Semantic threshold must be between 0.0 and 1.0")
        
        # Validate numeric settings
        if self.correction.max_corrections < 1:
            issues.append("Max corrections must be at least 1")
        
        if self.planning.max_planning_attempts < 1:
            issues.append("Max planning attempts must be at least 1")
        
        if self.execution.max_candidates_per_step < 1:
            issues.append("Max candidates per step must be at least 1")
        
        return issues

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ATCOTConfig(llm={self.llm.provider}, tools={len(self.tools)}, debug={self.debug_mode})"


def create_default_config() -> ATCOTConfig:
    """Create default ATCOT configuration."""
    return ATCOTConfig()


def load_config(config_path: Optional[str] = None, 
               use_env: bool = True) -> ATCOTConfig:
    """
    Load ATCOT configuration from various sources.
    
    Args:
        config_path: Path to JSON config file (optional)
        use_env: Whether to use environment variables
    
    Returns:
        ATCOTConfig instance
    """
    if config_path:
        config = ATCOTConfig.from_file(config_path)
    else:
        config = ATCOTConfig()
    
    if use_env:
        env_config = ATCOTConfig.from_env()
        # Merge environment config (env takes precedence)
        config.llm = env_config.llm
        config.debug_mode = env_config.debug_mode
        config.correction.max_corrections = env_config.correction.max_corrections
        config.correction.contradiction_threshold = env_config.correction.contradiction_threshold
        config.logging_config = env_config.logging_config
    
    # Validate configuration
    issues = config.validate()
    if issues:
        logger.warning(f"Configuration issues found: {issues}")
    
    # Setup logging
    config.setup_logging()
    
    return config

"""
Config Manager — Singleton for Application Configuration
===========================================================
Reads YAML config files and merges with CLI arguments.
Uses Singleton pattern to ensure one shared config instance.
"""

import os
from typing import Any, Dict, Optional
import yaml


class ConfigManager:
    """Singleton config manager for the application.
    
    Singleton ensures all modules share the same config state.
    
    Usage:
        config = ConfigManager()
        config.load("configs/train.yaml")
        lr = config.get("training.lr", default=0.01)
    """

    _instance = None
    _config: Dict = {}

    def __new__(cls):
        """Singleton: create instance only once, reuse after that."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to a .yaml config file.
        
        Returns:
            Parsed config dictionary.
        
        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}

        print(f"[ConfigManager] Loaded config from: {config_path}")
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key, supports dot-separated nested keys.
        
        Example: get("training.lr") -> config["training"]["lr"]
        
        Args:
            key: Config key (supports nested via dots).
            default: Default value if key not found.
        
        Returns:
            Config value or default.
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def merge_args(self, args: dict) -> None:
        """Merge CLI arguments into config (CLI overrides YAML).
        
        Priority: CLI args > YAML file > defaults.
        
        Args:
            args: Dictionary from argparse (Namespace.__dict__).
        """
        for key, value in args.items():
            if value is not None:
                self._config[key] = value

    @property
    def config(self) -> Dict[str, Any]:
        """Return the full config dictionary."""
        return self._config

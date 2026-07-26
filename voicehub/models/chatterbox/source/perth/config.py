"""
Configuration handling for Perth library.
"""

import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """
    Configuration management for Perth.

    Handles loading, saving, and accessing configuration settings for the library.
    """

    # Default configuration values
    _defaults = {
        # General settings
        "general": {
            "default_watermarker": "perth",
            "verbose": True,
        },
        # Perth-Net settings
        "perth": {
            "device": "cpu",
            "run_name": "implicit",
            "models_dir": None,  # Will be set to default location in __init__
        },
        # Audio processing settings
        "audio": {
            "default_sample_rate": 44100,
            "normalize": True,
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with default values and optional user config.

        Args:
            config_path: Path to a YAML configuration file to load
        """
        # Deep copy the defaults
        self._config = {}
        for section, values in self._defaults.items():
            self._config[section] = values.copy()

        # Set default models directory
        self._config["perth"]["models_dir"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "perth",
            "perth_net",
            "pretrained",
        )

        # Load user config if provided
        if config_path and os.path.exists(config_path):
            self.load(config_path)

    def load(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to a YAML configuration file
        """
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)

            # Merge with current config
            if user_config:
                for section, values in user_config.items():
                    if section in self._config:
                        self._config[section].update(values)
                    else:
                        self._config[section] = values
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    def save(self, config_path: str) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            config_path: Path to save the configuration to
        """
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            default: Default value to return if key is not found

        Returns:
            Configuration value or default
        """
        if section in self._config and key in self._config[section]:
            return self._config[section][key]
        return default

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Configuration section name

        Returns:
            Dictionary of configuration values for the section
        """
        return self._config.get(section, {}).copy()

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        return yaml.dump(self._config, default_flow_style=False)


# Singleton instance for global access
_config_instance = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to a configuration file to load

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    elif config_path:
        _config_instance.load(config_path)
    return _config_instance

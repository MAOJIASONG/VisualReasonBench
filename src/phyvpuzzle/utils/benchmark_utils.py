import yaml
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The configuration as a dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

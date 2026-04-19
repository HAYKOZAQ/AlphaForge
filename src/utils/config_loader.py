import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

def load_config():
    """Loads the global settings.yaml from the config directory and initializes environment variables."""
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Load environment variables from .env
    load_dotenv(project_root / ".env")
    
    config_path = project_root / "config" / "settings.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

def get_path(relative_path):
    """Utility to resolve absolute paths relative to the project root."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / relative_path

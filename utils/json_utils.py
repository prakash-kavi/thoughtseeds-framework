import os
import json
from typing import Dict, Any, Optional

def load_json(filepath: str, default=None) -> Optional[Dict[str, Any]]:
    """Load data from a JSON file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return default
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return default

def save_json(filepath: str, data: Dict[str, Any]) -> bool:
    """Save data to a JSON file with error handling"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")
        return False
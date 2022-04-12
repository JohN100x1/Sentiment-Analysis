import json
from pathlib import Path


def load_json(path_json: Path) -> dict:
    """Load the contents of JSON file as a dict specified by path_json."""
    with open(path_json) as f:
        return json.load(f)

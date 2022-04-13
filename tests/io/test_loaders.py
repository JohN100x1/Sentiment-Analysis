from json import JSONDecodeError
from pathlib import Path

import pytest

from reddit_analysis.io.loaders import load_json


class TestLoadJson:
    """Test load_json."""

    def test_success(self):
        path_json = Path(__file__).parent / "fake_file.json"
        assert load_json(path_json) == {"foo": "bar"}

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            assert load_json(Path("foo.json"))

    def test_invalid_file(self):
        with pytest.raises(JSONDecodeError):
            assert load_json(Path(__file__))

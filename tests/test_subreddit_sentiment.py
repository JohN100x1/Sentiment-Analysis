from pathlib import Path

from reddit_analysis.subreddit_sentiment import get_reddit_object


def test_get_reddit_object():
    path_json = Path(__file__).parent / "test_json.json"
    reddit = get_reddit_object(path_json=path_json)

    assert reddit.config.client_id == "foo_id"
    assert reddit.config.client_secret == "foo_secret"
    assert reddit.config.user_agent == "moo"
    assert reddit.config.username == "woo"
    assert reddit.config.password == "foobar"

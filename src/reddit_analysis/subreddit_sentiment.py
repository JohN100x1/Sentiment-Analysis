import json
import re
from pathlib import Path

import numpy as np
from praw import Reddit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from tensorflow import keras


def get_reddit_object(path_json: Path) -> Reddit:
    """
    Fetches details and returns reddit object.

    Parameters
    ----------
    path_json : Path, filename of details (json)

    Returns
    -------
    reddit : reddit object
    """
    with open(path_json) as f:
        data = json.load(f)
    reddit = Reddit(
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        user_agent=data["user_agent"],
        username=data["username"],
        password=data["password"],
    )
    return reddit


def get_top_sub_headlines(reddit: Reddit, sub: str, limit=1000):
    """
    Get subreddit top submission titles

    Parameters
    ----------
    reddit : Reddit object
    sub : str, subreddit name
    limit : int, top post limit

    Returns
    -------
    headlines : set, a set of top headlines from subreddit
    """
    headlines = []
    for submission in reddit.subreddit(sub).top(limit=limit):
        headlines.append(submission.title)
    return headlines


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    return text


class SentimentModel:
    def __init__(self, reddit, subs, limit=1000, val_ratio=0.1):
        self.subs = subs

        val_limit = int(limit * val_ratio)
        train_limit = limit - val_limit

        self.y_train = np.zeros((len(subs) * train_limit, len(subs)))
        self.y_val = np.zeros((len(subs) * val_limit, len(subs)))

        self.y_train1d = np.zeros(len(subs) * train_limit)
        self.y_val1d = np.zeros(len(subs) * val_limit)

        self.headline_train = []
        self.headline_val = []

        for i, sub in enumerate(self.subs):
            headlines = get_top_sub_headlines(reddit, sub, limit=limit)

            self.y_train[i * train_limit : (i + 1) * train_limit, i] = 1
            self.y_val[i * val_limit : (i + 1) * val_limit, i] = 1

            self.y_train1d[i * train_limit : (i + 1) * train_limit] = i
            self.y_val1d[i * val_limit : (i + 1) * val_limit] = i

            self.headline_train += headlines[:train_limit]
            self.headline_val += headlines[train_limit:]

        self.count_vec = CountVectorizer(
            preprocessor=preprocess_text, stop_words="english"
        )
        self.count_vec.fit(self.headline_train)

        # Trim to size
        self.y_val = self.y_val[: len(self.headline_val), :]
        self.y_val1d = self.y_val1d[: len(self.headline_val)]

        self.X_train = self.count_vec.transform(self.headline_train).toarray()
        self.X_val = self.count_vec.transform(self.headline_val).toarray()

        self.y_train = np.array(self.y_train)
        self.y_val = np.array(self.y_val)

    def train_keras_sequential(self):
        model_keras_sequential = keras.Sequential(
            [
                keras.layers.Dense(
                    50,
                    input_shape=(len(self.count_vec.vocabulary_),),
                    activation="relu",
                ),
                keras.layers.Dense(25, activation="relu"),
                keras.layers.Dense(len(self.subs), activation="sigmoid"),
            ]
        )
        model_keras_sequential.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        model_keras_sequential.fit(
            self.X_train, self.y_train, batch_size=32, epochs=10
        )

        expected = self.y_val
        predicted = model_keras_sequential.predict(self.X_val)
        print(sum((expected - predicted) <= 0.01) / len(expected))

    def train_naive_bayes(self):
        model_naive_bayes = GaussianNB()
        model_naive_bayes.fit(self.X_train, self.y_train1d)

        expected = self.y_val1d
        predicted = model_naive_bayes.predict(self.X_val)
        print(sum((expected - predicted) <= 0.01) / len(expected))

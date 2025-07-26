from sklearn.neighbors import KNeighborsClassifier
from typing import List
import numpy as np


class FAQClassifier:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="hamming",
            weights="distance",
        )
        self.answers = []

    def train(self, X: np.ndarray, answers: List[str]):
        self.answers = answers
        y = np.arange(len(answers))
        assert (
            X.dtype == np.bool_ or np.array_equal(X, X.astype(bool))
        ), "Input X must be binary"
        X = X.astype(bool)  # enforce bool dtype
        self.model.fit(X, y)

    def predict(self, query_vec: np.ndarray) -> str:
        query_vec = query_vec.astype(bool)
        idx = self.model.predict([query_vec])[0]
        return self.answers[idx]

    def predict_proba(self, query_vec: np.ndarray, top_k=3):
        query_vec = query_vec.astype(bool)
        distances, indices = self.model.kneighbors(
            [query_vec], n_neighbors=top_k
        )
        scores = 1 - distances[0]  # convert hamming distance to similarity
        results = [
            (self.answers[i], float(scores[j]))
            for j, i in enumerate(indices[0])
        ]
        return sorted(results, key=lambda x: -x[1])

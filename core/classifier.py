from sklearn.neighbors import KNeighborsClassifier
from typing import List
import numpy as np


class FAQClassifier:
    def __init__(self, n_neighbors=3):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="hamming",
        )
        self.answers = []

    def train(self, X: np.ndarray, answers: List[str]):
        self.answers = answers
        y = np.arange(len(answers))  # Each answer gets a unique label
        self.model.fit(X, y)

    def predict(self, query_vec: np.ndarray) -> str:
        idx = self.model.predict([query_vec])[0]
        return self.answers[idx]

    def predict_proba(self, query_vec: np.ndarray, top_k=3):
        distances, indices = self.model.kneighbors(
            [query_vec], n_neighbors=top_k
        )
        scores = 1 - distances[0]
        results = [
            (self.answers[i], float(scores[j]))
            for j, i in enumerate(indices[0])
        ]
        return sorted(results, key=lambda x: -x[1])
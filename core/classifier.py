from sklearn.neighbors import KNeighborsClassifier
from typing import List
import numpy as np


class FAQClassifier:
    def __init__(self, n_neighbors=5):
        # Use cosine distance, weight neighbors by distance for better ranking
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric='cosine',
            weights='distance'  # weighted voting to favor close neighbors
        )
        self.answers = []

    def train(self, X: np.ndarray, answers: List[str]):
        self.answers = answers
        y = np.arange(len(answers))  # Label each answer uniquely
        self.model.fit(X, y)

    def predict(self, query_vec: np.ndarray) -> str:
        idx = self.model.predict([query_vec])[0]
        return self.answers[idx]

    def predict_proba(self, query_vec: np.ndarray, top_k=3):
        distances, indices = self.model.kneighbors(
            [query_vec], n_neighbors=top_k
        )
        scores = 1 - distances[0]  # Convert cosine distance to similarity score
        results = [
            (self.answers[i], float(scores[j]))
            for j, i in enumerate(indices[0])
        ]
        return sorted(results, key=lambda x: -x[1])

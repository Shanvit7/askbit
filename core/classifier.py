from sklearn.neural_network import MLPClassifier
from typing import List
import numpy as np

class FAQClassifier:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(32,), 
            max_iter=500, 
            activation='relu',
            solver='adam',
            random_state=42
        )
        self.answers = []

    def train(self, X: np.ndarray, answers: List[str]):
        self.answers = answers
        y = np.arange(len(answers))  # each question maps to index of its answer
        self.model.fit(X, y)

    def predict(self, query_vec: np.ndarray) -> str:
        idx = self.model.predict([query_vec])[0]
        return self.answers[idx]

    def predict_proba(self, query_vec: np.ndarray, top_k=3):
        probs = self.model.predict_proba([query_vec])[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(self.answers[i], probs[i]) for i in top_indices if probs[i] > 0]

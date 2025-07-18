import numpy as np
from typing import List, Tuple


class BitEncoder:
    def __init__(self, embeddings=None):
        self.word_dim = 300
        self.embeddings = embeddings
        self.questions = []
        self.answers = []
        self.vocab = [] 
        self.encoded_matrix = None
        if self.embeddings:
            self.vocab = list(self.embeddings.keys())[:self.word_dim]

    def _text_to_vector(self, text: str) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Embeddings not provided. Pass GloVe vectors to BitEncoder.")

        words = text.lower().split()
        vectors = [self.embeddings[word] for word in words if word in self.embeddings]
        if not vectors:
            return np.zeros(self.word_dim)
        avg_vec = np.mean(vectors, axis=0)
        return (avg_vec > 0).astype(int)

    def fit(self, faq_pairs: List[Tuple[str, str]]):
        self.questions, self.answers = zip(*faq_pairs)
        self.encoded_matrix = np.array([
            self._text_to_vector(q + " " + a)
            for q, a in faq_pairs
        ])

    def encode(self, queries: List[str]) -> np.ndarray:
        return np.array([self._text_to_vector(q) for q in queries])

    def similarity_score(self, query: str):
        query_vec = self.encode([query])[0]
        return np.dot(self.encoded_matrix, query_vec)

    def retrieve_top_k(self, query: str, k=1):
        scores = self.similarity_score(query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.questions[i], self.answers[i], scores[i]) for i in top_indices if scores[i] > 0]

    def get_encoded_matrix(self):
        return self.encoded_matrix

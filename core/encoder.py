import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


class SbertBitEncoder:
    def __init__(self, bit_dim=384, model_name="all-MiniLM-L6-v2"):
        self.bit_dim = bit_dim
        self.model = SentenceTransformer(model_name)
        self.questions = []
        self.answers = []
        self.encoded_matrix = None

    def _text_to_vector(self, text: str) -> np.ndarray:
        dense = self.model.encode([text], normalize_embeddings=True)[0]
        bit_vector = (dense > 0).astype(int)
        return bit_vector

    def fit(self, faq_pairs: List[Tuple[str, str]]):
        self.questions, self.answers = zip(*faq_pairs)
        texts = [f"{q} {a}" for q, a in faq_pairs]
        self.encoded_matrix = np.array(
            [self._text_to_vector(text) for text in texts]
        )

    def encode(self, queries: List[str]) -> np.ndarray:
        return np.array([self._text_to_vector(q) for q in queries])

    def similarity_score(self, query: str):
        query_vec = self.encode([query])[0]
        return np.sum(
            self.encoded_matrix == query_vec,
            axis=1,
        ) / self.encoded_matrix.shape[1]

    def retrieve_top_k(self, query: str, k=1):
        scores = self.similarity_score(query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            (self.questions[i], self.answers[i], scores[i])
            for i in top_indices if scores[i] > 0
        ]

import numpy as np
from typing import List, Tuple
from collections import Counter
import math
import yake


class BitEncoder:
    def __init__(self, embeddings=None, keyword_boost=2.0, threshold_percentile=75):
        self.word_dim = 300
        self.embeddings = embeddings
        self.questions = []
        self.answers = []
        self.encoded_matrix = None
        self.threshold_percentile = threshold_percentile
        self.keyword_boost = keyword_boost
        self.word_freq = Counter()
        self.total_docs = 1  # avoid division by zero

        if self.embeddings:
            self.vocab = list(self.embeddings.keys())[:self.word_dim]
        self.yake_extractor = yake.KeywordExtractor(top=5, stopwords=None)

    def _extract_keywords(self, text: str, top_k=5) -> List[str]:
        return [kw for kw, _ in self.yake_extractor.extract_keywords(text)]

    def _text_to_vector(self, query: str, answer: str = "") -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Embeddings not provided.")

        def vectorize(text: str, keyword_set: set) -> np.ndarray:
            words = text.lower().split()
            vectors = []
            for word in words:
                vec = self.embeddings.get(word)
                if vec is not None:
                    freq = self.word_freq.get(word, 1)
                    idf = 1 + math.log((1 + self.total_docs) / freq)
                    boost = self.keyword_boost if word in keyword_set else 1.0
                    vectors.append(vec * idf * boost)
            return np.sum(vectors, axis=0) if vectors else np.zeros(self.word_dim)

        # Get keyword set
        keywords_q = set(self._extract_keywords(query))
        keywords_a = set(self._extract_keywords(answer))

        # Weighted Q and A encoding
        q_vec = vectorize(query, keywords_q)
        a_vec = vectorize(answer, keywords_a)
        combined = (q_vec * 1.5) + a_vec

        # Binarize based on top percentile threshold
        threshold = np.percentile(combined, self.threshold_percentile)
        return (combined > threshold).astype(int)

    def fit(self, faq_pairs: List[Tuple[str, str]]):
        self.questions, self.answers = zip(*faq_pairs)

        # Build document frequencies over entire FAQ set
        all_text = " ".join([q + " " + a for q, a in faq_pairs]).lower()
        self.word_freq = Counter(all_text.split())
        self.total_docs = len(faq_pairs)

        # Encode all FAQ pairs
        self.encoded_matrix = np.array([
            self._text_to_vector(q, a) for q, a in faq_pairs
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

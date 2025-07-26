import time
import joblib
from pathlib import Path
from typing import List, Tuple
import re
from core.encoder import SbertBitEncoder
from core.classifier import FAQClassifier
from services.logger import get_logger

logger = get_logger("faq_service")


class FAQService:
    def __init__(self):
        logger.info("🔧 Initializing FAQService with SbertBitEncoder...")
        self.encoder = SbertBitEncoder(bit_dim=128)
        self.classifier = FAQClassifier(n_neighbors=5)
        self.trained = False

    def fit(self, faq_pairs: List[Tuple[str, str]]):
        logger.info("🧠 Fitting bit encoder and training classifier...")
        self.encoder.fit(faq_pairs)
        encoded_X = self.encoder.encoded_matrix
        encoded_X = encoded_X.astype(bool)
        answers = list(self.encoder.answers)
        self.classifier.train(encoded_X, answers)
        self.trained = True
        logger.info("✅ Training complete.")

    def save(self, path: str = "model/askbit_faq_classifier.pkl"):
        Path("model").mkdir(exist_ok=True)
        joblib.dump({
            "questions": self.encoder.questions,
            "answers": self.encoder.answers,
            "encoded_matrix": self.encoder.encoded_matrix,
            "classifier": self.classifier.model,
        }, path)
        logger.info(f"💾 Model saved to: {path}")

    def load(self, path: str = "model/askbit_faq_classifier.pkl"):
        if not Path(path).exists():
            raise FileNotFoundError(
                "❌ Trained model not found. Please train first."
            )
        data = joblib.load(path)
        self.encoder.questions = data["questions"]
        self.encoder.answers = data["answers"]
        self.encoder.encoded_matrix = data["encoded_matrix"]
        self.classifier.model = data["classifier"]
        self.classifier.answers = list(self.encoder.answers)
        self.trained = True

    def answer_query(self, query: str, top_k: int = 1):
        if not self.trained:
            raise ValueError("Model not trained. Call `fit()` first.")

        def preprocess_text(text):
            text = text.lower().strip()
            text = re.sub(r'\btrail\b', 'trial', text)
            return text

        query = preprocess_text(query)

        start = time.time()
        query_vec = self.encoder.encode([query])[0].astype(bool)
        encoding_time = time.time() - start
        logger.info(f"⏱ Encoding took: {encoding_time:.4f}s")

        start = time.time()
        if top_k == 1:
            result = self.classifier.predict(query_vec)
        else:
            result = self.classifier.predict_proba(query_vec, top_k=top_k)
        prediction_time = time.time() - start
        logger.info(f"⏱ Prediction took: {prediction_time:.4f}s")

        return result

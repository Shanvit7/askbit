import time
import joblib
from pathlib import Path
from typing import List, Tuple
from core.bitencoder import BitEncoder
from core.classifier import FAQClassifier
from core.glove_loader import load_glove
from services.logger import get_logger

logger = get_logger("faq_service")


class FAQService:
    def __init__(self):
        logger.info(
            "🔧 [bold blue]Initializing FAQService and loading GloVe vectors..."
        )
        embeddings = load_glove()  # Eager load at init (with caching)
        self.encoder = BitEncoder(embeddings)
        self.classifier = FAQClassifier()
        self.trained = False

    def fit(self, faq_pairs: List[Tuple[str, str]]):
        logger.info("🧠 [green]Fitting encoder and training classifier...[/]")
        self.encoder.fit(faq_pairs)
        encoded_X = self.encoder.get_encoded_matrix()
        answers = list(self.encoder.answers)
        self.classifier.train(encoded_X, answers)
        self.trained = True
        logger.info("✅ [bold green]Training complete.[/]")

    def save(self, path: str = "model/askbit_faq_classifier.pkl"):
        Path("model").mkdir(exist_ok=True)
        joblib.dump({
            "questions": self.encoder.questions,
            "answers": self.encoder.answers,
            "encoded_matrix": self.encoder.encoded_matrix,
            "classifier": self.classifier.model,
        }, path)
        logger.info(f"💾 [cyan]Model saved to:[/] {path}")

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

        start = time.time()
        query_vec = self.encoder.encode([query])[0]
        encoding_time = time.time() - start
        logger.info(f"⏱ [dim]Encoding took:[/] {encoding_time:.2f}s")

        start = time.time()
        if top_k == 1:
            result = self.classifier.predict(query_vec)
        else:
            result = self.classifier.predict_proba(query_vec, top_k=top_k)
        prediction_time = time.time() - start
        logger.info(f"⏱ [dim]Prediction took:[/] {prediction_time:.2f}s")

        return result

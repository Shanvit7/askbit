import time
import joblib
from pathlib import Path
from typing import List, Tuple
from core.bitencoder import BitEncoder
from core.classifier import FAQClassifier
from core.glove_loader import load_glove
from services.logger import get_logger

logger = get_logger("classifier_service")


class ClassifierService:
    def __init__(self):
        logger.info("🔧 [bold blue]Initializing Classifier Service ...[/]")
        embeddings = load_glove()
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

    def save(self, path: str = "models/faq_classifier/model.pkl"):
        Path("models").mkdir(exist_ok=True)
        joblib.dump({
            "questions": self.encoder.questions,
            "answers": self.encoder.answers,
            "encoded_matrix": self.encoder.encoded_matrix,
            "classifier": self.classifier.model,
        }, path)
        logger.info(f"💾 [cyan]Model saved to:[/] {path}")

    def load(self, path: str = "models/faq_classifier/model.pkl"):
        if not Path(path).exists():
            raise FileNotFoundError("❌ Trained model not found. Please train first.")
        data = joblib.load(path)
        self.encoder.questions = data["questions"]
        self.encoder.answers = data["answers"]
        self.encoder.encoded_matrix = data["encoded_matrix"]
        self.classifier.model = data["classifier"]
        self.classifier.answers = list(self.encoder.answers)
        self.trained = True
        
    def answer_query(self, query: str, top_k: int = 1, margin: int = 2):
    if not self.trained:
        raise ValueError("Model not trained. Call `fit()` first.")

    # Always encode the query
    query_vec = self.encoder.encode([query])[0]

    # Get direct top-K matches by bit overlap
    bit_matches = self.encoder.retrieve_top_k(query, k=2)  # At least top 2

    # If only a single FAQ, just return it
    if len(bit_matches) == 1:
        return bit_matches[0][1] if top_k == 1 else [bit_matches[0]]

    # If a clear winner by margin, take it
    first_score = bit_matches[0][2]
    second_score = bit_matches[1][2] if len(bit_matches) > 1 else 0
    if (first_score - second_score) >= margin:
        return bit_matches[0][1] if top_k == 1 else [bit_matches[0]]

    # Otherwise, fall back to KNN for ranking
    if top_k == 1:
        result = self.classifier.predict(query_vec)
    else:
        result = self.classifier.predict_proba(query_vec, top_k=top_k)
    return result


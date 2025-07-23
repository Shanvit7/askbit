import cmd2
import json
import argparse
import shlex
import traceback
from pathlib import Path
import numpy as np
from services.faq import FAQService
from services.logger import get_logger

logger = get_logger("askbit_cli")

class AskBitApp(cmd2.Cmd):
    def __init__(self):
        super().__init__()
        self.faq_service = FAQService()
        self.intro = "Welcome to AskBit CLI. Type help or ? to list commands.\n"
        self.prompt = "(askbit) "

    def do_train(self, args):
        """Train the FAQ RAG system on your company's FAQ JSON."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "path",
            nargs="?",
            default="data/faq.json",
            help="Path to JSON FAQ data",
        )
        ns = parser.parse_args(args.split())

        path = ns.path
        if not Path(path).exists():
            self.perror(f"❌ File not found: {path}")
            return

        with open(path) as f:
            raw_data = json.load(f)
            faq_pairs = [
                (item["question"], item["answer"])
                for item in raw_data
                if item.get("question") and item.get("answer")
            ]

        self.faq_service.fit(faq_pairs)
        self.faq_service.save()
        self.poutput("✅ Company FAQ model trained and saved.")

    def do_ask(self, args):
        """Ask a question and get the best answer."""
        parser = argparse.ArgumentParser()
        parser.add_argument("question", help="Question to ask")
        ns = parser.parse_args(shlex.split(args))

        try:
            self.faq_service.load()
        except FileNotFoundError as e:
            self.perror(str(e))
            return

        try:
            answer = self.faq_service.answer_query(ns.question)
            self.poutput(f"🤖 Answer:\n{answer}")
        except Exception as e:
            logger.error(f"❌ {e}")
            traceback.print_exc()
            self.perror(f"❌ {e}")

    def do_vector(self, args):
        """Show raw bit vector and active bit indices for a query."""
        parser = argparse.ArgumentParser()
        parser.add_argument("query", help="Query to inspect")
        ns = parser.parse_args(shlex.split(args))

        try:
            self.faq_service.load()
        except FileNotFoundError as e:
            self.perror(str(e))
            return

        vector = self.faq_service.encoder.encode([ns.query])[0]
        bit_string = "".join(str(int(b)) for b in vector)
        active_bits = list(np.nonzero(vector)[0])
        self.poutput(f"🧮 Bit Vector — [{np.sum(vector)} active bits]:")
        self.poutput(bit_string)
        self.poutput(f"⚡ Active Bit Indices:\n{active_bits}")

    def do_topk(self, args):
        """Show the FAQ top-K matches for a query."""
        parser = argparse.ArgumentParser()
        parser.add_argument("query", help="Query to analyze")
        parser.add_argument("--topk", type=int, default=3)
        ns = parser.parse_args(shlex.split(args))

        try:
            self.faq_service.load()
        except FileNotFoundError as e:
            self.perror(str(e))
            return

        candidates = self.faq_service.encoder.retrieve_top_k(ns.query, k=ns.topk)
        if not candidates:
            self.poutput("No FAQ candidates found.")
            return

        for i, (q, a, score) in enumerate(candidates, 1):
            self.poutput(f"\n🔹 Match #{i}:")
            self.poutput(f"Q: {q}")
            self.poutput(f"A: {a}")
            self.poutput(f"⚙ Score: {score:.2%}")

    def do_keywords(self, args):
        """See what keywords YAKE extracts (for bit-boosting insights)."""
        parser = argparse.ArgumentParser()
        parser.add_argument("query", help="Query text")
        ns = parser.parse_args(shlex.split(args))

        try:
            self.faq_service.load()
        except FileNotFoundError as e:
            self.perror(str(e))
            return

        keywords = self.faq_service.encoder._extract_keywords(ns.query)
        self.poutput(f"🧠 YAKE Keywords for: '{ns.query}'")
        for idx, kw in enumerate(keywords, 1):
            self.poutput(f"{idx}. {kw}")

def main():
    app = AskBitApp()
    app.cmdloop()

if __name__ == "__main__":
    main()

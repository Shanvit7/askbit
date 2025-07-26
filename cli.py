import cmd2
import json
import argparse
import shlex
import traceback
from pathlib import Path
import numpy as np
from services.faq import FAQService
from services.logger import get_logger
import pexpect
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger("askbit_cli")


class AskBitApp(cmd2.Cmd):
    def __init__(self):
        super().__init__()
        self.faq_service = FAQService()
        self.intro = "Welcome to AskBit CLI. Type help or ? to list commands."
        self.prompt = "(askbit) "
        self._model_loaded = False

    def _load_model_once(self):
        if not self._model_loaded:
            try:
                self.faq_service.load()
                self._model_loaded = True
            except FileNotFoundError as e:
                self.perror(str(e))
                return False
        return True

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

        try:
            self.faq_service.fit(faq_pairs)
            self.faq_service.save()
            self.poutput("✅ Company FAQ model trained and saved.")
            self._model_loaded = True
        except Exception as e:
            logger.error(f"Error during training/saving: {e}")
            traceback.print_exc()
            self.perror(f"❌ Training or saving failed: {e}")

    def do_match(self, args):
        """Retrieve the best FAQ match for a question."""
        parser = argparse.ArgumentParser()
        parser.add_argument("question", help="Question to match")
        ns = parser.parse_args(shlex.split(args))

        if not self._load_model_once():
            return

        try:
            answer = self.faq_service.answer_query(ns.question)
            self.poutput(f"🤖 Retrieved Answer:\n{answer}")
        except Exception as e:
            logger.error(f"❌ {e}")
            traceback.print_exc()
            self.perror(f"❌ {e}")

    def do_ask(self, args):
        """Generate an answer using Llama 3 via Ollama."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "question",
            help="Question to ask and generate answer for",
        )
        parser.add_argument(
            "--topk",
            type=int,
            default=3,
            help="Number of FAQ entries to retrieve as context",
        )
        ns = parser.parse_args(shlex.split(args))

        if not self._load_model_once():
            return

        try:
            # Retrieve top-k FAQ pairs as context
            faq_context = self.faq_service.encoder.retrieve_top_k(
                ns.question,
                k=ns.topk,
            )

            if not faq_context:
                self.poutput("⚠️ No FAQ context found")

            # Build prompt for Llama 3
            context_texts = []
            for i, (q, a, score) in enumerate(faq_context, 1):
                context_texts.append(f"Q{i}: {q}\nA{i}: {a}")

            prompt_context = "\n\n".join(context_texts)
            prompt = (
                f"You are a friendly and helpful FAQ assistant.\n"
                f"Use the information below to answer the question"
                "thoughtfully and clearly.\n\n"
                f"Context:\n{prompt_context}\n\n"
                f"Question: {ns.question}\n\n"
                f"Answer in your own words, without copying or repeating the"
                f"context.\n"
                f"If unsure, say you don't know.\n"
                f"Answer:"
            )
            cmd = f'ollama run llama3 "{prompt}"'
            proc = pexpect.spawnu(cmd)
            self.poutput("🤖 Generated Answer (streaming):")

            while True:
                try:
                    line = proc.readline()
                    if line:
                        self.poutput(line.rstrip())
                    else:
                        break
                except pexpect.EOF:
                    break

            proc.close()
            if proc.exitstatus != 0:
                self.perror(f"❌ Ollama exited with code {proc.exitstatus}")

        except Exception as e:
            logger.error(f"❌ {e}")
            traceback.print_exc()
            self.perror(f"❌ {e}")

    def do_vector(self, args):
        """Show raw bit vector and active bit indices for a query."""
        parser = argparse.ArgumentParser()
        parser.add_argument("query", help="Query to inspect")
        ns = parser.parse_args(shlex.split(args))

        if not self._load_model_once():
            return

        vector = self.faq_service.encoder.encode([ns.query])[0]
        bit_vector = vector.astype(int)
        bit_string = "".join(str(b) for b in bit_vector)
        active_bits = list(np.nonzero(bit_vector)[0])
        self.poutput(f"🧮 Bit Vector — [{np.sum(bit_vector)} active bits]:")
        self.poutput(bit_string)
        self.poutput(f"⚡ Active Bit Indices:\n{active_bits}")

    def do_topk(self, args):
        """Show the FAQ top-K matches for a query."""
        parser = argparse.ArgumentParser()
        parser.add_argument("query", help="Query to analyze")
        parser.add_argument("--topk", type=int, default=3)
        ns = parser.parse_args(shlex.split(args))

        if not self._load_model_once():
            return

        candidates = self.faq_service.encoder.retrieve_top_k(
            ns.query,
            k=ns.topk,
        )
        if not candidates:
            self.poutput("No FAQ candidates found.")
            return

        for i, (q, a, score) in enumerate(candidates, 1):
            self.poutput(f"\n🔹 Match #{i}:")
            self.poutput(f"Q: {q}")
            self.poutput(f"A: {a}")
            self.poutput(f"⚙ Score: {score:.2%}")


def main():
    app = AskBitApp()
    app.cmdloop()


if __name__ == "__main__":
    main()

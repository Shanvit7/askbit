import cmd2
import json
import argparse
import shlex
from pathlib import Path
import numpy as np
from services.faq import FAQService


class AskBitApp(cmd2.Cmd):
    def __init__(self):
        super().__init__()
        self.faq_service = FAQService()
        self.intro = "Welcome to AskBit CLI. Type help or ? to list commands.\n"
        self.prompt = "(askbit) "

    def do_train(self, args):
        """Train the model from a JSON list of (question, answer) pairs."""
        parser = argparse.ArgumentParser()
        parser.add_argument("path", nargs="?", default="data/faq.json", help="Path to JSON FAQ data")
        ns = parser.parse_args(args.split())

        path = ns.path
        if not Path(path).exists():
            self.perror(f"❌ File not found: {path}")
            return

        with open(path) as f:
            raw_data = json.load(f)
            faq_pairs = [(item["question"], item["answer"]) for item in raw_data]

        self.faq_service.fit(faq_pairs)
        self.faq_service.save()
        self.poutput("✅ Model trained and saved.")

    def do_ask(self, args):
        """Ask a question and get the best answer."""
        parser = argparse.ArgumentParser()
        parser.add_argument("question", help="Question to ask")
        ns = parser.parse_args(shlex.split(args))

        try:
            self.faq_service.load()
            result = self.faq_service.answer_query(ns.question, top_k=1)
            if isinstance(result, str):
                self.poutput(f"🤖 Answer: {result}")
            else:
                answer, score = result[0]
                self.poutput(f"🤖 Answer: {answer} (confidence: {score:.2%})")
        except ValueError as e:
            self.perror(f"❌ {e}")

    def do_status(self, _):
        """Show model training status and internal details."""
        try:
            self.faq_service.load()
        except ValueError as e:
            self.perror(f"❌ {e}")
            return

        model = self.faq_service.classifier.model
        self.poutput("🧠 Model Status:")
        self.poutput(f"- Hidden Layers: {model.hidden_layer_sizes}")
        self.poutput(f"- Output Classes: {len(self.faq_service.classifier.answers)}")
        self.poutput(f"- Iterations Run: {model.n_iter_}")
        self.poutput(f"- Max Iterations: {model.max_iter}")
        self.poutput(f"- Converged: {model.n_iter_ < model.max_iter}")

    def do_vector(self, args):
        """Show raw bit vector for a given query."""
        parser = argparse.ArgumentParser()
        parser.add_argument("query", help="Query to inspect")
        ns = parser.parse_args(shlex.split(args))  
        self.faq_service.load()
        vector = self.faq_service.encoder.encode([ns.query])[0]
        bit_string = ''.join(str(b) for b in vector)
        self.poutput(f"🧮 Bit Vector ({np.sum(vector)} active bits):\n{bit_string}")

    def do_neurons(self, args):
        """Show hidden neuron activations for a query (with word labels)."""
        parser = argparse.ArgumentParser()
        parser.add_argument("query", help="Query to analyze")
        ns = parser.parse_args(shlex.split(args))

        self.faq_service.load()
        vec = self.faq_service.encoder.encode([ns.query])[0]  # Shape: (300,)

        model = self.faq_service.classifier.model
        weights = model.coefs_[0]       # Shape: (300, 32)
        biases = model.intercepts_[0]   # Shape: (32,)

        # Forward pass to hidden layer (ReLU activation)
        hidden_raw = np.dot(vec, weights) + biases
        hidden_activations = np.maximum(0, hidden_raw)

        # Find top input words that most influence each neuron
        vocab = self.faq_service.encoder.vocab
        top_k = 5
        neuron_keywords = []

        for neuron_index in range(weights.shape[1]):
         top_input_indices = np.argsort(weights[:, neuron_index])[::-1][:top_k]
         keywords = [
             vocab[i] if i < len(vocab) else f"bit{i}"
             for i in top_input_indices
         ]
         neuron_keywords.append(keywords)

        self.poutput(
         "\n📡 This output shows which *hidden neurons* in the model were most activated by your query.\n\n"
         "🧠 In a neural network, **neurons** are simple units that detect patterns in the input.\n"
         "Each neuron is like a tiny decision-maker that checks: 'Did I see a certain pattern of bits or words?'\n\n"
         "✨ These hidden neurons live in the middle of the network — between the input (your question as bits)\n"
         "and the output (the final answer prediction). They’re called 'hidden' because we don’t directly\n"
         "see what they’re doing during training — we just know they learn to respond to useful patterns.\n\n"
         "🔍 For each neuron, we show:\n"
         "  - How strongly it fired (activation level).\n"
         "  - Which input features (words or bits) most influence it.\n\n"
         "💡 When a neuron has *high activation*, it means your query matched a pattern that this neuron 'specializes in'.\n"
         "For example, a neuron might respond strongly when it sees finance-related terms, or when certain technical\n"
         "words co-occur — even if we never gave it those labels explicitly.\n\n"
         "So this view gives you a peek into what the model 'notices' inside your query — kind of like seeing\n"
         "which circuits light up when your brain recognizes something familiar.\n"
        )


        # Display neuron activations with layman labels
        self.poutput(f"🔬 Hidden Neuron Activations for: '{ns.query}'\n")
        for i, act in enumerate(hidden_activations):
          bar = "█" * int(act * 10) if act > 0 else ""
          keywords = ", ".join(neuron_keywords[i])
          self.poutput(f"Neuron {i + 1:02}: {bar} ({act:.3f}) — ⚡ Keywords: {keywords}")


def main():
    app = AskBitApp()
    app.cmdloop()


if __name__ == "__main__":
    main()

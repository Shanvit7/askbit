# AskBit: A Bit-Based Semantic RAG FAQ Assistant

AskBit is a FAQ and knowledge assistant that uses **bit vector encoding** of semantic sentence embeddings for high-speed retrieval—and integrated **generative RAG** capabilities via local Llama 3 using Ollama. AskBit gives you **blazing-fast semantic search** plus **fully offline, context-grounded generative answers** in an easy-to-use CLI.

---

## 🚀 Quickstart

### 1. Install dependencies (via [uv](https://github.com/astral-sh/uv))

```
uv pip install -r requirements.txt
```

### 2. Run AskBit CLI

```
make dev
```

This runs:

```
uv run python cli.py
```

Launching the interactive CLI interface:

```
Welcome to AskBit CLI. Type help or ? to list commands.
(askbit)
```

---

## 🛠️ CLI Commands

### 🧠 Train the model

```
(askbit) train data/faq.json
```

Train on a JSON file containing an array of Q&A pairs:

```
[
  {"question": "...", "answer": "..."},
  ...
]
```

### 💬 Retrieval-Augmented Generation (RAG) Answer

```
(askbit) ask "How do I reset my password?"
```

Returns a generated answer:  
- **Finds the most relevant FAQs using fast semantic search.**
- **Builds a context prompt and sends it to Llama 3 (via Ollama).**
- **Streams the generative answer live in your terminal.**

> **Requires [Ollama](https://ollama.com/) and the Llama 3 model installed locally (`ollama run llama3` must work in your terminal).**

### 🔎 Pure Retrieval

```
(askbit) match "How do I reset my password?"
```

Returns the best-matched answer based on semantic similarity, with no generation.

### 🗞 Debug / Inspect

```
(askbit) vector "How do I reset my password?"           # Show bit vector for a query
(askbit) topk "How do I reset my password?" --topk 3    # Show top 3 FAQ matches
(askbit) keywords "How do I reset my password?"         # See extracted keywords
```

---

## 📦 Architecture Overview

### 1. Input: FAQ Dataset

Example input:

```
faq = [
    ("How to reset my password?", "Click 'Forgot Password' on the login page."),
    ("What is the refund policy?", "Refunds are processed within 5 business days."),
    # ... more pairs
]
```

### 2. Semantic Bit Encoding

- Each question-answer pair is encoded as a **dense semantic vector** using SBERT (`sentence-transformers`).
- Embeddings are **binarized**: each float dimension >0 becomes 1, else 0.
- Results in a compact, efficient, semantic **bit vector** per item.

Example:

```
dense_vec = sbert_model.encode("example text", normalize_embeddings=True)
bit_vector = (dense_vec > 0).astype(int)
```

### 3. Binary KNN Retrieval

- Uses `sklearn` KNeighborsClassifier with **Hamming distance** for bitwise similarity.
- Finds the closest entry (or top-K) for any query—*semantics, not keywords!*

---

### 4. Retrieval-Augmented Generation (RAG) with Ollama Llama 3

- The best-matched FAQ entries (top-K context) are **assembled as a prompt**.
- This context and the user's query are sent to Llama 3 served locally by **Ollama**.
- Llama 3 generates a fluent, context-aware answer, which is streamed live back to the user in the terminal.

#### Example RAG prompt
```
Context:
Q1: How do I reset my password?
A1: Click 'Forgot Password' on the login page.

Q2: I can't log in, what should I do?
A2: Use the reset password link or contact support.

User question: How do I access my account if I forgot my password?
Answer (only with the answer, using the above context):
```

---

## 💡 How AskBit Works

1. **Training:**
   - Prepare FAQ Q&A pairs.
   - Encode each as binary semantic vectors.
   - Train KNN (Hamming) on these bits.

2. **Retrieval:**
   - Encode user query as bits.
   - Retrieve best FAQ matches with bitwise KNN.

3. **RAG/Generation (ask):**
   - Build a prompt of top-K FAQ Q&A.
   - Pass as context to Llama 3 via Ollama.
   - Stream the LLM's answer back in real time.

---

## 🧠 Why Bit Vectors + RAG?

- **Semantic search**: Paraphrased and fuzzy queries work out of the box.
- **Lightning-fast**: Bit ops are 50× faster than float vector math.
- **Tiny memory/ram**: 1M FAQs ≈ 48MB for bit vectors.
- **Truly useful RAG**: LLM is grounded in your actual company knowledge, avoids hallucinating.
- **Private and offline**: No API calls, all runs on your own machine.

## 🌐 Ollama Integration

- [Ollama](https://ollama.com/) makes it trivial to run Llama 3 and other powerful models locally with streaming output.
- You must have Ollama and the `llama3` model installed:
    - `brew install ollama`
    - `ollama run llama3` (test)
- AskBit streams answers directly from Ollama, enhancing both **retrieval and user experience**.

---

## ⚙️ Development Notes

- Dependencies in `requirements.txt`.
- Environment managed via `uv`.
- Run with:

```
make dev
```

- Extendable: swap out the LLM, tweak how context is constructed, or add more models.

---

## 📝 Summary

**AskBit is a blazing-fast, local RAG assistant—combining semantic bit vector search with offline generative power.** It supports robust FAQ tasks (retrieval), real conversational Q&A (generation), and is hackable, explainable, and private—designed for your own data.

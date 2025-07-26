# AskBit: A Bit-Based Semantic FAQ Retriever

AskBit is a lightweight, transparent, and fast FAQ assistant that learns to answer questions using **bit vector encoding** of semantic sentence embeddings and a **binary KNN classifier**. Built for high performance and interpretability, AskBit uses **binarized SBERT embeddings** for fast and semantically meaningful FAQ retrieval.

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

### 💬 Ask a question

```
(askbit) ask "How do I reset my password?"
```

Returns the best matched answer based on semantic similarity.

### 🗞 Debug / Inspect

```
(askbit) vector "How do I reset my password?"  # Show raw bit vector for the query
(askbit) topk "How do I reset my password?" --topk 3  # Show top 3 matching FAQ entries
(askbit) keywords "How do I reset my password?"  # Extract important keywords (via YAKE)
```

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

### 2. Semantic Bit Encoding: `SbertBitEncoder`

- Each question-answer pair is embedded as a **dense semantic vector** using **SBERT (Sentence-BERT)**.
- These vectors capture the *meaning* of entire sentences beyond simple word overlap.
- The dense float vector is then **binarized** to a bit vector (0s and 1s) by thresholding values (e.g., bits set to 1 if value > 0).
- Result: compact, semantically meaningful bit vectors.

Example binarization:

```
dense_vec = sbert_model.encode("example text", normalize_embeddings=True)
bit_vector = (dense_vec > 0).astype(int)
```

- Queries and FAQ entries are both embedded this way, enabling semantic matching.

### 3. Binary KNN Classifier: `FAQClassifier`

- Uses `sklearn`’s `KNeighborsClassifier` with **Hamming distance** metric operating on bit vectors.
- Learns to map bit-encoded queries to the most relevant FAQ answers.
- Supports:
  - Exact top-1 answer retrieval.
  - Top-k candidates with similarity scores.

Training labels are simple indices of answers:

```
y = np.arange(len(answers))
model.fit(bit_vectors, y)
```

### 4. Fast Semantic Retrieval With Bit Vectors

| Aspect           | Description                                |
|------------------|--------------------------------------------|
| Representation   | SBERT embeddings binarized into fixed-length vectors of 0/1 |
| Similarity       | Hamming distance (number of differing bits) |
| Classifier       | Weighted KNN to select best answer          |
| Benefit          | Semantic understanding + fast, compact storage |

## 💡 How AskBit Works

1. **Training:**
   - Load Q&A pairs.
   - Compute SBERT dense embeddings for each FAQ entry.
   - Binarize embeddings into bit vectors.
   - Train KNN classifier on these bit vectors indexed by answers.

2. **Querying:**
   - Preprocess user query text.
   - Encode query with the same SBERT model → binarize bits.
   - Use KNN with Hamming distance to find closest FAQ entries.
   - Return the answer corresponding to the nearest (top-k) bit vectors.

## 🧠 Why Bit Vectors?

- Traditional embeddings require heavy float computations.
- Bit vectors enable lightning-fast similarity using bitwise operations.
- Compact storage: bits consume less memory and speed up index traversal.
- By binarizing SBERT embeddings, AskBit merges the **semantic power of transformers** with **efficient binary retrieval**.

## ⚙️ Development Notes

- Environment managed via `uv` and dependencies declared in `requirements.txt`.
- Run the CLI using the stable command:

```
make dev
```

- Debugging tools and commands available within the CLI for inspecting bit vectors, keywords, and top matches.

## 📝 Summary

AskBit is a modern FAQ retriever combining state-of-the-art sentence embeddings (SBERT) with classic bit-level binary search techniques to deliver a fast, semantically aware, and lightweight question-answering assistant.
```

# AskBit: A Bit-Based ML FAQ Retriever

AskBit is a lightweight, transparent, and fast FAQ assistant that learns to answer questions using **bit vector encoding** and a **tiny neural network**. Built for learning and performance, uses lightweight binarized GloVe embeddings for fast and explainable learning.”

---

## 🚀 Quickstart

### 1. Install dependencies (via [uv](https://github.com/astral-sh/uv))

```bash
uv pip install -r requirements.txt
```

### 2. Start AskBit CLI

```bash
askbit
```

This will launch the interactive CLI:

```text
Welcome to AskBit CLI. Type help or ? to list commands.
(askbit)
```

---

## 🛠️ CLI Commands

### 🧠 Train the model

```bash
(askbit) train data/faq.json
```

Train on a list of Q\&A pairs (JSON file with `[{"question": ..., "answer": ...}, ...]`).

### 💬 Ask a question

```bash
(askbit) ask "How to reset my password?"
```

Returns the best matched answer.

### 🗞 Debug / Inspect

```bash
(askbit) neurons "How to reset my password?"  # Show hidden neuron activations
(askbit) vector "How to reset my password?"  # Show raw bit vector
(askbit) status                   #  Shows models training status and internal details.
```

---

## 📦 Architecture Overview

### 1. Input: Q\&A Dataset

```python
faq = [
    ("How to reset my password?", "Click 'Forgot Password' on the login page."),
    ("What is the refund policy?", "Refunds are processed within 5 business days."),
    # ... more pairs
]
```

---

### 2. Bit Encoding: `BitEncoder`

Each question-answer pair is embedded using preloaded **GloVe vectors**. We average all word embeddings in the input and **binarize** the resulting 300-dimensional vector:

```python
avg_vec = np.mean(vectors, axis=0)
bit_vector = (avg_vec > 0).astype(int)
```

This results in a compact and interpretable bit vector for each pair.

* Words not found in GloVe are ignored.
* Question and answer are **concatenated** before encoding.
* The resulting bit vectors are stored in a matrix used for classification and retrieval.

**Similarity search** can also be done using a simple dot product:

```python
score = np.dot(encoded_matrix, query_vec)
```

---

### 3. Neural Classifier: `FAQClassifier`

AskBit trains a small neural network (`MLPClassifier`) using:

* **Input:** Bit-encoded vector of each Q\&A pair
* **Output:** Index of the correct answer

Training target is a list of integers mapping each question to its answer.

```python
y = np.arange(len(answers))
model.fit(X, y)
```

At inference time:

* For `ask`: we pick the top-1 predicted answer index
* For `topk`: we sort class probabilities and return top-k answers

The classifier learns **nonlinear patterns** between questions and their corresponding answers.

---

## 🧠 What Is an MLP (Multilayer Perceptron)?

### Structure

* **Input layer:** Receives the bit vector
* **Hidden layers:** Learn patterns and combinations
* **Output layer:** Class probabilities per answer index

### Why Hidden Layers?

* Allow model to capture **non-linear** relationships
* E.g., "reset" + "password" → password-related intent

### Activation Functions

* Introduce non-linearity (e.g., ReLU, tanh)
* Prevent model from becoming a simple linear classifier

### Learning Process

1. Forward pass: compute prediction
2. Compare to true label (loss)
3. Backpropagation: adjust weights
4. Repeat over epochs

### 🔄 Rebuild the Package After Code Changes

If you make changes to the code, you need to rebuild the package to see the changes reflected. Use the following command to reinstall the package in editable mode:

```bash
uv pip install -e .
```

This ensures that any changes you make to the source files are immediately reflected when you run the application.

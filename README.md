# Dropout-Rate Tuning for Sentiment Analysis

> **Deep Learning Mini Project** — Compare how different dropout rates affect a
> neural network's ability to classify IMDB movie reviews as positive or negative.

---

## 📂 Project Structure

```
dropout-rate-tuning/
├── dropout_tuning.py        # Main script (train, evaluate, plot)
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── dropout_comparison.png   # (generated) Validation-loss plot
└── results.txt              # (generated) Comparison table
```

---

## 🔧 Setup Instructions

### Prerequisites

| Tool   | Version          |
|--------|------------------|
| Python | 3.8 or newer     |
| pip    | (bundled w/ Python) |

### 1. (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs **TensorFlow**, **Matplotlib**, and **NumPy**.

---

## ▶️ How to Run

```bash
python dropout_tuning.py
```

The script will:

1. Download the IMDB dataset (first run only, ~17 MB).
2. Train a small neural network **four times**, once for each dropout rate
   (0.1, 0.3, 0.5, 0.7).
3. Save a **validation-loss plot** (`dropout_comparison.png`).
4. Print and save a **comparison table** (`results.txt`).

**Estimated time:** 2 – 5 minutes on a regular laptop (CPU only).

---

## 💡 What Is Dropout?

**Dropout** is a regularization technique that helps prevent **overfitting** —
when a model memorizes the training data instead of learning general patterns.

During each training step, dropout **randomly "turns off"** a fraction of
neurons (the *dropout rate*). For example, `Dropout(0.3)` means 30 % of
neurons are switched off at random in every forward pass.

| Concept        | Explanation |
|----------------|-------------|
| **Rate = 0**   | No dropout — the model uses all neurons every time (more prone to overfitting). |
| **Rate = 0.1** | Only 10 % of neurons are dropped — very mild regularization. |
| **Rate = 0.5** | Half the neurons are dropped — strong regularization. |
| **Rate = 1.0** | All neurons are dropped — the model learns nothing (never used). |

Think of it like studying with flashcards: if you randomly hide some cards
each round, you're forced to learn the material deeply instead of just
memorizing the order.

---

## 📊 How to Read the Results

### Validation Loss Plot (`dropout_comparison.png`)

- **X-axis:** Epoch number (one full pass through the training data).
- **Y-axis:** Validation loss (binary cross-entropy) — *lower is better*.
- Each colored line represents a different dropout rate.

**What to look for:**

| Pattern                                       | Meaning |
|-----------------------------------------------|---------|
| Loss keeps going **down** smoothly            | The model is still learning; not overfitting yet. |
| Loss goes down then **curves back up**        | Overfitting — the model started memorizing training data. |
| Loss stays **flat and high**                  | Under-fitting — the model isn't learning enough (possibly too much dropout). |

### Comparison Table (`results.txt`)

| Column         | Meaning |
|----------------|---------|
| **Dropout**    | The dropout rate used for that run. |
| **Best Val Loss** | The lowest validation loss achieved across all epochs — *lower is better*. |
| **Test Accuracy**  | Accuracy on the held-out test set — *higher is better*. |

**Typical findings:**

- Very low dropout (0.1) may overfit → good training accuracy, lower test accuracy.
- Moderate dropout (0.3 – 0.5) often gives the **best** test accuracy.
- Very high dropout (0.7) may under-fit → both training and test accuracy suffer.

The "sweet spot" is usually somewhere in between — that's what this experiment
helps you find!

---

## ⚙️ Customizing the Experiment

Open `dropout_tuning.py` and change the constants near the top of the file:

```python
VOCAB_SIZE    = 10_000      # vocabulary size
MAX_LEN       = 256         # review length (tokens)
DROPOUT_RATES = [0.1, 0.3, 0.5, 0.7]   # rates to compare
EPOCHS        = 10          # epochs per run
BATCH_SIZE    = 512         # batch size
HIDDEN_UNITS  = 64          # neurons per hidden layer
```

---

## 📝 License

This project is provided for educational purposes. Feel free to modify and share.

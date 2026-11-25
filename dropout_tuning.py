#!/usr/bin/env python3
"""
=============================================================================
  Dropout-Rate Tuning for Sentiment Analysis (IMDB Movie Reviews)
=============================================================================

This script trains the SAME simple neural network four times, each time with
a different dropout rate (0.1, 0.3, 0.5, 0.7).  Everything else — model
architecture, optimizer, number of epochs, batch size — stays identical so
the comparison is fair.

After training, the script:
  1. Plots validation loss vs. epoch for all four rates (saved as a PNG).
  2. Prints a comparison table (best val-loss & test accuracy per rate).
  3. Saves the table to a text file.

Dataset : keras.datasets.imdb  (built-in, auto-downloaded)
Stack   : Python 3 · TensorFlow / Keras · Matplotlib · NumPy

Author  : Mini-project scaffold generated for DL lab
=============================================================================
"""

import os

# Ensure KERAS_HOME points to a writable local directory before any Keras/TF import
# to prevent Windows /tmp/.keras fallback permission errors
if "KERAS_HOME" not in os.environ:
    workspace_keras = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".keras")
    user_keras = os.path.expanduser("~/.keras")
    if os.path.exists(os.path.join(workspace_keras, "datasets")):
        os.environ["KERAS_HOME"] = workspace_keras
    elif os.path.exists(os.path.join(user_keras, "datasets")):
        os.environ["KERAS_HOME"] = user_keras
    else:
        os.environ["KERAS_HOME"] = workspace_keras

# Suppress noisy TF info logs (keep warnings & errors)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── 0. Dependency check ────────────────────────────────────────────────────
# Give a friendly message if a required package is missing, or auto-switch to
# an installed Python version (e.g. Python 3.10) where TensorFlow is installed.
import sys
import subprocess

_REQUIRED = {"tensorflow": "tensorflow", "matplotlib": "matplotlib", "numpy": "numpy"}
_missing = []
for _mod, _pip in _REQUIRED.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pip)

if _missing:
    # Check if another Python interpreter on this machine has the packages (e.g. Python 3.10 vs 3.13)
    def _find_compatible_python():
        candidates = [
            r"C:\Program Files\Python310\python.exe",
            r"C:\Program Files\Python311\python.exe",
            r"C:\Program Files\Python312\python.exe",
        ]
        for py_ver in ["-3.10", "-3.11", "-3.12"]:
            try:
                out = subprocess.check_output(
                    f"py {py_ver} -c \"import sys; print(sys.executable)\"",
                    shell=True,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if out and os.path.exists(out) and out not in candidates:
                    candidates.append(out)
            except Exception:
                pass
        for cand in candidates:
            if os.path.exists(cand) and os.path.abspath(cand).lower() != os.path.abspath(sys.executable).lower():
                try:
                    res = subprocess.run([cand, "-c", "import tensorflow, matplotlib, numpy"], capture_output=True)
                    if res.returncode == 0:
                        return cand
                except Exception:
                    pass
        return None

    _alt_py = _find_compatible_python()
    if _alt_py and __name__ == "__main__":
        print(f"\n[Notice] Current Python ({sys.version.split()[0]} at {sys.executable}) is missing: {', '.join(_missing)}", flush=True)
        print(f"[Notice] Switching automatically to compatible Python at: {_alt_py}\n", flush=True)
        sys.exit(subprocess.call([_alt_py, os.path.abspath(__file__)] + sys.argv[1:]))

    raise SystemExit(
        f"\n╔══════════════════════════════════════════════════════════╗\n"
        f"║  Missing packages in Python {sys.version.split()[0]}: {', '.join(_missing)}\n"
        f"║  Python path: {sys.executable}\n"
        f"║  Install them with:  {sys.executable} -m pip install {' '.join(_missing)}\n"
        f"╚══════════════════════════════════════════════════════════╝"
    )

# ── 1. Imports ─────────────────────────────────────────────────────────────
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── 2. User-configurable settings ─────────────────────────────────────────
# Change these if you want faster/slower runs or bigger/smaller experiments.

VOCAB_SIZE   = 10_000   # keep the top-N most frequent words
MAX_LEN      = 256      # pad / truncate every review to this many tokens
DROPOUT_RATES = [0.1, 0.3, 0.5, 0.7]   # rates to compare

EPOCHS       = 10       # training epochs per run  (increase for deeper study)
BATCH_SIZE   = 512      # larger → faster on GPU; reduce if you run out of RAM
HIDDEN_UNITS = 64       # neurons in each Dense hidden layer

# Where to save outputs (same folder as this script by default)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) or "."


# ── 3. Data loading & preprocessing ───────────────────────────────────────
def load_data():
    """
    Load the IMDB dataset, pad sequences to a fixed length, and return
    train / test splits as NumPy arrays.
    """
    print("Loading IMDB dataset …")
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=VOCAB_SIZE
    )

    # Pad sequences so every review has exactly MAX_LEN tokens.
    # Shorter reviews get zeros on the left; longer ones are truncated.
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test  = keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=MAX_LEN)

    print(f"  Training samples : {len(x_train)}")
    print(f"  Test samples     : {len(x_test)}")
    print(f"  Vocabulary size  : {VOCAB_SIZE}")
    print(f"  Sequence length  : {MAX_LEN}")
    return (x_train, y_train), (x_test, y_test)


# ── 4. Model builder ──────────────────────────────────────────────────────
def build_model(dropout_rate: float) -> keras.Model:
    """
    Build a small feedforward neural network for binary sentiment classification.

    Architecture
    ------------
    Embedding  →  GlobalAveragePooling  →  Dense(64, relu)  →  Dropout
                                        →  Dense(64, relu)  →  Dropout
                                        →  Dense(1, sigmoid)

    Only the dropout_rate changes between runs; everything else is fixed.
    """
    model = keras.Sequential([
        # Turn each integer token into a dense 32-dim vector
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32),

        # Average all token vectors into one fixed-size vector per review
        layers.GlobalAveragePooling1D(),

        # First hidden layer + dropout
        layers.Dense(HIDDEN_UNITS, activation="relu"),
        layers.Dropout(dropout_rate),

        # Second hidden layer + dropout
        layers.Dense(HIDDEN_UNITS, activation="relu"),
        layers.Dropout(dropout_rate),

        # Output: probability that the review is positive
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── 5. Training loop over dropout rates ───────────────────────────────────
def run_experiments(train_data, test_data):
    """
    Train the model once for each dropout rate.  Returns a dict:
        { rate: {"history": History, "test_loss": float, "test_acc": float} }
    """
    (x_train, y_train) = train_data
    (x_test,  y_test)  = test_data

    results = {}

    for rate in DROPOUT_RATES:
        print(f"\n{'='*60}")
        print(f"  Training with dropout = {rate}")
        print(f"{'='*60}")

        # Set the same random seed before each run for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        model = build_model(rate)

        start = time.time()
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,         # 20 % of training data held out
            verbose=1,
        )
        elapsed = time.time() - start

        # Evaluate on the held-out test set
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        results[rate] = {
            "history":   history,
            "test_loss": test_loss,
            "test_acc":  test_acc,
            "time_sec":  elapsed,
        }
        print(f"  → Test accuracy : {test_acc:.4f}  |  Time: {elapsed:.1f}s")

    return results


# ── 6. Plotting ────────────────────────────────────────────────────────────
def plot_val_loss(results: dict, save_path: str):
    """
    Plot validation loss across epochs for every dropout rate on one chart.
    """
    plt.figure(figsize=(9, 5))

    for rate, data in results.items():
        val_loss = data["history"].history["val_loss"]
        epochs   = range(1, len(val_loss) + 1)
        plt.plot(epochs, val_loss, marker="o", markersize=4,
                 label=f"Dropout = {rate}")

    plt.title("Validation Loss vs. Epoch for Different Dropout Rates", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (Binary Cross-Entropy)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n✅ Plot saved → {save_path}")


# ── 7. Summary table ──────────────────────────────────────────────────────
def print_and_save_table(results: dict, save_path: str):
    """
    Print a comparison table and also write it to a text file.
    """
    header = (
        f"\n{'='*62}\n"
        f"  COMPARISON TABLE — Dropout-Rate Tuning (IMDB Sentiment)\n"
        f"{'='*62}\n"
        f"  {'Dropout':>8}  │  {'Best Val Loss':>14}  │  {'Test Accuracy':>14}  │  {'Time (s)':>8}\n"
        f"  {'─'*8}──┼──{'─'*14}──┼──{'─'*14}──┼──{'─'*8}\n"
    )
    rows = ""
    for rate, data in results.items():
        best_val = min(data["history"].history["val_loss"])
        rows += (
            f"  {rate:>8.1f}  │  {best_val:>14.4f}  │  {data['test_acc'] * 100:>13.2f}%  │  {data['time_sec']:>7.1f}\n"
        )
    footer = f"  {'='*62}\n"

    table = header + rows + footer
    print(table)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"✅ Results saved → {save_path}")


# ── 8. Main entry point ───────────────────────────────────────────────────
def main():
    print("\n" + "🎬 " * 20)
    print("  Dropout-Rate Tuning for Sentiment Analysis")
    print("  Dataset: IMDB Movie Reviews (Keras built-in)")
    print("🎬 " * 20 + "\n")

    train_data, test_data = load_data()
    results = run_experiments(train_data, test_data)

    plot_path    = os.path.join(OUTPUT_DIR, "dropout_comparison.png")
    results_path = os.path.join(OUTPUT_DIR, "results.txt")

    plot_val_loss(results, plot_path)
    print_and_save_table(results, results_path)

    print("\n🏁 All done!  Check the project folder for outputs.")


if __name__ == "__main__":
    main()

# Day 8 — Naive Bayes Text Classifier (From Scratch) 📨
# Author: Stuart Abhishek
# Description: Train/evaluate a multinomial Naive Bayes classifier on CSV text data
#              with k-fold cross-validation, informative tokens, confusion matrix,
#              and a JSON model card. No external ML libs.

import csv
import json
import math
import random
import re
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Iterable, Set

# ----------------------- Data Loading -----------------------

def load_csv_text_label(path: str, text_col: str = "text", label_col: str = "label") -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if text_col in r and label_col in r and r[text_col] and r[label_col]:
                rows.append({ "text": r[text_col], "label": r[label_col] })
        return rows

# ----------------------- Tokenization -----------------------

_STOP = set("""
a an the and or but if then else for while of to in on at from by with without not be is are was were am do does did have has had this that these those you your we our i me my they them their he she it its as about into over under up down off out no yes can could would should will just
""".split())

def tokenize(text: str, keep_numbers=False) -> List[str]:
    text = text.lower()
    # Keep words and numbers as separate tokens
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)
    if not keep_numbers:
        tokens = [t for t in tokens if not t.isdigit()]
    return [t for t in tokens if t not in _STOP and len(t) > 1]

# ----------------------- Naive Bayes -----------------------

@dataclass
class NBModel:
    class_priors_log: Dict[str, float]
    cond_logprob: Dict[str, Dict[str, float]]  # class -> token -> log P(token|class)
    vocab: Set[str]
    class_token_counts: Dict[str, int]
    alpha: float

    def predict_proba(self, tokens: List[str]) -> Dict[str, float]:
        totals = {}
        vocab = self.vocab
        for c, logprior in self.class_priors_log.items():
            s = logprior
            cl_map = self.cond_logprob[c]
            for t in tokens:
                if t in vocab:
                    s += cl_map.get(t, cl_map.get("<UNK>", -20.0))
            totals[c] = s
        # log-softmax to normalized probs
        m = max(totals.values())
        exp = {c: math.exp(v - m) for c, v in totals.items()}
        Z = sum(exp.values())
        return {c: v / Z for c, v in exp.items()}

    def predict(self, tokens: List[str]) -> str:
        proba = self.predict_proba(tokens)
        return max(proba.items(), key=lambda x: x[1])[0]

def train_naive_bayes(
    texts: List[List[str]],
    labels: List[str],
    alpha: float = 1.0
) -> NBModel:
    classes = sorted(set(labels))
    class_counts = Counter(labels)
    total_docs = len(labels)
    priors_log = { c: math.log(class_counts[c] / total_docs) for c in classes }

    # Word counts per class
    token_counts_per_class: Dict[str, Counter] = { c: Counter() for c in classes }
    vocab: Set[str] = set()

    for toks, c in zip(texts, labels):
        token_counts_per_class[c].update(toks)
        vocab.update(toks)

    # Add <UNK> bucket
    vocab.add("<UNK>")

    # Total tokens per class
    class_token_counts = { c: sum(token_counts_per_class[c].values()) for c in classes }

    # Conditional log-probs with Laplace smoothing
    cond_logprob: Dict[str, Dict[str, float]] = { c: {} for c in classes }
    V = len(vocab)
    for c in classes:
        total = class_token_counts[c] + alpha * V
        for t in vocab:
            count = token_counts_per_class[c][t]
            cond_logprob[c][t] = math.log((count + alpha) / total)

    return NBModel(
        class_priors_log=priors_log,
        cond_logprob=cond_logprob,
        vocab=vocab,
        class_token_counts=class_token_counts,
        alpha=alpha,
    )

# ----------------------- Evaluation -----------------------

def kfold_indices(n: int, k: int = 5, seed: int = 42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    for i in range(k):
        val = folds[i]
        train = [j for t in range(k) if t != i for j in folds[t]]
        yield train, val

def metrics(labels_true: List[str], labels_pred: List[str]) -> Dict[str, float]:
    # Overall accuracy
    acc = sum(1 for a,b in zip(labels_true, labels_pred) if a == b) / len(labels_true)

    # Per-class precision/recall/F1 (macro-averaged)
    classes = sorted(set(labels_true))
    precs, recs, f1s = [], [], []
    for c in classes:
        tp = sum(1 for t,p in zip(labels_true, labels_pred) if t == c and p == c)
        fp = sum(1 for t,p in zip(labels_true, labels_pred) if t != c and p == c)
        fn = sum(1 for t,p in zip(labels_true, labels_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
        precs.append(precision); recs.append(recall); f1s.append(f1)

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(stats.mean(precs) if precs else 0.0, 4),
        "recall_macro": round(stats.mean(recs) if recs else 0.0, 4),
        "f1_macro": round(stats.mean(f1s) if f1s else 0.0, 4),
    }

def confusion_matrix(labels_true: List[str], labels_pred: List[str]) -> Tuple[List[str], List[List[int]]]:
    classes = sorted(set(labels_true))
    idx = {c:i for i,c in enumerate(classes)}
    mat = [[0 for _ in classes] for __ in classes]
    for t,p in zip(labels_true, labels_pred):
        mat[idx[t]][idx[p]] += 1
    return classes, mat

def most_informative_tokens(model: NBModel, topn: int = 15) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
    """
    For each pair of classes (A,B), return top tokens ranked by log-odds: log P(t|A) - log P(t|B).
    """
    classes = list(model.class_priors_log.keys())
    results = {}
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            A, B = classes[i], classes[j]
            diffs = []
            for t in model.vocab:
                if t == "<UNK>": 
                    continue
                diffs.append((t, model.cond_logprob[A].get(t, -20.0) - model.cond_logprob[B].get(t, -20.0)))
            diffs.sort(key=lambda x: x[1], reverse=True)
            results[(A,B)] = diffs[:topn]
            # also negative side for B vs A
            # caller can invert as needed
    return results

# ----------------------- CLI -----------------------

def prepare_xy(rows: List[dict], keep_numbers=False):
    X = [tokenize(r["text"], keep_numbers=keep_numbers) for r in rows]
    y = [r["label"] for r in rows]
    return X, y

def cross_validate(rows: List[dict], alpha=1.0, k=5, keep_numbers=False):
    X, y = prepare_xy(rows, keep_numbers=keep_numbers)
    n = len(y)
    all_metrics = []
    for tr_idx, va_idx in kfold_indices(n, k=k):
        Xtr = [X[i] for i in tr_idx]; ytr = [y[i] for i in tr_idx]
        Xva = [X[i] for i in va_idx]; yva = [y[i] for i in va_idx]
        model = train_naive_bayes(Xtr, ytr, alpha=alpha)
        yhat = [model.predict(tokens) for tokens in Xva]
        all_metrics.append(metrics(yva, yhat))
    # aggregate
    acc = round(stats.mean(m["accuracy"] for m in all_metrics), 4)
    prec = round(stats.mean(m["precision_macro"] for m in all_metrics), 4)
    rec = round(stats.mean(m["recall_macro"] for m in all_metrics), 4)
    f1 = round(stats.mean(m["f1_macro"] for m in all_metrics), 4)
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}

def fit_full(rows: List[dict], alpha=1.0, keep_numbers=False) -> NBModel:
    X, y = prepare_xy(rows, keep_numbers=keep_numbers)
    return train_naive_bayes(X, y, alpha=alpha)

def print_confusion(labels_true, labels_pred):
    classes, mat = confusion_matrix(labels_true, labels_pred)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "     " + " ".join(f"{c:>6}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = " ".join(f"{mat[i][j]:>6}" for j in range(len(classes)))
        print(f"{c:>4} {row}")

def save_model_card(path: str, model: NBModel, cv: dict, meta: dict):
    card = {
        "algorithm": "Multinomial Naive Bayes (from scratch)",
        "alpha": model.alpha,
        "class_priors_log": model.class_priors_log,
        "vocab_size": len(model.vocab),
        "class_token_counts": model.class_token_counts,
        "cross_validation": cv,
        "metadata": meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"\n📝 Model card saved to: {path}")

def interactive_demo(model: NBModel):
    print("\n🔎 Try it! Type a message to classify (or 'quit'):\n")
    while True:
        s = input("text> ").strip()
        if s.lower() in {"quit", "exit"}:
            break
        toks = tokenize(s)
        proba = model.predict_proba(toks)
        pred = max(proba.items(), key=lambda x: x[1])[0]
        print(f"Predicted: {pred}  |  Probabilities: { {k: round(v,3) for k,v in proba.items()} }\n")

def main():
    print("🧠 Day 8 — Naive Bayes Text Classifier (From Scratch)")
    path = input("Enter CSV path (must include columns 'text','label'): ").strip()
    try:
        rows = load_csv_text_label(path)
    except Exception as e:
        print("⚠️ Could not load CSV:", e)
        return
    if len(rows) < 30:
        print(f"⚠️ Only {len(rows)} rows loaded; try a larger dataset (>= 30).")
        return

    alpha = input("Laplace smoothing alpha [default 1.0]: ").strip()
    alpha = float(alpha) if alpha else 1.0
    keep_numbers = input("Keep numeric tokens? [y/N]: ").strip().lower() in ("y","yes","")

    # Cross-validation for generalization
    cv = cross_validate(rows, alpha=alpha, k=5, keep_numbers=keep_numbers)
    print("\n🔍 5-fold Cross-Validation (macro-averaged):")
    for k,v in cv.items():
        print(f"  {k}: {v}")

    # Fit full model
    model = fit_full(rows, alpha=alpha, keep_numbers=keep_numbers)

    # Evaluate on full data (reference)
    X, y = prepare_xy(rows, keep_numbers=keep_numbers)
    yhat = [model.predict(t) for t in X]
    mt = metrics(y, yhat)
    print("\n✅ Fit on full data (reference metrics):")
    for k,v in mt.items():
        print(f"  {k}: {v}")
    print_confusion(y, yhat)

    # Explainability: top tokens
    if len(set(y)) >= 2:
        info = most_informative_tokens(model, topn=12)
        print("\n💡 Most-informative tokens (log-odds):")
        for (A,B), toks in info.items():
            print(f"  {A} vs {B}: " + ", ".join([f"{t}({round(s,2)})" for t,s in toks]))

    # Save model card
    if input("\nSave model card JSON? [Y/n]: ").strip().lower() in ("","y","yes"):
        meta = {
            "csv_path": path,
            "text_column": "text",
            "label_column": "label",
            "notes": "Multinomial NB with Laplace smoothing; 5-fold CV; token stopwording; <UNK> bucket.",
        }
        save_model_card("Day-08/model_card_naive_bayes.json", model, cv, meta)

    # Interactive playground
    if input("Start interactive demo? [Y/n]: ").strip().lower() in ("","y","yes"):
        interactive_demo(model)

    print("\n🏁 Done. Classic NLP + ML implemented with clarity and scientific discipline.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
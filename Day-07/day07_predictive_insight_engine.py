# Day 7 — Predictive Insight Engine 📈
# Author: Stuart Abhishek
#
# Univariate Linear Regression (one feature -> one target) with:
# - CSV loading
# - z-score outlier handling (optional)
# - standardization (mean/STD)
# - gradient descent with early stopping
# - 5-fold cross-validation (R2, MAE, RMSE)
# - diagnostic plots (fitted line + residuals)
# - JSON "model card" export with coefficients & metrics
#
# Dependencies: standard library + matplotlib
# (No numpy/pandas used to highlight algorithmic clarity.)

import csv
import json
import math
import random
import statistics as stats
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# ---------------------------- Utilities ----------------------------

def load_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def numeric_columns(rows: List[dict]) -> List[str]:
    cols = []
    if not rows:
        return cols
    for k in rows[0].keys():
        try:
            float(rows[0][k])
            cols.append(k)
        except (ValueError, TypeError):
            pass
    return cols

def to_float_list(rows: List[dict], col: str) -> List[float]:
    vals = []
    for r in rows:
        try:
            vals.append(float(r[col]))
        except (ValueError, TypeError):
            continue
    return vals

def zscore_filter(x: List[float], y: List[float], thresh: float = 3.0) -> Tuple[List[float], List[float]]:
    """Remove pairs whose X or Y is an outlier by z-score > thresh."""
    if len(x) < 3:
        return x, y
    mx, sx = stats.mean(x), (stats.pstdev(x) or 1e-12)
    my, sy = stats.mean(y), (stats.pstdev(y) or 1e-12)
    nx, ny = [], []
    for xi, yi in zip(x, y):
        zx = abs((xi - mx) / sx)
        zy = abs((yi - my) / sy)
        if zx <= thresh and zy <= thresh:
            nx.append(xi); ny.append(yi)
    return nx, ny

@dataclass
class Standardizer:
    mean: float
    std: float

    def transform(self, v: List[float]) -> List[float]:
        s = self.std if self.std != 0 else 1e-12
        return [(t - self.mean) / s for t in v]

    def inverse(self, v: List[float]) -> List[float]:
        return [t * (self.std if self.std != 0 else 1e-12) + self.mean for t in v]

# ---------------------------- Model ----------------------------

@dataclass
class LinRegModel:
    w0: float  # bias
    w1: float  # slope
    x_scaler: Standardizer
    y_scaler: Standardizer

    def predict(self, x_raw: List[float]) -> List[float]:
        xs = self.x_scaler.transform(x_raw)
        yhat_std = [self.w0 + self.w1 * xi for xi in xs]
        return self.y_scaler.inverse(yhat_std)

def standardize_xy(x: List[float], y: List[float]) -> Tuple[List[float], List[float], Standardizer, Standardizer]:
    xsc = Standardizer(mean=stats.mean(x), std=stats.pstdev(x))
    ysc = Standardizer(mean=stats.mean(y), std=stats.pstdev(y))
    return xsc.transform(x), ysc.transform(y), xsc, ysc

def gradient_descent(xs: List[float], ys: List[float], lr=0.05, epochs=2000, tol=1e-7, patience=100) -> Tuple[float, float]:
    """Fit y = w0 + w1*x using GD on standardized X,Y. Returns w0, w1 in standardized space."""
    w0, w1 = 0.0, 0.0
    n = len(xs)
    best_loss = float("inf")
    best = (w0, w1)
    it_since_best = 0

    def mse():
        s = 0.0
        for xi, yi in zip(xs, ys):
            yhat = w0 + w1 * xi
            s += (yi - yhat) ** 2
        return s / n

    for _ in range(epochs):
        # gradients
        dw0, dw1 = 0.0, 0.0
        for xi, yi in zip(xs, ys):
            yhat = w0 + w1 * xi
            err = yhat - yi
            dw0 += err
            dw1 += err * xi
        dw0 /= n; dw1 /= n

        # update
        w0 -= lr * dw0
        w1 -= lr * dw1

        # early stopping on validation-like behavior (training loss)
        loss = mse()
        if loss + tol < best_loss:
            best_loss = loss
            best = (w0, w1)
            it_since_best = 0
        else:
            it_since_best += 1
            if it_since_best >= patience:
                break
    return best

# ---------------------------- Metrics & CV ----------------------------

def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    ym = stats.mean(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - ym) ** 2 for a in y_true) or 1e-12
    return 1 - ss_res / ss_tot

def mae(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))

def kfold_indices(n: int, k: int = 5) -> List[Tuple[List[int], List[int]]]:
    idx = list(range(n))
    random.Random(42).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    splits = []
    for i in range(k):
        val = folds[i]
        train = [j for t in range(k) if t != i for j in folds[t]]
        splits.append((train, val))
    return splits

def cross_val(x: List[float], y: List[float], k: int = 5) -> Dict[str, float]:
    r2s, maes, rmses = [], [], []
    n = len(x)
    for train_idx, val_idx in kfold_indices(n, k):
        xt = [x[i] for i in train_idx]; yt = [y[i] for i in train_idx]
        xv = [x[i] for i in val_idx];   yv = [y[i] for i in val_idx]

        xs_std, ys_std, xsc, ysc = standardize_xy(xt, yt)
        w0, w1 = gradient_descent(xs_std, ys_std, lr=0.05, epochs=4000, patience=200)
        model = LinRegModel(w0=w0, w1=w1, x_scaler=xsc, y_scaler=ysc)

        yhat = model.predict(xv)
        r2s.append(r2_score(yv, yhat))
        maes.append(mae(yv, yhat))
        rmses.append(rmse(yv, yhat))

    return {
        "R2_mean": round(stats.mean(r2s), 4),
        "R2_std": round(stats.pstdev(r2s), 4),
        "MAE_mean": round(stats.mean(maes), 4),
        "RMSE_mean": round(stats.mean(rmses), 4),
    }

# ---------------------------- Plotting ----------------------------

def plot_fit(x_raw: List[float], y_raw: List[float], model: LinRegModel):
    plt.scatter(x_raw, y_raw, alpha=0.7)
    xs_sorted = sorted(x_raw)
    ys_line = model.predict(xs_sorted)
    plt.plot(xs_sorted, ys_line)
    plt.title("Fitted Line")
    plt.xlabel("Feature (X)")
    plt.ylabel("Target (Y)")
    plt.grid(True)
    plt.show()

def plot_residuals(x_raw: List[float], y_raw: List[float], model: LinRegModel):
    yhat = model.predict(x_raw)
    residuals = [a - b for a, b in zip(y_raw, yhat)]
    plt.scatter(yhat, residuals, alpha=0.7)
    plt.axhline(0)
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted Y")
    plt.ylabel("Residual (Y - Ŷ)")
    plt.grid(True)
    plt.show()

# ---------------------------- CLI ----------------------------

def choose(prompt: str, options: List[str]) -> str:
    print(prompt)
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}")
    while True:
        sel = input("Select number: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(options):
            return options[int(sel) - 1]
        print("Invalid selection. Try again.")

def save_model_card(path: str, model: LinRegModel, cv_metrics: Dict[str, float], meta: dict):
    card = {
        "model_type": "Univariate Linear Regression (standardized, GD)",
        "coefficients_standardized": {"w0": model.w0, "w1": model.w1},
        "x_scaler": asdict(model.x_scaler),
        "y_scaler": asdict(model.y_scaler),
        "cross_validation": cv_metrics,
        "metadata": meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"📝 Model card saved to: {path}")

def main():
    print("📈 Predictive Insight Engine — Day 7")
    csv_path = input("Enter CSV path (e.g., data.csv): ").strip()
    try:
        rows = load_csv(csv_path)
    except Exception as e:
        print("⚠️ Failed to load CSV:", e)
        return

    num_cols = numeric_columns(rows)
    if len(num_cols) < 2:
        print("Need at least 2 numeric columns (one feature, one target).")
        return

    feature = choose("Choose FEATURE (X):", num_cols)
    target  = choose("Choose TARGET (Y):",  [c for c in num_cols if c != feature])

    x_all = to_float_list(rows, feature)
    y_all = to_float_list(rows, target)

    if len(x_all) != len(y_all) or len(x_all) < 10:
        n = min(len(x_all), len(y_all))
        print(f"⚠️ After parsing, usable pairs: {n}. Need >= 10.")
        return

    # Optional outlier filtering
    if input("Remove outliers with z-score > 3? [Y/n]: ").strip().lower() in ("", "y", "yes"):
        before = len(x_all)
        x_all, y_all = zscore_filter(x_all, y_all, 3.0)
        print(f"Outlier filter: {before} → {len(x_all)} usable pairs.")

    # Cross-validation BEFORE fitting on full data
    cv = cross_val(x_all, y_all, k=5)
    print("\n🔍 5-fold Cross-Validation")
    for k, v in cv.items():
        print(f"  {k}: {v}")

    # Fit on all data
    xs_std, ys_std, xsc, ysc = standardize_xy(x_all, y_all)
    w0, w1 = gradient_descent(xs_std, ys_std, lr=0.05, epochs=6000, patience=400)
    model = LinRegModel(w0=w0, w1=w1, x_scaler=xsc, y_scaler=ysc)

    # Final metrics on all data (for reference)
    yhat_all = model.predict(x_all)
    R2 = round(r2_score(y_all, yhat_all), 4)
    MAE = round(mae(y_all, yhat_all), 4)
    RMSE = round(rmse(y_all, yhat_all), 4)

    print("\n✅ Fitted on full data")
    print(f"  R²: {R2} | MAE: {MAE} | RMSE: {RMSE}")

    # Plots
    if input("Show fitted-line plot? [Y/n]: ").strip().lower() in ("", "y", "yes"):
        plot_fit(x_all, y_all, model)
    if input("Show residuals plot? [Y/n]: ").strip().lower() in ("", "y", "yes"):
        plot_residuals(x_all, y_all, model)

    # Save model card
    if input("Save model card JSON? [Y/n]: ").strip().lower() in ("", "y", "yes"):
        path = f"Day-07/model_card_{feature}_to_{target}.json"
        meta = {
            "csv_path": csv_path,
            "feature": feature,
            "target": target,
            "outlier_filter": "z-score > 3",
            "notes": "Univariate linear regression with standardization and GD; includes 5-fold CV metrics.",
        }
        save_model_card(path, model, cv, meta)

    print("\n🏁 Done. This pipeline shows end-to-end modeling discipline. 🧠")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
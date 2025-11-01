# Day 6 — Smart Data Analyzer 📊
# Author: Stuart Abhishek
# Description:
# Loads a CSV dataset and automatically performs statistical analysis,
# correlation detection, and simple insight generation.

import csv
import statistics as stats
from collections import defaultdict
import math
import itertools
import matplotlib.pyplot as plt

def load_csv(path):
    """Load CSV file into list of dicts."""
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    return data

def numeric_columns(data):
    """Return names of columns that are numeric."""
    numeric = []
    for key in data[0].keys():
        try:
            float(data[0][key])
            numeric.append(key)
        except ValueError:
            pass
    return numeric

def column_values(data, col):
    vals = []
    for row in data:
        try:
            vals.append(float(row[col]))
        except ValueError:
            continue
    return vals

def correlation(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return 0
    mean_x, mean_y = stats.mean(xs), stats.mean(ys)
    num = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,ys))
    den = math.sqrt(sum((x-mean_x)**2 for x in xs) * sum((y-mean_y)**2 for y in ys))
    return round(num/den, 3) if den else 0

def analyze(data):
    """Compute basic stats and correlations."""
    numeric = numeric_columns(data)
    report = {}
    for col in numeric:
        vals = column_values(data, col)
        if not vals:
            continue
        report[col] = {
            "count": len(vals),
            "mean": round(stats.mean(vals),2),
            "median": round(stats.median(vals),2),
            "stdev": round(stats.stdev(vals),2) if len(vals)>1 else 0
        }
    # Correlations
    pairs = list(itertools.combinations(numeric, 2))
    cors = []
    for a,b in pairs:
        xs, ys = column_values(data,a), column_values(data,b)
        r = correlation(xs,ys)
        if abs(r) >= 0.5:
            cors.append((a,b,r))
    return report, cors

def print_report(report, cors):
    print("📈 Summary Statistics")
    for col,info in report.items():
        print(f"• {col}: mean={info['mean']}, median={info['median']}, stdev={info['stdev']}, n={info['count']}")
    print("\n🤝 Significant Correlations (|r| ≥ 0.5)")
    if cors:
        for a,b,r in sorted(cors, key=lambda x: -abs(x[2])):
            direction = "direct" if r>0 else "inverse"
            print(f"  {a} ↔ {b}: r = {r} ({direction} correlation)")
    else:
        print("  None strong enough.")
    print("\n✨ Insights:")
    if cors:
        a,b,r = max(cors, key=lambda x: abs(x[2]))
        print(f"  Strongest link: {a} and {b} ({r}). Consider exploring cause-effect relationship.")
    print("\nReport complete ✅")

def plot_histograms(data, numeric_cols):
    """Generate histograms for each numeric column."""
    for col in numeric_cols:
        vals = column_values(data,col)
        plt.hist(vals,bins=10,alpha=0.7)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

def main():
    print("📊 Smart Data Analyzer — Day 6")
    path = input("Enter CSV file path (e.g., data.csv): ").strip()
    try:
        data = load_csv(path)
    except Exception as e:
        print("⚠️ Could not load file:", e)
        return

    report, cors = analyze(data)
    print_report(report, cors)

    if input("Plot histograms? [y/N]: ").lower().startswith('y'):
        plot_histograms(data, numeric_columns(data))

if __name__ == "__main__":
    main()
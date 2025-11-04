# Day 9 — Clustering Insight Engine 🌐
# Author: Stuart Abhishek
#
# Implements K-Means clustering + optional PCA (from scratch).
# Evaluates inertia, convergence, and visualizes clusters.

import csv
import math
import random
import statistics as stats
import json
from typing import List, Tuple
import matplotlib.pyplot as plt

# ---------- Data utilities ----------

def load_csv(path: str) -> List[List[float]]:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            try:
                rows.append([float(x) for x in r])
            except ValueError:
                pass
    return rows

def zscore_normalize(data: List[List[float]]) -> List[List[float]]:
    cols = len(data[0])
    means = [stats.mean([r[i] for r in data]) for i in range(cols)]
    stds  = [stats.pstdev([r[i] for r in data]) or 1e-12 for i in range(cols)]
    return [[(r[i]-means[i])/stds[i] for i in range(cols)] for r in data]

# ---------- K-Means from scratch ----------

def euclid(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def kmeans(data: List[List[float]], k=3, max_iter=200, tol=1e-4):
    n = len(data)
    dims = len(data[0])
    rng = random.Random(42)
    centroids = random.sample(data, k)
    for it in range(max_iter):
        # Assign
        clusters = [[] for _ in range(k)]
        for x in data:
            dists = [euclid(x, c) for c in centroids]
            idx = dists.index(min(dists))
            clusters[idx].append(x)
        # Update
        new_centroids = []
        for pts in clusters:
            if pts:
                new_centroids.append([stats.mean(p[i] for p in pts) for i in range(dims)])
            else:
                new_centroids.append(random.choice(data))
        # Check convergence
        shift = max(euclid(a, b) for a, b in zip(centroids, new_centroids))
        centroids = new_centroids
        if shift < tol:
            break
    inertia = sum(min(euclid(x, c) ** 2 for c in centroids) for x in data)
    return centroids, clusters, inertia, it + 1

# ---------- PCA for visualization ----------

def pca_2d(data: List[List[float]]) -> List[List[float]]:
    # Compute covariance matrix
    n, d = len(data), len(data[0])
    means = [stats.mean([r[i] for r in data]) for i in range(d)]
    X = [[r[i] - means[i] for i in range(d)] for r in data]
    cov = [[0]*d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum(X[t][i]*X[t][j] for t in range(n)) / (n-1)
    # Power iteration to get first 2 eigenvectors (simple version)
    def power_iter(A, num_iter=100):
        v = [random.random() for _ in range(len(A))]
        for _ in range(num_iter):
            Av = [sum(A[i][j]*v[j] for j in range(len(A))) for i in range(len(A))]
            norm = math.sqrt(sum(a*a for a in Av))
            v = [a/norm for a in Av]
        return v
    v1 = power_iter(cov)
    # Deflate for v2 (orthogonal component)
    Av1 = [sum(cov[i][j]*v1[j] for j in range(d)) for i in range(d)]
    l1 = sum(v1[i]*Av1[i] for i in range(d))
    cov2 = [[cov[i][j] - l1*v1[i]*v1[j] for j in range(d)] for i in range(d)]
    v2 = power_iter(cov2)
    # Project
    proj = [[sum(x[i]*v1[i] for i in range(d)), sum(x[i]*v2[i] for i in range(d))] for x in X]
    return proj

# ---------- Visualization ----------

def plot_clusters(data2d: List[List[float]], labels: List[int], centroids2d: List[List[float]]):
    k = len(set(labels))
    colors = plt.cm.get_cmap("tab10", k)
    for i, p in enumerate(data2d):
        plt.scatter(p[0], p[1], color=colors(labels[i]))
    for i, c in enumerate(centroids2d):
        plt.scatter(c[0], c[1], marker="*", color="black", s=180, edgecolor="white")
        plt.text(c[0], c[1], f"C{i}", fontsize=10, color="black")
    plt.title("K-Means Clusters (PCA 2D projection)")
    plt.grid(True)
    plt.show()

# ---------- CLI ----------

def main():
    print("🌐 Day 9 — Clustering Insight Engine")
    path = input("Enter CSV path (numeric columns only): ").strip()
    try:
        data = load_csv(path)
    except Exception as e:
        print("Error loading file:", e)
        return
    if len(data) < 10:
        print("Dataset too small.")
        return
    data = zscore_normalize(data)
    k = int(input("Number of clusters (k): ") or "3")
    centroids, clusters, inertia, iters = kmeans(data, k=k)
    print(f"\n✅ K-Means finished in {iters} iterations.")
    print(f"Inertia (sum of squared distances): {round(inertia,2)}")
    sizes = [len(c) for c in clusters]
    print("Cluster sizes:", sizes)

    # Project for visualization
    proj = pca_2d(data)
    # Label each point by nearest centroid
    labels = []
    for x in data:
        dists = [euclid(x, c) for c in centroids]
        labels.append(dists.index(min(dists)))
    centroids2d = [ [stats.mean([proj[i][0] for i in range(len(proj)) if labels[i]==j]),
                     stats.mean([proj[i][1] for i in range(len(proj)) if labels[i]==j])]
                    for j in range(k)]
    if input("Show 2D plot? [Y/n]: ").lower() in ("","y","yes"):
        plot_clusters(proj, labels, centroids2d)

    # Save model card
    if input("Save model card JSON? [Y/n]: ").lower() in ("","y","yes"):
        meta = {
            "csv_path": path,
            "k": k,
            "inertia": inertia,
            "iterations": iters,
            "cluster_sizes": sizes,
            "notes": "K-Means + PCA visualization implemented from scratch."
        }
        with open("Day-09/model_card_kmeans.json","w",encoding="utf-8") as f:
            json.dump(meta,f,indent=2)
        print("📝 Saved Day-09/model_card_kmeans.json")

if __name__ == "__main__":
    main()
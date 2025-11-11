# Day 15 — Neural Network From Scratch + Visualizer 🧩
# Author: Stuart Abhishek
#
# A minimal 2-layer neural net (input → hidden → output) built only with NumPy.
# Trains on synthetic 2D data to classify points into two classes.
# Displays live loss curve and decision boundary.

import numpy as np
import matplotlib.pyplot as plt

# -------------------- Data --------------------
def make_data(n=300, seed=0):
    np.random.seed(seed)
    X = np.random.randn(n, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # circle vs inside
    return X, y.reshape(-1, 1)

# -------------------- Model --------------------
class NeuralNet:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    @staticmethod
    def sigmoid(z): return 1 / (1 + np.exp(-z))
    @staticmethod
    def d_sigmoid(a): return a * (1 - a)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, out):
        m = len(X)
        dZ2 = out - y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (1 - np.tanh(self.Z1)**2)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient step
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def compute_loss(self, y, out):
        m = len(y)
        eps = 1e-9
        return -np.mean(y*np.log(out+eps) + (1-y)*np.log(1-out+eps))

    def fit(self, X, y, epochs=1000, plot_every=50):
        losses = []
        for epoch in range(1, epochs+1):
            out = self.forward(X)
            loss = self.compute_loss(y, out)
            self.backward(X, y, out)
            losses.append(loss)
            if epoch % plot_every == 0 or epoch == 1:
                print(f"Epoch {epoch}: loss={loss:.4f}")
        return losses

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# -------------------- Visualization --------------------
def plot_results(X, y, model):
    # Decision boundary
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, preds, cmap="coolwarm", alpha=0.4)
    plt.scatter(X[:,0], X[:,1], c=y[:,0], cmap="bwr", edgecolor="k")
    plt.title("Decision Boundary")
    plt.show()

def main():
    X, y = make_data()
    net = NeuralNet(2, 6, 1, lr=0.3)
    losses = net.fit(X, y, epochs=1000, plot_every=100)

    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plot_results(X, y, net)

if __name__ == "__main__":
    main()
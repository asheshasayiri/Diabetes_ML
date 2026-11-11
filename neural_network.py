import numpy as np


class NeuralNetwork:
    """
    Feedforward Neural Network — NumPy only.
    Architecture: Input → 16 → 8 → 1 (sigmoid output)
    """

    def __init__(self, input_size, leaky_alpha=0.01, l2_lambda=0.001):
        np.random.seed(42)

        self.layer_sizes = [input_size, 16, 8, 1]
        self.L = len(self.layer_sizes)
        self.leaky_alpha = leaky_alpha
        self.l2_lambda = l2_lambda

        self.weights = []
        self.biases = []

        for i in range(self.L - 1):
            fan_in = self.layer_sizes[i]
            w = np.random.randn(
                self.layer_sizes[i],
                self.layer_sizes[i + 1]
            ) * np.sqrt(2.0 / fan_in)   # He initialization
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        # Best weights for early stopping
        self._best_weights = None
        self._best_biases = None

    # ========================
    # ACTIVATIONS
    # ========================
    def leaky_relu(self, Z):
        return np.where(Z > 0, Z, self.leaky_alpha * Z)

    def leaky_relu_derivative(self, Z):
        return np.where(Z > 0, 1.0, self.leaky_alpha)

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    # ========================
    # FORWARD
    # ========================
    def forward(self, X):
        """Returns sigmoid probability in [0, 1]."""
        self.A = [X]
        self.Z = []

        A = X
        for i in range(self.L - 2):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.leaky_relu(Z)
            self.Z.append(Z)
            self.A.append(A)

        # Output layer — sigmoid
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.sigmoid(Z)
        self.Z.append(Z)
        self.A.append(A)

        return A   # shape (m, 1), values in (0, 1)

    # ========================
    # LOSS
    # ========================
    def compute_loss(self, y_pred, y_true):
        y_true = y_true.reshape(-1, 1)
        m = y_true.shape[0]
        eps = 1e-8

        bce = -(1 / m) * np.sum(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )
        l2 = (self.l2_lambda / (2 * m)) * sum(
            np.sum(w ** 2) for w in self.weights
        )
        return bce + l2

    # ========================
    # BACKWARD
    # ========================
    def backward(self, X, y):
        y = y.reshape(-1, 1)
        m = X.shape[0]

        self.dW = [None] * (self.L - 1)
        self.dB = [None] * (self.L - 1)

        dZ = self.A[-1] - y   # dL/dZ for sigmoid + BCE

        for i in reversed(range(self.L - 1)):
            A_prev = self.A[i]
            self.dW[i] = (1 / m) * np.dot(A_prev.T, dZ) + \
                         (self.l2_lambda / m) * self.weights[i]
            self.dB[i] = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self.leaky_relu_derivative(self.Z[i - 1])

    # ========================
    # GRADIENT CLIPPING
    # ========================
    def clip_gradients(self, max_norm=5.0):
        for i in range(self.L - 1):
            norm = np.linalg.norm(self.dW[i])
            if norm > max_norm:
                self.dW[i] *= (max_norm / norm)

    # ========================
    # UPDATE
    # ========================
    def update_params(self, lr):
        for i in range(self.L - 1):
            self.weights[i] -= lr * self.dW[i]
            self.biases[i]  -= lr * self.dB[i]

    # ========================
    # TRAIN (with early stopping)
    # ========================
    def train(self, X, y, X_val, y_val,
              epochs=200, lr=0.01,
              batch_size=32, lr_decay=0.001,
              patience=20):

        y     = y.reshape(-1)
        y_val = y_val.reshape(-1)
        m = X.shape[0]

        train_losses = []
        val_losses   = []

        best_val_loss  = np.inf
        patience_count = 0

        for epoch in range(epochs):
            current_lr = lr / (1 + lr_decay * epoch)

            # Shuffle
            perm    = np.random.permutation(m)
            X_shuff = X[perm]
            y_shuff = y[perm]

            # Mini-batch
            for i in range(0, m, batch_size):
                X_batch = X_shuff[i:i + batch_size]
                y_batch = y_shuff[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)
                self.clip_gradients()
                self.update_params(current_lr)

            # Loss tracking
            train_pred = self.forward(X)
            val_pred   = self.forward(X_val)
            train_loss = self.compute_loss(train_pred, y)
            val_loss   = self.compute_loss(val_pred, y_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:>4} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                self._best_weights = [w.copy() for w in self.weights]
                self._best_biases  = [b.copy() for b in self.biases]
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                    # Restore best weights
                    self.weights = self._best_weights
                    self.biases  = self._best_biases
                    break

        return train_losses, val_losses

    # ========================
    # PREDICT
    # ========================
    def predict(self, X, threshold=0.5):
        """Returns binary class labels (0 or 1)."""
        probs = self.forward(X)
        return (probs >= threshold).astype(int).reshape(-1)

    def predict_proba(self, X):
        """
        Returns (m, 2) array: [P(class=0), P(class=1)].
        forward() already applies sigmoid — NO second sigmoid here.
        """
        p1 = self.forward(X).reshape(-1, 1)   # shape (m, 1)
        p0 = 1 - p1
        return np.hstack([p0, p1])             # shape (m, 2)

    # ========================
    # EVALUATE
    # ========================
    def evaluate(self, X, y, threshold=0.5):
        """Returns Accuracy, Precision, Recall, F1 — pure NumPy."""
        y_pred = self.predict(X, threshold)
        y_true = y.reshape(-1)
        m = len(y_true)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        accuracy    = (tp + tn) / m
        precision   = tp / (tp + fp + 1e-8)
        recall      = tp / (tp + fn + 1e-8)
        f1          = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        print(f"  Accuracy    : {accuracy:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")
        print(f"  F1 Score    : {f1:.4f}")
        print(f"  Specificity : {specificity:.4f}")
        print(f"  Confusion   : TP={tp} FP={fp} FN={fn} TN={tn}")

        return {
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "specificity": specificity
        }

    # ========================
    # GRADIENT CHECK
    # ========================
    def gradient_check(self, X, y, eps=1e-5):
        """
        Compares analytical gradients vs numerical finite-difference gradients.
        Relative error < 1e-5 confirms backprop is correct.
        Run once on a small sample before full training.
        """
        self.forward(X)
        self.backward(X, y)

        analytical = np.concatenate([dw.ravel() for dw in self.dW])
        numerical  = np.zeros_like(analytical)

        # Flatten all weights into one vector for perturbation
        flat_w = np.concatenate([w.ravel() for w in self.weights])
        shapes = [w.shape for w in self.weights]

        for i in range(len(flat_w)):
            # Restore + perturb +eps
            w_plus = flat_w.copy(); w_plus[i] += eps
            self._set_flat_weights(w_plus, shapes)
            loss_plus = self.compute_loss(self.forward(X), y)

            # Perturb -eps
            w_minus = flat_w.copy(); w_minus[i] -= eps
            self._set_flat_weights(w_minus, shapes)
            loss_minus = self.compute_loss(self.forward(X), y)

            numerical[i] = (loss_plus - loss_minus) / (2 * eps)

        # Restore original weights
        self._set_flat_weights(flat_w, shapes)

        num   = np.linalg.norm(analytical - numerical)
        denom = np.linalg.norm(analytical) + np.linalg.norm(numerical) + 1e-8
        rel_error = num / denom

        status = "PASS ✅" if rel_error < 1e-5 else "FAIL ❌"
        print(f"Gradient check relative error: {rel_error:.2e}  →  {status}")
        return rel_error

    def _set_flat_weights(self, flat_w, shapes):
        idx = 0
        for i, shape in enumerate(shapes):
            size = shape[0] * shape[1]
            self.weights[i] = flat_w[idx:idx + size].reshape(shape)
            idx += size

    # ========================
    # SAVE / LOAD
    # ========================
    def save(self, path="model/weights.pkl"):
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @staticmethod
    def load(path="model/weights.pkl"):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

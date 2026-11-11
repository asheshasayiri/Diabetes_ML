import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data_preprocessing import DataPreprocessor
from neural_network import NeuralNetwork


def plot_loss(train_losses, val_losses, path="model/loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses,   label="Validation Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Loss curve saved → {path}")


if __name__ == "__main__":

    os.makedirs("model", exist_ok=True)

    # ========================
    # 1. Data Preprocessing
    # ========================
    dp = DataPreprocessor("dataset/diabetes.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = dp.preprocess()

    print(f"\nData shapes:")
    print(f"  Train : {X_train.shape}  |  y: {y_train.shape}")
    print(f"  Val   : {X_val.shape}    |  y: {y_val.shape}")
    print(f"  Test  : {X_test.shape}   |  y: {y_test.shape}")

    # ========================
    # 2. Gradient Check (small sample)
    # ========================
    print("\n--- Gradient Check (first 20 samples) ---")
    check_model = NeuralNetwork(input_size=X_train.shape[1])
    check_model.gradient_check(X_train[:20], y_train[:20])

    # ========================
    # 3. Train
    # ========================
    print("\n--- Training ---")
    model = NeuralNetwork(input_size=X_train.shape[1])

    train_losses, val_losses = model.train(
        X_train, y_train,
        X_val,   y_val,
        epochs=300,
        lr=0.01,
        batch_size=32,
        lr_decay=0.001,
        patience=25,
    )

    # ========================
    # 4. Evaluate
    # ========================
    print("\n--- Train Performance ---")
    train_metrics = model.evaluate(X_train, y_train)

    print("\n--- Test Performance ---")
    test_metrics  = model.evaluate(X_test,  y_test)

    gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    print(f"\nOverfitting gap (train acc - test acc): {gap:.4f}")
    if gap > 0.07:
        print("⚠️  Warning: possible overfitting")

    # ========================
    # 5. Save Model
    # ========================
    model.save("model/weights.pkl")

    # ========================
    # 6. Plot Loss
    # ========================
    plot_loss(train_losses, val_losses)

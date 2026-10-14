
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "creditcard.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=== Credit Card Fraud Detection Using AutoEncoder ===")

    # Load the original anonymized credit card transactions.
    df = pd.read_csv(DATA_PATH)
    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values.")

    print(f"Total transactions: {len(df):,}")
    print(f"Normal transactions: {(df['Class'] == 0).sum():,}")
    print(f"Fraudulent transactions: {(df['Class'] == 1).sum():,}")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Preserve the rare fraud cases in both data splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y,
        random_state=RANDOM_STATE
    )

    # Train only on legitimate transactions to learn normal behavior.
    X_normal = X_train.loc[y_train == 0]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_normal)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining AutoEncoder model...")

    model = AutoEncoder(
        hidden_neuron_list=[64, 32],
        epoch_num=10,
        batch_size=256,
        contamination=0.002,
        random_state=RANDOM_STATE,
        verbose=2,
    )

    model.fit(X_train_scaled)

    # Higher reconstruction errors indicate unusual transactions.
    scores = model.decision_function(X_test_scaled)
    predictions = model.predict(X_test_scaled)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, scores):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, scores):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"]
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("AutoEncoder Fraud Detection")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    results = pd.DataFrame({
        "actual_class": y_test.to_numpy(),
        "predicted_class": predictions,
        "anomaly_score": scores,
    })
    results.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    print("\nResults saved in the outputs folder.")
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()

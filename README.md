# Credit Card Fraud Detection Using PyOD AutoEncoder

## Project Overview

This project uses deep learning to identify potentially fraudulent credit card transactions. It implements the AutoEncoder algorithm from the PyOD library, which detects anomalies by learning normal transaction patterns and measuring reconstruction errors.

The project was developed for Assignment 4 of MSCS-633: Advanced Artificial Intelligence at the University of the Cumberlands.

## Dataset

The experiment uses the anonymized Credit Card Fraud Detection dataset available on Kaggle.

Dataset: https://www.kaggle.com/datasets/whenamancodes/fraud-detection

The dataset contains 284,807 transactions, including 492 fraudulent transactions and 284,315 legitimate transactions.

The dataset is not included in this repository because of its size. Download `creditcard.csv` from Kaggle and place it in the `data` folder.

## Technologies Used

- Python
- PyOD AutoEncoder
- PyTorch
- Pandas and NumPy
- Scikit-learn
- Matplotlib

## Implementation

The dataset is divided into 80% training data and 20% testing data using stratified sampling. The AutoEncoder is trained using only legitimate transactions to learn normal transaction patterns.

The model uses two hidden layers with 64 and 32 neurons, a batch size of 256, and 10 training epochs. Transactions with unusually high reconstruction errors are flagged as potential fraud.

## Experiment Results

| Metric | Result |
|---|---|
| Accuracy | 99.76% |
| ROC-AUC | 0.9613 |
| Average Precision | 0.6192 |
| Fraud Precision | 0.41 |
| Fraud Recall | 0.82 |
| Fraud F1-Score | 0.54 |

The model correctly detected 80 of the 98 fraudulent transactions in the test dataset. It missed 18 fraudulent transactions and incorrectly flagged 117 legitimate transactions.

These results demonstrate that the AutoEncoder can identify unusual transaction patterns, although further threshold tuning could reduce false positives.

## How to Run

1. Clone or download this repository.
2. Download the dataset from Kaggle and place `creditcard.csv` inside `data/`.
3. Install dependencies using `python -m pip install -r requirements.txt`.
4. Run the program using `python fraud_detection.py`.
5. Review the terminal output and the generated confusion matrix in the `outputs/` folder.

## Conclusion

The experiment demonstrates how deep learning can support credit card fraud detection through anomaly detection. The AutoEncoder achieved strong fraud recall, identifying approximately 82% of fraudulent transactions in the test set. However, the false positives show that model evaluation should consider precision and recall rather than accuracy alone.
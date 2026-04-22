"""
Customer Churn Prediction with KNN
Author: Lucas Barrios
Description: Predict customer churn for a telecommunications company using KNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


# STEP 1: LOAD AND EXPLORE DATA

churn_df = pd.read_csv('data/telco_churn.csv')

print(f"Dataset shape: {churn_df.shape}")
print(f"\nColumns: {list(churn_df.columns)}")
print(f"\nTarget distribution:")
print(churn_df['Churn'].value_counts())

# STEP 2: PREPROCESSING

df_churn = churn_df.copy()

# Drop customerID — useless for prediction
df_churn = df_churn.drop('customerID', axis=1)

# TotalCharges is stored as text — convert to numeric
df_churn['TotalCharges'] = pd.to_numeric(df_churn['TotalCharges'], errors='coerce')
df_churn['TotalCharges'] = df_churn['TotalCharges'].fillna(df_churn['TotalCharges'].median())

# Convert target to binary
df_churn['Churn'] = df_churn['Churn'].map({'Yes': 1, 'No': 0})

# Convert binary Yes/No columns
binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in binary_columns:
    df_churn[col] = df_churn[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})

# Convert gender
df_churn['gender'] = df_churn['gender'].map({'Male': 1, 'Female': 0})

# Convert MultipleLines
df_churn['MultipleLines'] = df_churn['MultipleLines'].map(
    {'Yes': 1, 'No': 0, 'No phone service': 0}
)

# One-hot encode remaining categorical columns
df_churn = pd.get_dummies(df_churn, columns=['InternetService', 'Contract', 'PaymentMethod'])

# Convert bool columns to integers
bool_columns = df_churn.select_dtypes(include=['bool']).columns
df_churn[bool_columns] = df_churn[bool_columns].astype(int)

print(f"\nDataset shape after preprocessing: {df_churn.shape}")
print(f"Any remaining non-numeric columns: {df_churn.select_dtypes(include=['object']).columns.tolist()}")

# STEP 3: SPLIT DATA

X_churn = df_churn.drop('Churn', axis=1)
y_churn = df_churn['Churn']

X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

print(f"\nTraining set: {X_train_churn.shape[0]} samples")
print(f"Test set: {X_test_churn.shape[0]} samples")

# STEP 4: TRAIN KNN MODEL

knn_churn = KNeighborsClassifier(n_neighbors=5)
knn_churn.fit(X_train_churn, y_train_churn)
print("\nKNN classifier trained successfully!")

y_pred_churn = knn_churn.predict(X_test_churn)

# STEP 5: EVALUATE MODEL

train_acc = accuracy_score(y_train_churn, knn_churn.predict(X_train_churn))
test_acc = accuracy_score(y_test_churn, y_pred_churn)
precision = precision_score(y_test_churn, y_pred_churn)
recall = recall_score(y_test_churn, y_pred_churn)
conf_matrix = confusion_matrix(y_test_churn, y_pred_churn)

print("\n=== Churn Model Performance ===")
print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")

print("\n=== Confusion Matrix ===")
print("                  Predicted")
print("              No Churn  Churn")
print(f"Actual No Churn   {conf_matrix[0,0]:4d}     {conf_matrix[0,1]:4d}")
print(f"       Churn      {conf_matrix[1,0]:4d}     {conf_matrix[1,1]:4d}")

print("\n=== Classification Report ===")
print(classification_report(y_test_churn, y_pred_churn,
      target_names=['No Churn', 'Churn']))

# STEP 6: EXPERIMENT WITH K VALUES

print("\n" + "="*50)
print("EXPERIMENTING WITH DIFFERENT K VALUES")
print("="*50)

k_values = [1, 3, 5, 7, 9, 11, 15]
results_churn = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_churn, y_train_churn)
    y_pred_temp = knn_temp.predict(X_test_churn)
    acc = accuracy_score(y_test_churn, y_pred_temp)
    prec = precision_score(y_test_churn, y_pred_temp)
    rec = recall_score(y_test_churn, y_pred_temp)
    results_churn.append({'K': k, 'Accuracy': acc, 'Precision': prec, 'Recall': rec})
    print(f"K={k:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

results_churn_df = pd.DataFrame(results_churn)
best_k_churn = results_churn_df.loc[results_churn_df['Accuracy'].idxmax(), 'K']
print(f"\nBest K value: {best_k_churn} (Accuracy: {results_churn_df['Accuracy'].max():.4f})")

plt.figure(figsize=(10, 6))
plt.plot(results_churn_df['K'], results_churn_df['Accuracy'], marker='o', label='Accuracy')
plt.plot(results_churn_df['K'], results_churn_df['Precision'], marker='s', label='Precision')
plt.plot(results_churn_df['K'], results_churn_df['Recall'], marker='^', label='Recall')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Score')
plt.title('Churn KNN Performance vs K Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('data/churn_k_comparison.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'data/churn_k_comparison.png'")
plt.show()
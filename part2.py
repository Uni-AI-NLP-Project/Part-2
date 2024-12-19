import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Define the target column (Diabetes_binary for binary classification)
y_true = df['Diabetes_binary']

# Get the majority class (most frequent value in y_true)
majority_class = y_true.value_counts().idxmax()

# Create predictions (always predict the majority class)
y_pred = [majority_class] * len(y_true)

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)

# Set zero_division to 1 to avoid warnings and handle cases with no positive predictions
precision = precision_score(y_true, y_pred, zero_division=1)
recall = recall_score(y_true, y_pred, zero_division=1)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

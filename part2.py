import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the Reddit dataset
df = pd.read_csv('Reddit_Data.csv')

# Define the target column (category for sentiment classification)
y_true = df['category']

# Get the majority class (most frequent value in y_true)
majority_class = y_true.value_counts().idxmax()

# Create predictions (always predict the majority class)
y_pred = [majority_class] * len(y_true)

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)

# Set zero_division to 1 to avoid warnings and handle cases with no positive predictions
precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
recall = recall_score(y_true, y_pred, average='macro', zero_division=1)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

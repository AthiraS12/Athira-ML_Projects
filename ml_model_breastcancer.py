import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset without specifying column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data = pd.read_csv(url, header=None)

# Drop the ID column (first column)
data = data.drop(columns=0)

# Encode the target variable (second column, index 1)
data[1] = data[1].map({"M": 1, "B": 0})

# Separate features and target
X = data.drop(columns=1)
y = data[1]

# Drop rows with missing values in the target variable
data = data.dropna(subset=[1])
X = data.drop(columns=1)
y = data[1]

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Since we didn't specify column names, the features will be indexed as integers
feature_importance_df = pd.DataFrame({'Feature': range(X_train.shape[1]), 'Importance': importances}).sort_values(by='Importance', ascending=False)

top_features = feature_importance_df.head(10)['Feature'].values
print("Top 10 important features:\n", top_features)

# Create new datasets containing only the top features
X_train_top = X_train[:, top_features]
X_test_top = X_test[:, top_features]

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train_top, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_top)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


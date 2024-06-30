# main.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lib.dataloader import load_data, preprocess_data, select_top_features
from lib.helper import plot_confusion_matrix


def main():
    """
    Main function to load data, preprocess it, train models, and evaluate performance.

    This script performs the following steps:
    1. Loads and preprocesses data from a specified URL.
    2. Splits the data into training and testing sets.
    3. Normalizes the features using StandardScaler.
    4. Trains a Random Forest classifier to select top features.
    5. Trains a Logistic Regression model using selected top features.
    6. Evaluates the model's performance using accuracy, classification report, and confusion matrix.
    7. Plots the confusion matrix.
    """

    # Load and preprocess the data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    X, y = load_data(url)
    X, y = preprocess_data(X, y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Select top features
    top_features = select_top_features(rf, X_train)
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
    plot_confusion_matrix(conf_matrix)


if __name__ == "__main__":
    main()

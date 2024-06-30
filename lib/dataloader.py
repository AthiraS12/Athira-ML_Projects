# lib/dataloader.py

import pandas as pd
from sklearn.impute import SimpleImputer


def load_data(url):
    data = pd.read_csv(url, header=None)
    data = data.drop(columns=0)
    data[1] = data[1].map({"M": 1, "B": 0})
    return data.drop(columns=1), data[1]


def preprocess_data(X, y):
    # Ensure y has no NaN values
    valid_rows = y.notna()
    X = X[valid_rows]
    y = y[valid_rows]

    # Impute missing values in X
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    return X, y


def select_top_features(model, X):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": range(X.shape[1]), "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    top_features = feature_importance_df.head(10)["Feature"].values
    return top_features

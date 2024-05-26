import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import matplotlib.pyplot as plt

from preprocessing_data import clean_and_select_features


def build_and_evaluate_model(X_train, y_train, X_test, y_test, k):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print("Accuracy = {}".format(accuracy))
    print(sklearn.metrics.classification_report(y_test, y_pred, digits=5))

    return accuracy


X_train, y_train, X_test, y_test = clean_and_select_features()

k = 5  # Number of neighbors in KNN
accuracy = build_and_evaluate_model(X_train, y_train, X_test, y_test, k)

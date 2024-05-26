import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from forward_selection import forward_feature_selection
from clean_data import clean_data_train, clean_data_test
from airline_services import services

from preprocessing_data import split_to_train_test


def build_and_evaluate_model(X_train, y_train, X_test, y_test, k):
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    model = forward_feature_selection(KNeighborsClassifier(n_neighbors=k))
    model.fit(X_train, y_train)
    print(model.get_feature_names_out())
    # y_pred = model.predict(X_test_scaled)

    # accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    # print("Accuracy = {}".format(accuracy))
    # print(sklearn.metrics.classification_report(y_test, y_pred, digits=5))

    # return accuracy


X_train, y_train, X_test, y_test = split_to_train_test(clean_data_train(), clean_data_test(), services, 'satisfaction')

k = 10  # Number of neighbors in KNN
build_and_evaluate_model(X_train, y_train, X_test, y_test, k)

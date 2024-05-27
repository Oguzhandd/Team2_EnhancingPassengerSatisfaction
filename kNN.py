import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from forward_selection import forward_feature_selection
from clean_data import clean_data_train, clean_data_test
from airline_services import services

from preprocessing_data import split_to_train_test


def build_and_evaluate_model(X_train, y_train, X_test, y_test, k):

    model = forward_feature_selection(KNeighborsClassifier(n_neighbors=k))
    model.fit(X_train, y_train)
    print(model.get_feature_names_out())


X_train, y_train, X_test, y_test = split_to_train_test(clean_data_train(), clean_data_test(), services, 'satisfaction')

k = 10  # Number of neighbors in KNN
build_and_evaluate_model(X_train, y_train, X_test, y_test, k)

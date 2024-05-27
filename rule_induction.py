from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from airline_services import services
from preprocessing_data import split_to_train_test
from clean_data import clean_data_train, clean_data_test
from forward_selection import forward_feature_selection
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from airline_services import services
from preprocessing_data import split_to_train_test
from clean_data import clean_data_train, clean_data_test
from forward_selection import forward_feature_selection
import pandas as pd

def rule_induction(X, y):
    decision_tree = DecisionTreeClassifier()

    # Forward selection features
    selected_features = forward_feature_selection(decision_tree)

    selected_features.fit(X, y)

    selected_feature_indices = selected_features.get_support(indices=True)
    selected_feature_names = X.columns[selected_feature_indices]

    print("Selected Features:", selected_feature_names)

    X_selected = X[selected_feature_names]

    parameters = {'criterion': ['gini', 'entropy'],
                  'max_depth': [None, 10, 20, 30, 40, 50],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4]}

    grid_search = GridSearchCV(decision_tree, parameters, cv=5)
    grid_search.fit(X_selected, y)

    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    best_decision_tree = DecisionTreeClassifier(**best_params)
    best_decision_tree.fit(X_selected, y)

    y_pred = best_decision_tree.predict(X_selected)

    accuracy = accuracy_score(y, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y, y_pred))
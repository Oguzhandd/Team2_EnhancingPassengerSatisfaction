import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing_data import processed_data_train, processed_data_test


def select(model):
    train = processed_data_train()
    test = processed_data_test()

    features = ['Type_of_Travel','Inflight_wifi_service','Online_boarding','Seat_comfort','Flight_Distance',
                'Inflight_entertainment','On-board_service','Leg_room_service','Cleanliness','Checkin_service', 
                'Inflight_service', 'Baggage_handling']

    target = ['satisfaction']

    # Test and train
    X_train = train[features]
    y_train = train[target].to_numpy()
    X_test = test[features]
    y_test = test[target].to_numpy()

    selected_features = []
    remaining_features = list(range(X_train.shape[1]))
    k_features = 5

    while len(selected_features) < k_features:
        best_accuracy = 0
        best_feature = None
        
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_train_subset = X_train[:, current_features]
            
            # Train model 
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_train_subset)
            accuracy = accuracy_score(y_train, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
        
        # Select best feature
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

    print(f"Selected features: {selected_features}")

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Retrain model
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy with selected features: {accuracy}")


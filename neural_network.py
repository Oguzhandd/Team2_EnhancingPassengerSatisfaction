import sklearn.preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing_data import processed_data_train, processed_data_test
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import sklearn.metrics
import time


train = processed_data_train()
test = processed_data_test()


def LR(features):
    # Build model
    
    target = ['satisfaction']

    # Test and train
    X_train = train[features]
    y_train = train[target].to_numpy()
    X_test = test[features]
    y_test = test[target].to_numpy()

    # Normalize Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    params_nn = {'hidden_layer_sizes': (30,30,30),
             'activation': 'logistic',
             'solver': 'lbfgs',
             'max_iter': 100}

    model = MLPClassifier(**params_nn)

    verbose=True
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train.ravel(), verbose=0)
    else:
        model.fit(X_train,y_train.ravel())
    y_pred = model.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_pred) 
    time_taken = time.time()-t0


    # Plot ROC Curve
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return accuracy, roc_auc, time_taken

features = ['Type_of_Travel','Inflight_wifi_service','Online_boarding','Seat_comfort','Flight_Distance',
               'Inflight_entertainment','On-board_service','Leg_room_service','Cleanliness','Checkin_service', 
               'Inflight_service', 'Baggage_handling']
ac, roc, tt = LR(features)
print("Accuracy of Neural Network .", ac)
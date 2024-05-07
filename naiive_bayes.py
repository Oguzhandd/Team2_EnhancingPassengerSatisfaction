from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from clean_data import clean_data_test, clean_data_train
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics
import time


train = clean_data_train()
test = clean_data_test()

#Importance of feature using Wrapper Method

features = ['Type_of_Travel','Inflight_wifi_service','Online_boarding','Seat_comfort','Flight_Distance',
               'Inflight_entertainment','On-board_service','Leg_room_service','Cleanliness','Checkin_service', 
               'Inflight_service', 'Baggage_handling']

def NB(features):
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

    params_nb = {}

    model = GaussianNB(**params_nb)

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





ac, roc, tt = NB(features)
print("Accuracy of Naiive Bayes .", ac)
    

   

     




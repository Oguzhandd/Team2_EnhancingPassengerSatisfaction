from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing_data
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics
import time

#from eli5.sklearn import PermutationImportance


train = preprocessing_data.processed_data_train()
test = preprocessing_data.processed_data_test()

#Select top10 feature

r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(train)
modified_data = pd.DataFrame(r_scaler.transform(train), columns=train.columns)
modified_data.head()

#Importance of feature using Wrapper Method
X = train.drop('satisfaction', axis=1)
y = train['satisfaction']
selector = SelectFromModel(rf(n_estimators=100, random_state=0))
selector.fit(X, y)
support = selector.get_support()
features = X.loc[:,support].columns.tolist()
print(features)
print(rf(n_estimators=100, random_state=0).fit(X,y).feature_importances_)
#perm = PermutationImportance(rf(n_estimators=100, random_state=0).fit(X,y),random_state=1).fit(X,y)
#eli5.show_weights(perm, feature_names = X.columns.tolist())


# Build model
features = ['Type_of_Travel','Inflight_wifi_service','Online_boarding','Seat_comfort','Flight_Distance',
            'Inflight_entertainment','On-board_service','Leg_room_service','Cleanliness','Checkin_service', 
            'Inflight_service', 'Baggage_handling']
target = ['satisfaction']

# Split into test and train
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
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc))
print("Time taken = {}".format(time_taken))
print(sklearn.metrics.classification_report(y_test,y_pred,digits=5))


#cm = sklearn.metrics.confusion_matrix( X_test, y_test, normalize = 'all')
#disp_rc = sklearn.metrics.RocCurveDisplay.from_predictions(model, X_test, y_test)
#plt.show()
#disp_cm = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
#disp_cm.plot()

def get_result():
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Time taken = {}".format(time_taken))
    print(sklearn.metrics.classification_report(y_test,y_pred,digits=5))    
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

ac, roc, tt = get_result()
    

   

     




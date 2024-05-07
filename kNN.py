import pandas as pd
from clean_data import clean_data_train, clean_data_test
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import matplotlib.pyplot as plt

def clean_and_select_features(train, test):
    train_cleaned = clean_data_train()
    test_cleaned = clean_data_test()

    features = ['Online_boarding', 'Inflight_wifi_service', 'Baggage_handling', 'Inflight_entertainment']

    target = 'satisfaction'  # Target variable

    X_train = train_cleaned[features]
    y_train = train_cleaned[target].to_numpy()  # Convert Series to NumPy array
    X_test = test_cleaned[features]
    y_test = test_cleaned[target].to_numpy()  # Convert Series to NumPy array

    return X_train, y_train, X_test, y_test


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

    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Not satisfied', 'Satisfied'])
    plt.yticks([0, 1], ['Not satisfied', 'Satisfied'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

    return accuracy

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

X_train, y_train, X_test, y_test = clean_and_select_features(train, test)

k = 5  # Number of neighbors in KNN
accuracy = build_and_evaluate_model(X_train, y_train, X_test, y_test, k)

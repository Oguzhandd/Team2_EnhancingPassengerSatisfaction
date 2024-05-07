import pandas as pd
from clean_data import clean_data_train, clean_data_test
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import matplotlib.pyplot as plt

# Function to clean the dataset and select features
def clean_and_select_features(train, test):
    # Clean the dataset
    train_cleaned = clean_data_train()
    test_cleaned = clean_data_test()

    # Selecting features
    features = ['Online_boarding', 'Inflight_wifi_service', 'Baggage_handling', 'Inflight_entertainment']

    target = 'satisfaction'  # Target variable

    # Assigning selected features for training and test data
    X_train = train_cleaned[features]
    y_train = train_cleaned[target].to_numpy()  # Convert Series to NumPy array
    X_test = test_cleaned[features]
    y_test = test_cleaned[target].to_numpy()  # Convert Series to NumPy array

    return X_train, y_train, X_test, y_test


# Function to build and evaluate the KNN model
def build_and_evaluate_model(X_train, y_train, X_test, y_test, k):
    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Building the KNN model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculating accuracy
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print("Accuracy = {}".format(accuracy))
    print(sklearn.metrics.classification_report(y_test, y_pred, digits=5))

    # Plotting the Confusion Matrix
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

# Loading the dataset
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

# Cleaning and selecting features from the dataset
X_train, y_train, X_test, y_test = clean_and_select_features(train, test)

# Building and evaluating the model
k = 5  # Number of neighbors in KNN
accuracy = build_and_evaluate_model(X_train, y_train, X_test, y_test, k)

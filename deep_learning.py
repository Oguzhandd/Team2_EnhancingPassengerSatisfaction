from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from airline_services import services
from forward_selection import forward_feature_selection
from preprocessing_data import split_to_train_test
from clean_data import clean_data_train, clean_data_test
def deep_learning(X_train, y_train, X_test, y_test, features):
    model = Sequential()
    model.add(Dense(64, input_shape=(len(features),), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    y_pred_train = (model.predict(X_train) > 0.5).astype("int32")
    y_pred_test = (model.predict(X_test) > 0.5).astype("int32")

    print("Classification Report for Test Set:")
    print(classification_report(y_test, y_pred_test))
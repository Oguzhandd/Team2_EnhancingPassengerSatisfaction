import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from forward_selection import forward_feature_selection
from clean_data import clean_data_train, clean_data_test
from preprocessing_data import split_to_train_test
from airline_services import services

def deep_learning(X_train, y_train, X_test, y_test, features):
    model = Sequential()
    model.add(Dense(64, input_shape=(len(features),), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Modeli derleme süresini ölç
    start_time = time.time()
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    compile_time = time.time() - start_time

    print(f"Model compile süresi: {compile_time:.2f} saniye")

    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    y_pred_train = (model.predict(X_train) > 0.5).astype("int32")
    y_pred_test = (model.predict(X_test) > 0.5).astype("int32")

    print("Classification Report for Test Set:")
    print(classification_report(y_test, y_pred_test))


if __name__ == "__main__":
    df_train_cleaned = clean_data_train()
    df_test_cleaned = clean_data_test()

    X_train, y_train, X_test, y_test = split_to_train_test(df_train_cleaned, df_test_cleaned, services, 'satisfaction')

    forward_selector = forward_feature_selection(classifier=RandomForestClassifier())
    forward_selector = forward_selector.fit(X_train, y_train)

    selected_features_indices = forward_selector.get_support(indices=True)
    selected_features = X_train.columns[selected_features_indices].tolist()

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    deep_learning(X_train_selected, y_train, X_test_selected, y_test, selected_features)

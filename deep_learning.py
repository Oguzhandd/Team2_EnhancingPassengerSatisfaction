from keras import Sequential
from keras.src.layers import Dense
from keras.src.layers import Input
from scikeras.wrappers import KerasRegressor
from keras.src.optimizers import Adam
from forward_selection import forward_feature_selection
from clean_data import clean_data_train, clean_data_test
from preprocessing_data import split_to_train_test
from airline_services import services


def deep_learning():
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model


df_train_cleaned = clean_data_train()
df_test_cleaned = clean_data_test()

X_train, y_train, X_test, y_test = split_to_train_test(df_train_cleaned, df_test_cleaned, services, 'satisfaction')

deep = forward_feature_selection(deep_learning())
deep = deep.fit(X_train, y_train)
selected_features = deep.get_feature_names_out()
print(selected_features)

deep = deep_learning()
X_train, y_train, X_test, y_test = split_to_train_test(df_train_cleaned, df_test_cleaned, selected_features, 'satisfaction')
deep.fit(X_train, y_train)
y_pred = deep.predict(X_test)

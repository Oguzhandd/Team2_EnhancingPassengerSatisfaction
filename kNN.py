from sklearn.neighbors import KNeighborsClassifier
from forward_selection import forward_feature_selection
from clean_data import clean_data_train, clean_data_test
from airline_services import services

from preprocessing_data import split_to_train_test

df_train_cleaned = clean_data_train()
df_test_cleaned = clean_data_test()

X_train, y_train, X_test, y_test = split_to_train_test(df_train_cleaned, df_test_cleaned, services, 'satisfaction')
model = forward_feature_selection(KNeighborsClassifier(n_neighbors=10))
model = model.fit(X_train, y_train)
selected_features = model.get_feature_names_out()

print(selected_features)

x_train, y_train, x_test, y_test = split_to_train_test(df_train_cleaned,
                                                       df_test_cleaned, selected_features, 'satisfaction')
model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

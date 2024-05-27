from sklearn.ensemble import RandomForestClassifier
from forward_selection import forward_feature_selection
from clean_data import clean_data_test, clean_data_train
from preprocessing_data import split_to_train_test
from airline_services import services

df_train_cleaned = clean_data_train()
df_test_cleaned = clean_data_test()

features = services
x_train, y_train, x_test, y_test = split_to_train_test(df_train_cleaned, df_test_cleaned, features, 'satisfaction')
random_forest = forward_feature_selection(RandomForestClassifier(n_estimators=10))
random_forest = random_forest.fit(x_train, y_train)
selected_features = random_forest.get_feature_names_out()

print(selected_features)

x_train, y_train, x_test, y_test = split_to_train_test(df_train_cleaned,
                                                       df_test_cleaned, selected_features, 'satisfaction')
random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_train)

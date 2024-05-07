from sklearn.ensemble import RandomForestClassifier
from forward_selection import forward_feature_selection
from clean_data import clean_data_test, clean_data_train
from preprocessing_data import split_to_train_test
from airline_services import services

features = services
x_train, y_train, x_test, y_test = split_to_train_test(clean_data_train(), clean_data_test(), features, 'satisfaction')
random_forest = forward_feature_selection(RandomForestClassifier(n_estimators=15))
random_forest = random_forest.fit(x_train, y_train)

print(random_forest.get_feature_names_out())

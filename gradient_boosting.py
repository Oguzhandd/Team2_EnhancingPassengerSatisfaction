from sklearn.ensemble import GradientBoostingClassifier

from airline_services import services
from clean_data import clean_data_train, clean_data_test
from forward_selection import forward_feature_selection
from preprocessing_data import split_to_train_test

features = services
x_train, y_train, x_test, y_test = split_to_train_test(clean_data_train(), clean_data_test(), features, 'satisfaction')
gb_model = forward_feature_selection(GradientBoostingClassifier(n_estimators=10))
selected_features = gb_model.get_feature_names_out()

print(selected_features)

x_train, y_train, x_test, y_test = split_to_train_test(clean_data_train(),
                                                       clean_data_test(), selected_features, 'satisfaction')
gb_model = GradientBoostingClassifier(n_estimators=10)
gb_model.fit(x_train, y_train)
y_pred = gb_model.predict(x_train)

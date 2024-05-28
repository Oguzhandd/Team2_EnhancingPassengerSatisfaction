from sklearn import svm
from clean_data import clean_data_train, clean_data_test  
from preprocessing_data import split_to_train_test
from airline_services import services
from forward_selection import forward_feature_selection

df_train_cleaned = clean_data_train()
df_test_cleaned = clean_data_test()

X_train, y_train, X_test, y_test = split_to_train_test(df_train_cleaned, df_test_cleaned, services, 'satisfaction')

sup_vec_machine = forward_feature_selection(svm.SVC())
sup_vec_machine = sup_vec_machine.fit(X_train,y_train)
selected_features = sup_vec_machine.get_feature_names_out()
print(selected_features)

x_train, y_train, x_test, y_test = split_to_train_test(df_train_cleaned,
                                                       df_test_cleaned, selected_features, 'satisfaction')
supvm = svm.SVC()
supvm.fit(x_train, y_train)
y_pred = supvm.predict(x_test)

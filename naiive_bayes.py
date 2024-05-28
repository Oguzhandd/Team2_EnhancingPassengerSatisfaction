from sklearn.naive_bayes import GaussianNB
from clean_data import clean_data_train, clean_data_test  # Functions for data cleaning
from preprocessing_data import split_to_train_test
from airline_services import services
from forward_selection import forward_feature_selection

df_train_cleaned = clean_data_train()
df_test_cleaned = clean_data_test()

X_train, y_train, X_test, y_test = split_to_train_test(df_train_cleaned, df_test_cleaned, services, 'satisfaction')

params_nb = {}

model = GaussianNB(**params_nb)
naiive_bayes = forward_feature_selection(model)
naiive_bayes.fit(X_train,y_train)
selected_features = naiive_bayes.get_feature_names_out()
print(selected_features)


x_train, y_train, x_test, y_test = split_to_train_test(df_train_cleaned,
                                                       df_test_cleaned, selected_features, 'satisfaction')

nb = GaussianNB(**params_nb)
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
   

     




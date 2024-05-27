from sklearn.linear_model import LogisticRegression
from clean_data import clean_data_train, clean_data_test  # Functions for data cleaning
from preprocessing_data import split_to_train_test
from airline_services import services
from forward_selection import forward_feature_selection

df_train_cleaned = clean_data_train()
df_test_cleaned = clean_data_test()

X_train, y_train, X_test, y_test = split_to_train_test(df_train_cleaned, df_test_cleaned, services, 'satisfaction')

logistic_reg = forward_feature_selection(LogisticRegression())
logistic_reg = logistic_reg.fit(X_train, y_train)
print(logistic_reg.get_feature_names_out())

x_train, y_train, x_test, y_test = split_to_train_test(df_train_cleaned,
                                                       df_test_cleaned, selected_features, 'satisfaction')
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_train)

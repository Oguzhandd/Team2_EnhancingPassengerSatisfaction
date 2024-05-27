from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from clean_data import clean_data_train, clean_data_test  # Functions for data cleaning
from preprocessing_data import split_to_train_test
from airline_services import services
from forward_selection import forward_feature_selection

df_train_cleaned = clean_data_train()
df_test_cleaned = clean_data_test()

X_train, y_train, X_test, y_test = split_to_train_test(df_train_cleaned, df_test_cleaned, services, 'satisfaction')

params_nn = {'hidden_layer_sizes': (30,30,30),
             'activation': 'logistic',
             'solver': 'lbfgs',
             'max_iter': 100}

neuralnet = forward_feature_selection(MLPClassifier(**params_nn))
neuralnet = neuralnet.fit(X_train,y_train)
print(neuralnet.get_feature_names_out())

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from clean_data import clean_data_train, clean_data_test  # Functions for data cleaning

def select_features(df):
    return df[['Online_boarding', 'Inflight_wifi_service', 'Baggage_handling', 'Inflight_entertainment']]

df_train_cleaned = clean_data_train()
df_test_cleaned = clean_data_test()

X_train = select_features(df_train_cleaned)
y_train = df_train_cleaned['satisfaction']
X_test = select_features(df_test_cleaned)
y_test = df_test_cleaned['satisfaction']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Decision Tree Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_val, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

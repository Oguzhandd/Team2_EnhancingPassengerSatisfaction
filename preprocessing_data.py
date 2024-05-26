from airline_services import services
from clean_data import *
from sklearn.preprocessing import LabelEncoder

train = clean_data_train()
test = clean_data_test()

# For Train set
lencoders = {}
for col in train.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    train[col] = lencoders[col].fit_transform(train[col])

# For Test Set
lencoders_t = {}
for col in test.select_dtypes(include=['object']).columns:
    lencoders_t[col] = LabelEncoder()
    test[col] = lencoders_t[col].fit_transform(test[col])
# Detect outliers
Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1

# Removal of outliers from dataset
train = train[~((train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))).any(axis=1)]


def processed_data_train():
    return train


# return cleaned data of test set
def processed_data_test():
    return test


def split_to_train_test(train, test, features, target):
    train_cleaned = train
    test_cleaned = test

    features = features

    target = target  # Target variable

    X_train = train_cleaned[features]
    y_train = train_cleaned[target].to_numpy()  # Convert Series to NumPy array
    X_test = test_cleaned[features]
    y_test = test_cleaned[target].to_numpy()  # Convert Series to NumPy array

    return X_train, y_train, X_test, y_test

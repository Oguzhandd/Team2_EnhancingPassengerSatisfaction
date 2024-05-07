import pandas as pd

# Load data
df1 = pd.read_csv("datasets/train.csv")
df2 = pd.read_csv("datasets/test.csv")

# Drop Null raws
df1 = df1.dropna()
df2 = df2.dropna()


# return cleaned data of train set
def clean_data_train():
    return df1


# return cleaned data of test set
def clean_data_test():
    return df2

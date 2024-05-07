import pandas as pd

# Load data
df1 = pd.read_csv("datasets/train.csv")
df2 = pd.read_csv("datasets/test.csv")
#
# print("Train set:\n")
# print(len(df1))
# print(df1.isnull().sum())
# print("\n")
# print("Test set:\n")
# print(len(df2))
# print(df2.isnull().sum())
# print("\n\n")


# Drop Null Raws
df1 = df1.dropna()
df2 = df2.dropna()


# return cleaned data of train set
def clean_data_train():
    return df1


# return cleaned data of test set
def clean_data_test():
    return df2

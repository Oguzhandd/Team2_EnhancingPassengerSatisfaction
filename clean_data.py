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

df1 = df1.dropna()
df2 = df2.dropna()

# print("Train set (no nulls):\n")
# print(len(df1))
# print(df1.isnull().sum())
# print("\n")
# print("Test set (no nulls):\n")
# print(len(df2))
# print(df2.isnull().sum())
# print("\n\n")
print("Total dataset: " + str(len(df1) + len(df2)))


# return cleaned data of test set
def clean_data_test():
    return df1


# return cleaned data of train set
def clean_data_train():
    return df2

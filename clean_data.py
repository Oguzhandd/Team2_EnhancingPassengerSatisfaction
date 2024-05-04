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



#Drop Unnamed: 0 column
df1 = df1.drop('Unnamed: 0',axis=1)
df2 = df2.drop('Unnamed: 0',axis=1)

#Change blanks with '_'
df1.columns =[c.replace(' ','_') for c in df1.columns]
df2.columns =[c.replace(' ','_') for c in df2.columns]

# Imputing missing value with mean - Train and Test
df1['Arrival_Delay_in_Minutes'] = df1['Arrival_Delay_in_Minutes'].fillna(df1['Arrival_Delay_in_Minutes'].mean())
df2['Arrival_Delay_in_Minutes'] = df2['Arrival_Delay_in_Minutes'].fillna(df2['Arrival_Delay_in_Minutes'].mean())

# Replace NaN with mode for categorical variables - Train and Test
df1['Gender'] = df1['Gender'].fillna(df1['Gender'].mode()[0])
df1['Customer_Type'] = df1['Customer_Type'].fillna(df1['Customer_Type'].mode()[0])
df1['Type_of_Travel'] = df1['Type_of_Travel'].fillna(df1['Type_of_Travel'].mode()[0])
df1['Class'] = df1['Class'].fillna(df1['Class'].mode()[0])
df2['Gender'] = df2['Gender'].fillna(df2['Gender'].mode()[0])
df2['Customer_Type'] = df2['Customer_Type'].fillna(df2['Customer_Type'].mode()[0])
df2['Type_of_Travel'] = df2['Type_of_Travel'].fillna(df2['Type_of_Travel'].mode()[0])
df2['Class'] = df2['Class'].fillna(df2['Class'].mode()[0])


# print("Train set (no nulls):\n")
# print(len(df1))
# print(df1.isnull().sum())
# print("\n")
# print("Test set (no nulls):\n")
# print(len(df2))
# print(df2.isnull().sum())
# print("\n\n")
print("Total dataset: " + str(len(df1) + len(df2)))


# return cleaned data of train set
def clean_data_train():
    return df1


# return cleaned data of test set
def clean_data_test():
    return df2

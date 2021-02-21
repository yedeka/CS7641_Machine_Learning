from sklearn.preprocessing import LabelEncoder
import pandas as pd

"""
This method is used to load and analyze the wine data data set to cleanse it further for the classification problem
"""


def loadExploreDS():
    incomeDF = pd.read_csv("income_evaluation.csv")
    ds_shape = incomeDF.shape
    print("ds_shape")
    print(ds_shape)
    columns = incomeDF.columns
    print("columns")
    print(columns)
    # Strip the column names of spaces
    for col in incomeDF.columns:
        col_stripped = col.strip()
        incomeDF = incomeDF.rename(columns={col: col_stripped})
    # Drop unrelated columns with large number of variants
    incomeDF = incomeDF.drop(["capital-gain", "capital-loss", "fnlwgt", "native-country"], axis=1)
    # Strip strings of the categorical values from the dataset to ensure that the conversion does not yield false discrete values
    for column in incomeDF[["education", "marital-status", "occupation", "race", "relationship", "sex", "workclass"]]:
        incomeDF[column] = incomeDF[column].str.strip()

    print("Education Unique => ", incomeDF["education"].unique())
    print("Marital Status unique => ", incomeDF["marital-status"].unique())
    print("Occupation unique => ", incomeDF["occupation"].unique())
    print("Race unique => ", incomeDF["race"].unique())
    print("Relationship Unique => ", incomeDF["relationship"].unique())
    print("Sex Unique => ", incomeDF["sex"].unique())
    print("Workclass unique => ", incomeDF["workclass"].unique())

    # Since one hot encoder does not accept strings we will use label encoding
    le = LabelEncoder()

    incomeDF['education'] = le.fit_transform(incomeDF['education'])
    incomeDF['marital-status'] = le.fit_transform(incomeDF['marital-status'])
    incomeDF['occupation'] = le.fit_transform(incomeDF['occupation'])
    incomeDF['race'] = le.fit_transform(incomeDF['race'])
    incomeDF['relationship'] = le.fit_transform(incomeDF['relationship'])
    incomeDF['sex'] = le.fit_transform(incomeDF['sex'])
    incomeDF['workclass'] = le.fit_transform(incomeDF['workclass'])
    incomeDF['income'] = le.fit_transform(incomeDF['income'])

    print("Income unique => ", incomeDF["income"].unique())
    print("Distribution => ", incomeDF["income"].value_counts())

    print(incomeDF.head(8))

    return incomeDF

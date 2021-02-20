from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import loadWineDataSet

"""
This method is used to load and analyze the wine data data set to cleanse it further for the classification problem
"""

def loadExploreDS():
    wds = loadWineDataSet()
    ds_shape = wds.shape
    wds.rename(
        columns={'fixed acidity': 'fixed_acidity', 'citric acid': 'citric_acid', 'volatile acidity': 'volatile_acidity',
                 'residual sugar': 'residual_sugar', 'free sulfur dioxide': 'free_sulfur_dioxide',
                 'total sulfur dioxide': 'total_sulfur_dioxide'}, inplace=True)
    columns = wds.columns
    uniqueY = wds['quality'].unique()
    valueCnt = wds['quality'].value_counts()
    missingvals = wds.isnull().sum()
    # Create Classification version of target variable
    wds['goodquality'] = [1 if x >=6 else 0 for x in wds['quality']]
    qualityCounts = wds['goodquality'].value_counts()
    # print(qualityCounts)

    corr = wds.corr()['quality'].sort_values(ascending=False)
    print(corr)
    # print(abs(corr) > 0.2)
    # corr.plot(kind='bar')
    # matplotlib.pyplot.subplots(figsize=(15, 10))
    # sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
    #            cmap=sns.diverging_palette(220, 20, as_cmap=True))

    wineDict = {
        "dataSet": wds,
        "shape": ds_shape,
        "missingVals": missingvals,
        "columns": columns,
        "ydistribution": qualityCounts
    }
    return wineDict

"""
This method is used to convert the multicategory data into only two categories for simplification.
"""

def prepareDataForModelling(wineDF):
    features = wineDF.drop(['quality','goodquality'], axis=1)
    output = wineDF['goodquality']
    print(features.shape)
    print(output.shape)
    # Normalize feature variables
    normalized_features = StandardScaler().fit_transform(X=features)
    return {
        "features": normalized_features,
        "output": output
    }

    '''X_train, X_test, y_train, y_test = train_test_split(normalized_features, output, test_size=0.2, random_state=0)
    return {
        "X_train" : X_train,
        "X_test" : X_test,
        "y_train" : y_train,
        "y_test":  y_test,
        "features": normalized_features,
        "output": output
    }'''


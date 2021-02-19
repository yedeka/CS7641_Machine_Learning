import matplotlib
from numpy import mean, std, linspace
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RepeatedKFold, GridSearchCV, \
    StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.model_selection import LearningCurve

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
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, output, test_size=0.2, random_state=0)
    return {
        "X_train" : X_train,
        "X_test" : X_test,
        "y_train" : y_train,
        "y_test":  y_test,
        "features": normalized_features,
        "output": output
    }

"""
This method is used to perform grid search analysis in order to find out the maximum depth for the decision tree to be used for prunning.
"""

def performgridSearch(data):
    x_train = data["X_train"]
    y_train = data["y_train"]
    x_test = data["X_test"]
    y_test = data["y_test"],
    # prepare the cross-validation procedure
    # cv = KFold(n_splits=10, random_state=1, shuffle=True)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = DecisionTreeClassifier(random_state=1)
    # Perform grid search
    param_grid = {'max_depth': [3, 5, 6, 7]}
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=-1)
    # fitting the model for grid search
    grid.fit(x_train, y_train)
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(x_test)
    # print classification report
    print(classification_report)

    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(data["features"], data["output"])  # Fit the data to the visualizer
    visualizer.show()

    # evaluate model
    scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    # perform grid search
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

"""
This method is used for performing actual decision tree analysis on the data to generate the final results with hyper parameters
obtained from grid search
"""
def analyzeDecisionTree(dataSet):
    print('Calling analyze tree')
    x_train = dataSet["X_train"]
    y_train = dataSet["y_train"]
    x_test = dataSet["X_test"]
    y_test = dataSet["y_test"]
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model with hyper parameters obtained from grid search
    model = DecisionTreeClassifier(random_state=1, max_depth=7)
    model.fit(x_train, y_train)
    y_pred1 = model.predict(x_test)
    print(classification_report(y_test, y_pred1))
    print('Done with analyze tree')

    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(dataSet["features"], dataSet["output"])  # Fit the data to the visualizer
    visualizer.show()



"""
Main method to perform all the decision tree related tasks on the wine dataset.
"""

def performDecisionTree():
    winedict = loadExploreDS()
    dataset = winedict["dataSet"]
    preppedData = prepareDataForModelling(dataset)
    performgridSearch(preppedData)
    analyzeDecisionTree(preppedData)
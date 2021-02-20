import matplotlib
from numpy import mean, std, linspace
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RepeatedKFold, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.model_selection import LearningCurve

from wineQualityUtil import loadExploreDS, prepareDataForModelling
'''
This method is used to perform decision tree learning to get the baseline scores for the model
'''

def performDecisionTreeBaseline(features, output, test_population):
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=test_population)
    # Aplly decision tree without any hyper parameter tuning
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred1 = model.predict(x_test)
    print("Baseline Data Start ------------------------------------------------------------------")
    print(classification_report(y_test, y_pred1))
    print("Baseline Data End ------------------------------------------------------------------")
    # Plotting the learning curve for the baseline model
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(features, output)  # Fit the data to the visualizer
    visualizer.show()


"""
This method is used to perform grid search analysis in order to find out the maximum depth for the decision tree to be used for prunning.
"""

def performgridSearch(features, output, testpopulation):
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=testpopulation)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = DecisionTreeClassifier()
    # Perform grid search
    param_grid = {'max_depth': [3, 5, 6, 7, 9, 10]}
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
    visualizer.fit(features, output)  # Fit the data to the visualizer
    visualizer.show()

'''
Find out the performance of the tuned model
'''
def performDecisionTreeTuned(features, output, test_population):
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=test_population)
    # Aplly decision tree without any hyper parameter tuning
    model = DecisionTreeClassifier(random_state=1, max_depth=7)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(y_test, y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = float(cm.diagonal().sum()) / len(y_test)
    # Plotting the learning curve for the baseline model
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(features, output)  # Fit the data to the visualizer
    visualizer.show()


"""
Main method to perform all the decision tree related tasks on the wine dataset.
"""

def performDecisionTree():
    winedict = loadExploreDS()
    dataset = winedict["dataSet"]
    preppedData = prepareDataForModelling(dataset)
    features = preppedData["features"]
    output = preppedData['output']
    test_population = 0.2
    performDecisionTreeBaseline(features, output, test_population)
    performgridSearch(features, output, test_population)
    performDecisionTreeTuned(features, output, test_population)
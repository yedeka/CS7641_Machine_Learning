from matplotlib import axes
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.model_selection import LearningCurve, ValidationCurve

import numpy as np

from termDepositUtil import loadExploreDataSet, clean_data

'''
Gather the baseline parameters for decision tree without any tuning and grid searching
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
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(features, output)  # Fit the data to the visualizer
    visualizer.show()

'''
perform grid search to find out the best hyperparameter to be used for tuning the decision tree
'''
def performGridSearch(features, output, testpopulation):
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=testpopulation)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = DecisionTreeClassifier()
    # Perform grid search
    param_grid = {'max_leaf_nodes': [30,40,50,80,100,130,150,185,187,200,210,220], 'splitter':['random']}
    # param_grid = {'min_samples_leaf': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'splitter': ["random"]}
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=-1)
    # fitting the model for grid search
    grid.fit(x_train, y_train)
    print("Grid search Data Start ------------------------------------------------------------------")
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(x_test)
    # print classification report
    print(grid.best_score_)
    print("Grid search Data End ------------------------------------------------------------------")

'''
Find out the performance of the tuned model
'''
def performDecisionTreeTuned(features, output, test_population):
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=test_population)
    # Aplly decision tree without any hyper parameter tuning
    model = DecisionTreeClassifier(max_leaf_nodes= 100, splitter='random')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(y_test, y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    # Plotting the learning curve for the baseline model
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(features, output)  # Fit the data to the visualizer
    visualizer.show()

    viz = ValidationCurve(model, param_name='max_leaf_nodes',
                          param_range=[30,40,50,80,100,130,150,185,187,200,210,220], cv=10, scoring="r2")
    viz.fit(features, output)
    viz.show()

def performDecisionTree():
    bankDS = loadExploreDataSet()
    dataset = bankDS["dataset"]
    # plotCatClmns(dataset)
    # plotNumClmns(dataset)
    # Cleanse the data by dropping irrelevant columns, imputing the noisy columns and converting truthy columns to corresponding binary values
    cleaned_data = clean_data(dataset)
    features = cleaned_data.drop(['deposit_bool'], axis=1)
    output = cleaned_data['deposit_bool']
    test_population = 0.2
    performDecisionTreeBaseline(features, output, test_population)
    performGridSearch(features, output, test_population)
    performDecisionTreeTuned(features, output, test_population)

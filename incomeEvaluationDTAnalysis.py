from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.model_selection import LearningCurve, ValidationCurve

import incomeEvaluationUtil
import numpy as np


def performBaselineDT(x_train, x_test, y_train, y_test):
    # Aplly decision tree without any hyper parameter tuning
    model = DecisionTreeClassifier(random_state=120)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Baseline Data Start ------------------------------------------------------------------")
    print(classification_report(y_test, y_pred))
    print("Baseline Data End ------------------------------------------------------------------")
    # Plotting the learning curve for the baseline model
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, train_sizes=sizes, n_jobs=4, scoring='accuracy'
    )
    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show()


def performGridSearch(x_train, x_test, y_train, y_test):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = DecisionTreeClassifier(random_state=120)
    # Perform grid search
    # param_grid = {'max_leaf_nodes': [30,40,50,80,100,130,150,185,187,200,210,220]}
    # param_grid = {'min_samples_leaf': [10,20,30,40,50,60,70,80,90,100]}
    param_grid = {'max_depth': [1,2,3,4,5,6,7,8,9,10]}

    grid = GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=-1, scoring='accuracy')
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
def performDecisionTreeTuned( x_train, x_test, y_train, y_test):
    # Aplly decision tree without any hyper parameter tuning
    model = DecisionTreeClassifier(max_depth= 7, random_state=120)
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
    cv = StratifiedKFold(n_splits=12)
    sizes = np.linspace(0.3, 1.0, 10)
    visualizer = LearningCurve(
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(x_train, y_train)
    visualizer.show()

    viz = ValidationCurve(model, param_name='max_depth',
                          param_range=[3,5,7,9,11], cv=10, scoring='accuracy')
    viz.fit(x_train, y_train)
    viz.show()



def performDecisionTree():
    incomeDF = incomeEvaluationUtil.loadExploreDS()
    features = incomeDF.drop(['income'], axis=1)
    output = incomeDF['income']
    testPopulation = 0.2
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=testPopulation)
    performBaselineDT(x_train, x_test, y_train, y_test)
    performGridSearch(x_train, x_test, y_train, y_test)
    performDecisionTreeTuned(x_train, x_test, y_train, y_test)
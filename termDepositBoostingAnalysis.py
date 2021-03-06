from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, StratifiedKFold
from yellowbrick.model_selection import LearningCurve, ValidationCurve

import numpy as np
from termDepositUtil import loadExploreDataSet, clean_data

def performBoostingBaseline(x_train, x_test, y_train, y_test):
    # Create adaboost classifer object
    model = AdaBoostClassifier(random_state=45)
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
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show()

'''
perform grid search to find out the best hyperparameter to be used for tuning the decision tree
'''
def performGridSearch(x_train, x_test, y_train, y_test):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = AdaBoostClassifier(random_state=45)
    # Perform grid search
    param_grid = {'n_estimators': range(100,3000,100), 'learning_rate':[0.0001,0.001,0.01,0.1,0.5,0.7]}
    # param_grid = {'n_estimators': range(100, 3000, 100)}
    # svc = SVC(probability=True, kernel='linear')
    # param_grid = {'n_estimators': range(100, 2100, 100)}
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=20)
    # fitting the model for grid search
    grid.fit(x_train, y_train)
    print("Grid search Data Start ------------------------------------------------------------------")
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(x_test)
    # print classification report
    print(grid.best_score_)
    print("Grid search Data End ------------------------------------------------------------------")

def performBoostingTuned(x_train, x_test, y_train, y_test):
    # Aplly decision tree without any hyper parameter tuning
    # model = AdaBoostClassifier(n_estimators=300, learning_rate=0.5)
    # model = AdaBoostClassifier(n_estimators=1400, learning_rate=0.1)
    model = AdaBoostClassifier(n_estimators=2200, learning_rate=0.1,random_state=45)
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
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show()

    viz = ValidationCurve(model, param_name='n_estimators',
                          param_range=[2000,2100,2200,2300,2400], cv=10, scoring="r2")
    '''viz = ValidationCurve(model, param_name='n_estimators',
                          param_range=[100,120,140,160,180,200,220,240,260,300], cv=10, scoring="r2")'''
    viz.fit(x_train, y_train)
    viz.show()


def performBoosting():
    bankDS = loadExploreDataSet()
    dataset = bankDS["dataset"]
    cleaned_data = clean_data(dataset)
    features = cleaned_data.drop(['deposit_bool'], axis=1)
    output = cleaned_data['deposit_bool']
    test_population = 0.2
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=test_population)
    performBoostingBaseline(x_train, x_test, y_train, y_test)
    performGridSearch(x_train, x_test, y_train, y_test)
    performBoostingTuned(x_train, x_test, y_train, y_test)

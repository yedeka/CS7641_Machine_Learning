'''
This function performs the KNN classification without tuning to determine the baseline results for KNN classification
'''
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold, GridSearchCV
import numpy as np
from yellowbrick.model_selection import LearningCurve, ValidationCurve

from termDepositUtil import loadExploreDataSet, clean_data
'''
Perform baseline analysis for SVM model
'''

def performSVMBaseLine(x_train, x_test, y_train, y_test):
    model = svm.SVC(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Baseline Data Start ------------------------------------------------------------------")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix baseline")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Baseline Data End ------------------------------------------------------------------")
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=5)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show()

def performGridSearch(x_train, x_test, y_train, y_test):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = svm.SVC(random_state=42)
    # Perform grid search
    # param_grid = {'gamma': [0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001]}
    param_grid = {'gamma': [0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000007, 0.000008, 0.000009, 0.00001]}
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=-1,scoring='accuracy')
    # fitting the model for grid search
    grid.fit(x_train, y_train)
    print("Grid search Data Start ------------------------------------------------------------------")
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(x_test)
    # print classification report
    print(grid.best_score_)
    print("Grid search Data End ------------------------------------------------------------------")

def performSVNTuned(x_train, x_test, y_train, y_test):
    # Aplly decision tree without any hyper parameter tuning
    #model = svm.SVC(random_state=42,gamma=0.00001)
    model = svm.SVC(random_state=42, gamma=0.000004)
    # model = svm.SVC(random_state=42, gamma=0.000008)
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
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show()

    viz = ValidationCurve(model, param_name='gamma',
                          param_range=[0.000002, 0.000003,0.000004,0.000005,0.000006], cv=10,
                          scoring="accuracy")
    viz.fit(x_train, y_train)
    viz.show()


def performSVM():
    bankDS = loadExploreDataSet()
    dataset = bankDS["dataset"]
    cleaned_data = clean_data(dataset)
    features = cleaned_data.drop(['deposit_bool'], axis=1)
    output = cleaned_data['deposit_bool']
    test_population = 0.2
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=test_population)
    performSVMBaseLine(x_train, x_test, y_train, y_test)
    performGridSearch(x_train, x_test, y_train, y_test)
    performSVNTuned(x_train, x_test, y_train, y_test)

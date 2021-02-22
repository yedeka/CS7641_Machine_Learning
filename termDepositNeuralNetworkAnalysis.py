from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from yellowbrick.model_selection import LearningCurve, ValidationCurve
from termDepositUtil import loadExploreDataSet, clean_data

import numpy as np

def performNNBaseLine(x_train, x_test, y_train, y_test):
    model = MLPClassifier(random_state = 50)
    model.fit(x_train, y_train)

    predict_train = model.predict(x_train)
    predict_test = model.predict(x_test)

    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))


    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=5)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv,scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show()

def performGridSearch(x_train, x_test, y_train, y_test):
    print("Performing grid search")
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = MLPClassifier(random_state = 50)
    # Perform grid search
    # param_grid = {'momentum': [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]}
    # param_grid = {'alpha': [0.0001, 0.05]}
    param_grid = {'activation':['identity', 'logistic', 'tanh','relu'], 'solver':['lbfgs', 'sgd','adam']}
    grid = GridSearchCV(model, param_grid, scoring='accuracy',refit=True, verbose=3, n_jobs=-1)
    # fitting the model for grid search
    grid.fit(x_train, y_train)
    print("Grid search Data Start ------------------------------------------------------------------")
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(x_test)
    # print classification report
    print(grid.best_score_)
    print("Grid search Data End ------------------------------------------------------------------")

def performNNTuned(x_train, x_test, y_train, y_test):
    # Aplly decision tree with hyper parameter tuning
    model = MLPClassifier(activation='tanh')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(y_test, y_pred))
    print("Tuned Data End ------------------------------------------------------------------")
    print("Tuned Data Confusion Matrix ------------------------------------------------------------------")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = float(cm.diagonal().sum()) / len(y_test)
    # Plotting the learning curve for the tuned model
    cv = StratifiedKFold(n_splits=12)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show()

    viz = ValidationCurve(model, param_name='activation',
                          param_range=['identity', 'logistic', 'tanh'], scoring='accuracy' ,cv=10)
    viz.fit(x_train, y_train)
    viz.show()



def performNN():
    bankDS = loadExploreDataSet()
    dataset = bankDS["dataset"]
    cleaned_data = clean_data(dataset)
    features = cleaned_data.drop(['deposit_bool'], axis=1)
    output = cleaned_data['deposit_bool']
    test_population = 0.2
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=test_population)
    performNNBaseLine(x_train, x_test, y_train, y_test)
    performGridSearch(x_train, x_test, y_train, y_test)
    performNNTuned(x_train, x_test, y_train, y_test)
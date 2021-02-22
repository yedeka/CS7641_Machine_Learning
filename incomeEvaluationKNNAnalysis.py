from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.model_selection import LearningCurve, ValidationCurve
import numpy as np
import incomeEvaluationUtil

def performKNNBaseline(x_train, x_test, y_train, y_test):
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Baseline Data Start ------------------------------------------------------------------")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix baseline")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Baseline Data End ------------------------------------------------------------------")
    # Plot learning curve for baseline model
    cv = StratifiedKFold(n_splits=12)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show()

def performGridSearch( x_train, x_test, y_train, y_test):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = KNeighborsClassifier()
    # Perform grid search
    param_grid = {'n_neighbors':range(11,100), 'metric': ['euclidean', 'manhattan', 'minkowski']}
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

def performKNNTuned(x_train, x_test, y_train, y_test):
    # Aplly decision tree with hyper parameter tuning
    # model = KNeighborsClassifier(n_neighbors=28,metric='manhattan')
    model = KNeighborsClassifier(n_neighbors=72, metric='manhattan')
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

    viz = ValidationCurve(model, param_name='n_neighbors',
                          param_range=[22,24,26,28,30,32,34], cv=10, scoring='accuracy')
    viz.fit(x_train, y_train)
    viz.show()


def performKNN():
    incomeDF = incomeEvaluationUtil.loadExploreDS()
    features = incomeDF.drop(['income'], axis=1)
    output = incomeDF['income']
    testPopulation = 0.2
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=testPopulation)
    performKNNBaseline(x_train, x_test, y_train, y_test)
    performGridSearch(x_train, x_test, y_train, y_test)
    performKNNTuned(x_train, x_test, y_train, y_test)
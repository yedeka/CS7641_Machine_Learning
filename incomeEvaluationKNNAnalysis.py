from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.model_selection import LearningCurve
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

def performKNN():
    incomeDF = incomeEvaluationUtil.loadExploreDS()
    features = incomeDF.drop(['income'], axis=1)
    output = incomeDF['income']
    testPopulation = 0.2
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=testPopulation)
    performKNNBaseline(x_train, x_test, y_train, y_test)
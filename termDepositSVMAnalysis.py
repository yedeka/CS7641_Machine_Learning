'''
This function performs the KNN classification without tuning to determine the baseline results for KNN classification
'''
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from yellowbrick.model_selection import LearningCurve

from termDepositUtil import loadExploreDataSet, clean_data


def performSVMBaseLine(features, output, testPopulation):
    # prepare data for splitting into training set and testing set
    print("Start baseline performance")
    print(np.unique(output))
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=testPopulation)
    model = svm.SVC()
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
        model, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show()

def performSVM():
    bankDS = loadExploreDataSet()
    dataset = bankDS["dataset"]
    cleaned_data = clean_data(dataset)
    features = cleaned_data.drop(['deposit_bool'], axis=1)
    output = cleaned_data['deposit_bool']
    test_population = 0.2
    performSVMBaseLine(features, output, test_population)

from utils import loadFraudDataSet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
import numpy as np

# This function is used to load the fraud dataset and populate it's features in a dictionary
def loadNExploreDS():
    fraudDs = loadFraudDataSet()
    ds_shape = fraudDs.shape
    columns = fraudDs.columns
    uniqueY = fraudDs['Class'].unique()
    valueCnt = fraudDs['Class'].value_counts()
    fraudDict = {
        "dataSet": fraudDs,
        "shape": ds_shape,
        "columns": columns,
        "uniqueY": uniqueY,
        "classValues": valueCnt
    }
    return fraudDict

# This function is used for feature preprocessing and normalization
def perform_preprocessing(fraudDF):
    fraud_df = fraudDF.drop(['Time'], axis=1)
    # normalize the amount column values to match with other columns
    fraud_df['norm_amount'] = StandardScaler().fit_transform(
        fraud_df['Amount'].values.reshape(-1, 1))
    fraud_df = fraud_df.drop(['Amount'], axis=1)
    return fraud_df

#This function is used to split the data set into training set and test set
def splitDataSet(fraudDF):
    x = fraudDF.drop(['Class'], axis=1)
    y = fraudDF[['Class']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return{
        "train_x":x_train,
        "train_y": y_train,
        "test_x": x_test,
        "test_y": y_test,
    }

def perform_baseline_classification(dataSet):
    train_x = dataSet["train_x"]
    train_y = dataSet["train_y"]
    test_x = dataSet["test_x"]
    test_y = dataSet["test_y"]

    dt_classifier = DecisionTreeClassifier()
    # train model by using fit method
    print("Model training starts........")
    dt_classifier.fit(train_x, train_y.values.ravel())
    acc_score = dt_classifier.score(test_x, test_y)
    print(f'Accuracy of model on test dataset :- {acc_score}')
    # predict result using test dataset
    y_pred = dt_classifier.predict(test_x)
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(test_y, y_pred)}")
    # classification report for f1-score
    print(f"Classification Report :- \n {classification_report(test_y, y_pred)}")

def applySampling(originalDF):
    class_val = originalDF['Class'].value_counts()
    print(f"Number of samples for each class :- \n {class_val}")
    non_fraud = class_val[0]
    fraud = class_val[1]
    print(f"Non Fraudulent Numbers :- {non_fraud}")
    print(f"Fraudulent Numbers :- {fraud}")
    # Match both target samples to same level by undersampling
    nonfraud_indices = originalDF[originalDF.Class == 0].index
    fraud_indices = np.array(originalDF[originalDF.Class == 1].index)
    #Take random samples from non fraud indices that are simillar to fraudulant samples
    # Here fraud is the number of random samples to be taken equivalent to the fraud samples
    random_normal_indices = np.random.choice(nonfraud_indices, fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)
    print(f"random non-Fraud indices count :- {random_normal_indices.shape}")
    # concatenate both indices of fraud and non fraud
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    print(f"Under sample indices count :- {under_sample_indices.shape}")
    # extract all features from whole data for under sample indices only
    under_sample_data = originalDF.iloc[under_sample_indices, :]
    print(f"under_sample_data count :- {under_sample_data.shape}")
    x_undersample_data = under_sample_data.drop(['Class'], axis=1)
    y_undersample_data = under_sample_data[['Class']]
    # now split dataset to train and test datasets as before
    X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
        x_undersample_data, y_undersample_data, test_size=0.2, random_state=0)
    dt_classifier = DecisionTreeClassifier(max_depth=11)
    # train model by using fit method
    print("Model training start........")
    dt_classifier.fit(X_train_sample, y_train_sample.values.ravel())
    print("Model training completed")
    acc_score = dt_classifier.score(X_test_sample, y_test_sample)
    print(f'Accuracy of model on test dataset :- {acc_score}')
    # predict result using test dataset
    y_pred = dt_classifier.predict(X_test_sample)
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test_sample, y_pred)}")
    # classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test_sample, y_pred)}")
    print(f"AROC score :- \n {roc_auc_score(y_test_sample, y_pred)}")


def performDecisionTree():
    fraud_dict = loadNExploreDS()
    xformed_fraud_ds = perform_preprocessing(fraud_dict["dataSet"])
    trainTestDS = splitDataSet(xformed_fraud_ds)
    # perform_baseline_classification(trainTestDS)
    applySampling(xformed_fraud_ds)
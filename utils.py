import pandas as pd

def loadFraudDataSet():
    dataset = pd.read_csv("./data/creditcard.csv");
    return dataset

def loadWineDataSet():
    dataset = pd.read_csv("./data/winequality-white.csv",";");
    return dataset;

def loadBankDataSet():
    dataset = pd.read_csv("./data/bank.csv");
    return dataset;
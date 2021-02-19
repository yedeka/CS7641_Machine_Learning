import pandas as pd

def loadWineDataSet():
    dataset = pd.read_csv("./data/winequality-white.csv",";");
    return dataset;

def loadBankDataSet():
    dataset = pd.read_csv("./data/bank.csv");
    return dataset;
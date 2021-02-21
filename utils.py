import pandas as pd

def loadIncomeDataSet():
    dataset = pd.read_csv("income_evaluation.csv");
    return dataset;

def loadBankDataSet():
    dataset = pd.read_csv("bank.csv");
    return dataset;
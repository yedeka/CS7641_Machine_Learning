import termDepositDTAnalysis
import termDepositSVMAnalysis
import termDepositBoostingAnalysis
import termDepositKnnAnalysis
import incomeEvaluationDTAnalysis

import incomeEvaluationUtil

def performIncomeExperiment():
    incomeEvaluationDTAnalysis.performDecisionTree()

def peformTermDepositExperiment():
     #termDepositDTAnalysis.performDecisionTree()
     termDepositBoostingAnalysis.performBoosting()
     #termDepositKnnAnalysis.performKNN()
     #termDepositSVMAnalysis.performSVM()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    peformTermDepositExperiment()
    #performIncomeExperiment()


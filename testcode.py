import termDepositDTAnalysis
import termDepositSVMAnalysis
import termDepositBoostingAnalysis
import termDepositKnnAnalysis
import termDepositNeuralNetworkAnalysis
import incomeEvaluationDTAnalysis
import incomeEvaluationBoostingAnalysis
import incomeEvaluationKNNAnalysis
import incomeValidationSVMAnalysis
import incomeEvaluationNeuralNetworkAnalysis
import time

def performIncomeExperiment():
    #incomeEvaluationDTAnalysis.performDecisionTree()
    #incomeEvaluationBoostingAnalysis.performBoosting()
    knnStart = time.time()
    incomeEvaluationKNNAnalysis.performKNN()
    knnend = time.time()
    print("Time for income evaluation KNN => ", knnend - knnStart)
    # incomeValidationSVMAnalysis.performSVM()
    '''nnStart = time.time()
    incomeEvaluationNeuralNetworkAnalysis.performNN()
    nnend = time.time()
    print("Time for income evaluation Neural Network => ", nnend - nnStart)'''

def peformTermDepositExperiment():
     #termDepositDTAnalysis.performDecisionTree()
     #termDepositBoostingAnalysis.performBoosting()
     nnStart = time.time()
     termDepositKnnAnalysis.performKNN()
     nnend = time.time()
     print("Time for Term Deposit KNN => ", nnend - nnStart)
     #termDepositSVMAnalysis.performSVM()
     '''nnStart = time.time()
     termDepositNeuralNetworkAnalysis.performNN()
     nnend = time.time()
     print("Time for Term Deposit Neural Network => ",nnend-nnStart)'''
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #peformTermDepositExperiment()
    performIncomeExperiment()


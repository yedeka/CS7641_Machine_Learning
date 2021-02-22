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
    dtStart = time.time()
    incomeEvaluationDTAnalysis.performDecisionTree()
    dtend = time.time()
    print("Time for income evaluation KNN => ", dtend - dtStart)
    boostStart = time.time()
    incomeEvaluationBoostingAnalysis.performBoosting()
    dtend = time.time()
    print("Time for income evaluation boosting => ", dtend - boostStart)
    knnStart = time.time()
    incomeEvaluationKNNAnalysis.performKNN()
    knnend = time.time()
    print("Time for income evaluation KNN => ", knnend - knnStart)
    nnStart = time.time()
    incomeValidationSVMAnalysis.performSVM()
    nnend = time.time()
    print("Time for income evaluation SVM => ", nnend - nnStart)
    nnStart = time.time()
    incomeEvaluationNeuralNetworkAnalysis.performNN()
    nnend = time.time()
    print("Time for income evaluation Neural Network => ", nnend - nnStart)

def peformTermDepositExperiment():
     nnStart = time.time()
     termDepositDTAnalysis.performDecisionTree()
     nnend = time.time()
     print("Time for Term Deposit Decision Tree => ", nnend - nnStart)
     termDepositBoostingAnalysis.performBoosting()
     nnStart = time.time()
     termDepositKnnAnalysis.performKNN()
     nnend = time.time()
     print("Time for Term Deposit KNN => ", nnend - nnStart)
     nnStart = time.time()
     termDepositSVMAnalysis.performSVM()
     nnend = time.time()
     print("Time for Term Deposit SVM => ", nnend - nnStart)
     nnStart = time.time()
     termDepositNeuralNetworkAnalysis.performNN()
     nnend = time.time()
     print("Time for Term Deposit Neural Network => ",nnend-nnStart)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    peformTermDepositExperiment()
    performIncomeExperiment()


import termDepositDTAnalysis
import termDepositBoostingAnalysis
import termDepositKnnAnalysis
import termDepositSVMAnalysis
import wineQualityDTAnalysis

def peformTermDepositExperiment():
    termDepositDTAnalysis.performDecisionTree()
    # termDepositBoostingAnalysis.performBoosting()
    # termDepositKnnAnalysis.performKNN()
    # termDepositSVMAnalysis.performSVM()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # performWineExperiment()
    peformTermDepositExperiment()


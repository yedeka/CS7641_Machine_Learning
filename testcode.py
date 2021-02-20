import termDepositDTAnalysis
import termDepositBoostingAnalysis
import termDepositKnnAnalysis
import wineQualityDTAnalysis

def peformTermDepositExperiment():
    # termDepositDTAnalysis.performDecisionTree()
    # termDepositBoostingAnalysis.performBoosting()
    termDepositKnnAnalysis.performKNN()

''' def performWineExperiment():
    wineQualityDTAnalysis.performDecisionTree() '''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # performWineExperiment()
    peformTermDepositExperiment()


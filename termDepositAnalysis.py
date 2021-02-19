from utils import loadBankDataSet

import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

def plotCategorical(bankdf):
    cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20, 15))
    counter = 0
    for cat_column in cat_columns:
        value_counts = bankdf[cat_column].value_counts()

        trace_x = counter // 3
        trace_y = counter % 3
        x_pos = np.arange(0, len(value_counts))

        axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label=value_counts.index)
        axs[trace_x, trace_y].set_title(cat_column)
        for tick in axs[trace_x, trace_y].get_xticklabels():
            tick.set_rotation(90)
        counter += 1
    plt.savefig("Categorical_Data")
    # plt.show()

def plotNumerical(bankdf):
    num_columns = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(30, 15))

    counter = 0
    for num_column in num_columns:
        trace_x = counter // 3
        trace_y = counter % 3

        axs[trace_x, trace_y].hist(bankdf[num_column])
        axs[trace_x, trace_y].set_title(num_column)

        counter += 1
    plt.savefig("Numerical_Data")
    # plt.show()
    bankdf[['pdays', 'campaign', 'previous']].describe()
    print(len(bankdf[bankdf['pdays'] > 400]) / len(bankdf) * 100)
    # 81% of values of pdays are -1 we should drop it since the significance of -1 is not known
    print(len(bankdf[bankdf['pdays'] == -1]) / len(bankdf) * 100)
    # Analysis of campaign column
    print(len(bankdf[bankdf['campaign'] > 34]) / len(bankdf) * 100)
    #Analysis of previous column
    print(len(bankdf[bankdf['previous'] > 34]) / len(bankdf) * 100)
    #Analysis of deposit column
    value_counts = bankdf['deposit'].value_counts()
    print(value_counts)

def loadExploreDataSet():
    bankds = loadBankDataSet()
    print(bankds.describe())
    ds_shape = bankds.shape
    print(ds_shape)
    columns = bankds.columns
    print(columns)
    missingvals = bankds.isnull().sum()
    print(missingvals)

    bankdsinfo = {
        "dataset": bankds,
        "shape": ds_shape,
        "missingVals": missingvals,
    }
    return bankdsinfo

if __name__ == '__main__':
    bankDS = loadExploreDataSet()
    plotCategorical(bankDS["dataset"])
    plotNumerical(bankDS["dataset"])
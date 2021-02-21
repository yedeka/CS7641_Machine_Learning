import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
Function to load the data set and exploring the same with some initial dataset level basic stats
'''

def loadBankDataSet():
    dataset = pd.read_csv("bank.csv");
    return dataset

def loadExploreDataSet():
    bankds = loadBankDataSet()
    ds_shape = bankds.shape
    columns = bankds.columns
    missingvals = bankds.isnull().sum()
    print(bankds['deposit'].value_counts())
    bankdsinfo = {
        "dataset": bankds,
        "shape": ds_shape,
        "missingVals": missingvals,
    }
    return bankdsinfo


"""
Function to plot categorical data to understand the limits and outlisers for data cleansing    
"""


def plotCatClmns(bankdf):
    category_clmns = ['contact', 'default', 'education', 'housing', 'job', 'loan', 'marital', 'month', 'poutcome']
    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(30, 15))
    counter = 0
    for cat_column in category_clmns:
        value_counts = bankdf[cat_column].value_counts()

        x_loc = counter // 3
        y_loc = counter % 3
        x_pos = np.arange(0, len(value_counts))

        axs[x_loc, y_loc].bar(x_pos, value_counts.values, tick_label=value_counts.index)
        axs[x_loc, y_loc].set_title(cat_column)
        for tick in axs[x_loc, y_loc].get_xticklabels():
            tick.set_rotation(90)
        counter += 1
    plt.savefig("Categorical_Data.png")
    # plt.show()

    # Impact of job type on term deposit
    job_df = pd.DataFrame()

    job_df['yes'] = bankdf[bankdf['deposit'] == 'yes']['job'].value_counts()
    job_df['no'] = bankdf[bankdf['deposit'] == 'no']['job'].value_counts()
    job_df['percentage'] = job_df['yes'] / job_df['no']
    job_df['diff'] = job_df['yes'] - job_df['no']
    print(job_df)

    # Impact of marital status on term deposit
    ms_df = pd.DataFrame()
    ms_df['yes'] = bankdf[bankdf['deposit'] == 'yes']['marital'].value_counts()
    ms_df['no'] = bankdf[bankdf['deposit'] == 'no']['marital'].value_counts()
    ms_df['percentage'] = ms_df['yes'] / ms_df['no']
    ms_df['diff'] = ms_df['yes'] - ms_df['no']
    print(ms_df)

    # Impact of education on deposit
    ed_df = pd.DataFrame()
    ed_df['yes'] = bankdf[bankdf['deposit'] == 'yes']['education'].value_counts()
    ed_df['no'] = bankdf[bankdf['deposit'] == 'no']['education'].value_counts()
    ed_df['percentage'] = ed_df['yes'] / ed_df['no']
    ed_df['diff'] = ed_df['yes'] - ed_df['no']
    print(ed_df)

    # Impact of type of contact on deposit
    contact_df = pd.DataFrame()
    contact_df['yes'] = bankdf[bankdf['deposit'] == 'yes']['contact'].value_counts()
    contact_df['no'] = bankdf[bankdf['deposit'] == 'no']['contact'].value_counts()
    contact_df['percentage'] = contact_df['yes'] / contact_df['no']
    contact_df['diff'] = contact_df['yes'] - contact_df['no']
    print(contact_df)

    ''' 
    This tells us that
    Customers with 'blue-collar' and 'services' jobs are less likely to subscribe for term deposit.
    Married customers are less likely to subscribe for term deposit.
    Customers with 'cellular' type of contact are less likely to subscribe for term deposit. '''


"""
Function to plot numerical data to understand the limits and outlisers for data cleansing    
"""


def plotNumClmns(bankdf):
    num_clmns = ['balance', 'campaign', 'day', 'duration', 'pdays', 'previous']

    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(30, 15))

    counter = 0
    for num_column in num_clmns:
        x_loc = counter // 3
        y_loc = counter % 3

        axs[x_loc, y_loc].hist(bankdf[num_column])
        axs[x_loc, y_loc].set_title(num_column)

        counter += 1
    plt.savefig("Numerical_Data.png")
    # plt.show()

    print(bankdf[['pdays', 'campaign', 'previous']].describe())
    print(len(bankdf[bankdf['pdays'] > 400]) / len(bankdf) * 100)
    # 74% of values of pdays are -1 we should drop it since the significance of -1 is not known
    print(len(bankdf[bankdf['pdays'] == -1]) / len(bankdf) * 100)
    # Analysis of campaign column
    print(len(bankdf[bankdf['campaign'] > 34]) / len(bankdf) * 100)
    # Analysis of previous column
    print(len(bankdf[bankdf['previous'] > 34]) / len(bankdf) * 100)
    # Analysis of deposit column
    value_counts = bankdf['deposit'].value_counts()
    print(value_counts)

    bal_df = pd.DataFrame()
    bal_df['yes'] = (bankdf[bankdf['deposit'] == 'yes'][['deposit', 'balance']].describe())['balance']
    bal_df['no'] = (bankdf[bankdf['deposit'] == 'no'][['deposit', 'balance']].describe())['balance']
    print(bal_df)


'''
Function to convert the booleans into 1 and 0.
'''


def get_dummy_from_bool(row, column_name):
    return 1 if row[column_name] == 'yes' else 0


'''
Cleanse the values if they are above threshold then return the mean values.
'''


def get_correct_values(row, column_name, threshold, df):
    ''' Returns mean value if value in column_name is above threshold'''
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].mean()
        return mean


'''
Clean the bank marketing dataframe to do the following 
    1] Convert boolean values to 0/1.
    2] Impute the values of the columns where values are below thresholds.
    3] Convert categorical columns into dummy relevant values.
    4] Drop the irrelevant columns.     
'''


def clean_data(df):
    cleaned_df = df.copy()
    print('Columns before ')
    print(cleaned_df.columns)
    # Convert boolean columns into 0/1 columns
    truthy_clmns = ['default', 'deposit', 'housing', 'loan']
    for truthy_col in truthy_clmns:
        cleaned_df[truthy_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, truthy_col), axis=1)

    cleaned_df = cleaned_df.drop(columns=truthy_clmns)
    # Transform categorical columns into equivalent dummy values
    cat_columns = ['contact', 'education', 'job', 'marital', 'month', 'poutcome']
    for cat_col in cat_columns:
        cleaned_df = pd.concat([cleaned_df.drop(cat_col, axis=1),
                                pd.get_dummies(cleaned_df[cat_col], prefix=cat_col, prefix_sep='_',
                                               drop_first=True, dummy_na=False)], axis=1)

    # drop irrelevant columns
    cleaned_df = cleaned_df.drop(columns=['pdays'])

    # impute noisy columns
    cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df), axis=1)
    cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df), axis=1)

    cleaned_df = cleaned_df.drop(columns=['campaign', 'previous'])

    print('Columns After ')
    print(cleaned_df.columns)
    print(cleaned_df.head())
    return cleaned_df

if __name__ == '__main__':
    ds = loadExploreDataSet()
    plotCatClmns(ds["dataset"])
    plotNumClmns(ds["dataset"])

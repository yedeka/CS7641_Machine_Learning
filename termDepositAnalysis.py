from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.model_selection import LearningCurve

from utils import loadBankDataSet

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
Function to load the data set and exploring the same with some initial dataset level basic stats
'''
def loadExploreDataSet():
    bankds = loadBankDataSet()
    ds_shape = bankds.shape
    columns = bankds.columns
    missingvals = bankds.isnull().sum()

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
    # 81% of values of pdays are -1 we should drop it since the significance of -1 is not known
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

'''
Gather the baseline parameters for decision tree without any tuning and grid searching
'''
def performDecisionTreeBaseline(features, output, test_population):
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=test_population)
    # Aplly decision tree without any hyper parameter tuning
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred1 = model.predict(x_test)
    print("Baseline Data Start ------------------------------------------------------------------")
    print(classification_report(y_test, y_pred1))
    print("Baseline Data End ------------------------------------------------------------------")
    # Plotting the learning curve for the baseline model
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(features, output)  # Fit the data to the visualizer
    visualizer.show()

'''
perform grid search to find out the best hyperparameter to be used for tuning the decision tree
'''
def performGridSearch(features, output, testpopulation):
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=testpopulation)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # create model
    model = DecisionTreeClassifier()
    # Perform grid search
    param_grid = {'max_leaf_nodes': [30,40,50,80,100,130,150,185,187,200,210,220], 'splitter':['random']}
    # param_grid = {'min_samples_leaf': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'splitter': ["random"]}
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=-1)
    # fitting the model for grid search
    grid.fit(x_train, y_train)
    print("Grid search Data Start ------------------------------------------------------------------")
    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(x_test)
    # print classification report
    print(grid.best_score_)
    print("Grid search Data End ------------------------------------------------------------------")

'''
Find out the performance of the tuned model
'''
def performDecisionTreeTuned(features, output, test_population):
    # prepare data for splitting into training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=test_population)
    # Aplly decision tree without any hyper parameter tuning
    model = DecisionTreeClassifier(max_leaf_nodes= 100, splitter='random')
    model.fit(x_train, y_train)
    y_pred1 = model.predict(x_test)
    print("Tuned Data Start ------------------------------------------------------------------")
    print(classification_report(y_test, y_pred1))
    print("Tuned Data End ------------------------------------------------------------------")
    # Plotting the learning curve for the baseline model
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=12)
    sizes = np.linspace(0.3, 1.0, 10)
    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4
    )
    visualizer.fit(features, output)  # Fit the data to the visualizer
    visualizer.show()

def performDecisionTree():
    bankDS = loadExploreDataSet()
    dataset = bankDS["dataset"]
    # plotCatClmns(dataset)
    # plotNumClmns(dataset)
    # Cleanse the data by dropping irrelevant columns, imputing the noisy columns and converting truthy columns to corresponding binary values
    cleaned_data = clean_data(dataset)
    features = cleaned_data.drop(['deposit_bool'], axis=1)
    output = cleaned_data['deposit_bool']
    test_population = 0.2
    #performDecisionTreeBaseline(features, output, test_population)
    #performGridSearch(features, output, test_population)
    performDecisionTreeTuned(features, output, test_population)

import pandas as pd
import statsmodels.api as sm
import numpy as np
import os
from sklearn.metrics import recall_score

"""
We have a database (ULL_database) with information about primary and secondary education students in the Canary Islands 
for 4 academic years. There is information about their academic performance and 
contextual information (about their families, teachers, and school). The database contains a subset of data 
in the form of panel data, meaning information about the same students at different points in time (ULL_panel_data).

Machine learning algorithms can be used to predict at-risk students. 
A student is considered at risk if they are anticipated to have low academic performance in the future. 
Detecting these students would allow for corrective measures to be taken in advance.

As a measure of academic performance, we have the variables "scores".
We have academic performance in Mathematics and in Spanish Language

We specify a model to predict at-risk students. Utilizing the panel data,
the model aims to forecast whether the student will be at risk in the future (in 6th grade)
based on various predictors of current academic performance (3rd grade).

Each observation (row) in ULL_panel_data is a student, with their academic performance in sixth grade 
and their predictors of academic performance from third grade (columns).
"""
directory = "C:/Users..."
os.chdir(directory)

# Load data
data = pd.read_csv('ULL_panel_data.csv', sep=';')

# Select only the data we want to work for
data = data[['id_student_16_19', 'score_MAT', 'score_LEN', 'score_MAT3', 'score_LEN3', 'a1',
             'mother_education', 'father_education', 'mother_occupation', 'father_occupation', 
             'inmigrant_second_gen', 'start_schooling_age', 'books', 'f12a', 'public_private', 
             'capital_island', 'd14', 'ESCS', 'id_school']]

# Drop observations with missing data in any of the variables that we will use in the models
    # Here, synthetic data methods can be used instead to fill in missing values
missing_columns = ['score_MAT3', 'a1', 'mother_education', 'father_education',
    'mother_occupation', 'father_occupation', 'inmigrant_second_gen',
    'start_schooling_age', 'books', 'f12a', 'public_private',
    'capital_island', 'd14']

data = data.dropna(subset=missing_columns)

# Generate quartiles of scores in sixth grade
data['scores_MATq'] = pd.qcut(data['score_MAT'], 4, labels=["1", "2", "3","4"])
data['scores_MATq'] = data['scores_MATq'].astype(int)
data['scores_LENq'] = pd.qcut(data['score_LEN'], 4, labels=["1", "2", "3","4"])
data['scores_LENq'] = data['scores_LENq'].astype(int)

# Generate median and percentiles 25 and 75 of socioeconomic status (ESCS)
median_ESCS = data['ESCS'].median()
p25_ESCS = data['ESCS'].quantile(0.25)
p75_ESCS = data['ESCS'].quantile(0.75)
data['ESCS_median'] = pd.Series([None] * len(data))  # Inicializa con valores nulos
data.loc[data['ESCS'] >= median_ESCS, 'ESCS_median'] = 1
data.loc[data['ESCS'] < median_ESCS, 'ESCS_median'] = 2
data['ESCS_p25_p75'] = pd.Series([None] * len(data))  # Inicializa con valores nulos
data.loc[data['ESCS'] >= p75_ESCS, 'ESCS_p25_p75'] = 1
data.loc[data['ESCS'] < p25_ESCS, 'ESCS_p25_p75'] = 2
data.loc[(data['ESCS'] >= p25_ESCS) & (data['ESCS'] < p75_ESCS), 'ESCS_p25_p75'] = None

# Some data corrections to make the final results
data['d14'] = data['d14'].apply(lambda x: 1 if x == 1 else 0) # Variable d14 top category(4) is the "bad" category (more than 50% of teachers change school), so the results must be inverted

"""
Models
The goal of the model is to predict the academic performance in sixth grade (Y_t)
using information from the same student in third grade, specifically:
1: Academic performance in third grade (Y_t-1)
2: Sensitive factors or circumstances (C)
3: Predictors uncorrelated with circumstances, also called "effort" (X)

Model 1:    Y_t = α + β1Y_t-1 + ε
Model 2:    Y_t = α + β1Y_t-1 + β2C + ε
Model 3:    First step: Y_t-1 = α + β2C + ν
                Recover the prediction of Y_t-1 (academic performance due to circumstances, C): Y_t-1_hat
                Recover the residual ν (academic performance due to effort, X): ν_hat
            Second step: Y_t = α + β1Y_t-1_hat + β2ν_hat + ε
                Recover the prediction of Y_t only due to Y_t-1_hat (only due to circumstances)
                Recover the prediction of Y_t only due to ν_hat (only due to effort)
"""

"""
In theory...
Model 1: Using only the academic performance in third grade (benchmark)
Model 2: Using the academic performance + circumstances in third grade (less fair - more socially desirable)
Model 3: Using the circumstances + effort in third grade (close to Model 2)
    Prediction exclusively of circumstances of Model 3 (much less fair - much more socially desirable)
    Prediction exclusively of effort of Model 3 (much more fair - much less socially desirable)

Let's prove it
"""
# Variables for the models
Y_t_1 = "score_MAT3"
C = data[["a1", "mother_education", "father_education", "mother_occupation", "father_occupation", 
      "inmigrant_second_gen", "start_schooling_age", "books", "f12a", "public_private", "capital_island", "d14"]]
Circumstances = ["a1", "mother_education", "father_education", "mother_occupation", "father_occupation", 
      "inmigrant_second_gen", "start_schooling_age", "books", "f12a", "public_private", "capital_island", "d14"]

# Dummy variables (all variables C are categorical variables)
dummy_variables = pd.get_dummies(C ,columns = Circumstances ,drop_first = True)

# Join Y_t_1 + C
data_combined = pd.concat([data[Y_t_1], dummy_variables], axis=1)

# Model 1
model1 = sm.OLS(data["score_MAT"], sm.add_constant(data[Y_t_1])).fit()
print(model1.summary())
data['model1_pred'] = model1.fittedvalues

# Model 2
model2 = sm.OLS(data["score_MAT"], sm.add_constant(data_combined)).fit()
print(model2.summary())
data['model2_pred'] = model2.fittedvalues

# Model 3
model3 = sm.OLS(data["score_MAT3"], sm.add_constant(dummy_variables)).fit()
print(model3.summary())
    # First step
data['Y_t_1_hat'] = model3.fittedvalues
data['ν_hat'] = model3.resid
    # Second step
model4 = sm.OLS(data["score_MAT"], sm.add_constant(data[["Y_t_1_hat", "ν_hat"]])).fit()
print(model4.summary())
data['model3_pred'] = model4.fittedvalues
        # Prediction exclusively of circumstances
data['model3_pred_circum'] = model4.params['const'] + model4.params['Y_t_1_hat'] * data['Y_t_1_hat']
        # Prediction exclusively of effort
mean_circu = data['Y_t_1_hat'].mean()
data['mean_circu'] = mean_circu
data['model3_pred_effort'] = (model4.params['const'] + 
                          model4.params['ν_hat'] * data['ν_hat'] + 
                          model4.params['Y_t_1_hat'] * mean_circu)

# Transform predictions(continuous) to quartiles(categorical)
data['scores_MAT_pred1'] = pd.qcut(data['model1_pred'], 4, labels=["1", "2", "3","4"])
data['scores_MAT_pred1'] = data['scores_MAT_pred1'].astype(int)
data['scores_MAT_pred2'] = pd.qcut(data['model2_pred'], 4, labels=["1", "2", "3","4"])
data['scores_MAT_pred2'] = data['scores_MAT_pred2'].astype(int)
data['scores_MAT_pred3'] = pd.qcut(data['model3_pred'], 4, labels=["1", "2", "3","4"])
data['scores_MAT_pred3'] = data['scores_MAT_pred3'].astype(int)
data['scores_MAT_pred_C'] = pd.qcut(data['model3_pred_circum'], 4, labels=["1", "2", "3","4"])
data['scores_MAT_pred_C'] = data['scores_MAT_pred_C'].astype(int)
data['scores_MAT_pred_X'] = pd.qcut(data['model3_pred_effort'], 4, labels=["1", "2", "3","4"])
data['scores_MAT_pred_X'] = data['scores_MAT_pred_X'].astype(int)

# Transform predictions(continuous) to terciles(categorical)
"""
data['scores_MAT_pred1_t'] = pd.qcut(data['model1_pred'], 3, labels=["1", "2", "3"])
data['scores_MAT_pred1_t'] = data['scores_MAT_pred1_t'].astype(int)
data['scores_MAT_pred2_t'] = pd.qcut(data['model2_pred'], 3, labels=["1", "2", "3"])
data['scores_MAT_pred2_t'] = data['scores_MAT_pred2_t'].astype(int)
data['scores_MAT_pred3_t'] = pd.qcut(data['model3_pred'], 3, labels=["1", "2", "3"])
data['scores_MAT_pred3_t'] = data['scores_MAT_pred3_t'].astype(int)
data['scores_MAT_pred_C_t'] = pd.qcut(data['model3_pred_circum'], 3, labels=["1", "2", "3"])
data['scores_MAT_pred_C_t'] = data['scores_MAT_pred_C_t'].astype(int)
data['scores_MAT_pred_X_t'] = pd.qcut(data['model3_pred_effort'], 3, labels=["1", "2", "3"])
data['scores_MAT_pred_X_t'] = data['scores_MAT_pred_X_t'].astype(int)
"""
# Transform predictions(continuous) to percentiles but percentiles 2 and 3 equal (between 25 and 75 percentil)
data['scores_MAT_pred1_t'] = data['scores_MAT_pred1'].apply(lambda x: 1 if x == 1 else (2 if x == 2 or x == 3 else 3))
data['scores_MAT_pred2_t'] = data['scores_MAT_pred2'].apply(lambda x: 1 if x == 1 else (2 if x == 2 or x == 3 else 3))
data['scores_MAT_pred3_t'] = data['scores_MAT_pred3'].apply(lambda x: 1 if x == 1 else (2 if x == 2 or x == 3 else 3))
data['scores_MAT_pred_C_t'] = data['scores_MAT_pred_C'].apply(lambda x: 1 if x == 1 else (2 if x == 2 or x == 3 else 3))
data['scores_MAT_pred_X_t'] = data['scores_MAT_pred_X'].apply(lambda x: 1 if x == 1 else (2 if x == 2 or x == 3 else 3))

"""
We focus on Equalized Odds (Equality of opportunity) 
    (Other fairness measures are possible)
To calculate Equalized Odds we first calculate recall or sensitivity:
TP / (TP + FN)
and then we calculate the ratio of recall among different groups to obtain Equalized Odds
Recall is calculated for Low and High academic performance
Low academic performance: Below the median or 25th percentile
High academic performance: Above the median or above 75th percentile (top 25 percent)
"""
# Recall below 25 percentile and above 75 percentile (top level vs rest of levels))
def compute_recall(data, categories, levels, top_level, filename_suffix):
    recall_dfs = []
    for category in categories:
        recall_results = {}
        for i in [1, 4]:
            condition_top = (data['scores_MATq'] == i) & (data[category] == top_level)
            condition_low = (data['scores_MATq'] == i) & (data[category] != top_level)
            scores_MATq_top = data.loc[condition_top, 'scores_MATq']
            scores_MATq_low = data.loc[condition_low, 'scores_MATq']

            for variable in ["pred1", "pred2", "pred3", "pred_C", "pred_X"]:
                scores_MAT_variable_top = data.loc[condition_top, f'scores_MAT_{variable}']
                scores_MAT_variable_low = data.loc[condition_low, f'scores_MAT_{variable}']

                recall_top = recall_score(scores_MATq_top, scores_MAT_variable_top, average='micro')
                recall_low = recall_score(scores_MATq_low, scores_MAT_variable_low, average='micro')

                recall_results[f'recall_top_{variable}_{i}'] = recall_top
                recall_results[f'recall_low_{variable}_{i}'] = recall_low

        recall_df = pd.DataFrame(list(recall_results.items()), columns=['Metric', 'Recall'])
        recall_df[['Group', 'Model', 'Percentile']] = recall_df['Metric'].str.extract(r'recall_(top|low)_(\w+)_(\d+)')
        recall_df['Recall'] = recall_df['Recall'].round(4)
        recall_df['Variable'] = category
        recall_df['Prediction'] = '25_75'
        recall_dfs.append(recall_df[['Group', 'Model', 'Percentile', 'Recall', 'Variable', 'Prediction']])
    return recall_dfs

# Recall between 25 percentile and 75 percentile (top level vs rest of levels))
def compute_recall_terciles(data, categories, levels, top_level, filename_suffix):
    recall_dfs = []
    for category in categories:
        recall_results = {}
        i = 2
        condition_top = (data['scores_MATq'] == i) & (data[category] == top_level)
        condition_low = (data['scores_MATq'] == i) & (data[category] != top_level)
        scores_MATq_top = data.loc[condition_top, 'scores_MATq']
        scores_MATq_low = data.loc[condition_low, 'scores_MATq']

        for variable in ["pred1_t", "pred2_t", "pred3_t", "pred_C_t", "pred_X_t"]:
            scores_MAT_variable_top = data.loc[condition_top, f'scores_MAT_{variable}']
            scores_MAT_variable_low = data.loc[condition_low, f'scores_MAT_{variable}']

            recall_top = recall_score(scores_MATq_top, scores_MAT_variable_top, average='micro')
            recall_low = recall_score(scores_MATq_low, scores_MAT_variable_low, average='micro')

            recall_results[f'recall_top_{variable}_{i}'] = recall_top
            recall_results[f'recall_low_{variable}_{i}'] = recall_low

        recall_df = pd.DataFrame(list(recall_results.items()), columns=['Metric', 'Recall'])
        recall_df[['Group', 'Model', 'Tercile']] = recall_df['Metric'].str.extract(r'recall_(top|low)_(\w+)_(\d+)')
        recall_df['Recall'] = recall_df['Recall'].round(4)
        recall_df['Variable'] = category
        recall_df['Prediction'] = 'between25_75'
        recall_dfs.append(recall_df[['Group', 'Model', 'Tercile', 'Recall', 'Variable', 'Prediction']])
    return recall_dfs

# Recall below and above the median (top level vs rest of levels))
def compute_recall_median(data, categories, levels, top_level, filename_suffix):
    recall_dfs = []
    score_pairs = [(1, 2), (3, 4)]
    for category in categories:
        recall_results = {}
        for pair in score_pairs:
            condition_top = ((data['scores_MATq'] == pair[0]) | (data['scores_MATq'] == pair[1])) & (data[category] == top_level)
            condition_low = ((data['scores_MATq'] == pair[0]) | (data['scores_MATq'] == pair[1])) & (data[category] != top_level)
            scores_MATq_top = data.loc[condition_top, 'scores_MATq']
            scores_MATq_low = data.loc[condition_low, 'scores_MATq']
            scores_MATq_top_binary = scores_MATq_top.apply(lambda x: 1 if x in pair else 0)
            scores_MATq_low_binary = scores_MATq_low.apply(lambda x: 1 if x in pair else 0)

            for variable in ["pred1", "pred2", "pred3", "pred_C", "pred_X"]:
                scores_MAT_variable_top = data.loc[condition_top, f'scores_MAT_{variable}']
                scores_MAT_variable_low = data.loc[condition_low, f'scores_MAT_{variable}']
                scores_MAT_variable_top_binary = scores_MAT_variable_top.apply(lambda x: 1 if x in pair else 0)
                scores_MAT_variable_low_binary = scores_MAT_variable_low.apply(lambda x: 1 if x in pair else 0)
                
                recall_top = recall_score(scores_MATq_top_binary, scores_MAT_variable_top_binary, average='binary')
                recall_low = recall_score(scores_MATq_low_binary, scores_MAT_variable_low_binary, average='binary')
                
                recall_results[f'recall_top_{variable}_{pair[0]}_{pair[1]}'] = recall_top
                recall_results[f'recall_low_{variable}_{pair[0]}_{pair[1]}'] = recall_low

        recall_df = pd.DataFrame(list(recall_results.items()), columns=['Metric', 'Recall'])
        recall_df[['Group', 'Model', 'Pair1', 'Pair2']] = recall_df['Metric'].str.extract(r'recall_(top|low)_(\w+)_(\d+)_(\d+)')
        recall_df['Recall'] = recall_df['Recall'].round(4)
        recall_df['Variable'] = category
        recall_df['Prediction'] = 'median'
        recall_dfs.append(recall_df[['Group', 'Model', 'Pair1', 'Pair2', 'Recall', 'Variable', 'Prediction']])
    return recall_dfs

# Collect results in DataFrames
recall_dfs_25_75 = []
recall_dfs_25_75.extend(compute_recall(data, ["f12a"], levels=5, top_level=5, filename_suffix="25_75"))
recall_dfs_25_75.extend(compute_recall(data, ["mother_education", "father_education", "mother_occupation", "father_occupation", "books"], levels=4, top_level=4, filename_suffix="25_75"))
recall_dfs_25_75.extend(compute_recall(data, ["start_schooling_age"], levels=3, top_level=1, filename_suffix="25_75"))
recall_dfs_25_75.extend(compute_recall(data, ["inmigrant_second_gen", "public_private", "capital_island", "a1", "ESCS_median", "ESCS_p25_p75", "d14"], levels=2, top_level=1, filename_suffix="25_75"))

recall_dfs_between25_75 = []
recall_dfs_between25_75.extend(compute_recall_terciles(data, ["f12a"], levels=5, top_level=5, filename_suffix="between25_75"))
recall_dfs_between25_75.extend(compute_recall_terciles(data, ["mother_education", "father_education", "mother_occupation", "father_occupation", "books"], levels=4, top_level=4, filename_suffix="between25_75"))
recall_dfs_between25_75.extend(compute_recall_terciles(data, ["start_schooling_age"], levels=3, top_level=1, filename_suffix="between25_75"))
recall_dfs_between25_75.extend(compute_recall_terciles(data, ["inmigrant_second_gen", "public_private", "capital_island", "a1", "ESCS_median", "ESCS_p25_p75", "d14"], levels=2, top_level=1, filename_suffix="between25_75"))

recall_dfs_median = []
recall_dfs_median.extend(compute_recall_median(data, ["f12a"], levels=5, top_level=5, filename_suffix="median"))
recall_dfs_median.extend(compute_recall_median(data, ["mother_education", "father_education", "mother_occupation", "father_occupation", "books"], levels=4, top_level=4, filename_suffix="median"))
recall_dfs_median.extend(compute_recall_median(data, ["start_schooling_age"], levels=3, top_level=1, filename_suffix="median"))
recall_dfs_median.extend(compute_recall_median(data, ["inmigrant_second_gen", "public_private", "capital_island", "a1", "ESCS_median", "ESCS_p25_p75", "d14"], levels=2, top_level=1, filename_suffix="median"))

# Combine DataFrames
combined_df_25_75 = pd.concat(recall_dfs_25_75, ignore_index=True)
combined_df_between25_75 = pd.concat(recall_dfs_between25_75, ignore_index=True)
combined_df_median = pd.concat(recall_dfs_median, ignore_index=True)

# Pivot tables
pivot_combined_df_25_75 = combined_df_25_75.pivot_table(index=['Variable', 'Group', 'Percentile'], columns='Model', values='Recall').reset_index()
pivot_combined_df_25_75 = pivot_combined_df_25_75[['Variable', 'Group', 'Percentile', 'pred1', 'pred2', 'pred3', 'pred_C', 'pred_X']]
pivot_combined_df_25_75_sorted = pivot_combined_df_25_75.sort_values(by=['Percentile', 'Variable', 'Group'], ascending=[True, True, False])
pivot_combined_df_between25_75 = combined_df_between25_75.pivot_table(index=['Variable', 'Group', 'Tercile'], columns='Model', values='Recall').reset_index()
pivot_combined_df_between25_75 = pivot_combined_df_between25_75[['Variable', 'Group', 'Tercile', 'pred1_t', 'pred2_t', 'pred3_t', 'pred_C_t', 'pred_X_t']]
pivot_combined_df_between25_75_sorted = pivot_combined_df_between25_75.sort_values(by=['Tercile', 'Variable', 'Group'], ascending=[True, True, False])
pivot_combined_df_median = combined_df_median.pivot_table(index=['Variable', 'Group', 'Pair1', 'Pair2'], columns='Model', values='Recall').reset_index()
pivot_combined_df_median = pivot_combined_df_median[['Variable', 'Group', 'Pair1', 'Pair2', 'pred1', 'pred2', 'pred3', 'pred_C', 'pred_X']]
pivot_combined_df_median_sorted = pivot_combined_df_median.sort_values(by=['Pair1', 'Pair2', 'Variable', 'Group'], ascending=[True, True, True, False])

# Calculate equalized odds for each variable
def calculate_odds(row_value, top_value):
    return row_value / top_value if top_value != 0 else None

final_data_25_75 = []

for variable in pivot_combined_df_25_75_sorted['Variable'].unique():
    variable_df = pivot_combined_df_25_75_sorted[pivot_combined_df_25_75_sorted['Variable'] == variable]
    for percentile in variable_df['Percentile'].unique():
        top_row = variable_df[(variable_df['Group'] == 'top') & (variable_df['Percentile'] == percentile)]
        if not top_row.empty:
            top_row = top_row.iloc[0]
            temp_data = []
            for _, row in variable_df[variable_df['Percentile'] == percentile].iterrows():
                odds_row = {
                    'Variable': row['Variable'],
                    'Group': row['Group'],
                    'Percentile': row['Percentile'],
                    'pred1': row['pred1'],
                    'pred2': row['pred2'],
                    'pred3': row['pred3'],
                    'pred_C': row['pred_C'],
                    'pred_X': row['pred_X'],
                    'pred1_odds': calculate_odds(row['pred1'], top_row['pred1']),
                    'pred2_odds': calculate_odds(row['pred2'], top_row['pred2']),
                    'pred3_odds': calculate_odds(row['pred3'], top_row['pred3']),
                    'pred_C_odds': calculate_odds(row['pred_C'], top_row['pred_C']),
                    'pred_X_odds': calculate_odds(row['pred_X'], top_row['pred_X']),
                }
                temp_data.append(odds_row)
            final_data_25_75.extend(temp_data)

final_data_25_75_sorted = pd.DataFrame(final_data_25_75)

final_data_between25_75 = []

for variable in pivot_combined_df_between25_75_sorted['Variable'].unique():
    variable_df = pivot_combined_df_between25_75_sorted[pivot_combined_df_between25_75_sorted['Variable'] == variable]
    for tercile in variable_df['Tercile'].unique():
        top_row = variable_df[(variable_df['Group'] == 'top') & (variable_df['Tercile'] == tercile)]
        if not top_row.empty:
            top_row = top_row.iloc[0]
            temp_data = []
            for _, row in variable_df[variable_df['Tercile'] == tercile].iterrows():
                odds_row = {
                    'Variable': row['Variable'],
                    'Group': row['Group'],
                    'Tercile': row['Tercile'],
                    'pred1_t': row['pred1_t'],
                    'pred2_t': row['pred2_t'],
                    'pred3_t': row['pred3_t'],
                    'pred_C_t': row['pred_C_t'],
                    'pred_X_t': row['pred_X_t'],
                    'pred1_odds': calculate_odds(row['pred1_t'], top_row['pred1_t']),
                    'pred2_odds': calculate_odds(row['pred2_t'], top_row['pred2_t']),
                    'pred3_odds': calculate_odds(row['pred3_t'], top_row['pred3_t']),
                    'pred_C_odds': calculate_odds(row['pred_C_t'], top_row['pred_C_t']),
                    'pred_X_odds': calculate_odds(row['pred_X_t'], top_row['pred_X_t']),
                }
                temp_data.append(odds_row)
            final_data_between25_75.extend(temp_data)

final_data_between25_75_sorted = pd.DataFrame(final_data_between25_75)


final_data_median = []

for variable in pivot_combined_df_median_sorted['Variable'].unique():
    variable_df = pivot_combined_df_median_sorted[pivot_combined_df_median_sorted['Variable'] == variable]
    for pair in variable_df[['Pair1', 'Pair2']].drop_duplicates().values:
        pair1, pair2 = pair
        top_row = variable_df[(variable_df['Group'] == 'top') & (variable_df['Pair1'] == pair1) & (variable_df['Pair2'] == pair2)]
        if not top_row.empty:
            top_row = top_row.iloc[0]
            temp_data = []
            for _, row in variable_df[(variable_df['Pair1'] == pair1) & (variable_df['Pair2'] == pair2)].iterrows():
                odds_row = {
                    'Variable': row['Variable'],
                    'Group': row['Group'],
                    'Pair1': row['Pair1'],
                    'Pair2': row['Pair2'],
                    'pred1': row['pred1'],
                    'pred2': row['pred2'],
                    'pred3': row['pred3'],
                    'pred_C': row['pred_C'],
                    'pred_X': row['pred_X'],
                    'pred1_odds': calculate_odds(row['pred1'], top_row['pred1']),
                    'pred2_odds': calculate_odds(row['pred2'], top_row['pred2']),
                    'pred3_odds': calculate_odds(row['pred3'], top_row['pred3']),
                    'pred_C_odds': calculate_odds(row['pred_C'], top_row['pred_C']),
                    'pred_X_odds': calculate_odds(row['pred_X'], top_row['pred_X']),
                }
                temp_data.append(odds_row)
            final_data_median.extend(temp_data)

final_data_median_sorted = pd.DataFrame(final_data_median)


category_order = ['a1', 'mother_education', 'father_education', 'mother_occupation', 'father_occupation', 'books', 'd14', 'inmigrant_second_gen', 
                  'public_private', 'capital_island', 'start_schooling_age', 'f12a', 'ESCS_median', 'ESCS_p25_p75']

final_data_25_75_sorted['Variable'] = pd.Categorical(final_data_25_75_sorted['Variable'], categories=category_order, ordered=True)
final_data_25_75_sorted = final_data_25_75_sorted.sort_values(by='Variable')
final_data_25_75_sorted = final_data_25_75_sorted[['Variable', 'Group', 'Percentile', 'pred1', 'pred1_odds', 'pred2', 'pred2_odds', 'pred3', 'pred3_odds', 'pred_C', 'pred_C_odds', 'pred_X', 'pred_X_odds']]
final_data_25_75_sorted = final_data_25_75_sorted.sort_values(by=['Percentile', 'Variable', 'Group'], ascending=[True, True, False])
final_data_between25_75_sorted['Variable'] = pd.Categorical(final_data_between25_75_sorted['Variable'], categories=category_order, ordered=True)
final_data_between25_75_sorted = final_data_between25_75_sorted.sort_values(by='Variable')
final_data_between25_75_sorted = final_data_between25_75_sorted[['Variable', 'Group', 'Tercile', 'pred1_t', 'pred1_odds', 'pred2_t', 'pred2_odds', 'pred3_t', 'pred3_odds', 'pred_C_t', 'pred_C_odds', 'pred_X_t', 'pred_X_odds']]
final_data_between25_75_sorted = final_data_between25_75_sorted.sort_values(by=['Tercile', 'Variable', 'Group'], ascending=[True, True, False])
final_data_median_sorted['Variable'] = pd.Categorical(final_data_median_sorted['Variable'], categories=category_order, ordered=True)
final_data_median_sorted = final_data_median_sorted.sort_values(by='Variable')
final_data_median_sorted = final_data_median_sorted[['Variable', 'Group', 'Pair1', 'Pair2', 'pred1', 'pred1_odds', 'pred2', 'pred2_odds', 'pred3', 'pred3_odds', 'pred_C', 'pred_C_odds', 'pred_X', 'pred_X_odds']]
final_data_median_sorted = final_data_median_sorted.sort_values(by=['Pair1', 'Pair2', 'Variable', 'Group'], ascending=[True, True, True, False])

# Export to Excel
with pd.ExcelWriter('Results.xlsx') as writer:
    final_data_25_75_sorted.to_excel(writer, sheet_name='25_75', index=False, float_format='%.4f')
    final_data_median_sorted.to_excel(writer, sheet_name='Median', index=False, float_format='%.4f')
    final_data_between25_75_sorted.to_excel(writer, sheet_name='between25_75', index=False, float_format='%.4f')
    data.to_excel(writer, sheet_name='data', index=False, float_format='%.4f')
    

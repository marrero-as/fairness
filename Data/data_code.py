import pandas as pd
import os

directory = "C:/Users/..."
os.chdir(directory)

# Load data
original_data = pd.read_csv('original.csv', sep=',')

# Keep only the longitudinal information (panel data)
"""
In the database, there is information about the same students in 3th grade of primary education in 2015/16 academic course 
and 3 years later, in 6th grade of primary education in 2018/19 academic course. 
These students are identified with the column id_student_16_19
"""
original_data = original_data.sort_values(by=['id_student_16_19', 'id_grade'])  # Sort by id_student_16_19 and id_grade
original_data = original_data.dropna(subset=['id_student_16_19']) # Drop rows where id_student_16_19 is NaN

# There are students with repeated ids for the same course. These students cannot be in 6th grade in 2015/16. We remove them.
original_data['repeated'] = original_data.groupby(['id_grade', 'id_student_16_19']).transform('size') # Create 'repeated' column
original_data = original_data[~((original_data['repeated'] != 1) & (original_data['id_grade'] == 6) & (original_data['id_year'] == 2016))] # Drop rows where repeated != 1 and id_grade = 6 and id_year = 2016
original_data = original_data.drop(columns=['repeated']) # Drop 'repeated' column

# There are students with observations only in th grade or only in 6th grade. We remove them
original_data['unique'] = original_data.groupby('id_student_16_19')['id_grade'].transform('size') # Create 'unique' column
original_data = original_data[~((original_data['unique'] == 1) & (original_data['id_grade'] == 3))] # Drop rows where unique == 1 and id_grade == 3
original_data = original_data[~((original_data['unique'] == 1) & (original_data['id_grade'] == 6))] # Drop rows where unique == 1 and id_grade == 6
original_data = original_data.drop(columns=['unique']) # Drop 'unique' column

# There are students with scores in one grade but not in another, we remove them
# Filter for score_MAT
original_data['missing'] = original_data['score_MAT'].notna().astype(int)
original_data['missing_min'] = original_data.groupby('id_student_16_19')['missing'].transform('min')
original_data = original_data[original_data['missing_min'] != 0]
original_data = original_data.drop(columns=['missing', 'missing_min'])
# Filter for score_LEN
original_data['missing'] = original_data['score_LEN'].notna().astype(int)
original_data['missing_min'] = original_data.groupby('id_student_16_19')['missing'].transform('min')
original_data = original_data[original_data['missing_min'] != 0]
original_data = original_data.drop(columns=['missing', 'missing_min'])

# Create data for model. 6th grade academic performance explained by variables from the past (3th grade)
panel_3th = original_data[original_data['id_grade'] != 6] # Drop rows where id_grade == 6
panel_3th = panel_3th.rename(columns={ # Rename columns
    'score_MAT': 'score_MAT3',
    'level_MAT': 'level_MAT3',
    'score_LEN': 'score_LEN3',
    'level_LEN': 'level_LEN3'
})
original_data = original_data[['id_student_16_19', 'id_grade', 'score_MAT', 'level_MAT', 'score_LEN', 'level_LEN']] # Keep relevant columns
original_data = original_data[original_data['id_grade'] != 3] # Drop rows where id_grade == 3
ULL_panel_data = original_data.merge(panel_3th, on='id_student_16_19', how='left') # Merge with panel_3th data

ULL_panel_data.to_csv('ULL_panel_data.csv', index=False) # Generate the data to run the application

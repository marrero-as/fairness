
### This repository contains the source code to replicate the results of the following paper:
Marrero, A.S., Marrero, G., Bethencourt, C., Liam, J. Calegari, R. "AI-fairness and equality of opportunity: a case study on educational achievement". AEQUITAS 2024 - Workshop on Fairness and Bias in AI.

### How to reproduce the results in the paper
This is the project structure:

+ data/ Input data files.
+ replication_results/ Generated the results of the paper

All code is written in Python. You will need the following packages: pandas, statsmodels, numpy, scikit-learn

The files need to be run in order: 
1. Download original.csv file from __[zenodo](https://zenodo.org/records/11171863)__
3. Run data_code.py in the data folder. This code generates ULL_panel_data.csv
4. Run code.py in the replication_results folder. This code uses ULL_panel_data.csv and generates the results of the paper in an excel file: Results.xlsx

### Explanation
+ original.csv contains a dataset on academic performance of primary and secondary education students from the Canary Islands.
Each row refers to a single student at a given grade (3th and 6th grades of primary education and 4th grade of secondary education) and academic year (2015-16, 2016-17, 2017-18, 2018-19). Longitudinal data (panel data) is also included: students in 3th grade of primary education in 2015-16 are sampled again in 6th grade of primary education in 2018-19.
+ The columns of the dataset represent relevant features collected for each student: identifiers, academic performance in various subjects, and responses to questionnaires addressed to the students, their families, their teachers, and the school principals. In the data folder, there is an Excel file (data_dictionary.xlsx) where you can find the definitions of each variable and the possible values they can take.
+ data_code.py in the data folder takes the complete database (original.csv) and retains only the longitudinal data. Since our goal is to predict students' future academic performance using current information, the code merges the students' academic performance in 6th grade with the information from 3rd grade, generating ULL_panel_data.csv. Each row in ULL_panel_data.csv represents a student who was surveyed in both 3rd grade and 6th grade. The columns include their academic performance in mathematics in 6th grade and information about a set of variables (called circumstances) in 3rd grade.
+ code.py in the replication_results folder takes ULL_panel_data.csv and generates the results of the paper in Results.xlsx. The first sheet (25_75) includes the Tables 6 and 8 of the paper (percentiles 25 and 75, indicated by 1 and 4). The third sheet (between25_75) includes Table 7 (prediction between 25 and 75 percentiles). The second sheet (Median) includes prediction below and above the median (not included in the paper and indicated by pair 1 and 2, below the median, and pair 3 and 4, above the median). Lastly, the fourth sheet (data) includes the prediction of all regression models estimated in the paper. 

### Citation
When using this archive, please cite this paper: 
Marrero, A.S., Marrero, G., Bethencourt, C., Liam, J. Calegari, R. "AI-fairness and equality of opportunity: a case study on educational achievement". AEQUITAS 2024 - Workshop on Fairness and Bias in AI.

If you use the original dataset, please cite this paper: 
Giovanelli, J., Magnini, M., James, L., Ciatto, G., Marrero, A. S., Borghesi, A., Marrero, G. A., & Calegari, R. (2024). Unfair Inequality in Education: A Benchmark for AI-Fairness Research (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11171863

### License
+ Software is distributed under the terms of the __[MIT License](https://opensource.org/licenses/MIT)__.


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

### License
+ Databases and their contents are distributed under the terms of the __[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)__.
+ Software is distributed under the terms of the __[MIT License](https://opensource.org/licenses/MIT)__.

This directory contains python script to train and evaluate classifiers using cleaned and preprocessed manually annonated jod ads data (df_manual) and classify cleaned and preprocessed scraped job ads data (df_jobs).

* Cleaned and preprocessed mannualy annotated job ads data (df_manual) is stored in the data directory. You can find the data in:
  - [data](../data) &rarr; [final dfs](../data/final%20dfs/) &rarr; [df_manual_for_training.pkl](../data/final%20dfs/df_manual_for_training.pkl)
* Cleaned and preprocessed job ads data (df_jobs) is stored in the scraping directory. You can find the data in:
  - [data](../data) &rarr; [final dfs](../data/final%20dfs/) &rarr; [df_jobs_for_classification.pkl](../data/final%20dfs/df_jobs_for_classification.pkl)
* Classification and evaluation output can be found in the following directories:
    - Classifiers: [data](../data) &rarr; [classification models](../data/classification%20models)
    - Visuals: [data](../data) &rarr; [plots](../data/plots)
    - Tables: [data](../data) &rarr; [output tables](../data/output%20tables)

***Note that classifiers are imported from [estimators_get_pipe.py](./estimators_get_pipe.py) in this directory.***
This directory contains python necessary imports, functions, forked repositories, and styles for use in all other python scripts in this repository. Note that the imports.py script also contains variables and common functions that are used in other scripts.

* All data are stored in the data directory under the final dfs directory. You can find the data in:
  - [data](../data) &rarr; [final dfs](../data/final%20dfs/)
  - Cleaned and preprocessed manually annotated job ads data (df_manual) is stored in the data directory. You can find the data in:
  - [data](../data) &rarr; [final dfs](../data/final%20dfs/) &rarr; [df_manual_for_analysis.pkl](../data/final%20dfs/df_manual_for_analysis.pkl)
  - Unclassified cleaned and preprocessed job ads data (df_jobs) is stored in the scraping directory. You can find the data in:
    - [data](../data) &rarr; [final dfs](../data/final%20dfs/) &rarr; [df_jobs_for_classification.pkl](../data/final%20dfs/df_jobs_for_classification.pkl)
  - Classified (not cleaned) job ads data (df_jobs) is stored in the scraping directory. You can find the data in:
    - [data](../data) &rarr; [final dfs](../data/final%20dfs/) &rarr; [df_jobs_classified.pkl](../data/final%20dfs/df_jobs_classified.pkl)
  - Classified cleaned and preprocessed job ads data (df_jobs) is stored in the scraping directory. You can find the data in:
    - [data](../data) &rarr; [final dfs](../data/final%20dfs/) &rarr; [df_jobs_for_analysis.pkl](../data/final%20dfs/df_jobs_for_analysis.pkl)
* Analysis output can be found in the following directories:
  - Classifiers: [data](../data) &rarr; [classification models](../data/classification%20models)
  - Visuals: [data](../data) &rarr; [plots](../data/plots)
  - Tables: [data](../data) &rarr; [output tables](../data/output%20tables)

***Note that imports and some functions are imported from [setup_module](../setup_module) directory.***

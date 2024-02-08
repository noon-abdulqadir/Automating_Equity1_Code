This directory contains python script to clean and preprocess scraped data for both manually annotated job ads (transformed into df_manual) and raw job ads (transformed into df_jobs).

* Mannualy annotated data is stored this directory under [Coding Material](./Coding%20Material/). You can also find the data in:
  - [data](../data) &rarr; [content analysis + ids + sectors](../data/content%20analysis%20+%20ids%20+%20sectors) &rarr; [Coding Material](../data/content%20analysis%20+%20ids%20+%20sectors/Coding%20Material)
* Raw data is stored in the scraping directory. You can find the data in:
  - [1. Scraping](../1.%20Scraping) &rarr; *\<name of platform>* &rarr; Data
* All data are stored in the data directory under the final dfs directory. You can find the data in:
  - [data](../data) &rarr; [final dfs](../data/final%20dfs/)
  - [df_manual_for_training.pkl](../data/final%20dfs/df_manual_for_training.pkl)
  - [df_jobs_for_classification.pkl](../data/final%20dfs/df_jobs_for_classification.pkl)
  - [df_jobs_classified.pkl](../data/final%20dfs/df_jobs_classified.pkl)
  - [df_jobs_for_analysis.pkl](../data/final%20dfs/df_jobs_for_analysis.pkl)

***Note that imports and some functions are imported from [setup_module](../setup_module) directory.***

***Codebook:*** To view the codebook, [click here](../Sector%20Keywords,%20Codebook,%20and%20Classification%20Metrics/Codebook.md) or navigate to the markdown file titled ```Codebook.md``` under the directory titled ```Sector Keywords, Codebook, and Classification Metrics```.

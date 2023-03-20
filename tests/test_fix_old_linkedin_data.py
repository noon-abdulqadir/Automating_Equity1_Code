# with open('/Users/nyxinsane/Documents/Google Drive (nyXiNsane)/Credentials/Apps/Pip, Conda, and Brew/conda_packages.txt', 'r') as f:
#     con = [line.rstrip(' \n') for line in f]
# with open('/Users/nyxinsane/Documents/Google Drive (nyXiNsane)/Credentials/Apps/Pip, Conda, and Brew/conda_packages.txt', 'w') as f:
#     for i in set([c.split('=')[0] for c in con]):
#         f.write(f'{i.lower()}\n')
# -*- coding: utf-8 -*-
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Install packages and import
# %%
# #################################### PLEASE INSTALL LATEST CHROME WEBDRIVER #####################################
# Uncomment to run as required
# #     --install-option="--chromedriver-version= *.**" \
#   --install-option="--chromedriver-checksums=4fecc99b066cb1a346035bf022607104,058cd8b7b4b9688507701b5e648fd821"
# %%
# ##### COPY THE LINES IN THIS COMMENT TO THE TOP OF NEW SCRIPTS #####
# # Function to import this package to other files
# import os
# import sys
# from pathlib import Path
# code_dir = None
# code_dir_name = 'Code'
# unwanted_subdir_name = 'Analysis'
# for _ in range(5):
#     parent_path = str(Path.cwd().parents[_]).split('/')[-1]
#     if (code_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):
#         code_dir = str(Path.cwd().parents[_])
#         if code_dir is not None:
#             break
# main_dir = str(Path(code_dir).parents[0])
# scraped_data = f'{code_dir}/scraped_data'
# sys.path.append(code_dir)
# from setup_module.imports import *
# from setup_module.params import *
# from setup_module.scraping import *
# from setup_module.classification import *
# from setup_module.vectorizers_classifiers import *
# warnings.filterwarnings("ignore", category=DeprecationWarning)
#
#
# %%
import os
import sys
from pathlib import Path

code_dir = None
code_dir_name = 'Code'
unwanted_subdir_name = 'Analysis'

for _ in range(5):

    parent_path = str(Path.cwd().parents[_]).split('/')[-1]

    if (code_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):

        code_dir = str(Path.cwd().parents[_])

        if code_dir is not None:
            break

main_dir = str(Path(code_dir).parents[0])
scraped_data = f'{code_dir}/scraped_data'
sys.path.append(code_dir)

# %%
from setup_module.imports import *
from setup_module.scraping import *

# %%

fpath = '/Users/nyxinsane/Library/CloudStorage/OneDrive-UvA/Automating Equity/Study 1/Study1_Code/scraped_data/LinkedIn/Data/'

keywords = []
for file in os.listdir(fpath):
    if '_old' in file:
        if 'dict_' in file and file.endswith('.json'):
            keyword = file.split('dict_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
            keywords.append(keyword)
        if 'df_' in file and (file.endswith('.csv') or file.endswith('.xlsx')):
            keyword = file.split('df_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
            keywords.append(keyword)

for keyword in list(set(keywords)):
    data_dict = []
    for file in os.listdir(fpath):
        if keyword in file and (file.endswith('.json') or file.endswith('.csv') or file.endswith('.xlsx')):

            if file.endswith('.json'):
                if '_old' in file:
                    # print(file)
                    old_name = file
                    with open(f'{fpath}{file}', 'r', encoding='utf-8') as f:
                        data_old_dict = json.load(f)
                    # print(keyword, ' data_old_dict: ', len(data_old_dict))
                elif '_old' not in file:
                    # print(file)
                    final_name = file
                    with open(f'{fpath}{file}', 'r', encoding='utf-8') as f:
                        try:
                            data_new_dict = json.load(f)

                    # print(keyword, ' data_new_dict: ', len(data_new_dict))
                        except json.JSONDecodeError:
                            data_new_dict = list(csv.DictReader(f))
                try:
                    if data_new_dict and data_old_dict:
                        data_new_dict.extend(data_old_dict)
                        for my_dict in data_new_dict:
                            if my_dict not in data_dict:
                                data_dict.append(my_dict)
                    elif data_new_dict and not data_old_dict:
                        data_dict.extend(data_new_dict)
                    elif data_old_dict and not data_new_dict:
                        data_dict.extend(data_old_dict)
                    elif not data_new_dict and not data_old_dict:
                        pass
                except(NameError):
                    with contextlib.suppress(NameError):
                        if data_new_dict:
                            data_dict.extend(data_new_dict)
                    with contextlib.suppress(NameError):
                        if data_old_dict:
                            data_dict.extend(data_old_dict)

                with contextlib.suppress(NameError):
                    with open(f'{fpath}{final_name}', 'w', encoding='utf-8') as f:
                        json.dump(data_dict, f)
                with contextlib.suppress(NameError, FileNotFoundError):
                    os.remove(f'{fpath}{old_name}')

            elif file.endswith('.csv') or file.endswith('.xlsx'):
                if '_old' in file:
                    # print(file)
                    old_name = file
                    data_old_df = pd.read_csv(f'{fpath}{file}', encoding='utf-8')
                    # print(keyword, ' data_old_df: ', len(data_old_df))
                elif '_old' not in file:
                    # print(file)
                    final_name = file
                    data_new_df = pd.read_csv(f'{fpath}{file}', encoding='utf-8')
                    # print(keyword, ' data_new_df: ', len(data_new_df))
                try:
                    if len(data_new_df) > 0 and len(data_old_df) > 0:
                        data_new_df = data_new_df.append(data_old_df)
                        # data_new_df = data_new_df.drop_duplicates()
                    elif len(data_new_df) > 0 and len(data_old_df) == 0:
                        pass
                    elif len(data_new_df) == 0 and len(data_old_df) > 0:
                        data_new_df = data_old_df
                    elif len(data_new_df) == 0 and len(data_old_df) == 0:
                        pass
                except(NameError):
                    with contextlib.suppress(NameError):
                        if len(data_new_df) > 0:
                            pass
                    with contextlib.suppress(NameError):
                        if len(data_old_df) > 0:
                            data_new_df = data_old_df
                with contextlib.suppress(NameError):
                    data_new_df.to_csv(f'{fpath}{final_name}', index=False)
                with contextlib.suppress(NameError, FileNotFoundError):
                    os.remove(f'{fpath}{old_name}')

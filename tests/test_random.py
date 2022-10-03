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
# sys.path.append(code_dir)
# from setup_module.imports import *
# from setup_module.params import *
# from setup_module.scraping import *
# from setup_module.classification import *
# from setup_module.vectorizers_classifiers import *
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# %matplotlib notebook
# %matplotlib inline
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
sys.path.append(code_dir)

# %%
from setup_module.imports import *
from setup_module.scraping import *

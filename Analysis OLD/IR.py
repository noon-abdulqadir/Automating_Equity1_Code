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

from setup_module.classification import *
from setup_module.imports import *
from setup_module.params import *
from setup_module.scraping import *
from setup_module.vectorizers_classifiers import *

warnings.filterwarnings('ignore')

# %%
args = get_args()
save_enabled = args['save_enabled']
parent_dir = args['data_save_path']
front_columns = ['Coder ID', 'Job ID', 'OG_Sentence ID', 'Sentence ID', 'Sentence']
cal_columns = [
    'Warmth',
    'Competence',
    'Task_Mentioned',
    'Task_Warmth',
    'Task_Competence',
]
k_alpha_dict = {}
ir_all_dict = {}
coders_numbers = [1, 2]
coder_score_dict = defaultdict(list)
coder = 'all'
reliability_dir = f'{args["parent_dir"]}Reliability Checks/'

############# INTRACODER Coder 1 #############
if coder == 1:
    ir_file_name = 'INTRACODER1'
    df1a = pd.read_excel(
        f'{reliability_dir}Pair 1 - Intra/Job ID - p_ce05575325f3b0f1_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1b = pd.read_excel(
        f'{reliability_dir}Pair 2 - Intra/Job ID - p_ca008a8d67189539_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1c = pd.read_excel(
        f'{reliability_dir}Pair 3 - Intra/Job ID - p_9acfa03a05f2542f_Rhea- Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1d = pd.read_excel(
        f'{reliability_dir}Pair 4 - Intra/Job ID - p_3d626cbfef055cb4_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1e = pd.read_excel(
        f'{reliability_dir}Pair 5 - Intra/Job ID - p_1b37ad5237066811_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )

    df1 = pd.concat([df1a, df1b, df1c, df1d, df1e])
    print('Length of df1:', len(df1))

    df2a = pd.read_excel(
        f'{reliability_dir}Pair 1 - Intra/OLD Job ID - p_ce05575325f3b0f1_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2b = pd.read_excel(
        f'{reliability_dir}Pair 2 - Intra/OLD Job ID - p_ca008a8d67189539_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2c = pd.read_excel(
        f'{reliability_dir}Pair 3 - Intra/OLD Job ID - p_9acfa03a05f2542f_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2d = pd.read_excel(
        f'{reliability_dir}Pair 4 - Intra/OLD Job ID - p_3d626cbfef055cb4_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2e = pd.read_excel(
        f'{reliability_dir}Pair 5 - Intra/OLD Job ID - p_1b37ad5237066811_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )

    df2 = pd.concat([df2a, df2b, df2c, df2d, df2e])
    print('Length of df2:', len(df2))

############# INTRACODER Coder 2 #############
elif coder == 2:
    ir_file_name = 'INTRACODER2'
    df1a = pd.read_excel(
        f'{reliability_dir}Pair 8 Intra/OLD Job ID - p_a087b464a6a092fa_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1b = pd.read_excel(
        f'{reliability_dir}Pair 11 Intra/Job ID - 4052472440_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1c = pd.read_excel(
        f'{reliability_dir}Pair 12 Intra/Job ID - p_7674c23f38f94dcf_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1d = pd.read_excel(
        f'{reliability_dir}Pair 13 Intra/Job ID - p_42ea0a6f52e862d4_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1e = pd.read_excel(
        f'{reliability_dir}Pair 14 Intra/Job ID - p_9f364da9030d1ce6_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )

    df1 = pd.concat([df1a, df1b, df1c, df1d, df1e])
    print('Length of df1:', len(df1))

    df2a = pd.read_excel(
        f'{reliability_dir}Pair 8 Intra/PAIRED Job ID - p_a087b464a6a092fa_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2b = pd.read_excel(
        f'{reliability_dir}Pair 11 Intra/PAIRED Job ID - 4052472440_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2c = pd.read_excel(
        f'{reliability_dir}Pair 12 Intra/PAIRED Job ID - p_7674c23f38f94dcf_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2d = pd.read_excel(
        f'{reliability_dir}Pair 13 Intra/PAIRED Job ID - p_42ea0a6f52e862d4_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2e = pd.read_excel(
        f'{reliability_dir}Pair 14 Intra/PAIRED Job ID - p_9f364da9030d1ce6_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )

    df2 = pd.concat([df2a, df2b, df2c, df2d, df2e])
    print('Length of df2:', len(df2))

############# INTERCODER #############
elif coder == 'all':
    ir_file_name = 'INTERCODER'

    df1a = pd.read_excel(
        f'{reliability_dir}Pair 6 - Inter/PAIRED INTER - Job ID - p_15a42cd4b082799e_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1b = pd.read_excel(
        f'{reliability_dir}Pair 7 - Inter/Job ID - 3768944208_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1c = pd.read_excel(
        f'{reliability_dir}Pair 9 Inter/Job ID - 4039450758_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1d = pd.read_excel(
        f'{reliability_dir}Pair 10 Inter/Job ID - p_5c738f7cef046a48_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df1e = pd.read_excel(
        f'{reliability_dir}Pair 16 Inter/Job ID - p_9acfa03a05f2542f_Rhea- Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )

    df1 = pd.concat([df1a, df1b, df1c, df1d, df1e])

    df2a = pd.read_excel(
        f'{reliability_dir}Pair 6 - Inter/OLD Job ID - p_15a42cd4b082799e_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2b = pd.read_excel(
        f'{reliability_dir}Pair 7 - Inter/Job ID - 3768944208_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2c = pd.read_excel(
        f'{reliability_dir}Pair 9 Inter/PAIRED Job ID - 4039450758_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2d = pd.read_excel(
        f'{reliability_dir}Pair 10 Inter/PAIRED Job ID - p_5c738f7cef046a48_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )
    df2e = pd.read_excel(
        f'{reliability_dir}Pair 16 Inter/PAIRED Job ID - p_9acfa03a05f2542f_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )

    df2 = pd.concat([df2a, df2b, df2c, df2d, df2e])

    df1 = pd.read_excel(
        f'{reliability_dir}Pair 6 - Inter/PAIRED INTER - Job ID - p_15a42cd4b082799e_Rhea - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )

    df2 = pd.read_excel(
        f'{reliability_dir}Pair 6 - Inter/OLD Job ID - p_15a42cd4b082799e_Coder_Name - Codebook (Automating Equity).xlsx',
        index_col=0,
        engine='openpyxl',
    )

    print('Length of df1:', len(df1))
    print('Length of df2:', len(df2))

print('-' * 20)
print('\n')
print(f'Results for {ir_file_name}')
print('-' * 20)
print('\n')
df1['Coder ID'] = 1
df1['OG_Sentence ID'] = df1.index + 1

df2['Coder ID'] = 2
df2['OG_Sentence ID'] = df2.index + 1

df_concat_coder_all = pd.concat([df1, df2])
df_concat_coder_all['Sentence ID'] = (
    df_concat_coder_all.groupby(['Sentence']).ngroup() + 1
)

for column in df_concat_coder_all[cal_columns]:
    k_alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
        df_concat_coder_all,
        experiment_col='Sentence ID',
        annotator_col='Coder ID',
        class_col=str(column),
    )
    print('-' * 20)
    k_alpha_dict['K-alpha ' + str(column)] = k_alpha
    print(f"Krippendorff's alpha ({str(column)}): ", k_alpha)

    print('-' * 20)
print('\n')
for column in df_concat_coder_all[cal_columns]:
    for index, coder_number in enumerate(coders_numbers):
        coder = (
            df_concat_coder_all.loc[
                df_concat_coder_all['Coder ID'] == coder_number, str(column)
            ]
            .astype(int)
            .to_list()
        )
        coder_score_dict[str(column)].append(
            [[int(index + 1), i, coder[i]] for i in range(len(coder))]
        )

for column in cal_columns:
    counter = 0
    formatted_codes = coder_score_dict[str(column)][counter]
    while counter < len(coders_numbers) - 1:
        try:
            formatted_codes += coder_score_dict[str(column)][counter + 1]
            counter += 1
        except Exception:
            break

    ratingtask = agreement.AnnotationTask(data=formatted_codes)

    ir_all_dict['IR K-alpha ' + str(column)] = ratingtask.alpha()
    # ir_all_dict["IR Cohen-kappa " + str(column)] = ratingtask.kappa()
    ir_all_dict['IR Scott-pi ' + str(column)] = ratingtask.pi()

    # print("-" * 20, "\n")
    print(f"Krippendorff's alpha ({str(column)}):", ratingtask.alpha())
    # print(f"Cohen's Kappa ({str(column)}): ", ratingtask.kappa())
    print(f"Scott's pi ({str(column)}): ", ratingtask.pi())
    print('-' * 20, '\n')

    if save_enabled is True:
        with open(parent_dir + f'{column}_FINAL_IR_all_{ir_file_name}.json', 'w', encoding='utf8') as f:
            json.dump(ir_all_dict, f)
print('-' * 20)


# %%

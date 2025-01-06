# -*- coding: utf-8 -*-
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import os # type:ignore # isort:skip # fmt:skip # noqa # nopep8
import sys # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from pathlib import Path # type:ignore # isort:skip # fmt:skip # noqa # nopep8

mod = sys.modules[__name__]

code_dir = None
code_dir_name = 'Code'
unwanted_subdir_name = 'Analysis'

if code_dir_name not in str(Path.cwd()).split('/')[-1]:
    for _ in range(5):

        parent_path = str(Path.cwd().parents[_]).split('/')[-1]

        if (code_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):

            code_dir = str(Path.cwd().parents[_])

            if code_dir is not None:
                break
else:
    code_dir = str(Path.cwd())
sys.path.append(code_dir)

# %load_ext autoreload
# %autoreload 2


# %%
from setup_module.imports import *  # type:ignore # isort:skip # fmt:skip # noqa # nopep8

warnings.filterwarnings('ignore', category=DeprecationWarning)

# %% [markdown]
# ### Analyses

# %%
# Function to plot
def qq_plot(x):
    (osm, osr), (slope, intercept, r) = scipy.stats.probplot(x, dist='norm', plot=None)
    plt.plot(osm, osr, '.', osm, slope * osm + intercept)
    plt.xlabel('Quantiles', fontsize=14)
    plt.ylabel('Quantiles Obs', fontsize=14)
    plt.show()


# %%
# Function to create dummy variables
def encodeY(Y):

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y


# Or use LabelBinarizer()

# %%
# Function to find cosine similarity score: 1−cos(x, y) = (x * y)/(||x||*||y||)
def cosine_similarity(x, y):
    x_sqrt = np.sqrt(np.dot(x, x))
    y_sqrt = np.sqrt(np.dot(y, y))
    if y_sqrt != 0:
        return np.dot(x, y.T) / (x_sqrt * y_sqrt)
    elif y_sqrt == 0:
        return 0


# %% [markdown]
# ### intercoder_reliability_to_csv

# %%
# Calculate k-alpha
def IR_kalpha(
    df_concat_coder_all,
    save_enabled=False,
    cal_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    k_alpha_dict=None,
    args=get_args(),
):
    print(
        f'NOTE: The function "IR_kalpha" contains the following optional (default) arguments:\n{get_default_args(IR_kalpha)}'
    )

    if k_alpha_dict is None:
        k_alpha_dict = {}

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

    if save_enabled is True:
        with open(f'{args["parent_dir"]}K-alpha.json', 'w', encoding='utf8') as f:
            json.dump(k_alpha_dict, f)
    elif save_enabled is False:
        print('No K-alpha save enabled.')

    return k_alpha_dict


# %%
# Calculate all IR
def IR_all(
    df_concat_coder_all,
    coders_numbers,
    save_enabled=False,
    cal_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    coder_score_dict=defaultdict(list),
    ir_all_dict=None,
    args=get_args(),
):
    print(
        f'NOTE: The function "IR_all" contains the following optional (default) arguments:\n{get_default_args(IR_all)}'
    )

    if ir_all_dict is None:
        ir_all_dict = {}

    for column in df_concat_coder_all[cal_columns]:
        for index, coder_number in enumerate(coders_numbers):
            coder = (
                df_concat_coder_all.loc[
                    df_concat_coder_all['Coder ID'].astype(str) == str(coder_number), str(column)
                ]
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
        ir_all_dict['IR Cohen-kappa ' + str(column)] = ratingtask.kappa()
        ir_all_dict['IR Scott-pi ' + str(column)] = ratingtask.pi()

        print('-' * 20, '\n')
        print(f"Krippendorff's alpha ({str(column)}):", ratingtask.alpha())
        print(f"Cohen's Kappa ({str(column)}): ", ratingtask.kappa())
        print(f"Scott's pi ({str(column)}): ", ratingtask.pi())
    print('-' * 20, '\n')

    if save_enabled is True:
        with open(f'{args["parent_dir"]}IR_all.json', 'w', encoding='utf8') as f:
            json.dump(ir_all_dict, f)
    elif save_enabled is False:
        print('No IR save enabled.')

    return (coder_score_dict, ratingtask, ir_all_dict)


# %%
def IR_all_final(
    coder,
    k_alpha_dict=None,
    ir_all_dict=None,
    coders_numbers=[1, 2],
    coder_score_dict=defaultdict(list),
    save_enabled=False,
    front_columns=['Coder ID', 'Job ID', 'OG_Sentence ID', 'Sentence ID', 'Sentence'],
    cal_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    args=get_args(),
):
    if k_alpha_dict is None:
        k_alpha_dict = {}
    if ir_all_dict is None:
        ir_all_dict = {}

    reliability_dir=f'{args["content_analysis_dir"]}Reliability Checks/'
    # print('-' * 20)
    # print('\n')
    # print(f'Results for {ir_file_name}')
    # print('-' * 20)
    # print('\n')
    ############# INTRACODER Coder 1 #############
    if coder == 1:
        ir_file_name = 'INTRACODER1'
        df1a = pd.read_excel(
            f'{reliability_dir}Pair 1 - Intra/Job ID - p_ce05575325f3b0f1_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df1b = pd.read_excel(
            f'{reliability_dir}Pair 2 - Intra/Job ID - p_ca008a8d67189539_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df1c = pd.read_excel(
            f'{reliability_dir}Pair 3 - Intra/Job ID - p_9acfa03a05f2542f_Rhea- Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df1d = pd.read_excel(
            f'{reliability_dir}Pair 4 - Intra/Job ID - p_3d626cbfef055cb4_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df1e = pd.read_excel(
            f'{reliability_dir}Pair 5 - Intra/Job ID - p_1b37ad5237066811_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )

        df1 = pd.concat([df1a, df1b, df1c, df1d, df1e])

        df2a = pd.read_excel(
            f'{reliability_dir}Pair 1 - Intra/OLD Job ID - p_ce05575325f3b0f1_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df2b = pd.read_excel(
            f'{reliability_dir}Pair 2 - Intra/OLD Job ID - p_ca008a8d67189539_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df2c = pd.read_excel(
            f'{reliability_dir}Pair 3 - Intra/OLD Job ID - p_9acfa03a05f2542f_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df2d = pd.read_excel(
            f'{reliability_dir}Pair 4 - Intra/OLD Job ID - p_3d626cbfef055cb4_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df2e = pd.read_excel(
            f'{reliability_dir}Pair 5 - Intra/OLD Job ID - p_1b37ad5237066811_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )

        df2 = pd.concat([df2a, df2b, df2c, df2d, df2e])

    ############# INTRACODER Coder 2 #############
    elif coder == 2:
        ir_file_name = 'INTRACODER2'
        df1a = pd.read_excel(
            f'{reliability_dir}Pair 8 - Intra/OLD Job ID - p_a087b464a6a092fa_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df1b = pd.read_excel(
            f'{reliability_dir}Pair 11 - Intra/Job ID - 4052472440_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df1c = pd.read_excel(
            f'{reliability_dir}Pair 12 - Intra/Job ID - p_7674c23f38f94dcf_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df1d = pd.read_excel(
            f'{reliability_dir}Pair 13 - Intra/Job ID - p_42ea0a6f52e862d4_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df1e = pd.read_excel(
            f'{reliability_dir}Pair 14 - Intra/Job ID - p_9f364da9030d1ce6_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
        )

        df1 = pd.concat([df1a, df1b, df1c, df1d, df1e])

        df2a = pd.read_excel(
            f'{reliability_dir}Pair 8 Intra/PAIRED Job ID - p_a087b464a6a092fa_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df2b = pd.read_excel(
            f'{reliability_dir}Pair 11 Intra/PAIRED Job ID - 4052472440_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df2c = pd.read_excel(
            f'{reliability_dir}Pair 12 Intra/PAIRED Job ID - p_7674c23f38f94dcf_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df2d = pd.read_excel(
            f'{reliability_dir}Pair 13 Intra/PAIRED Job ID - p_42ea0a6f52e862d4_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
        )
        df2e = pd.read_excel(
            f'{reliability_dir}Pair 14 Intra/PAIRED Job ID - p_9f364da9030d1ce6_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
        )

        df2 = pd.concat([df2a, df2b, df2c, df2d, df2e])

    ############# INTERCODER #############
    elif coder == 'all':
        ir_file_name = 'INTERCODER'

        df1 = pd.read_excel(
            f'{reliability_dir}Pair 6 - Inter/PAIRED INTER - Job ID - p_15a42cd4b082799e_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
        )

        df2 = pd.read_excel(
            f'{reliability_dir}Pair 6 - Inter/OLD Job ID - p_15a42cd4b082799e_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
        )

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
        ir_all_dict['IR Cohen-kappa ' + str(column)] = ratingtask.kappa()
        ir_all_dict['IR Scott-pi ' + str(column)] = ratingtask.pi()

        print('-' * 20, '\n')
        print(f"Krippendorff's alpha ({str(column)}):", ratingtask.alpha())
        print(f"Cohen's Kappa ({str(column)}): ", ratingtask.kappa())
        print(f"Scott's pi ({str(column)}): ", ratingtask.pi())
        print('-' * 20, '\n')

        if save_enabled is True:
            with open(
                f'{args["parent_dir"]}{column}_FINAL_IR_all_{ir_file_name}.json', 'w', encoding='utf8') as f:
                json.dump(ir_all_dict, f)
    print('-' * 20)

    return ir_all_dict

# %%
ir_all_dict = IR_all_final(coder = 'all')

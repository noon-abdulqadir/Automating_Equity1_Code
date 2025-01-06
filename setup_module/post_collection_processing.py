# -*- coding: utf-8 -*-
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
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
from setup_module.imports import *
from setup_module.scraping import *

warnings.filterwarnings('ignore', category=DeprecationWarning)

# %% [markdown]
# post_collection_processing
# %%
# Function to create a metadata dict
def create_metadata(args=get_args()):

    if args['print_enabled'] is True:
        print('Creating Metadata Dict.')

    columns = args['columns_fill_list'] + args['columns_drop_list']

    metadata_dict = {
        columns[0]: {
            'type': 'str',
            'description': 'unique identifier for each job ad',
            'example': 'pj_245d9fed724ade0b',
        },  # Job ID
        'Language': {
            'type': 'str',
            'description': 'language of job ad',
            'example': 'en',
        },  # Language
        'Job Description': {
            'type': 'str',
            'description': 'job posting complete text',
        },  # Job Description
        # Sentence
        columns[1]: {'type': 'str', 'description': 'sentence from job ad'},
        columns[2]: {
            'type': 'str',
            'description': 'search term used for data collection',
            'list': args['keywords_list'],
        },  # Search Keyword
        columns[3]: {
            'type': 'str',
            'description': 'gender categorization of job based on Standard Industrial Classifications (SBI2008; 2018)',
            'example': 'Female',
        },  # Gender
        columns[4]: {
            'type': 'str',
            'description': 'age categorization of job based on Standard Industrial Classifications (SBI2008; 2018)',
            'example': 'Older',
        },  # Age
        columns[5]: {
            'type': 'str',
            'description': 'name of job search online platforms from which data was collected',
            'list': args['site_list'],
        },  # Platform
        columns[6]: {
            'type': 'str',
            'description': 'title of advertised job position',
            'example': 'Store Assistant',
        },  # Job Title
        columns[7]: {
            'type': 'str',
            'description': 'name of company that posted job ad',
            'example': 'Picnic',
        },  # Company Name
        columns[8]: {
            'type': 'str',
            'description': 'city of advertised job position',
            'example': 'Amstelveen',
        },  # Location
        columns[9]: {
            'type': 'float',
            'description': 'ONLY AVAILABLE FOR GLASSDOOR PLATFORM JOB ADS - rating of company out of 5 stars',
            'example': 4.7,
        },  # Rating
        columns[10]: {
            'type': ['str', 'list'],
            'description': 'industry of advertised job position',
            'example': ['Research', 'Chemicals', 'Food Production'],
        },  # Industry
        columns[11]: {
            'type': ['str', 'list'],
            'description': 'sector of advertised job position',
            'example': ['Engineering', 'Information Technology'],
        },  # Sector
        columns[12]: {
            'type': 'str',
            'description': 'IN DUTCH - ONLY AVAILABLE FOR GLASSDOOR PLATFORM JOB ADS - type of company ownership',
            'example': 'Beursgenoteerd bedrijf',
        },  # Type of ownership
        columns[13]: {
            'type': 'str',
            'description': 'IN DUTCH - ONLY AVAILABLE FOR GLASSDOOR PLATFORM JOB ADS - type of employment for advertised job position',
        },  # Employment Type
        columns[14]: {
            'type': 'str',
            'description': 'IN DUTCH - ONLY AVAILABLE FOR GLASSDOOR PLATFORM JOB ADS - level of seniority for advertised job position',
        },  # Seniority Level
        columns[15]: {
            'type': 'str',
            'description': 'URL of company advertising job position',
        },  # Company URL
        columns[16]: {
            'type': 'str',
            'description': 'URL of advertised job position',
        },  # Job URL
        columns[17]: {
            'type': 'str',
            'description': 'IN DUTCH - time passed between job ad being posted and job ad being collected',
            'example': '2u',
        },  # Job Age
        columns[18]: {
            'type': 'str',
            'description': 'IN DUTCH - time passed between job ad being posted and job ad being collected',
            'example': '2u',
        },  # Job Age Number
        columns[19]: {
            'type': 'str',
            'description': 'date on which job ad was posted',
            'example': '2020-12-30',
        },  # Job Date
        columns[20]: {
            'type': 'str',
            'description': 'date on which job ad was collected',
            'example': '2020-12-30',
        },
    }

    if args['print_enabled'] is True:
        print(f'Saving Metadata dict to {code_dir}/metadata.json')
    with open(f'{code_dir}/metadata.json', 'w', encoding='utf8') as f:
        json.dump(metadata_dict, f)

    return metadata_dict


# %%
# # Function to merge metadata
# def save_metadata(df_jobs, df_file_name, save_path, args=get_args()):

#     if (not df_jobs.empty) and (len(df_jobs != 0)):
#         if args['print_enabled'] ==True:
#             print('Attaching Metadata to existing DF.')
#         metadata_dict = create_metadata()
#         metadata_key = 'metadat.iot'
#         metadata_dict_json = json.dumps(metadata_dict)
#         df_jobs_pyarrow = pa.Table.from_pandas(df_jobs)
#         existing_metadata = df_jobs_pyarrow.schema.metadata
#         combined_metadata = {
#             metadata_key.encode(): metadata_dict_json.encode(),
#             **existing_metadata,
#         }
#         df_jobs_pyarrow = df_jobs_pyarrow.replace_schema_metadata(combined_metadata)
#         if args['print_enabled'] ==True:
#             print('Saving DF as .parquet file.')
#         pq.write_table(
#             df_jobs_pyarrow,
#             save_path + df_file_name.replace('.csv', '_pyarrow.parquet'),
#             compression='GZIP',
#         )
#     elif (df_jobs.empty) and (len(df_jobs == 0)):
#         df_jobs_pyarrow = pa.Table.from_pandas(df_jobs)

#     return df_jobs_pyarrow


# %%
# Fix broken LinkedIn Files
def fix_broken_linkedin_files(glob_path):
    data_dict = {}
    data_list = []

    if glob_path.endswith('.json'):

        with open(glob_path, encoding = 'utf-8') as csv_file_handler:
            csv_reader = csv.DictReader(csv_file_handler)

            for rows in csv_reader:
                first_key = str(list(rows.keys())[0])
                key = rows[first_key]
                data_dict[key] = rows

        for num in data_dict:
            data_list.append(data_dict[num])

        with open(glob_path, 'w', encoding = 'utf-8') as json_file_handler:
            json_file_handler.write(json.dumps(data_list, indent = 4))


# %%
# # Clean df and drop duplicates and -1 for job description
# def clean_df(
#     df_jobs: pd.DataFrame,
#     id_dict_new = False,
#     int_variable: str = 'Job ID',
#     str_variable: str = 'Job Description',
#     gender: str = 'Gender',
#     age: str = 'Age',
#     language: str = 'en',
#     nan_list = [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan'],
#     reset=True,
#     args=get_args(),
# ) -> pd.DataFrame:

#     df_jobs.columns = df_jobs.columns.to_series().progress_apply(lambda x: x.strip())
#     df_jobs.dropna(axis='index', how='all', inplace=True)
#     df_jobs.dropna(axis='columns', how='all', inplace=True)
#     df_jobs.drop(
#         df_jobs.columns[
#             df_jobs.columns.str.contains(
#                 'unnamed|index|level', regex=True, case=False, flags=re.I
#             )
#         ],
#         axis='columns',
#         inplace=True,
#         errors='ignore',
#     )
#     df_jobs[int_variable] = df_jobs[int_variable].progress_apply(lambda x: str(x).lower().strip())

#     if reset is True:
#         df_jobs = set_gender_age_sects_lang(df_jobs, str_variable=str_variable, id_dict_new=id_dict_new)

#     subset_list=[int_variable, str_variable, gender, age]
#     print('Cleaning DF')
#     df_jobs.drop_duplicates(
#         subset=[str_variable],
#         keep='first',
#         inplace=True,
#         ignore_index=True,
#     )

# #     df_jobs = df_jobs.loc[
# #         (
# #             df_jobs[str_variable]
# #             .swifter.progress_bar(args['print_enabled'])
# #             .progress_bar(args['print_enabled'])
# #             progress_apply(lambda x: isinstance(x, str))
# #         )
# #         & (df_jobs[str_variable] != -1)
# #         & (df_jobs[str_variable] != '-1')
# #         & (df_jobs[str_variable] != None)
# #         & (df_jobs[str_variable] != 'None')
# #         & (df_jobs[str_variable] != np.nan)
# #         & (df_jobs[str_variable] != 'nan')
# #     ]

#     df_jobs.drop(
#         df_jobs[
#             (df_jobs[str_variable].isin(nan_list)) |
#             (df_jobs[str_variable].isnull()) |
#             (df_jobs[str_variable].isna())
#         ].index,
#         axis='index',
#         inplace=True,
#         errors='ignore'

#     )


#     print('Detecting Language.')
#     df_jobs = detect_language(df_jobs, str_variable)
#     if 'Language' in df_jobs.columns:
#         try:
#             df_jobs.drop(df_jobs.index[df_jobs['Language'] != str(language)], axis='index', inplace=True, errors='ignore')

#         except:
#             df_jobs = df_jobs.loc[(df_jobs['Language'] == str(language))]

#     if 'Search Keyword' in df_jobs.columns:
#         for w_keyword, r_keyword in keyword_trans_dict.items():
#             df_jobs.loc[(df_jobs['Search Keyword'] == str(w_keyword)), 'Search Keyword'] = r_keyword

#     df_jobs.reset_index(inplace=True, drop=True)

#     return df_jobs


# # %%
# # Lang detect
# def detect_language(df_jobs: pd.DataFrame, str_variable = 'Job Description', args=get_args()) -> pd.DataFrame:
#     if args['print_enabled'] is True:
#         print('Starting language detection...')

#     # df_jobs['Language'] = language
#     try:
#         df_jobs['Language'] = df_jobs[str_variable].swifter.progress_bar(args['print_enabled']).progress_apply(detect_language_helper)

#     except Exception as e:
#         if args['print_enabled'] is True:
#             print('Language not detected.')
#     else:
#         if args['print_enabled'] is True:
#             print('Language detection complete.')

#     return df_jobs


# # %%
# def detect_language_helper(x, language='en'):

#     x = ''.join([i for i in x if i not in list(string.punctuation)])

#     if not x or x in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan'] or x.isspace() or x.replace(' ', '').isdigit():
#         return 'NO LANGUAGE DETECTED'
#     # try:
#     #     lang = WhatTheLang().predict_lang(x)
#     #     return lang if lang not in ['CANT_PREDICT', 'nl', language] else detect(x)
#     # except ValueError:
#     try:
#         return detect(x)
#     except langdetect.LangDetectException:
#         return 'NO LANGUAGE DETECTED'


# %%
# Function to order categories
def categorize_df_gender_age(
    df,
):
    # Arrange Categories
    try:
        df['Gender'] = df['Gender'].astype('category').cat.reorder_categories(order_gender, ordered=True)

        df['Gender'] = pd.Categorical(
            df['Gender'], categories=order_gender, ordered=True
        )
    except ValueError as e:
        print(e)
    try:
        df['Age'] = df['Age'].astype('category').cat.reorder_categories(order_age, ordered=True)

        df['Age'] = pd.Categorical(
            df['Age'], categories=order_age, ordered=True
        )
    except ValueError as e:
        print(e)

    return df



# %%
# Function to dummy code gender and age
def dummy_code_df_gender_age(df, print_info=False, args=get_args()):
    # Gender Recode
    df.loc[df['Gender'] == 'Female', ['Gender_Female']] = 1
    df.loc[df['Gender'] != 'Female', ['Gender_Female']] = 0

    df.loc[df['Gender'] == 'Mixed Gender', ['Gender_Mixed']] = 1
    df.loc[df['Gender'] != 'Mixed Gender', ['Gender_Mixed']] = 0

    df.loc[df['Gender'] == 'Male', ['Gender_Male']] = 1
    df.loc[df['Gender'] != 'Male', ['Gender_Male']] = 0

    # Age Recode
    df.loc[df['Age'] == 'Older', ['Age_Older']] = 1
    df.loc[df['Age'] != 'Older', ['Age_Older']] = 0

    df.loc[df['Age'] == 'Mixed Age', ['Age_Mixed']] = 1
    df.loc[df['Age'] != 'Mixed Age', ['Age_Mixed']] = 0

    df.loc[df['Age'] == 'Young', ['Age_Younger']] = 1
    df.loc[df['Age'] != 'Young', ['Age_Younger']] = 0

    # Gender Recode
    df.loc[df['Gender'] == 'Female', ['Gender_Num']] = 1
    df.loc[df['Gender'] == 'Mixed Gender', ['Gender_Num']] = 2
    df.loc[df['Gender'] == 'Male', ['Gender_Num']] = 3

    # Age Recode
    df.loc[df['Age'] == 'Older', ['Age_Num']] = 1
    df.loc[df['Age'] == 'Mixed Age', ['Age_Num']] = 2
    df.loc[df['Age'] == 'Young', ['Age_Num']] = 3

    if print_info is True:
        df_gender_age_info(df)

    return df


# %%
# Funtion to print df gender and age info
def df_gender_age_info(
    df,
    ivs_all=ivs_all,
):
    # Print Info
    print('\nDF INFO:\n')
    df.info()

    for iv in ivs_all:
        try:
            print('='*20)
            print(f'{iv}:')
            print('-'*20)
            print(f'{iv} Counts:\n{df[f"{iv}"].value_counts()}')
            print('-'*20)
            print(f'{iv} Percentages:\n{df[f"{iv}"].value_counts(normalize=True).mul(100).round(1).astype(float)}')
            try:
                print('-'*20)
                print(f'{iv} Mean: {df[f"{iv}"].mean().round(3).astype(float)}')
                print('-'*20)
                print(f'{iv} Standard Deviation: {df[f"{iv}"].std().round(3).astype(float)}')
            except Exception:
                pass
        except Exception:
            print(f'{iv} not available.')

    print('\n')


# %%
# Funtion to print df gender and age info
def df_warm_comp_info(
    df, dvs_all=['Warmth', 'Warmth_Probability', 'Competence', 'Competence_Probability'], print_info=False,
):
    # Print Info
    print('\nDF INFO:\n')
    df.info()

    if print_info is True:
        for dv in dvs_all:
            if '_Probability' not in dv:
                try:
                    print('='*20)
                    print(f'{dv}:')
                    print('-'*20)
                    print(f'{dv} Counts:\n{df[f"{dv}"].value_counts()}')
                    print('-'*20)
                    print(f'{dv} Percentages:\n{df[f"{dv}"].value_counts(normalize=True).mul(100).round(1).astype(float)}')
                    print('-'*20)
                    print(f'{dv} Means: {df[f"{dv}"].mean().round(3).astype(float)}')
                    print('-'*20)
                    print(f'{dv} Standard Deviation: {df[f"{dv}"].std().round(3).astype(float)}')
                except Exception:
                    print(f'{dv} not available.')

    print('\n')


# %%
# Function to plot df values
def value_count_df_plot(which_df, main_cols, num_unique_values=100, filter_lt_pct=.05, height=1300, width=1500, align='left'):
    cols = []
    nunique = []
    df = which_df #specify which df you're using
    which_df = df

    for c in which_df.columns:
        cols.append(c)
        nunique.append(which_df[c].nunique())
    df_cols = pd.DataFrame(nunique)
    df_cols['cols'] = cols
    df_cols.columns = ['nunique','column']
    df_cols = df_cols[['column','nunique']]
    df_cols_non_unique = df_cols[
            (df_cols['nunique'] <= df_cols.shape[0])
            & (df_cols['nunique'] > 1)
        ].sort_values(by='nunique',ascending=True)
    merch_cols = list(df_cols_non_unique.column)
    # print(df_cols_non_unique.shape)
    df_cols_non_unique.head()

    #lte 30 unique values in any column
    num_unique_values = num_unique_values
    df_cols_non_unique = df_cols_non_unique[df_cols_non_unique['nunique'] <= num_unique_values]
    list_non_unique_cols = list(df_cols_non_unique['column'])
    print('total number of cols with lte', num_unique_values, 'unique values:', len(list_non_unique_cols))

    #include main cols
    main_cols = [main_cols]
    number_of_main_cols = len(main_cols)

    #append main cols with interesting cols
    interestin_cols = main_cols + list_non_unique_cols

    #specify interesting cols
    df1 = df.loc[:, df.columns.isin(interestin_cols)]
    df1 = df1.iloc[:,:-1]

    #get value counts for each value in each col
    def value_counts_col(df,col):
        df = df
        value_counts_df = pd.DataFrame(round(df[col].value_counts(normalize=True),2).reset_index())
        value_counts_df.columns = ['value','value_counts']
        value_counts_df['feature'] = col
        return value_counts_df

    all_cols_df = []
    for i in df1.columns[number_of_main_cols:]:
        dfs = value_counts_col(df1,i)
        all_cols_df.append(dfs)

    #append column values to end of column
    which_df = pd.concat(all_cols_df)
    which_df['value'] = which_df['value'].fillna('null')
    which_df['feature_value'] = which_df['feature'] + '_' + which_df['value'].map(str)
    which_df = which_df.drop(['value','feature'], axis='columns', errors='ignore')
    which_df = which_df[['feature_value','value_counts']]
    which_df = which_df.sort_values(by='value_counts',ascending=False)

    #filter out less than x% features
    filter_lt_pct = filter_lt_pct
    which_df = which_df[which_df['value_counts'] >= filter_lt_pct]

    print('df shape:', which_df.shape,'\n')

    #table plot
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=list(which_df.columns)
                    ,fill_color='black'
                    ,align=align
                    ,font=dict(color='white', size=12)
                    ,line_color=['darkgray','darkgray','darkgray']
                   )
        ,cells=dict(values=[which_df['feature_value'],which_df['value_counts']]
                   ,fill_color='black'
                   ,align=align
                   ,font=dict(color='white', size=12)
                   ,line_color=['darkgray','darkgray','darkgray']
                   )
    )
    ])
    fig2.update_layout(height=100, margin=dict(r=0, l=0, t=0, b=0))
    fig2.show()

    #bar plot
    fig1 = px.bar(which_df
                     ,y='feature_value'
                     ,x='value_counts'
                     ,height=height
                     ,width=width
                     ,template='plotly_dark'
                     ,text='value_counts'
                     ,title='<b>Outlier Values of Interesting Cols within Dataset'
                    )
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(showgrid=False)
    fig1.show()

    # return fig1,fig2


# %% Function to visaulize
def get_viz(df_name, df_df, dataframes, args=get_args()):
    from setup_module.params import analysis_columns, image_save_format

    # Visualize data balance
    dataframes[df_name]['Warmth'].value_counts()
    dataframes[df_name]['Competence'].value_counts()
    warm_comp_count = (
        dataframes[df_name][analysis_columns]
        .reset_index()
        .groupby(analysis_columns)
        .count()
        .sort_values(by='index')
    )
    fig, ax = plt.subplots()
    fig.suptitle(f'{df_name}: Warmth and Competence Sentence Counts', fontsize=16.0)
    warm_comp_count.plot(kind='barh', stacked=True, legend=True, color='blue', ax=ax).grid(
        axis='y'
    )
    if args['save_enabled'] is True:
        fig.savefig(f'{args["plot_save_path"]}{df_name} - Warmth and Competence Sentence Counts.{image_save_format}', format=image_save_format, dpi=3000, bbox_inches='tight')

    fig.show()
    plt.pause(0.1)


# %%
def set_language_requirement(
    df_jobs,
    str_variable = 'Job Description',
    dutch_requirement_pattern = r'[Ll]anguage: [Dd]utch|[Dd]utch [Pp]referred|[Dd]utch [Re]quired|[Dd]utch [Ll]anguage|[Pp]roficient in [Dd]utch|[Ss]peak [Dd]utch|[Kk]now [Dd]utch',
    english_requirement_pattern = r'[Ll]anguage: [Ee]nglish|[Ee]nglish [Pp]referred|[Ee]nglish [Re]quired|[Ee]nglish [Ll]anguage|[Pp]roficient in [Ee]nglish|[Ss]peak [Ee]nglish|[Kk]now [Ee]nglish',
    args=get_args(),
    ):
    # Language requirements
    # Dutch
    print('Setting Dutch language requirements.')
    if 'Dutch Requirement' in df_jobs.columns:
        df_jobs.drop(columns=['Dutch Requirement'], inplace=True)
    df_jobs['Dutch Requirement'] = np.where(
        df_jobs[str_variable].str.contains(dutch_requirement_pattern),
        'Yes',
        'No',
    )

    # English
    print('Setting English language requirements.')
    if 'English Requirement' in df_jobs.columns:
        df_jobs.drop(columns=['English Requirement'], inplace=True)
    df_jobs['English Requirement'] = np.where(
        df_jobs[str_variable].str.contains(english_requirement_pattern),
        'Yes',
        'No',
    )

    return df_jobs


# %%
def set_sector_and_percentage(
    df_jobs,
    sector_dict_new=False,
    age_limit = 45,
    age_ratio = 10,
    gender_ratio = 20,
    args=get_args(),
):

    sbi_english_keyword_list = args['sbi_english_keyword_list']
    sbi_english_keyword_dict = args['sbi_english_keyword_dict']
    sbi_sectors_dict = args['sbi_sectors_dict']
    sbi_sectors_dict_full = args['sbi_sectors_dict_full']
    sbi_sectors_dom_gen = args['sbi_sectors_dom_gen']
    sbi_sectors_dom_age = args['sbi_sectors_dom_age']
    trans_keyword_list = args['trans_keyword_list']
    df_sectors = get_sector_df_from_cbs()

    if sector_dict_new is True:
        sector_vs_job_id_dict = make_job_id_v_sector_key_dict()
    elif sector_dict_new is False:
        with open(validate_path(f'{args["parent_dir"]}job_id_vs_sector_all.json'), encoding='utf-8') as f:
            sector_vs_job_id_dict = json.load(f)

    # Set Sectors
    print('Setting sector.')
    if 'Sector' in df_jobs.columns:
        df_jobs.drop(columns=['Sector'], inplace=True)
    for sect, sect_dict in sector_vs_job_id_dict.items():
        for keyword, job_ids in sect_dict.items():
            df_jobs.loc[df_jobs['Job ID'].astype(str).progress_apply(lambda x: x.lower().strip()).isin([str(i) for i in job_ids]), 'Sector'] = str(sect).lower().strip()
    # if 'Search Keyword' in df_jobs.columns:
    #     if df_jobs['Sector'].isnull().values.any() or df_jobs['Sector'].isnull().sum() > 0 or df_jobs['Sector'].isna().values.any() or df_jobs['Sector'].isna().sum() > 0:
    #         df_sectors = get_sector_df_from_cbs()
    #         for idx, row in df_sectors.iterrows():
    #             if isinstance(row[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')], list) and df_jobs['Job ID'].astype(str).progress_apply(lambda x: x.lower().strip()).isin([str(i) for i in row[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')]]):
    #                     print(row[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Sector Name')])

    print('Setting sector code and percentages.')
    # Add gender and age columns
    sect_cols = ['Sector Code', '% Female', '% Male', '% Older', '% Younger']
    for col in sect_cols:
        if col in df_jobs.columns:
            df_jobs.drop(columns=col, inplace=True)
    df_jobs = df_jobs.reindex(columns=[*df_jobs.columns, *sect_cols], fill_value=np.nan)

    # Set Percentages
    # Open df
    df_sectors = get_sector_df_from_cbs()
    for index, row in df_jobs.iterrows():
        for idx, r in df_sectors.iterrows():
            if str(row['Sector']).strip().lower() == str(r[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Sector Name')]).strip().lower():
                df_jobs.loc[index, 'Sector Code'] = r[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Code')]
                df_jobs.loc[index, '% Female'] = r[('Gender', 'Female', '% per Sector')]
                df_jobs.loc[index, '% Male'] = r[('Gender', 'Male', '% per Sector')]
                df_jobs.loc[index, '% Older'] = r[('Age', f'Older (>= {age_limit} years)', '% per Sector')]
                df_jobs.loc[index, '% Younger'] = r[('Age', f'Younger (< {age_limit} years)', '% per Sector')]

    print('Done setting sector percentages.')

    return df_jobs


# # %%
# def set_sector_and_percentage_helper(df_jobs, keyword, trans_keyword_list, args=get_args()):

#     for index, row in df_jobs.iterrows():
#         if row['Sector'] in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
#             df_jobs.loc[(df_jobs['Job ID'].astype(str).progress_apply(lambda x: x.strip().lower()).isin([i.strip().lower() for i in job_ids if isinstance (i, str)])) | (df_jobs['Search Keyword'].astype(str).progress_apply(lambda x: x.strip().lower()) == keyword.strip().lower()), 'Sector'] = sect.capitalize()
#             for index, row in df_jobs.iterrows():
#                 if row['Sector'] in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
#                     df_jobs.loc[(df_jobs['Search Keyword'].astype(str).progress_apply(lambda x: x.strip().lower()) == keyword.strip().lower()), 'Sector'] = sect.capitalize()
#                     if row['Sector'] in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
#                         trans_keyword_list.append(keyword.strip().lower())
#                         trans_keyword_list = save_trans_keyword_list(trans_keyword_list)

#     return df_jobs


# %%
def set_gender_age(
    df_jobs,
    id_dict_new=False,
    args=get_args(),
):

    sbi_english_keyword_list = args['sbi_english_keyword_list']
    sbi_english_keyword_dict = args['sbi_english_keyword_dict']
    sbi_sectors_dict = args['sbi_sectors_dict']
    sbi_sectors_dict_full = args['sbi_sectors_dict_full']
    sbi_sectors_dom_gen = args['sbi_sectors_dom_gen']
    sbi_sectors_dom_age = args['sbi_sectors_dom_age']
    trans_keyword_list = args['trans_keyword_list']

    if id_dict_new is True:
        job_id_dict = make_job_id_v_genage_key_dict()

    elif id_dict_new is False:
        with open(validate_path(f'{args["parent_dir"]}job_id_vs_all.json'), encoding='utf8') as f:
            job_id_dict = json.load(f)

    print('Setting gender and age.')
    # Add gender and age columns
    gen_age_cols = ['Gender', 'Age']
    for col in gen_age_cols:
        if col in df_jobs.columns:
            df_jobs.drop(columns=col, inplace=True)
    df_jobs = df_jobs.reindex(columns=[*df_jobs.columns, *gen_age_cols], fill_value=np.nan)

    # Gender
    print('Setting gender.')
    try:
        for sect, cat in sbi_sectors_dom_gen.items():
            df_jobs.loc[df_jobs['Sector'].astype(str).progress_apply(lambda x: x.lower().strip()) == str(sect).lower().strip(), 'Gender'] = str(cat)
    except Exception as e:
        for cat in ['Mixed Gender', 'Male', 'Female']:
            df_jobs.loc[df_jobs['Job ID'].astype(str).progress_apply(lambda x: x.lower().strip()).isin([str(i) for i in job_id_dict[cat]]), 'Gender'] = str(cat)

    # Age
    print('Setting age.')
    try:
        for sect, cat in sbi_sectors_dom_age.items():
            df_jobs.loc[df_jobs['Sector'].astype(str).progress_apply(lambda x: x.lower().strip()) == str(sect).lower().strip(), 'Age'] = str(cat)
    except Exception as e:
        for cat in ['Mixed Age', 'Young', 'Older']:
            df_jobs.loc[df_jobs['Job ID'].astype(str).progress_apply(lambda x: x.lower().strip()).isin([str(i) for i in job_id_dict[cat]]), 'Age'] = str(cat)

    print('Categorizing gender and age')
    df_jobs = categorize_df_gender_age(df_jobs)
    df_jobs = dummy_code_df_gender_age(df_jobs)

    print('Done setting gender and age.')

    return df_jobs


# %%
def set_gender_age_sects_lang(df_jobs, str_variable, id_dict_new=False, args=get_args()):

    # now = datetime.datetime.now().total_seconds()

    df_jobs=set_language_requirement(df_jobs, str_variable=str_variable)
    # print(f'Time taken for set_language_requirement: {now-datetime.datetime.now().total_seconds()}')
    # now = datetime.datetime.now().total_seconds()
    df_jobs=set_sector_and_percentage(df_jobs, sector_dict_new=id_dict_new)
    # print(f'Time taken for set_sector_and_percentage: {now-datetime.datetime.now().total_seconds()}')
    # now = datetime.datetime.now().total_seconds()
    df_jobs=set_gender_age(df_jobs, id_dict_new=id_dict_new)
    # print(f'Time taken for set_gender_age: {now-datetime.datetime.now().total_seconds()}')

    return df_jobs


# %%
# Load and merge existing dict and df
def load_merge_dict_df(
    keyword: str,
    save_path: str,
    df_file_name: str,
    json_file_name: str,
    args=get_args()
):
    # df_jobs
    if is_non_zero_file(save_path + df_file_name.lower()) is True:
        if args['print_enabled'] is True:
            print(
                f'A DF with the name "{df_file_name.lower()}" already exists.\nNew data will be appended to the file.'
            )
        df_old_jobs = pd.read_csv(save_path + df_file_name.lower())
        if not df_old_jobs.empty:
            df_old_jobs = clean_df(df_old_jobs)
        else:
            print(f'{df_file_name} is empty!')

    elif is_non_zero_file(save_path + df_file_name.lower()) is False:
        if args['print_enabled'] is True:
            print(f'No DF with the name "{df_file_name.lower()}" found.')
        df_old_jobs = pd.DataFrame()
    if args['print_enabled'] is True:
        print(f'Old jobs DF of length: {df_old_jobs.shape[0]}.')

    # jobs
    if is_non_zero_file(save_path + json_file_name.lower()) is True:
        if args['print_enabled'] is True:
            print(
                f'A list of dicts with the name "{json_file_name.lower()}" already exists.\nNew data will be appended to this file.'
            )
        with open(save_path + json_file_name, encoding='utf8') as f:
            old_jobs = json.load(f)
        # old_jobs = remove_dupe_dicts(old_jobs)
    elif is_non_zero_file(save_path + json_file_name.lower()) is False:
        if args['print_enabled'] is True:
            print(f'No list of dicts with the name "{json_file_name.lower()}" found.')
        old_jobs = []
    if args['print_enabled'] is True:
        print(f'Old jobs dict of length: {len(old_jobs)}.')

    # Merge dicts and df
    if df_old_jobs is not None:
        # Convert old df to jobs
        try:
            jobs_from_df_old_jobs = df_old_jobs.reset_index().to_dict('records')
        except Exception:
            jobs_from_df_old_jobs = df_old_jobs.reset_index(drop=True).to_dict(
                'records'
            )

        # Merge jobs from df to jobs from file
        if args['print_enabled'] is True:
            print('Merging DF with jobs into new list.')
        old_jobs.extend(jobs_from_df_old_jobs)
        # old_jobs = remove_dupe_dicts(old_jobs)
        jobs = []
        for myDict in old_jobs:
            if myDict not in jobs:
                jobs.append(myDict)

        if is_non_zero_file(save_path + df_file_name.lower()) is True or is_non_zero_file(save_path + json_file_name.lower()) is True:
            with open(save_path + json_file_name, 'w', encoding='utf8') as f:
                json.dump(jobs, f)
    elif df_old_jobs is None:
        jobs = old_jobs

    if args['print_enabled'] is True:
        print('-' * 20)
        if len(jobs) > 0:
            print(
                f'List of dicts of length {len(jobs)} was loaded for {jobs[0]["Search Keyword"]}.'
            )

        elif len(jobs) == 0:
            print(f'List of dicts of length {len(jobs)} was loaded for {keyword}.')
        print('-' * 20)

    return jobs, df_old_jobs


# %%
# Function to save df as csv
def save_df(
    keyword: str,
    df_jobs,
    save_path: str,
    keyword_file: str,
    df_file_name: str,
    print_enabled: bool = False,
    clean_enabled: bool = True,
    args=get_args(),
):
    if print_enabled is True:
        print(f'Saving {keyword} jobs data to df...')

    if (not df_jobs.empty) and (len(df_jobs != 0)):
        if print_enabled is True:
            print(f'Cleaning {keyword} df.')

        # Drop duplicates and -1 for job description
        if clean_enabled is True:
            df_jobs = clean_df(df_jobs)

        # Search keyword
        try:
            search_keyword = df_jobs['Search Keyword'].iloc[0].lower().replace("-Noon's MacBook Pro",'')
        except KeyError:
            df_jobs.reset_index(drop=True, inplace=True)
            search_keyword = df_jobs['Search Keyword'].iloc[0].lower().replace("-Noon's MacBo an and.  ok Pro",'')
        except IndexError:
            print(len(df_jobs))

        # Save df to csv
        if print_enabled is True:
            print(f'Saving {keyword.lower()} jobs df of length {len(df_jobs.index)} to csv as {df_file_name.lower()} in location {save_path}')

        df_jobs.to_csv(save_path + df_file_name, mode='w', sep=',', header=True, index=True)
        df_jobs.to_csv(save_path + df_file_name.split(args['file_save_format_backup'])[0]+'txt', mode='w', sep=',', header=True, index=True)

        # if (not df_jobs.empty) and (len(df_jobs != 0)):
        #     try:
        #         df_jobs_pyarrow = save_metadata(df_jobs, df_file_name, save_path)
        #     except Exception:
        #         pass

    elif df_jobs.empty:
        if print_enabled is True:
            print(f'Jobs DataFrame is empty since no jobs results were found for {str(keyword)}. Moving on to next search.')

    return df_jobs


# %%
# Post collection cleanup
def site_loop(site, site_list, site_from_list, args, df_list_from_site=None):

    if site_from_list is True:
        df_list_from_site = []
        for site in tqdm.tqdm(site_list):
            if args['print_enabled'] is True:
                print('-' * 20)
                print(f'Cleaning up LIST OF DFs for {site}.')
            glob_paths = glob.glob(f'{scraped_data}/{site}/Data/*.json')+glob.glob(f'{scraped_data}/{site}/Data/*.csv')+glob.glob(f'{scraped_data}/{site}/Data/*.xlsx')

            yield site, df_list_from_site, glob_paths

    elif site_from_list is False:
        df_list_from_site = None
        if args['print_enabled'] is True:
            print('-' * 20)
            print('Cleaning up LIST OF DFs from all sites.')
        glob_paths = glob.glob(f'{scraped_data}/*/Data/*.json')+glob.glob(f'{scraped_data}/*/Data/*.csv')+glob.glob(f'{scraped_data}/*/Data/*.xlsx')

        yield site, df_list_from_site, glob_paths


# %%
def site_save(site, df_jobs, args, chunk_size = 1024 * 1024):
    if args['save_enabled'] is True:
        print(f'Saving df_{site}_all_jobs.{args["file_save_format"]}')
        with open(args['df_dir'] + f'df_{site}_all_jobs.{args["file_save_format"]}', 'wb') as f:
            pickle.dump(df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Done saving df_{site}_all_jobs.{args["file_save_format"]}')


# %%
def keyword_loop(keyword, keywords_from_list, glob_paths, args, translator, df_list_from_keyword=None):

    if keywords_from_list is True:
        df_list_from_keyword = []
        for glob_path in glob_paths:
            if 'dict_' in glob_path and glob_path.endswith('.json'):
                keyword = glob_path.split('dict_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
            elif 'df_' in glob_path and (glob_path.endswith('.csv') or glob_path.endswith('.xlsx')):
                keyword = glob_path.split('df_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
            if '_' in keyword:
                keyword = ' '.join(keyword.split('_')).strip().lower()

            if args['print_enabled'] is True:
                print(f'Post collection cleanup for {keyword}.')
            yield keyword, df_list_from_keyword

    elif keywords_from_list is False:
        keyword=keyword
        df_list_from_keyword = None
        if args['print_enabled'] is True:
            print(f'Post collection cleanup for {keyword}.')
        yield keyword, df_list_from_keyword


# %%
def keyword_save(keyword, site, df_jobs, args):
    if args['save_enabled'] is True:
        print(f'Saving df_{site}_{keyword}_all_jobs.{args["file_save_format"]}')
        with open(args['df_dir'] + f'df_{site}_{keyword}_all_jobs.{args["file_save_format"]}', 'wb') as f:
            pickle.dump(df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Done saving df_{site}_{keyword}_all_jobs.{args["file_save_format"]}')


# %%
def post_cleanup(
    site_from_list=True,
    keywords_from_list=True,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    job_id_save_enabled=False,
    job_sector_save_enabled=False,
    keyword='',
    site='',
    keywords_list=None,
    all_save_path = f'job_id_vs_all.json',
    int_variable: str = 'Job ID',
    str_variable: str = 'Job Description',
    args=get_args(),
    translator = Translator(),
    translate_keywords=True,
    fix_na_enabled = True,
    save_enabled=True
):

    print(
        f'NOTE: The function "post_cleanup" contains the following optional (default) arguments:\n{get_default_args(post_cleanup)}'
    )
    print('-' * 20)

    if keywords_list is None:
        keywords_list = []

    # Get original collected sectors
    trans_keyword_list = get_trans_keyword_list()

    for site, df_list_from_site, glob_paths in tqdm.tqdm(site_loop(site=site, site_list=site_list, site_from_list=site_from_list, args=args)):
        for keyword, df_list_from_keyword in tqdm.tqdm(keyword_loop(keyword=keyword, keywords_from_list=keywords_from_list, glob_paths=glob_paths, args=args, translator=translator)):

            trans_keyword = keyword.strip().lower()

            if translate_keywords is True and detect(trans_keyword) != 'en':
                while True:
                    try:
                        trans_keyword = translator.translate(trans_keyword).text.strip().lower()
                    except Exception as e:
                        time.sleep(0.3)
                        continue
                    break

            for w_keyword, r_keyword in keyword_trans_dict.items():
                if str(trans_keyword.lower()) == w_keyword.lower():
                    trans_keyword = r_keyword.strip().lower()
                trans_keyword_list.append(trans_keyword)

            try:
                df_jobs = post_cleanup_helper(keyword, site)
                print(f'DF {trans_keyword.title()} collected.')

                if df_jobs.empty and args['print_enabled'] is True:
                    print(f'DF {trans_keyword.title()} not collected yet.')

            except Exception:
                if args['print_enabled'] is True:
                    print(f'An error occured with finding DF {keyword}.')
                df_jobs = pd.DataFrame()

            else:
                if args['print_enabled'] is True:
                    print(f'Cleaning up LIST OF DFs for {keyword}.')

                if site_from_list is True and keywords_from_list is True:
                    if (not df_jobs.empty) and (len(df_jobs != 0)):
                        df_list_from_keyword.append(df_jobs)
                    df_list_from_site.append(df_list_from_keyword)
                    df_jobs = df_list_from_site
                    # site_save(site, df_jobs, args=args)
                elif site_from_list is True and keywords_from_list is False:
                    if (not df_jobs.empty) and (len(df_jobs != 0)):
                        df_list_from_site.append(df_jobs)
                    df_jobs = df_list_from_site
                    # site_save(site, df_jobs, args=args)
                elif site_from_list is False and keywords_from_list is True:
                    if (not df_jobs.empty) and (len(df_jobs != 0)):
                        df_list_from_keyword.append(df_jobs)
                    df_jobs = df_list_from_keyword
                #     keyword_save(keyword, site, df_jobs, args=args)
                # elif site_from_list is False and keywords_from_list is False:
                #     keyword_save(keyword, site, df_jobs, args=args)

    if fix_na_enabled is True:
        for lst in df_jobs:
            for df in lst:
                if isinstance(df, pd.DataFrame):
                    df = set_gender_age_sects_lang(df, str_variable=str_variable)

    if translate_keywords is True:
        trans_keyword_list = save_trans_keyword_list(trans_keyword_list)

    if save_enabled is True:
        print(f'Saving df_jobs_post_cleanup.{args["file_save_format"]}')
        with open(args['df_dir'] + f'df_jobs_post_cleanup.{args["file_save_format"]}', 'wb') as f:
            pickle.dump(df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Done saving df_jobs_post_cleanup.{args["file_save_format"]}')

        # print(f'Saving df_jobs_post_cleanup.{args["file_save_format_backup"]}')
        # with open(df_dir + f'df_jobs_post_cleanup.{file_save_format_backup}', 'w', newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(df_jobs)
        # print(f'Done saving df_jobs_post_cleanup.{args["file_save_format_backup"]}')

    if job_id_save_enabled is True:
        job_id_dict = make_job_id_v_genage_key_dict(site_from_list=False)
    if job_sector_save_enabled is True:
        sector_vs_job_id_dict = make_job_id_v_sector_key_dict(site_from_list=False)

    return df_jobs


# %%
# Get keywords and files for post collection df cleanup
def post_cleanup_helper(keyword, site, args=get_args()):
    (
        keyword_url,
        keyword_file,
        save_path,
        json_file_name,
        df_file_name,
        logs_file_name,
        filemode,
    ) = main_info(keyword, site)

    jobs, df_old_jobs = load_merge_dict_df(
        keyword, save_path, df_file_name, json_file_name
    )

    if is_non_zero_file(save_path + df_file_name.lower()) is True or is_non_zero_file(save_path + json_file_name.lower()) is True:
        with open(save_path + json_file_name, 'w', encoding='utf8') as f:
            json.dump(jobs, f)
        df_jobs = pd.DataFrame(jobs)
        if (not df_jobs.empty) and (len(df_jobs != 0)):
            # Save df as csv
            if args['save_enabled'] is True:
                df_jobs = save_df(
                    keyword,
                    df_jobs,
                    save_path,
                    keyword_file.lower(),
                    df_file_name.lower(),
                )
        elif (df_jobs.empty) or (len(df_jobs == 0)):
            if args['print_enabled'] is True:
                print(
                    f'Jobs DataFrame is empty since no jobs results were found for {str(keyword)}.'
                )

    elif is_non_zero_file(save_path + df_file_name.lower()) is False or is_non_zero_file(save_path + json_file_name.lower()) is False:
        if args['print_enabled'] is True:
            print(f'No jobs file found for {keyword} in path: {save_path}.')
        df_jobs = pd.DataFrame()

    return df_jobs


# %%
# Function to clean from old folder
def clean_from_old(
    site=None,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    files = None,
    exten_to_find = ['.json','.csv','.xlsx'],
    translator = Translator(),
    args=get_args(),
):
    if files is None:
        files = []

    if site is None and files == []:
        try:
            for site in site_list:
                for file_ in glob.glob(f'{scraped_data}/{site}/Data/*.json')+glob.glob(f'{scraped_data}/{site}/Data/*.csv')+glob.glob(f'{scraped_data}/{site}/Data/*.xlsx'):
                    files.append(file)
        except Exception as e:
            for file_ in glob.glob(f'{scraped_data}/*/Data/*.json')+glob.glob(f'{scraped_data}/*/Data/*.csv')+glob.glob(f'{scraped_data}/*/Data/*.xlsx'):
                files.append(file_)

    elif site is not None and files == []:
        for file_ in glob.glob(f'{scraped_data}/{site}/Data/*.json')+glob.glob(f'{scraped_data}/{site}/Data/*.csv')+glob.glob(f'{scraped_data}/{site}/Data/*.xlsx'):
            files.append(file_)

    for file_ in tqdm.tqdm(files):
        if site is None:
            site = file_.split(f'{code_dir}/')[1].split('/Data')[0].strip()
        if 'dict_' in file_ and file_.endswith('.json'):
            keyword = file_.split('dict_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
        elif 'df_' in file_ and (file_.endswith('.csv') or file_.endswith('.xlsx')):
            keyword = file_.split('df_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
        if '_' in keyword:
            keyword = ' '.join(keyword.split('_')).strip().lower()

        if detect(keyword) != 'en':
            while True:
                try:
                    trans_keyword = translator.translate(keyword).text.strip().lower()
                except Exception as e:
                    time.sleep(0.3)
                    continue
                break

        else:
            trans_keyword = keyword

        for w_keyword, r_keyword in keyword_trans_dict.items():
            if trans_keyword and trans_keyword != keyword:
                if str(trans_keyword.strip().lower()) == w_keyword.strip().lower():
                    trans_keyword = r_keyword.strip().lower()
                else:
                    trans_keyword = trans_keyword.strip().lower()

        print(f'Getting data for {trans_keyword}.')
        if trans_keyword != keyword:
            print(f'Translated from: {keyword}.')

        (
            keyword_url,
            keyword_file,
            save_path,
            json_file_name,
            df_file_name,
            logs_file_name,
            filemode,
        ) = main_info(keyword, site)
        if trans_keyword != keyword:
            (
                trans_keyword_url,
                trans_keyword_file,
                trans_save_path,
                trans_json_file_name,
                trans_df_file_name,
                trans_logs_file_name,
                trans_filemode,
            ) = main_info(trans_keyword, site)

        if is_non_zero_file(file_) is True:
            df_jobs = pd.DataFrame()

            if file_.endswith('.json'):
                try:
                    df_jobs_json = pd.read_json(file_, orient='records')
                except ValueError:
                    with open(file_) as f:
                        df_jobs_json = pd.DataFrame(json.load(f))
                df_jobs = df_jobs.append(df_jobs_json, ignore_index=True)
                if trans_keyword != keyword and is_non_zero_file(trans_save_path + trans_json_file_name.lower()) is True:
                    trans_df_jobs_json = pd.read_json(trans_save_path + trans_json_file_name.lower(), orient='records')
                    df_jobs = df_jobs.append(trans_df_jobs_json, ignore_index=True)

            if file_.endswith('.csv'):
                df_jobs_csv = pd.read_csv(file_)
                df_jobs = df_jobs.append(df_jobs_csv, ignore_index=True)
                if trans_keyword != keyword and is_non_zero_file(trans_save_path + trans_df_file_name.lower()) is True:
                    trans_df_jobs_csv = pd.read_csv(trans_save_path + trans_df_file_name.lower())
                    df_jobs = df_jobs.append(trans_df_jobs_csv, ignore_index=True)

            if file_.endswith('.xlsx'):
                df_jobs_xlsx = pd.read_excel(file_)
                df_jobs = df_jobs.append(df_jobs_xlsx, ignore_index=True)
                if trans_keyword != keyword and is_non_zero_file(trans_save_path + trans_df_file_name.lower()) is True:
                    trans_df_jobs_xlsx = pd.read_excel(trans_save_path + trans_df_file_name.lower().replace('csv', 'xlsx'))
                    df_jobs = df_jobs.append(trans_df_jobs_xlsx, ignore_index=True)

            if (not df_jobs.empty) and (len(df_jobs != 0)):
                df_jobs = clean_df(df_jobs)
                jobs = df_jobs.to_dict(orient='records')

                if is_non_zero_file(save_path + df_file_name.lower()) is True or is_non_zero_file(save_path + json_file_name.lower()) is True:
                    with open(save_path + json_file_name, 'w', encoding='utf8') as f:
                        json.dump(jobs, f)

                df_jobs = save_df(
                    keyword=keyword,
                    df_jobs=df_jobs,
                    save_path=save_path,
                    keyword_file=keyword_file.lower(),
                    df_file_name=df_file_name.lower(),
                    clean_enabled = False,
                )

        else:
            print(f'Data for {site} {keyword} is empty.')

    return df_jobs


# %%
# Function to match job sector to larger sectors
def make_job_id_v_sector_key_dict(
    site_from_list=False,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    all_save_path = f'job_id_vs_sector',
    args=get_args(),
    ):

    print(
        f'NOTE: The function "make_job_id_v_sector_key_dict" contains the following optional (default) arguments:\n{get_default_args(make_job_id_v_sector_key_dict)}'
    )

    sib_5_loc = validate_path(f'{args["parent_dir"]}Sectors + Age and Gender Composition of Industires and Jobs/Found Data/SBI_ALL_NACE_REV2.csv')

    # Get keywords and paths to df_jobs
    if site_from_list is True:
        for site in site_list:
            if args['print_enabled'] is True:
                print(f'Getting job ids for {site}.')
            df_jobs_paths = list((glob.glob(f'{scraped_data}/{site}/Data/*.csv')))
            sector_vs_job_id_dict = make_job_id_v_sector_key_dict_helper(
                df_jobs_paths=df_jobs_paths, all_save_path=f'{all_save_path}_{site}',
            )
    elif site_from_list is False:
        df_jobs_paths = glob.glob(f'{scraped_data}/*/Data/*.csv')
        sector_vs_job_id_dict = make_job_id_v_sector_key_dict_helper(
            df_jobs_paths=df_jobs_paths, all_save_path=f'{all_save_path}_all',
        )

    return sector_vs_job_id_dict


# %%
def make_job_id_v_sector_key_dict_helper(
    df_jobs_paths,
    all_save_path,
    sector_vs_job_id_dict=defaultdict(lambda: defaultdict(list)),
    args=get_args(),
):

    sbi_english_keyword_list = args['sbi_english_keyword_list']
    sbi_english_keyword_dict = args['sbi_english_keyword_dict']
    sbi_sectors_dict = args['sbi_sectors_dict']
    sbi_sectors_dict_full = args['sbi_sectors_dict_full']
    sbi_sectors_dom_gen = args['sbi_sectors_dom_gen']
    sbi_sectors_dom_age = args['sbi_sectors_dom_age']
    trans_keyword_list = args['trans_keyword_list']

    for path in df_jobs_paths:
        df_jobs = pd.read_csv(path)

        for index, row in tqdm.tqdm(df_jobs.iterrows()):
            if row['Search Keyword'] not in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
                search_keyword = str(row['Search Keyword'].strip().lower().replace("-Noon's MacBook Pro",'').strip().lower())
                for w_keyword, r_keyword in keyword_trans_dict.items():
                    if search_keyword == w_keyword.lower():
                        df_jobs.loc[index, 'Search Keyword'] = r_keyword.strip().lower()
                        df_jobs.to_csv(path)
                trans_keyword_list.append(search_keyword)

                for sector, keywords_list in args['sbi_sectors_dict_full'].items():
                    if search_keyword in str(keywords_list):
                        sector_vs_job_id_dict[str(sector)][str(search_keyword)].append(str(row['Job ID']))

                # for code, sect_dict in sbi_sectors_dict.items():
                #     if str(row['Search Keyword']) in str(sect_dict['Used_Sector_Keywords']):
                #         sector_vs_job_id_dict[str(sect_dict['Sector_Name'])][str(row['Search Keyword'])].append(row['Job ID'])

    trans_keyword_list = save_trans_keyword_list(trans_keyword_list)

    if args['save_enabled'] is True:
        with open(f'{args["parent_dir"]}{all_save_path}.json', 'w', encoding='utf8') as f:
            json.dump(sector_vs_job_id_dict, f)
    elif args['save_enabled'] is False:
        print('No job id matching save enabled.')

    return sector_vs_job_id_dict


# %%
# Function to match job IDs with gender and age in dict
def make_job_id_v_genage_key_dict(
    site_from_list=False,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    all_save_path = f'job_id_vs',
    args=get_args()):

    print(
        f'NOTE: The function "make_job_id_v_genage_key_dict" contains the following optional (default) arguments:\n{get_default_args(make_job_id_v_genage_key_dict)}'
    )

    # Get keywords and paths to df_jobs
    if site_from_list is True:
        for site in tqdm.tqdm(site_list):
            if args['print_enabled'] is True:
                print(f'Getting job ids for {site}.')
            df_jobs_paths = list((glob.glob(f'{scraped_data}/{site}/Data/*.csv')))
            job_id_dict = make_job_id_v_genage_key_dict_helper(
                df_jobs_paths=df_jobs_paths, all_save_path=f'{all_save_path}_{site}'
            )
    elif site_from_list is False:
        df_jobs_paths = glob.glob(f'{scraped_data}/*/Data/*.csv')
        job_id_dict = make_job_id_v_genage_key_dict_helper(
            df_jobs_paths=df_jobs_paths, all_save_path=f'{all_save_path}_all'
        )

    return job_id_dict


# %%
def make_job_id_v_genage_key_dict_helper(
    df_jobs_paths,
    all_save_path,
    job_id_dict=defaultdict(list),
    args=get_args(),
):

    trans_keyword_list = args['trans_keyword_list']

    for path in tqdm.tqdm(df_jobs_paths):
        df_jobs = pd.read_csv(path)
        for index, row in tqdm.tqdm(df_jobs.iterrows()):

            if row['Search Keyword'] not in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
                search_keyword = str(row['Search Keyword'].replace("-Noon's MacBook Pro",'').strip().lower())
                for w_keyword, r_keyword in keyword_trans_dict.items():
                    if search_keyword == w_keyword.lower():
                        df_jobs.loc[index, 'Search Keyword'] = r_keyword.strip().lower()
                        df_jobs.to_csv(path)
                trans_keyword_list.append(search_keyword)

                for (
                    fem_keyword,
                    male_keyword,
                    gen_keyword,
                    old_keyword,
                    young_keyword,
                    age_keyword,
                ) in itertools.zip_longest(
                    args['keywords_womenvocc'],
                    args['keywords_menvocc'],
                    args['keywords_genvsect'],
                    args['keywords_oldvocc'],
                    args['keywords_youngvocc'],
                    args['keywords_agevsect'],
                ):
                    if search_keyword == str(fem_keyword).strip().lower().replace("-Noon's MacBook Pro",''):
                        job_id_dict['Female'].append(str(row['Job ID']))
                    else:
                        job_id_dict['Mixed Gender'].append(str(row['Job ID']))
                    if search_keyword == str(male_keyword).strip().lower().replace("-Noon's MacBook Pro",''):
                        job_id_dict['Male'].append(str(row['Job ID']))
                    else:
                        job_id_dict['Mixed Gender'].append(str(row['Job ID']))

                    if search_keyword == str(old_keyword).strip().lower().replace("-Noon's MacBook Pro",''):
                        job_id_dict['Older'].append(str(row['Job ID']))
                    else:
                        job_id_dict['Mixed Age'].append(str(row['Job ID']))
                    if search_keyword == str(young_keyword).strip().lower().replace("-Noon's MacBook Pro",''):
                        job_id_dict['Young'].append(str(row['Job ID']))
                    else:
                        job_id_dict['Mixed Age'].append(str(row['Job ID']))

    trans_keyword_list = save_trans_keyword_list(trans_keyword_list)
    if args['save_enabled'] is True:
        with open(f'{args["parent_dir"]}{all_save_path}.json', 'w', encoding='utf8') as f:
            json.dump(job_id_dict, f)

    elif args['save_enabled'] is False:
        print('No job id matching save enabled.')

    return job_id_dict


# %%
# Function to split job ads to sentences
def split_df_jobs_to_df_sent(
    df_for_analysis,
    lst_col='Job Description',
    pattern=r'[\n\r]+|(?<=[a-z]\.)(?=\s*[A-Z])|(?<=[a-z])(?=[A-Z])',
    args=get_args(),
):
    dff = df_for_analysis.assign(
        **{
            lst_col: df_for_analysis[lst_col]
            .swifter.progress_bar(args['print_enabled'])
            progress_apply(lambda x: sent_tokenize(x))
        }
    )
    df_final = pd.DataFrame(
        {
            col: np.repeat(dff[col].values, dff[lst_col].str.len())
            for col in dff.columns.difference([lst_col])
        }
    ).assign(**{lst_col: np.concatenate(dff[lst_col].values)})[dff.columns.to_list()]

    return df_final


# %%
# Function to split job descriptions to sentences
def split_to_sentences(df_jobs, df_sentence_list=None, args=get_args()):
    print('-' * 20)
    print(
        f'NOTE: The function "get_args" which is used by the "split_to_sentences" function contains the following optional (default) arguments:\n{get_default_args(get_args)}\nYou can change these arguments by calling "get_args",  passing the desired variable values to "get_args" then passing "get_args" to "split_to_sentences".'
    )
    print('-' * 20)

    if df_sentence_list is None:
        df_sentence_list = []

    if isinstance(df_jobs, list):
        df_list = df_jobs
        if args['print_enabled'] is True:
            print(f'LIST OF {len(df_list)} DFs passed.')
            print('-' * 20)
        for df_jobs in df_list:
            if isinstance(df_jobs, pd.DataFrame):
                if (
                    (not df_jobs.empty)
                    and all(df_jobs['Language'] == str(args['language']))
                    and (df_jobs is not None)
                ):
                    if args['print_enabled'] is True:
                        print(f'DF OF LENGTH {len(df_jobs)} passed.')
                    try:
                        if args['print_enabled'] is True:
                            print(
                                f'Processing DF from platform: {df_jobs["Platform"].iloc[0]}'
                            )
                        (
                            search_keyword,
                            job_id,
                            age,
                            args,
                            sentence_list,
                            sentence_dict,
                            df_sentence,
                            df_sentence_all,
                        ) = split_to_sentences_helper(df_jobs, args)
                        df_sentence_list.append(df_sentence)
                        if args['txt_save'] is True:
                            if args['print_enabled'] is True:
                                print(
                                    f'Saving {df_jobs["Search Keyword"].iloc[0]} DF to txt.'
                                )
                            write_all_to_txt(search_keyword, job_id, age, df_jobs, args)
                        elif args['txt_save'] is False:
                            if args['print_enabled'] is True:
                                print(
                                    f'No txt save enabled for DF {df_jobs["Search Keyword"].iloc[0]}.'
                                )
                    except Exception:
                        pass
                elif (
                    (df_jobs.empty)
                    or all(df_jobs['Language'] != str(args['language']))
                    or (df_jobs is None)
                ):
                    if df_jobs.empty:
                        if args['print_enabled'] is True:
                            print('DF is empty.')
                    elif all(df_jobs['Language'] != str(args['language'])):
                        if args['print_enabled'] is True:
                            print(
                                f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                            )
            elif isinstance(df_jobs, list):
                df_sentence, df_sentence_list = split_to_sentences(df_jobs)
                df_sentence_list.append(df_sentence)
    #         pbar.finish()
    elif isinstance(df_jobs, pd.DataFrame):
        if (
            (not df_jobs.empty)
            and all(df_jobs['Language'] == str(args['language']))
            and (df_jobs is not None)
        ):
            if args['print_enabled'] is True:
                print(f'DF OF LENGTH {len(df_jobs)} passed.')
            try:
                if args['print_enabled'] is True:
                    print(f'Processing DF from platform: {df_jobs["Platform"].iloc[0]}')
                    (
                        search_keyword,
                        job_id,
                        age,
                        args,
                        sentence_list,
                        sentence_dict,
                        df_sentence,
                        df_sentence_all,
                    ) = split_to_sentences_helper(df_jobs, args)
                df_sentence_list.append(df_sentence)
                if args['txt_save'] is True:
                    if args['print_enabled'] is True:
                        print(f'Saving {df_jobs["Search Keyword"].iloc[0]} DF to txt.')
                    write_all_to_txt(search_keyword, job_id, age, df_jobs, args)
                elif args['txt_save'] is False:
                    if args['print_enabled'] is True:
                        print(
                            f'No txt save enabled for DF {df_jobs["Search Keyword"].iloc[0]}.'
                        )
            except Exception:
                pass
        elif (
            (df_jobs.empty)
            or all(df_jobs['Language'] != str(args['language']))
            or (df_jobs is None)
        ):
            if df_jobs.empty:
                if args['print_enabled'] is True:
                    print('DF is empty.')
            elif all(df_jobs['Language'] != str(args['language'])):
                if args['print_enabled'] is True:
                    print(
                        f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                    )

    try:
        if (not df_sentence.empty) and (args['save_enabled'] is True):
            df_sentence_all.to_pickle(args['parent_dir'] + f'df_sentence_all_jobs.{args["file_save_format"]}')
            # pickle.dump(df_sentence_all, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif args['save_enabled'] is False:
            if args['print_enabled'] is True:
                print('No sentence save enabled.')
    except Exception:
        df_sentence = pd.DataFrame()
        if args['print_enabled'] is True:
            print('No sentence df found.')

    return df_sentence, df_sentence_list


# %%
def split_to_sentences_helper(df_jobs, args=get_args()):
    if (not df_jobs.empty) and (len(df_jobs != 0)):
        if args['print_enabled'] is True:
            print(
                f'DF {str(df_jobs["Search Keyword"].iloc[0])} of length {df_jobs.shape[0]} passed.'
            )
        try:
            search_keyword = '_'.join(
                str(df_jobs['Search Keyword'].iloc[0]).lower().split(' ').replace("-Noon's MacBook Pro",'')
            )
            if (df_jobs['Job ID'] == df_jobs['Job ID'].iloc[0]).all():
                job_id = str(df_jobs['Job ID'].iloc[0])
                age = str(df_jobs['Age'].iloc[0])
                (
                    sentence_list,
                    sentence_dict,
                    df_sentence,
                    df_sentence_all,
                ) = sent_tokenize_and_save_df(search_keyword, job_id, age, df_jobs, args)
            else:
                job_ids = list(df_jobs['Job ID'].unique())
                ages = list(df_jobs['Age'].unique())
                for job_id in job_ids:
                    for age in ages:
                        (
                            sentence_list,
                            sentence_dict,
                            df_sentence,
                            df_sentence_all,
                        ) = sent_tokenize_and_save_df(search_keyword, job_id, age, df_jobs, args)

                        yield (
                            search_keyword,
                            job_id,
                            age,
                            args,
                            sentence_list,
                            sentence_dict,
                            df_sentence,
                            df_sentence_all,
                        )

        except Exception as e:
            if args['print_enabled'] is True:
                print(e.json())
            (
                search_keyword,
                job_id,
                age,
                sentence_list,
                sentence_dict,
                df_sentence,
                df_sentence_all,
            ) = assign_all(7, None)
    elif df_jobs.empty:
        (
            search_keyword,
            job_id,
            age,
            sentence_list,
            sentence_dict,
            df_sentence,
            df_sentence_all,
        ) = assign_all(7, None)

    return (
        search_keyword,
        job_id,
        age,
        args,
        sentence_list,
        sentence_dict,
        df_sentence,
        df_sentence_all,
    )


# %%
# Function to tokenize and clean job descriptions from df based on language
def sent_tokenize_and_save_df(search_keyword, job_id, age, df_jobs, args=get_args()):
    if (not df_jobs.empty) and all(df_jobs['Language'] == str(args['language'])):
        path_to_csv = str(
            args['parent_dir']
            + f'Sentences DF/{str(args["language"])}/{age}/{str(" ".join(search_keyword.split("_")))}'
        )
        pathlib.Path(path_to_csv).mkdir(parents=True, exist_ok=True)

        lang_num = df_jobs.loc[
            df_jobs.Language == str(args['language']), 'Language'
        ].count()
        if args['print_enabled'] is True:
            print(f'{lang_num} jobs with language {str(args["language"])} found.')
        if lang_num > 0:
            if args['print_enabled'] is True:
                print(
                    f'Tokenizing DF {str(" ".join(search_keyword.split("_")))} of length {df_jobs.shape[0]} to sentences.'
                )
            sentence_dict = {}
            for index, row in df_jobs.iterrows():
                pattern = r'[\n]+|[,]{2,}|[|]{2,}|[\n\r]+|(?<=[a-z]\.)(?=\s*[A-Z])|(?=\:+[A-Z])'

                # sentence_list = []
                if row.loc['Language'] == str(args['language']):
#                     sentence_list = [re.split(pattern, sent) for sent in list(sent_tokenize(row['Job Description']))]
                    sentence_list = [
                        sent
                        for sentence in list(nlp(row['Job Description']).sents)
                        for sent in re.split(pattern, sentence)
                        if len(sent) != 0
                    ]
                    sentence_dict[str(row.loc['Job ID'])] = list(sentence_list)
                    sentence_dict['Search Keyword'] = row['Search Keyword']
                    sentence_dict['Gender'] = row['Gender']
                    sentence_dict['Age'] = row['Age']

            # Create DF sentence from sentence dict
            df_sentence_all = pd.DataFrame()
            for key, lst in sentence_dict.items():
                # if (key != "Search Keyword") and (key != "Gender") and (key != "Age"):
                df_sentence = pd.DataFrame(
                    [(key, sent) for sent in lst], columns=args['columns_fill_list']
                )
                df_sentence = df_sentence.reindex(
                    columns=[
                        *df_sentence.columns.to_list(),
                        *[
                            col
                            for col in args['columns_list']
                            if col not in args['columns_fill_list']
                        ],
                    ],
                    fill_value=0,
                )
                df_sentence_all = pd.concat([df_sentence, df_sentence_all])

                if not df_sentence.empty:
                    if args['print_enabled'] is True:
                        print(
                            f'Saving sentences DF {sentence_dict["Search Keyword"]} of length {df_sentence.shape[0]} and job ID {df_sentence["Job ID"].iloc[0]} to csv.'
                        )
                    df_sentence.to_csv(
                        path_to_csv
                        + f'/Job ID - {df_sentence["Job ID"].iloc[0]}_sentences_df.csv',
                        mode='w',
                        sep=',',
                        header=True,
                        index=True,
                    )
                    # Write DF to excel
                    if args['excel_save'] is True:
                        if args['print_enabled'] is True:
                            print(
                                f'Saving {df_jobs["Search Keyword"].iloc[0]} DF to excel.'
                            )
                        write_sentences_to_excel(
                            search_keyword, job_id, age, df_sentence, args
                        )
                elif df_sentence.empty:
                    if args['print_enabled'] is True:
                        print('Sentence DF is empty.')
                    (
                        sentence_list,
                        sentence_dict,
                        df_sentence,
                        df_sentence_all,
                    ) = assign_all(4, None)

            # Create DF which inclues Search Keyword and Age
            df_sentence_all['Search Keyword'] = sentence_dict['Search Keyword']
            df_sentence_all['Gender'] = sentence_dict['Gender']
            df_sentence_all['Age'] = sentence_dict['Age']
            df_sentence_all = df_sentence_all[
                ['Search Keyword', 'Gender', 'Age']
                + [
                    col
                    for col in df_sentence_all.columns
                    if col not in ['Search Keyword', 'Gender', 'Age']
                ]
            ]

            if not df_sentence_all.empty:
                if args['print_enabled'] is True:
                    print(
                        f'Saving ALL sentences DF {sentence_dict["Search Keyword"]} of length {df_sentence_all.shape[0]} and job ID {df_sentence_all["Job ID"].iloc[0]} to csv.'
                    )
                df_sentence_all.to_csv(
                    path_to_csv + f'/ALL_{search_keyword}_sentences_df.{args["file_save_format_backup"]}',
                    mode='w',
                    sep=',',
                    header=True,
                    index=True,
                )
            elif df_sentence_all.empty:
                if args['print_enabled'] is True:
                    print('ALL sentence DF is empty.')
                sentence_list, sentence_dict, df_sentence, df_sentence_all = assign_all(
                    4, None
                )

        elif lang_num <= 0:
            if args['print_enabled'] is True:
                print(f'No {str(args["language"])} language jobs found.')
            sentence_list, sentence_dict, df_sentence, df_sentence_all = assign_all(
                4, None
            )

    elif (df_jobs.empty) or all(df_jobs['Language'] != str(args['language'])):
        if df_jobs.empty:
            if args['print_enabled'] is True:
                print('DF is empty.')
        elif all(df_jobs['Language'] != str(args['language'])):
            if args['print_enabled'] is True:
                print(
                    f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                )
        sentence_list, sentence_dict, df_sentence, df_sentence_all = assign_all(4, None)

    return sentence_list, sentence_dict, df_sentence, df_sentence_all


# %%
# Function to save individual sentences in excel file
def write_sentences_to_excel(search_keyword, job_id, age, df_sentence, args=get_args()):
    if (df_sentence is not None) and (not df_sentence.empty):
        path_to_txt = str(
            args['parent_dir']
            + f'Jobs EXECL/{str(args["language"])}/{age}/{str(" ".join(search_keyword.split("_")))}'
        )
        pathlib.Path(path_to_txt).mkdir(parents=True, exist_ok=True)
        # Create column dict for excel file
        column_dict = [{'header': str(col)} for col in args['columns_list']]

        # Create a Pandas Excel writer
        writer = pd.ExcelWriter(
            path_to_txt
            + f'/Job ID - {df_sentence["Job ID"].iloc[0]}_Coder_Name - Codebook (Automating Equity).xlsx',
        )

        # Check datatype
        if isinstance(df_sentence, pd.DataFrame):
            df_sentence, workbook, worksheet = write_sentences_to_excel_helper(
                1, writer, df_sentence, args
            )
        elif isinstance(df_sentence, list):
            df_list = df_sentence
            for i, df in enumerate(df_list, 1):
                if isinstance(df, pd.DataFrame):
                    df_sentence = df
                    df_sentence, workbook, worksheet = write_sentences_to_excel_helper(
                        i, writer, df_sentence, args
                    )

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        try:
            workbook.close()
        except Exception:
            pass


# %%
def write_sentences_to_excel_helper(i, writer, df_sentence, args=get_args()):
    try:
        # Convert the dataframe to anExcel object.
        df_sentence.to_excel(writer, sheet_name=f'Sheet {i}')

        # Get the objects from the dataframe writer object.
        workbook = writer.book
        worksheet = writer.sheets[f'Sheet {i}']

        # Add a format for the header cells.
        header_format = workbook.add_format(args['format_props'])

        # Add a format for columns
        first_row = 1  # Excluding header
        first_col = 3  # Excluding index, Job ID and Sentence
        last_row = len(df_sentence)
        last_col = len(args['columns_list'])
        worksheet.data_validation(
            first_row, first_col, last_row, last_col, args['validation_props']
        )

    except Exception as e:
        if args['print_enabled'] is True:
            print(e.json())

    return (df_sentence, workbook, worksheet)


# %%
# Function to save full job description text in txt file
def write_all_to_txt(search_keyword, job_id, age, df_jobs, args=get_args()):

    if isinstance(df_jobs, list):
        df_list = df_jobs
        for df_jobs in df_list:
            if isinstance(df_jobs, pd.DataFrame):
                if (not df_jobs.empty) and all(
                    df_jobs['Language'] == str(args['language'])
                ):
                    try:
                        write_all_to_txt_helper(
                            search_keyword, job_id, age, df_jobs, args
                        )
                    except Exception as e:
                        if args['print_enabled'] is True:
                            print(e.json())
                elif (df_jobs.empty) or all(
                    df_jobs['Language'] != str(args['language'])
                ):
                    if df_jobs.empty:
                        if args['print_enabled'] is True:
                            print('DF is empty.')
                    elif all(df_jobs['Language'] != str(args['language'])):
                        if args['print_enabled'] is True:
                            print(
                                f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                            )
            elif isinstance(df_jobs, list):
                write_all_to_txt(search_keyword, job_id, age, df_jobs, args)
    #         pbar.finish()
    elif isinstance(df_jobs, pd.DataFrame):
        if (not df_jobs.empty) and all(df_jobs['Language'] == str(args['language'])):
            write_all_to_txt_helper(search_keyword, job_id, age, df_jobs, args)
        elif (df_jobs.empty) or all(df_jobs['Language'] != str(args['language'])):
            if df_jobs.empty:
                if args['print_enabled'] is True:
                    print('DF is empty.')
            elif all(df_jobs['Language'] != str(args['language'])):
                if args['print_enabled'] is True:
                    print(
                        f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                    )


# %%
def write_all_to_txt_helper(search_keyword, job_id, age, df_jobs, args=get_args()):
    path_to_txt = (
        str(args['parent_dir'])
        + f'Jobs TXT/{args["language"]}/{age}/{" ".join(df_jobs["Search Keyword"].iloc[0].split("_"))}'
    )
    pathlib.Path(path_to_txt).mkdir(parents=True, exist_ok=True)

    df_jobs.drop(
        [x for x in args['columns_drop_list'] if x in df_jobs.columns], axis='columns', inplace=True, errors='ignore'
    )
    df_jobs.drop(
        df_jobs.columns[df_jobs.columns.str.contains('Age', case=False)],
        axis='columns',
        inplace=True,
        errors='ignore',
    )

    for index, row in df_jobs.iterrows():
        if row['Language'] == str(args['language']):
            with open(path_to_txt + f'/Job ID - {str(row["Job ID"])}.txt', 'a') as f:
                f.write(row['Job Description'])


# %%
# Function to send batches of excel files to google drive
def send_new_excel_to_gdrive(
    files_to_upload_number=20,
    coders_dict='',
    coders_from_dict=True,
    language='en',
    move_txt_file=True,
    gender_list=['Female', 'Male', 'Mixed Gender'],
    done_job_excel_list=None,
    new_job_excel_list=None,
    new_batch_job_txt_list=None,
    args=get_args(),
):
    dest_path=validate_path(f'{args["content_analysis_dir"]}')

    print('-' * 20)
    print(
        f'NOTE: The function "send_new_excel_to_gdrive" contains the following optional (default) arguments:\n{get_default_args(send_new_excel_to_gdrive)}.'
    )
    print('-' * 20)

    if done_job_excel_list is None:
        done_job_excel_list = []
    if new_job_excel_list is None:
        new_job_excel_list = []
    if new_batch_job_txt_list is None:
        new_batch_job_txt_list = []

    if coders_from_dict is True:
        with open(f'{args["parent_dir"]}coders_dict.json', encoding='utf8') as f:
            coders_dict = json.load(f)
    elif coders_from_dict is False:
        pass
    with open(f'{args["parent_dir"]}batch_counter_dict.json', encoding='utf8') as f:
        batch_counter_dict = json.load(f)

    for coder_number, coder_name in coders_dict.items():
        coder_dest_folder = validate_path(dest_path + f'{coder_name} Folder/')
        if os.path.isdir(coder_dest_folder):
            for coder_dest_folder_path, batch_folder_names, done_job_excel in os.walk(
                coder_dest_folder
            ):
                for batch_number in batch_folder_names:
                    batch_counter_dict[coder_name].extend(
                        int(i)
                        for i in re.findall(r'\d+', batch_number)
                        if int(i) not in batch_counter_dict[coder_name]
                    )
                for done_job_excel_name in done_job_excel:
                    if ('Job ID - ') and ('.xlsx') in done_job_excel_name:
                        if is_non_zero_file(coder_dest_folder_path + '/' + done_job_excel_name) is True:
                            done_job_excel_list.append(
                                validate_path(
                                    coder_dest_folder_path + '/' + done_job_excel_name
                                )
                            )

        try:
            batch_counter_dict[coder_name] = list(set(batch_counter_dict[coder_name]))
            done_job_excel_list = list(set(done_job_excel_list))
        except Exception:
            pass

        excel_source_folder = validate_path(
            f'{args["parent_dir"]}Jobs EXECL/{str(language)}'
        )
        if os.path.isdir(excel_source_folder):
            for (
                gender_occ_source_dir_path,
                all_dir_file_names,
                new_job_excel,
            ) in os.walk(excel_source_folder):
                for new_job_excel_name in new_job_excel:
                    if ('Job ID - ') and ('.xlsx') in new_job_excel_name:
                        if (is_non_zero_file(gender_occ_source_dir_path + '/' + new_job_excel_name) is True
                            and (
                                new_job_excel
                                != any(
                                    done_job_excel
                                    for done_job_excel in done_job_excel_list
                                )
                            )
                            and ('.DS_Store' not in new_job_excel)
                        ):
                            new_job_excel_list.append(
                                validate_path(
                                    gender_occ_source_dir_path
                                    + '/'
                                    + new_job_excel_name
                                )
                            )
        new_job_excel_list = list(set(new_job_excel_list))

        if len(new_job_excel_list) > int(files_to_upload_number):
            new_batch_job_excel_list = random.sample(
                new_job_excel_list, int(files_to_upload_number)
            )
        elif len(new_job_excel_list) <= int(files_to_upload_number):
            new_batch_job_excel_list = new_job_excel_list
            if args['print_enabled'] is True:
                print(
                    f'Less than 12 excel jobs remaining. Moving final {len(new_batch_job_excel_list)} jobs.'
                )

        if move_txt_file is True:
            for new_batch_job_excel in new_batch_job_excel_list:
                new_batch_job_txt_list.append(
                    str(new_batch_job_excel)
                    .replace('Jobs EXECL', 'Jobs TXT')
                    .replace('_Coder_Name - Codebook (Automating Equity).xlsx', '.txt')
                )
            new_batch_job_txt_list = list(set(new_batch_job_txt_list))

        if len(new_batch_job_excel_list) > 0:
            if os.path.isdir(coder_dest_folder):
                for (
                    coder_dest_folder_path,
                    batch_folder_names,
                    done_job_excel,
                ) in os.walk(coder_dest_folder):
                    path_to_next_batch = coder_dest_folder + str(
                        f'{coder_name} Folder - Batch {max(int(i) for v in batch_counter_dict.values() for i in v) + 1}/'
                    )
                    pathlib.Path(path_to_next_batch).mkdir(parents=True, exist_ok=True)
                    for (
                        new_batch_job_excel,
                        new_batch_job_txt,
                    ) in itertools.zip_longest(
                        new_batch_job_excel_list, new_batch_job_txt_list
                    ):
                        try:
                            shutil.move(new_batch_job_excel, path_to_next_batch)
                            if move_txt_file is True:
                                shutil.move(new_batch_job_txt, path_to_next_batch)
                        except Exception:
                            pass
        elif len(new_batch_job_excel_list) <= 0:
            if args['print_enabled'] is True:
                print('No more files to move.')

    for coder_number, coder_name in list(coders_dict.items()):
        for coder_dest_folder_path, batch_folder_names, done_job_excel in os.walk(
            coder_dest_folder
        ):
            for batch_number in batch_folder_names:
                batch_counter_dict[coder_name].extend(
                    int(i)
                    for i in re.findall(r'\d+', batch_number)
                    if i not in batch_counter_dict[coder_name]
                )
        try:
            batch_counter_dict[coder_name] = list(set(batch_counter_dict[coder_name]))
        except Exception:
            pass

    if args['save_enabled'] is True:
        with open(f'{args["parent_dir"]}batch_counter_dict.json', 'w', encoding='utf8') as f:
            json.dump(batch_counter_dict, f)
    elif args['save_enabled'] is False:
        if args['print_enabled'] is True:
            print('No batch counter save enabled.')



# %%
# Function to open and clean dfs
def open_and_clean_excel(
    EXCEL_PATHS=defaultdict(list),
    front_columns=['Coder ID', 'Job ID', 'OG_Sentence ID', 'Sentence ID', 'Sentence'],
    cal_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    int_variable: str = 'Job ID',
    str_variable='Sentence',
    reset=True,
    args=get_args(),
):

    dest_path=validate_path(f'{args["content_analysis_dir"]}')

    for lst in EXCEL_PATHS.values():
        lst = list(set(lst))
    if len(EXCEL_PATHS) < 5:
        if args['print_enabled'] is True:
            print(
                f'NOTE: The function "open_and_clean_excel" contains the following optional (default) arguments:\n{get_default_args(open_and_clean_excel)}'
            )
    with open(f'{args["parent_dir"]}coders_dict.json', encoding='utf8') as f:
        coders_dict = json.load(f)
    for coder_number, coder_name in coders_dict.items():
        coder_dest_folder = validate_path(f'{dest_path}{coder_name} Folder/')
        if os.path.isdir(coder_dest_folder):
            for coder_dest_folder_path, batch_folder_names, done_job_excel in os.walk(
                coder_dest_folder
            ):
                for done_job_excel_name in done_job_excel:
                    if (
                        (len(done_job_excel) != 0)
                        and ('Job ID - ' in done_job_excel_name)
                        and ('.xlsx' in done_job_excel_name)
                        and ('.txt' not in done_job_excel_name)
                        and (is_non_zero_file(coder_dest_folder_path + '/' + done_job_excel_name) is True)
                    ):
                        EXCEL_PATHS[coder_name].append(
                            validate_path(
                                coder_dest_folder_path + '/' + done_job_excel_name
                            )
                        )
                    elif (
                        (len(done_job_excel) == 0)
                        or (
                            ('Job ID - ' not in done_job_excel_name)
                            and ('.xlsx' not in done_job_excel_name)
                            and ('.txt' in done_job_excel_name)
                        )
                        and (
                            is_non_zero_file(coder_dest_folder_path + '/' + done_job_excel_name) is False
                        )
                    ):
                        coders_dict.pop(coder_number, None)
        if args['print_enabled'] is True:
            print(
                f'{len(EXCEL_PATHS[coder_name])} valid excel files found for coder {coder_name}.'
            )

    if len(EXCEL_PATHS) == 0:
        if args['print_enabled'] is True:
            print('No valid excel files found for any coders.')
        (
            coders_list,
            coders_dict,
            coders_numbers,
            df_coder_list,
            df_concat_coder_all,
        ) = assign_all(5, None)
    elif len(EXCEL_PATHS) != 0:
        if args['print_enabled'] is True:
            print('-' * 20)
        coders_list = list(coders_dict.values())
        coders_numbers = list(coders_dict.keys())
        df_coder_list = []
        for index, (coder_key, CODER_EXCEL_PATH) in enumerate(EXCEL_PATHS.items()):
            for path in CODER_EXCEL_PATH:
                if args['print_enabled'] is True:
                    print(path)
                # file_extension = path.lower().split('.')[-1]
                if path.endswith('xlsx'):
                    df_coder = pd.read_excel(
                        validate_path(path), index_col=0, engine='openpyxl'
                    )
                elif path.endswith('xls'):
                    df_coder = pd.read_excel(validate_path(path), index_col=0)
                else:
                    raise Exception('File not supported')

                if df_coder.columns.str.contains('^Unnamed').all():
                    break
                else:
                    df_coder = clean_df(df_coder, str_variable=str_variable, reset=reset)
                    df_coder.drop(
                        df_coder.columns[
                            df_coder.columns.str.contains('Coder Remarks', case=False)
                        ],
                        axis='columns',
                        inplace=True,
                        errors='ignore',
                    )
                    df_coder['Job ID'].fillna(method='ffill', inplace=True)
                    for k, v in coders_dict.items():
                        if v == coder_key:
                            df_coder['Coder ID'] = k
                    df_coder[f'OG_{str_variable} ID'] = df_coder.index + 1
                    df_coder = df_coder.fillna(0)
                    df_coder[str_variable] = df_coder[str_variable].progress_apply(lambda sentence: sentence.strip().lower().replace('[^\w\s]', ''))
                    if df_coder[str_variable].isna().sum() > 0:
                        if args['print_enabled'] is True:
                            print(
                                f'{df_coder[str_variable].isna().sum()} missing sentences found.'
                            )
                    df_coder_list.append(df_coder)

        # pbar.finish()

        if len(df_coder_list) >= 1:
            df_concat_coder_all = pd.concat(df_coder_list)
            df_concat_coder_all[f'{str_variable} ID'] = (
                df_concat_coder_all.groupby([str_variable]).ngroup() + 1
            )
            df_concat_coder_all = df_concat_coder_all[
                front_columns
                + [
                    col
                    for col in df_concat_coder_all.columns
                    if col not in front_columns
                ]
            ]
            df_concat_coder_all.loc[:, cal_columns] = (
                df_concat_coder_all.loc[:, cal_columns]
                .swifter.progress_bar(args['print_enabled'])
                progress_apply(pd.to_numeric, downcast='integer', errors='coerce')
            )
            df_concat_coder_all = clean_df(df_concat_coder_all, str_variable=str_variable, reset=reset)
            df_concat_coder_all.index = range(df_concat_coder_all.shape[0])
            if args['print_enabled'] is True:
                print(f'Total of {len(df_concat_coder_all)} sentences in the dataset.')
            for var in cal_columns:
                if (df_concat_coder_all[str(var)] == 1).sum() + (
                    df_concat_coder_all[str(var)] == 0
                ).sum() == len(df_concat_coder_all):
                    if args['print_enabled'] is True:
                        print(
                            f'Sum of "present" and "not present" {str(var)} labels is equal to length of dataset.'
                        )
                else:
                    if args['print_enabled'] is True:
                        print(
                            f'Sum of "present" and "not present" {str(var)} labels is NOT equal to length of dataset.'
                        )
                    raise ValueError('Problem with candidate trait labels count.')
        elif len(df_coder_list) <= 1:
            df_concat_coder_all = df_coder_list

    if args['print_enabled'] is True:
        print('-' * 20)

    return (
        coders_list,
        coders_dict,
        coders_numbers,
        df_coder_list,
        df_concat_coder_all,
    )

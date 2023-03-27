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
from setup_module.scraping import *
from setup_module.imports import *
from setup_module.classification import *
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

warnings.filterwarnings('ignore', category=DeprecationWarning)

# %%
# Classification
# Set args
args = get_args()
parent_dir = args['parent_dir']
df_dir = args['df_dir']
data_save_path = args['data_save_path']
models_save_path = args['models_save_path']
table_save_path = args['table_save_path']
plot_save_path = args['plot_save_path']
print_enabled = args['print_enabled']
plots_enabled = args['plots_enabled']
save_enabled = args['save_enabled']
age_limit = args['age_limit']
age_ratio = args['age_ratio']
gender_ratio = args['gender_ratio']
file_save_format = args['file_save_format']
file_save_format_backup = args['file_save_format_backup']
image_save_format = args['image_save_format']

text_col = 'Job Description spacy_sentencized'
n_gram = '3grams_gensim'
preprocessing_enabled = True
stemming_enabled = False
lemmatization_enabled = True
numbers_cleaned = True

analysis_df_from_manual = True
task_enabled = False
# language_model_enabled = False

# CHANGE THIS TO GET NEW DATA FROM JSON FILES
start_new_preprocessing = False
###
id_dict_new = start_new_preprocessing
from_processing_function = start_new_preprocessing
classification_final_from_post_cleanup = start_new_preprocessing
classification_final_from_df_jobs_list = start_new_preprocessing
###
stop_words = set(sw.words('english'))
pattern = {'True': r'[^a-zA-Z]', 'False': r'[^a-zA-Z0-9]'}
main_from_function = True
preprocessed_from_function = True
ngrams_from_funtion = True
embedding_from_function = True
ngrams_enabled = True
ngrams_from_funtion = True
drop_cols_enabled = False
embedding_libraries_list = ['nltk', 'gensim']
nltk_ngrams_dict = {2: nltk.bigrams, 3: nltk.trigrams}
ngrams_list = [1, 2, 3, 123]
embedding_models_dict = {
    'w2v': [build_train_word2vec, word2vec_embeddings, Word2Vec],
    'ft': [build_train_fasttext, fasttext_embeddings, FastText],
}

use_dict_for_classifiers_vectorizers = True
select_best_enabled = True
optimization_enabled = False
gs = gridspec.GridSpec(2, 2)

# Validation split ratios
n_jobs = 1
train_ratio = 0.75
test_ratio = 0.10
validation_ratio = 0.15
test_split = test_size = 1 - train_ratio
validation_split = test_ratio / (test_ratio + validation_ratio)

# Cross-validation
random_state = 42
random.seed(random_state)
np.random.seed(random_state)
partition = True
cv = RepeatedStratifiedKFold(
    n_splits=10, n_repeats=3, random_state=random_state)
# cv = StratifiedShuffleSplit(n_splits=10, train_size=train_ratio, test_size=test_ratio, random_state=random_state)
# cv = KFold(n_splits=5, random_state=random_state, shuffle=True)
# cv = LeaveOneOut()

# Resampling
class_weight = 'balanced'
resampling_enabled = True
resample_enn = SMOTEENN(
    enn=EditedNearestNeighbours(sampling_strategy='majority'))
resample_tome = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# Undersampling
rus = RandomUnderSampler(random_state=random_state, replacement=True)
tl = RandomOverSampler(sampling_strategy='majority')
nm = NearMiss()
# Oversampling
ros = RandomOverSampler(random_state=random_state)
smote = SMOTE()
# Sampling Used
resampling_method = resample_tome

t = time.time()
cores = multiprocessing.cpu_count()
model_sizes = [300, 100]
scoring = 'recall'
scores = ['recall', 'accuracy', 'precision', 'f1',
          'roc_auc', 'explained_variance', 'matthews_corrcoef']
search_enabled = False
metrics_list = [
    'Mean Validation Score',
    'Explained Variance',
    'Accuracy',
    'Precision',
    'Recall',
    'F1-score',
    'ROC',
    'AUC',
    'Matthews Correlation Coefficient',
    f'{scoring.title()} Best Threshold',
    f'{scoring.title()} Best Score',
    'Log Loss/Cross Entropy',
    'Classification Report',
    'Confusion Matrix',
    'Accuracy_opt',
    'Precision_opt',
    'Recall_opt',
    'F1-score_opt',
    'Matthews Correlation Coefficient_opt',
    'Classification Report_opt',
    'Confusion Matrix_opt',
]

if task_enabled is False:
    analysis_columns = ['Warmth', 'Competence']
    # df_jobs_labaled_save_path = f'{args["df_dir"]}df_jobs_labeled_final_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned})'
    if search_enabled is False:
        pickle_file_name = 'Classifiers Table.pkl'
        csv_file_name = 'Classifiers Table.csv'
        excel_file_name = 'Classifiers Table.xlsx'
        latex_file_name = 'Classifiers Table.tex'
        markdown_file_name = 'Classifiers Table.md'
    elif search_enabled is True:
        pickle_file_name = 'Classifiers Table_search.pkl'
        csv_file_name = 'Classifiers Table Search.csv'
        excel_file_name = 'Classifiers Table Search.xlsx'
        latex_file_name = 'Classifiers Table Search.tex'
        markdown_file_name = 'Classifiers Table Search.md'

    # if language_model_enabled is True:
    #     csv_file_name = 'Classifiers Table Language Models.csv'
    #     excel_file_name = 'Classifiers Table Language Models.xlsx'

elif task_enabled is True:
    analysis_columns = [
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ]
    # df_jobs_labaled_save_path = f'{df_dir}df_jobs_labeled_final_preprocessed_WITH_TASK_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned})'
    if search_enabled is False:
        csv_file_name = 'Classifiers Table_WITH_TASK.csv'
        excel_file_name = 'Classifiers Table_WITH_TASK.xlsx'
    elif search_enabled is True:
        csv_file_name = 'Classifiers Table Search_WITH_TASK.csv'
        excel_file_name = 'Classifiers Table Search_WITH_TASK.xlsx'

    # if language_model_enabled is True:
    #     csv_file_name = 'Classifiers Table Language Models_WITH_TASK.csv'
    #     excel_file_name = 'Classifiers Table Language Models_WITH_TASK.xlsx'

# if id_dict_new is True:
#     job_id_dict = make_job_id_v_genage_key_dict()
#     sector_vs_job_id_dict = make_job_id_v_sector_key_dict()

# elif id_dict_new is False:
#     with open(validate_path(f'{parent_dir}job_id_vs_all.json'), encoding='utf-8') as f:
#         job_id_dict = json.load(f)

#     with open(validate_path(f'{parent_dir}job_id_vs_sector_all.json'), encoding='utf-8') as f:
#         sector_vs_job_id_dict = json.load(f)

# if from_processing_function is True:

#     df_jobs_labeled = simple_preprocess_df()

# elif from_processing_function is False:
#     df_jobs_labaled_save_path = f'{args["df_dir"]}df_jobs_labeled_final_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned})'

#     try:
#         df_jobs_labeled = pd.read_pickle(
#             validate_path(f'{df_jobs_labaled_save_path}.{file_save_format}'))
#         df_jobs_labeled.to_pickle(validate_path(
#             f'{df_jobs_labaled_save_path}.{file_save_format}'))

#     except Exception:
#         df_jobs_labeled = pd.read_csv(
#             validate_path(f'{df_jobs_labaled_save_path}.{file_save_format_backup}'))
#         df_jobs_labeled.to_csv(validate_path(
#             f'{df_jobs_labaled_save_path}.{file_save_format_backup}'))


# Final Classification
final_warmth_vectorizer = 'TfidfVectorizer'
final_warmth_classifier = 'LogisticRegression'

final_competence_vectorizer = 'UnionBOW'
final_competence_classifier = 'VotingClassifier'


# %%
# Analysis
# Setting Variables
# ivs = ['Gender', 'Age']
# gender_order = ['Female', 'Mixed Gender', 'Male']
# age_order = ['Older Worker', 'Mixed Age', 'Younger']
# ivs_dict = {'Gender': gender_order, 'Age': age_order}

# dvs = {}
# dvs_probas = {}
alpha = 0.050
normality_tests_labels = ['Statistic', 'p-value']

if analysis_df_from_manual is False:
    df_loc = f'_outliers_age_limit-{age_limit}_age_ratio-{age_ratio}_gender_ratio-{gender_ratio}'
    dataframes = {'df': None, 'df_mean': None}
    dv_cols = ['Warmth', 'Warmth_Probability',
               'Competence', 'Competence_Probability']

elif analysis_df_from_manual is True:
    df_loc = '_outliers'
    dataframes = {'df_manual': None, 'df_manual_mean': None}
    dv_cols = ['Warmth', 'Competence']

zscores_list = [0, 1.96, 2.58, 3.29]

outliers_remove = True

# for dv in dv_cols:
#     if outliers_remove is False:
#         zscores_list = [0]

#         for zscores in zscores_list:
#             if 'Probability' not in dv:
#                 dvs[f'{dv}_Zscore{zscores}'] = f'{dv}'
#             elif 'Probability' in dv:
#                 dvs_probas[f'{dv}_Probability_Zscore{zscores}'] = f'{dv}'

#     elif outliers_remove is True:

#         for zscores in zscores_list:
#             if 'Probability' not in dv:
#                 dvs[f'{dv}_Zscore{zscores}'] = f'{dv}_Outliers_Removed_Zscore{zscores}'
#             elif 'Probability' in dv:
#                 dvs_probas[
#                     f'{dv}_Probability_Zscore{zscores}'
#                 ] = f'{dv}_Outliers_Removed_Zscore{zscores}'

for dv in dv_cols:
    if 'Probability' not in dv:
        dvs[f'{dv}'] = f'{dv}'
    if 'Probability' in dv:
        dvs[f'{dv}_Probability'] = f'{dv}'

# %%

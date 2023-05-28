# -*- coding: utf-8 -*-
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Install packages and import

# %%
import os  # type:ignore # isort:skip # fmt:skip # noqa # nopep8
import sys  # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from pathlib import Path  # type:ignore # isort:skip # fmt:skip # noqa # nopep8

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
    code_dir = Path.cwd()

# %load_ext autoreload
# %autoreload 2

# %%
from dotenv.main import load_dotenv  # type:ignore # isort:skip # fmt:skip # noqa # nopep8

envrc_path = Path.cwd().parents[0].joinpath('.envrc')
load_dotenv(dotenv_path=envrc_path)
conda_env_name = os.environ.get('CONDA_ENV_NAME')
conda_env_path = os.environ.get('CONDA_ENV_PATH')
set_conda = False
from_file = True

if set_conda:
    os.system('conda init --all')
    os.system(f'conda activate {conda_env_name}')

    with open(f'{code_dir}/imported_modules.txt', 'r') as f:
        imported_modules = f.readlines()

    if from_file:
        for lib in imported_modules:
            lib = lib.strip()
            if lib != '':
                try:
                    globals()[lib] = __import__(lib)
                except ImportError:
                    print(f'Installing {lib}')
                    try:
                        os.system(
                            f'conda install --name {conda_env_name} --yes {lib}')
                    except Exception:
                        os.system(f'{conda_env_path}/bin/pip install {lib}')

try:
    import argparse
    import ast
    import collections
    import contextlib
    import copy
    import csv
    import datetime
    import functools
    import gc
    import glob
    import importlib
    import inspect
    import itertools
    import json
    import logging
    import logging.handlers
    import math
    import multiprocessing
    import operator
    import pathlib
    import pickle
    import platform
    import pprint
    import random
    import shutil
    import socket
    import string
    import subprocess
    import tempfile
    import time
    import typing
    import unicodedata
    import warnings
    from collections import Counter, defaultdict
    from io import StringIO
    from random import randrange
    from subprocess import call
    from threading import Thread
    from typing import Dict, List, Optional, Set, Tuple

    import cbsodata
    import en_core_web_sm
    import gensim
    import gensim.downloader as gensim_api
    import imblearn
    import IPython
    import IPython.core
    import joblib
    import lxml
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    import matplotlib.image as img
    import matplotlib.pyplot as plt
    import nltk
    import nltk.data
    import numpy as np
    import openpyxl
    import optuna
    import pandas as pd
    import pingouin as pg
    import regex as re
    import requests
    import scipy
    import seaborn as sns
    import selenium.webdriver as webdriver
    import selenium.webdriver.support.ui as ui
    import simpledorff
    import sklearn
    import sklearn as sk
    import spacy
    import statsmodels
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import statsmodels.regression.mixed_linear_model as sm_mlm
    import statsmodels.stats.api as sms
    import textblob
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import tqdm
    import tqdm.auto as tqdm_auto
    import transformers
    import urllib3
    import xgboost as xgb
    import xlsxwriter
    from accelerate import Accelerator, notebook_launcher
    from bs4 import BeautifulSoup
    from dotenv.main import load_dotenv
    from gensim import corpora, models
    from gensim.corpora import Dictionary
    from gensim.models import (
        CoherenceModel,
        FastText,
        KeyedVectors,
        TfidfModel,
        Word2Vec,
    )
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phraser, Phrases
    from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
    from gensim.similarities import (
        SoftCosineSimilarity,
        SparseTermSimilarityMatrix,
        WordEmbeddingSimilarityIndex,
    )
    from gensim.test.utils import common_texts, datapath, get_tmpfile
    from gensim.utils import save_as_line_sentence, simple_preprocess
    from googletrans import Translator

    # from http_request_randomizer.requests.proxy.requestProxy import RequestProxy
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.datasets import make_imbalance
    from imblearn.metrics import (
        classification_report_imbalanced,
        geometric_mean_score,
        make_index_balanced_accuracy,
    )
    from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
    from imblearn.under_sampling import (
        EditedNearestNeighbours,
        NearMiss,
        RandomUnderSampler,
        TomekLinks,
    )
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.display import HTML, Image, Markdown, display
    from ipywidgets import FloatSlider, interactive
    from joblib import parallel_backend
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    from nltk import (
        WordNetLemmatizer,
        agreement,
        bigrams,
        pos_tag,
        regexp_tokenize,
        sent_tokenize,
        trigrams,
        word_tokenize,
        wordpunct_tokenize,
    )
    from nltk.corpus import abc
    from nltk.corpus import stopwords
    from nltk.corpus import stopwords as sw
    from nltk.corpus import wordnet
    from nltk.corpus import wordnet as wn
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer
    from nltk.tokenize import WordPunctTokenizer
    from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype
    from scipy import spatial, stats
    from scipy.special import softmax
    from scipy.stats import (
        anderson,
        chi2_contingency,
        f_oneway,
        levene,
        mannwhitneyu,
        normaltest,
        shapiro,
        stats,
        ttest_ind,
    )
    from selenium.common.exceptions import *
    from selenium.common.exceptions import (
        ElementClickInterceptedException,
        ElementNotVisibleException,
        NoAlertPresentException,
        NoSuchElementException,
        TimeoutException,
    )
    from selenium.webdriver import ActionChains, Chrome
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import Select, WebDriverWait
    from sklearn import feature_selection, linear_model, metrics, set_config, svm, utils
    from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
    from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import (
        load_files,
        load_iris,
        make_classification,
        make_regression,
    )
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        AdaBoostClassifier,
        BaggingClassifier,
        BaggingRegressor,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
        StackingClassifier,
        StackingRegressor,
        VotingClassifier,
        VotingRegressor,
    )
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.feature_extraction.text import (
        CountVectorizer,
        FeatureHasher,
        TfidfVectorizer,
    )
    from sklearn.feature_selection import (
        SelectFdr,
        SelectFpr,
        SelectFromModel,
        SelectFwe,
        SelectKBest,
        SelectPercentile,
        chi2,
        f_classif,
        f_regression,
        mutual_info_classif,
        mutual_info_regression,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import (
        Lasso,
        LassoCV,
        LogisticRegression,
        PassiveAggressiveClassifier,
        Perceptron,
        SGDClassifier,
    )
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        fowlkes_mallows_score,
        log_loss,
        make_scorer,
        matthews_corrcoef,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import (
        GridSearchCV,
        HalvingGridSearchCV,
        HalvingRandomSearchCV,
        KFold,
        LeaveOneOut,
        PredefinedSplit,
        RandomizedSearchCV,
        RepeatedStratifiedKFold,
        ShuffleSplit,
        StratifiedKFold,
        StratifiedShuffleSplit,
        cross_val_score,
        cross_validate,
        learning_curve,
        train_test_split,
        validation_curve,
    )
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
    from sklearn.preprocessing import (
        Binarizer,
        FunctionTransformer,
        LabelBinarizer,
        LabelEncoder,
        MinMaxScaler,
        OneHotEncoder,
        StandardScaler,
        scale,
    )
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils import (
        check_array,
        check_consistent_length,
        check_random_state,
        check_X_y,
    )
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.utils.metaestimators import available_if
    from sklearn.utils.validation import (
        check_is_fitted,
        column_or_1d,
        has_fit_parameter,
    )
    from spacy.matcher import Matcher
    from statsmodels.formula.api import ols
    from statsmodels.graphics.factorplots import interaction_plot
    from statsmodels.iolib.summary2 import summary_col
    from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
    from statsmodels.regression.linear_model import RegressionResults
    from statsmodels.stats.diagnostic import het_white
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from textblob import TextBlob, Word
    from textblob.en.inflect import pluralize, singularize
    from tqdm.contrib.itertools import product as tqdm_product
    from transformers import (
        AdamW,
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        BertConfig,
        BertForPreTraining,
        BertForSequenceClassification,
        BertModel,
        BertTokenizer,
        BertTokenizerFast,
        BitsAndBytesConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        EarlyStoppingCallback,
        GPT2Config,
        GPT2ForSequenceClassification,
        GPT2Model,
        GPT2TokenizerFast,
        GPTJConfig,
        GPTJForSequenceClassification,
        GPTJModel,
        GPTNeoXConfig,
        GPTNeoXForSequenceClassification,
        GPTNeoXModel,
        GPTNeoXTokenizerFast,
        LlamaConfig,
        LlamaForSequenceClassification,
        LlamaModel,
        LlamaTokenizerFast,
        OpenAIGPTConfig,
        OpenAIGPTForSequenceClassification,
        OpenAIGPTTokenizerFast,
        TextClassificationPipeline,
        TFGPTJForSequenceClassification,
        TFGPTJModel,
        TokenClassificationPipeline,
        Trainer,
        TrainingArguments,
        get_linear_schedule_with_warmup,
        pipeline,
    )
    from transformers.integrations import (
        TensorBoardCallback,
        is_optuna_available,
        is_ray_available,
    )
    from webdriver_manager.chrome import ChromeDriverManager
    from xgboost import XGBClassifier

except ImportError as error:
    module_name = str(error).split('named')[1]
    print(f'The library {module_name} is not installed. Installing now.')
    # !conda install --channel apple --yes {module_name}

# from icecream import ic
# import bokeh
# import cardinality
# import libmaths as lm
# import plotly
# import plotly.express as px
# import plotly.graph_objects as go
# import progressbar
# import xorbits.pandas as xpd
# import tensorflow as tf

# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras import layers, models
# from tensorflow.keras import preprocessing
# from tensorflow.keras import preprocessing as kprocessing
# import swifter
# from whatthelang import WhatTheLang
# from xorbits.numpy import arange, argmax, cumsum
# from yellowbrick.text import TSNEVisualizer
# from keras.layers import Activation, Dense
# from keras.models import Sequential

# imported_modules = dir()
# with open(f'{code_dir}imported_modules.txt', 'w') as f:
#     for lib in imported_modules:
#         if '__' not in str(lib) and '_' not in str(lib):
#             f.write(f'{lib}\n')


# %%
# Set paths
# MAIN DIR
main_dir = f'{str(Path(code_dir).parents[0])}/'

# code_dir
code_dir = f'{code_dir}/'
sys.path.append(code_dir)

# scraping dir
scraped_data = f'{code_dir}1. Scraping/'

# data dir
data_dir = f'{code_dir}data/'

# df save sir
df_save_dir = f'{data_dir}final dfs/'

# lang models dir
llm_path = f'{data_dir}Language Models/'

# models dir
models_save_path = f'{data_dir}classification models/'

# output tables dir
table_save_path = f'{data_dir}output tables/'

# plots dir
plot_save_path = f'{data_dir}plots/'

# Make sure path exist and make dir if not
all_dir_list = [
    data_dir, df_save_dir, llm_path, models_save_path, table_save_path, plot_save_path
]
for proj_dir in all_dir_list:
    if not os.path.exists(proj_dir):
        os.mkdir(proj_dir)

# scraped_data sites_dir
site_list = ['Indeed', 'Glassdoor', 'LinkedIn']
for site in site_list:
    if not os.path.exists(f'{scraped_data}{site}'):
        os.mkdir(f'{scraped_data}{site}')

# scraped_data CBS dir
if not os.path.exists(f'{scraped_data}CBS'):
    os.mkdir(f'{scraped_data}CBS')

# %%


# %%
# Set LM settings
# Preprocessing
# NLTK variables
nltk_path = f'{llm_path}nltk'
nltk.data.path.append(nltk_path)
if not os.path.exists(nltk_path):
    os.mkdir(nltk_path)

nltk_libs = [
    'words', 'stopwords', 'punkt', 'averaged_perceptron_tagger',
    'omw-1.4', 'wordnet', 'maxent_ne_chunker', 'vader_lexicon'
]
available_nltk_libs = list(
    set(
        nltk_dir.split('.zip')[0].split('/')[-1]
        for nltk_dir in glob.glob(f'{nltk_path}/*/*')
    )
)

for nltk_lib in list(set(available_nltk_libs) ^ set(nltk_libs)):
    nltk.download(nltk_lib, download_dir=nltk_path)

# nltk.download_shell()

stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
sentim_analyzer = SentimentIntensityAnalyzer()

# Spacy variables
nlp = spacy.load('en_core_web_sm')
# nlp = en_core_web_sm.load()
# nlp = spacy.load('en_core_web_trf')

# Gensim
gensim_path = f'{str(llm_path)}gensim/'
if not os.path.exists(nltk_path):
    os.mkdir(gensim_path)
gensim_api.base_dir = os.path.dirname(gensim_path)
gensim_api.BASE_DIR = os.path.dirname(gensim_path)
gensim_api.GENSIM_DATA_DIR = os.path.dirname(gensim_path)
glove_path = f'{gensim_path}glove/'
fasttext_path = os.path.abspath(f'{gensim_path}fasttext-wiki-news-subwords-300')

# Classification
# Model variables
t = time.time()
n_jobs = -1
n_splits = 10
n_repeats = 3
random_state = 42
refit = True
class_weight = 'balanced'
cv = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
)
scoring = 'recall'
scores = ['recall', 'accuracy', 'f1', 'roc_auc',
          'explained_variance', 'matthews_corrcoef']
scorers = {
    'precision_score': make_scorer(precision_score, zero_division=0),
    'recall_score': make_scorer(recall_score, zero_division=0),
    'accuracy_score': make_scorer(accuracy_score, zero_division=0),
}
protocol = pickle.HIGHEST_PROTOCOL
analysis_columns = ['Warmth', 'Competence']
text_col = 'Job Description spacy_sentencized'
classified_columns = ['Warmth_Probability', 'Competence_Probability']
metrics_dict = {
    f'{scoring.title()} Best Score': np.nan,
    f'{scoring.title()} Best Threshold': np.nan,
    'Train - Mean Cross Validation Score': np.nan,
    f'Train - Mean Cross Validation - {scoring.title()}': np.nan,
    f'Train - Mean Explained Variance - {scoring.title()}': np.nan,
    'Test - Mean Cross Validation Score': np.nan,
    f'Test - Mean Cross Validation - {scoring.title()}': np.nan,
    f'Test - Mean Explained Variance - {scoring.title()}': np.nan,
    'Explained Variance': np.nan,
    'Accuracy': np.nan,
    'Balanced Accuracy': np.nan,
    'Precision': np.nan,
    'Average Precision': np.nan,
    'Recall': np.nan,
    'F1-score': np.nan,
    'Matthews Correlation Coefficient': np.nan,
    'Brier score': np.nan,
    'Fowlkes–Mallows Index': np.nan,
    'R2 Score': np.nan,
    'ROC': np.nan,
    'AUC': np.nan,
    'Log Loss/Cross Entropy': np.nan,
    'Cohen’s Kappa': np.nan,
    'Geometric Mean': np.nan,
    'Classification Report': np.nan,
    'Imbalanced Classification Report': np.nan,
    'Confusion Matrix': np.nan,
    'Normalized Confusion Matrix': np.nan,
}

# Set random seed
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
cores = multiprocessing.cpu_count()

# Transformer variables
max_length = 512
returned_tensor = 'pt'
cpu_counts = torch.multiprocessing.cpu_count()
device = torch.device('mps') if torch.has_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available(
) else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device_name = str(device.type)
print(f'Using {device_name.upper()}')
# Set random seed
random_state = 42
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
cores = multiprocessing.cpu_count()
torch.Generator(device_name).manual_seed(random_state)
cores = multiprocessing.cpu_count()
accelerator = Accelerator()
torch.autograd.set_detect_anomaly(True)
os.environ.get('TOKENIZERS_PARALLELISM')
hyperparameter_tuning = True
best_trial_args = [
    'num_train_epochs', 'learning_rate', 'weight_decay', 'warmup_steps',
]
training_args_dict = {
    'seed': random_state,
    'resume_from_checkpoint': False,
    'overwrite_output_dir': True,
    'logging_steps': 500,
    'evaluation_strategy': 'steps',
    'eval_steps': 500,
    'save_strategy': 'steps',
    'save_steps': 500,
    'use_mps_device': bool(device_name == 'mps' and torch.backends.mps.is_available()),
    'metric_for_best_model': 'eval_recall',
    'optim': 'adamw_torch',
    'load_best_model_at_end': True,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 20,
    # The below metrics are used by hyperparameter search
    'num_train_epochs': 3,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 100,
}
training_args_dict_for_best_trial = {
    arg_name: arg_
    for arg_name, arg_ in training_args_dict.items()
    if arg_name not in best_trial_args
}

# Plotting variables
pp = pprint.PrettyPrinter(indent=4)
tqdm.tqdm.pandas(desc='progress-bar')
tqdm_auto.tqdm.pandas(desc='progress-bar')
# # tqdm.notebook.tqdm().pandas(desc='progress-bar')
tqdm_auto.notebook_tqdm().pandas(desc='progress-bar')
# pbar = progressbar.ProgressBar(maxval=10)
font = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 10
}
mpl.style.use(f'{code_dir}/setup_module/apa.mplstyle-main/apa.mplstyle')
mpl.rcParams['text.usetex'] = False
mpl.rc('font', **font)
plt.style.use('tableau-colorblind10')
plt.rc('font', **font)
colorblind_hex_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmap_colorblind = mpl.colors.LinearSegmentedColormap.from_list(name='cmap_colorblind', colors=colorblind_hex_colors)
with contextlib.suppress(ValueError):
    plt.colormaps.register(cmap=cmap_colorblind)

colorblind_hex_colors_blues_and_grays = [
    colorblind_hex_colors[i]
    for i in [9, 2, 6, 7, 4, 0]
]
colorblind_hex_colors_blues_and_grays = sorted(
    colorblind_hex_colors_blues_and_grays * 3,
    key=colorblind_hex_colors_blues_and_grays.index
)

cmap_colorblind_blues_and_grays = mpl.colors.LinearSegmentedColormap.from_list(name='colorblind_hex_colors_blues_and_grays', colors=colorblind_hex_colors_blues_and_grays)
with contextlib.suppress(ValueError):
    plt.colormaps.register(cmap=cmap_colorblind_blues_and_grays)
plt.set_cmap(cmap_colorblind_blues_and_grays)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')

# Display variables
# csv.field_size_limit(sys.maxsize)
# IPython.core.page = print
# IPython.display.clear_output
# display(HTML('<style>.container { width:90% !important; }</style>'))
# InteractiveShell.ast_node_interactivity = 'all'
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# import pretty_errors
# pretty_errors.configure(
#     separator_character = '*',
#     filename_display    = pretty_errors.FILENAME_EXTENDED,
#     line_number_first   = True,
#     display_link        = True,
#     lines_before        = 5,
#     lines_after         = 2,
#     line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
#     code_color          = '  ' + pretty_errors.default_config.line_color,
#     truncate_code       = True,
#     display_locals      = True
# )
# pretty_errors.replace_stderr()
# lux.config.default_display = "lux"
# lux.config.plotting_backend = "matplotlib"

errors = (
    TypeError,
    AttributeError,
    ElementClickInterceptedException,
    ElementNotInteractableException,
    NoSuchElementException,
    NoAlertPresentException,
    TimeoutException,
)

# %%
# Analysis
# Set Variables
nan_list = [
    None, 'None', [], '[]', '"', -1, '-1', 0, '0', 'nan', np.nan, 'Nan', u'\uf0b7', u'\u200b', u'', u' ', u'  ', u'   ', '', ' ', '  ', '   ',
]
non_whitespace_nan_list = nan_list[:nan_list.index('')]

sentence_beginners = 'A|The|This|There|Then|What|Where|When|How|Ability|Support|Provide|Liaise|Contribute|Collaborate|Build|Advise|Detail|Avail|Must|Minimum|Excellent|Fluent'
pattern_1 = r'[\n]+[\s]*|[\,\s]{3,}(?<![A-Z]+)(?=[A-Z])|[\|\s]{3,}(?<![A-Z]+)(?=[A-Z])|[\:]+[\s]*(?<![A-Z]+)(?=[A-Z])|[\;]+[\s]*(?<![A-Z]+)(?=[A-Z])|[\n\r]+[\s]*(?<![A-Z]+)(?=[A-Z])'
pattern_2 = r'(?<=[a-z]\.+|\:+|\;+|\S)(?<![\(|\&]+)(?<![A-Z]+)(?=[A-Z])'
pattern_3 = rf'\s+(?={sentence_beginners})\s*'
pattern = re.compile(f'{pattern_1} | {pattern_2} | {pattern_3}', re.VERBOSE)

dutch_requirement_pattern = r'[Dd]utch [Pp]referred | [Dd]utch [Re]quired | [Dd]utch [Ll]anguage |[Pp]roficient in [Dd]utch |[Ss]peak [Dd]utch | [Kk]now [Dd]utch | [Ff]luent in [Dd]utch | [Dd]utch [Nn]ative | * [Dd]utch [Ll]evel | [Dd]utch [Ss]peaking | [Dd]utch [Ss]peaker | [iI]deally [Dd]utch'
english_requirement_pattern = r'[Ee]nglish [Pp]referred | [Ee]nglish [Re]quired | [Ee]nglish [Ll]anguage |[Pp]roficient in [Ee]nglish |[Ss]peak [Ee]nglish | [Kk]now [Ee]nglish | [Ff]luent in [Ee]nglish | [Ee]nglish [Nn]ative | * [Ee]nglish [Ll]evel | [Ee]nglish [Ss]peaking | [Ee]nglish [Ss]peaker | [iI]deally [Ee]nglish'

alpha = np.float64(0.050)
normality_tests_labels = ['Statistic', 'p-value']
ngrams_list=[1, 2, 3, 123]
embedding_libraries_list = ['spacy', 'nltk', 'gensim']
dvs = [
    'Warmth', 'Competence',
]
dvs_prob = [
    'Warmth_Probability', 'Competence_Probability',
]
dvs_all = [
    'Warmth', 'Competence', 'Warmth_Probability', 'Competence_Probability',
]
ivs = ['Gender', 'Age']
ivs_all = [
    'Gender',
    'Gender_Num',
    'Gender_Female',
    'Gender_Mixed',
    'Gender_Male',
    'Gender_Female_n',
    'Gender_Male_n',
    'Gender_Female_% per Sector',
    'Gender_Male_% per Sector',
    'Age',
    'Age_Num',
    'Age_Older',
    'Age_Mixed',
    'Age_Younger',
    'Age_Older_n',
    'Age_Younger_n',
    'Age_Older_% per Sector',
    'Age_Younger_% per Sector',
]
ivs_cat_and_perc = [
    'Gender',
    'Age',
    'Gender_Female_% per Sector',
    'Gender_Male_% per Sector',
    'Age_Older_% per Sector',
    'Age_Younger_% per Sector',
]
ivs_dummy_and_perc = [
    'Gender_Female',
    'Gender_Mixed',
    'Gender_Male',
    'Gender_Female_% per Sector',
    'Gender_Male_% per Sector',
    'Age_Older',
    'Age_Mixed',
    'Age_Younger',
    'Age_Older_% per Sector',
    'Age_Younger_% per Sector',
]
ivs_gender_dummy_and_perc = [
    'Gender_Female',
    'Gender_Mixed',
    'Gender_Male',
    'Gender_Female_% per Sector',
    'Gender_Male_% per Sector',
]
ivs_age_dummy_and_perc = [
    'Age_Older',
    'Age_Mixed',
    'Age_Younger',
    'Age_Older_% per Sector',
    'Age_Younger_% per Sector',
]
ivs_count_and_perc = [
    'Gender_Female_n',
    'Gender_Male_n',
    'Gender_Female_% per Sector',
    'Gender_Male_% per Sector',
    'Age_Older_n',
    'Age_Younger_n',
    'Age_Older_% per Sector',
    'Age_Younger_% per Sector',
]
ivs_count = [
    'Gender_Female_n',
    'Gender_Male_n',
    'Age_Older_n',
    'Age_Younger_n',
]
ivs_gender_count = [
    'Gender_Female_n',
    'Gender_Male_n',
]
ivs_age_count = [
    'Age_Older_n',
    'Age_Younger_n',
]
ivs_perc = [
    'Gender_Female_% per Sector',
    'Gender_Male_% per Sector',
    'Age_Older_% per Sector',
    'Age_Younger_% per Sector',
]
ivs_gender_perc = [
    'Gender_Female_% per Sector',
    'Gender_Male_% per Sector',
]
ivs_age_perc = [
    'Age_Older_% per Sector',
    'Age_Younger_% per Sector',
]
ivs_dummy = [
    'Gender_Female',
    'Gender_Mixed',
    'Gender_Male',
    'Age_Older',
    'Age_Mixed',
    'Age_Younger',
]
ivs_num = [
    'Gender_Num',
    'Age_Num',
]

ivs_dummy_num = [
    'Gender_Num',
    'Gender_Female',
    'Gender_Mixed',
    'Gender_Male',
    'Age_Num',
    'Age_Older',
    'Age_Mixed',
    'Age_Younger',
]
ivs_num_and_perc = [
    'Gender_Num',
    'Gender_Female_% per Sector',
    'Gender_Male_% per Sector',
    'Age_Num',
    'Age_Older_% per Sector',
    'Age_Younger_% per Sector',
]
ivs_gender_dummy_num = [
    'Gender_Num',
    'Gender_Female',
    'Gender_Mixed',
    'Gender_Male',
]
ivs_gender_dummy = [
    'Gender_Female',
    'Gender_Mixed',
    'Gender_Male',
]
ivs_age_dummy_num = [
    'Age_Num',
    'Age_Older',
    'Age_Mixed',
    'Age_Younger',
]
ivs_age_dummy = [
    'Age_Older',
    'Age_Mixed',
    'Age_Younger',
]
gender_order = ['Female', 'Mixed Gender', 'Male']
age_order = ['Older', 'Mixed Age', 'Younger']
platform_order = ['LinkedIn', 'Indeed', 'Glassdoor']
ivs_dict = {'Gender': gender_order, 'Age': age_order}
# Models dict
sm_models = {
    'Logistic': sm.Logit,
    'OLS': sm.OLS,
}
# DVs dict
dvs_for_analysis = {
    'binary': ['Categorical Warmth and Competence', dvs],
    'probability': ['Probability Warmth and Competence', dvs_prob],
    'binary and probability': ['Categorical and Probability Warmth and Competence', dvs_all],
}
# IVs dict
ivs_for_analysis = {
    'categories': ['Categorical Gender and Age', ivs_dummy],
    'percentages': ['PPS Gender and Age', ivs_perc],
    'categories and percentages': ['Categorical and PPS Gender and Age', ivs_dummy_and_perc],
}
cat_list = [
    'Job ID',
    'Gender',
    'Gender_Female',
    'Gender_Male',
    'Gender_Mixed',
    'Age',
    'Age_Older',
    'Age_Younger',
    'Gender_Mixed',
    'Language',
    'English Requirement in Sentence',
    'Dutch Requirement in Sentence'
]
controls = [
        '% Sector per Workforce',
        'Job Description num_words',
        'English Requirement in Job Ad_Yes', 'Dutch Requirement in Job Ad_Yes',
        # Main controls = [:4], Extra controls = [4:]
        # 'Platform_Indeed', 'Platform_Glassdoor',
        # Main controls = [:6], Extra controls = [6:]
        # 'Platform_LinkedIn',
        # 'English Requirement in Job Ad', 'Dutch Requirement in Job Ad',
        # 'Platform',
        # 'Job Description num_unique_words',
        # 'Job Description num_chars',
        # 'Job Description num_chars_no_whitespact_and_punt',
        # 'Industry', 'Sector_n',
]

# n_grams_counts = []
# for embedding_library, ngram_num in tqdm_product(embedding_libraries_list, ngrams_list):
#     controls.extend(
#         [
#             f'Job Description {embedding_library}_{ngram_num}grams_count',
#             f'Job Description {embedding_library}_{ngram_num}grams_abs_word_freq',
#             f'Job Description {embedding_library}_{ngram_num}grams_abs_word_perc',
#             f'Job Description {embedding_library}_{ngram_num}grams_abs_word_perc_cum'
#         ]
#     )

# %%
# Commonly used functions
def get_df_info(df, ivs_all=None):
    if ivs_all is None:
        ivs_all = [
            'Gender',
            'Gender_Num',
            'Gender_Female',
            'Gender_Mixed',
            'Gender_Male',
            'Gender_Female_n',
            'Gender_Male_n',
            'Gender_Female_% per Sector',
            'Gender_Male_% per Sector',
            'Age',
            'Age_Num',
            'Age_Older',
            'Age_Mixed',
            'Age_Younger',
            'Age_Older_n',
            'Age_Younger_n',
            'Age_Older_% per Sector',
            'Age_Younger_% per Sector',
        ]
    # Print Info
    print('\nDF INFO:\n')
    df.info()

    for iv in ivs_all:
        try:
            print('='*20)
            print(f'{iv}:')
            print('-'*20)
            if len(df[iv].value_counts()) < 5:
                print(f'{iv} Counts:\n{df[iv].value_counts()}')
                print('-'*20)
                print(f'{iv} Percentages:\n{df[iv].value_counts(normalize=True).mul(100).round(1).astype(float)}')
                print('-'*20)
            print(f'Min {iv} value: {df[iv].min().round(3).astype(float)}')
            print(f'Max {iv} value: {df[iv].max().round(3).astype(float)}')
            with contextlib.suppress(Exception):
                print('-'*20)
                print(f'{iv} Mean: {df[iv].mean().round(2).astype(float)}')
                print('-'*20)
                print(f'{iv} Standard Deviation: {df[iv].std().round(2).astype(float)}')
        except Exception:
            print(f'{iv} not available.')

    print('\n')


# Function to order categories
def categorize_df_gender_age(df, gender_order=None, age_order=None, ivs=None):
    if gender_order is None:
        gender_order = ['Female', 'Mixed Gender', 'Male']
    if age_order is None:
        age_order = ['Older', 'Mixed Age', 'Younger']
    if ivs is None:
        ivs = ['Gender', 'Age']
    # Arrange Categories
    for iv in ivs:
        if iv == 'Gender':
            order = gender_order
        elif iv == 'Age':
            order = age_order
        with contextlib.suppress(ValueError):
            df[iv] = df[iv].astype('category').cat.reorder_categories(order, ordered=True)

            df[iv] = pd.Categorical(
                df[iv], categories=order, ordered=True
            )
            df[f'{iv}_Num'] = pd.to_numeric(df[iv].cat.codes).astype('int64')

    return df


# %%
def get_word_num_and_frequency(row, text_col):

    with open(f'{data_dir}punctuations.txt', 'rb') as f:
        custom_punct_chars = pickle.load(f)
    row['Job Description num_words'] = len(str(row[text_col]).split())
    row['Job Description num_unique_words'] = len(set(str(row[text_col]).split()))
    row['Job Description num_chars'] = len(str(row[text_col]))
    row['Job Description num_chars_no_whitespact_and_punt'] = len(
        [
            c
            for c in str(row[text_col])
            if c not in custom_punct_chars and c not in list(string.punctuation) and c in list(string.printable) and c not in list(string.whitespace) and c != ' '
        ]
    )
    row['Job Description num_punctuations'] = len(
        [
            c
            for c in str(row[text_col])
            if c in custom_punct_chars and c in list(string.punctuation) and c in list(string.printable) and c not in list(string.whitespace) and c != ' '
        ]
    )

    return row



# %%
# Fix Keywords
keyword_trans_dict = {
    'landbouw': 'agriculture',
    'manage drivers': 'transportation',
    'renting and other business support': 'business support',
    'other business support': 'business support',
    'mijnbouw': 'mining',
    'bosbouw': 'forestry',
    'gas for': 'gas',
    'gas vooraad': 'gas',
    'productie': 'production',
    'sociologen': 'sociologist',
    'leraren van basisschool': 'primary school teacher',
    'ere leraren': 'honorary teacher',
    'other teacher': 'teacher',
    'andere leraren': 'teacher',
    'buyinging': 'buying',
    'accommodatie': 'accommodation',
    'vissen': 'fishing',
    'grooth': 'great',
    'opleiding': 'education',
    'ingenieur': 'engineer',
    'engineers': 'engineer',
    'communicatie': 'communication',
    'auteur': 'author',
    'auteurs': 'author',
    'authors': 'author',
    'publieke administratie': 'public administration',
    'verkoop onroerend goed': 'selling real estate',
    'educational': 'education',
    'marketingmanager': 'marketing manager',
    'marketingmanagers': 'marketing manager',
    'food servin': 'food serving',
    'voedsel dienen': 'food serving',
    'etensservin': 'food serving',
    'sales': 'sale',
    'verkoop': 'sale',
    'sold': 'sale',
    'sell': 'sale',
    'uitverkoop': 'sale',
    'pedagoog': 'educationalist',
    'educationalists': 'educationalist',
    'educatie': 'education',
    'educator': 'education',
    'psycholoog': 'psychologist',
    'psychologists': 'psychologist',
    'logistieke manager': 'logistics manager',
    'logistieke managers': 'logistics manager',
    'logistic': 'logistics',
    'koop': 'buying',
    'buy': 'buying',
    'ere serviceactiviteiten': 'honorary service activity',
    'serviceactiviteiten': 'service activity',
    'directeur': 'director',
    'informatie': 'information',
    'serve accommodation': 'accommodation',
    'psychologen': 'psychologist',
    'linguïsten': 'linguist',
    'linguïst': 'linguist',
    'linguïst': 'linguist',
    'sales of real estate': 'selling real estate',
    'socioloog': 'sociologist',
    'opslag': 'storage',
    'educatief': 'education',
    'elektriciteit': 'electricity',
    'elektrotechnische ingenieur': 'electrical engineer',
    'elektrotechnische ingenieurs': 'electrical engineer',
    'ingenieurs': 'engineer',
    'ingenieur': 'engineer',
    'toepassings ontwikkelaar': 'application developer',
    'toepassings ontwikkelaars': 'application developer',
    'application developers': 'application developer',
    'water voorraad': 'water supply',
    'fysiotherapeuten': 'physiotherapist',
    'cultuur': 'culture',
    'career developmentsspecialist': 'career development specialist',
    'carrière ontwikkelingspecialisten': 'career development specialist',
    'carrière ontwikkelingspecialist': 'career development specialist',
    'ict-manager': 'ict manager',
    'ict-managers': 'ict manager',
    'ict managers': 'ict manager',
    'manager care institution': 'manager of healthcare institution',
    'managers care institution': 'manager of healthcare institution',
    'manager healthcare institution': 'manager of healthcare institution',
    'managers healthcare institution': 'manager of healthcare institution',
    'manager of care institution': 'manager of healthcare institution',
    'managers of care institution': 'manager of healthcare institution',
    'manager healthcare institution': 'manager of healthcare institution',
    'managers healthcare institution': 'manager of healthcare institution',
    'managers of healthcare institution': 'manager of healthcare institution',
    'manager care institutions': 'manager of healthcare institution',
    'managers care institutions': 'manager of healthcare institution',
    'manager healthcare institutions': 'manager of healthcare institution',
    'managers healthcare institutions': 'manager of healthcare institution',
    'manager of care institutions': 'manager of healthcare institution',
    'managers of care institutions': 'manager of healthcare institution',
    'manager healthcare institutions': 'manager of healthcare institution',
    'managers healthcare institutions': 'manager of healthcare institution',
    'managers of healthcare institutions': 'manager of healthcare institution',
    'forestrymanager of healthcare institution': 'manager of healthcare institution',
    'gezondheid en maatschappelijk werkactiviteit': 'healthcare',
    'doctors': 'doctor',
    'dokter': 'doctor',
    'dokters': 'doctor',
    'sociale werkzaamheden': 'social work',
    'sociaal werker': 'social work',
    'social work activities': 'social work activity',
    'sports': 'sport',
    'groothandel': 'wholesale',
    'wholesale and retail': 'wholesale',
    'andere serviceactiviteiten': 'other service activity',
    'specialized services manager': 'specialised services manager',
    'specialized business service': 'specialised business service',
    'specialized nurse': 'specialised nurse',
    'recreatie': 'recreation',
    'netwerk specialisten': 'network specialist',
    'netwerkspecialisten': 'network specialist',
    'adverse': 'staff',
    'bulletin': 'staff',
    'other service activity': 'staff',
    'afvalbeheer': 'waste management'}

# with open(f'{code_dir}/1. Scraping/CBS/Data/keyword_trans_dict.txt', 'w') as f:
#     json.dump(keyword_trans_dict, f)

# %%

# with open(f'{code_dir}/1. Scraping/CBS/Data/keyword_trans_dict.txt', 'w') as f:
#     json.dump(keyword_trans_dict, f)

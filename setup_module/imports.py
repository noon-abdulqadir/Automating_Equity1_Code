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
    code_dir = str(Path.cwd())
sys.path.append(code_dir)

# %load_ext autoreload
# %autoreload 2

# %%
from dotenv.main import load_dotenv  # type:ignore # isort:skip # fmt:skip # noqa # nopep8

env_path = Path(code_dir).joinpath('.envrc')
load_dotenv(dotenv_path=env_path)
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
    import urllib
    import warnings
    from collections import Counter, defaultdict
    from io import StringIO
    from random import randrange
    from subprocess import call
    from threading import Thread
    from typing import Dict, List, Optional, Set, Tuple

    import cbsodata
    import en_core_web_sm
    import evaluate
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
    import shap
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
    import urllib3
    import xgboost as xgb
    import xlsxwriter
    import yaml
    from bs4 import BeautifulSoup
    from cluestar import plot_text
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
    from html2image import Html2Image
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
    from IPython.core.display import HTML
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.display import HTML, Image, Markdown, display
    from ipywidgets import FloatSlider, fixed, interact, interact_manual, interactive
    from joblib import parallel_backend
    from matplotlib.animation import FuncAnimation
    from matplotlib_inline.backend_inline import set_matplotlib_formats
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
    from selenium import webdriver
    from selenium.common.exceptions import *
    from selenium.common.exceptions import (
        ElementClickInterceptedException,
        ElementNotVisibleException,
        NoAlertPresentException,
        NoSuchElementException,
        TimeoutException,
        WebDriverException,
    )
    from selenium.webdriver import ActionChains, Chrome, ChromiumEdge
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chromium.options import ChromiumOptions
    from selenium.webdriver.chromium.service import ChromiumService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import Select, WebDriverWait
    from skimpy import skim
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
    from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        AdaBoostClassifier,
        BaggingClassifier,
        BaggingRegressor,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
        RandomForestRegressor,
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
        mean_absolute_error,
        mean_squared_error,
        precision_recall_curve,
        precision_score,
        r2_score,
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
    from statsmodels.sandbox.regression.gmm import IV2SLS
    from statsmodels.stats.diagnostic import het_white
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from summarytools import dfSummary, tabset
    from textblob import TextBlob, Word
    from textblob.en.inflect import pluralize, singularize
    from torcheval.metrics.functional.classification import (
        binary_accuracy,
        binary_precision,
        binary_recall,
    )
    from tqdm.contrib.itertools import product as tqdm_product
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
scores = ['recall', 'accuracy', 'f1', 'roc_auc', 'explained_variance', 'matthews_corrcoef']
scorers = {
    'precision_score': make_scorer(precision_score, zero_division=0),
    'recall_score': make_scorer(recall_score, zero_division=0),
    'accuracy_score': make_scorer(accuracy_score, zero_division=0),
}
protocol = pickle.HIGHEST_PROTOCOL
text_col = 'Job Description spacy_sentencized'
analysis_columns = ['Warmth', 'Competence']
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
    'Brier Score': np.nan,
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
# accelerator = Accelerator()
torch.autograd.set_detect_anomaly(True)
os.environ.get('TOKENIZERS_PARALLELISM')
os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
os.environ.get('TRANSFORMERS_CACHE')
openai_token = os.environ['OPENAI_API_KEY']
huggingface_token = os.environ['HUGGINGFACE_API_KEY']
skip_fitted_estimators = True
evaluate_estimator_on_concat = False
hyperparameter_tuning = False
shap.initjs()
best_trial_args = ['num_train_epochs', 'learning_rate', 'weight_decay', 'warmup_steps',]
training_args_dict = {
    'seed': random_state,
    'resume_from_checkpoint': True,
    'overwrite_output_dir': True,
    'logging_steps': 500,
    'evaluation_strategy': 'steps',
    'eval_steps': 500,
    'save_strategy': 'steps',
    'save_steps': 500,
    'use_mps_device': bool(device_name == 'mps' and torch.backends.mps.is_available()),
    'metric_for_best_model': 'Recall',
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
set_matplotlib_formats('png')
plt.style.use('tableau-colorblind10')
plt.rc('font', **font)
plt.rcParams['font.family'] = font['family']
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
pd.set_option('display.float_format', '{:.3f}'.format)
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

example_sentences = {
    'warmth_example_sentence': 'As a senior member of the team, fostering collaboration and encouraging best practices in ways of working and knowledge sharing.',
    'competence_example_sentence': 'The IT security team works closely together with the Risk Management department on the topics Information Security and Privacy.',
    'both_warmth_and_competence_example_sentence': 'Acquiring deep knowledge of IQVIA data sources, acting as an advisor to other members of the consulting team',
    'neither_warmth_nor_competence_example_sentence': 'The role is open for candidates based in remote locations in the Region Europe.'
}

alpha = np.float64(0.050)
normality_tests_labels = ['Statistic', 'p-value']
ngrams_list=[1, 2, 3, 123]
embedding_libraries_list = ['spacy', 'nltk', 'gensim']

with open(f'{data_dir}warmth_competence_words.json', 'r') as f:
    warmth_competence_words = json.load(f)

dvs = [
    'Warmth', 'Competence',
]
dvs_predicted = [f'{dv}_predicted' for dv in dvs]
dvs_prob = [
    'Warmth_Probability', 'Competence_Probability',
]
dvs_prob_predicted = [f'{dv}_predicted' for dv in dvs_prob]
dvs_all = [
    'Warmth', 'Competence', 'Warmth_Probability', 'Competence_Probability',
]
dvs_all_predicted = [f'{dv}_predicted' for dv in dvs_all]
dvs_combined = dvs_all + dvs_predicted
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
    'Interaction_Female_Older_% per Sector',
    'Interaction_Female_Younger_% per Sector',
    'Interaction_Male_Older_% per Sector',
    'Interaction_Male_Younger_% per Sector',
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
ivs_dummy_perc_and_perc_interactions = [
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
    'Interaction_Female_Older_% per Sector',
    'Interaction_Female_Younger_% per Sector',
    'Interaction_Male_Older_% per Sector',
    'Interaction_Male_Younger_% per Sector',
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
ivs_perc_interactions = [
    'Interaction_Female_Older_% per Sector',
    'Interaction_Female_Younger_% per Sector',
    'Interaction_Male_Older_% per Sector',
    'Interaction_Male_Younger_% per Sector',
]
ivs_perc_and_perc_interactions = [
    'Gender_Female_% per Sector',
    'Gender_Male_% per Sector',
    'Age_Older_% per Sector',
    'Age_Younger_% per Sector',
    'Interaction_Female_Older_% per Sector',
    'Interaction_Female_Younger_% per Sector',
    'Interaction_Male_Older_% per Sector',
    'Interaction_Male_Younger_% per Sector',
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
ivs_dict: dict[str, list[str]] = {'Gender': gender_order, 'Age': age_order}
# # Models dict
# sm_models = {
#     'Logistic': sm.Logit,
#     'OLS': sm.OLS,
# }
# # DVs dict for analysis
# dvs_for_analysis = {
#     'probability': ['Probability Warmth and Competence', dvs_prob],
#     'binary': ['Categorical Warmth and Competence', dvs],
#     'binary and probability': ['Categorical and Probability Warmth and Competence', dvs_all],
# }

# # Make extra IV dicts
# ivs_dummy_for_analysis = [iv for iv in ivs_dummy if 'Mixed' not in iv]
# ivs_dummy_and_perc_for_analysis = [iv for iv in ivs_dummy_and_perc if 'Mixed' not in iv]
# ivs_dummy_perc_and_perc_interactions_for_analysis = [iv for iv in ivs_dummy_perc_and_perc_interactions if 'Mixed' not in iv]

# # IVs dict for analysis
# ivs_for_analysis = {
#     'categories, percentages, and interactions': [
#         'Categorical, PPS, and PPS Interactions Gender and Age',
#         ivs_dummy_perc_and_perc_interactions_for_analysis
#     ],
#     'categories and percentages': [
#         'Categorical and PPS Gender and Age',
#         ivs_dummy_and_perc_for_analysis
#     ],
#     'percentages and interactions': [
#         'PPS and PPS Interactions',
#         ivs_perc_and_perc_interactions
#     ],
#     'categories': [
#         'Categorical Gender and Age',
#         ivs_dummy_for_analysis
#     ],
#     'percentages': [
#         'PPS Gender and Age',
#         ivs_perc
#     ],
#     'interactions': [
#         'PPS Interactions',
#         ivs_perc_interactions
#     ],
# }
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
        'Job Description spacy_sentencized_num_words', '% Sector per Workforce',
        'Sector Job Advertisement Count', 'Keywords Count',
        'English Requirement in Job Ad_Yes', 'Dutch Requirement in Job Ad_Yes',
        # Main controls = [:4], Extra controls = [4:]
        # 'Platform_Indeed', 'Platform_Glassdoor',
        # Main controls = [:6], Extra controls = [6:]
        # 'Platform_LinkedIn',
        # 'English Requirement in Job Ad', 'Dutch Requirement in Job Ad',
        # 'Platform',
        # 'Job Description_num_unique_words',
        # 'Job Description_num_chars',
        # 'Job Description_num_chars_no_whitespact_and_punt',
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
def show_and_close_plots(plt):
    plt.rc('font', **font)
    plt.rcParams['font.family'] = font['family']
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

# %%
def close_plots(plt):
    plt.rc('font', **font)
    plt.rcParams['font.family'] = font['family']
    plt.clf()
    plt.cla()
    plt.close()

# %%
def shorten_file_path(file_path: str, max_length: int = 260) -> str:

    if len(file_path) > max_length:
        parent_dir, file_name = os.path.split(file_path)
        while len(parent_dir) + len(file_name) > max_length:
            file_name = file_name[:-1]
        return os.path.join(parent_dir, file_name)
    return file_path

# %%
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
            'Interaction_Female_Older_% per Sector',
            'Interaction_Female_Younger_% per Sector',
            'Interaction_Male_Older_% per Sector',
            'Interaction_Male_Younger_% per Sector',
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
                print(f'{iv} Mean: {df[iv].mean().round(3).astype(float)}')
                print('-'*20)
                print(f'{iv} Standard Deviation: {df[iv].std().round(3).astype(float)}')
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
    row[f'{text_col}_num_words'] = len(str(row[text_col]).split())
    row[f'{text_col}_num_unique_words'] = len(set(str(row[text_col]).split()))
    row[f'{text_col}_num_chars'] = len(str(row[text_col]))
    row[f'{text_col}_num_chars_no_whitespact_and_punt'] = len(
        [
            c
            for c in str(row[text_col])
            if c not in custom_punct_chars and c not in list(string.punctuation) and c in list(string.printable) and c not in list(string.whitespace) and c != ' '
        ]
    )
    row[f'{text_col}_num_punctuations'] = len(
        [
            c
            for c in str(row[text_col])
            if c in custom_punct_chars and c in list(string.punctuation) and c in list(string.printable) and c not in list(string.whitespace) and c != ' '
        ]
    )

    return row

# %%
# Function to write full regressions reports to excel
def save_df_full_summary_excel(
    df_full_summary,
    title,
    text_to_add_list,
    file_save_path,
    sheet_name=None,
    startrow=None,
    startcol=None,
):
    if sheet_name is None:
        sheet_name = 'All'
    if startrow is None:
        startrow = 1
    if startcol is None:
        startcol = 1

    # Define last rows and cols locs
    header_range = 1
    endrow = startrow + header_range + df_full_summary.shape[0]
    endcol = startcol + df_full_summary.shape[1]

    # Remove NAs
    df_full_summary = df_full_summary.fillna('')

    # Write
    writer = pd.ExcelWriter(f'{file_save_path}.xlsx')
    df_full_summary.to_excel(writer, sheet_name=sheet_name, merge_cells=True, startrow=startrow, startcol=startcol)
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(startrow, 1, None, None, {'hidden': True}) # hide the index column

    # Title
    worksheet.merge_range(1, startcol, 1, endcol, title, workbook.add_format({'italic': True, 'font_name': 'Times New Roman', 'font_size': 12, 'font_color': 'black', 'align': 'left', 'top': True, 'bottom': True, 'left': False, 'right': False}))

    # Main body
    body_max_row_idx, body_max_col_idx = df_full_summary.shape

    for c, r in tqdm_product(range(body_max_col_idx), range(body_max_row_idx)):
        row_to_write = startrow + header_range + r
        col_to_write = startcol + 1 + c # 1 is for index
        body_formats = {'num_format': '0.00', 'font_name': 'Times New Roman', 'font_size': 12, 'font_color': 'black', 'align': 'center', 'text_wrap': True, 'left': False, 'right': False}

        if r == 0:
            body_formats |= {'top': True, 'bottom': True, 'left': False, 'right': False}
            worksheet.set_column(col_to_write, col_to_write, 10)

        if r == body_max_row_idx-1:
            body_formats |= {'bottom': True}

        if c == 0:
            body_formats |= {'align': 'left'}
            worksheet.set_column(col_to_write, col_to_write, 15)

        worksheet.write(row_to_write, col_to_write, df_full_summary.iloc[r, c], workbook.add_format(body_formats))

    # Add Note
    note_format = {'italic': True, 'font_name': 'Times New Roman', 'font_size': 10, 'font_color': 'black', 'align': 'left', 'left': False, 'right': False}
    worksheet.merge_range(endrow, startcol, endrow, endcol, 'Note.', workbook.add_format(note_format))
    # Add text
    for i, text in enumerate(text_to_add_list):
        worksheet.merge_range(endrow + 1 + i , startcol, endrow + 1 + i, endcol, text, workbook.add_format(note_format))

    writer.close()

# %%
# Function to make full regression report
def make_full_report(
    results, dv, analysis_type, model_name, dvs_name, ivs_name, ivs_type, df_name,
    regression_info_dict=None, regressor_order=None, text_to_add_list=None, title=None, model_names=None
):
    '''
    Make a full report for a regression analysis.
    results: statsmodels regression results object or list of results objects
    dv: str, dependent variable name
    '''

    if regression_info_dict is None:
        # Regression info dict
        regression_info_dict = {
            'F': lambda x: f'{x.fvalue:.3f}',
            'F (p-value)': lambda x: f'{x.f_pvalue:.3f}',
            'df_model': lambda x: f'{x.df_model:.0f}',
            'df_resid': lambda x: f'{x.df_resid:.0f}',
            'df_total': lambda x: f'{x.df_resid + x.df_model + 1:.0f}',
            'R-squared': lambda x: f'{x.rsquared:.3f}',
            'R-squared Adj.': lambda x: f'{x.rsquared_adj:.3f}',
            'Unstandardized Coefficent B (b)': lambda x: f'{x.params[0]:.3f}',
            'Standard Error (SE)': lambda x: f'{x.bse[0]:.3f}',
            'Standardized Coefficient b* (β)': lambda x: f'{x.params[0] / x.model.endog.std():.3f}',
            't': lambda x: f'{x.tvalues[0]:.3f}',
            't (p-value)': lambda x: f'{x.pvalues[0]:.3f}',
            '95% CI': lambda x: f'{x.conf_int().iloc[0, 1]:.3f} - {x.conf_int().iloc[0, 1]:.3f}',
            'Log-Likelihood': lambda x: f'{x.llf:.3f}',
            'Pseudo R2': lambda x: f'{x.prsquared:.3f}',
            'AIC': lambda x: f'{x.aic:.3f}',
            'BIC': lambda x: f'{x.bic:.3f}',
            'ICC': lambda x: f'{x.rsquared / (x.rsquared + (x.nobs - 1) * x.mse_resid):.3f}',
            'RMSE': lambda x: f'{x.mse_resid ** 0.5:.3f}',
            'RMSE (std)': lambda x: f'{x.mse_resid ** 0.5 / x.model.endog.std():.3f}',
            'Omnibus': lambda x: f'{sms.omni_normtest(x.resid).statistic:.3f}',
            'Omnibus (p-value)': lambda x: f'{sms.omni_normtest(x.resid).pvalue:.3f}',
            'Skew': lambda x: f'{sms.jarque_bera(x.resid)[-2]:.3f}',
            'Kurtosis': lambda x: f'{sms.jarque_bera(x.resid)[-1]:.3f}',
            'Jarque-Bera (JB)': lambda x: f'{sms.jarque_bera(x.resid)[0]:.3f}',
            'Jarque-Bera (p-value)': lambda x: f'{sms.jarque_bera(x.resid)[1]:.3f}',
            'Model Name': lambda x: f'{x.model.__class__.__name__}',
            'N': lambda x: f'{int(x.nobs):d}',
            # 'Summary': lambda x: f'{x.summary()}',
            # 'F (p-value - FDR)': lambda x: f'{x.f_pvalue_fdr:.3f}',
            # 'F (p-value - Bonferroni)': lambda x: f'{x.f_pvalue_bonf:.3f}',
            # 't (p-value - FDR)': lambda x: f'{x.pvalues_fdr[1]:.3f}',
            # 't (p-value - Bonferroni)': lambda x: f'{x.pvalues_bonf[1]:.3f}',
        }
        if isinstance(results, list):
            results_to_check = results[0]
        else:
            results_to_check = results
        if all('const' in x for x in zip(results_to_check.params.index, results_to_check.bse.index, results_to_check.tvalues.index, results_to_check.pvalues.index)):
            regression_info_dict = regression_info_dict | {
                'Intercept': lambda x: f'{x.params["const"]:.5f}',
                'Intercept (std)': lambda x: f'{x.bse["const"]:.5f}',
                'Intercept t': lambda x: f'{x.tvalues["const"]:.5f}',
                'Intercept t (p-value)': lambda x: f'{x.pvalues["const"]:.5f}',
                'Intercept (95% CI)': lambda x: f'{x.conf_int().loc["const"][0]:.5f} - {x.conf_int().loc["const"][1]:.5f}',
            }
    if model_names is None:
        if isinstance(results, list):
            model_names = [
                f'{results_to_check.model.endog_names.split("_")[0] if "_" in results_to_check.model.endog_names else results_to_check.model.endog_names} Model {i}'
                for i in range(len(results))
            ]
            model_names[0] = model_names[0].replace('Model 0', 'Full Model')
        else:
            model_names = [
                f'{results.model.endog_names.split("_")[0] if "_" in results.model.endog_names else results.model.endog_names}'
            ]

    order_type = 'unordered' if regressor_order is None else 'ordered'
    if text_to_add_list is None:
        text_to_add_list = []
        if regressor_order is not None:
            text_to_add_list.append('Models are ordered by independent variable type.')

        else:
            text_to_add_list.append('Models are ordered by coefficient size, largest to smallest.')

    if title is None:
        title = f'{model_name} {analysis_type}: {dvs_name} x {ivs_name}'

    try:
        # Statsmodels summary_col
        full_summary = summary_col(
            results,
            stars=True,
            info_dict=regression_info_dict,
            regressor_order=regressor_order,
            float_format='%0.3f',
            model_names=model_names,
        )
        if isinstance(results, list) and len(results) > 4:
            full_summary.tables[0][full_summary.tables[0].filter(regex='Full Model').columns[0]].loc['Unstandardized Coefficent B (b)': '95% CI'] = ''

        # Add title and notes
        full_summary.add_title(title)
        text_to_add_list.extend(full_summary.extra_txt)
        for text in text_to_add_list:
            full_summary.add_text(text)
        # Save
        save_name = shorten_file_path(f'{table_save_path}{model_name} {df_name} - ALL {dv} {order_type} {analysis_type} on {ivs_type}')
        df_full_summary = pd.read_html(full_summary.as_html())[0]
        df_full_summary.to_csv(f'{save_name}.csv')
        df_full_summary.style.to_latex(f'{save_name}.tex', hrules=True)
        save_df_full_summary_excel(df_full_summary, title, text_to_add_list, save_name)

        return full_summary
    except IndexError as e:
        print(f'Making full report for {model_names[0]} due to the following error: {e}')
        return None

# %%
# Function to get regression standardized coefficients
def get_standardized_coefficients(results):

    # # Get standardized regression coefficients
    # std = np.asarray(constant.std(0))

    # if 'const' in results.params and 'const' in constant:
    #     std[0] = 1
    # tt = results.t_test(np.diag(std))
    # tt.c_names = results.model.exog_names

    # t-test
    std = results.model.exog.std(0)
    if 'const' in results.params:
        std[0] = 1
    tt = results.t_test(np.diag(std))
    if results.model.__class__.__name__ == 'MixedLM' or 'Group Var' in results.model.exog_names:
        offset = slice(None, -1)
        tt.c_names = results.model.exog_names[offset]
    else:
        offset = slice(None, None)
        tt.c_names = results.model.exog_names

    # Make df with standardized and unstandardized coefficients
    df_std_coef = pd.DataFrame(
        {
            'coef': results.params[offset].progress_apply(lambda x: f'{x:.5f}'),
            'std err': results.bse[offset].progress_apply(lambda x: f'{x:.5f}'),
            'std coef': (results.params[offset] / results.model.exog[offset].std(axis=0)).progress_apply(lambda x: f'{x:.5f}'),
            't': results.tvalues[offset].progress_apply(lambda x: f'{x:.5f}'),
            'P>|t|': results.pvalues[offset].progress_apply(lambda x: f'{x:.5f}'),
            '[0.025': results.conf_int()[0][offset].progress_apply(lambda x: f'{x:.5f}'),
            '0.975]': results.conf_int()[1][offset].progress_apply(lambda x: f'{x:.5f}'),
        }
    )
    # if 'Group Var' in df_std_coef.index:
    #     df_std_coef = df_std_coef.drop('Group Var', axis='index')
    # # Add standardized coefficients and other data from t-test
    # df_std_coef['std coef'] = tt.effect
    # df_std_coef['std err'] = tt.sd
    # df_std_coef['t'] = tt.statistic
    # df_std_coef['P>|t|'] = tt.pvalue
    # df_std_coef['[0.025'] = tt.conf_int()[:, 0]
    # df_std_coef['0.975]'] = tt.conf_int()[:, 1]
    # df_std_coef['var'] = [names[i] for i in range(len(results.model.exog_names))]
    # df_std_coef = df_std_coef.sort_values('std coef', ascending=False)
    df_std_coef = df_std_coef.reset_index().rename(columns={'index': 'var'})
    df_std_coef = df_std_coef.rename(
        columns={
            'var': 'Variable',
            'coef': 'Unstandardized Coefficent B (b)',
            'std err': 'Standard Error',
            'std coef':'Standardized Coefficient b* (β)',
            't': 't-value',
            'P>|t|': 'p-value',
            '[0.025': '95% CI Lower',
            '0.975]': '95% CI Upper'
        }
    )
    # Reorder columns
    df_std_coef = df_std_coef[[
        'Variable',
        'Unstandardized Coefficent B (b)',
        'Standard Error',
        'Standardized Coefficient b* (β)',
        't-value',
        'p-value',
        '95% CI Lower',
        '95% CI Upper'
    ]]

    return tt, df_std_coef

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
    'grooth': 'wholesaling',
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

with open(f'{code_dir}/1. Scraping/CBS/Data/keyword_trans_dict.txt', 'w') as f:
    json.dump(keyword_trans_dict, f)


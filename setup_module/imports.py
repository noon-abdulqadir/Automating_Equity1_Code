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
# %matplotlib notebook
# %matplotlib inline
# %%
import importlib
import os
import sys
from pathlib import Path

mod = sys.modules[__name__]

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
# %load_ext autoreload
# %autoreload 2

# %%
env_name = 'study1'
from_file = False

# os.system('source /opt/homebrew/Caskroom/miniforge/base/bin/activate')
# os.system(f'conda init --all')
# os.system(f'conda activate {env_name}')

# with open(f'{code_dir}/install_modules.txt', 'r') as f:
#     install_modules = f.readlines()

# if from_file:
#     for module in install_modules:
#         module = module.strip()
#         if module != '':
#             try:
#                 importlib.import_module(module)
#             except ImportError:
#                 print(f'Installing {module}')
#                 try:
#                     os.system(f'conda install --channel apple --yes {module} ')
#                 except Exception:
#                     os.system(f'python -m pip install {module}')

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
    import re
    import shutil
    import socket
    import string
    import tempfile
    import time
    import typing
    import unicodedata
    import warnings
    from collections import defaultdict
    from io import StringIO
    from random import randrange
    from subprocess import call
    from threading import Thread
    from typing import Dict, List, Optional, Set, Tuple

    import bokeh
    import cardinality
    import cbsodata
    import en_core_web_sm
    import gensim
    import gensim.downloader as gensim_api
    import IPython
    import IPython.core
    import joblib
    import langdetect
    import libmaths as lm
    import lxml
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    import matplotlib.image as img
    import matplotlib.pyplot as plt
    import nltk
    import nltk.data
    import numpy as np
    import openpyxl
    import pandas as pd
    import pingouin as pg
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    import pretty_errors
    import progressbar
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pygwalker as pyg
    import requests
    import researchpy as rp
    import scipy
    import seaborn as sns
    import selenium.webdriver as webdriver
    import selenium.webdriver.support.ui as ui
    import sh
    import simpledorff
    import sklearn as sk
    import spacy
    import specification_curve as specy
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import swifter
    import tensorflow as tf
    import torch
    import torch.nn as nn
    import tqdm
    # import tqdm.auto
    import transformers
    import urllib3
    import xgboost as xgb
    import xlsxwriter
#     import xorbits.pandas as xpd

    # from accelerate import Accelerator
    from bs4 import BeautifulSoup
    from gensim import corpora, models
    from gensim.corpora import Dictionary
    from gensim.models import (CoherenceModel, FastText, KeyedVectors,
                               TfidfModel, Word2Vec)
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phraser, Phrases
    from gensim.parsing.preprocessing import (preprocess_string,
                                              remove_stopwords)
    from gensim.similarities import (SoftCosineSimilarity,
                                     SparseTermSimilarityMatrix,
                                     WordEmbeddingSimilarityIndex)
    from gensim.test.utils import common_texts, datapath, get_tmpfile
    from gensim.utils import save_as_line_sentence, simple_preprocess
    from googletrans import Translator
    from http_request_randomizer.requests.proxy.requestProxy import \
        RequestProxy
    from icecream import ic
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.datasets import make_imbalance
    from imblearn.metrics import classification_report_imbalanced
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import (EditedNearestNeighbours, NearMiss,
                                         RandomUnderSampler, TomekLinks)
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.display import HTML, Image, Markdown, display
    from ipywidgets import FloatSlider, interactive
    from joblib import parallel_backend
    from keras.layers import Activation, Dense
    from keras.models import Sequential
    from langdetect import DetectorFactory, detect, detect_langs
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    from nltk import (WordNetLemmatizer, agreement, bigrams, pos_tag,
                      regexp_tokenize, sent_tokenize, trigrams, word_tokenize,
                      wordpunct_tokenize)
    from nltk.corpus import abc
    from nltk.corpus import stopwords as sw
    from nltk.corpus import wordnet as wn
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer
    from nltk.tokenize import WordPunctTokenizer
    from pandas.api.types import (is_numeric_dtype, is_object_dtype,
                                  is_string_dtype)
    from plot_metric.functions import BinaryClassification
    from scipy import spatial, stats
    from scipy.stats import (anderson, chi2_contingency, f_oneway,
                             mannwhitneyu, normaltest, shapiro, stats)
    from selenium.common.exceptions import *
    from selenium.common.exceptions import (ElementClickInterceptedException,
                                            ElementNotVisibleException,
                                            NoAlertPresentException,
                                            NoSuchElementException,
                                            TimeoutException)
    from selenium.webdriver import ActionChains, Chrome
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.desired_capabilities import \
        DesiredCapabilities
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import Select, WebDriverWait
    from sentence_transformers import SentenceTransformer, losses, util
    from sklearn import feature_selection, metrics, set_config, svm, utils
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import load_files
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                                  BaggingRegressor, ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier, StackingClassifier,
                                  StackingRegressor, VotingClassifier,
                                  VotingRegressor)
    from sklearn.feature_extraction.text import (CountVectorizer,
                                                 FeatureHasher,
                                                 TfidfVectorizer)
    from sklearn.feature_selection import (SelectFdr, SelectFpr,
                                           SelectFromModel, SelectFwe,
                                           SelectKBest, SelectPercentile, chi2,
                                           f_classif, f_regression,
                                           mutual_info_classif,
                                           mutual_info_regression)
    from sklearn.impute import SimpleImputer
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import (LogisticRegression,
                                      PassiveAggressiveClassifier, Perceptron,
                                      SGDClassifier)
    from sklearn.manifold import TSNE
    from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                                 balanced_accuracy_score, brier_score_loss,
                                 classification_report, cohen_kappa_score,
                                 confusion_matrix, f1_score, log_loss,
                                 make_scorer, matthews_corrcoef,
                                 precision_recall_curve, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import (GridSearchCV, KFold, LeaveOneOut,
                                         RandomizedSearchCV,
                                         RepeatedStratifiedKFold, ShuffleSplit,
                                         StratifiedKFold,
                                         StratifiedShuffleSplit,
                                         cross_val_score, cross_validate,
                                         learning_curve, train_test_split)
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
    from sklearn.preprocessing import (Binarizer, FunctionTransformer,
                                       LabelBinarizer, LabelEncoder,
                                       MinMaxScaler, OneHotEncoder,
                                       StandardScaler, scale)
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils import (check_consistent_length, check_random_state,
                               check_X_y)
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.utils.validation import (check_is_fitted, column_or_1d,
                                          has_fit_parameter)
    from statannotations.Annotator import Annotator
    from statsmodels.formula.api import ols
    from statsmodels.graphics.factorplots import interaction_plot
    from statsmodels.stats.diagnostic import het_white
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras import layers, models
    from tensorflow.keras import preprocessing
    from tensorflow.keras import preprocessing as kprocessing
    from textblob import TextBlob, Word
    from textblob.en.inflect import pluralize, singularize
    # from transformers import (
    #     AutoConfig,
    #     AutoModel,
    #     AutoTokenizer,
    #     BertConfig,
    #     BertModel,
    #     BertPreTrainedModel,
    #     BertTokenizer,
    #     TFBertModel,
    # )
    from transformers.trainer_pt_utils import get_parameter_names
    from webdriver_manager.chrome import ChromeDriverManager
    # from whatthelang import WhatTheLang
    from xgboost import XGBClassifier
#     from xorbits.numpy import arange, argmax, cumsum
    from yellowbrick.text import TSNEVisualizer

except ImportError as error:
    module_name = str(error).split('named')[1]
    print(f'The library {module_name} is not installed. Installing now.')
    # !conda install --channel apple --yes {module_name}

# %%
# Tweak Settings
model_download_dir = os.path.expanduser(f'{code_dir}/data/Language Models/')

nltk_path = f'{str(model_download_dir)}nltk/'
nltk.data.path.append(nltk_path)

gensim_path = f'{str(model_download_dir)}gensim/'
gensim_api.base_dir = os.path.dirname(gensim_path)
gensim_api.BASE_DIR = os.path.dirname(gensim_path)
gensim_api.GENSIM_DATA_DIR = os.path.dirname(gensim_path)
glove_path = f'{gensim_path}glove/'
fasttext_path = os.path.abspath(f'{gensim_path}fasttext-wiki-news-subwords-300')

IPython.core.page = print
IPython.display.clear_output
display(HTML('<style>.container { width:90% !important; }</style>'))
InteractiveShell.ast_node_interactivity = 'all'
csv.field_size_limit(sys.maxsize)
warnings.filterwarnings('ignore', category=DeprecationWarning)
pretty_errors.configure(
    separator_character = '*',
    filename_display    = pretty_errors.FILENAME_EXTENDED,
    line_number_first   = True,
    display_link        = True,
    lines_before        = 5,
    lines_after         = 2,
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color          = '  ' + pretty_errors.default_config.line_color,
    truncate_code       = True,
    display_locals      = True
)
pretty_errors.replace_stderr()

errors = (
    TypeError,
    AttributeError,
    ElementClickInterceptedException,
    ElementNotInteractableException,
    NoSuchElementException,
    NoAlertPresentException,
    TimeoutException,
)
pp = pprint.PrettyPrinter(indent=4)
tqdm.tqdm_notebook().pandas(desc='progress-bar')
pbar = progressbar.ProgressBar(maxval=10)
# %matplotlib widget
# %matplotlib notebook
# %matplotlib inline
mpl.use('MacOSX')
mpl.style.use(f'{code_dir}/setup_module/apa.mplstyle-main/apa.mplstyle')
mpl.rcParams['text.usetex'] = False
font = {'family': 'arial', 'weight': 'normal', 'size': 10}
mpl.rc('font', **font)
plt.style.use('ggplot')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', '{:.3f}'.format)
# lux.config.default_display = "lux"
# lux.config.plotting_backend = "matplotlib"

# nltk.download('words', download_dir = nltk_path)
# nltk.download('punkt', download_dir = nltk_path)
# nltk.download('stopwords', download_dir = nltk_path)
# nltk.download('omw-1.4', download_dir=f'{nltk_path}')
# nltk.download('wordnet', download_dir=f'{nltk_path}')
# nltk.download('averaged_perceptron_tagger', download_dir = nltk_path)
# nltk.download('maxent_ne_chunker', download_dir = nltk_path)
# nltk.download('vader_lexicon', download_dir = nltk_path)
# nltk.download_shell()

# nlp = en_core_web_sm.load()
nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('en_core_web_trf')


# %%
# Analysis
# Setting Variables
random_state = 42
random.seed(random_state)
np.random.seed(random_state)
DetectorFactory.seed = random_state

ivs = ['Gender', 'Age']
ivs_all = [
        'Gender',
        'Gender_Num',
        'Gender_Female',
        'Gender_Mixed',
        'Gender_Male',
        'Age',
        'Age_Num',
        'Age_Older',
        'Age_Mixed',
        'Age_Younger',
    ]
ivs_all_dummy_num = [
        'Gender_Num',
        'Gender_Female',
        'Gender_Mixed',
        'Gender_Male',
        'Age_Num',
        'Age_Older',
        'Age_Mixed',
        'Age_Younger',
    ]
ivs_all_dummy = [
        'Gender_Female',
        'Gender_Mixed',
        'Gender_Male',
        'Age_Older',
        'Age_Mixed',
        'Age_Younger',
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
ivs_age_dummy_num = [
        'Age_Older',
        'Age_Mixed',
        'Age_Younger',
    ]
order_gender = ['Female', 'Mixed Gender', 'Male']
order_age = ['Older', 'Mixed Age', 'Younger']
ivs_dict = {'Gender': order_gender, 'Age': order_age}

dvs = {}

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
    'English Requirement',
    'Dutch Requirement'
]

# %%
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
    'opleiding':'education',
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


# %%
# with open(f'{code_dir}/scraped_data/CBS/Data/keyword_trans_dict.txt', 'w') as f:
#     json.dump(keyword_trans_dict, f)

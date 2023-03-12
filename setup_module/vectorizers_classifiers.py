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
from setup_module.params import *
# from setup_module.scraping import *
# from setup_module.post_collection_processing import *
# from setup_module.classification import *

warnings.filterwarnings('ignore', category=DeprecationWarning)


# %%
# Vectorizers

## Word Embedding
### Word2Vec
# nlp = spacy.load('en_core_web_sm')
# w2v = KeyedVectors.load_word2vec_format(
#     f'{gensim_path}word2vec-google-news-300/word2vec-google-news-300.gz',
#     binary=True,
# )
# w2v.init_sims(replace=True)

# params_w2v = {
#     'Word2Vec__ngram_range': [(1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (1, 3)],
#     'Word2Vec__max_features': [None, 5000, 10000, 50000],
#     'Word2Vec__min_count': [5.0, 10.0, 15.0, 20.0, 25.0],
#     'Word2Vec__window': [2.0, 5.0, 10.0, 12.0, 15.0],
#     'Word2Vec__vector_size': [50.0, 100.0, 150.0, 200.0, 250.0],
#     'Word2Vec__iter': [50.0, 100.0, 150.0, 200.0, 250.0],
#     'Word2Vec__alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
#     'Word2Vec__min_df': [0.5, 1.0, 1.5, 2.0],
#     'Word2Vec__max_df': [0.25, 0.5, 0.75, 1.0, 1.5],
#     'Word2Vec__max_features': [None, 5000, 10000, 50000],
#     'Word2Vec__ngram_range': [(1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (1, 3)],
#     'Word2Vec__sublinear_tf': [True, False],
#     'Word2Vec__binary': [True, False],
#     'Word2Vec__norm': [None, 'l1', 'l2'],
#     'Word2Vec__min_n': [1, 2, 3, 4, 5],
#     'Word2Vec__max_n': [1, 2, 3, 4, 5],
#     'Word2Vec__workers': [1, 2, 3, 4, 5],
#     'Word2Vec__sorted_vocab': [True, False],
#     'Word2Vec__batch_words': [1000, 2000, 3000, 4000, 5000],
#     'Word2Vec__callbacks': [
#         None,
#         'keras.callbacks.TensorBoard',
#         'keras.callbacks.CSVLogger',
#     ],
#     'Word2Vec__epochs': [1, 2, 3, 4, 5],
#     'Word2Vec__shuffle': [True, False],
#     'Word2Vec__validation_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
# }

# ### Gensim FastText
# ft = gensim.models.FastText(sample=6e-5, min_alpha=0.0007, workers=multiprocessing.cpu_count() - 1)
# params_ft = {
#     'FastText__min_count': [5.0, 10.0, 15.0, 20.0, 25.0],
#     'FastText__window': [2.0, 5.0, 10.0, 12.0, 15.0],
#     'FastText__vector_size': [50.0, 100.0, 150.0, 200.0, 250.0],
#     'FastText__alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
#     'FastText__min_alpha': [
#         0.0001,
#         0.0002,
#         0.0003,
#         0.0004,
#         0.0005,
#         0.0006,
#         0.0007,
#         0.0008,
#         0.0009,
#         0.001,
#     ],
#     'FastText__epochs': [50.0, 100.0, 150.0, 200.0, 250.0],
#     'FastText__min_n': [1.0, 2.0, 3.0, 4.0, 5.0],
#     'FastText__max_n': [1.0, 2.0, 3.0, 4.0, 5.0],
#     'FastText__sg': [0, 1],
#     'FastText__workers': [cores - 1],
# }

### Word Emedding FeatureUnion
# em = FeatureUnion(
#     transformer_list=[('Word2Vec', w2v), ('FastText', ft)]
# )
# params_em = {**{'Word2Vec__' + k: v for k, v in params_w2v.items()}, **{'FastText__' + k: v for k, v in params_ft.items()}}

# params_em_pipe = {**params_w2v_pipe, **params_ft_pipe}

## BOW
### CountVectorizer
count = CountVectorizer()
params_count = {
    'analyzer': 'word',
    'ngram_range': (1, 3),
    'lowercase': 'True',
#     'stop_words': 'english',
    # 'max_df': 0.95,
    # 'min_df': 0.05,
}

params_count_pipe = {
    'CountVectorizer__analyzer': ['word'],
    'CountVectorizer__ngram_range': [(1, 3)],
    'CountVectorizer__lowercase': [True, False],
    'CountVectorizer__max_df': [0.9, 0.85, 0.8, 0.75, 0.70],
    'CountVectorizer__min_df': [0.10, 0.15, 0.2, 0.25, 0.30],
}

### TfidfVectorizer
tfidf = TfidfVectorizer()
params_tfidf = {
    'analyzer': 'word',
    'ngram_range': (1, 3),
    'lowercase': 'True',
#     'max_features': 10000,
    'use_idf': 'True',
#     'stop_words': 'english',
    # 'max_df': 0.95,
    # 'min_df': 0.05,
}

params_tfidf_pipe = {
#     'TfidfVectorizer__stop_words': ['english'],
    'TfidfVectorizer__analyzer': ['word'],
    'TfidfVectorizer__ngram_range': [(1, 3)],
    'TfidfVectorizer__lowercase': [True, False],
#     'TfidfVectorizer__max_features': [None, 5000, 10000, 50000],
    'TfidfVectorizer___use_idf': [True],
#     'TfidfVectorizer___smooth_idf': [True, False],
    'TfidfVectorizer__max_df': [0.9, 0.85, 0.8, 0.75, 0.70],
    'TfidfVectorizer__min_df': [0.10, 0.15, 0.2, 0.25, 0.30],
}

### BOW FeatureUnion
bow = FeatureUnion(
    transformer_list=[('CountVectorizer', count), ('TfidfVectorizer', tfidf)]
)
params_bow = {
    **{'CountVectorizer__' + k: v for k, v in params_count.items()},
    **{'TfidfVectorizer__' + k: v for k, v in params_tfidf.items()},
}

params_bow_pipe = {**params_count_pipe, **params_tfidf_pipe}

## Vectorizers Dict
vectorizers = {
    'CountVectorizer': [count, params_count],
    'TfidfVectorizer': [tfidf, params_tfidf],
    'UnionBOW': [bow, params_bow],
    # "UnionWordEmbedding": [em, params_em],
}

vectorizers_pipe = {
    'CountVectorizer': [count, params_count_pipe],
    'TfidfVectorizer': [tfidf, params_tfidf_pipe],
    'UnionBOW': [bow, params_bow_pipe],
    # "UnionWordEmbedding": [em, params_em_pipe],
}

## Vectorizers List
vectorizers_lst = [
    count.set_params(**params_count),
    tfidf.set_params(**params_tfidf),
    bow.set_params(**params_bow),
]

# Selectors
selector = SelectKBest(score_func=chi2, k='all')
selector_name = selector.__class__.__name__

# model_selector = SelectFromModel()
# model_selector_name = model_selector.__class__.__name__

### SelectKBest
selectkbest = SelectKBest()
params_selectkbest = {'score_func': 'chi2', 'k': 'all'}

params_selectkbest_pipe = {
    'SelectKBest__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
    'SelectKBest__k': [100, 500, 1000, 5000, 10000, 'all'],
}

### SelectPercentile
selectpercentile = SelectPercentile()
params_selectpercentile = {'score_func': 'chi2', 'percentile': 10}

params_selectpercentile_pipe = {
    'SelectPercentile__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
    'SelectPercentile__percentile': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}

### SelectFpr
selectfpr = SelectFpr()
params_selectfpr = {'score_func': 'chi2', 'alpha': 0.05}

params_selectfpr_pipe = {
    'SelectFpr__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
    'SelectFpr__alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
}

### SelectFdr
selectfdr = SelectFdr()
params_selectfdr = {'score_func': 'chi2', 'alpha': 0.05}

params_selectfdr_pipe = {
    'SelectFdr__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
    'SelectFdr__alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
}

### SelectFwe
selectfwe = SelectFwe()
params_selectfwe = {'score_func': 'chi2', 'alpha': 0.05}

params_selectfwe_pipe = {
    'SelectFwe__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
    'SelectFwe__alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
}

## Selectors Dict
selectors = {
    'SelectKBest': [selectkbest, params_selectkbest],
    'SelectPercentile': [selectpercentile, params_selectpercentile],
    'SelectFpr': [selectfpr, params_selectfpr],
    'SelectFdr': [selectfdr, params_selectfdr],
    'SelectFwe': [selectfwe, params_selectfwe],
}

selectors_pipe = {
    'SelectKBest': [selectkbest, params_selectkbest_pipe],
    'SelectPercentile': [selectpercentile, params_selectpercentile_pipe],
    'SelectFpr': [selectfpr, params_selectfpr_pipe],
    'SelectFdr': [selectfdr, params_selectfdr_pipe],
    'SelectFwe': [selectfwe, params_selectfwe_pipe],
}

## Selectors List
selectors_lst = [
    selectkbest.set_params(**params_selectkbest),
    selectpercentile.set_params(**params_selectpercentile),
    selectfpr.set_params(**params_selectfpr),
    selectfdr.set_params(**params_selectfdr),
    selectfwe.set_params(**params_selectfwe),
]

# Classifiers
### Dummy Classifier
dummy = DummyClassifier()
params_dummy_freq = {'strategy': 'most_frequent', 'random_state': random_state}
params_dummy_stratified = {'strategy': 'stratified', 'random_state': random_state}
params_dummy_uniform = {'strategy': 'uniform', 'random_state': random_state}

params_dummy_pipe = {
    'DummyClassifier__strategy': [
        'stratified',
        'most_frequent',
        'prior',
        'uniform',
        'constant',
    ],
    'DummyClassifier__random_state': [42, 200],
}

### Multinomial Naive Bayes
nb = MultinomialNB()
params_nb = {'alpha': 0.1, 'fit_prior': True, 'class_prior': None}

params_nb_pipe = {
    'MultinomialNB__fit_prior': [True, False],
    'MultinomialNB__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'MultinomialNB__class_prior': [None, [0.1, 0.9], [0.2, 0.8]],
}

### Bernoulli Naive Bayes
bnb = BernoulliNB()
params_bnb = {'alpha': 0.1, 'fit_prior': True, 'class_prior': None}

params_bnb_pipe = {
    'BernoulliNB__fit_prior': [True, False],
    'BernoulliNB__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'BernoulliNB__class_prior': [None, [0.1, 0.9], [0.2, 0.8]],
}

### Gaussian Naive Bayes
gnb = GaussianNB()
params_gnb = {'var_smoothing': 1e-9}

params_gnb_pipe = {
    'GaussianNB__var_smoothing': [
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1e0,
        1e1,
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
        1e7,
        1e8,
        1e9,
    ],
}

### KNeighbors Classifier
knn = KNeighborsClassifier()
params_knn = {
    'n_neighbors': 3,
    'weights': 'uniform',
    'algorithm': 'auto',
    'leaf_size': 30,
    'p': 2,
    'metric': 'minkowski',
    'metric_params': None,
    'n_jobs': n_jobs,
}

params_knn_pipe = {
    'KNeighborsClassifier__weights': ['uniform', 'distance'],
    'KNeighborsClassifier__n_neighbors': [2, 5, 15, 30, 45, 64],
    'KNeighborsClassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'KNeighborsClassifier__leaf_size': [30, 50, 100, 200, 300, 500],
    'KNeighborsClassifier__p': [1, 2, 3, 4, 5],
    'KNeighborsClassifier__metric': [
        'minkowski',
        'euclidean',
        'cityblock',
        'chebyshev',
        'cosine',
        'correlation',
        'hamming',
        'jaccard',
        'l1',
        'l2',
        'manhattan',
    ],
    'KNeighborsClassifier__n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'KNeighborsClassifier__metric_params': [None, {'p': 2}, {'p': 3}],
}

### Logistic Regression
lr = LogisticRegression()
params_lr = {
    'penalty': 'l2',
    'dual': False,
    'tol': 0.0001,
    'C': 1.0,
    'fit_intercept': True,
    'intercept_scaling': 1,
    'class_weight': class_weight,
    'random_state': random_state,
    'solver': 'liblinear',
    'max_iter': 100,
    'multi_class': 'ovr',
    'verbose': 0,
    'warm_start': False,
    'n_jobs': n_jobs,
}

params_lr_pipe = {
    'LogisticRegression__penalty': ['l1', 'l2'],
    'LogisticRegression__random_state': [42, 200],
    'LogisticRegression__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'LogisticRegression__max_iter': [100, 200, 300, 500, 1000],
    'LogisticRegression__multi_class': ['ovr', 'multinomial'],
    'LogisticRegression__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'LogisticRegression__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'LogisticRegression__n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

### Passive Aggressive
pa = PassiveAggressiveClassifier()
params_pa = {
    'C': 1.0,
    'fit_intercept': True,
    'max_iter': 1000,
    'tol': 0.0001,
    'class_weight': class_weight,
    'verbose': 0,
    'random_state': random_state,
    'loss': 'hinge',
    'n_jobs': n_jobs,
}

params_pa_pipe = {
    'PassiveAggressiveClassifier__loss': ['hinge', 'squared_hinge'],
    'PassiveAggressiveClassifier__n_iter': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'PassiveAggressiveClassifier__random_state': [42, 200],
    'PassiveAggressiveClassifier__fit_intercept': [True, False],
    'PassiveAggressiveClassifier__class_weight': [None, 'balanced'],
    'PassiveAggressiveClassifier__n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PassiveAggressiveClassifier__tol': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'PassiveAggressiveClassifier__max_iter': [
        100,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
    ],
    'PassiveAggressiveClassifier__C': [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
}

### Stochastic Gradient Descent Aggressive
sgd = SGDClassifier()
params_sgd = {
    'fit_intercept': True,
    'max_iter': 1000,
    'tol': 0.0001,
    'class_weight': class_weight,
    'verbose': 0,
    'random_state': random_state,
    'loss': 'hinge',
    'n_jobs': n_jobs,
}

params_sgd_pipe = {
    'SGDClassifier__loss': ['hinge', 'squared_hinge'],
    'SGDClassifier__n_iter': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'SGDClassifier__random_state': [42, 200],
    'SGDClassifier__fit_intercept': [True, False],
    'SGDClassifier__class_weight': [None, 'balanced'],
    'SGDClassifier__n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'SGDClassifier__tol': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'SGDClassifier__max_iter': [
        100,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
    ],
}

### SVM
svm = LinearSVC()
params_svm = {
    'penalty': 'l2',
    'loss': 'hinge',
    'dual': True,
    'tol': 0.0001,
    'C': 1.0,
    'fit_intercept': True,
    'intercept_scaling': 1,
    'class_weight': class_weight,
    'random_state': random_state,
    'max_iter': 1000,
    'multi_class': 'ovr',
    'verbose': 0,
}

params_svm_pipe = {
    'LinearSVC__penalty': ['l1', 'l2'],
    'LinearSVC__loss': ['hinge', 'squared_hinge'],
    'LinearSVC__random_state': [42, 200],
    'LinearSVC__dual': [True, False],
    'LinearSVC__tol': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'LinearSVC__max_iter': [
        100,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
    ],
    'LinearSVC__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'LinearSVC__fit_intercept': [True, False],
    'LinearSVC__class_weight': [None, 'balanced'],
    'LinearSVC__intercept_scaling': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'LinearSVC__multi_class': ['ovr', 'crammer_singer'],
}

### Decision Tree
dt = DecisionTreeClassifier()
params_dt = {
    'criterion': 'gini',
    'splitter': 'best',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features': None,
    'random_state': random_state,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
}

params_dt_pipe = {
    'DecisionTreeClassifier__max_depth': [5, 10],
    'DecisionTreeClassifier__criterion': ['gini', 'entropy'],
    'DecisionTreeClassifier__random_state': [42, 200],
    'DecisionTreeClassifier__splitter': ['best', 'random'],
    'DecisionTreeClassifier__min_samples_split': [
        2,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ],
    'DecisionTreeClassifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'DecisionTreeClassifier__min_weight_fraction_leaf': [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    'DecisionTreeClassifier__max_features': [None, 'auto', 'sqrt', 'log2'],
    'DecisionTreeClassifier__max_leaf_nodes': [
        None,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ],
    'DecisionTreeClassifier__min_impurity_decrease': [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
}

### Random Forest
rf = RandomForestClassifier()
params_rf = {
    'n_estimators': 10,
    'criterion': 'log_loss',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': n_jobs,
    'random_state': random_state,
    'verbose': 0,
    'warm_start': False,
    'class_weight': class_weight,
}

params_rf_pipe = {
    'RandomForestClassifier__n_estimators': [10, 20],
    'RandomForestClassifier__n_jobs': [-1],
    'RandomForestClassifier__max_depth': [5, 10],
    'RandomForestClassifier__max_feature': [*np.arange(0.1, 1.1, 0.1)],
    'RandomForestClassifier__random_state': [42, 200],
    'RandomForestClassifier__criterion': ['gini', 'entropy', 'log_loss'],
    'RandomForestClassifier__min_samples_split': [
        2,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ],
    'RandomForestClassifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'RandomForestClassifier__min_weight_fraction_leaf': [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    'RandomForestClassifier__max_leaf_nodes': [
        None,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ],
    'RandomForestClassifier__min_impurity_decrease': [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    'RandomForestClassifier__bootstrap': [True, False],
    'RandomForestClassifier__oob_score': [True, False],
    'RandomForestClassifier__class_weight': [None, 'balanced'],
}

### Extra Trees
et = ExtraTreesClassifier()
params_et = {
    'n_estimators': 10,
    'criterion': 'log_loss',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': n_jobs,
    'random_state': random_state,
    'verbose': 0,
    'warm_start': False,
    'class_weight': class_weight,
}

params_et_pipe = {
    'ExtraTreesClassifier__n_estimators': [10, 20],
    'ExtraTreesClassifier__n_jobs': [-1],
    'ExtraTreesClassifier__max_depth': [5, 10],
    'ExtraTreesClassifier__max_feature': [*np.arange(0.1, 1.1, 0.1)],
    'ExtraTreesClassifier__random_state': [42, 200],
    'ExtraTreesClassifier__criterion': ['gini', 'entropy', 'log_loss'],
    'ExtraTreesClassifier__min_samples_split': [
        2,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ],
    'ExtraTreesClassifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'ExtraTreesClassifier__min_weight_fraction_leaf': [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    'ExtraTreesClassifier__max_leaf_nodes': [
        None,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ],
    'ExtraTreesClassifier__min_impurity_decrease': [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    'ExtraTreesClassifier__bootstrap': [True, False],
    'ExtraTreesClassifier__oob_score': [True, False],
    'ExtraTreesClassifier__class_weight': [None, 'balanced'],
}

### Gradient Boosting
gbc = GradientBoostingClassifier()
params_gbc = {
    'loss': 'deviance',
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 1.0,
    'criterion': 'friedman_mse',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_depth': 3,
    'min_impurity_decrease': 0.0,
    'init': None,
    'random_state': random_state,
    'max_features': None,
    'verbose': 0,
    'max_leaf_nodes': None,
    'warm_start': False,
}

params_gbc_pipe = {
    'GradientBoostingClassifier__max_depth': [3, 5],
    'GradientBoostingClassifier__min_samples_leaf': [1, 2],
    'GradientBoostingClassifier__criterion': ['gini', 'entropy'],
    'GradientBoostingClassifier__random_state': [42, 200],
    'GradientBoostingClassifier__n_estimators': [50, 100, 150],
    'GradientBoostingClassifier__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'GradientBoostingClassifier__loss': ['deviance', 'exponential'],
    'GradientBoostingClassifier__subsample': [*np.arange(0.1, 1.1, 0.1)],
    'GradientBoostingClassifier__min_samples_split': [
        2,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ],
    'GradientBoostingClassifier__min_weight_fraction_leaf': [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    'GradientBoostingClassifier__max_features': [None, 'auto', 'sqrt', 'log2'],
    'GradientBoostingClassifier__max_leaf_nodes': [
        None,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ],
    'GradientBoostingClassifier__min_impurity_decrease': [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
}

### AdaBoost
ada = AdaBoostClassifier()
params_ada = {
    'base_estimator': None,
    'n_estimators': 50,
    'learning_rate': 1.0,
    'algorithm': 'SAMME.R',
    'random_state': random_state,
}

params_ada_pipe = {
    'AdaBoostClassifier__max_depth': [3, 5],
    'AdaBoostClassifier__min_samples_leaf': [1, 2],
    'AdaBoostClassifier__criterion': ['gini', 'entropy'],
    'AdaBoostClassifier__random_state': [42, 200],
    'AdaBoostClassifier__n_estimators': [50, 100, 150],
    'AdaBoostClassifier__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'AdaBoostClassifier__base_estimator': [
        SVC(probability=True, kernel='linear'),
        LogisticRegression(),
        MultinomialNB(),
    ],
}

### XGBoost
xgb = XGBClassifier()
params_xgb = {
    'nthread':4, #when use hyperthread, xgboost may become slower
    'objective':'binary:logistic',
    'learning_rate': 0.05, #so called `eta` value
    'max_depth': 6,
    'min_child_weight': 11,
    'silent': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'n_estimators': 1000, #number of trees, change it to 1000 for better results
    'missing':-999,
    'seed': 1337,
    'eval_metric': 'auc',
    'sample_type': 'weighted',
    'verbosity': '0',
}

params_xgb_pipe = {
    'xgb__max_depth': [3, 4, 5, 6],
    'xgb__min_child_weight': [1, 3, 5, 7, 9],
    'xgb__learning_rate': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
    'xgb__n_estimators': [500, 1000, 1500, 2000],
    'xgb__subsample': [*np.arange(0.1, 1.1, 0.1)],
    'xgb__colsample_bytree': [*np.arange(0.1, 1.1, 0.1)],
    'xgb__seed': [42, 200],
    'xgb__nthread': [1, 2, 3, 4],
    'xgb__objective': ['binary:logitraw', 'binary:logistic', 'binary:hinge'],
    'xgb__eval_metric': ['auc', 'logloss'],
    'xgb__sample_type': ['weighted', 'uniform'],
}

### MLP Classifier
mlpc = MLPClassifier()
params_mlpc = {
    'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 'auto',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'power_t': 0.5,
    'max_iter': 200,
    'shuffle': True,
    'random_state': random_state,
    'tol': 0.0001,
    'verbose': False,
    'warm_start': False,
    'momentum': 0.9,
    'nesterovs_momentum': True,
    'early_stopping': False,
    'validation_fraction': 0.1,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-08,
}

params_mlpc_pipe = {
    'MLPClassifier__hidden_layer_sizes': [(100,), (50,), (25,), (10,), (5,), (1,)],
    'MLPClassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'MLPClassifier__solver': ['lbfgs', 'sgd', 'adam'],
    'MLPClassifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'MLPClassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'MLPClassifier__max_iter': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'MLPClassifier__random_state': [42, 200],
}

mlpr = MLPRegressor()
params_mlpr = {
    'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 'auto',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'power_t': 0.5,
    'max_iter': 200,
    'shuffle': True,
    'random_state': random_state,
    'tol': 0.0001,
    'verbose': False,
    'warm_start': False,
    'momentum': 0.9,
    'nesterovs_momentum': True,
    'early_stopping': False,
    'validation_fraction': 0.1,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-08,
}

params_mlpr_pipe = {
    'MLPRegressor__hidden_layer_sizes': [(100,), (50,), (25,), (10,), (5,), (1,)],
    'MLPRegressor__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'MLPRegressor__solver': ['lbfgs', 'sgd', 'adam'],
    'MLPRegressor__alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'MLPRegressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'MLPRegressor__max_iter': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'MLPRegressor__random_state': [42, 200],
}

### Sequential
# seq = keras.Sequential()
# params_seq = {
#     'layers': [
#         Dense(12, input_dim=8, activation='relu', name='layer1'),
#         Dense(8, activation='relu', name='layer2'),
#         Dense(1, activation='sigmoid', name='layer3'),
#     ]
# }

# params_seq_pipe = {
#     'keras.Sequential__layers': [
#         [
#             Dense(12, input_dim=8, activation='relu', name='layer1'),
#             Dense(8, activation='relu', name='layer2'),
#             Dense(1, activation='sigmoid', name='layer3'),
#         ],
#         [
#             Dense(12, input_dim=8, activation='relu', name='layer1'),
#             Dense(8, activation='relu', name='layer2'),
#             Dense(1, activation='sigmoid', name='layer3'),
#         ],
#         [
#             Dense(12, input_dim=8, activation='relu', name='layer1'),
#             Dense(8, activation='relu', name='layer2'),
#             Dense(1, activation='sigmoid', name='layer3'),
#         ]
#     ]
# }

# seq = Sequential()
# seq.add(Dense(12, input_dim=8, activation='relu'))
# seq.add(Dense(8, activation='relu'))
# seq.add(Dense(1, activation='sigmoid'))
# seq.compile(loss='binary_crossentropy', optimizer='adam', metrics=['precision', 'recall', 'accuracy'])

## Stacking and Voting Classifiers
estimators = [
    ('Multinomial Naive Bayes', MultinomialNB()),
    (
        'Logistic Regression',
        LogisticRegression(random_state=random_state, class_weight=class_weight),
    ),
]

### Voting Classifier
voting_classifier = VotingClassifier(estimators=estimators)
params_voting = {'n_jobs': n_jobs, 'voting': 'soft'}

params_voting_pipe = {
    'VotingClassifier__estimators': [
        ('dummy', dummy, params_dummy_freq),
        ('dummy', dummy, params_dummy_stratified),
        ('dummy', dummy, params_dummy_uniform),
        ('nb', nb, params_nb),
        ('bnb', bnb, params_bnb),
        ('gnb', gnb, params_gnb),
        ('knn', knn, params_knn),
        ('lr', lr, params_lr),
        ('pa', pa, params_pa),
        ('sgd', sgd, params_sgd),
        ('svm', svm, params_svm),
        ('dt', dt, params_dt),
        ('rf', rf, params_rf),
        ('gbc', gbc, params_gbc),
        ('ada', ada, params_ada),
        ('xgb', xgb, params_xgb),
        ('mlpc', mlpc, params_mlpc),
        ('mlpr', mlpr, params_mlpr),
    ],
    'VotingClassifier__voting': ['hard', 'soft'],
    'VotingClassifier__weights': [None, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
}

### Stacking Classifier
stacking_classifier = StackingClassifier(estimators=estimators)
# final_estimator = LogisticRegression(random_state=random_state, class_weight=class_weight)
final_estimator = RandomForestClassifier(
    random_state=random_state, class_weight={0: 1, 1: 2}
)
params_stacking = {
    'n_jobs': n_jobs,
    'final_estimator': final_estimator,
    'cv': KFold(n_splits=2),
}

params_stacking_pipe = {
    'StackingClassifier__estimator': [
        ('dummy', dummy),
        ('nb', nb),
        ('bnb', bnb),
        ('gnb', gnb),
        ('knn', knn),
        ('lr', lr),
        ('pa', pa),
        ('sgd', sgd),
        ('svm', svm),
        ('dt', dt),
        ('rf', rf),
        ('gbc', gbc),
        ('ada', ada),
        ('mlpc', mlpc),
    ],
    'StackingClassifier__cv': [3, 5, 7, 9, 11, 13, 15],
    'StackingClassifier__n_jobs': [-1],
    'StackingClassifier__stack_method': ['predict_proba', 'decision_function'],
    'StackingClassifier__passthrough': [True, False],
}

## Classifiers dict
classifiers = {
    'DummyClassifier_MostFrequent': [dummy, params_dummy_freq],
    'DummyClassifier_Stratified': [dummy, params_dummy_stratified],
    'DummyClassifier_Uniform': [dummy, params_dummy_uniform],
    'MultinomialNB': [nb, params_nb],
    'BernoulliNB': [bnb, params_bnb],
    'GaussianNB': [gnb, params_gnb],
    'KNeighborsClassifier': [knn, params_knn],
    'LogisticRegression': [lr, params_lr],
    'PassiveAggressiveClassifier': [pa, params_pa],
    'SGDClassifier': [sgd, params_sgd],
    'LinearSVC': [svm, params_svm],
    'DecisionTreeClassifier': [dt, params_dt],
    'RandomForestClassifier': [rf, params_rf],
    'GradientBoostingClassifier': [gbc, params_gbc],
    'AdaBoostClassifier': [ada, params_ada],
    'XGBClassifier': [xgb, params_xgb],
    'MLPClassifier': [mlpc, params_mlpc],
    'MLPRegressor': [mlpr, params_mlpr],
    'VotingClassifier': [voting_classifier, params_voting],
    'StackingClassifier': [stacking_classifier, params_stacking],
}

## Classifiers Pipe dict
classifiers_pipe = {
    'DummyClassifier': [dummy, params_dummy_pipe],
    'MultinomialNB': [nb, params_nb_pipe],
    'BernoulliNB': [bnb, params_bnb_pipe],
    'GaussianNB': [gnb, params_gnb_pipe],
    'KNeighborsClassifier': [knn, params_knn_pipe],
    'LogisticRegression': [lr, params_lr_pipe],
    'PassiveAggressiveClassifier': [pa, params_pa_pipe],
    'SGDClassifier': [sgd, params_sgd_pipe],
    'LinearSVC': [svm, params_svm_pipe],
    'DecisionTreeClassifier': [dt, params_dt_pipe],
    'RandomForestClassifier': [rf, params_rf_pipe],
    'GradientBoostingClassifier': [gbc, params_gbc_pipe],
    'AdaBoostClassifier': [ada, params_ada_pipe],
    'XGBClassifier': [xgb, params_xgb_pipe],
    'MLPClassifier': [mlpc, params_mlpc_pipe],
    'MLPRegressor': [mlpr, params_mlpr_pipe],
    'VotingClassifier': [voting_classifier, params_voting_pipe],
    'StackingClassifier': [stacking_classifier, params_stacking_pipe],
}

## Classifiers List
classifiers_lst = [
    dummy.set_params(**params_dummy_freq),
    dummy.set_params(**params_dummy_stratified),
    dummy.set_params(**params_dummy_uniform),
    nb.set_params(**params_nb),
    bnb.set_params(**params_bnb),
    gnb.set_params(**params_gnb),
    knn.set_params(**params_knn),
    lr.set_params(**params_lr),
    pa.set_params(**params_pa),
    sgd.set_params(**params_sgd),
    svm.set_params(**params_svm),
    dt.set_params(**params_dt),
    rf.set_params(**params_rf),
    et.set_params(**params_et),
    gbc.set_params(**params_gbc),
    ada.set_params(**params_ada),
    mlpc.set_params(**params_mlpc),
    stacking_classifier.set_params(**params_stacking),
    voting_classifier.set_params(**params_voting),
]
#     DummyClassifier(strategy='stratified', random_state=random_state),
#     MultinomialNB(),
#     GaussianNB(),
#     KNeighborsClassifier(
#         n_neighbors=3, n_jobs=n_jobs, algorithm='brute', metric='cosine', p=2, weights='uniform'
#     ),
#     LogisticRegression(random_state=random_state, class_weight=class_weight),
#     PassiveAggressiveClassifier(random_state=random_state, class_weight=class_weight),
#     LinearSVC(random_state=random_state, class_weight=class_weight),
#     DecisionTreeClassifier(random_state=random_state, class_weight=class_weight),
#     RandomForestClassifier(random_state=random_state, class_weight={0: 1, 1: 2}),
#     ExtraTreesClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state, 'criterion'='log_loss'),
#     GradientBoostingClassifier(
#         n_estimators=100, learning_rate=1, random_state=random_state
#     ),
#     AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=random_state),
#     MLPClassifier(
#         hidden_layer_sizes=(100,),
#         activation='relu',
#         solver='adam',
#         alpha=0.0001,
#         batch_size='auto',
#         learning_rate='constant',
#         learning_rate_init=0.001,
#         power_t=0.5,
#         max_iter=200,
#         shuffle=True,
#         random_state=random_state,
#         tol=0.0001,
#         verbose=False,
#         warm_start=False,
#         momentum=0.9,
#         nesterovs_momentum=True,
#         early_stopping=False,
#         validation_fraction=0.1,
#         beta_1=0.9,
#         beta_2=0.999,
#         epsilon=1e-08,
#     ),
#     # keras.Sequential(
#     #     [
#     #         Dense(12, input_dim=8, activation='relu', name='layer1'),
#     #         Dense(8, activation='relu', name='layer2'),
#     #         Dense(1, activation='sigmoid', name='layer3'),
#     #     ]
#     # ),
#     stacking_classifier,
#     voting_classifier
# ]

# Scorers dict
scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
}

# DF to store results
if use_dict_for_classifiers_vectorizers is True:
    index = pd.MultiIndex.from_product(
        [list(map(lambda classifier: classifier, classifiers.keys()))],
        names=['Classifiers'],
    )
    columns = pd.MultiIndex.from_product(
        [
            analysis_columns,
            list(map(lambda vectorizer: vectorizer, vectorizers.keys())),
            metrics_list,
        ],
        names=['Variable', 'Vectorizer', 'Measures'],
    )
    table_df = pd.DataFrame(index=index, columns=columns)

elif use_dict_for_classifiers_vectorizers is False:
    index = pd.MultiIndex.from_product(
        [
            list(
                map(
                    lambda classifier: classifier.__class__.__name__
                    if classifier.__class__.__name__ != 'DummyClassifier'
                    else classifier.__class__.__name__
                    + f' - {str(classifier.strategy).title()}',
                    classifiers_lst,
                )
            )
        ],
        names=['Classifiers'],
    )
    columns = pd.MultiIndex.from_product(
        [
            analysis_columns,
            list(
                map(lambda vectorizer: vectorizer.__class__.__name__, vectorizers_lst)
            ),
            metrics_list,
        ],
        names=['Variable', 'Vectorizer', 'Measures'],
    )
    table_df = pd.DataFrame(index=index, columns=columns)

# %%

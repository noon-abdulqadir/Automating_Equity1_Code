# %%
import os  # isort:skip # fmt:skip # noqa # nopep8
import sys  # isort:skip # fmt:skip # noqa # nopep8
from pathlib import Path  # isort:skip # fmt:skip # noqa # nopep8

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

sys.path.append(code_dir)
# %load_ext autoreload
# %autoreload 2

# %%
from setup_module.imports import *  # isort:skip # fmt:skip # noqa # nopep8

# %% [markdown]
# ### READ DATA

# %%
# Variables
# Sklearn variables
method = 'Supervised'
final_models_save_path = f'{models_save_path}{method} Results/'
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
scores = [
    'recall', 'accuracy', 'f1', 'roc_auc',
    'explained_variance', 'matthews_corrcoef'
]
scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
}
analysis_columns = ['Warmth', 'Competence']
text_col = 'Job Description spacy_sentencized'
metrics_dict = {
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
    'Fowlkes–Mallows Index': np.nan,
    'ROC': np.nan,
    'AUC': np.nan,
    f'{scoring.title()} Best Threshold': np.nan,
    f'{scoring.title()} Best Score': np.nan,
    'Log Loss/Cross Entropy': np.nan,
    'Cohen’s Kappa': np.nan,
    'Geometric Mean': np.nan,
    'Classification Report': np.nan,
    'Imbalanced Classification Report': np.nan,
    'Confusion Matrix': np.nan,
    'Normalized Confusion Matrix': np.nan,
}

# Transformer variables
max_length = 512
returned_tensor = 'pt'
cpu_counts = torch.multiprocessing.cpu_count()
device = torch.device('mps') if torch.has_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available(
) else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device_name = str(device.type)
print(f'Using {device_name.upper()}')
# Set random seed
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
DetectorFactory.seed = random_state
cores = multiprocessing.cpu_count()
bert_model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizerFast.from_pretrained(
    bert_model_name, strip_accents=True
)
bert_model = BertForSequenceClassification.from_pretrained(
    bert_model_name
).to(device)

# Plotting variables
pp = pprint.PrettyPrinter(indent=4)
tqdm.tqdm.pandas(desc='progress-bar')
tqdm_auto.tqdm.pandas(desc='progress-bar')
tqdm.notebook.tqdm().pandas(desc='progress-bar')
tqdm_auto.notebook_tqdm().pandas(desc='progress-bar')
# pbar = progressbar.ProgressBar(maxval=10)
mpl.use('MacOSX')
mpl.style.use(f'{code_dir}/setup_module/apa.mplstyle-main/apa.mplstyle')
mpl.rcParams['text.usetex'] = True
font = {'family': 'arial', 'weight': 'normal', 'size': 10}
mpl.rc('font', **font)
plt.style.use('tableau-colorblind10')
plt.set_cmap('Blues')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', '{:.2f}'.format)

# %% [markdown]
# ## Vectorizers

# %%
# CountVectorizer
count_ = CountVectorizer()
count_params = {
    #     'TfidfVectorizer__stop_words': ['english'],
    'CountVectorizer__analyzer': ['word'],
    'CountVectorizer__ngram_range': [(1, 3)],
    'CountVectorizer__lowercase': [True, False],
    'CountVectorizer__max_df': [0.90, 0.85, 0.80, 0.75, 0.70],
    'CountVectorizer__min_df': [0.10, 0.15, 0.20, 0.25, 0.30],
}
count = [count_, count_params]

# TfidfVectorizer
tfidf_ = TfidfVectorizer()
tfidf_params = {
    #     'TfidfVectorizer__stop_words': ['english'],
    'TfidfVectorizer__analyzer': ['word'],
    'TfidfVectorizer__ngram_range': [(1, 3)],
    'TfidfVectorizer__lowercase': [True, False],
    'TfidfVectorizer___use_idf': [True, False],
    'TfidfVectorizer__max_df': [0.90, 0.85, 0.80, 0.75, 0.70],
    'TfidfVectorizer__min_df': [0.10, 0.15, 0.20, 0.25, 0.30],
}
tfidf = [tfidf_, tfidf_params]

# Vectorizers List
vectorizers_list = [
    count,
    tfidf
]

# BOW FeatureUnion
transformer_list = []
bow_params = {}
for vectorizer_and_params in vectorizers_list:
    transformer_list.append(
        (vectorizer_and_params[0].__class__.__name__, vectorizer_and_params[0])
    )
    for k, v in vectorizer_and_params[1].items():
        bow_params[f'FeatureUnion__{k}'] = v

bow_ = FeatureUnion(
    transformer_list=[transformer_list]
)
bow = [bow_, bow_params]

# Vectorizers List append bow
vectorizers_list.append(bow)

# Vectorizers Dict
vectorizers_pipe = {
    vectorizer_and_params[0].__class__.__name__: vectorizer_and_params
    for vectorizer_and_params in vectorizers_list
}


# %% [markdown]
# ## Selectors

# %%
# SelectKBest
selectkbest_ = SelectKBest()
selectkbest_params = {
    'SelectKBest__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
    'SelectKBest__k': ['all'],
}
selectkbest = [selectkbest_, selectkbest_params]

# SelectPercentile
selectperc_ = SelectPercentile()
selectperc_params = {
    'SelectPercentile__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
    'SelectPercentile__percentile': [30, 40, 50, 60, 70, 80],
}
selectperc = [selectperc_, selectperc_params]

# SelectFpr
selectfpr_ = SelectFpr()
selectfpr_params = {
    'SelectFpr__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
}
selectfpr = [selectfpr_, selectfpr_params]

# SelectFdr
selectfdr_ = SelectFdr()
selectfdr_params = {
    'SelectFdr__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
}
selectfdr = [selectfdr_, selectfdr_params]

# SelectFwe
selectfwe_ = SelectFwe()
selectfwe_params = {
    'SelectFwe__score_func': [f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression],
}
selectfwe = [selectfwe_, selectfwe_params]

# Selectors List
selectors_list = [
    selectkbest,
    # selectperc, selectfpr, selectfdr, selectfwe
]
# Selectors Dict
selectors_pipe = {
    selector_and_params[0].__class__.__name__: selector_and_params
    for selector_and_params in selectors_list
}


# %% [markdown]
# ## Resamplers

# %%
# Resamplers
# SMOTETomek Resampler
smotetomek_ = SMOTETomek()
smotetomek_params = {
    'SMOTETomek__random_state': [random_state],
    'SMOTETomek__tomek': [TomekLinks(sampling_strategy='majority', n_jobs=n_jobs)],
}
smotetomek = [smotetomek_, smotetomek_params]

# Resampler List
resamplers_list = [
    smotetomek,
]

# Resampler Dict
resamplers_pipe = {
    resampler_and_params[0].__class__.__name__: resampler_and_params
    for resampler_and_params in resamplers_list
}

# %% [markdown]
# ## Classifiers

# %%
# Classifiers
# Dummy Classifier
dummy_ = DummyClassifier()
dummy_params = {
    'DummyClassifier__strategy': [
        'stratified',
        'most_frequent',
        'prior',
        'uniform',
    ],
    'DummyClassifier__random_state': [random_state],
}

dummy = [dummy_, dummy_params]

# Multinomial Naive Bayes
nb_ = MultinomialNB()
nb_params = {
    'MultinomialNB__fit_prior': [True, False],
    # 'MultinomialNB__alpha': [0.1, 0.2, 0.3],
}

nb = [nb_, nb_params]

# Bernoulli Naive Bayes
bnb_ = BernoulliNB()
bnb_params = {
    'BernoulliNB__fit_prior': [True],
    'BernoulliNB__alpha': [0.1, 0.2, 0.3],
}

bnb = [bnb_, bnb_params]

# Gaussian Naive Bayes
gnb_ = GaussianNB()
gnb_params = {
    'GaussianNB__var_smoothing': [1e-9],
}

gnb = [gnb_, gnb_params]

# KNeighbors Classifier
knn_ = KNeighborsClassifier()
knn_params = {
    'KNeighborsClassifier__weights': ['uniform', 'distance'],
    'KNeighborsClassifier__n_neighbors': [2, 5, 15],
    'KNeighborsClassifier__algorithm': ['auto'],
    # 'KNeighborsClassifier__p': [1, 2, 3, 4, 5],
    # 'KNeighborsClassifier__metric': [
    #     'minkowski',
    #     'euclidean',
    #     'cosine',
    #     'correlation',
    # ],
    # 'KNeighborsClassifier__leaf_size': [30, 50, 100, 200, 300, 500],
    # 'KNeighborsClassifier__metric_params': [None, {'p': 2}, {'p': 3}],
}

knn = [knn_, knn_params]

# Logistic Regression
lr_ = LogisticRegression()
lr_params = {
    'LogisticRegression__class_weight': [class_weight],
    'LogisticRegression__random_state': [random_state],
    'LogisticRegression__fit_intercept': [True, False],
    'LogisticRegression__multi_class': ['auto'],
    'LogisticRegression__solver': ['liblinear'],
    'LogisticRegression__C': [0.01, 1, 100],
    # 'LogisticRegression__penalty': ['elasticnet'],
    # 'LogisticRegression__max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],,
}

lr = [lr_, lr_params]

# Passive Aggressive
pa_ = PassiveAggressiveClassifier()
pa_params = {
    'PassiveAggressiveClassifier__loss': ['hinge', 'squared_hinge'],
    'PassiveAggressiveClassifier__random_state': [random_state],
    'PassiveAggressiveClassifier__fit_intercept': [True, False],
    'PassiveAggressiveClassifier__class_weight': [class_weight],
    'PassiveAggressiveClassifier__shuffle': [True, False],
    'PassiveAggressiveClassifier__C': [0.01, 1, 100],
    # 'PassiveAggressiveClassifier__max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],,
}

pa = [pa_, pa_params]

# Perceptron
ptron_ = linear_model.Perceptron()
ptron_params = {
    'Perceptron__penalty': ['elasticnet'],
    'Perceptron__random_state': [random_state],
    'Perceptron__fit_intercept': [True, False],
    'Perceptron__class_weight': [class_weight],
    'Perceptron__shuffle': [True, False],
    # 'Perceptron__max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],,
}

ptron = [ptron_, ptron_params]

# Stochastic Gradient Descent Aggressive
sgd_ = SGDClassifier()
sgd_params = {
    'SGDClassifier__loss': ['hinge', 'squared_hinge'],
    'SGDClassifier__random_state': [random_state],
    'SGDClassifier__fit_intercept': [True, False],
    'SGDClassifier__class_weight': [class_weight],
    # 'SGDClassifier__max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],
}

sgd = [sgd_, sgd_params]

# SVM
svm_ = LinearSVC()
svm_params = {
    'LinearSVC__loss': ['hinge', 'squared_hinge'],
    'LinearSVC__random_state': [random_state],
    'LinearSVC__fit_intercept': [True, False],
    'LinearSVC__class_weight': [class_weight],
    'LinearSVC__C': [0.01, 1, 100],
    'LinearSVC__max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],
    # 'LinearSVC__multi_class': ['ovr', 'crammer_singer'],
}

svm = [svm_, svm_params]

# Decision Tree
dt_ = DecisionTreeClassifier()
dt_params = {
    'DecisionTreeClassifier__max_depth': [2, 5, 10],
    'DecisionTreeClassifier__criterion': ['gini', 'entropy', 'log_loss'],
    'DecisionTreeClassifier__random_state': [random_state],
    'DecisionTreeClassifier__splitter': ['best', 'random'],
    'DecisionTreeClassifier__class_weight': [class_weight],
    # 'DecisionTreeClassifier__max_features': ['auto'],
}

dt = [dt_, dt_params]

# Random Forest
rf_ = RandomForestClassifier()
rf_params = {
    'RandomForestClassifier__max_depth': [2, 5, 10],
    'RandomForestClassifier__n_estimators': [10, 20],
    'RandomForestClassifier__criterion': ['gini', 'entropy', 'log_loss'],
    'RandomForestClassifier__random_state': [random_state],
    'RandomForestClassifier__class_weight': [class_weight],
    'RandomForestClassifier__oob_score': [True],
    # 'RandomForestClassifier__max_features': ['auto'],
}

rf = [rf_, rf_params]

# Extra Trees
et_ = ExtraTreesClassifier()
et_params = {
    'RandomForestClassifier__max_depth': [2, 5, 10],
    'RandomForestClassifier__n_estimators': [10, 20],
    'ExtraTreesClassifier__max_feature': ['auto'],
    'ExtraTreesClassifier__random_state': [random_state],
    'ExtraTreesClassifier__criterion': ['gini', 'entropy', 'log_loss'],
    'ExtraTreesClassifier__class_weight': [class_weight],
}

et = [et_, et_params]

# Gradient Boosting
gbc_ = GradientBoostingClassifier()
gbc_params = {
    'GradientBoostingClassifier__random_state': [random_state],
    'GradientBoostingClassifier__loss': ['log_loss', 'deviance', 'exponential'],
    # 'GradientBoostingClassifier__max_features': ['auto'],
}

gbc = [gbc_, gbc_params]

# AdaBoost
ada_ = AdaBoostClassifier()
ada_params = {
    'AdaBoostClassifier__criterion': ['gini', 'entropy'],
    'AdaBoostClassifier__random_state': [random_state],
    'AdaBoostClassifier__n_estimators': [50, 100, 150],
    'AdaBoostClassifier__base_estimator': [
        SVC(kernel='linear'),
        LogisticRegression(),
        MultinomialNB(),
        DecisionTreeClassifier(),
    ],
}

ada = [ada_, ada_params]

# XGBoost
xgb_ = XGBClassifier()
xgb_params = {
    'XGBClassifier__seed': [random_state],
    'XGBClassifier__eval_metric': ['logloss'],
    'XGBClassifier__objective': ['binary:logistic'],
}

xgb = [xgb_, xgb_params]

# MLP Classifier
mlpc_ = MLPClassifier()
mlpc_params = {
    'MLPClassifier__hidden_layer_sizes': [(100,), (50,), (25,), (10,), (5,), (1,)],
    'MLPClassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'MLPClassifier__solver': ['lbfgs', 'sgd', 'adam'],
    'MLPClassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'MLPClassifier__random_state': [random_state],
}

mlpc = [mlpc_, mlpc_params]

# MLP Regressor
mlpr_ = MLPRegressor()
mlpr_params = {
    'MLPRegressor__hidden_layer_sizes': [(100,), (50,), (25,), (10,), (5,), (1,)],
    'MLPRegressor__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'MLPRegressor__solver': ['lbfgs', 'sgd', 'adam'],
    'MLPRegressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'MLPRegressor__random_state': [random_state],
}

mlpr = [mlpr_, mlpr_params]

# Classifiers List
classifers_list = [
    dummy, nb, knn, lr, pa, ptron, svm, dt, rf, ada, xgb, mlpc,
    # bnb, gnb, sgd, et, gbc, mlpr
]

# Classifiers Dict
classifiers_pipe = {
    classifier_and_params[0].__class__.__name__: classifier_and_params
    for classifier_and_params in classifers_list
}

# Voting and Stacking Classifiers
# Estimators for Voting and Stacking Classifiers
voting_stacking_estimators = [
    (classifier_and_params[0].__class__.__name__, classifier_and_params[0])
    for classifier_and_params in classifers_list
]

# Voting Classifier
voting_ = VotingClassifier(estimators=voting_stacking_estimators)
voting_params = {
    'VotingClassifier__voting': ['soft', 'hard'],
    'VotingClassifier__weights': [None],
}

voting = [voting_, voting_params]

# Stacking Classifier
stacking_ = StackingClassifier(estimators=voting_stacking_estimators)
stacking_params = {
    'StackingClassifier__stack_method': ['auto', 'predict_proba', 'decision_function', 'predict'],
    'StackingClassifier__passthrough': [True, False],
}

stacking = [stacking_, stacking_params]

# Add stacking and voting classifiers to classifiers list and pipe dict
classifers_list.append(voting)
classifiers_pipe[voting[0].__class__.__name__] = voting
classifers_list.append(stacking)
classifiers_pipe[stacking[0].__class__.__name__] = stacking
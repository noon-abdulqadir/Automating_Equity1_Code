# %%
import os  # type:ignore # isort:skip # fmt:skip # noqa # nopep8
import sys  # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from pathlib import Path  # type:ignore # isort:skip # fmt:skip # noqa # nopep8

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
from setup_module.imports import *  # type:ignore # isort:skip # fmt:skip # noqa # nopep8

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
accelerator = Accelerator()
torch.autograd.set_detect_anomaly(True)
os.environ.get('TOKENIZERS_PARALLELISM')

# Plotting variables
pp = pprint.PrettyPrinter(indent=4)
tqdm.tqdm.pandas(desc='progress-bar')
tqdm_auto.tqdm.pandas(desc='progress-bar')
# tqdm.notebook.tqdm().pandas(desc='progress-bar')
tqdm_auto.notebook_tqdm().pandas(desc='progress-bar')
# pbar = progressbar.ProgressBar(maxval=10)
mpl.style.use(f'{code_dir}/setup_module/apa.mplstyle-main/apa.mplstyle')
mpl.rcParams['text.usetex'] = False
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
# ### Helper function to join model names and params into pipe params


def make_pipe_list(model, params):
    return [model, {f'{model.__class__.__name__}__{param_name}': param_value for param_name, param_value in params.items()}]

# %% [markdown]
# ## Vectorizers


# %%
# CountVectorizer
count_ = CountVectorizer()
count_params = {
    'analyzer': ['word'],
    'ngram_range': [(1, 3)],
    'lowercase': [True, False],
    'max_df': [0.85, 0.80, 0.75],
    'min_df': [0.15, 0.20, 0.25],
}
count = make_pipe_list(count_, count_params)

# TfidfVectorizer
tfidf_ = TfidfVectorizer()
tfidf_params = {
    'analyzer': ['word'],
    'ngram_range': [(1, 3)],
    'lowercase': [True, False],
    'use_idf': [True, False],
    'max_df': [0.85, 0.80, 0.75],
    'min_df': [ 0.15, 0.20, 0.25],
}
tfidf = make_pipe_list(tfidf_, tfidf_params)

# Vectorizers List
vectorizers_list = [
    count,
    tfidf
]

# BOW FeatureUnion
### BOW FeatureUnion
bow_ = FeatureUnion(
    transformer_list=[('CountVectorizer', count[0]), ('TfidfVectorizer', tfidf[0])]
)

bow_params = count[1] | tfidf[1]

bow = make_pipe_list(bow_, bow_params)

# Vectorizers List append bow
vectorizers_list.append([bow_.set_params(**{key: value[0] for key, value in bow_params.items()})])

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
    'score_func': [f_classif, chi2, f_regression],
    'k': ['all'],
}
selectkbest = make_pipe_list(selectkbest_, selectkbest_params)

# SelectPercentile
selectperc_ = SelectPercentile()
selectperc_params = {
    'score_func': [f_classif, chi2, f_regression],
    'percentile': [30, 40, 50, 60, 70, 80],
}
selectperc = make_pipe_list(selectperc_, selectperc_params)

# SelectFpr
selectfpr_ = SelectFpr()
selectfpr_params = {
    'score_func': [f_classif, chi2, f_regression],
}
selectfpr = make_pipe_list(selectfpr_, selectfpr_params)

# SelectFdr
selectfdr_ = SelectFdr()
selectfdr_params = {
    'score_func': [f_classif, chi2, f_regression],
}
selectfdr = make_pipe_list(selectfdr_, selectfdr_params)

# SelectFwe
selectfwe_ = SelectFwe()
selectfwe_params = {
    'score_func': [f_classif, chi2, f_regression],
}
selectfwe = make_pipe_list(selectfwe_, selectfwe_params)

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
    'random_state': [random_state],
    'tomek': [TomekLinks(sampling_strategy='majority', n_jobs=n_jobs)],
}
smotetomek = make_pipe_list(smotetomek_, smotetomek_params)

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
    'strategy': [
        'stratified',
        'most_frequent',
        'prior',
        'uniform',
    ],
    'random_state': [random_state],
}
dummy = make_pipe_list(dummy_, dummy_params)

# Multinomial Naive Bayes
nb_ = MultinomialNB()
nb_params = {
    'fit_prior': [True, False],
    'alpha': [0.1, 0.2, 0.3],
}
nb = make_pipe_list(nb_, nb_params)

# Bernoulli Naive Bayes
bnb_ = BernoulliNB()
bnb_params = {
    'fit_prior': [True, False],
    'alpha': [0.1, 0.2, 0.3],
}
bnb = make_pipe_list(bnb_, bnb_params)

# Gaussian Naive Bayes
gnb_ = GaussianNB()
gnb_params = {
    'var_smoothing': [1e-9],
}
gnb = make_pipe_list(gnb_, gnb_params)

# KNeighbors Classifier
knn_ = KNeighborsClassifier()
knn_params = {
    'weights': ['uniform', 'distance'],
    'n_neighbors': [5, 15, 30, 50],
    'algorithm': ['auto'],
    # 'p': [1, 2, 3, 4, 5],
    # 'metric': [
    #     'minkowski',
    #     'euclidean',
    #     'cosine',
    #     'correlation',
    # ],
    # 'leaf_size': [30, 50, 100, 200, 300, 500],
    # 'metric_params': [None, {'p': 2}, {'p': 3}],
}
knn = make_pipe_list(knn_, knn_params)

# Logistic Regression
lr_ = LogisticRegression()
lr_params = {
    'class_weight': [class_weight],
    'random_state': [random_state],
    'fit_intercept': [True, False],
    'multi_class': ['auto'],
    'solver': ['liblinear'],
    'C': [0.01, 0.5, 1, 5, 10, 15],
    'max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],
    # 'penalty': ['elasticnet'],
}
lr = make_pipe_list(lr_, lr_params)

# Passive Aggressive
pa_ = PassiveAggressiveClassifier()
pa_params = {
    'loss': ['squared_hinge'],
    'random_state': [random_state],
    'fit_intercept': [True, False],
    'class_weight': [class_weight],
    'shuffle': [True, False],
    'C': [0.01, 0.5, 1, 5, 10, 15],
    'average': [True, False],
    'max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],
}
pa = make_pipe_list(pa_, pa_params)

# Perceptron
ptron_ = linear_model.Perceptron()
ptron_params = {
    'penalty': ['elasticnet'],
    'random_state': [random_state],
    'fit_intercept': [True, False],
    'class_weight': [class_weight],
    'shuffle': [True, False],
    'max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],
}
ptron = make_pipe_list(ptron_, ptron_params)

# Stochastic Gradient Descent Aggressive
sgd_ = SGDClassifier()
sgd_params = {
    'loss': ['squared_hinge'],
    'random_state': [random_state],
    'fit_intercept': [True, False],
    'class_weight': [class_weight],
    'max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],
}
sgd = make_pipe_list(sgd_, sgd_params)

# SVM
svm_ = LinearSVC()
svm_params = {
    'loss': ['squared_hinge'],
    'random_state': [random_state],
    'fit_intercept': [True, False],
    'class_weight': [class_weight],
    'C': [0.01, 0.5, 1, 5, 10, 15],
    'max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],
    'dual': [False]
    # 'multi_class': ['ovr', 'crammer_singer'],
}
svm = make_pipe_list(svm_, svm_params)

# Decision Tree
dt_ = DecisionTreeClassifier()
dt_params = {
    'max_depth': [2, 5, 10],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'random_state': [random_state],
    'splitter': ['best', 'random'],
    'class_weight': [class_weight],
    # 'max_features': ['auto'],
}
dt = make_pipe_list(dt_, dt_params)

# Random Forest
rf_ = RandomForestClassifier()
rf_params = {
    'max_depth': [2, 5, 10],
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'random_state': [random_state],
    'class_weight': [class_weight],
    # 'oob_score': [True],
    # 'max_features': ['auto'],
}
rf = make_pipe_list(rf_, rf_params)

# Extra Trees
et_ = ExtraTreesClassifier()
et_params = {
    'max_depth': [2, 5, 10],
    'n_estimators': [50, 100, 150],
    'max_feature': ['auto'],
    'random_state': [random_state],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'class_weight': [class_weight],
}
et = make_pipe_list(et_, et_params)

# Gradient Boosting
gbc_ = GradientBoostingClassifier()
gbc_params = {
    'random_state': [random_state],
    'loss': ['log_loss', 'deviance', 'exponential'],
    # 'max_features': ['auto'],
}
gbc = make_pipe_list(gbc_, gbc_params)

# XGBoost
xgb_ = XGBClassifier()
xgb_params = {
    'seed': [random_state],
    'eval_metric': ['logloss'],
    'objective': ['binary:logistic'],
}
xgb = make_pipe_list(xgb_, xgb_params)

# MLP Classifier
mlpc_ = MLPClassifier()
mlpc_params = {
    'hidden_layer_sizes': [(100,), (50,), (25,), (10,), (5,), (1,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'random_state': [random_state],
    'max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],
}
mlpc = make_pipe_list(mlpc_, mlpc_params)

# MLP Regressor
mlpr_ = MLPRegressor()
mlpr_params = {
    'hidden_layer_sizes': [(100,), (50,), (25,), (10,), (5,), (1,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [400000, 500000, 600000, 700000, 800000, 900000, 1000000],
    'random_state': [random_state],
}
mlpr = make_pipe_list(mlpr_, mlpr_params)

# AdaBoostClassifier
ada_ = AdaBoostClassifier()
ada_params = {
    'random_state': [random_state],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'algorithm': ['SAMME', 'SAMME.R'],
    'estimator': [
        SVC(probability=True, kernel='linear'),
        LogisticRegression(),
        MultinomialNB(),
    ],
}
ada = make_pipe_list(ada_, ada_params)

# Classifiers List
classifier_ignore_list = [
    et, bnb, gnb, gbc, sgd,
]
classifiers_list = [
    dummy, knn, lr, svm, dt, rf, xgb, mlpc, mlpr, pa, ptron, et, bnb, gnb, gbc, sgd
]
classifiers_list_all = [
    classifier_and_params
    for classifier_and_params in classifiers_list
    if classifier_and_params not in classifier_ignore_list
]
classifiers_list_linear = [
    lr, svm, sgd, pa, ptron, mlpc, mlpr
]
classifiers_list_nonlinear = [
    classifier_and_params
    for classifier_and_params in classifiers_list_all
    if classifier_and_params not in classifiers_list_linear
]

# Classifiers Dict
# All
classifiers_pipe_all = {
    classifier_and_params[0].__class__.__name__: classifier_and_params
    for classifier_and_params in classifiers_list_all
}
## Linear
classifiers_pipe_linear = {
    classifier_and_params[0].__class__.__name__: classifier_and_params
    for classifier_and_params in classifiers_list_linear
}
# Nonlinear
classifiers_pipe_nonlinear = {
    classifier_and_params[0].__class__.__name__: classifier_and_params
    for classifier_and_params in classifiers_list_nonlinear
}

# Ensemble Classifiers
# Estimators for Ensemble Classifiers
voting_stacking_estimators = [
    # classifier_and_params[0].set_params(**{key.replace(f'{classifier_and_params[0].__class__.__name__}__', ''): value[0]
    # for key, value in classifier_and_params[1].items()})
    (classifier_and_params[0].__class__.__name__,
    classifier_and_params[0].set_params(**{key.replace(f'{classifier_and_params[0].__class__.__name__}__', ''): value[0]
    for key, value in classifier_and_params[1].items()}))
    for classifier_and_params in classifiers_list_all
    if hasattr(classifier_and_params[0], 'fit')
    and hasattr(classifier_and_params[0], 'predict')
    and hasattr(classifier_and_params[0], 'predict_proba')
    # and hasattr(classifier_and_params[0], 'decision_function')
    and classifier_and_params[0].__class__.__name__ != 'MLPRegressor'
    # and classifier_and_params[0].__class__.__name__ != 'MLPClassifier'
]

# Voting Classifier
voting_params = {
    'voting': ['soft', 'hard'],
    'weights': [None],
}
voting_ = VotingClassifier(estimators=voting_stacking_estimators)
voting = make_pipe_list(voting_, voting_params)

# Stacking Classifier
stacking_params = {
    'stack_method': ['auto', 'predict_proba', 'decision_function', 'predict'],
    'passthrough': [True, False],
}
stacking_ = StackingClassifier(estimators=voting_stacking_estimators)
stacking = make_pipe_list(stacking_, stacking_params)

# Ensemble Classifiers
classifiers_list_ensemble = [
    voting, stacking
]
classifiers_pipe_ensemble = {
    classifier_and_params[0].__class__.__name__: classifier_and_params
    for classifier_and_params in classifiers_list_ensemble
}
# %%

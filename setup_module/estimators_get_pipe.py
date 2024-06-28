# %%
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

# %%
import evaluate
from accelerate import Accelerator
import transformers
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
    GPTNeoXTokenizerFast,
    LlamaConfig,
    LlamaForSequenceClassification,
    LlamaTokenizer,
    LlamaTokenizerFast,
    MegatronBertForSequenceClassification,
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

accelerator = Accelerator()


# %% [markdown]
# ### READ DATA

# %%
# Variables
# Sklearn Variables
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

# Transformer variables
max_length = 512
returned_tensor = 'pt'
cpu_counts = torch.multiprocessing.cpu_count()
device = torch.device('mps') if torch.has_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available(
) else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device_name = str(device.type)
print(f'Using {device_name.upper()}')
accelerator = Accelerator()
torch.autograd.set_detect_anomaly(True)
os.environ.get('TOKENIZERS_PARALLELISM')
os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
os.environ.get('TRANSFORMERS_CACHE')
openai_token = os.environ['OPENAI_API_KEY']
huggingface_token = os.environ['HUGGINGFACE_API_KEY']
# load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4'
quantization_config_dict = {
    'load_in_8bit': True,
    'llm_int8_skip_modules': ['lm_head'],
}
skip_fitted_estimators = True
evaluate_estimator_on_concat = False
hyperparameter_tuning = False
shap.initjs()


# %% [markdown]
# Supervised Pipeline
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

# Vectorizers Dict
vectorizers_pipe = {
    vectorizer_and_params[0].__class__.__name__: vectorizer_and_params
    for vectorizer_and_params in vectorizers_list
}

# BOW FeatureUnion
### BOW FeatureUnion
bow_ = FeatureUnion(
    transformer_list=[('CountVectorizer', count[0]), ('TfidfVectorizer', tfidf[0])]
)
bow_params = {**count[1], **tfidf[1]}
bow = make_pipe_list(bow_, bow_params)

# Vectorizers List and Dict append bow
vectorizers_list.append(bow_.set_params(**{key: value[0] for key, value in bow_params.items()}))
vectorizers_pipe[bow[0].__class__.__name__] = bow


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
    'max_iter': [700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000],
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
    'max_iter': [700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000],
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
    'max_iter': [700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000],
}
ptron = make_pipe_list(ptron_, ptron_params)

# Stochastic Gradient Descent Aggressive
sgd_ = SGDClassifier()
sgd_params = {
    'loss': ['squared_hinge'],
    'random_state': [random_state],
    'fit_intercept': [True, False],
    'class_weight': [class_weight],
    'max_iter': [700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000],
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
    'max_iter': [700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000],
    'dual': [False]
}
svm = make_pipe_list(svm_, svm_params)

# SVC
svc_ = SVC()
svc_params = {
    'random_state': [random_state],
    'class_weight': [class_weight],
    'C': [0.01, 0.5, 1, 5, 10, 15],
    'max_iter': [700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000],
}
svc = make_pipe_list(svc_, svc_params)

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
    'loss': ['log_loss', 'log_loss', 'exponential'],
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
    'solver': ['sgd', 'adam'],#['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'random_state': [random_state],
    'max_iter': [700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000],
}
mlpc = make_pipe_list(mlpc_, mlpc_params)

# Classifiers List
classifier_ignore_list = [
    svc, et, bnb, gnb, ptron, #svm, mlpc, pa, gbc
]
classifiers_list_all = [
    dummy, nb, knn, lr, dt, rf, svm, svc, xgb, mlpc, pa, ptron, sgd, gbc, et, bnb, gnb,
]
classifiers_list = [
    classifier_and_params
    for classifier_and_params in classifiers_list_all
    if classifier_and_params not in classifier_ignore_list
]
classifiers_list_linear = [
    lr, nb, svm, svc, sgd, mlpc, pa, ptron, gbc,
]
classifiers_list_linear = [
    classifier_and_params
    for classifier_and_params in classifiers_list_all
    if classifier_and_params not in classifier_ignore_list
]
classifiers_list_nonlinear = [
    classifier_and_params
    for classifier_and_params in classifiers_list_all
    if classifier_and_params not in classifier_ignore_list
    and classifier_and_params not in classifiers_list_linear
]

# Classifiers Dict
# All
classifiers_pipe = {
    classifier_and_params[0].__class__.__name__: classifier_and_params
    for classifier_and_params in classifiers_list
    if classifier_and_params not in classifier_ignore_list
}
## Linear
classifiers_pipe_linear = {
    classifier_and_params[0].__class__.__name__: classifier_and_params
    for classifier_and_params in classifiers_list_linear
    if classifier_and_params not in classifier_ignore_list
}
# Nonlinear
classifiers_pipe_nonlinear = {
    classifier_and_params[0].__class__.__name__: classifier_and_params
    for classifier_and_params in classifiers_list_nonlinear
    if classifier_and_params not in classifier_ignore_list
}

# Ensemble Classifiers
ada_params = {
    'random_state': [random_state],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'algorithm': ['SAMME', 'SAMME.R'],
}
bagging_params = {
    'random_state': [random_state],
    'n_estimators': [50, 100, 150],
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0],
}
# Voting Classifier
## Estimators for VotingClassifier
voting_estimators = [
    (classifier_and_params[0].__class__.__name__,
    classifier_and_params[0].set_params(**{key.replace(f'{classifier_and_params[0].__class__.__name__}__', ''): value[0]
    for key, value in classifier_and_params[1].items()}))
    for classifier_and_params in classifiers_list
    if classifier_and_params not in classifier_ignore_list
    and classifier_and_params[0].__class__.__name__ != 'DummyClassifier'
    and hasattr(classifier_and_params[0], 'fit')
    and hasattr(classifier_and_params[0], 'predict')
    and hasattr(classifier_and_params[0], 'predict_proba')
]
voting_ = VotingClassifier(estimators=voting_estimators)
voting_params = {
    'voting': ['soft'],
}
voting = make_pipe_list(voting_, voting_params)

# AdaBoostClassifier for VotingClassifier
ada_voting_estimators = [
    classifier_and_params
    for classifier_and_params in voting_estimators
    if classifier_and_params[0] != 'DummyClassifier'
    and 'sample_weight' in inspect.getfullargspec(classifier_and_params[1].fit)[0]
]
ada_voting_estimators_params = {
    key.replace(f'{voting[0].__class__.__name__}__', ''): value[0]
    for key, value in voting[1].items()
}
ada_voting_ = AdaBoostClassifier(estimator=VotingClassifier(ada_voting_estimators).set_params(**ada_voting_estimators_params))
ada_voting_params = {
    'random_state': [random_state],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'algorithm': ['SAMME', 'SAMME.R'],
}
ada_voting = make_pipe_list(ada_voting_, ada_voting_params)

# BaggingClassifier for VotingClassifier
bagging_voting_estimators = [
    classifier_and_params
    for classifier_and_params in voting_estimators
    if classifier_and_params[0] != 'DummyClassifier'
    and 'sample_weight' in inspect.getfullargspec(classifier_and_params[1].fit)[0]
]
bagging_voting_estimators_params = {
    key.replace(f'{voting[0].__class__.__name__}__', ''): value[0]
    for key, value in voting[1].items()
}
bagging_voting_ = BaggingClassifier(estimator=VotingClassifier(bagging_voting_estimators).set_params(**bagging_voting_estimators_params))
bagging_voting = make_pipe_list(bagging_voting_, bagging_params)

# Stacking Classifier
## Estimators for VotingClassifier
stacking_estimators = [
    (classifier_and_params[0].__class__.__name__,
    classifier_and_params[0].set_params(**{key.replace(f'{classifier_and_params[0].__class__.__name__}__', ''): value[0]
    for key, value in classifier_and_params[1].items()}))
    for classifier_and_params in classifiers_list
    if classifier_and_params not in classifier_ignore_list
    and classifier_and_params[0].__class__.__name__ != 'DummyClassifier'
    and hasattr(classifier_and_params[0], 'predict_proba')
    and hasattr(classifier_and_params[0], 'decision_function')
]
stacking_ = StackingClassifier(estimators=stacking_estimators)
stacking_params = {
    'stack_method': ['auto', 'predict_proba', 'decision_function', 'predict'],
    'passthrough': [True, False],
}
stacking = make_pipe_list(stacking_, stacking_params)

# AdaBoostClassifier for StackingClassifier
ada_stacking_estimators = [
    classifier_and_params
    for classifier_and_params in stacking_estimators
    if classifier_and_params[0] != 'DummyClassifier' and 'sample_weight' in inspect.getfullargspec(classifier_and_params[1].fit)[0]
]
ada_stacking_estimators_params = {
    key.replace(f'{stacking[0].__class__.__name__}__', ''): value[0]
    for key, value in stacking[1].items()
}
ada_stacking_ = AdaBoostClassifier(estimator=StackingClassifier(ada_stacking_estimators).set_params(**ada_stacking_estimators_params))
ada_stacking = make_pipe_list(ada_stacking_, ada_params)

# BaggingClassifier for StackingClassifier
bagging_stacking_estimators = [
    classifier_and_params
    for classifier_and_params in stacking_estimators
    if classifier_and_params[0] != 'DummyClassifier'
    and 'sample_weight' in inspect.getfullargspec(classifier_and_params[1].fit)[0]
]
bagging_stacking_estimators_params = {
    key.replace(f'{stacking[0].__class__.__name__}__', ''): value[0]
    for key, value in stacking[1].items()
}
bagging_stacking_ = BaggingClassifier(estimator=StackingClassifier(bagging_stacking_estimators).set_params(**bagging_stacking_estimators_params))
bagging_stacking_params = {
    'random_state': [random_state],
    'n_estimators': [50, 100, 150],
    'max_samples': [0.5, 0.75, 1],
    'max_features': [0.5, 0.75, 1],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
}
bagging_stacking = make_pipe_list(bagging_stacking_, bagging_params)

# Ensemble Classifiers
classifiers_list_ensemble = [
    voting,
    stacking,
    ada_voting,
    # ada_stacking,
    bagging_voting,
    # bagging_stacking,
]
classifiers_pipe_ensemble = {
    classifier_and_params[0].__class__.__name__: classifier_and_params
    for classifier_and_params in classifiers_list_ensemble
}

# Add Ensemble Classifiers to Classifiers List and Dict
classifiers_list.extend(classifiers_list_ensemble)
classifiers_pipe |= classifiers_pipe_ensemble

# %% [markdown]
# Transformers Pipeline
transformers_pipe = {
    'BertForSequenceClassification': {
        'model_name': 'bert-base-uncased',
        'config': BertConfig,
        'tokenizer': BertTokenizerFast,
        'model': BertForSequenceClassification,
    },
    'GPT2ForSequenceClassification': {
        'model_name': 'gpt2',
        'config': GPT2Config,
        'tokenizer': GPT2TokenizerFast,
        'model': GPT2ForSequenceClassification,
    },
    'OpenAIGPTForSequenceClassification': {
        'model_name': 'openai-gpt',
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizerFast,
        'model': OpenAIGPTForSequenceClassification,
    },
    # 'FlanT5ForSequenceClassification': {
    #     'model_name': 'google/flan-t5-base',
    #     'config': AutoConfig,
    #     'tokenizer': AutoTokenizer,
    #     'model': AutoModelForSequenceClassification,
    # },
    # 'GPTJForSequenceClassification': {
    #     'model_name': 'EleutherAI/gpt-j-6b',
    #     'config': GPTJConfig,
    #     'tokenizer': AutoTokenizer,
    #     'model': GPTJForSequenceClassification,
    # },
    # 'GPTNeoXForSequenceClassification': {
    #     'model_name': 'EleutherAI/gpt-neox-20b',
    #     'config': GPTNeoXConfig,
    #     'tokenizer': GPTNeoXTokenizerFast,
    #     'model': GPTNeoXForSequenceClassification,
    # },
    # 'LlamaForSequenceClassification': {
    #     'model_name': 'meta-llama/Llama-2-7b',
    #     'config': LlamaConfig,
    #     'tokenizer': LlamaTokenizerFast,
    #     'model': LlamaForSequenceClassification,
    # },
    # 'PALM': {
    #     'model_name': 'conceptofmind/palm-1b',
    #     'config': AutoConfig,
    #     'tokenizer': AutoTokenizer,
    #     'model': AutoModelForSequenceClassification,
    # },
    # 'MegatronBertForSequenceClassification': {
    #     'model_name': 'nvidia/megatron-bert-uncased-345m',
    #     'config': BertConfig,
    #     'tokenizer': BertTokenizerFast,
    #     'model': MegatronBertForSequenceClassification,
    # },
    # 'Falcon': {
    #     'model_name': 'tiiuae/falcon-40b',
    #     'config': AutoConfig,
    #     'tokenizer': AutoTokenizer,
    #     'model': AutoModelForSequenceClassification,
    # },
}

# %%

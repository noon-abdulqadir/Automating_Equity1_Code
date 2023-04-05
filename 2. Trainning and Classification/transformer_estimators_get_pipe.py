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
transformer_model_names = ['bert-base-uncased', 'ydshieh/tiny-random-gptj-for-sequence-classification']


# %% [markdown]
# Transformer Dict

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
    # 'GPTJForSequenceClassification': {
    #     'model_name': 'ydshieh/tiny-random-gptj-for-sequence-classification',
    #     'config': GPTJConfig,
    #     'tokenizer': AutoTokenizer,
    #     'model': GPTJForSequenceClassification,
    # },
    # 'OpenAIGPTForSequenceClassification': {
    #     'model_name': 'openai-gpt',
    #     'config': OpenAIGPTConfig,
    #     'tokenizer': OpenAIGPTTokenizerFast,
    #     'model': OpenAIGPTForSequenceClassification,
    # },
}

# %%

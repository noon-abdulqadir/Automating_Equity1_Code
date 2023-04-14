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
from estimators_get_pipe import * # type:ignore # isort:skip # fmt:skip # noqa # nopep8


# %%
# Set random seed
random_state = 42
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
cores = multiprocessing.cpu_count()

# Transformer variables
method = 'Transformers'
results_save_path = f'{models_save_path}{method} Results/'
with open(f'{data_dir}{method}_results_save_path.txt', 'w') as f:
    f.write(results_save_path)
if not os.path.exists(results_save_path):
    os.makedirs(results_save_path)
done_xy_save_path = f'{results_save_path}Search+Xy/'
with open(f'{data_dir}{method}_done_xy_save_path.txt', 'w') as f:
    f.write(done_xy_save_path)
if not os.path.exists(done_xy_save_path):
    os.makedirs(done_xy_save_path)

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
    'Mean Cross Validation Train Score': np.nan,
    f'Mean Cross Validation Train - {scoring.title()}': np.nan,
    f'Mean Explained Train Variance - {scoring.title()}': np.nan,
    'Mean Cross Validation Test Score': np.nan,
    f'Mean Cross Validation Test - {scoring.title()}': np.nan,
    f'Mean Explained Test Variance - {scoring.title()}': np.nan,
    'Explained Variance': np.nan,
    'Accuracy': np.nan,
    'Balanced Accuracy': np.nan,
    'Precision': np.nan,
    'Recall': np.nan,
    'F1-score': np.nan,
    'Matthews Correlation Coefficient': np.nan,
    'Fowlkes–Mallows Index': np.nan,
    'R2 Score': np.nan,
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
    'Normalized Confusion Matrix': np.nan
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
best_trial_args = [
    'num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size', 'learning_rate', 'warmup_steps', 'weight_decay'
]
training_args_dict = {
    'seed': random_state,
    # 'resume_from_checkpoint': True,
    # 'overwrite_output_dir': True,
    'logging_steps': 100,
    'evaluation_strategy': 'epoch',
    'save_strategy': 'epoch',
    # 'torch_compile': bool(transformers.file_utils.is_torch_available()),
    'use_mps_device': bool(device_name == 'mps' and torch.backends.mps.is_available()),
    'optim': 'adamw_torch',
    # 'save_total_limit': 1,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'recall',
    'label_names':[0, 1],
    # The below metrics are used by hyperparameter search
    'num_train_epochs': 3,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 20,
    'learning_rate': 5e-5,
    'warmup_steps': 100,
    'weight_decay': 0.01,
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
# # Functions
#

# %%
def get_existing_files(
    results_save_path= results_save_path,
    estimator_names_list=None,
    vectorizer_names_list=None,
    classifier_names_list=None,
):
    if estimator_names_list is None:
        estimator_names_list = []

    print(f'Searching for existing estimators in directory:\n{results_save_path}')

    for estimators_file in tqdm.tqdm(glob.glob(f'{results_save_path}*.pkl')):
        if f'{method} Estimator - ' in estimators_file:

            col=estimators_file.split(f'{method} Estimator - ')[-1].split(' - ')[0]
            vectorizer_name=estimators_file.split(f'{col} - ')[-1].split(' + ')[0]
            classifier_name=estimators_file.split(f'{vectorizer_name} + ')[-1].split(' (Save_protocol=')[0]

            estimator_names_list.append(f'{col} - {vectorizer_name} + {classifier_name}')

    return (
        list(set(estimator_names_list))
    )


# %%
# Function to place Xy and CV data in df and save
def save_Xy(
    X_train, y_train,
    X_test, y_test,
    X_val, y_val,
    col,
    results_save_path=results_save_path,
    method=method, done_xy_save_path=done_xy_save_path,
    compression=None, protocol=None, path_suffix=None, data_dict=None
):
    if compression is None:
        compression = False
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    if path_suffix is None:
        path_suffix = f' - {col} - (Save_protocol={protocol}).pkl'
    if data_dict is None:
        data_dict = {}

    # Check data
    check_consistent_length(X_train, y_train)
    check_consistent_length(X_test, y_test)
    check_consistent_length(X_val, y_val)

    # Make df_train_data
    df_train_data = pd.DataFrame(
        {
            'X_train': X_train,
            'y_train': y_train,
        },
    )
    # Make df_test_data
    df_test_data = pd.DataFrame(
        {
            'X_test': X_test,
            'y_test': y_test,
        },
    )
    # Make df_test_data
    df_val_data = pd.DataFrame(
        {
            'X_val': X_val,
            'y_val': y_val,
        },
    )

    # Assign dfs to variables
    data_dict['df_train_data'] = df_train_data
    data_dict['df_test_data'] = df_test_data
    data_dict['df_val_data'] = df_val_data

    # Save files
    print('='*20)
    for file_name, file_ in data_dict.items():
        save_path = f'{models_save_path}{file_name}{path_suffix}'
        print(f'Saving Xy {file_name} at {save_path}')
        file_.to_pickle(
            save_path, protocol=protocol
        )
    print(f'Done saving Xy!\n{list(data_dict.keys())}')
    print('='*20)


# %%
def get_class_weights(
    X_train, y_train,
    X_test, y_test,
    X_val, y_val,
):
    # Get train class weights
    train_class_weights = compute_class_weight(class_weight = class_weight, classes = np.unique(y_train), y = y_train)
    train_class_weights_ratio = train_class_weights[0]/train_class_weights[1]
    train_class_weights_dict = dict(zip(np.unique(y_train), train_class_weights))

    # Get train class weights
    test_class_weights = compute_class_weight(class_weight = class_weight, classes = np.unique(y_train), y = y_test)
    test_class_weights_ratio = test_class_weights[0]/test_class_weights[1]
    test_class_weights_dict = dict(zip(np.unique(y_test), test_class_weights))

    # Get val class weights
    val_class_weights = compute_class_weight(class_weight = class_weight, classes = np.unique(y_train), y = y_val)
    val_class_weights_ratio = val_class_weights[0]/val_class_weights[1]
    val_class_weights_dict = dict(zip(np.unique(y_val), val_class_weights))

    return (
        train_class_weights, train_class_weights_ratio, train_class_weights_dict,
        test_class_weights, test_class_weights_ratio, test_class_weights_dict,
        val_class_weights, val_class_weights_ratio, val_class_weights_dict,
    )

# %%
def print_Xy(
    X_train, y_train,
    X_test, y_test,
    X_val, y_val,
    train_class_weights, train_class_weights_ratio, train_class_weights_dict,
    test_class_weights, test_class_weights_ratio, test_class_weights_dict,
    val_class_weights, val_class_weights_ratio, val_class_weights_dict,
):
    # Check for consistent length
    check_consistent_length(X_train, y_train)
    check_consistent_length(X_test, y_test)
    check_consistent_length(X_val, y_val)

    print('Done splitting data into training and testing sets.')
    print('='*20)
    print(f'Training set shape: {y_train.shape}')
    print('-'*10)
    print(f'Training set example:\n{X_train[0]}')
    print('~'*10)
    print(f'Testing set shape: {y_test.shape}')
    print('-'*10)
    print(f'Testing set example:\n{X_test[0]}')
    print('~'*10)
    print(f'Validation set shape: {y_val.shape}')
    print('-'*10)
    print(f'Validation set example:\n{X_val[0]}')
    print('~'*10)
    print(f'Training data class weights:\nRatio = {train_class_weights_ratio:.2f} (0 = {train_class_weights[0]:.2f}, 1 = {train_class_weights[1]:.2f})')
    print('-'*10)
    print(f'Testing data class weights:\nRatio = {test_class_weights_ratio:.2f} (0 = {test_class_weights[0]:.2f}, 1 = {test_class_weights[1]:.2f})')
    print('-'*10)
    print(f'Validation data class weights:\nRatio = {val_class_weights_ratio:.2f} (0 = {val_class_weights[0]:.2f}, 1 = {val_class_weights[1]:.2f})')
    print('='*20)


# %%
def split_data(df, col, analysis_columns=analysis_columns, text_col=text_col):

    train_ratio = 0.75
    test_ratio = 0.10
    validation_ratio = 0.15
    test_split = test_size = 1 - train_ratio
    validation_split = test_ratio / (test_ratio + validation_ratio)

    # Split
    print('='*20)
    print('Splitting data into training, testing, and validation sets:')
    print(f'Ratios: train_size = {train_ratio}, test size = {test_ratio}, validation size = {validation_ratio}')

    df = df.dropna(subset=analysis_columns, how='any')
    df = df.loc[df[text_col].str.len() >= 5]
    print(f'DF length: {len(df)}')

    train, test = train_test_split(
        df, train_size=1-test_split, test_size=test_split, random_state=random_state
    )
    val, test = train_test_split(
        test, test_size=validation_split, random_state=random_state
    )

    X_train = np.array(list(train[text_col].astype('str').values))
    y_train = column_or_1d(train[col].astype('int64').values.tolist(), warn=True)

    X_test = np.array(list(test[text_col].astype('str').values))
    y_test = column_or_1d(test[col].astype('int64').values.tolist(), warn=True)

    X_val = np.array(list(val[text_col].astype('str').values))
    y_val = column_or_1d(val[col].astype('int64').values.tolist(), warn=True)

    # Get class weights
    (
        train_class_weights, train_class_weights_ratio, train_class_weights_dict,
        test_class_weights, test_class_weights_ratio, test_class_weights_dict,
        val_class_weights, val_class_weights_ratio, val_class_weights_dict,
    ) = get_class_weights(
        X_train, y_train,
        X_test, y_test,
        X_val, y_val,
    )
    # Print info
    print_Xy(
        X_train, y_train,
        X_test, y_test,
        X_val, y_val,
        train_class_weights, train_class_weights_ratio, train_class_weights_dict,
        test_class_weights, test_class_weights_ratio, test_class_weights_dict,
        val_class_weights, val_class_weights_ratio, val_class_weights_dict,
    )

    return (
        train, X_train, y_train,
        test, X_test, y_test,
        val, X_val, y_val,
        train_class_weights, train_class_weights_ratio, train_class_weights_dict,
        test_class_weights, test_class_weights_ratio, test_class_weights_dict,
        val_class_weights, val_class_weights_ratio, val_class_weights_dict,
    )


# %%
class ToDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, encoded):
        self.encodings = encodings
        self.encoded = encoded

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], device=device).clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.encoded[idx], device=device).clone().detach()
        return item

    def __len__(self):
        return len(self.encoded)


# %%
def print_Xy_encodings(
    X_train, y_train, train_dataset,
    X_test, y_test, test_dataset,
    X_val, y_val, val_dataset,
):
    # Check for consistent length
    check_consistent_length(X_train, y_train, train_dataset)
    check_consistent_length(X_test, y_test, test_dataset)
    check_consistent_length(X_val, y_val, val_dataset)

    # Check encodings
    assert all(y_train == train_dataset.encoded), 'y_train and train_dataset encoded are not the same'
    assert all(y_test == test_dataset.encoded), 'y_test and test_dataset encoded are not the same'

    print('Done encoding training, testing, and validation sets.')
    print('='*20)
    print(f'Training set encodings example:\n{" ".join(train_dataset.encodings[0].tokens[:30])}')
    print('~'*10)
    print(f'Testing set encodings example:\n{" ".join(test_dataset.encodings[0].tokens[:30])}')
    print('~'*10)
    print(f'Validation set encodings example:\n{" ".join(val_dataset.encodings[0].tokens[:30])}')
    print('='*20)


# %%
def encode_data(
    X_train, y_train,
    X_test, y_test,
    X_val, y_val,
    tokenizer,
):
    print('='*20)
    print(f'Encoding training, testing, and validation sets with {f"{tokenizer=}".split("=")[0]}.from_pretrained using {tokenizer.name_or_path}.')

    X_train_encodings = tokenizer(
        X_train.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors=returned_tensor
    ).to(device)
    train_dataset = ToDataset(X_train_encodings, y_train)

    X_test_encodings = tokenizer(
        X_test.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors=returned_tensor
    ).to(device)
    test_dataset = ToDataset(X_test_encodings, y_test)

    X_val_encodings = tokenizer(
        X_val.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors=returned_tensor
    ).to(device)
    val_dataset = ToDataset(X_val_encodings, y_val)

    # Print info
    print_Xy_encodings(
        X_train, y_train, train_dataset,
        X_test, y_test, test_dataset,
        X_val, y_val, val_dataset,
    )

    return (
        X_train_encodings, train_dataset,
        X_test_encodings, test_dataset,
        X_val_encodings, val_dataset,
    )


# %%
def load_Xy(
    col,
    results_save_path=results_save_path, method=method,
    data_dict=None, protocol=None, path_suffix=None,
):
    if data_dict is None:
        data_dict = {}
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    if path_suffix is None:
        path_suffix = f' - {col} - (Save_protocol={protocol}).pkl'

    print('+'*30)
    print(f'{"="*10} Loading Xy from previous for {col} {"="*10}')
    print('+'*30)
    # Read all dfs
    for file_path in glob.glob(f'{models_save_path}*{path_suffix}'):
        file_name = file_path.split(f'{models_save_path}')[-1].split(path_suffix)[0]
        print(f'Loading {file_name} from {file_path}')
        if path_suffix in file_path and 'df_' in file_name and 'cv_results' not in file_name:
            data_dict[file_name] = pd.read_pickle(file_path)

    # Train data
    df_train_data = data_dict['df_train_data']
    X_train = df_train_data['X_train'].values
    y_train = df_train_data['y_train'].values
    # Test data
    df_test_data = data_dict['df_test_data']
    X_test = df_test_data['X_test'].values
    y_test = df_test_data['y_test'].values
    # Val data
    df_val_data = data_dict['df_val_data']
    X_val = df_val_data['X_val'].values
    y_val = df_val_data['y_val'].values

    print(f'Done loading Xy from previous for {col}!')

    # Get class weights
    (
        train_class_weights, train_class_weights_ratio, train_class_weights_dict,
        test_class_weights, test_class_weights_ratio, test_class_weights_dict,
        val_class_weights, val_class_weights_ratio, val_class_weights_dict,
    ) = get_class_weights(
        X_train, y_train,
        X_test, y_test,
        X_val, y_val,
    )

    # Print info
    print_Xy(
        X_train, y_train,
        X_test, y_test,
        X_val, y_val,
        train_class_weights, train_class_weights_ratio, train_class_weights_dict,
        test_class_weights, test_class_weights_ratio, test_class_weights_dict,
        val_class_weights, val_class_weights_ratio, val_class_weights_dict,
    )
    return (
        X_train, y_train,
        X_test, y_test,
        X_val, y_val,
        train_class_weights, train_class_weights_ratio, train_class_weights_dict,
        test_class_weights, test_class_weights_ratio, test_class_weights_dict,
        val_class_weights, val_class_weights_ratio, val_class_weights_dict,
    )


# %%
# Optuna hyperparameter tuning
# https://huggingface.co/docs/transformers/v4.27.2/en/hpo_train#hyperparameter-search-using-trainer-api
def optuna_hp_space(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
        'num_train_epochs': trial.suggest_int('num_train_epochs', 1, 5),
        'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16, 32, 64, 128]),
        'per_device_eval_batch_size': trial.suggest_categorical('per_device_eval_batch_size', [4, 8, 16, 32, 64, 128]),
        'warmup_steps': trial.suggest_int('warmup_steps', 0, 500),
        'weight_decay': trial.suggest_float('weight_decay', 1e-12, 1e-1, log=True),
    }

# %%
# Compute objective for hyperparameter tuning
# https://github.com/huggingface/transformers/issues/13019
def compute_objective(metrics_dict):
    return metrics_dict['recall']

# %%
def compute_metrics_with_y_pred(
    y_labels, y_pred,
    pos_label=None, labels=None, zero_division=None, alpha=None
):
    if pos_label is None:
        pos_label = 1
    if labels is None:
        labels = np.unique(y_pred)
    if zero_division is None:
        zero_division = 0
    if alpha is None:
        alpha = 0.1

    print('Computing metrics using y_pred.')
    # Using y_pred
    explained_variance = metrics.explained_variance_score(y_labels, y_pred)
    accuracy = metrics.accuracy_score(y_labels, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_labels, y_pred)
    precision = metrics.precision_score(y_labels, y_pred, pos_label=pos_label, labels=labels, zero_division=zero_division)
    recall = metrics.recall_score(y_labels, y_pred, pos_label=pos_label, labels=labels, zero_division=zero_division)
    f1 = metrics.f1_score(y_labels, y_pred, pos_label=pos_label,labels=labels, zero_division=zero_division)
    mcc = metrics.matthews_corrcoef(y_labels, y_pred)
    fm = metrics.fowlkes_mallows_score(y_labels, y_pred)
    r2 = metrics.r2_score(y_labels, y_pred)
    kappa = metrics.cohen_kappa_score(y_labels, y_pred, labels=labels)
    gmean_iba = imblearn.metrics.make_index_balanced_accuracy(alpha=alpha, squared=True)(geometric_mean_score)
    gmean = gmean_iba(y_labels, y_pred)
    report = metrics.classification_report(y_labels, y_pred, labels=labels, zero_division=zero_division)
    imblearn_report = classification_report_imbalanced(y_labels, y_pred, labels=labels, zero_division=zero_division)
    cm = metrics.confusion_matrix(y_labels, y_pred, labels=labels)
    cm_normalized = metrics.confusion_matrix(y_labels, y_pred, normalize='true', labels=labels)

    return (
        explained_variance, accuracy, balanced_accuracy, precision,
        recall, f1, mcc, fm, r2, kappa, gmean, report, imblearn_report, cm, cm_normalized
    )


# %%
def compute_metrics_with_y_pred_prob(
    y_labels, y_pred_prob,
    pos_label=None
):
    if pos_label is None:
        pos_label = 1

    print('Computing metrics using y_pred_prob.')
    average_precision = metrics.average_precision_score(y_labels, y_pred_prob)
    roc_auc = metrics.roc_auc_score(y_labels, y_pred_prob)
    fpr, tpr, threshold = metrics.roc_curve(y_labels, y_pred_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    loss = metrics.log_loss(y_labels, y_pred_prob)
    precision_pr, recall_pr, threshold_pr = metrics.precision_recall_curve(y_labels, y_pred_prob, pos_label=1)

    return (
        average_precision, roc_auc, auc,
        fpr, tpr, threshold, loss,
        precision_pr, recall_pr, threshold_pr
    )


# %%
def compute_metrics_all(
    y_labels, y_pred, y_pred_prob
):
    # Get metrics
    print('='*20)
    # Using y_pred
    if y_pred:
        print('-'*20)
        (
            explained_variance, accuracy, balanced_accuracy, precision,
            recall, f1, mcc, fm, r2, kappa, gmean, report, imblearn_report, cm, cm_normalized
        ) = compute_metrics_with_y_pred(
            y_labels, y_pred
        )
    # Using y_pred_prob
    if y_pred_prob:
        print('-'*20)
        (
            average_precision, roc_auc, auc,
            fpr, tpr, threshold, loss,
            precision_pr, recall_pr, threshold_pr
        ) = compute_metrics_with_y_pred_prob(
            y_labels, y_pred_prob
        )

    # Place metrics into dict
    print('-'*20)
    print('Appending metrics to dict.')
    metrics_dict = {
        # f'{scoring.title()} Best Score': float(best_train_score),
        # f'{scoring.title()} Best Threshold': threshold,
        # 'Train - Mean Cross Validation Score': float(cv_train_scores),
        # f'Train - Mean Cross Validation - {scoring.title()}': float(cv_train_recall),
        # f'Train - Mean Explained Variance - {scoring.title()}': float(cv_train_explained_variance_recall),
        # 'Test - Mean Cross Validation Score': float(cv_test_scores),
        # f'Test - Mean Cross Validation - {scoring.title()}': float(cv_test_recall),
        # f'Test - Mean Explained Variance - {scoring.title()}': float(cv_test_explained_variance_recall),
        'Explained Variance': float(explained_variance),
        'Accuracy': float(accuracy),
        'Balanced Accuracy': float(balanced_accuracy),
        'Precision': float(precision),
        'Average Precision': float(average_precision),
        'Recall': float(recall),
        'F1-score': float(f1),
        'Matthews Correlation Coefficient': float(mcc),
        'Fowlkes–Mallows Index': float(fm),
        'R2 Score': float(r2),
        'ROC': float(roc_auc),
        'AUC': float(auc),
        'Log Loss/Cross Entropy': float(loss),
        'Cohen’s Kappa': float(kappa),
        'Geometric Mean': float(gmean),
        'Classification Report': report,
        'Imbalanced Classification Report': imblearn_report,
        'Confusion Matrix': cm,
        'Normalized Confusion Matrix': cm_normalized,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
    }

    return metrics_dict


# %%
def clean_metrics_dict(metrics_dict, prefix_to_remove):
    for metric_name in list(metrics_dict):
        if metric_name.startswith(prefix_to_remove):
            new_metric_name = ' '.join(metric_name.split(prefix_to_remove)[-1].split('_')).strip()
        if not new_metric_name[0].isupper():
            new_metric_name = new_metric_name.title()
        if new_metric_name == 'Loss':
            metrics_dict['Log Loss/Cross Entropy'] = metrics_dict.pop(metric_name)
        else:
            metrics_dict[new_metric_name] = metrics_dict.pop(metric_name)

    return metrics_dict


# %%
# Function to get y_pred and y_pred_prob
def preprocess_logits_for_metrics_y_pred_prob(y_pred_logits, y_labels):

    print('-'*20)
    print(f'Preprocessing y_pred logits and labels for {col}:')
    print('-'*20)

    if isinstance(y_pred_logits, tuple):
        y_pred_logits = y_pred_logits[0]

    if not torch.is_tensor(y_pred_logits):
        y_pred_logits_tensor = torch.tensor(y_pred_logits, device=device)
    else:
        y_pred_logits_tensor = y_pred_logits.to(device)

    print(f'y_pred_logits shape: {y_pred_logits_tensor.shape}, {y_pred_logits_tensor.dtype}')
    print('-'*20)

    # Get y_pred_prob
    # https://stackoverflow.com/questions/34240703/what-are-logits-what-is-the-difference-between-softmax-and-softmax-cross-entrop
    print('-'*20)
    print('Getting y_pred_prob through softmax of y_pred_logits.')
    try:
        y_pred_prob_array = torch.nn.functional.softmax(y_pred_logits_tensor, dim=-1).clone().detach().cpu().numpy()
        print('Using torch.nn.functional.softmax.')
    except Exception:
        y_pred_prob_array = scipy.special.softmax(y_pred_logits, axis=-1)
        print('Using scipy.special.softmax.')

    print(f'y_pred_prob_array shape: {y_pred_prob_array.shape}, {y_pred_prob_array.dtype}')

    y_pred_logits_tensor.detach()

    return torch.tensor(y_pred_prob_array, device=device)


# %%
# Function to get y_pred and y_pred_prob
def preprocess_logits_for_metrics_y_pred(y_pred_prob_array):

    # https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    print('-'*20)
    print(f'Getting y_pred from y_pred_prob for {col}:')
    print('-'*20)

    if isinstance(y_pred_prob_array, tuple):
        y_pred_prob_array = y_pred_prob_array[0]

    if not torch.is_tensor(y_pred_prob_array):
        y_pred_prob_tensor = torch.tensor(y_pred_prob_array, device=device)
    else:
        y_pred_prob_tensor = y_pred_prob_array.to(device)

    print(f'y_pred_prob_array shape: {y_pred_prob_tensor.shape}. {y_pred_prob_tensor.dtype}')
    print('-'*20)

    # Get y_pred
    print('-'*20)
    print('Getting y_pred through argmax of y_pred_prob.')
    try:
        y_pred_array = torch.argmax(y_pred_prob_tensor, axis=-1).clone().detach().cpu().numpy()
        print('Using torch.argmax.')
    except Exception:
        y_pred_array = y_pred_prob.argmax(axis=-1)
        print('Using np.argmax.')

    print(f'y_pred_array shape: {y_pred_array.shape}')

    return y_pred_array


# %%
def compute_metrics(
    predicted_results_from_eval,
):
    y_pred_prob_array, y_labels_array = predicted_results_from_eval

    # Get y_pred_prob
    (
        y_pred_array
    ) = preprocess_logits_for_metrics_y_pred(y_pred_prob_array)

    # Get the the whole of the last column, which is the  probability of 1, and flatten to list
    print('-'*20)
    print('Flattening y_labels , y_pred_array, and y_pred_prob_array, then extracting probabilities of 1.')
    y_labels = y_labels_array.flatten().tolist()
    y_pred = y_pred_array.flatten().tolist()
    y_pred_prob = y_pred_prob_array[:, -1].flatten().tolist()
    print(f'y_pred_prob length: {len(y_pred_prob)}')
    print(f'y_labels length: {len(y_labels)}')
    print('-'*20)

    return compute_metrics_all(y_labels, y_pred, y_pred_prob)


# %%
# Function to get y_pred and y_pred_prob
def preprocess_logits_for_metrics_in_compute_metrics(y_pred_logits):

    # Get y_pred
    print('-'*20)
    y_pred_logits_tensor = torch.tensor(y_pred_logits, device=device)
    print('Getting y_pred through argmax of y_pred_logits...')
    try:
        y_pred_array = torch.argmax(y_pred_logits_tensor, axis=-1).clone().detach().cpu().numpy()
        print('Using torch.argmax.')
    except Exception:
        y_pred_array = y_pred_logits.argmax(axis=-1)
        print('Using np.argmax.')
    print(f'y_pred_array shape: {y_pred_array.shape}')
    print('-'*20)
    print('Flattening y_pred...')
    y_pred = [bert_label2id[l] for l in y_pred_array.flatten().tolist()]
    print(f'y_pred length: {len(y_pred)}')
    print('-'*20)

    # Get y_pred_prob
    print('-'*20)
    print('Getting y_pred_prob through softmax of y_pred_logits...')
    try:
        y_pred_prob_array = torch.nn.functional.softmax(y_pred_logits_tensor, dim=-1).clone().detach().cpu().numpy()
        print('Using torch.nn.functional.softmax.')
    except Exception:
        y_pred_prob_array = scipy.special.softmax(y_pred_logits, axis=-1)
        print('Using scipy.special.softmax.')
    # from: https://discuss.huggingface.co/t/different-results-predicting-from-trainer-and-model/12922
    assert all(y_pred_prob_array.argmax(axis=-1) == y_pred_array), 'Argmax of y_pred_prob_array does not match y_pred_array.'
    print(f'y_pred_prob shape: {y_pred_prob_array.shape}')
    print('-'*20)
    print('Flattening y_pred_prob and extracting probabilities of 1...')
    y_pred_prob = y_pred_prob_array[:, -1].flatten().tolist()
    print(f'y_pred length: {len(y_pred_prob)}')
    print('-'*20)

    y_pred_logits_tensor.detach()

    return (
        y_pred_array, y_pred, y_pred_prob_array, y_pred_prob
    )


# %%
def compute_metrics_from_logits(
    predicted_results_from_eval,
):
    # Get predictions
    print('-'*20)
    print(f'Getting y_pred logits and ids for {col}:')
    y_pred_logits, y_labels = predicted_results_from_eval
    print(f'y_pred_logits shape: {y_pred_logits.shape}')
    print(f'y shape: {y_labels.shape}')
    print('-'*20)

    # Get y_test_pred and y_test_pred_prob
    (
        y_pred_array, y_pred, y_pred_prob_array, y_pred_prob
    ) = preprocess_logits_for_metrics_in_compute_metrics(y_pred_logits)

    return compute_metrics_all(y_labels, y_pred, y_pred_prob)


# %%
def evaluation(
    metrics_dict, df_metrics,
    col, vectorizer_name, classifier_name
):
    # Print metrics
    print('=' * 20)
    print('~' * 20)
    print(' Metrics:')
    print('~' * 20)
    print(f'Classification Report:\n {metrics_dict["Classification Report"]}')
    print('-' * 20)
    for metric_name, metric_value in metrics_dict.items():
        if metric_name not in ['Runtime', 'Samples Per Second', 'Steps Per Second']:
            with contextlib.suppress(TypeError, ValueError):
                metric_value = float(metric_value)
            if isinstance(metric_name, (int, float)):
                df_metrics.loc[
                    (classifier_name), (col, vectorizer_name, metric_name)
                ] = metric_value
                print(f'{metric_name}: {round(metric_value, 2)}')
            else:
                df_metrics.loc[
                    (classifier_name), (col, vectorizer_name, metric_name)
                ] = str(metric_value)
                print(f'{metric_name}: {metric_value}')
            print('-' * 20)

    print('=' * 20)

    return df_metrics


# %%
# Function to place Xy data in df and save
def save_Xy_estimator(
    X_train, y_train, train_dataset,
    X_test, y_test, y_test_pred, y_test_pred_prob, test_dataset,
    X_val, y_val, y_val_pred, y_val_pred_prob, val_dataset,
    estimator, accelerator,
    col, vectorizer_name, classifier_name,
    results_save_path=results_save_path,
    method=method, done_xy_save_path=done_xy_save_path,
    path_suffix=None, data_dict=None,
    compression=None, protocol=None,
):
    if data_dict is None:
        data_dict = {}
    if compression is None:
        compression = False
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    if path_suffix is None:
        path_suffix = f' - {col} - {vectorizer_name} + {classifier_name} (Save_protocol={protocol}).pkl'

    # Check predicted data
    check_consistent_length(X_train, y_train, train_dataset)
    check_consistent_length(X_test, y_test, y_test_pred, y_test_pred_prob, test_dataset)
    check_consistent_length(X_val, y_val, y_val_pred, y_val_pred_prob, val_dataset)

    # Make data dict
    data_dict['Estimator'] = estimator
    data_dict['accelerator'] = accelerator
    data_dict['df_train_data'] = pd.DataFrame(
        {
            'X_train': X_train,
            'y_train': y_train,
            'train_dataset': train_dataset,
        },
    )
    # Make df_test_data
    data_dict['df_test_data'] = pd.DataFrame(
        {
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_pred_prob': y_test_pred_prob,
            'test_dataset': test_dataset,
        },
    )
    # Make df_val_data
    data_dict['df_val_data'] = pd.DataFrame(
        {
            'X_val': X_val,
            'y_val': y_val,
            'y_val_pred': y_val_pred,
            'y_val_pred_prob': y_val_pred_prob,
            'val_dataset': val_dataset,
        },
    )

    # Save files
    print('='*20)
    saved_files_list = []
    for file_name, file_ in data_dict.items():
        save_path = (
            done_xy_save_path
            if file_name not in ['Estimator', 'accelerator']
            else results_save_path
        )
        print(f'Saving {file_name} at {save_path}')
        if not isinstance(file_, pd.DataFrame) and file_name == 'Estimator' and 'df_' not in file_name:
            # Save as .model
            file_.save_model(f'{save_path}{method} {file_name}{path_suffix.replace("pkl", "model")}')
            saved_files_list.append(file_name)
        elif not isinstance(file_, pd.DataFrame) and file_name == 'accelerator' and 'df_' not in file_name:
            file_.save(estimator.state, f'{save_path}{method} Estimator{path_suffix.replace("pkl", "model")}/accelerator')
            saved_files_list.append(file_name)
        elif isinstance(file_, pd.DataFrame) and file_name != 'Estimator' and file_name != 'accelerator' and 'df_' in file_name:
            file_.to_pickle(
                f'{save_path}{method} {file_name}{path_suffix}', protocol=protocol
            )
            saved_files_list.append(file_name)

    assert set(data_dict.keys()) == set(saved_files_list), f'Not all files were saved! Missing: {set(data_dict.keys()) ^ set(saved_files_list)}'
    print(f'Done saving Xy, labels and estimator!\n{list(data_dict.keys())}')
    print('='*20)


# %%
# Assert that all classifiers were used
def assert_all_classifiers_used(
    estimators_list=None, used_classifiers=None, results_save_path=results_save_path, method=method, classifiers_pipe=transformers_pipe,
):
    if estimators_list is None:
        estimators_list = []
    if used_classifiers is None:
        used_classifiers = []

    for estimator_path in glob.glob(f'{results_save_path}{method} Estimator - *.pkl'):
        classifier_name = estimator_path.split(f'{results_save_path}{method} ')[1].split(' + ')[1].split(' (Save_protocol=')[0]
        used_classifiers.append(classifier_name)

    assert set(list(classifiers_pipe.keys())) == set(used_classifiers), f'Not all classifiers were used!\nAvaliable Classifiers:\n{set(list(classifiers_pipe.keys()))}\nUsed Classifiers:\n{set(used_classifiers)}\nLeftout Classifiers:\n{set(list(classifiers_pipe.keys())) ^ set(used_classifiers)}'
    print('All classifiers were used!')


# %% [markdown]
# # Training
#

# %% [markdown]
# ### READ DATA
#

# %%
with open(f'{data_dir}df_manual_len.txt', 'r') as f:
    df_manual_len = int(f.read())

df_manual = pd.read_pickle(f'{df_save_dir}df_manual_for_trainning.pkl')
assert len(df_manual) == df_manual_len, f'DATAFRAME MISSING DATA! DF SHOULD BE OF LENGTH {df_manual_len} BUT IS OF LENGTH {len(df_manual)}'
print(f'Dataframe loaded with shape: {df_manual.shape}')


# %%
print('#'*40)
print('Starting!')
print('#'*40)

analysis_columns = ['Warmth', 'Competence']
text_col = 'Job Description spacy_sentencized'

# Get existing estimators
estimator_names_list = get_existing_files()

for col in tqdm.tqdm(analysis_columns):

    print('-'*20)
    print(f'{"="*30} TRAINING DATASET OF LENGTH {len(df_manual)} ON {col.upper()} {"="*30}')
    print('-'*20)
    print(
        f'classifiers to be used ({len(list(transformers_pipe.keys()))}):\n{list(transformers_pipe.keys())}'
    )
    assert len(df_manual[df_manual[str(col)].map(df_manual[str(col)].value_counts() > 1)]) != 0, f'Dataframe has no {col} values!'

    if len(glob.glob(f'{results_save_path}{method} df_*_data - {col} - (Save_protocol=*).pkl')) == 3:
        # Load previous Xy
        print('Loading previous Xy.')
        (
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            train_class_weights, train_class_weights_ratio, train_class_weights_dict,
            test_class_weights_dict, test_class_weights_ratio, test_class_weights_dict,
            val_class_weights, val_class_weights_ratio, val_class_weights_dict,
        ) = load_Xy(
            col
        )
    else:
        print('Splitting data.')
        # Split data
        (
            train, X_train, y_train,
            test, X_test, y_test,
            val, X_val, y_val,
            train_class_weights, train_class_weights_ratio, train_class_weights_dict,
            test_class_weights, test_class_weights_ratio, test_class_weights_dict,
            val_class_weights, val_class_weights_ratio, val_class_weights_dict,
        ) = split_data(
            df_manual, col,
        )
        # Save Xy data
        save_Xy(
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            col,
        )

    # Get model_name, tokenizer, model and optimizer
    for transformer_name, transformer_dict in tqdm.tqdm(transformers_pipe.items()):
        model_name = transformer_dict['model_name']
        config = transformer_dict['config'].from_pretrained(model_name)
        tokenizer = transformer_dict['tokenizer'].from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))
        model = transformer_dict['model'].from_pretrained(model_name, config=config,).to(device)
        if model.config.pad_token_id is None:
            try:
                model.config.pad_token_id = tokenizer.pad_token_id
            except:
                model.config.pad_token_id = model.config.eos_token_id
        def model_init(trial): return model
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        vectorizer_name = ''.join(model.name_or_path.split('-')).upper()
        classifier_name = model.__class__.__name__
        output_dir = training_args_dict['output_dir'] = training_args_dict_for_best_trial['output_dir'] = f'{results_save_path}{method} Estimator - {col} - {vectorizer_name} + {classifier_name} (Save_protocol={pickle.HIGHEST_PROTOCOL}).model'
        # log_dir = training_args_dict['logging_dir'] = training_args_dict_for_best_trial['logging_dir'] = f'{results_save_path}{method} Estimator - {col} - {vectorizer_name} + {classifier_name} (Save_protocol={pickle.HIGHEST_PROTOCOL}).log'

        # Encode data
        (
            X_train_encodings, train_dataset,
            X_test_encodings, test_dataset,
            X_val_encodings, val_dataset,
        ) = encode_data(
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            tokenizer,
        )

        if f'{col} - {vectorizer_name} + {classifier_name}' in estimator_names_list:
            print('-'*20)
            print(
                f'Already trained {col} - {vectorizer_name} + {classifier_name}'
            )
            print('-'*20)
            continue

        # Accelerate model
        (
            model, tokenizer, optimizer, train_dataset, test_dataset, val_dataset
        ) = accelerator.prepare(
            model, tokenizer, optimizer, train_dataset, test_dataset, val_dataset
        )
        # model.eval()

        # Initialize Trainer
        print('-'*20)
        print('='*30)
        print(f'{"="*30} Initializing Trainer using {vectorizer_name} + {classifier_name} {"="*30}')
        print('+'*30)

        # Make trainer with fine-tuning arguments
        print('-'*20)
        print('Passing data and arguments to Trainer.')
        estimator = Trainer(
            # model_init=model_init,
            # args=TrainingArguments(**training_args_dict_for_best_trial),
            model=model,
            args=TrainingArguments(**training_args_dict),
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics_y_pred_prob,
            compute_metrics=compute_metrics,
            data_collator=transformers.DataCollatorWithPadding(tokenizer),
        )
        if estimator.place_model_on_device:
            estimator.model.to(device)

        # Hyperparameter search
        print('-'*20)
        print(f'Starting hyperparameter search for {col}.')
        best_trial = estimator.hyperparameter_search(
            direction='maximize',
            backend='optuna',
            n_trials=10,
            hp_space=optuna_hp_space,
            sampler=optuna.samplers.TPESampler(seed=random_state),
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
            compute_objective=compute_objective,
            n_jobs=n_jobs,
        )
        estimator.save_state()
        estimator.save_metrics('all', metrics_dict)
        estimator.save_model(output_dir)
        accelerator.save(estimator.state, f'{output_dir}/accelerator')
        print('Done hyperparameter search!')
        print('-'*20)

        # Train trainer
        print('-'*20)
        print(f'Starting training for {col}.')
        estimator.train()#trial=best_trial)
        estimator.save_state()
        estimator.save_metrics('all', metrics_dict)
        estimator.save_model(output_dir)
        accelerator.save(estimator.state, f'{output_dir}/accelerator')
        print('Done training!')
        print('-'*20)

        # Evaluate
        print('-'*20)
        print(f'Evaluating estimator for {col}.')
        eval_metrics_dict = estimator.evaluate()
        y_val_pred = eval_metrics_dict.pop('eval_y_pred')
        y_val_pred_prob = eval_metrics_dict.pop('eval_y_pred_prob')
        eval_metrics_dict = clean_metrics_dict(eval_metrics_dict, list(eval_metrics_dict.keys())[0].split('_')[0])
        print('Done evaluating!')

        # Get predictions
        print(f'Getting prediction results for {col}.')
        y_test_pred_logits, y_test_labels, test_metrics_dict = estimator.predict(test_dataset)
        y_test_pred = test_metrics_dict.pop('test_y_pred')
        y_test_pred_prob = test_metrics_dict.pop('test_y_pred_prob')
        test_metrics_dict = clean_metrics_dict(test_metrics_dict, list(test_metrics_dict.keys())[0].split('_')[0])
        print('Done predicting!')
        print('-'*20)

        # Save model
        print('-'*20)
        print(f'Saving model for {col}.')
        save_Xy_estimator(
            X_train, y_train, train_dataset,
            X_test, y_test, y_test_pred, y_test_pred_prob, test_dataset,
            X_val, y_val, y_val_pred, y_val_pred_prob, val_dataset,
            estimator, accelerator,
            col, vectorizer_name, classifier_name,
        )
        print('Done training!')
        print('-'*20)

# Assert that all classifiers were used
assert_all_classifiers_used(classifiers_pipe=classifiers_pipe)
print('#'*40)
print('DONE!')
print('#'*40)


# %%

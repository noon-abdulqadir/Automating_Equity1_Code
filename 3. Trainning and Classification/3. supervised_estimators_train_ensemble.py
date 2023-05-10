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
    code_dir = Path.cwd()
sys.path.append(code_dir)

# %load_ext autoreload
# %autoreload 2


# %%
from setup_module.imports import * # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from estimators_get_pipe import * # type:ignore # isort:skip # fmt:skip # noqa # nopep8


# %% [markdown]
# ### Set variables

# %%
# Variables
method = 'Supervised'
classifiers_type = 'ensemble'
if classifiers_type == 'nonlinear':
    classifiers_pipe = classifiers_pipe_nonlinear
elif classifiers_type == 'linear':
    classifiers_pipe = classifiers_pipe_linear
elif classifiers_type == 'ensemble':
    classifiers_pipe = classifiers_pipe_ensemble
elif classifiers_type == 'all':
    classifiers_pipe = classifiers_pipe

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
    'Brier score loss': np.nan,
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
random_state = 42
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
cores = multiprocessing.cpu_count()


# %% [markdown]
# # Functions

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
    models_save_path=models_save_path, results_save_path=results_save_path,
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
def split_data(df, col, text_col=text_col, analysis_columns=analysis_columns):

    train_ratio = 0.75
    test_ratio = 0.10
    validation_ratio = 0.15
    test_split = test_size = 1 - train_ratio
    validation_split = test_ratio / (test_ratio + validation_ratio)

    # Split
    print('='*20)
    print('Splitting data into training and testing:')
    print(f'Ratios: train_size = {train_ratio}, test size = {test_ratio}')

    df = df.dropna(subset=analysis_columns, how='any')
    df = df.loc[df[text_col].apply(len) >= 5]
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
def load_Xy(
    col,
    models_save_path=models_save_path, results_save_path=results_save_path, method=method,
    path_suffix=None, data_dict=None, protocol=None,
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
    print(f'Done loading Xy from previous for {col}!')

    return (
        X_train, y_train,
        X_test, y_test,
        X_val, y_val,
        train_class_weights, train_class_weights_ratio, train_class_weights_dict,
        test_class_weights_dict, test_class_weights_ratio, test_class_weights_dict,
        val_class_weights, val_class_weights_ratio, val_class_weights_dict,
    )


# %%
# Function to normalize unusual classifiers after fitting
def normalize_after_fitting(estimator, X_train, y_train, X_test, y_test, searchcv):
    # Classifiers to normalize = ['GaussianNB', 'DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'Perceptron', 'Sequential']

    # Get feature importance if classifier provides them and use as X
    if any(hasattr(estimator, feature_attr) for feature_attr in ['feature_importances_', 'coef_']):
        feature_selector = SelectFromModel(estimator, prefit=True)
        X_train = feature_selector.transform(X_train)
        X_test = X_test[:, feature_selector.get_support()]
        df_feature_importances = pd.DataFrame(
            {
                'features': X_test.values,
                'feature_importances': estimator.feature_importances_
            }
        )
        df_feature_importances = df_feature_importances.sort_values('feature_importances', ascending=False)
        print(df_feature_importances.head(20))
        print(f'Best estimator has feature_importances of shape:\n{estimator}')
    else:
        df_feature_importances = None

    # For perceptron: calibrate classifier to get prediction probabilities
    if not hasattr(searchcv, 'predict_proba') and not hasattr(searchcv, '_predict_proba_lr') and hasattr(searchcv, 'decision_function'):
        searchcv = CalibratedClassifierCV(
            searchcv, cv=cv, method='sigmoid'
        ).fit(X_train, y_train)

    # For Sequential classifier: compile for binary classification, optimize with adam and score on recall
    if classifier_name == 'Sequential':
        searchcv.compile(
            loss='binary_crossentropy', optimizer='adamw', metrics=list(scoring)
        ).fit(X_train, y_train)

    return (
        estimator, X_train, y_train, X_test, y_test, searchcv, df_feature_importances
    )


# %%
# Function to place Xy and CV data in df and save
def save_Xy_search_cv_estimator(
    grid_search, searchcv, cv_results,
    X_train, y_train, y_train_pred, y_train_pred_prob,
    X_test, y_test, y_test_pred, y_test_pred_prob,
    X_val, y_val, y_val_pred, y_val_pred_prob,
    df_feature_importances,
    estimator,
    col, vectorizer_name, classifier_name,
    results_save_path=results_save_path,
    method=method, done_xy_save_path=done_xy_save_path,
    path_suffix=None, data_dict=None,
    compression=None, protocol=None,
):
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    if path_suffix is None:
        path_suffix = f' - {col} - {vectorizer_name} + {classifier_name} (Save_protocol={protocol}).pkl'
    if data_dict is None:
        data_dict = {}
    if compression is None:
        compression = False

    # Check predicted data
    check_consistent_length(X_train, y_train, y_train_pred, y_train_pred_prob)
    check_consistent_length(X_test, y_test, y_test_pred, y_test_pred_prob)
    check_consistent_length(X_val, y_val, y_val_pred, y_val_pred_prob)

    # Make data dict
    data_dict['Estimator'] = estimator
    data_dict['Grid Search'] = grid_search
    data_dict['SearchCV'] = searchcv
    # Make df_cv_results
    data_dict['df_cv_results'] = pd.DataFrame(
        cv_results
    )
    # Make df_train_data
    data_dict['df_train_data'] = pd.DataFrame(
        {
            'X_train': X_train,
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'y_train_pred_prob': y_train_pred_prob,
        },
    )
    # Make df_test_data
    data_dict['df_test_data'] = pd.DataFrame(
        {
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_pred_prob': y_test_pred_prob,
        },
    )
    # Make df_val_data
    data_dict['df_val_data'] = pd.DataFrame(
        {
            'X_val': X_val,
            'y_val': y_val,
            'y_val_pred': y_val_pred,
            'y_val_pred_prob': y_val_pred_prob,
        },
    )

    # Make df_feature_importances
    if df_feature_importances is not None:
        data_dict['df_feature_importances'] = df_feature_importances

    # Save files
    print('='*20)
    saved_files_list = []
    for file_name, file_ in data_dict.items():
        save_path = done_xy_save_path if file_name != 'Estimator' else results_save_path
        print(f'Saving {file_name} at {save_path}')
        if 'df_' not in file_name:
            with open(
                f'{save_path}{method} {file_name}{path_suffix}', 'wb'
            ) as f:
                joblib.dump(file_, f, compress=compression, protocol=protocol)
        else:
            file_.to_pickle(
                f'{save_path}{method} {file_name}{path_suffix}', protocol=protocol
            )
        saved_files_list.append(file_name)
    assert set(data_dict.keys()) == set(saved_files_list), f'Not all files were saved! Missing: {set(data_dict.keys()) ^ set(saved_files_list)}'
    print(f'Done saving Xy, CV data, and estimator!\n{list(data_dict.keys())}')
    print('='*20)


# %%
# Assert that all classifiers were used
def assert_all_classifiers_used(
    classifiers_pipe, estimators_list=None, used_classifiers=None, results_save_path=results_save_path, method=method
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

# %% [markdown]
# ### READ DATA

# %%
with open(f'{data_dir}df_manual_len.txt', 'r') as f:
    df_manual_len = int(f.read())

df_manual = pd.read_pickle(f'{df_save_dir}df_manual_for_training.pkl')
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
        f'Vectorizers to be used ({len(list(vectorizers_pipe.values()))}):\n{list(vectorizers_pipe.keys())}'
    )
    print(
        f'Total number of vectorizer parameters = {sum([len(list(vectorizers_pipe.values())[i]) for i in range(len(vectorizers_pipe))])}'
    )
    print(
        f'Selectors to be used ({len(list(selectors_pipe.values()))}):\n{list(selectors_pipe.keys())}'
    )
    print(
        f'Total number of selector parameters = {sum([len(list(selectors_pipe.values())[i][1]) for i in range(len(selectors_pipe))])}'
    )
    print(
        f'Resamplers to be used ({len(list(resamplers_pipe.keys()))}):\n{list(resamplers_pipe.keys())}'
    )
    print(
        f'Total number of resamplers parameters = {sum([len(list(resamplers_pipe.values())[i][1]) for i in range(len(resamplers_pipe))])}'
    )
    print(
        f'classifiers to be used ({len(list(classifiers_pipe.keys()))}):\n{list(classifiers_pipe.keys())}'
    )
    print(
        f'Total number of classifiers parameters = {sum([len(list(classifiers_pipe.values())[i][1]) for i in range(len(classifiers_pipe))])}'
    )

    assert len(df_manual[df_manual[str(col)].map(df_manual[str(col)].value_counts() > 1)]) != 0, f'Dataframe has no {col} values!'

    if len(glob.glob(f'{models_save_path}df_*_data - {col} - (Save_protocol=*).pkl')) == 3:
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

    for (
        vectorizer_name, vectorizer_and_params
    ), (
        selector_name, selector_and_params
    ), (
        resampler_name, resampler_and_params
    ), (
        classifier_name, classifier_and_params
    ) in tqdm_product(
        vectorizers_pipe.items(), selectors_pipe.items(), resamplers_pipe.items(), classifiers_pipe.items()
    ):

        if f'{col} - {vectorizer_name} + {classifier_name}' in estimator_names_list:
            print('-'*20)
            print(
                f'Already trained {col} - {vectorizer_name} + {classifier_name}'
            )
            print('-'*20)
            continue

        ## Normalize Xy for unusual classifiers before fitting
        if classifier_name == 'GaussianNB':
            X_train = X_train.todense()
            X_test = X_test.todense()
            X_val = X_val.todense()

        # Identify names and params
        vectorizer = vectorizer_and_params[0]
        vectorizer_params = vectorizer_and_params[-1]

        selector = selector_and_params[0]
        selector_params = selector_and_params[-1]

        resampler = resampler_and_params[0]
        resampler_params = resampler_and_params[-1]

        classifier = classifier_and_params[0]
        classifier_params = classifier_and_params[-1]

        # Pipeline
        ## Steps
        if col == 'Warmth':
            steps = [
                (vectorizer_name, vectorizer),
                (selector_name, selector),
                (resampler_name, resampler),
                (classifier_name, classifier)
            ]
        else:
            steps = [
                (vectorizer_name, vectorizer),
                (selector_name, selector),
                (classifier_name, classifier)
            ]

        ## Params
        param_grid = {
            **vectorizer_params,
            **selector_params,
            **classifier_params,
        }

        ## Pipeline
        pipe = imblearn.pipeline.Pipeline(steps=steps)

        # Search
        print('-'*20)
        print(f'{"="*30} Using GridSearchCV {"="*30}')
        print('-'*20)
        print(f'GridSearchCV with:\nPipe:\n{pipe}\nParams:\n{param_grid}')
        print('+'*30)

        # Use if StratifiedKFold causes issues
        # cv = PredefinedSplit(test_fold=[-1]*len(X_train) + [0]*len(X_val))
        # Pass arguments to gridsearch
        grid_search = HalvingGridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=True,
            verbose=1,
            error_score='raise',
            refit=refit,
            random_state=random_state,
            scoring=scorers['recall_score'],
        )

        with joblib.parallel_backend(backend='loky', n_jobs=n_jobs):
            # Fit SearchCV
            print('Fitting GridSearchCV')
            searchcv = grid_search.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

        # Reidentify and name best estimator and params
        estimator = searchcv.best_estimator_
        cv_results = searchcv.cv_results_
        vectorizer = estimator[0]
        vectorizer_params = vectorizer.get_params()
        vectorizer_name = vectorizer.__class__.__name__
        selector = estimator[1]
        selector_params = selector.get_params()
        selector_name = selector.__class__.__name__
        classifier = estimator[-1]
        classifier_params = classifier.get_params()
        classifier_name = classifier.__class__.__name__
        if col == 'Warmth':
            resampler = estimator[-2]
            resampler_params = resampler.get_params()
            resampler_name = resampler.__class__.__name__

        # Normalize Xy for unusual classifiers after fitting
        (
            estimator, X_train, y_train, X_test, y_test, searchcv, df_feature_importances
        ) = normalize_after_fitting(
            estimator, X_train, y_train, X_test, y_test, searchcv
        )

        # Set prediction probability attribute
        if hasattr(searchcv, 'predict_proba'):
            searchcv_predict_attr = searchcv.predict_proba
        elif hasattr(searchcv, '_predict_proba_lr'):
            searchcv_predict_attr = searchcv._predict_proba_lr

        # Get predictions and probabilities
        y_train_pred = estimator.predict(X_train)
        y_train_pred_prob = searchcv_predict_attr(X_train)[:, 1]

        y_test_pred = searchcv.predict(X_test)
        y_test_pred_prob = searchcv_predict_attr(X_test)[:, 1]

        y_val_pred = searchcv.predict(X_val)
        y_val_pred_prob = searchcv_predict_attr(X_val)[:, 1]

        # Save Xy and CV data
        save_Xy_search_cv_estimator(
            grid_search, searchcv, cv_results,
            X_train, y_train, y_train_pred, y_train_pred_prob,
            X_test, y_test, y_test_pred, y_test_pred_prob,
            X_val, y_val, y_val_pred, y_val_pred_prob,
            df_feature_importances, estimator,
            col, vectorizer_name, classifier_name,
        )

# Assert that all classifiers were used
assert_all_classifiers_used(classifiers_pipe=classifiers_pipe)
print('#'*40)
print('DONE!')
print('#'*40)


# %%

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
from setup_module.imports import * # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from estimators_get_pipe import * # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from setup_module import specification_curve_fork as specy # type:ignore # isort:skip # fmt:skip # noqa # nopep8


# %% [markdown]
# ### Set variables

# %%
# Variables
method = 'Supervised'
with open(f'{data_dir}{method}_results_save_path.txt', 'r') as f:
    results_save_path = f.read()
with open(f'{data_dir}{method}_done_xy_save_path.txt', 'r') as f:
    done_xy_save_path = f.read()
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
    'num_train_epochs', 'learning_rate',
]
training_args_dict = {
    'seed': random_state,
    'resume_from_checkpoint': True,
    'overwrite_output_dir': True,
    'logging_steps': 500,
    'evaluation_strategy': 'steps',
    'eval_steps': 500,
    'save_strategy': 'steps',
    'save_steps': 500,
    # 'metric_for_best_model': 'recall',
    # 'torch_compile': bool(transformers.file_utils.is_torch_available()),
    'use_mps_device': bool(device_name == 'mps' and torch.backends.mps.is_available()),
    'optim': 'adamw_torch',
    'load_best_model_at_end': True,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 20,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    # The below metrics are used by hyperparameter search
    'num_train_epochs': 3,
    'learning_rate': 5e-5,
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
class ToDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            key: val[idx].clone().detach().to(device)
            for key, val in self.encodings.items()
        }

    def __len__(self):
        return len(self.encodings['input_ids'])


# %%
class ImbTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = self._calculate_class_weights(self.train_dataset)

    def _calculate_class_weights(self, dataset):
        # Count the number of samples in each class
        class_counts = torch.zeros(self.model.config.num_labels)
        for label in dataset.labels:
            class_counts[label] += 1

        # Calculate the inverse frequency of each class
        inv_frequencies = 1 / class_counts

        # Normalize the inverse frequencies so that they sum up to 1
        sum_inv_frequencies = torch.sum(inv_frequencies)
        return inv_frequencies / sum_inv_frequencies

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# %%
# Function to get y_pred and y_pred_prob
def preprocess_logits_for_metrics_in_compute_metrics(y_pred_logits):

    # Create a DeepSpeed engine
    engine, _ = deepspeed.initialize()

    # Get y_pred
    print('-'*20)
    y_pred_logits_tensor = torch.tensor(y_pred_logits, device=device)
    print('Getting y_pred through argmax of y_pred_logits...')
    try:
        y_pred_array = engine.module(torch.argmax(y_pred_logits_tensor, axis=-1)).cpu().numpy()
        print('Using torch.argmax.')
    except Exception:
        y_pred_array = engine.module(y_pred_logits.argmax(axis=-1))
        print('Using np.argmax.')
    print(f'y_pred_array shape: {y_pred_array.shape}')
    print('-'*20)
    print('Flattening y_pred...')
    y_pred = y_pred_array.flatten().tolist()
    print(f'y_pred length: {len(y_pred)}')
    print('-'*20)

    # Get y_pred_prob
    print('-'*20)
    print('Getting y_pred_prob through softmax of y_pred_logits...')
    try:
        y_pred_prob_array = engine.module(torch.nn.functional.softmax(y_pred_logits_tensor, dim=-1)).cpu().numpy()
        print('Using torch.nn.functional.softmax.')
    except Exception:
        y_pred_prob_array = engine.module(scipy.special.softmax(y_pred_logits, axis=-1))
        print('Using scipy.special.softmax.')
    # from: https://discuss.huggingface.co/t/different-results-predicting-from-trainer-and-model/12922
    assert all(y_pred_prob_array.argmax(axis=-1) == y_pred_array), 'Argmax of y_pred_prob_array does not match y_pred_array.'
    print(f'y_pred_prob shape: {y_pred_prob_array.shape}')
    print('-'*20)
    print('Flattening y_pred_prob and extracting probabilities of 1...')
    y_pred_prob = y_pred_prob_array[:, -1].flatten().tolist()
    print(f'y_pred length: {len(y_pred_prob)}')
    print('-'*20)

    y_pred_logits_tensor.clone().detach()

    return (
        y_pred_array, y_pred, y_pred_prob_array, y_pred_prob
    )


# %% [markdown]
# # Classifying

# %% [markdown]
# ### READ DATA

# %%
with open(f'{data_dir}df_jobs_len.txt', 'r') as f:
    df_jobs_len = int(f.read())

df_jobs = pd.read_pickle(f'{df_save_dir}df_jobs_for_classification.pkl')
assert len(df_jobs) == df_jobs_len, f'DATAFRAME MISSING DATA! DF SHOULD BE OF LENGTH {df_jobs_len} BUT IS OF LENGTH {len(df_jobs)}'
print(f'Dataframe loaded with shape: {df_jobs.shape}')


# %%
print('#'*40)
print('Starting!')
print('#'*40)

protocol = pickle.HIGHEST_PROTOCOL
analysis_columns = ['Warmth', 'Competence']
text_col = 'Job Description spacy_sentencized'
classified_columns = ['Warmth_Probability', 'Competence_Probability']
final_estimators_dict = {
    'Warmth': {
        'vectorizer_name': 'BERTBASEUNCASED',
        'classifier_name': 'BertForSequenceClassification',
    },
    'Competence': {
        'vectorizer_name': 'BERTBASEUNCASED',
        'classifier_name': 'BertForSequenceClassification',
    },
}

for col in tqdm.tqdm(analysis_columns):
    print('-'*20)
    final_estimators_dict[col]['path_suffix'] = path_suffix = f' - {col} - {(vectorizer_name := final_estimators_dict[col]["vectorizer_name"])} + {(classifier_name := final_estimators_dict[col]["classifier_name"])} (Save_protocol={protocol})'

    if classifier_name in list(classifiers_pipe.keys()):
        method = 'Supervised'
        with open(f'{data_dir}{method}_results_save_path.txt', 'r') as f:
            results_save_path = f.read()
        print('-'*20)
        print(f'Using {classifier_name} from {method} pipeline.')
        print('Loading Supervised Estimator.')
        with open(
            f'{results_save_path}{method} Fitted Estimator {path_suffix}.pkl', 'rb'
        ) as f:
            estimator = joblib.load(f)
        print('Done loading Supervised Estimator!')

        print('-'*20)
        print('Classifying data.')
        X = np.array(list(df_jobs[text_col].astype('str').values))
        df_jobs[col] = estimator.predict(X)
        if hasattr(estimator, 'predict_proba'):
            # Get the the whole of the last column, which is the  probability of 1, and flatten to list
            df_jobs[f'{col}_Probability'] = estimator.predict_proba(X)[:, -1]

        print(f'Done classifying data using {classifier_name} for {col}!')
        print('-'*20)

    elif classifier_name in list(transformers_pipe.keys()):
        method = 'Transformers'
        with open(f'{data_dir}{method}_results_save_path.txt', 'r') as f:
            results_save_path = f.read()
        print('-'*20)
        print(f'Using {classifier_name} from {method} pipeline.')
        print('Loading Transformer Estimator.')
        model = transformers_pipe[classifier_name]['model']
        tokenizer = transformers_pipe[classifier_name]['tokenizer']
        config = transformers_pipe[classifier_name]['config']
        estimator_dir = f'{results_save_path}{method} Fitted Estimator{path_suffix}.model'
        output_dir = training_args_dict['output_dir'] = training_args_dict_for_best_trial['output_dir'] = final_estimators_dict[col]['output_dir'] = f'{results_save_path}{method} Final Estimator{path_suffix}.model'
        fitted_estimator = model.from_pretrained(estimator_dir)
        tokenizer = tokenizer.from_pretrained(estimator_dir)
        config = config.from_pretrained(f'{estimator_dir}/config.json')
        # if col == 'Warmth':
        #     Trainer = ImbTrainer
        # else:
        #     Trainer = Trainer

        # Tokenize
        X = df_jobs[text_col].astype('str').values.tolist()
        encodings = tokenizer(
        X, truncation=True, padding=True, max_length=max_length, return_tensors=returned_tensor
        ).to(device)
        dataset = ToDataset(encodings)
        print('Done loading Transformer Estimator!')
        # Accelerate model
        (
            fitted_estimator, tokenizer, dataset
        ) = accelerator.prepare(
            fitted_estimator, tokenizer, dataset
        )

        # HACK
        # Deepseed
        from deepspeed import DeepSpeedEngine
        from deepspeed.ops.adam import FusedAdam
        from transformers.integrations import deepspeed

        # define the DeepSpeed configuration
        deepspeed_config = DeepSpeedConfig(
            zero_allow_untested_optimizer=True,
            zero_optimization_level=3,
            zero_optimization={},
            enable_zero=False,
            partition_activations=False,
            partition_weights=[1],
            fp16={'enabled': True},
            offload_optimizer={'device': 'nvme', 'nvme_path': '/raid/'},
        )
        engine, _, _ = DeepSpeedEngine.initialize(model=fitted_estimator, local_rank=0, config=deepspeed_config)

        # Get predictions
        print(f'Getting prediction results for {col}.')
        estimator = Trainer(
            model=engine.module,# HACK
            tokenizer=tokenizer,
            # args=TrainingArguments(**training_args_dict),
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics_y_pred_prob,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            # data_collator=transformers.DataCollatorWithPadding(tokenizer),
        )
        if estimator.place_model_on_device:
            estimator.model.to(device)

        print('-'*20)
        print('Classifying data.')
        y_pred_logits, y_labels = estimator.predict(dataset)
        y_pred_array, y_pred, y_pred_prob_array, y_pred_prob = preprocess_logits_for_metrics_in_compute_metrics(y_pred_logits)
        df_jobs[col] = y_pred
        df_jobs[f'{col}_Probability'] = y_pred_prob

        print(f'Done classifying data using {classifier_name} for {col}!')
        print('-'*20)


# %% [markdown]
# ## Inspect classified data

# %%
assert len(df_jobs) > 0 and isinstance(df_jobs, pd.DataFrame), f'ERORR: LENGTH OF DF = {len(df_jobs)}'
df_jobs.to_pickle(f'{df_save_dir}df_jobs_for_analysis.pkl')
df_jobs.to_csv(f'{df_save_dir}df_jobs_for_analysis.csv', index=False)


# %%
df_jobs = df_jobs.dropna(subset=['Warmth', 'Competence', 'Warmth_Probability', 'Competence_Probability'])


# %%
df_jobs.info()


# %%
df_jobs.describe()


# %%
get_df_info(df_jobs, ivs_all=[analysis_columns])


# %%
get_df_info(df_jobs, ivs_all=[classified_columns])


# %% [markdown]
# ### Plot classified data
#

# %%
# Counts plot of classifed warmthh and competence
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(x='Warmth', data=df_jobs, ax=ax[0], palette='colorblind')
sns.countplot(x='Competence', data=df_jobs, ax=ax[1], palette='colorblind')
plt.show()


# %%
# Box plot of warmth and competence probabilities
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(x='Warmth', y='Warmth_Probability', data=df_jobs, ax=ax[0], palette='colorblind')
sns.boxplot(x='Competence', y='Competence_Probability', data=df_jobs, ax=ax[1], palette='colorblind')
plt.show()


# %%
# Specification curve analysis
print(f'Running Logit specification curve analysis with:\nDEPENDENT VARIABLES = {dvs}\nINDEPENDENT VARIABLES = {ivs_dummy_and_perc}\nCONTROLS = {controls}')
sc = specy.SpecificationCurve(df=dj_jobs, y_endog=dvs, x_exog=ivs_dummy, controls=controls)
sc.fit(estimator=sm.Logit)
sc.plot(show_plot=True)


# %%
# Specification curve analysis
print(f'Running OLS specification curve analysis with:\nDEPENDENT VARIABLES = {dvs_prob}\nINDEPENDENT VARIABLES = {ivs_dummy_and_perc}\nCONTROLS = {controls}')
sc = specy.SpecificationCurve(df=dj_jobs, y_endog=dvs_prob, x_exog=ivs_perc, controls=controls)
sc.fit(estimator=sm.OLS)
sc.plot(show_plot=True)

# %% [markdown]
# ### Save dataframe
#

# %%
assert len(df_jobs) > 0 and isinstance(df_jobs, pd.DataFrame), f'ERORR: LENGTH OF DF = {len(df_jobs)}'
df_jobs.to_pickle(f'{df_save_dir}df_jobs_for_analysis.pkl')
df_jobs.to_csv(f'{df_save_dir}df_jobs_for_analysis.csv', index=False)


# %%

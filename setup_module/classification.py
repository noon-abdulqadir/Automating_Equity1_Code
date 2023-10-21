# -*- coding: utf-8 -*-
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
from setup_module.imports import *
from setup_module.scraping import *

import evaluate
from accelerate import Accelerator

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


warnings.filterwarnings('ignore', category=DeprecationWarning)


# %%
# preprocessing
# Function to get new data, clean it, and make excel dfs


def get_new_data(cleanup_return_enabled=True, main_from_file=True, args=get_args()):
    if main_from_file is False:
        df_jobs = post_cleanup(keywords_from_list=True, site_from_list=True)
    elif main_from_file is True:
        with open(args['df_dir'] + f'df_jobs_post_cleanup.{args["file_save_format"]}', 'rb') as f:
            df_jobs = pickle.load(f)

    df_sentence = split_to_sentences(df_jobs)
    if cleanup_return_enabled is True:

        return df_jobs, df_sentence


# %%
# Open traing and testing dfs
def open_and_clean_labeled_excel(
    id_dict_new=False,
    df_labeled_dict={},
    dict_save=True,
    analysis_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    args=get_args(),
):

    # Open and clean labeled excel files
    (
        coders_list,
        coders_dict,
        coders_numbers,
        df_coder_list,
        df_jobs_labeled,
    ) = open_and_clean_excel()

    # Tune DF
    df_jobs_labeled = df_jobs_labeled.rename(
        columns={'Sentence': 'Job Description'})
    df_jobs_labeled = df_jobs_labeled[
        ['Job ID', 'Job Description'] + [str(col) for col in analysis_columns]
    ]
    df_jobs_labeled.index = range(df_jobs_labeled.shape[0])

    if args['print_enabled'] is True:
        print(f'Number of DF sentences: {len(df_jobs_labeled)}')
    word_count = (
        df_jobs_labeled['Job Description']
        .progress_apply(lambda x: len(x.split(' ')))
        .sum()
    )
    if args['print_enabled'] is True:
        print(f'Number of DF words: {word_count}')

    # # Set Age and Gender IVs
    df_jobs_labeled = set_gender_age_sects_lang(
        df_jobs=df_jobs_labeled, id_dict_new=id_dict_new, str_variable='Job Description')

    # # Gender and Age Info
    get_df_info(df_jobs_labeled)

    # # Detect Language
    # df_jobs_labeled = detect_language(df_jobs_labeled)

    # profile = ProfileReport(df_jobsn, title='Pandas Profiling Report', explorative=True)
    # print(profile.to_json())

    if args['save_enabled'] is True:
        df_jobs_labeled.to_csv(
            f'{args["df_dir"]}df_jobs_labeled_unprocessed.{args["file_save_format_backup"]}')

        df_jobs_labeled.to_pickle(
            f'{args["df_dir"]}df_jobs_labeled_unprocessed.{args["file_save_format"]}')

    # Make different DFs for each variable in training then place them in dict
    for column, col in tqdm_product(df_jobs_labeled.columns, analysis_columns):
        if str(col) == str(column):
            df_labeled_dict[f'{column}'] = df_jobs_labeled[
                ['Job Description', 'Gender', 'Age', str(column)]
            ]

    if dict_save is True:

        with open(f'{args["parent_dir"]}df_labeled_dict.{args["file_save_format"]}', 'wb') as f:
            pickle.dump(df_labeled_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        with contextlib.suppress(Exception):
            with open(f'{args["parent_dir"]}df_labeled_dict.json', 'w', encoding='utf8') as f:
                json.dump(df_labeled_dict, f)

    if args['save_enabled'] is True:
        plot_df(df_jobs_labeled, label='Jobs DataFrame')

    return (df_jobs_labeled, df_labeled_dict)


# %%
# Open unlabled excel files
def open_and_clean_unlabeled_excel(
    main_from_file=True,
    EXCEL_PATHS=None,
    df_unlabeled_list=None,
    language='en',
    get_new_data_enabled=True,
    cleanup_return_enabled=True,
    stable_path_excel='Jobs EXECL/',
    analysis_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    args=get_args(),
):
    if EXCEL_PATHS is None:
        EXCEL_PATHS = []
    if df_unlabeled_list is None:
        df_unlabeled_list = []

    if get_new_data_enabled is True:
        print('Running post cleanup and splitting sentences.')
        if cleanup_return_enabled is True:
            df_jobs, df_sentence = get_new_data(cleanup_return_enabled)
        elif cleanup_return_enabled is False:
            get_new_data(cleanup_return_enabled)
    elif get_new_data_enabled is False:
        print('Using older version of df_jobs.')
        # with open(args["df_dir"] + f'df_jobs_post_cleanup.{args["file_save_format"]}', 'rb') as f:
        #     df_jobs = pickle.load(f)

    if main_from_file is False:
        stable_path = validate_path(
            validate_path(args['parent_dir'] +
                          stable_path_excel + f'{language}/')
        )

        if os.path.isdir(stable_path):
            # for stable_folder, sub_folder, files_list in os.walk(stable_path):
            #     for file in files_list:
            #         if len(files_list) != 0 and 'Job ID - ' in file and '.xlsx' in file and os.path.isfile(f'{stable_folder}/{file}'):
            EXCEL_PATHS = [
                validate_path(f'{stable_folder}/{file}')
                for stable_folder, sub_folder, files_list in os.walk(stable_path)
                for file in files_list
                if len(files_list) != 0
                and 'Job ID - ' in file
                and '.xlsx' in file
                and is_non_zero_file(f'{stable_folder}/{file}') is True
            ]
        if len(EXCEL_PATHS) != 0:
            for path in EXCEL_PATHS:
                if (
                    'blank-header'
                    not in validate(path)['tables'][0]['errors'][0]['code']
                ):
                    print(
                        'An error was found:',
                        validate(file)['tables'][0]['errors'][0]['code'],
                    )
                df_jobs_unlabeled_full = pd.read_excel(
                    validate_path(path), index_col=0, engine='openpyxl'
                )
                if df_jobs_unlabeled_full.columns.str.contains('^Unnamed').all():
                    break
                df_jobs_unlabeled_full = df_jobs_unlabeled_full.rename(
                    columns={'Sentence': 'Job Description'}
                )
                df_jobs_unlabeled_full = clean_df(
                    df_jobs_unlabeled_full
                ).drop(
                    df_jobs_unlabeled_full.columns[
                        df_jobs_unlabeled_full.columns.str.contains(
                            'Coder Remarks', case=False
                        )
                    ],
                    axis='columns',
                    inplace=True,
                    errors='ignore',
                )
                df_jobs_unlabeled_full = open_and_clean_unlabeled_excel_helper(
                    df_jobs_unlabeled_full)
                df_unlabeled_list.append(df_jobs_unlabeled_full)
    elif main_from_file is True:
        df_jobs_temp = []
        for jobs_list in df_jobs:
            if isinstance(jobs_list, list):
                for df in jobs_list:
                    if isinstance(df, pd.DataFrame):
                        df_jobs_temp.append(df)
                    elif isinstance(df, list):
                        for d in df:
                            if isinstance(d, pd.DataFrame):
                                df_jobs_temp.append(df)
                            else:
                                print('Too many layers for jobs DF.')
            elif isinstance(jobs_list, pd.DataFrame):
                df_jobs_temp.append(jobs_list)
            df_jobs_unlabeled_full = pd.concat(df_jobs_temp)
            df_jobs_unlabeled_full = open_and_clean_unlabeled_excel_helper(
                df_jobs_unlabeled_full)
            df_unlabeled_list.append(df_jobs_unlabeled_full)
    # pbar.finish()
    if len(df_unlabeled_list) > 1:
        df_jobs_unlabeled = pd.concat(df_unlabeled_list)
    else:
        if isinstance(df_unlabeled_list, pd.DataFrame):
            df_jobs_unlabeled = df_unlabeled_list
        if isinstance(df_unlabeled_list, list):
            df_jobs_unlabeled = pd.concat(
                [df for df in df_unlabeled_list if isinstance(df, pd.DataFrame)])

    df_jobs_unlabeled['Sentence ID'] = (
        df_jobs_unlabeled.groupby(['Job Description']).ngroup() + 1
    )
    sent_id_col = df_jobs_unlabeled.pop('Sentence ID')
    df_jobs_unlabeled = df_jobs_unlabeled[
        [
            col
            for col in df_jobs_unlabeled.columns
            if (col not in analysis_columns) and ('Sentence ID' not in col)
        ]
        # + analysis_columns
    ]
    df_jobs_unlabeled = df_jobs_unlabeled.reindex(
        columns=df_jobs_unlabeled.columns.tolist() + analysis_columns)
    df_jobs_unlabeled = df_jobs_unlabeled.drop_duplicates()
    df_jobs_unlabeled.insert(1, sent_id_col.name, sent_id_col)
    df_jobs_unlabeled = clean_df(
        df_jobs_unlabeled, str_variable='Job Description'
    )
    df_jobs_unlabeled.index = range(df_jobs_unlabeled.shape[0])

    if args['save_enabled'] is True:
        df_jobs_unlabeled.to_pickle(validate_path(
            f'{args["df_dir"]}df_jobs_unlabeled.{args["file_save_format"]}'))
        df_jobs_unlabeled.to_csv(validate_path(
            f'{args["df_dir"]}df_jobs_unlabeled.{args["file_save_format_backup"]}'))

    return df_jobs_unlabeled


# %%
def open_and_clean_unlabeled_excel_helper(df_jobs_unlabeled_full):

    df_jobs_unlabeled_full = df_jobs_unlabeled_full.dropna(
        subset=['Job ID', 'Job Description'])
    df_jobs_unlabeled_full = df_jobs_unlabeled_full.drop_duplicates()
    df_jobs_unlabeled_full['Job ID'] = df_jobs_unlabeled_full['Job ID'].fillna(
        method='ffill')
    df_jobs_unlabeled_full['Job Description'] = (
        df_jobs_unlabeled_full['Job Description']
        .str.strip()
        .str.lower()
        .str.replace('[^\w\s]', '')
    )
    if df_jobs_unlabeled_full['Job Description'].isna().sum() > 0:
        print(
            f'{df_jobs_unlabeled_full["Job Description"].isna().sum()} missing sentences found.'
        )

    return df_jobs_unlabeled_full


# %%
# Function to tokenize with/without stemming and lemmatization
def stem_lem(
    word,
    stemming_enabled,
    lemmatization_enabled,
    stemmer=SnowballStemmer('english'),
    lemmatizer=WordNetLemmatizer(),
):
    if (lemmatization_enabled is True):
        word = lemmatizer.lemmatize(word)

    if (stemming_enabled is True):
        word = stemmer.stem(word)

    return word


# %%
def custom_tokenizer(row, stemming_enabled, lemmatization_enabled, numbers_cleaned, pattern, stop_words, return_tokens=False):

    tokens = [
        stem_lem(str(unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')),
                 stemming_enabled=stemming_enabled, lemmatization_enabled=lemmatization_enabled)
        for word in preprocess_documents(re.sub(pattern[str(numbers_cleaned)], ' ', row.strip().lower()))
        #                 for word in simple_preprocess(re.sub(pattern[str(numbers_cleaned)], ' ', row.strip().lower()), deacc=True)
        if (word not in stop_words) and (word.isalpha())
    ]
    if return_tokens is True:
        return tokens
    elif return_tokens is False:
        return ' '.join(tokens)

# %%
# Function to create bi and tri grams


def get_gensim_n_grams(unigram_sentences):

    # Bigrams
    bigram = Phraser(Phrases(
        unigram_sentences, connector_words=ENGLISH_CONNECTOR_WORDS, min_count=1, threshold=1))
    bigram_sentences = bigram[unigram_sentences]

    # Trigrams
    trigram = Phraser(Phrases(
        bigram_sentences, connector_words=ENGLISH_CONNECTOR_WORDS, min_count=1, threshold=1))
    trigram_sentences = trigram[bigram_sentences]

    return bigram, trigram, list(bigram_sentences), list(trigram_sentences)


# %%
def get_corpus_and_dictionary(row, ngram_number, embedding_library):

    row[f'{ngram_number}grams_{embedding_library}_dictionary'] = corpora.Dictionary(
        [
            row[f'{ngram_number}grams_{embedding_library}']
        ]
    )

    row[f'{ngram_number}grams_{embedding_library}_corpus'] = [
        row[f'{ngram_number}grams_{embedding_library}_dictionary'].doc2bow(
            row[f'{ngram_number}grams_{embedding_library}']
        )
    ]

    row[f'{ngram_number}grams_{embedding_library}_tfidf'] = TfidfModel(
        row[f'{ngram_number}grams_{embedding_library}_corpus']
    )

    # row[f'{ngram_number}grams_{embedding_library}_tfidf_matrix'] = [
    #     row[f'{ngram_number}grams_{embedding_library}_tfidf'][doc]
    #     for doc in row[f'{ngram_number}grams_{embedding_library}_corpus']
    # ]

    return row


# %%
def get_word_num_and_frequency(row, text_col):

    row['num_words'] = len(str(row[f'{text_col}']).split())
    row['num_unique_words'] = len(set(str(row[f'{text_col}']).split()))
    row['num_chars'] = len(str(row[f'{text_col}']))
    row['num_punctuations'] = len(
        [c for c in str(row[f'{text_col}']) if c in string.punctuation])

    return row


# %%
def get_abs_frequency(row, text_col, ngram_number, embedding_library):

    abs_word_freq = defaultdict(int)
    for word in row[f'{ngram_number}grams_{embedding_library}']:
        abs_word_freq[word] += 1

        abs_wtd_df = (
            pd.DataFrame.from_dict(abs_word_freq, orient='index')
            .rename(columns={0: 'abs_word_freq'})
            .sort_values(by=['abs_word_freq'], ascending=False)
        )
        abs_wtd_df.insert(
            1, 'abs_word_perc', value=abs_wtd_df['abs_word_freq'] / abs_wtd_df['abs_word_freq'].sum())
        abs_wtd_df.insert(2, 'abs_word_perc_cum',
                          abs_wtd_df['abs_word_perc'].cumsum())

        row[f'{ngram_number}grams_{embedding_library}_abs_word_freq'] = str(
            abs_wtd_df['abs_word_freq'].to_dict())
        row[f'{ngram_number}grams_{embedding_library}_abs_word_perc'] = str(
            abs_wtd_df['abs_word_perc'].to_dict())
        row[f'{ngram_number}grams_{embedding_library}_abs_word_perc_cum'] = str(
            abs_wtd_df['abs_word_perc_cum'].to_dict())

    return row


# %%
def convert_frequency(value, freq):

    return dict(functools.reduce(operator.add, map(collections.Counter, [ast.literal_eval(row) for idx, row in value['df'][f'{freq}'].items() if isinstance(row, str) and str(row) != 'nan' and type(row) != float and len(row) != 0])))


# %%
def build_train_word2vec(
    sector, df, ngram_number, embedding_library, sectors_enabled=False, size=300, print_enabled=True,
    words=['she', 'he', 'support', 'leader', 'management', 'team',
           'business', 'customer', 'risk', 'build', 'computer', 'programmer'],
    t=time.time(), cores=multiprocessing.cpu_count(), args=get_args(),
):
    sentences = df[f'{ngram_number}grams_{embedding_library}'].values

    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=size,
        min_count=0,
        window=2,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
        workers=cores - 1,
        sg=1,
    )

    w2v_model.build_vocab(sentences, progress_per=10000)
    if sectors_enabled is True and sector != None and print_enabled is True:
        print(
            f'Time to train the model for {sector} {size}: {round((time.time() - t) / 60, 2)} mins')

    w2v_model.train(
        sentences,
        total_examples=w2v_model.corpus_count,
        epochs=30,
        report_delay=1,
    )

    if sectors_enabled is True and sector != None and print_enabled is True:
        print(
            f'Time to build w2v_vocab for {sector} {size}: {round((time.time() - t) / 60, 2)} mins')
    w2v_vocab = list(w2v_model.wv.index_to_key)

    if args['print_enabled'] is True:
        print(f'Checking words form list of length {len(words)}')
        print(f'WORDS LIST: {words}')

        if sectors_enabled is True and sector != None:
            for word in words:
                print(f'Checking word:\n{word.upper()}:')
                try:
                    # print(f'{sector} 300: {w2v_model_300.wv[word]}')
                    # print(f'{sector} 100: {w2v_model_100.wv[word]}')
                    print(
                        f'Length of {sector} {size} model vobal: {len(w2v_vocab)}')
                    print(
                        f'{sector} {size} - Positive most similar to {word}: {w2v_model.wv.most_similar(positive=word, topn=5)}')
                    print(
                        f'{sector} {size} - Negative most similar to {word}: {w2v_model.wv.most_similar(negative=word, topn=5)}')

                except KeyError as e:
                    print(e)

    return w2v_vocab, w2v_model


# %%
def word2vec_embeddings(sentences, w2v_vocab, w2v_model, size=300):

    sentences = [word for word in sentences if word in w2v_vocab]

    return np.mean(w2v_model.wv[sentences], axis='index') if sentences else np.zeros(size)


# %%
# def build_train_glove(
#     sector, df, ngram_number, embedding_library, sectors_enabled = False, size = 300, print_enabled = True,
#     words = ['she', 'he', 'support', 'leader', 'management', 'team', 'business', 'customer', 'risk', 'build', 'computer', 'programmer'],
#     t = time.time(), cores = multiprocessing.cpu_count(), args=get_args(),
# ):
#     sentences = df[f'{ngram_number}grams_{embedding_library}'].values
#     glove_model = Glove(
#         sentences=sentences,
#         size=size,
#         min_count=0,
#         window=2,
#         sample=6e-5,
#         alpha=0.03,
#         min_alpha=0.0007,
#         negative=20,
#         workers=cores - 1,
#         sg = 1,
#     )

#     glove_model.build_vocab(sentences, progress_per=10000)
#     if sectors_enabled is True and sector != None and print_enabled is True:
#         print(f"Time to train the model for {sector} {size}: {round((time.time() - t) / 60, 2)} mins")

#     glove_model.train(
#         sentences,
#         total_examples=glove_model.corpus_count,
#         epochs=30,
#         report_delay=1,
#     )

#     if sectors_enabled is True and sector != None and print_enabled is True:
#         print(f"Time to build vocab for {sector} {size}: {round((time.time() - t) / 60, 2)} mins")
#     vocab = list(glove_model.wv.index_to_key)

#     if args['print_enabled'] is True:
#         print(f'Checking words form list of length {len(words)}')
#         print(f'WORDS LIST: {words}')

#         if sectors_enabled is True and sector != None:
#             for word in words:
#                 print(f'Checking word:\n{word.upper()}:')
#                 try:
#                     # print(f'{sector} 300: {glove_model_300.wv[word]}')
#                     # print(f'{sector} 100: {glove_model_100.wv[word]}')
#                     print(f'Length of {sector} {size} model vobal: {len(vocab)}')
#                     print(f'{sector} {size} - Positive most similar to {word}: {glove_model.wv.most_similar(positive=word, topn=5)}')
#                     print(f'{sector} {size} - Negative most similar to {word}: {glove_model.wv.most_similar(negative=word, topn=5)}')

#                 except KeyError as e:
#                     print(e)

#     return glove_vocab, glove_model


# # %%
# def glove_embeddings(sentences, glove_vocab, glove_model, size=300):

#     sentences = [word for word in sentences if word in glove_vocab]

#     return np.mean(w2v_model.wv[sentences], axis='index') if sentences else np.zeros(size)


# %%
def build_train_fasttext(
    sector, df, ngram_number, embedding_library, sectors_enabled=False, size=300, print_enabled=True,
    words=['she', 'he', 'support', 'leader', 'management', 'team',
           'business', 'customer', 'risk', 'build', 'computer', 'programmer'],
    t=time.time(), cores=multiprocessing.cpu_count(), args=get_args(),
):
    sentences = df[f'{ngram_number}grams_{embedding_library}'].values

    ft_model = FastText(
        sentences=sentences,
        vector_size=size,
        min_count=0,
        window=2,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
        workers=cores - 1,
        sg=1,
    )

    ft_model.build_vocab(sentences, progress_per=10000)
    if sectors_enabled is True and sector != None and print_enabled is True:
        print(
            f'Time to train the model for {sector} {size}: {round((time.time() - t) / 60, 2)} mins')

    ft_model.train(
        sentences,
        total_examples=ft_model.corpus_count,
        epochs=30,
        report_delay=1,
    )

    if sectors_enabled is True and sector != None and print_enabled is True:
        print(
            f'Time to build vocab for {sector} {size}: {round((time.time() - t) / 60, 2)} mins')
    ft_vocab = list(ft_model.wv.index_to_key)

    if args['print_enabled'] is True:
        print(f'Checking words form list of length {len(words)}')
        print(f'WORDS LIST: {words}')

        if sectors_enabled is True and sector != None:
            for word in words:
                print(f'Checking word:\n{word.upper()}:')
                try:
                    # print(f'{sector} 300: {ft_model_300.wv[word]}')
                    # print(f'{sector} 100: {ft_model_100.wv[word]}')
                    print(
                        f'Length of {sector} {size} model vobal: {len(ft_vocab)}')
                    print(
                        f'{sector} {size} - Positive most similar to {word}: {ft_model.wv.most_similar(positive=word, topn=5)}')
                    print(
                        f'{sector} {size} - Negative most similar to {word}: {ft_model.wv.most_similar(negative=word, topn=5)}')

                except KeyError as e:
                    print(e)

    return ft_vocab, ft_model


# %%
def fasttext_embeddings(sentences, ft_vocab, ft_model, size=300):

    sentences = [word for word in sentences if word in ft_vocab]

    return np.mean(ft_model.wv[sentences], axis='index') if sentences else np.zeros(size)


# %%
# def sent2vec_embedding(sentences, sent2vec_model = sent2vec.Sent2vecModel(), args=get_args()):

#     sent2vec_model.load_model('model.bin', inference_mode=True)
#     bi_embs, vocab = sent2vec_model.get_bigram_embeddings()

#     return model.embed_sentence(sentences)


# %%
def get_glove(glove_file=validate_path(f'{glove_path}glove.840B.300d.txt')):

    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf8') as glove:

        for line in glove:
            values = line.split()
            word = values[0]

            with contextlib.suppress(ValueError):
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

    print(f'Found {len(embeddings_index)} word vectors.')

    return embeddings_index


# %%
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# %%
# def huggingface_similarity(sentences, model = BertModel.from_pretrained('bert-base-uncased', trust_remote_code=True), args=get_args()):

#     embeddings = model.encode(sentences, show_progress_bar=True)
#     cos_sim = util.cos_sim(embeddings, embeddings)

#     #Sort list by the highest cosine similarity score
#     all_sentence_combinations = sorted([[cos_sim[i][j], i, j] for i in range(len(cos_sim)-1) for j in range(i+1, len(cos_sim))], key=lambda x: x[0], reverse=True)

#     if args['print_enabled'] is True:
#         print('Top-5 most similar pairs:')
#         for score, i, j in all_sentence_combinations[:5]:
#             print(f'{sentences[i]} \t {sentences[j]} \t {cos_sim[i][j]:.4f}')

#     if args['save_enabled'] is True:
#         model.to_json_file('bert_config.json')

#     return all_sentence_combinations, embeddings


# %%
def get_sentiment(df_jobs_to_be_processed, text_col, sentiment_range=(-1, 1)):

    df_jobs_to_be_processed['sentiment'] = df_jobs_to_be_processed[text_col].progress_apply(
        lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'] if isinstance(x, str) else np.nan)

    # rescaled
    if sentiment_range != (-1, 1):
        df_jobs_to_be_processed['sentiment'] = preprocessing.MinMaxScaler(
            feature_range=sentiment_range).fit_transform(df_jobs_to_be_processed[['sentiment']])
    # print(df_jobs_to_be_processed[['sentiment']].describe().T)

    return df_jobs_to_be_processed


# %%
# reference: https://www.machinelearningplus.com/nlp/cosine-similarity/
def create_soft_cossim_matrix(documents, cat_order, similarity_matrix):
    len_array = np.arange(len(documents))
    xx, yy = np.meshgrid(len_array, len_array)
    cossim_mat = pd.DataFrame([[round(similarity_matrix.inner_product(
        documents[i], documents[j], normalized=(True, True)), 2) for i, j in zip(x, y)] for y, x in zip(xx, yy)])
    cossim_mat.columns = cat_order
    cossim_mat.index = cat_order

    return cossim_mat


# %%
# Remove punctuations, tokenize, and lemmatize
def simple_preprocess_df(
    df_jobs_to_be_processed=pd.DataFrame(),
    stem_lem=stem_lem,
    args=get_args(),
):
    from setup_module.params import (
        drop_cols_enabled,
        embedding_from_function,
        embedding_libraries_list,
        embedding_models_dict,
        id_dict_new,
        lemmatization_enabled,
        main_from_function,
        n_gram,
        ngrams_enabled,
        ngrams_from_funtion,
        ngrams_list,
        nltk_ngrams_dict,
        numbers_cleaned,
        pattern,
        preprocessed_from_function,
        preprocessing_enabled,
        stemming_enabled,
        stop_words,
        text_col,
    )

    if embedding_from_function is True:
        print('df_jobs_labeled from embedding function.')
        if ngrams_from_funtion is True:
            print('df_jobs_labeled from n_grams function.')
            if preprocessed_from_function is True:
                print('df_jobs_labeled from preprocessed function.')
                if main_from_function is True:
                    print('df_jobs_labeled from main function.')
                    if args['print_enabled'] is True:
                        print('Opening df_jobs_to_be_processed from file.')
                    df_jobs_to_be_processed, df_labeled_dict = open_and_clean_labeled_excel(
                        id_dict_new=id_dict_new)
                elif main_from_function is False:
                    print('Using given version of main df_jobs_labeled.')

                    if df_jobs_to_be_processed.empty:
                        try:
                            df_jobs_to_be_processed = pd.read_pickle(
                                f'{args["df_dir"]}df_jobs_labeled_unprocessed.{args["file_save_format"]}'
                            )
                        except Exception:
                            df_jobs_to_be_processed = pd.read_csv(
                                f'{args["df_dir"]}df_jobs_labeled_unprocessed.{args["file_save_format_backup"]}'
                            )
                            for col in df_jobs_to_be_processed.columns:
                                if 'grams_' in col:
                                    df_jobs_to_be_processed[col] = df_jobs_to_be_processed[col].progress_apply(
                                        lambda x: ast.literal_eval(str(x)))

#                         if 'Unnamed: 0' in df_jobs_to_be_processed.columns:
                        df_jobs_to_be_processed = df_jobs_to_be_processed.drop(
                            ['Unnamed: 0'],
                            axis='columns',
                            errors='ignore',
                        )

                # Remove stopwords
                if preprocessing_enabled is True:
                    df_jobs_to_be_processed['Job Description_cleaned'] = (
                        df_jobs_to_be_processed['Job Description']
                        .progress_apply(
                            lambda row: custom_tokenizer(str(row), numbers_cleaned=numbers_cleaned, stemming_enabled=stemming_enabled,
                                                         lemmatization_enabled=lemmatization_enabled, pattern=pattern, stop_words=stop_words, return_tokens=False)
                        )
                    )

                    # Get word frequencies
                    df_jobs_to_be_processed = df_jobs_to_be_processed.progress_apply(
                        lambda row: get_word_num_and_frequency(row=row, text_col=text_col), axis='columns')

                    # Get sentiment
                    df_jobs_to_be_processed = get_sentiment(
                        df_jobs_to_be_processed, text_col=text_col)

                    if args['save_enabled'] is True:
                        print('Saving df_jobs_labeled from preprocessed function.')
                        df_jobs_to_be_processed.to_pickle(
                            f'{args["df_dir"]}df_jobs_labeled_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}')
                        df_jobs_to_be_processed.to_csv(
                            f'{args["df_dir"]}df_jobs_labeled_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}')

            elif preprocessed_from_function is False:
                print('Using given version of preprocessed df_jobs_labeled.')

                try:
                    df_jobs_to_be_processed = pd.read_pickle(
                        f'{args["df_dir"]}df_jobs_labeled_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}'
                    )
                except Exception:
                    df_jobs_to_be_processed = pd.read_csv(
                        f'{args["df_dir"]}df_jobs_labeled_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}'
                    )
                    for col in df_jobs_to_be_processed.columns:
                        if 'grams_' in col:
                            df_jobs_to_be_processed[col] = df_jobs_to_be_processed[col].progress_apply(
                                lambda x: ast.literal_eval(str(x)))

#                 if 'Unnamed: 0' in df_jobs_to_be_processed.columns:
                df_jobs_to_be_processed = df_jobs_to_be_processed.drop(
                    ['Unnamed: 0'], axis='columns', errors='ignore')

            # Tokenize on Ngrams
            if ngrams_enabled is True:

                # Tokenize
                # Custom
                df_jobs_to_be_processed['1grams_all'] = df_jobs_to_be_processed['Job Description_cleaned'].progress_apply(lambda sentences: ast.literal_eval(custom_tokenizer(str(
                    sentences), numbers_cleaned=numbers_cleaned, stemming_enabled=stemming_enabled, lemmatization_enabled=lemmatization_enabled, pattern=pattern, stop_words=stop_words, return_tokens=True)))
                # BERT
                BERTMODEL = 'bert-base-uncased'
                bert_tokenizer = BertTokenizer.from_pretrained(
                    BERTMODEL, strip_accents=True, trust_remote_code=True)
                df_jobs_to_be_processed['1grams_bert'] = df_jobs_to_be_processed['Job Description_cleaned'].progress_apply(
                    lambda sentence: bert_tokenizer.tokenize(str(sentence)).to(device))

                if args['save_enabled'] is True:
                    print('Saving df_jobs_labeled after tokenization.')
                    df_jobs_to_be_processed.to_pickle(
                        f'{args["df_dir"]}df_jobs_labeled_tokenized_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}')
                    df_jobs_to_be_processed.to_csv(
                        f'{args["df_dir"]}df_jobs_labeled_tokenized_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}')

                try:
                    df_jobs_to_be_processed = pd.read_pickle(
                        f'{args["df_dir"]}df_jobs_labeled_tokenized_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}'
                    )
                except Exception:
                    df_jobs_to_be_processed = pd.read_csv(
                        f'{args["df_dir"]}df_jobs_labeled_tokenized_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}'
                    )
                    for col in df_jobs_to_be_processed.columns:
                        if 'grams_' in col:
                            df_jobs_to_be_processed[col] = df_jobs_to_be_processed[col].progress_apply(
                                lambda x: ast.literal_eval(str(x)))

#                 if 'Unnamed: 0' in df_jobs_to_be_processed.columns:
                df_jobs_to_be_processed = df_jobs_to_be_processed.drop(
                    ['Unnamed: 0'], axis='columns', errors='ignore')

                # Gensim
                bigram_transformer, trigram_transformer, df_jobs_to_be_processed['2grams_gensim'], df_jobs_to_be_processed['3grams_gensim'] = get_gensim_n_grams(
                    df_jobs_to_be_processed['1grams_all'])
                df_jobs_to_be_processed['123grams_gensim'] = df_jobs_to_be_processed['1grams_all'] + \
                    df_jobs_to_be_processed['2grams_gensim'] + \
                    df_jobs_to_be_processed['3grams_gensim']

                # NLTK
                for ngram_number, ngram_nltk in nltk_ngrams_dict.items():
                    df_jobs_to_be_processed[f'{str(ngram_number)}grams_nltk'] = df_jobs_to_be_processed['1grams_all'].reset_index(
                        drop=True).progress_apply(lambda unigram: list(ngram_nltk(unigram)))

                df_jobs_to_be_processed['123grams_nltk'] = df_jobs_to_be_processed['1grams_all'] + \
                    df_jobs_to_be_processed['2grams_nltk'] + \
                    df_jobs_to_be_processed['3grams_nltk']

                if args['save_enabled'] is True:
                    print('Saving df_jobs_labeled from n_grams function.')
                    df_jobs_to_be_processed.to_pickle(
                        f'{args["df_dir"]}df_jobs_labeled_ngram_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}')
                    df_jobs_to_be_processed.to_csv(
                        f'{args["df_dir"]}df_jobs_labeled_ngram_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}')

                try:
                    df_jobs_to_be_processed = pd.read_pickle(
                        f'{args["df_dir"]}df_jobs_labeled_ngram_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}'
                    )
                except Exception:
                    df_jobs_to_be_processed = pd.read_csv(
                        f'{args["df_dir"]}df_jobs_labeled_ngram_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}'
                    )
                    for col in df_jobs_to_be_processed.columns:
                        if 'grams_' in col:
                            df_jobs_to_be_processed[col] = df_jobs_to_be_processed[col].progress_apply(
                                lambda x: ast.literal_eval(str(x)))

#                 if 'Unnamed: 0' in df_jobs_to_be_processed.columns:
                df_jobs_to_be_processed = df_jobs_to_be_processed.drop(
                    ['Unnamed: 0'], axis='columns', errors='ignore')

                # Get absolute word frequencies
                for embedding_library, ngram_number in tqdm_product(embedding_libraries_list, ngrams_list):
                    if ngram_number == 1:
                        embedding_library = 'all'
                    df_jobs_to_be_processed = df_jobs_to_be_processed.progress_apply(lambda row: get_abs_frequency(
                        row=row, text_col=text_col, ngram_number=ngram_number, embedding_library=embedding_library), axis='columns')

                # Get dictionary and corpus
                for ngram_number in ngrams_list:
                    if ngram_number != 1:
                        df_jobs_to_be_processed = df_jobs_to_be_processed.progress_apply(lambda row: get_corpus_and_dictionary(
                            row=row, ngram_number=ngram_number, embedding_library='gensim'), axis='columns')

                if args['save_enabled'] is True:
                    print('Saving df_jobs_labeled from n_grams function.')
                    df_jobs_to_be_processed.to_pickle(
                        f'{args["df_dir"]}df_jobs_labeled_corpus_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}')
                    df_jobs_to_be_processed.to_csv(
                        f'{args["df_dir"]}df_jobs_labeled_corpus_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}')

        elif ngrams_from_funtion is False:
            print('Using given version of corpus df_jobs_labeled.')

            try:
                df_jobs_to_be_processed = pd.read_pickle(
                    f'{args["df_dir"]}df_jobs_labeled_corpus_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}'
                )
            except Exception:
                df_jobs_to_be_processed = pd.read_csv(
                    f'{args["df_dir"]}df_jobs_labeled_corpus_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}'
                )
                for col in df_jobs_to_be_processed.columns:
                    if 'grams_' in col:
                        df_jobs_to_be_processed[col] = df_jobs_to_be_processed[col].progress_apply(
                            lambda x: ast.literal_eval(str(x)))

#             if 'Unnamed: 0' in df_jobs_to_be_processed.columns:
            df_jobs_to_be_processed = df_jobs_to_be_processed.drop(
                ['Unnamed: 0'], axis='columns', errors='ignore')

            # Get embeddings
            for embedding_library, ngram_number in tqdm_product(embedding_libraries_list, ngrams_list):
                if ngram_number == 1:
                    embedding_library = 'all'

                print(
                    f'Building {ngram_number}grams_{embedding_library} model and vocabulary.')

                for embed_model_name, embed_func_list in embedding_models_dict.items():
                    build_train_func, embed_func, model_loader = embed_func_list

                    vocab, model = build_train_func(
                        df=df_jobs_to_be_processed,
                        ngram_number=ngram_number,
                        embedding_library=embedding_library,
                        sector=None,
                    )
                    print(f'Getting {embed_model_name} embeddings.')
                    df_jobs_to_be_processed[
                        f'{ngram_number}grams_{embedding_library}_mean_{embed_model_name}_embeddings'
                    ] = df_jobs_to_be_processed[
                        f'{ngram_number}grams_{embedding_library}'
                    ].progress_apply(
                        lambda sentences: embed_func(sentences, vocab, model)
                    )
                    if args['save_enabled'] is True:
                        model.save(
                            f'{args["embeddings_save_path"]}{ngram_number}grams_{embedding_library}_{embed_model_name}_model.model')

                # Sent2Vec
                print('Getting sent2vec embeddings.')
                embeddings_index = get_glove()
                df_jobs_to_be_processed[f'{ngram_number}grams_{embedding_library}_sent2vec_embeddings'] = df_jobs_to_be_processed[f'{ngram_number}grams_{embedding_library}'].progress_apply(
                    lambda sentences: sent2vec(sentences, embeddings_index=embeddings_index, external_glove=True, extra_preprocessing_enabled=False))
                print('Done getting sent2vec embeddings.')

                # # HuggingFace
                # print('Getting huggingface embeddings.')
                # bert_model=TFBertModel.from_pretrained('bert-base-uncased', trust_remote_code=True)
                # df_jobs_to_be_processed[f'{ngram_number}grams_{embedding_library}_huggingface_embeddings'] = df_jobs_to_be_processed[f'{ngram_number}grams_{embedding_library}'].progress_apply(lambda sentences: bert_model(sentences))
                # print('Done getting huggingface embeddings.')

            if args['save_enabled'] is True:
                print(
                    'Saving df_jobs_labeled from word2vec and fasttext embedding function.')
                df_jobs_to_be_processed.to_pickle(
                    f'{args["df_dir"]}df_jobs_labeled_embeddings_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}')
                df_jobs_to_be_processed.to_csv(
                    f'{args["df_dir"]}df_jobs_labeled_embeddings_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}')

    elif embedding_from_function is False:
        print('Using given version of embeddings df_jobs_labeled.')

        try:
            df_jobs_to_be_processed = pd.read_pickle(
                f'{args["df_dir"]}df_jobs_labeled_embeddings_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}'
            )
        except Exception:
            df_jobs_to_be_processed = pd.read_csv(
                f'{args["df_dir"]}df_jobs_labeled_embeddings_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}'
            )
            for col in df_jobs_to_be_processed.columns:
                if 'grams_' in col:
                    df_jobs_to_be_processed[col] = df_jobs_to_be_processed[col].progress_apply(
                        lambda x: ast.literal_eval(str(x)))

#         if 'Unnamed: 0' in df_jobs_to_be_processed.columns:
        df_jobs_to_be_processed = df_jobs_to_be_processed.drop(
            ['Unnamed: 0'], axis='columns', errors='ignore')

    if drop_cols_enabled is True:
        df_jobs_to_be_processed = df_jobs_to_be_processed.drop(
            ['Gender', 'Age'], axis='columns', errors='ignore')

    if args['save_enabled'] is True:

        df_jobs_to_be_processed.to_pickle(
            f'{args["df_dir"]}df_jobs_labeled_final_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format"]}')
        df_jobs_to_be_processed.to_csv(
            f'{args["df_dir"]}df_jobs_labeled_final_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{args["file_save_format_backup"]}')

    return df_jobs_to_be_processed


# %% Function to for inital visualization
def get_and_viz_df_dict(dataframes, df_loc, args=get_args()):

    for df_name, df_df in dataframes.items():
        print('='*80)
        print(df_name.upper())
        print('='*80)
        dataframes[df_name] = pd.read_pickle(
            f'{args["df_dir"]}{df_name}{df_loc}.{args["file_save_format"]}')
        dataframes[df_name] = categorize_df_gender_age(dataframes[df_name])
        dataframes[df_name].info()
        if 'mean' not in df_name:
            # Visualize data balance
            get_viz(df_name, df_df, dataframes)

    return dataframes


# %%
# # Function to clean df and get data and target
# def df_jobs_labeled_to_df_jobs(df_jobs_labeled, col, text_col='Job Description_cleaned'):
#     # Clean DF
#     df_jobs = df_jobs_labeled.drop(
#         [
#             columns
#             for columns in df_jobs_labeled.columns
#             if (str(col) != str(columns))
#             and (str(text_col) != str(columns))
#         ],
#         axis='columns',
#     )

#     df_jobs = df_jobs_labeled.dropna(subset=[col, text_col], how='any')

#     if (df_jobs[str(col)].isna().sum() > 0) or (df_jobs[str(col)].isnull().sum() > 0):
#         print(f'{df_jobs.isna()} missing values found in df_jobs')

#     # df_jobs.fillna('0')

#     # Specify data and target columns and make array
#     df_jobs[f'{str(text_col)}_data'] = df_jobs[f'{str(text_col)}'].astype('str').values
#     df_jobs[f'{str(col)}_target'] = df_jobs[str(col)].astype('int64').values
#     data = df_jobs[f'{str(text_col)}'].astype('str').values
#     target = df_jobs[str(col)].astype('int64').values

#     return df_jobs, data, target


# %%
def resample_data(X_train, y_train, col, resampling_enabled, resampling_method):
    if (resampling_enabled is True) and (col == 'Warmth'):
        print(f'Resampling {col} to fix imbalance.')
        print('=' * 20)
        print('-' * 20)
        print(f'Original dataset shape {y_train.shape}')
        print(f'Original dataset, counts of label "1": {sum(y_train==1)}')
        print(f'Original dataset, counts of label "0": {sum(y_train==0)}')

        X_train_resampled, y_train_resampled = resampling_method.fit_resample(
            X_train, y_train
        )
        print(f'Resampled dataset shape {y_train.shape}')
        print(f'Resampled dataset, counts of label "1": {sum(y_train==1)}')
        print(f'Resampled dataset, counts of label "0": {sum(y_train==0)}')
        print('-' * 20)

    return X_train_resampled, y_train_resampled


# %%
# Plot training dfs
def plot_df(
    df_jobs,
    label,
    analysis_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    bins=[0, 0.25, 0.5, 1, 1.25, 1.5],
    color='blue',
):
    for col in analysis_columns:
        plt.hist(df_jobs[str(col)], bins=bins, color=color)
        plt.xlabel(f'{str(label)} - {str(col)} Historgram')
        plt.ylabel('Count')
        plt.xticks([0.25, 1.25], [0, 1])
        plt.show()

    # Compute the correlation matrix
    corr = df_jobs.loc[:, [col for col in analysis_columns]].corr()

    # Generate a mask for the upper triangle (hide the upper triangle)
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr, mask=mask, square=True, linewidths=0.25, cmap='YlOrRd', vmin=0, vmax=1
    )
    plt.show()


# %%
def plot_confusion_matrix_percentage(
    col, cm, classifier_name, vectorizer_name, cmap=cmap_colorblind_blues_and_grays
):
    cm_title = f'{col} Confusion matrix: {classifier_name} + {vectorizer_name}'
    print('\n')
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = [f'{value:0.0f}\n' for value in cm.flatten()]
    group_percentages = [f'{value:.2%}' for value in cm.flatten() / np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(cm.shape[0], cm.shape[1])
    heatmap = sns.heatmap(
        cm, cmap='PuBu', annot=labels, fmt='', annot_kws={'size': 12.0}
    )
    plt.xlabel('Predicted', fontsize=12.0)
    plt.ylabel('Actual', fontsize=12.0)
    plt.title(
        f'{col} Confusion matrix: {classifier_name} + {vectorizer_name}',
        fontsize=14.0,
    )
    # plt.tight_layout()
    plt.show()
    print('-' * 20)

    return heatmap


def evaluate_prediction(
    X_test,
    y_test,
    my_tags,
    df_jobs_labeled,
    col,
    scoring,
    print_enabled,
    title='Confusion matrix',
):
    with contextlib.suppress(Exception):
        y_test_pred = classifier.predict(X_test)

        print('-' * 20)
        print(
            f'Recall: {metrics.recall_score(y_test, y_test_pred, pos_label = 1, labels = [1,0])}'
        )
        print(f'Accuracy: {metrics.accuracy_score(y_test, y_test_pred)}')
        print(
            f'Precision: {metrics.precision_score(y_test, y_test_pred, pos_label = 1, labels = [1,0])}'
        )
        best_threshold, best_score = calculate_best_threshold(
            y_test, y_test_pred, scoring, print_enabled)
        print(f'Best {scoring} Threshold: {best_threshold}')
        print(f'Best {scoring} Score: {best_score}')
        print(
            f'Matthews correlation coefficient: {metrics.matthews_corrcoef(y_test, y_test_pred)}'
        )
        print(f'F1: {metrics.f1_score(y_test, y_test_pred)}')
        print('\n')
        cm = confusion_matrix(y_test, y_test_pred, labels=my_tags)
        print(f'Confusion matrix:\n{cm}')
        print('(row=expected, col=predicted)')

        cm_normalized = cm.astype(
            'float') / cm.sum(axis='columns')[:, np.newaxis]
        plot_confusion_matrix(
            df_jobs_labeled, col, cm_normalized, my_tags, title + ' Normalized'
        )
        skplt.metrics.plot_confusion_matrix(
            y_test, y_test_pred, normalize=True)
        plt.show()


def get_label_n(y, y_pred, n=None):

    # enforce formats of inputs
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    check_consistent_length(y, y_pred)
    y_len = len(y)  # the length of targets

    # calculate the percentage of outliers
    if n is not None:
        outliers_fraction = n / y_len
    else:
        outliers_fraction = np.count_nonzero(y) / y_len

    threshold = np.percentile(y_pred, 100 * (1 - outliers_fraction))
    y_pred = (y_pred > threshold).astype('int')

    return y_pred


def precision_n_scores(y, y_pred, n=None):

    # turn raw prediction decision scores into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return metrics.precision_score(y, y_pred)


def evaluate_print(clf_name, y, y_pred):

    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    print('{clf_name} ROC:{roc}, precision @ rank n:{prn}'.format(
        clf_name=clf_name,
        roc=np.round(roc_auc_score(y, y_pred), decimals=4),
        prn=np.round(precision_n_scores(y, y_pred), decimals=4)))


# %%
def explain_model(test, y_test, y_test_pred, y_test_prob_pred, y_train):
    # select observation
    i = 0
    txt_instance = test[f'{str(text_col)}'].astype('str').iloc[i]
    # check true value and predicted value
    print('True:', y_test[i], '--> Pred:', y_test_pred[i],
          '| Prob:', round(np.max(y_test_prob_pred[i]), 2))
    # show explanation
    explainer = lime_text.LimeTextExplainer(class_names=np.unique(y_train))

    return explainer.explain_instance(txt_instance, model.y_test_prob_pred, num_features=3)

# %%
# Search Optimization


def grid_search_wrapper(
    X_test,
    y_test,
    X_train,
    y_train,
    clf,
    params_grid,
    cv,
    scorers,
    n_jobs=-1,
    refit_score='recall_score',
):
    grid_search = GridSearchCV(
        clf,
        param_grid=params_grid,
        scoring=scorers,
        refit=refit_score,
        cv=cv,
        return_train_score=True,
        n_jobs=n_jobs,
    )
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)

    print(f'Best params for {refit_score}')
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print(
        f'\nConfusion matrix of {vectorizer_name} with {classifier_name} optimized for {refit_score} on the test data:'
    )
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            columns=['pred_neg', 'pred_pos'],
            index=['neg', 'pos'],
        )
    )

    return grid_search

# %%


def calculate_best_threshold(y_test, y_test_pred, scoring, print_enabled):
    best_threshold = -1
    best_score = -1
    for threshold in np.arange(0.01, 0.801, 0.01):
        threshold = np.round(threshold, 2)

        # globals()[f'metrics.{scoring.lower()}_score']
        if scoring.lower() == 'recall':
            scorer = metrics.recall_score
        elif scoring.lower() == 'f1 score':
            scorer = metrics.f1_score
        else:
            raise ValueError(f'{scoring.title()} is not a valid score')

        emb_model_score = scorer(y_true=y_test, y_pred=(
            y_test_pred > threshold).astype(int))
        if emb_model_score > best_score:
            best_score = emb_model_score
            best_threshold = threshold
        if print_enabled:
            print(f'{scoring.title()} at threshold {threshold}: {emb_model_score}')
    print(f'{scoring.title()} at best threshold {best_threshold}: {best_score}')

    return best_threshold, best_score
# %%


def predict(
    X_test,
    y_test,
    classifier,
    classifier_name,
    col,
    scoring,
    df_jobs_labeled,
    print_enabled,
):

    with contextlib.suppress(Exception):
        my_tags = df_jobs_labeled[str(col)].unique()
        y_test_pred = classifier.predict(X_test)

        evaluate_print(classifier_name + '   |   ', y_test, y_test_pred)
        report_test = classification_report(y_test, y_test_pred)
        evaluate_prediction(
            X_test,
            y_test,
            my_tags,
            df_jobs_labeled,
            col,
            scoring,
            print_enabled,
        )

    return report_test


# %%
def evaluation(y_test, y_test_pred, scoring, print_enabled, title='Confusion Matrix'):
    cm = metrics.confusion_matrix(y_test, y_test_pred)
    precision = metrics.precision_score(
        y_test, y_test_pred, pos_label=1, labels=[1, 0])
    recall = metrics.recall_score(
        y_test, y_test_pred, pos_label=1, labels=[1, 0])
    accuracy = metrics.accuracy_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_test_pred)
    best_threshold, best_score = calculate_best_threshold(
        y_test, y_test_pred, scoring, print_enabled)
    report = classification_report(y_test, y_test_pred)

    print('=' * 20)
    print(f'Classification Report:\n', report)
    print('-' * 20)
    print(f'Recall score: {recall}')
    print('-' * 20)
    print(f'Accuracy score: {accuracy}')
    print('-' * 20)
    print(f'Precision score: {precision}')
    print('-' * 20)
    print(f'F1 score: {f1}')
    print('-' * 20)
    print(f'Matthews correlation coefficient: {mcc}')
    print('-' * 20)
    print(f'Best {scoring} Threshold: {best_threshold}')
    print('-' * 20)
    print(f'Best {scoring} Score: {best_score}')
    print('-' * 20)
    print(f'Confusion Matrix:\n', cm)
    print('=' * 20)

    return cm, precision, recall, accuracy, f1, mcc, best_threshold, best_score, report


# %%
def evaluate_multi_classif(y_test, predicted, y_test_pred, scoring='recall', print_enabled=False, figsize=(15, 5)):
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    # Accuracy, Precision, Recall
    mcc = metrics.matthews_corrcoef(y_test, predicted)
    best_threshold, best_score = calculate_best_threshold(
        y_test, predicted, scoring, print_enabled)
    report = classification_report(y_test, predicted)

    accuracy = metrics.accuracy_score(y_test, predicted)
    f1 = metrics.f1_score(y_test, predicted)
    print('Accuracy:',  round(accuracy, 2))
    print('F1:', round(f1, 2))
    print('Detail:')
    print(report)

    # Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=cmap_colorblind_blues_and_grays, cbar=False)
    ax.set(xlabel='Pred', ylabel='True', xticklabels=classes,
           yticklabels=classes, title='Confusion matrix')
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    # Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test_array[:, i], y_test_pred[:, i])
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(
            classes[i], metrics.auc(fpr, tpr)))
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate',
              ylabel='True Positive Rate (Recall)', title='Receiver operating characteristic')
    ax[0].legend(loc='lower right')
    ax[0].grid(True)

    # Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_array[:, i], y_test_pred[:, i])
        ax[1].plot(recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(
            classes[i], metrics.auc(recall, precision)))
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall',
              ylabel='Precision', title='Precision-Recall curve')
    ax[1].legend(loc='best')
    ax[1].grid(True)
    plt.show()

    print('=' * 20)
    print(f'Classification Report:\n', report)
    print('-' * 20)
    print(f'Recall score: {recall}')
    print('-' * 20)
    print(f'Accuracy score: {accuracy}')
    print('-' * 20)
    print(f'Precision score: {precision}')
    print('-' * 20)
    print(f'F1 score: {f1}')
    print('-' * 20)
    print(f'Matthews correlation coefficient: {mcc}')
    print('-' * 20)
    print(f'Best {scoring} Score: {best_score}')
    print('-' * 20)
    print(f'Confusion Matrix:\n', cm)
    print('=' * 20)

    return cm, precision, recall, accuracy, f1, mcc, best_threshold, best_score, report


# %%
# Optimize
def optimization(
    model,
    X_test,
    X_train,
    y_test,
    y_train,
    y_test_pred,
    y_test_prob_log_pred,
    vectorizer,
    vectorizer_name,
    classifier,
    classifier_name,
    train_acc,
    test_acc,
    scoring
):
    # AUC
    if (
        hasattr(classifier, 'predict_log_proba')
        and hasattr(classifier, 'predict_proba')
    ) or (classifier_name == 'DecisionTreeClassifier'):
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test, y_test_prob_log_pred[:, 1], pos_label=1
        )
        roc_auc = metrics.auc(fpr, tpr)
        print('\n')
        print('-' * 20)
        print(
            f'ROC AUC for {classifier_name} with {vectorizer_name}:\n', roc_auc)

        # ROC
        plt.figure(figsize=(4, 4))
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc}')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.savefig('roccurve.svg', bbox_inches='tight')
        metrics.plot_roc_curve(classifier, X_test, y_test)
        plt.show()

        df_ROC_plot = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
        print('ROC CURVE FOR PREDICTED')
        bc = plot_metric_functions.BinaryClassification(y_test, y_test_pred, labels=['0', '1'])
        # Figures
        plt.figure(figsize=(5, 5))
        bc.plot_roc_curve()
        plt.show()
        # print(ggplot(df_ROC_plot, aes(x = 'fpr', ymin = 0, ymax = 'tpr')) + geom_line(aes(y = 'tpr')) + geom_area(alpha = 0.2) + ggtitle(f'ROC Curve w/ AUC = {str(roc_auc)}'))

        y_prob_log_pred = model.predict_log_proba(X_train)[:, 1]
        j_scores = tpr - fpr
        j_ordered = sorted(zip(j_scores, thresholds))
        optimal_threshold = j_ordered[-1][1]
        print(f'Optimal threshold: ', np.exp(optimal_threshold))
        y_test_pred_new = np.where(
            y_prob_log_pred[:, 1] > optimal_threshold, 1, 0)
        print(f'New y_test_pred:\n{y_test_pred_new}')

        # print('-'*20)
        # print(f'Training Accuracy: {np.mean(train_acc)}')
        # print('\n')
        # print(f'Validation Accuracy: {np.mean(test_acc)}')
        # print('\n')
        if hasattr(classifier, 'best_params_') and hasattr(classifier, 'best_score_'):
            print('Grid Search Best Params:')
            print('\n')
            print(model.best_params_)
            print('-' * 20)

        print(
            f'SCORES FOR {classifier_name} WITH {vectorizer_name} AFTER OPTIMIZATION:'
        )
        cm, precision, recall, accuracy, f1, mcc, best_threshold, best_score, report = evaluation(
            y_test, y_test_pred, scoring, print_enabled)

    return y_test_pred_new, df_ROC_plot

# # %%
# def load_glove_with_vocabulary(vocabulary_map, feature_names, print_enabled, glove_file = validate_path(f'{glove_path}glove.840B.300d.txt')):

#     unique_words = set(feature_names)
#     if print_enabled:
#         print(f'Number of unique words: {len(unique_words)}')

#     glove = open(glove_file, 'r', encoding='utf8')

#     emb_list = [None] * len(vocabulary_map)

#     if print_enabled:
#         print(f'Length of vocabulary: {len(vocabulary_map)}')

#     found_words = 0
#     for i, line in enumerate(glove):
#         splitLine = line.split(' ')
#         word = splitLine[0]
#         if word in vocabulary_map:
#             found_words += 1
#             embedding = np.array([float(val) for val in splitLine[1:]], dtype=np.float32)
#             emb_list[vocabulary_map[word]] = embedding

#     if print_enabled:
#         print(f'Loaded GloVe vectors for {found_words} words. Generating random vectors for the rest.')

#     full_emb_list = []
#     glove_mean = -0.00584
#     glove_std = 0.452
#     full_emb_list = [np.random.normal(glove_mean, glove_std, (1, 300)) if emb is None else emb for emb in emb_list ]

#     if print_enabled:
#         print(f'Done. Vectors loaded : {len(full_emb_list)}')

#     return np.vstack(full_emb_list)


# # %%
# # this function creates a normalized vector for the whole sentence
# def sent2vec(sentences, embeddings_index=None, external_glove=True, extra_preprocessing_enabled=False):

#     if external_glove is False and embeddings_index is None:
#         embeddings_index= get_glove()

#     if extra_preprocessing_enabled is False:
#         words = sentences

#     elif extra_preprocessing_enabled is True:
#         stop_words = set(sw.words('english'))
#         words = str(sentences).lower()
#         words = word_tokenize(words)
#         words = [w for w in words if (w not in stop_words) and (w.isalpha())]

#     M = []

#     try:
#         for w in words:
#             try:
#                 M.append(embeddings_index[w])
#             except Exception:
#                 continue

#         M = np.array(M)
#         v = M.sum(axis='index')
#         if type(v) != np.ndarray:
#             return np.zeros(300)

#         return v / np.sqrt((v ** 2).sum())

#     except Exception:
#         return np.zeros(300)


# # %%
# def get_feature_name_and_refit_X_train_on_chi_test(train, X_train, vectorizer, refit_vectorizer, p_value_limit = 0.95):

#     feature_names = vectorizer.get_feature_names_out()
#     y = train[str(col)].astype('int64')
#     X_names = feature_names

#     dtf_features = pd.DataFrame()

#     for cat in np.unique(y):
#         chi2, p = feature_selection.chi2(X_train, y==cat)
#         dtf_features = dtf_features.append(pd.DataFrame({'feature':X_names, 'score':1-p, 'y':cat}))
#         dtf_features = dtf_features.sort_values(['y','score'], ascending=[True,False])
#         dtf_features = dtf_features[dtf_features['score']>p_value_limit]

#     X_names = dtf_features['feature'].unique().tolist()

#     vectorizer = refit_vectorizer(vocabulary=X_names)
#     X_train = vectorizer.fit_transform(train[f'{str(text_col)}'].astype('str'))

#     return X_train, vectorizer, dtf_features, X_names, feature_names


# # %%
# def train_offset(X, data_type):

#     word_indices_list = []
#     offsets_list = []

#     offset = 0
#     for i in range(0, X.shape[0]):
#         offsets_list.append(offset)
#         row, col = X.getrow(i).nonzero()
#         word_indices_list.append(torch.tensor(torch.from_numpy(col), dtype=torch.int64))
#         offset += len(row)

#     words = torch.cat(word_indices_list)
#     offsets = torch.tensor(offsets_list, dtype=torch.int64)

#     print(f'{str(data_type).title()} words shape: {words.shape}')
#     print(f'{str(data_type).title()} offsets shape: {offsets.shape}')
#     print(f'Created words and offsets for {str(data_type)} data')

#     return words, offsets

# %%
# class BagOfEmbeddings(nn.Module):
#     def __init__(self, embedding_weights, hidden_dim=100, dropout=0.5, embedding_mode='mean'):
#         super(BagOfEmbeddings, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.dropout = nn.Dropout(p=dropout)
#         embedding_size = embedding_weights.shape[1]
#         self.embedding = nn.EmbeddingBag(embedding_weights.shape[0], embedding_size, mode=embedding_mode)
#         self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float())
#         self.embedding.weight.requires_grad = False
#         self.final_layer = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(embedding_size, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, words, offsets):
#         x = self.embedding(words, offsets)

#         return self.final_layer(x)

# # %%
# def get_batch(words, offsets, targets, start_index, size):
#     first_word_index = offsets[start_index]
#     offsets_end_index = start_index + size
#     if offsets_end_index > offsets.shape[0]:
#         offsets_end_index = offsets.shape[0]
#         last_word_index = words.shape[0]
#     else:
#         last_word_index = offsets[offsets_end_index]
#     if targets is not None:
#         return words[first_word_index:last_word_index], offsets[start_index:offsets_end_index] - offsets[start_index], targets[start_index:offsets_end_index]
#     else:
#         return words[first_word_index:last_word_index], offsets[start_index:offsets_end_index] - offsets[start_index], None

# # %%
# def run_training(epochs, emb_model, optimizer, loss_fn,
#                 all_words, all_offsets,
#                 all_targets, batch_size=32):

#     print(f'Training samples: {all_offsets.shape[0]}')
#     batch_losses = []
#     for e in range(epochs):
#         emb_model.train()
#         start_index = 0
#         batch_nr = 0
#         print(f'Starting epoch {(e + 1)}')
#         while start_index < all_offsets.shape[0]:
#             batch_nr += 1
#             words, offsets, target = get_batch(all_words, all_offsets, all_targets, start_index, batch_size)
#             start_index += batch_size
#             optimizer.zero_grad()
#             output = emb_model(words, offsets)
#             loss = loss_fn(output.squeeze(), target)
#             loss.backward()
#             optimizer.step()
#             if batch_nr % 1000 == 0:
#                 batch_losses.append(loss.item())
#                 print(f'Epoch: {e + 1}, batch: {batch_nr}, loss: {loss.item():.5f}')

#     return batch_losses

# # %%
# def run_test(emb_model, loss_fn,
#                 all_words, all_offsets,
#                 all_targets, batch_size=256):

#     print('Test samples: ', all_offsets.shape[0])
#     batch_losses = []
#     outputs = []
#     emb_model.eval()
#     start_index = 0
#     batch_nr = 0
#     print('Starting testing')
#     while start_index < all_offsets.shape[0]:
#         batch_nr += 1
#         words, offsets, target = get_batch(all_words, all_offsets, all_targets, start_index, batch_size)
#         start_index += batch_size
#         output = emb_model(words, offsets)
#         outputs.append(torch.sigmoid(output))
#         if loss_fn:
#             loss = loss_fn(output.squeeze(), target)
#             if batch_nr % 100 == 0:
#                 batch_losses.append(loss.item())
#                 print(f'Batch: {batch_nr}, loss: {loss.item():.5f}')

#     return batch_losses, torch.cat(outputs)

# %%

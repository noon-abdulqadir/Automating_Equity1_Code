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

from setup_module.classification import *

# %%
from setup_module.imports import *
from setup_module.params import *
from setup_module.scraping import *

warnings.filterwarnings('ignore', category=DeprecationWarning)





# %%
# Tokenizers
# Load Word2Vec Model
w2v_model = Word2Vec.load(
    validate_path(
        f'{args["w2v_save_path"]}word2vec_model.model'
    )
)

df_jobs_labeled = pd.read_pickle(
    f'{args["df_dir"]}df_jobs_labeled_embeddings_preprocessed_stemming({stemming_enabled})_lemmatization({lemmatization_enabled})_numbers_cleaned({numbers_cleaned}).{file_save_format}'
)

n_gram = '3grams_gensim'
plots_enabled=True
print_enabled=True
for col in tqdm.tqdm(analysis_columns):
    print('-' * 20)
    print('\n')
    print(f'============================ STARTING PROCESSING {col.upper()} ============================')
    print('\n')
    print('-' * 20)
    if (
        len(
            df_jobs_labeled[
                df_jobs_labeled[str(col)].map(
                    df_jobs_labeled[str(col)].value_counts() > 50
                )
            ]
        )
        != 0
    ):
        print('Splitting data into training and test sets.')
        df_jobs_labeled = df_jobs_labeled.dropna(subset=['Warmth', 'Competence', text_col], how='any')

        train, test = train_test_split(
            df_jobs_labeled, test_size=test_split, train_size = 1-test_split, random_state=random_state
        )

        y_train = column_or_1d(train[str(col)].astype('int64').values, warn=True)

        y_test = column_or_1d(test[str(col)].astype('int64').values, warn=True)

        # Load Word2Vec Model
        w2v_model_gensim = Word2Vec.load(
            validate_path(
                f'{args["embeddings_save_path"]}123grams_gensim_word2vec_model.model'
            )
        )
        w2v_model_nltk = Word2Vec.load(
            validate_path(
                f'{args["embeddings_save_path"]}123grams_nltk_word2vec_model.model'
            )
        )

        ft_model_gensim = FastText.load(
            validate_path(
                f'{args["embeddings_save_path"]}123grams_gensim_fasttext_model.model'
            )
        )
        ft_model_nltk = FastText.load(
            validate_path(
                f'{args["embeddings_save_path"]}123grams_nltk_fasttext_model.model'
            )
        )

        word2vec_model300 = gensim_api.load('word2vec-google-news-300')
        glove_model300 = gensim_api.load('glove-wiki-gigaword-300')
        fasttext_model300 = gensim_api.load('fasttext-wiki-news-subwords-300')


        word_embedding_models = {'Word2Vec': word2vec_model300, 'GLoVe': glove_model300, 'fastText': fasttext_model300}

        ## Tensorflow/Keras
        tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ',
                                                oov_token='NaN',
                                                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

        ## tokenize train text
        print(f'Getting keras_embeddings for train dataset.')
        train_tokens = [ast.literal_eval(row) for idx, row in train[f'{n_gram}'].items() if isinstance(row, str) and str(row) != 'nan' and type(row) != float and len(row) != 0]
        tokenizer.fit_on_texts(iter(iter(train_tokens)))
        vocabulary_map = tokenizer.word_index
        ## create sequence
        lst_text2seq_train= tokenizer.texts_to_sequences(iter(iter(train_tokens)))
        ## padding sequence
        X_train = kprocessing.sequence.pad_sequences(lst_text2seq_train,
                            maxlen=15, padding='post', truncating='post')

        ## tokenize test text
        print(f'Getting keras_embeddings for test dataset.')
        test_tokens = [ast.literal_eval(row) for idx, row in test[f'{n_gram}'].items() if isinstance(row, str) and str(row) != 'nan' and type(row) != float and len(row) != 0]
        ## create sequence
        lst_text2seq_test= tokenizer.texts_to_sequences(iter(iter(test_tokens)))
        ## padding sequence
        X_test = kprocessing.sequence.pad_sequences(lst_text2seq_test,
                            maxlen=15, padding='post', truncating='post')

        if plots_enabled:
            print('Plotting tokenizer heatmap.')
            sns.heatmap(X_test==0, vmin=0, vmax=1, cbar=False)
            plt.show()

        if print_enabled:
            i = 0
            len_txt = len(train[text_col].iloc[i].split())

            ## list of text: ["I like this", ...]
            print('from: ', train[text_col].iloc[i], '| len:', len_txt)

            ## sequence of token ids: [[1, 2, 3], ...]
            len_tokens = len(X_train[i])
            print('to: ', X_train[i], '| len:', len(X_train[i]))

            ## vocabulary: {"I":1, "like":2, "this":3, ...}
            print('check: ', train[text_col].iloc[i].split()[0],
                    ' -- idx in vocabulary -->',
                    vocabulary_map[train_tokens[i][i].split()[0]])

            ## words not in vocabulary?
            if len_txt-len_tokens != 0:
                missing_words = [word for word in train[text_col].iloc[i].split() if word not in vocabulary_map.keys()]
                print('!!!', len_txt-len_tokens, 'words not in vocabulary:', missing_words)

            print('vocabulary: ', dict(list(vocabulary_map.items())[:5]), '... (padding element, 0)')

        ## start the matrix (length of vocabulary x vector size) with all 0s
        embeddings = np.zeros((len(vocabulary_map)+1, 300))

        for word, idx in vocabulary_map.items():
            ## update the row with vector
            with contextlib.suppress(Exception):
                embeddings[idx] =  w2v_model[word]

        if print_enabled:
            word = 'work'
            print('dic[word]:', vocabulary_map[word], '|idx')
            print('embeddings[idx]:', embeddings[vocabulary_map[word]].shape, '|vector')

        ## code attention layer
        def attention_layer(inputs, neurons):
            x = layers.Permute((2,1))(inputs)
            x = layers.Dense(neurons, activation='softmax')(x)
            x = layers.Permute((2,1), name='attention')(x)
            x = layers.multiply([inputs, x])
            return x


        ## input
        x_in = layers.Input(shape=(15,))
        ## embedding
        x = layers.Embedding(input_dim=embeddings.shape[0],
                                output_dim=embeddings.shape[1],
                                weights=[embeddings],
                                input_length=15, trainable=False)(x_in)
        ## apply attention
        x = attention_layer(x, neurons=15)
        ## 2 layers of bidirectional lstm
        x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2,
                                    return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)
        ## final dense layers
        x = layers.Dense(64, activation='relu')(x)
        y_out = layers.Dense(3, activation='softmax')(x)
        ## compile
        model = models.Model(x_in, y_out)
        model.compile(loss='sparse_categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

        if print_enabled:
            model.summary()

        ## encode y
        dic_y_mapping = dict(enumerate(np.unique(y_train)))
        inverse_dic = {v:k for k,v in dic_y_mapping.items()}
        y_train = np.array([inverse_dic[y] for y in y_train])
        ## train
        training = model.fit(x=X_train, y=y_train, batch_size=256,
                                epochs=10, shuffle=True, verbose=0,
                                validation_split=0.3)

        ## plot loss and accuracy
        metrics = [k for k in training.history.keys() if ('loss' not in k) and ('val' not in k)]
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        ax[0].set(title='Training')
        ax11 = ax[0].twinx()
        ax[0].plot(training.history['loss'], color='black')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss', color='black')
        for metric in metrics:
            ax11.plot(training.history[metric], label=metric)
        ax11.set_ylabel('Score', color='steelblue')
        ax11.legend()
        ax[1].set(title='Validation')
        ax22 = ax[1].twinx()
        ax[1].plot(training.history['val_loss'], color='black')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss', color='black')
        for metric in metrics:
            ax22.plot(training.history[f'val_{metric}'], label=metric)
        ax22.set_ylabel('Score', color='steelblue')
        plt.show()

        ## test
        y_test_pred = model.predict(X_test)
        predicted = [dic_y_mapping[np.argmax(pred)] for pred in y_test_pred]
        cm, precision, recall, accuracy, f1, mcc, best_threshold, best_score, report = evaluate_multi_classif(y_test, predicted, y_test_pred, figsize=(15,5))

        ## select observation
        i = 0
        txt_instance = test[f'{text_col}'].iloc[i]
        ## check true value and predicted value
        print('True:', y_test[i], '--> Pred:', predicted[i], '| Prob:', round(np.max(y_test_pred[i]),2))

        ## show explanation
        ### 1. preprocess input
        corpus = test[text_col]
        tokens = [ast.literal_eval(row) for idx, row in test[f'{n_gram}'].items() if isinstance(row, str) and str(row) != 'nan' and type(row) != float and len(row) != 0]
        X_instance = kprocessing.sequence.pad_sequences(
                        tokenizer.texts_to_sequences(corpus), maxlen=15,
                        padding='post', truncating='post')

        ### 2. get attention weights
        layer = [layer for layer in model.layers if 'attention' in layer.name][0]
        func = K.function([model.input], [layer.output])
        weights = func(X_instance)[0]
        weights = np.mean(weights, axis=2).flatten()

        ### 3. rescale weights, remove null vector, map word-weight
        weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(weights).reshape(-1,1)).reshape(-1)
        weights = [weights[n] for n,idx in enumerate(X_instance[0]) if idx != 0]
        dic_word_weigth = {word:weights[n] for n,word in enumerate(tokens[0]) if word in
                            tokenizer.word_index.keys()}

        ### 4. barplot
        if dic_word_weigth:
            dtf = pd.DataFrame.from_dict(dic_word_weigth, orient='index',
                                    columns=['score'])
            dtf.sort_values(by='score',
                ascending=True).tail(3).plot(kind='barh',
                legend=False).grid(axis='x')
            plt.show()
        else:
            print('--- No word recognized ---')

        ### 5. produce html visualization
        text = []
        for word in tokens[0]:
            weight = dic_word_weigth.get(word)
            if weight is not None:
                text.append('<b><span style="background-color:rgba(100,149,237,' + str(weight) + ');">' + word + '</span></b>')
            else:
                text.append(word)
        text = ' '.join(text)

        ### 6. visualize on notebook
        print('\033[1m'+'Text with highlighted words')
        from IPython.core.display import HTML, display
        display(HTML(text))




        ## bert tokenizer
        ## distil-bert tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True, is_split_into_words=True)
        nlp = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        lst_vocabulary = list(tokenizer.vocab.keys())

        train = train.dropna()
        corpus = train[text_col].astype(str).values.to_list()
        maxlen = 50

        ## add special tokens
        maxqnans = np.int((maxlen-20)/2)
        corpus_tokenized = ['[CLS] '+
                        ' '.join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '',
                        str(txt).lower().strip()))[:maxqnans])+
                        ' [SEP] ' for txt in corpus]

        ## generate masks
        masks = [[1]*len(txt.split(' ')) + [0]*(maxlen - len(
                    txt.split(' '))) for txt in corpus_tokenized]

        ## padding
        txt2seq = [txt + ' [PAD]'*(maxlen-len(txt.split(' '))) if len(txt.split(' ')) != maxlen else txt for txt in corpus_tokenized]

        ## generate idx
        idx = [tokenizer.encode(seq.split(' ')) for seq in txt2seq]

        ## generate segments
        segments = []
        for seq in txt2seq:
            temp, i = [], 0
            for token in seq.split(' '):
                temp.append(i)
                if token == '[SEP]':
                    i += 1
            segments.append(temp)
        ## feature matrix
        X_train = [np.asarray(idx, dtype='int32'),
                    np.asarray(masks, dtype='int32'),
                    np.asarray(segments, dtype='int32')]

        i = 0
        print('txt: ', train[text_col].iloc[0])
        print('tokenized:', [tokenizer.convert_ids_to_tokens(idx) for idx in X_train[0][i].to_list()])
        print('idx: ', X_train[0][i])
        print('mask: ', X_train[1][i])
        print('segment: ', X_train[2][i])

        ## inputs
        idx = layers.Input((50), dtype='int32', name='input_idx')
        masks = layers.Input((50), dtype='int32', name='input_masks')
        segments = layers.Input((50), dtype='int32', name='input_segments')
        ## pre-trained bert
        nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased')
        bert_out, _ = nlp([idx, masks, segments])
        ## fine-tuning
        x = layers.GlobalAveragePooling1D()(bert_out)
        x = layers.Dense(64, activation='relu')(x)
        y_out = layers.Dense(len(np.unique(y_train)),
                                activation='softmax')(x)
        ## compile
        model = models.Model([idx, masks, segments], y_out)
        for layer in model.layers[:4]:
            layer.trainable = False
        model.compile(loss='sparse_categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])
        model.summary()

        ## inputs
        idx = layers.Input((50), dtype='int32', name='input_idx')
        masks = layers.Input((50), dtype='int32', name='input_masks')
        ## pre-trained bert with config
        config = transformers.DistilBertConfig(dropout=0.2,
                    attention_dropout=0.2)
        config.output_hidden_states = False
        nlp = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
        bert_out = nlp(idx, attention_mask=masks)[0]
        ## fine-tuning
        x = layers.GlobalAveragePooling1D()(bert_out)
        x = layers.Dense(64, activation='relu')(x)
        y_out = layers.Dense(len(np.unique(y_train)),
                                activation='softmax')(x)
        ## compile
        model = models.Model([idx, masks], y_out)
        for layer in model.layers[:3]:
            layer.trainable = False
        model.compile(loss='sparse_categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])
        model.summary()

        ## encode y
        dic_y_mapping = {n:label for n,label in
                            enumerate(np.unique(y_train))}
        inverse_dic = {v:k for k,v in dic_y_mapping.items()}
        y_train = np.array([inverse_dic[y] for y in y_train])
        ## train
        training = model.fit(x=X_train, y=y_train, batch_size=64,
                                epochs=1, shuffle=True, verbose=1,
                                validation_split=0.3)
        ## test
        y_test_pred = model.predict(X_test)
        predicted = [dic_y_mapping[np.argmax(pred)] for pred in y_test_pred]
        cm, precision, recall, accuracy, f1, mcc, best_threshold, best_score, report = evaluate_multi_classif(y_test, predicted, y_test_pred, figsize=(15,5))

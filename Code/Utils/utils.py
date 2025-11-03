import os, sys

# sys.path.append('/mnt/scratch_b/users/t/tsingalis/Documents/nnParafac2')
sys.path.append('/media/data/tsingalis/nnParafac2/')
sys.path.append('/media/data/tsingalis/nnParafac2/Code/')

ROOT = '/media/blue/tsingalis/nnParafac2/'

import re
import jieba  # For chinece
import pandas as pd
from fugashi import Tagger  # For Japanese

from sklearn import metrics

# import skbayes
# from skbayes.linear_models import EBLogisticRegression, VBLogisticRegression

# check_estimator(EBLogisticRegression)
# check_estimator(VBLogisticRegression)

from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

import scipy

from scipy.sparse import issparse

import json
import numpy as np

from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk.stem.snowball import GermanStemmer
from nltk.stem.snowball import ItalianStemmer
from nltk.stem.snowball import RussianStemmer
from nltk.stem.snowball import SpanishStemmer

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
# from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# from Code.SpacyVectorizer import SpacyLemmaCountVectorizer
# from Code.SpacyVectorizer import SpacyPipeProcessor

from functools import lru_cache

from copy import deepcopy

from Code.Utils.preprocessing import Tokenizer

ja_tagger = Tagger('-Owakati -b 5000')

fr_stemmer = FrenchStemmer()
en_stemmer = EnglishStemmer()
de_stemmer = GermanStemmer()
it_stemmer = ItalianStemmer()
ru_stemmer = RussianStemmer()
es_stemmer = SpanishStemmer()

en_stop_words = stopwords.words('english')
fr_stop_words = stopwords.words('french')
it_stop_words = stopwords.words('italian')
de_stop_words = stopwords.words('german')
es_stop_words = stopwords.words('spanish')
ru_stop_words = stopwords.words('russian')

from transformers import BertTokenizer

en_bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
de_bert_tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")

# import Stemmer
#
# fr_stemmer = Stemmer.Stemmer('french')
# en_stemmer = Stemmer.Stemmer('english')

import spacy

# https://stackoverflow.com/questions/66530272/tokenize-text-very-slow-when-doing-it
fr_tok_spacy = spacy.load('fr_core_news_sm', disable=["tagger", "ner", "parser"])
en_tok_spacy = spacy.load('en_core_web_sm', disable=["tagger", "ner", "parser"])
it_tok_spacy = spacy.load('it_core_news_sm', disable=["tagger", "ner", "parser"])
de_tok_spacy = spacy.load('de_core_news_sm', disable=["tagger", "ner", "parser"])
es_tok_spacy = spacy.load('es_core_news_sm', disable=["tagger", "ner", "parser"])
ru_tok_spacy = spacy.load('ru_core_news_sm', disable=["tagger", "ner", "parser"])

"""Moses supported languages
    ca: Catalan
    cs: Czech
    de: German
    el: Greek
    en: English
    es: Spanish
    fi: Finnish
    fr: French
    hu: Hungarian
    is: Icelandic
    it: Italian
    lv: Latvian
    nl: Dutch
    pl: Polish
    pt: Portugese
    ro: Romanian
    ru: Russian
    sk: Slovak
    sl: Slovene
    sv: Swedish
    ta: Tamil
"""

de_moses_tokenizer = Tokenizer('de')
en_moses_tokenizer = Tokenizer('en')
# ja_moses_tokenizer = Tokenizer('ja')
fr_moses_tokenizer = Tokenizer('fr')
it_moses_tokenizer = Tokenizer('it')
ru_moses_tokenizer = Tokenizer('ru')
es_moses_tokenizer = Tokenizer('es')

with open(os.path.join(ROOT, 'Code/Utils/ja_stop_words.json'), encoding='utf-8') as data_file:
    ja_stop_words = json.load(data_file)

with open(os.path.join(ROOT, 'Code/Utils/zh_stop_words.json'), encoding='utf-8') as data_file:
    zh_stop_words = json.load(data_file)


def read_aligned_raw_data(data_folder, language, n_samples=None, masked=True):
    if masked:
        file_train = os.path.join(data_folder, language + '_masked.txt')
    else:
        file_train = os.path.join(data_folder, language + '.txt')

    # f = open(file_train, 'r', encoding="UTF-8")
    # data = f.readlines()

    with open(file_train, 'r', encoding="UTF-8") as f:
        data = f.readlines()

    data = [x.rstrip('\n') for x in data]
    df_data = pd.DataFrame(data, columns=['Text'])
    # if n_samples == -1 or n_samples > len(df_data):
    #     return df_data
    # else:
    #     return df_data.sample(n=n_samples, random_state=1)

    return df_data


@lru_cache(maxsize=10000)
def en_stem(word):
    return en_stemmer.stem(word.lower().strip())
    # return en_stemmer.stemWord(word)


def en_tokenize(text, tok_lib):
    if tok_lib == 'spacy':
        stems = []
        doc = en_tok_spacy(text)
        for token in doc:
            if token.is_alpha and not token.is_stop and not token.is_punct:
                stems.append(token.lemma_)
        return stems
    elif tok_lib == 'nltk':
        text = re.sub('[^a-zA-Z0-9]+', ' ', text)
        tokens = [word for word in nltk.word_tokenize(text, language='english') if word.isalpha()]
        stems = [en_stem(t) for t in tokens if t not in en_stop_words]
        return stems
    elif tok_lib == 'moses':
        tokens = [word for word in en_moses_tokenizer.tokenize(text).split(' ') if word.isalpha()]
        stems = [en_stem(t) for t in tokens if t not in en_stop_words]
        # return tokens
    elif tok_lib == 'laser':
        tokens = [word for word in
                  TokenLine(line=text, lang='en', lower_case=True, romanize=False).split(' ')
                  if word.isalpha()]
        stems = [en_stem(t) for t in tokens if t not in en_stop_words]
        return stems
    elif tok_lib == 'BertTokenizer':
        # tokens = [word for word in en_bert_tokenizer.tokenize(text) if word.isalpha()]
        stems = [en_stem(t) for t in en_bert_tokenizer.tokenize(text) if t not in en_stop_words]
        # return stems
    else:
        raise ValueError('Tokenizer should be "spacy" or "nltk"')

    return stems


@lru_cache(maxsize=10000)
def fr_stem(word):
    return fr_stemmer.stem(word.lower().strip())
    # return fr_stemmer.stemWord(word)


def fr_tokenize(text, tok_lib):
    if tok_lib == 'spacy':
        stems = []
        doc = fr_tok_spacy(text)
        for token in doc:
            # print(token, token.lemma_)
            if token.is_alpha and not token.is_stop and not token.is_punct:
                stems.append(token.lemma_)
    elif tok_lib == 'nltk':
        text = re.sub('[^a-zA-Z0-9]+', ' ', text)
        tokens = [word for word in nltk.word_tokenize(text, language='french') if word.isalpha()]
        stems = [fr_stem(t) for t in tokens if t not in fr_stop_words]
    elif tok_lib == 'moses':
        tokens = [word for word in fr_moses_tokenizer.tokenize(text).split(' ') if word.isalpha()]
        stems = [fr_stem(t) for t in tokens if t not in fr_stop_words]
    elif tok_lib == 'BertTokenizer':
        tokens = [word for word in fr_bert_tokenizer.tokenize(text) if word.isalpha()]
        stems = [fr_stem(t) for t in tokens if t not in de_stop_words]
    else:
        raise ValueError('Tokenizer should be "spacy" or "nltk"')

    return stems

@lru_cache(maxsize=10000)
def de_stem(word):
    return de_stemmer.stem(word)
    # return fr_stemmer.stemWord(word)


def de_tokenize(text, tok_lib):
    if tok_lib == 'spacy':
        stems = []
        doc = de_tok_spacy(text)
        for token in doc:
            # print(token, token.lemma_)
            if token.is_alpha and not token.is_stop and not token.is_punct:
                stems.append(token.lemma_)
    elif tok_lib == 'nltk':
        tokens = [word for word in nltk.word_tokenize(text, language='german') if word.isalpha()]
        stems = [de_stem(t) for t in tokens if t not in de_stop_words]
    elif tok_lib == 'moses':
        tokens = [word for word in de_moses_tokenizer.tokenize(text).split(' ') if word.isalpha()]
        stems = [de_stem(t) for t in tokens if t not in de_stop_words]
        # return tokens
    elif tok_lib == 'laser':
        return TokenLine(line=text, lang='de', lower_case=True, romanize=False)
    elif tok_lib == 'BertTokenizer':
        # tokens = [word for word in de_bert_tokenizer.tokenize(text) if word.isalpha()]
        stems = [de_stem(t) for t in de_bert_tokenizer.tokenize(text) if t not in de_stop_words]
    else:
        raise ValueError('Tokenizer should be "spacy" or "nltk"')

    return stems


@lru_cache(maxsize=10000)
def it_stem(word):
    return it_stemmer.stem(word)
    # return fr_stemmer.stemWord(word)


def it_tokenize(text, tok_lib):
    if tok_lib == 'spacy':
        stems = []
        doc = it_tok_spacy(text)
        for token in doc:
            # print(token, token.lemma_)
            if token.is_alpha and not token.is_stop and not token.is_punct:
                stems.append(token.lemma_)
    elif tok_lib == 'moses':
        tokens = [word for word in it_moses_tokenizer.tokenize(text).split(' ') if word.isalpha()]
        stems = [it_stem(t) for t in tokens if t not in it_stop_words]
    elif tok_lib == 'nltk':
        tokens = [word for word in nltk.word_tokenize(text, language='italian') if word.isalpha()]
        stems = [it_stem(t) for t in tokens if t not in it_stop_words]
    else:
        raise ValueError('Tokenizer should be "spacy" or "nltk"')

    return stems


@lru_cache(maxsize=10000)
def es_stem(word):
    return es_stemmer.stem(word)
    # return fr_stemmer.stemWord(word)


def es_tokenize(text, tok_lib):
    if tok_lib == 'spacy':
        stems = []
        doc = es_tok_spacy(text)
        for token in doc:
            # print(token, token.lemma_)
            if token.is_alpha and not token.is_stop and not token.is_punct:
                stems.append(token.lemma_)
    elif tok_lib == 'moses':
        tokens = [word for word in es_moses_tokenizer.tokenize(text).split(' ') if word.isalpha()]
        stems = [es_stem(t) for t in tokens if t not in es_stop_words]
    elif tok_lib == 'nltk':
        tokens = [word for word in nltk.word_tokenize(text, language='spanish') if word.isalpha()]
        stems = [es_stem(t) for t in tokens if t not in es_stop_words]
    else:
        raise ValueError('Tokenizer should be "spacy" or "nltk"')

    return stems


@lru_cache(maxsize=10000)
def ru_stem(word):
    return ru_stemmer.stem(word)
    # return fr_stemmer.stemWord(word)


def ru_tokenize(text, tok_lib):
    if tok_lib == 'spacy':
        stems = []
        doc = ru_tok_spacy(text)
        for token in doc:
            # print(token, token.lemma_)
            if token.is_alpha and not token.is_stop and not token.is_punct:
                stems.append(token.lemma_)
    elif tok_lib == 'moses':
        tokens = [word for word in ru_moses_tokenizer.tokenize(text).split(' ') if word.isalpha()]
        stems = [ru_stem(t) for t in tokens if t not in ru_stop_words]
    elif tok_lib == 'nltk':
        tokens = [word for word in nltk.word_tokenize(text, language='russian') if word.isalpha()]
        stems = [ru_stem(t) for t in tokens if t not in ru_stop_words]
    else:
        raise ValueError('Tokenizer should be "spacy" or "nltk"')

    return stems


def zh_tokenize(text, tok_lib):
    tokens = ' '.join(jieba.lcut(text)).split()
    stop_words = zh_stop_words + stopwords.words('english')
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens


def ja_tokenize(text, tok_lib):
    text = re.sub(r"[0-9]+", '', re.sub(r"\W", ' ', text))
    tokens = [word.surface for word in ja_tagger(text)]
    stop_words = ja_stop_words + stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


LASER = '/media/blue/tsingalis/LASER/'
MOSES_BDIR = LASER + '/tools-external/moses-tokenizer/tokenizer/'
MOSES_TOKENIZER = MOSES_BDIR + 'tokenizer.perl -q -no-escape -threads 20 -l '
MOSES_LC = MOSES_BDIR + 'lowercase.perl'
NORM_PUNC = MOSES_BDIR + 'normalize-punctuation.perl -l '
DESCAPE = MOSES_BDIR + 'deescape-special-chars.perl'
REM_NON_PRINT_CHAR = MOSES_BDIR + 'remove-non-printing-char.perl'
# Romanization (and lower casing)
ROMAN_LC = 'python3 ' + LASER + '/source/lib/romanize_lc.py -l '

# Mecab tokenizer for Japanese
MECAB = LASER + '/tools-external/mecab'

from subprocess import check_output


def TokenLine(line, lang='en', lower_case=True, romanize=False):
    assert lower_case, 'lower case is needed by all the models'
    roman = lang if romanize else 'none'
    tok = check_output(
        REM_NON_PRINT_CHAR
        + '|' + NORM_PUNC + lang
        + '|' + DESCAPE
        + '|' + MOSES_TOKENIZER + lang
        + ('| python3 -m jieba -d ' if lang == 'zh' else '')
        + ('|' + MECAB + '/bin/mecab -O wakati -b 50000 ' if lang == 'ja' else '')
        + '|' + ROMAN_LC + roman,
        input=line,
        encoding='UTF-8',
        shell=True)
    return tok.strip()


def tokenize_data(language, df_train, df_dev=None, df_test=None, vocab=None, ngram_range=(1, 1),
                  cased=True, use_tfidf=True, norm='l2', tok_lib='nltk', tfidf_transformer=None, analyzer='word',
                  max_features=None):
    if language == 'english':
        tokenizer = en_tokenize
    elif language == 'french':
        tokenizer = fr_tokenize
    elif language == 'german':
        tokenizer = de_tokenize
    elif language == 'italian':
        tokenizer = it_tokenize
    elif language == 'spanish':
        tokenizer = es_tokenize
    elif language == 'russian':
        tokenizer = ru_tokenize
    elif language == 'chinese':
        tokenizer = zh_tokenize
    elif language == 'japanese':
        tokenizer = ja_tokenize
    else:
        raise ValueError('Select correct language')

    class_data_tokenized_train, class_data_tokenized_dev, class_data_tokenized_test = None, None, None

    vectorizer = CountVectorizer(tokenizer=lambda text: tokenizer(text, tok_lib),
                                 ngram_range=ngram_range,
                                 lowercase=not cased,
                                 vocabulary=vocab,
                                 token_pattern=r"(?u)\b\w\w+\b",
                                 max_features=max_features,
                                 binary=False,
                                 analyzer=analyzer)

    class_data_tokenized_train = vectorizer.fit_transform(tqdm(df_train['Text'].values.tolist(),
                                                               desc=f'Fit CountVectorizer in '
                                                                    f'training data language {language}'))

    print('Task language: {} -- dataset size: {} -- vocab size: {}'.format(language,
                                                                           len(df_train['Text'].values.tolist()),
                                                                           len(vectorizer.vocabulary_)))

    if df_dev is not None:
        class_data_tokenized_dev = vectorizer.transform(tqdm(df_dev['Text'].values.tolist(),
                                                             desc=f'Transform CountVectorizer in '
                                                                  f'dev data language {language}'))
    if df_test is not None:
        class_data_tokenized_test = vectorizer.transform(tqdm(df_test['Text'].values.tolist(),
                                                              desc=f'Transform CountVectorizer in '
                                                                   f'test data language {language}'))

    if use_tfidf:
        if tfidf_transformer is None:
            tfidf_transformer = TfidfTransformer(use_idf=True, norm=norm)
        class_data_tokenized_train = tfidf_transformer.fit_transform(class_data_tokenized_train)
        if df_dev is not None:
            class_data_tokenized_dev = tfidf_transformer.transform(class_data_tokenized_dev)
        if df_test is not None:
            class_data_tokenized_test = tfidf_transformer.transform(class_data_tokenized_test)
    else:
        class_data_tokenized_train = preprocessing.normalize(class_data_tokenized_train, norm=norm, axis=1)
        if df_dev is not None:
            class_data_tokenized_dev = preprocessing.normalize(class_data_tokenized_dev, norm=norm, axis=1)
        if df_test is not None:
            class_data_tokenized_test = preprocessing.normalize(class_data_tokenized_test, norm=norm, axis=1)

    return class_data_tokenized_train, class_data_tokenized_dev, \
        class_data_tokenized_test, vectorizer, tfidf_transformer, vectorizer.vocabulary_


def tokenize_datav2(language, df_train, df_dev=None, df_test=None,
                    vocab=None, ngram_range=(1, 1), tok_lib='nltk',
                    cased=True, use_tfidf=True, norm='l2',
                    tfidf_transformer=None):
    class_data_tokenized_train, class_data_tokenized_dev, class_data_tokenized_test = None, None, None
    train_docs, dev_docs, test_docs = None, None, None
    if tok_lib == 'nltk':
        print(f'Train samples: {len(df_train)}')
        train_docs = df_train['Text'].values.tolist()
        if df_dev is not None:
            print(f'Dev samples: {len(df_dev)}')
            dev_docs = df_dev['Text'].values.tolist()
        if df_test is not None:
            print(f'Test samples: {len(df_test)}')
            test_docs = df_test['Text'].values.tolist()

        if language == 'english':
            tokenizer = en_tokenize
        elif language == 'french':
            tokenizer = fr_tokenize
        elif language == 'german':
            tokenizer = de_tokenize
        elif language == 'italian':
            tokenizer = it_tokenize
        elif language == 'spanish':
            tokenizer = es_tokenize
        elif language == 'russian':
            tokenizer = ru_tokenize
        elif language == 'chinese':
            tokenizer = zh_tokenize
        elif language == 'japanese':
            tokenizer = ja_tokenize
        else:
            raise ValueError('Select correct language')

        vectorizer = CountVectorizer(analyzer='char',
                                     tokenizer=lambda text: tokenizer(text, tok_lib),
                                     ngram_range=ngram_range,
                                     lowercase=not cased,
                                     vocabulary=vocab,
                                     binary=False)

        class_data_tokenized_train = vectorizer.fit_transform(train_docs)

    elif tok_lib == 'spacy':
        spp = SpacyPipeProcessor(en_tok_spacy, n_process=-1)

        print(f'Train samples: {len(df_train["Text"])}')
        train_docs = spp(df_train["Text"].values.tolist())
        if df_dev is not None:
            print(f'Dev samples: {len(df_dev)}')
            dev_docs = spp(df_dev["Text"].values.tolist())
        if df_test is not None:
            print(f'Test samples: {len(df_test)}')
            test_docs = spp(df_test["Text"].values.tolist())

        vectorizer = SpacyLemmaCountVectorizer(ignore_chars='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                                               ngram_range=ngram_range,
                                               lowercase=not cased,
                                               vocabulary=vocab,
                                               binary=False)

        class_data_tokenized_train = vectorizer.fit_transform(train_docs)
    else:
        raise ValueError('Tokenizer should be "spacy" or "nltk"')

    # class_data_tokenized_train = vectorizer.fit_transform(df_train)

    if vocab is None:
        vocab = vectorizer.vocabulary_

    print('Task dataset {} vocab size {}'.format(language, len(vectorizer.vocabulary_)))

    if df_dev is not None:
        class_data_tokenized_dev = vectorizer.transform(dev_docs)
    if df_test is not None:
        class_data_tokenized_test = vectorizer.transform(test_docs)
    if use_tfidf:
        if tfidf_transformer is None:
            tfidf_transformer = TfidfTransformer(use_idf=True, norm=norm)
        class_data_tokenized_train = tfidf_transformer.fit_transform(class_data_tokenized_train)
        if df_dev is not None:
            class_data_tokenized_dev = tfidf_transformer.transform(class_data_tokenized_dev)
        if df_test is not None:
            class_data_tokenized_test = tfidf_transformer.transform(class_data_tokenized_test)
    else:
        class_data_tokenized_train = preprocessing.normalize(class_data_tokenized_train, norm=norm, axis=1)
        if df_dev is not None:
            class_data_tokenized_dev = preprocessing.normalize(class_data_tokenized_dev, norm=norm, axis=1)
        if df_test is not None:
            class_data_tokenized_test = preprocessing.normalize(class_data_tokenized_test, norm=norm, axis=1)

    return class_data_tokenized_train, class_data_tokenized_dev, \
        class_data_tokenized_test, vectorizer, tfidf_transformer, vocab


def encode_multi_labels(labels_train, labels_dev=None, labels_test=None):
    # Create label matrices for train and test
    from sklearn.preprocessing import MultiLabelBinarizer
    # combine labels because there is not
    """
    labels_train_encoded = ['_'.join(map(str, sorted(l_tr))) for l_tr in labels_train]
    labels_dev_encoded = ['_'.join(map(str, sorted(l_dev))) for l_dev in labels_dev]
    labels_test_encoded = ['_'.join(map(str, sorted(l_tst))) for l_tst in labels_test]

    import itertools
    all_labels_train_encoded = set(labels_train_encoded)
    common_labels = set(labels_dev_encoded) & set(labels_test_encoded)
    # Remove train samples whose labels are not in the dev and test set.
    idx_tr = [bool(set(l_tr) & common_labels) for l_tr in set(labels_train_encoded)]
    idx_dev = [l_dev in all_labels_train_encoded for l_dev in labels_dev_encoded]
    idx_test = [l_tst in all_labels_train_encoded for l_tst in labels_test_encoded]

    # labels_train = labels_train[idx_tr]
    labels_dev = labels_dev[idx_dev]
    labels_test = labels_test[idx_test]
    """
    le = MultiLabelBinarizer().fit(labels_train)
    labels_train = le.transform(labels_train)

    if labels_dev is not None:
        # UserWarning: unknown class(es) [203, 21, 252, 274, 442, 446, 450, 451] will be ignored.
        # To avoid this warning we remove the unseen classes.
        labels_dev = [np.intersect1d(l_dev, le.classes_) for l_dev in labels_dev]
        labels_dev = le.transform(labels_dev)
    if labels_test is not None:
        # UserWarning: unknown class(es) [203, 252, 430, 436, 442, 446, 448, 449, 450, 451] will be ignored.
        # To avoid this warning we remove the unseen classes.
        labels_test = [np.intersect1d(l_tst, le.classes_) for l_tst in labels_test]
        labels_test = le.transform(labels_test)

    labels = labels_train, labels_dev, labels_test
    return labels, le.classes_, le


def encode_labels(labels_train, labels_dev=None, labels_test=None):
    # Create label matrices for train and test
    le = preprocessing.LabelEncoder()
    labels_train = le.fit_transform(labels_train)

    if labels_dev is not None:
        labels_dev = le.transform(labels_dev)
    if labels_test is not None:
        labels_test = le.transform(labels_test)

    labels = labels_train, labels_dev, labels_test
    return labels, le.classes_, le


def compute_class(X_tr_src, X_tr_trg,
                  X_dev_src, X_tst_trg,
                  lb_tr_src, lb_tr_trg,
                  lb_dev_src, lb_tst_trg,
                  clf_name=None,
                  sub_clf_name=None,
                  solver='lbfgs',
                  penalty='l2',
                  random_state=88):
    print("Fitting the classifier to the training set")

    clip_negative = False
    if clip_negative:
        if np.any(X_tr_src < 0):
            print('Negative value found in X_tr_src')
            X_tr_src = X_tr_src.clip(min=0)

        if np.any(X_tr_trg < 0):
            print('Negative value found in X_tr_trg')
            X_tr_trg = X_tr_trg.clip(min=0)

        if np.any(X_dev_src < 0):
            print('Negative value found in X_dev_src')
            X_dev_src = X_dev_src.clip(min=0)

        if np.any(X_tst_trg < 0):
            print('Negative value found in X_tst_trg')
            X_tst_trg = X_tst_trg.clip(min=0)

    if clf_name == LogisticRegression.__name__ or clf_name == LinearSVC.__name__ or clf_name == SVC.__name__:
        param_grid = {"C": np.logspace(-4, 4, num=15)}
    elif clf_name == OneVsRestClassifier.__name__:
        if sub_clf_name == LogisticRegression.__name__:
            param_grid = {"estimator__C": np.logspace(-4, 4, num=14)}
        elif sub_clf_name == SGDClassifier.__name__:
            param_grid = {"estimator__alpha": np.logspace(-4, 4, num=14)}
        elif sub_clf_name == KNeighborsClassifier.__name__:
            param_grid = {'estimator__n_neighbors': (1, 10, 1)}
        else:
            raise ValueError('sub_clf_name in OneVsRestClassifier should be "LogisticRegression"')
    elif clf_name == MultinomialNB.__name__:
        param_grid = {"alpha": [50, 15, 10, 5, 1, 0.5, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.001]}
    elif clf_name == GaussianMixture.__name__:
        param_grid = {'n_components': np.array([1, 2, 3, 4])}
    elif clf_name == GaussianNB.__name__:
        param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
    elif clf_name == ComplementNB.__name__:
        param_grid = {"alpha": [50, 15, 10, 5, 1, 0.5, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.001]}
    elif clf_name == AdaBoostClassifier.__name__:
        param_grid = {'estimator__C': np.logspace(-4, 4, num=15),
                      'n_estimators': [1, 5, 10, 15, 20, 25, 40, 50],
                      'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5]}
    elif clf_name == RandomForestClassifier.__name__:
        param_grid = {'n_estimators': [64, 128, 256],
                      'max_depth': [2, 4, 8, 16, 36, 64],
                      "min_samples_split": [5, 10]}
    elif clf_name == MLPClassifier.__name__:
        param_grid = {
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }
    # elif clf_name == EBLogisticRegression.__name__:
    #     param_grid = {"alpha": np.logspace(-4, 4, num=15)}
    # elif clf_name == VBLogisticRegression.__name__:
    #     param_grid = {"a": np.logspace(-4, 4, num=15), "b": np.logspace(-2, 6, num=15)}
    elif clf_name == KNeighborsClassifier.__name__:
        param_grid = {
            'n_neighbors': (1, 10, 1),
            'leaf_size': (20, 40, 1),
            'p': (1, 2),
            'weights': ('uniform', 'distance'),
            'metric': ('minkowski', 'chebyshev')}

    # if scaler_name == MinMaxScaler.__name__:
    #     scaler = MinMaxScaler()
    #     X_tr_src = scaler.fit_transform(X_tr_src)
    #     X_dev_src = scaler.transform(X_dev_src)
    #     X_tst_trg = scaler.transform(X_tst_trg)
    #     X_tr_trg = scaler.fit_transform(X_tr_trg)

    if clf_name == LogisticRegression.__name__:
        # clf = LogisticRegression(solver='liblinear', max_iter=20000,
        #                          random_state=random_state, dual=False,
        #                          multi_class='ovr',
        #                          penalty=penalty,
        #                          # n_jobs=40
        #                          )
        # if X_tr_src.shape[0] < X_tr_src.shape[1]:
        #     dual = True
        #     solver = 'liblinear'
        # else:
        #     dual = False
        #     solver = 'liblinear'
        clf = LogisticRegression(solver=solver, max_iter=20000,
                                 random_state=random_state, dual=False,
                                 # multi_class='ovr',
                                 penalty=penalty,
                                 # n_jobs=40
                                 )
    elif clf_name == KNeighborsClassifier.__name__:
        clf = KNeighborsClassifier(algorithm='auto')
    elif clf_name == MultinomialNB.__name__:
        # clf = OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))
        clf = MultinomialNB(fit_prior=True, class_prior=None)
    elif clf_name == ComplementNB.__name__:
        # clf = OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))
        clf = ComplementNB(fit_prior=True, class_prior=None)
    elif clf_name == GaussianMixture.__name__:
        clf = GaussianMixture()
    elif clf_name == LinearSVC.__name__:
        # clf = LinearSVC(max_iter=20000)
        clf = SVC(probability=True, kernel='linear')
        # clf = OneVsRestClassifier(SVC(probability=True, kernel='linear'))
    elif clf_name == SVC.__name__:
        clf = SVC(max_iter=20000, gamma="auto")
    elif clf_name == GaussianNB.__name__:
        # clf = OneVsRestClassifier(GaussianNB())
        clf = GaussianNB()
    elif clf_name == AdaBoostClassifier.__name__:
        if sub_clf_name == 'LinearSVC':
            clf = AdaBoostClassifier(SVC(probability=True, kernel='linear'))
        elif sub_clf_name == 'SVC':
            clf = AdaBoostClassifier(SVC(probability=True, kernel='rbf', gamma="auto"))
        elif sub_clf_name == 'LogisticRegression':
            clf = AdaBoostClassifier(LogisticRegression(solver=solver, max_iter=20000,
                                                        random_state=random_state,
                                                        penalty=penalty, n_jobs=-1))
    elif clf_name == RandomForestClassifier.__name__:
        clf = RandomForestClassifier(random_state=0)
    elif clf_name == MLPClassifier.__name__:
        clf = MLPClassifier(random_state=0, max_iter=3000)
    # elif clf_name == EBLogisticRegression.__name__:
    #     clf = EBLogisticRegression(n_iter=150, n_iter_solver=115, verbose=True)
    # elif clf_name == VBLogisticRegression.__name__:
    #     clf = VBLogisticRegression()
    if clf_name == OneVsRestClassifier.__name__:
        if sub_clf_name == LogisticRegression.__name__:
            clf = OneVsRestClassifier(LogisticRegression(random_state=0, max_iter=50000, n_jobs=-1,
                                                         dual=False, warm_start=True))
        elif sub_clf_name == SGDClassifier.__name__:
            clf = OneVsRestClassifier(SGDClassifier(loss="hinge", penalty="l2", tol=1e-3, n_jobs=-1,
                                                    max_iter=50000, verbose=0, warm_start=True))
        elif sub_clf_name == KNeighborsClassifier.__name__:
            clf = OneVsRestClassifier(KNeighborsClassifier(algorithm='auto', n_jobs=-1))

    # Create a classifier copy for parameter selection
    clf_search = deepcopy(clf)

    if issparse(X_tr_src) and issparse(X_dev_src):
        X_combined = scipy.sparse.vstack([X_tr_src, X_dev_src])
        lb_combined = scipy.sparse.hstack([lb_tr_src, lb_dev_src])
    elif ~issparse(X_tr_src) and ~issparse(X_dev_src):
        X_combined = np.vstack([X_tr_src, X_dev_src])
        lb_combined = np.concatenate([lb_tr_src, lb_dev_src], axis=0)
    else:
        raise ValueError('Both train and dev set has to be sparse or dense.')

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = np.hstack((np.ones(X_tr_src.shape[0], dtype=int) * -1, np.zeros(X_dev_src.shape[0], dtype=int)))
    ps = PredefinedSplit(split_index)
    # # Optimize parameters using dev set
    if clf_name == AdaBoostClassifier.__name__ \
            or clf_name == MLPClassifier.__name__:
            # or clf_name == VBLogisticRegression.__name__\
        # # Optimize parameters using dev set
        search = RandomizedSearchCV(clf_search, param_distributions=param_grid, cv=ps,
                                    n_jobs=-1, verbose=1, refit=False).fit(X_combined, lb_combined)
    elif clf_name == AdaBoostClassifier.__name__:
        search = HalvingGridSearchCV(clf_search, param_grid,
                                     resource='n_estimators',
                                     max_resources=10, random_state=0).fit(X_combined, lb_combined)
    else:
        from scipy import sparse

        search = GridSearchCV(clf_search, param_grid=param_grid, cv=ps,
                              n_jobs=-1, verbose=1, refit=False).fit(X_combined, list(lb_combined))

    print(search.best_params_)

    clf.set_params(**search.best_params_).fit(X_tr_src, list(lb_tr_src))

    accuracies = {}
    f1_scores = {}
    recall_scores = {}
    micro_f1 = {}
    macro_f1 = {}
    precision_scores = {}
    predictions = {}
    for X_data, ld_data, type_data in zip([X_tr_src, X_tr_trg, X_tst_trg, X_dev_src],
                                          [lb_tr_src, lb_tr_trg, lb_tst_trg, lb_dev_src],
                                          ['train_src', 'train_trg', 'test_trg', 'dev_src']):
        y_pred = clf.predict(X_data)
        # y_proba = clf.predict_proba(X_data)

        acc = accuracy_score(ld_data, y_pred)
        f1 = f1_score(ld_data, y_pred, average='weighted', labels=np.unique(y_pred))
        recall = recall_score(ld_data, y_pred, average='weighted', labels=np.unique(y_pred))
        precision = precision_score(ld_data, y_pred, average='weighted', labels=np.unique(y_pred))

        micro_f1.update({type_data: metrics.f1_score(ld_data, y_pred, average="micro", labels=np.unique(y_pred))})
        macro_f1.update({type_data: metrics.f1_score(ld_data, y_pred, average="macro", labels=np.unique(y_pred))})
        recall_scores.update({type_data: recall})
        precision_scores.update({type_data: precision})

        f1_scores.update({type_data: f1})
        accuracies.update({type_data: acc})
        predictions.update({type_data: y_pred})
        # print('<<--- Classification {} accuracy {} %--->>'.format(type_data, acc * 100))

    return accuracies, f1_scores, recall_scores, precision_scores, micro_f1, macro_f1, predictions


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_folder=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if save_folder:
        plt.savefig(os.path.join(save_folder, '{}_cm.eps'.format(accuracy)), format='eps')

    # plt.show()

    plt.close(fig)

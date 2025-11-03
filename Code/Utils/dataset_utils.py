import os
import torch
import json
import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset

from sklearn.preprocessing import LabelEncoder

from Code.Utils.timer import Timer
from Code.Utils.utils import tokenize_data, encode_labels, encode_multi_labels

time = Timer()


# _DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_coo_tensor(mat, device):
    mat = mat.tocoo().astype(np.float32)
    values = mat.data
    indices = np.vstack((mat.row, mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mat.shape

    t = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)

    # t = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)
    return t


def read_bucc(language):
    bucc_lang_keys = {'german': 'de', 'english': 'en'}

    # Input files for BUCC2018 shared task
    bucc_dataset_root = '/media/blue/tsingalis/nnParafac2/Data/Bucc/german/'

    lang_file = os.path.join(bucc_dataset_root, f"de-en.training.{bucc_lang_keys[language]}")
    print("Read source file")
    source = {}
    with open(lang_file, encoding="utf8") as fIn:
        for line in fIn:
            id, sentence = line.strip().split("\t", maxsplit=1)
            source[id] = sentence

    # n_samples = 10000
    # source = {k: source[k] for k in sorted(source.keys())[:n_samples]}

    ### Encode source sentences
    source_ids = list(source.keys())
    source_sentences = [source[id] for id in source_ids]

    # German, French etc !!!!!!!!!!!!
    df_source = pd.DataFrame({'Text': source_sentences})

    return df_source


def read_parallel(tknz_params, args, parallel_datasets, device, random_state=1, n_samples=None, save_exp_folder=None):
    from Code.Utils.utils import read_aligned_raw_data, tokenize_data
    dataset_length = {'WikiMatrix': 100000, 'Europarlv8': 100000,
                      'OpenSubtitles2018': 100000, 'Tanzil': 100000,
                      'GlobalVoices': 100000, 'UnitedNations': 100000,
                      'WikiMedia': 100000}

    df_parallel_data_dict = {}
    data_dict, tf_idf_transformers_dict, vocabs_dict = dict(), dict(), dict()
    n_corpus = 0
    for language in ['english', args.target_lang]:
        df_parallel_data = []
        for parallel_dataset in parallel_datasets:
            print(f'--- Process {parallel_dataset} ---')
            dataset_folder = os.path.join(args.root, 'Data', parallel_dataset, f'english-{args.target_lang}')
            try:
                _df_parallel_data = read_aligned_raw_data(dataset_folder, language,
                                                          n_samples=None, masked=args.masked)
            except FileNotFoundError:
                print(f'Language pair english-{args.target_lang} not found in {parallel_dataset}')
                continue
            _df_parallel_data['corpus_name'] = parallel_dataset
            df_parallel_data.append(_df_parallel_data)
            n_corpus += 1

            print(f'Load {parallel_dataset} -- language {language} -- n_samples {len(_df_parallel_data)}')
        df_parallel_data = pd.concat(df_parallel_data).reset_index()
        df_parallel_data_dict.update({language: df_parallel_data})
        # time.start()

    n_en_data = len(df_parallel_data_dict['english'])
    n_trg_data = len(df_parallel_data_dict[args.target_lang])

    # This is needed to sample parallel document, see below where random_state is used.
    assert n_en_data == n_trg_data

    for language in ['english', args.target_lang]:
        # We need a fixed random state to sample parallel rows
        df_parallel_data = df_parallel_data_dict[language].groupby('corpus_name', group_keys=False).apply(
            lambda x: x.sample(min(len(x), 2 * round(n_samples / n_corpus)), random_state=random_state))
        tokenize_data_results = tokenize_data(language=language,
                                              df_train=df_parallel_data,
                                              df_dev=None,
                                              df_test=None,
                                              vocab=None,
                                              tfidf_transformer=None,
                                              **tknz_params)

        _data_train, _data_dev, _data_test, _vectorizer_dict, _tf_idf_transformer, _vocab = tokenize_data_results

        # time.stop(tag='Tokenize {} -- Vocab size {}'.format(language, len(_vocab)), verbose=True)

        data_dict.update({language: to_coo_tensor(_data_train, device).t().coalesce()})

        # data_dict.update({language: torch.from_numpy(_data_train.toarray()).float().to(_DEVICE_).T})
        tf_idf_transformers_dict.update({language: _tf_idf_transformer})
        vocabs_dict.update({language: _vocab})

    if save_exp_folder is not None:
        for language in ['english', args.target_lang]:

            if not os.path.exists(os.path.join(save_exp_folder, f'vocab_{language}.json')):
                with open(os.path.join(save_exp_folder, f'vocab_{language}.json'), 'w', encoding='utf8') as jf:
                    json.dump(vocabs_dict[language], jf, indent=4, sort_keys=True, ensure_ascii=False)

            if not os.path.exists(os.path.join(save_exp_folder, f'tf_idf_transformer_{language}.pkl')):
                with open(os.path.join(save_exp_folder, f'tf_idf_transformer_{language}.pkl'), 'wb') as fin:
                    pickle.dump(tf_idf_transformers_dict[language], fin)

            if not os.path.exists(os.path.join(save_exp_folder, f'vectorizer_{language}.pkl')):
                with open(os.path.join(save_exp_folder, f'vectorizer_{language}.pkl'), 'wb') as fin:
                    pickle.dump(tf_idf_transformers_dict[language], fin)

    return data_dict, _vectorizer_dict, tf_idf_transformers_dict, vocabs_dict


def align_dataset(data_dict, target_lang, pseudo_labels):
    # Use the predicted labels as the actual labels (pseudo-labeling) in target language for alignment
    data_dict[target_lang]['Label'] = pseudo_labels

    for language in ['english', target_lang]:
        data_dict[language].rename(columns={'Text': f'Text_{language}'}, inplace=True)
        data_dict[language].rename(columns={'Label': f'Label_{language}'}, inplace=True)

    merged_data = data_dict['english'].merge(
        data_dict[target_lang],
        left_on=['Label_english', data_dict['english'].groupby('Label_english').cumcount()],
        right_on=[f'Label_{target_lang}', data_dict[target_lang].groupby(f'Label_{target_lang}').cumcount()]
    )

    for language in ['english', target_lang]:
        data_dict.update(
            {
                language: merged_data[[f'Label_{language}', f'Text_{language}']].rename(
                    columns={f'Text_{language}': 'Text', f'Label_{language}': 'Label'}, inplace=False)
            }
        )
    for language in ['english', target_lang]:
        data_dict[language].sort_values('Label', inplace=True)
        data_dict[language].reset_index(drop=True, inplace=True)
    return data_dict


def noisy_labels(data_dict, target_lang, perc_noise=0.1):
    gnd_labels = data_dict[target_lang]['Label'].copy().values
    mask = np.random.rand(len(data_dict[target_lang])) < perc_noise  # flip the labels

    unique_labels = set(list(gnd_labels))
    data_dict[target_lang].loc[mask, 'Label'] = data_dict[target_lang][mask].apply(
        lambda row: np.random.choice(list(unique_labels - set([row['Label']]))), axis=1)

    noisy_labels = data_dict[target_lang]['Label'].values
    print(f'Pseudo-labels accuracy for {target_lang}: {sum(gnd_labels == noisy_labels) / len(gnd_labels) * 100}')

    return noisy_labels


def read_raw_data_amazon(data_folder, language, domain, n_samples=None):
    file_train = os.path.join(data_folder, language.lower(), domain, "train.review")
    file_dev = os.path.join(data_folder, language.lower(), domain, "dev.review")
    file_test = os.path.join(data_folder, language.lower(), domain, "test.review")

    df_train = pd.read_csv(file_train, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])
    df_dev = pd.read_csv(file_dev, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])
    df_test = pd.read_csv(file_test, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])

    # df_train['Text'] = df_train['Text'].str[:50]
    # df_dev['Text'] = df_dev['Text'].str[:50]
    # df_test['Text'] = df_test['Text'].str[:50]

    return df_train[:n_samples], df_dev[:n_samples], df_test[:n_samples]


def read_raw_data_mldoc(data_folder, language, n_samples=None, num_train=1000):
    file_train = os.path.join(data_folder, language, language.lower() + ".train." + str(num_train))
    file_dev = os.path.join(data_folder, language, language.lower() + ".dev")
    file_test = os.path.join(data_folder, language, language.lower() + ".test")

    df_train = pd.read_csv(file_train, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])
    df_dev = pd.read_csv(file_dev, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])
    df_test = pd.read_csv(file_test, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])

    return df_train[:n_samples], df_dev[:n_samples], df_test[:n_samples]


def stratified_sample_df(df, col, n_samples, min_samples_per_class=10, seed=1):
    """ Convert each multilabel vector to a unique string """
    df[col + '_encoded'] = LabelEncoder().fit_transform(['_'.join(map(str, sorted(lbl))) for lbl in df[col].values])

    df_ = df.groupby(col + '_encoded').filter(lambda x: len(x) >= min_samples_per_class)
    df_ = df_.sample(n_samples, replace=True, random_state=seed)

    df_.reset_index(level=0, drop=True)
    df_.drop(col + '_encoded', axis=1, inplace=True)

    return df_


def read_task_dataset(vocabs, tf_idf_transformers, tknz_params, device, args, n_samples=None,
                      task=None, subtask=None, pseudo_labels=None):
    print(f'Read {task} dataset')
    dataset_folder = os.path.join(args.root, 'Data', task)

    data_train_dict, data_dev_dict, data_test_dict = dict(), dict(), dict()
    label_train_dict, label_dev_dict, label_test_dict = dict(), dict(), dict()

    tf_idf_transformers_dict, vocabs_dicts = dict(), dict()

    multi_eurlex_languages = {'english': 'en', 'french': 'fr', 'german': 'de'}

    for language in ['english', args.target_lang]:

        if task == 'MlDoc':
            df_train, df_dev, df_test = read_raw_data_mldoc(dataset_folder, language, n_samples)
        elif task == 'Amazon':
            df_train, df_dev, df_test = read_raw_data_amazon(dataset_folder, language, subtask, n_samples=n_samples)
        else:
            tr_multi_eurlex_dataset = load_dataset('multi_eurlex',
                                                   split="train[:30%]",
                                                   language=multi_eurlex_languages[language],
                                                   label_level=args.label_level,
                                                   keep_in_memory=True,
                                                   num_proc=os.cpu_count())
            dev_multi_eurlex_dataset = load_dataset('multi_eurlex',
                                                    split="validation",
                                                    language=multi_eurlex_languages[language],
                                                    label_level=args.label_level,
                                                    keep_in_memory=True,
                                                    num_proc=os.cpu_count())
            tst_multi_eurlex_dataset = load_dataset('multi_eurlex',
                                                    split="test",
                                                    language=multi_eurlex_languages[language],
                                                    label_level=args.label_level,
                                                    keep_in_memory=True,
                                                    num_proc=os.cpu_count())

            from concurrent.futures import ThreadPoolExecutor

            def extract_text(item, key):
                return item[key]

            # # Assuming multi_eurlex_dataset is your dataset
            # train_texts = tr_multi_eurlex_dataset['text']
            #
            # # Use ThreadPoolExecutor to parallelize the loop
            # with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            #     train_txt = list(executor.map(lambda item:
            #                                   extract_text(item, multi_eurlex_languages[language]),
            #                                   train_texts))

            # train_txt = [i['en'] for i in tr_multi_eurlex_dataset['text']]
            # train_labels = [i for i in tr_multi_eurlex_dataset['labels']]
            train_txt = tr_multi_eurlex_dataset['text']
            train_labels = tr_multi_eurlex_dataset['labels']
            df_train = pd.DataFrame({'Text': train_txt, 'Label': train_labels})

            # dev_txt = [i[multi_eurlex_languages[language]] for i in dev_multi_eurlex_dataset['text']]
            # dev_labels = [i for i in dev_multi_eurlex_dataset['labels']]
            dev_txt = dev_multi_eurlex_dataset['text']
            dev_labels = dev_multi_eurlex_dataset['labels']
            df_dev = pd.DataFrame({'Text': dev_txt, 'Label': dev_labels})

            # test_txt = [i[multi_eurlex_languages[language]] for i in tst_multi_eurlex_dataset['text']]
            # test_labels = [i for i in tst_multi_eurlex_dataset['labels']]
            test_txt = tst_multi_eurlex_dataset['text']
            test_labels = tst_multi_eurlex_dataset['labels']
            df_test = pd.DataFrame({'Text': test_txt, 'Label': pd.Series(test_labels)})

        # if pseudo_labels is not None:
        # df_dev.rename(columns={'Text': f'{language}'}, inplace=True)
        # df_test.rename(columns={'Text': f'{language}'}, inplace=True)
        # n_samples = 100

        data_train_dict.update({language: df_train})
        data_dev_dict.update({language: df_dev})
        data_test_dict.update({language: df_test})

    # pseudo_labels = noisy_labels(data_train_dict, args.target_lang, perc_noise=0.3)

    # Encode labels
    # labels_train_dict, labels_dev_dict, labels_test_dict, label_encoder_dict = {}, {}, {}, {}
    for language in ['english', args.target_lang]:
        df_train, df_dev, df_test = data_train_dict[language], data_dev_dict[language], data_test_dict[language]

        if task == 'MlDoc' or task == 'Amazon':
            labels, classes, label_encoder = encode_labels(df_train['Label'].values,
                                                           df_dev['Label'].values,
                                                           df_test['Label'].values)
        else:

            labels, classes, label_encoder = encode_multi_labels(df_train['Label'].values,
                                                                 df_dev['Label'].values,
                                                                 df_test['Label'].values)

        labels_train, labels_dev, labels_test = labels

        # labels_train_dict.update({language: label_encoder.inverse_transform(labels_train)})
        # labels_dev_dict.update({language: label_encoder.inverse_transform(labels_dev)})
        # labels_test_dict.update({language: label_encoder.inverse_transform(labels_test)})
        # label_encoder_dict.update({language: label_encoder})

        data_train_dict[language]['Label_raw'] = label_encoder.inverse_transform(labels_train)
        data_dev_dict[language]['Label_raw'] = label_encoder.inverse_transform(labels_dev)
        data_test_dict[language]['Label_raw'] = label_encoder.inverse_transform(labels_test)

        data_train_dict[language]['Label'] = labels_train.tolist()
        data_dev_dict[language]['Label'] = labels_dev.tolist()
        data_test_dict[language]['Label'] = labels_test.tolist()

    if task == 'MultiEurlex' and n_samples is not None:
        for language in ['english', args.target_lang]:
            df_train_sampled = stratified_sample_df(data_train_dict[language], 'Label_raw', n_samples,
                                                    min_samples_per_class=50, seed=1)
            df_dev_sampled = stratified_sample_df(data_dev_dict[language], 'Label_raw', n_samples,
                                                  min_samples_per_class=50, seed=1)
            df_test_sampled = stratified_sample_df(data_test_dict[language], 'Label_raw', n_samples,
                                                   min_samples_per_class=50, seed=1)

            df_train_sampled.drop('Label_raw', axis=1, inplace=True)
            df_dev_sampled.drop('Label_raw', axis=1, inplace=True)
            df_test_sampled.drop('Label_raw', axis=1, inplace=True)

            data_train_dict.update({language: df_train_sampled})
            data_dev_dict.update({language: df_dev_sampled})
            data_test_dict.update({language: df_test_sampled})

    if pseudo_labels is not None:
        data_train_dict = align_dataset(data_train_dict, args.target_lang, pseudo_labels)

    for language in ['english', args.target_lang]:
        df_train, df_dev, df_test = data_train_dict[language], data_dev_dict[language], data_test_dict[language]

        print(f'Load {task} -- language {language} -- n_samples {len(df_train)}')

        parallel_vocab, parallel_tfidf_transformer = vocabs[language], tf_idf_transformers[language]

        time.start()
        tokenize_data_results = tokenize_data(language=language,
                                              df_train=df_train,
                                              df_dev=df_dev,
                                              df_test=df_test,
                                              vocab=parallel_vocab,
                                              tfidf_transformer=parallel_tfidf_transformer,
                                              **tknz_params)

        _data_train, _data_dev, _data_test, _, _tfidf_transformer, _vocab = tokenize_data_results
        time.stop(tag='Tokenize {}'.format(language), verbose=True)

        tf_idf_transformers_dict.update({language: _tfidf_transformer})
        vocabs_dicts.update({language: _vocab})

        # labels, classes, label_encoder = encode_labels(df_train['Label'].values,
        #                                                df_dev['Label'].values,
        #                                                df_test['Label'].values)

        # labels_train, labels_dev, labels_test = labels

        # Replace text data with arithmetic data
        data_train_dict.update({language: to_coo_tensor(_data_train, device).t().coalesce()})

        data_dev_dict.update({language: torch.from_numpy(_data_dev.toarray()).float().to('cpu').T})

        data_test_dict.update({language: torch.from_numpy(_data_test.toarray()).float().to('cpu').T})

        label_train_dict.update({language: list(df_train['Label'].values)})

        label_dev_dict.update({language: list(df_dev['Label'].values)})

        label_test_dict.update({language: list(df_test['Label'].values)})

    labels = label_train_dict, label_dev_dict, label_test_dict
    data = data_train_dict, data_dev_dict, data_test_dict

    return data, labels, label_encoder, classes


def read_aligned_mldoc_amazon(rank, run, wiki_n_samples, tknz_params, args, exp_folder):
    # save_exp_folder = os.path.join(args.output, 'Exps', 'WikiMatrix', exp_folder,
    #                                f'S{wiki_n_samples}', args.target_lang)

    # pseudo_labels_root = os.path.join(exp_folder, args.task, f'R{rank}', f'run{run}', 'models')

    if args.sub_task is not None:
        pseudo_labels_root = os.path.join(exp_folder, args.task, args.sub_task, f'R{rank}', f'run{run}', 'models')
    else:
        pseudo_labels_root = os.path.join(exp_folder, args.task, f'R{rank}', f'run{run}', 'models')

    with open(os.path.join('../..', pseudo_labels_root, f'predictions_train_trg.txt'), 'r') as f:
        pseudo_labels = [int(x.rstrip()) for x in f]

    # root = os.path.join(args.output, 'Exps', 'WikiMatrix', exp_folder, f'S{wiki_n_samples}')

    vocabs_dict, tf_idf_transformers_dict = dict(), dict()
    for language in ['english', args.target_lang]:
        with open(os.path.join(exp_folder, f"vocab_{language}.json")) as json_file:
            vocabs_dict.update({language: json.load(json_file)})
        with open(os.path.join(exp_folder, f'tf_idf_transformer_{language}.pkl'), 'rb') as f:
            tf_idf_transformers_dict.update({language: pickle.load(f)})

    data, _, _, _, _ = read_task_dataset(vocabs_dict, tf_idf_transformers_dict,
                                         tknz_params,
                                         args,
                                         n_samples=None,
                                         task=args.task,
                                         subtask=args.sub_task,
                                         pseudo_labels=pseudo_labels,
                                         device=device)
    data, _, _ = data

    # save_exp_folder = os.path.join(args.output, 'Exps', 'WikiMatrix', exp_folder, args.task, args.subtask)
    # Path(save_exp_folder).mkdir(parents=True, exist_ok=True)

    return data, tf_idf_transformers_dict, vocabs_dict

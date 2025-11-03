"""
References:

"""

import os
import sys

sys.path.append('/mnt/scratch_b/users/t/tsingalis/Documents/nnParafac2')
# sys.path.append('/media/data/tsingalis/nnParafac2/')

from Code.Utils.utils import compute_class
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import argparse

from Code.Utils.timer import Timer

import json

from Code.Utils.dataset_utils import read_task_dataset

from pathlib import Path
import numpy as np

import torch
from torch import Tensor
from torch import mm as ddmm
from torch.sparse import mm as sdmm

import re

time = Timer()

import tensorly as tl
import matplotlib.pyplot as plt

tl.set_backend('pytorch')


def dsmm(mat1: Tensor, mat2: Tensor) -> Tensor:
    return sdmm(mat2.t(), mat1.t()).t()


def map_data(data, U, S=None, device='cpu'):
    U = [u.to(device) for u in U]

    if S is not None:
        S = [s.to(device) for s in S]
        invS = []
        for id_modality, language in enumerate(['english', args.target_lang]):
            indices = torch.stack([torch.LongTensor([i, i]).to(device=device)
                                   for i in torch.arange(len(S[id_modality]))], dim=1)

            values = torch.cat(
                [torch.FloatTensor([1. / (s + 1e-8)]).to(device=device) for s in S[id_modality].values()])

            invS.append(torch.sparse.FloatTensor(indices, values, S[id_modality].shape).coalesce())

    Y = dict()
    for id_modality, language in enumerate(['english', args.target_lang]):
        if data[language].is_sparse:
            # print('------------------------------- Sparse Data matrix in inference -------------------------------')
            if S is not None:
                _Y = dsmm(sdmm(invS[id_modality], U[id_modality].T), data[language].to(device))
            else:
                _Y = dsmm(U[id_modality].T, data[language].to(device))
        else:
            if S is not None:
                _Y = ddmm(sdmm(invS[id_modality], U[id_modality].T), data[language].to(device))
            else:
                _Y = ddmm(U[id_modality].T, data[language])
        if device == 'cpu':
            _Y = _Y.numpy()

        Y.update({language: _Y})
    return Y


def check_run_folder(exp_folder):
    run = 1
    if args.pseudo_labels:
        run_folder = os.path.join(exp_folder, f'run{args.loaded_run}_pseudo_{run}')
    else:
        run_folder = os.path.join(exp_folder, 'run{}'.format(run))

    if not os.path.exists(run_folder):
        Path(run_folder).mkdir(parents=True, exist_ok=True)

        print("Path {} created".format(run_folder))
        return run_folder

    while os.path.exists(run_folder):
        run += 1
        if args.pseudo_labels:
            run_folder = os.path.join(exp_folder, f'run{args.loaded_run}_pseudo_{run}')
        else:
            run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    print("Path {} created".format(run_folder))
    return run_folder


def save_json(obj, path):
    with open(path, 'w') as jf:
        json.dump(obj, jf, indent=4, sort_keys=True)


def normalize_data(X_tr, X_dev, X_tst):
    from sklearn.preprocessing import Normalizer

    print('Normalize data...')
    scaler = Normalizer()
    X_tr = scaler.fit_transform(X_tr)
    X_dev = scaler.transform(X_dev)
    X_tst = scaler.transform(X_tst)

    return X_tr, X_dev, X_tst


def save_stats(labels, predictions, classes, save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Test accuracy of target lang
    report = classification_report(labels, predictions, output_dict=True)

    with open(os.path.join(save_path, 'report.json'), 'w') as jf:
        json.dump(report, jf, indent=4, sort_keys=True)

    with plt.rc_context(rc={'figure.max_open_warning': 0}):
        cm = ConfusionMatrixDisplay.from_predictions(
            labels, predictions, display_labels=classes,
            xticks_rotation=45
        )
        cm.figure_.savefig(os.path.join(save_path, 'ConfusionMatrix.eps'),
                           bbox_inches='tight')

        ncm = ConfusionMatrixDisplay.from_predictions(
            labels, predictions, display_labels=classes,
            normalize='true', values_format='.2%',
            xticks_rotation=45
        )
        ncm.figure_.savefig(os.path.join(save_path, 'normConfusionMatrix.eps'),
                            bbox_inches='tight')


def save_dev_train_test_acc(run_folder, language, accuracies):
    with open(os.path.join(run_folder, language, 'dev_acc.txt'), 'a') as file_object:
        file_object.write(str(accuracies['dev']) + "\n")

    with open(os.path.join(run_folder, language, 'test_acc.txt'), 'a') as file_object:
        file_object.write(str(accuracies['test']) + "\n")

    with open(os.path.join(run_folder, language, 'train_acc.txt'), 'a') as file_object:
        file_object.write(str(accuracies['train']) + "\n")


def main():
    # tknz_params = {
    #     'cased': False,
    #     'use_tfidf': True,
    #     'ngram_range': (1, args.ngram_range),
    #     'tok_lib': args.tokenizer,
    #     'norm': 'l2'}

    time.start()

    print('--------------------------------------------')
    print('Testing model rank: {}...'.format(args.loaded_rank))
    print('--------------------------------------------')

    load_exp_folder = str(os.path.join(args.root, 'models', '-'.join(args.parallel_dataset), args.alg,
                                       f'english-{args.target_lang}',
                                       f'run{args.loaded_run}'))

    with open(os.path.join(load_exp_folder, f'vocab_{args.target_lang}.json')) as json_file:
        vocab = json.load(json_file)

    with open(os.path.join(load_exp_folder, 'tknz_params.json')) as json_file:
        tknz_params = json.load(json_file)
    tknz_params['ngram_range'] = tuple(tknz_params['ngram_range'])

    with open(os.path.join(load_exp_folder, 'vocab_english.json')) as json_file:
        vocab_english = json.load(json_file)

    import pickle
    with open(os.path.join(load_exp_folder, 'tf_idf_transformer_english.pkl'), 'rb') as handle:
        tf_idf_transformer_english = pickle.load(handle)

    with open(os.path.join(load_exp_folder, f'tf_idf_transformer_{args.target_lang}.pkl'), 'rb') as handle:
        tf_idf_transformer = pickle.load(handle)

    tf_idf_transformers_dict, vocabs_dict = dict(), dict()
    vocabs_dict.update({'english': vocab_english})
    vocabs_dict.update({f'{args.target_lang}': vocab})

    tf_idf_transformers_dict.update({'english': tf_idf_transformer_english})
    tf_idf_transformers_dict.update({f'{args.target_lang}': tf_idf_transformer})

    data, data_labels, label_encoder, classes = read_task_dataset(vocabs=vocabs_dict,
                                                                  tf_idf_transformers=tf_idf_transformers_dict,
                                                                  tknz_params=tknz_params,
                                                                  args=args,
                                                                  n_samples=None,
                                                                  task=args.target_task,
                                                                  subtask=args.sub_task,
                                                                  pseudo_labels=None,
                                                                  device=device)

    model_exp_folder = os.path.join(load_exp_folder, f'R{args.loaded_rank}', 'models')

    labels_train, labels_dev, labels_test = data_labels
    data_train, data_dev, data_test = data

    import pathlib

    dev_src_acc, test_trg, train_src, train_trg = [], [], [], []
    # model_names = list(map(lambda x: f'{x}.pkl',
    #                        sorted([int(p.stem) for p in pathlib.Path(model_exp_folder).glob('*.pkl')])))

    model_numerical_names = np.unique(sorted([int(re.findall(r'\d+', p.stem).pop())
                                              for p in pathlib.Path(model_exp_folder).glob('*.pkl')]))

    # if args.parallel_dataset == 'WikiMedia' or args.parallel_dataset == 'Bible' or args.parallel_dataset == 'Europarlv8':
    #     umodel_names = list(map(lambda x: f'u{x}.pkl', model_numerical_names))
    #     smodel_names = list(map(lambda x: f's{x}.pkl', model_numerical_names))
    # else:
    #     umodel_names = list(map(lambda x: f'{x}.pkl', model_numerical_names))

    umodel_names = list(map(lambda x: f'u{x}.pkl', model_numerical_names))
    smodel_names = list(map(lambda x: f's{x}.pkl', model_numerical_names))

    for umode_name, smode_name in zip(umodel_names, smodel_names):
        # checkpoint_U = torch.load(mode_name)
        print(f'Load model {umode_name}')
        iter = int(re.findall(r'\d+', umode_name).pop())

        # if iter < 50:
        #     continue
        checkpoint_U = [u for u in torch.load(os.path.join(model_exp_folder, umode_name))]
        checkpoint_S = [s.coalesce() for s in torch.load(os.path.join(model_exp_folder, smode_name))]

        X_tr = map_data(data_train, checkpoint_U, S=None)
        X_dev = map_data(data_dev, checkpoint_U, S=None)
        X_tst = map_data(data_test, checkpoint_U, S=None)

        predict_labels = compute_class(X_tr_src=X_tr['english'].T,
                                       X_tr_trg=X_tr[args.target_lang].T,
                                       X_dev_src=X_dev['english'].T,
                                       X_tst_trg=X_tst[args.target_lang].T,
                                       lb_tr_src=labels_train['english'],
                                       lb_tr_trg=labels_train[args.target_lang],
                                       lb_dev_src=labels_dev['english'],
                                       lb_tst_trg=labels_test[args.target_lang],
                                       clf_name=args.clf,
                                       sub_clf_name=args.sub_clf_name)

        accuracies, f1_scores, recall_scores, precision_scores, micro_f1, macro_f1, predictions = predict_labels

        if args.sub_task is not None:
            save_exp_folder = os.path.join(load_exp_folder, f'R{args.loaded_rank}', 'results',
                                           args.clf, args.target_task, args.sub_task)
        else:
            save_exp_folder = os.path.join(load_exp_folder, f'R{args.loaded_rank}', 'results',
                                           args.clf, args.target_task)

        Path(save_exp_folder).mkdir(parents=True, exist_ok=True)

        print(f"save_exp_folder: {save_exp_folder}")

        Path(os.path.join(save_exp_folder,
                          umode_name.replace('.pkl', ''))).mkdir(parents=True, exist_ok=True)

        # Save the predictions of the target language in the train set, e.g., german, italian, etc.
        with open(os.path.join(save_exp_folder, umode_name.replace('.pkl', ''),
                               'predictions_train_trg.txt'), 'w') as file_object:
            file_object.write("\n".join(str(item) for item in predictions['train_trg']))

        if args.target_task != 'MultiEurlex':
            save_stats(labels_test[args.target_lang], predictions['test_trg'], classes,
                       os.path.join(save_exp_folder, 'test_trg'))

        dev_src_acc.append((iter, accuracies['dev_src']))
        test_trg.append((iter, accuracies['test_trg']))
        train_src.append((iter, accuracies['train_src']))
        train_trg.append((iter, accuracies['train_trg']))

        with open(os.path.join(save_exp_folder, 'dev_src_acc.txt'), 'w') as file:
            file.write('\n'.join(f'{e}\t{item}' for e, item in dev_src_acc))

        with open(os.path.join(save_exp_folder, 'test_trg_acc.txt'), 'w') as file:
            file.write('\n'.join(f'{e}\t{item}' for e, item in test_trg))

        with open(os.path.join(save_exp_folder, 'train_src_acc.txt'), 'w') as file:
            file.write('\n'.join(f'{e}\t{item}' for e, item in train_src))

        with open(os.path.join(save_exp_folder, 'train_trg_acc.txt'), 'w') as file:
            file.write('\n'.join(f'{e}\t{item}' for e, item in train_trg))

    with open(os.path.join(save_exp_folder, 'max_dev_src_acc.txt'), 'w') as file:
        file.write(str(max([item for _, item in dev_src_acc])))

    with open(os.path.join(save_exp_folder, 'max_test_trg_acc.txt'), 'w') as file:
        file.write(str(max([item for _, item in test_trg])))

    with open(os.path.join(save_exp_folder, 'max_train_src_acc.txt'), 'w') as file:
        file.write(str(max([item for _, item in train_src])))

    with open(os.path.join(save_exp_folder, 'max_train_trg_acc.txt'), 'w') as file:
        file.write(str(max([item for _, item in train_trg])))

    # End of epochs

    time.stop(tag='Testing', verbose=True)
    print('End of training!')


def none_or_str(v):
    if v == 'None':
        return ''
    if v in ['books', 'dvd', 'music']:
        return v
    else:
        raise argparse.ArgumentTypeError("'Books' or 'DVD' or 'Music' is expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--alg", default=None, type=str, required=True,
                        choices=['nnParafac2', 'Parafac2', 'aoadmm'],
                        help="The alg. to use: nnParafac2 or Parafac2 or aoadmm")

    parser.add_argument("--clf", default=None, type=str, required=True,
                        choices=['LogisticRegression', 'MultinomialNB', 'SVC', 'MLPClassifier',
                                 'KNeighborsClassifier', 'RandomForestClassifier', 'LinearSVC',
                                 'OneVsRestClassifier', 'AdaBoostClassifier', 'ComplementNB'],
                        help="The classifier to use.")

    parser.add_argument("--sub_clf_name", default=None, type=str, required=False,
                        choices=['LogisticRegression', 'MultinomialNB', 'SVC', 'MLPClassifier',
                                 'KNeighborsClassifier', 'RandomForestClassifier', 'LinearSVC',
                                 'SGDClassifier', 'AdaBoostClassifier'],
                        help="The sub-classifier to use.")

    parser.add_argument("--scaler", default=None, type=str, required=False,
                        choices=['MinMaxScaler'],
                        help="The scaler to use. Is necessary for NB classifier.")

    parser.add_argument("--target_task", default=None, type=str, required=True,
                        choices=['MlDoc', 'Amazon', 'MultiEurlex'],
                        help="The target task to perform")

    parser.add_argument("--sub_task",
                        nargs='?', const='', type=none_or_str, default=None, choices=['dvd', 'music', 'books'],
                        help="The sub_task to run the algorithm")

    parser.add_argument("--save_model", action='store_true',
                        help="Set --save_model in the parameters to save the model")

    parser.add_argument('--masked', action='store_true',
                        help="Set -- masked in the parameters to save the model")

    parser.add_argument("--target_lang", default=None, type=str, required=True,
                        choices=['french', 'german', 'italian', 'spanish', 'russian', 'chinese', 'japanese'],
                        help="The name of the target language.")

    # parser.add_argument("--tokenizer", default='nltk', type=str,
    #                     required=True, choices=['nltk', 'spacy'],
    #                     help="The tokenizer to use..")
    #
    # parser.add_argument("--ngram_range", default=1, type=int, required=False,
    #                     help="The ngram_range.")

    parser.add_argument("--root", default='./', type=str, required=True,
                        help="root of the project.")

    # parser.add_argument("--output", default='./', type=str, required=True,
    #                     help="output of the project.")

    parser.add_argument('--pseudo_labels', action='store_true',
                        help="Set --pseudo_labels in the parameters to use pseudo labels")

    parser.add_argument("--loaded_rank", default=None, type=int, required=False,
                        help="The rank of the models to load the pseudo labels.")

    parser.add_argument("--loaded_run", default=None, type=int, required=True,
                        help="The run of the models to load the pseudo labels.")

    parser.add_argument("--u_init", default="nndsvd", type=str, required=False,
                        choices=["random", "nndsvd", "nndsvda", "nndsvdar"],
                        help="The initialization type of U in nnParafac2")

    parser.add_argument("--label_level", default='level_1',
                        help="The level of labels of multi_eurlex.")

    parser.add_argument("--perc_samples", default=None, type=float,
                        help="The percentage of samples to load in dataset.")

    parser.add_argument("--parallel_dataset", default=None, type=str, required=True,
                        action='append',
                        choices=['WikiMatrix', 'WikiMedia', 'Bible', 'Bucc',
                                 'Tanzil', 'GlobalVoices', 'Tatoeba2023',
                                 'Europarlv8', 'CCAlignedv1', 'CCMatrixv1',
                                 'NewsCommentary', 'OpenSubtitles2018', 'MultiUNv1'],
                        help="The parallel or comparable corpora to load.")

    args = parser.parse_args()

    print(args)

    if args.target_task == 'Amazon':
        assert args.sub_task is not None, 'Subtask must be not none for Amazon'

    if args.pseudo_labels:
        assert args.loaded_rank is not None, 'loaded_rank is none while pseudo_labels is true'
        assert args.loaded_run is not None, 'loaded_run is none while pseudo_labels is true'

    # if args.clf == 'MultinomialNB' and args.alg == 'Parafac2':
    #     assert args.scaler is not None, 'scaler is none while alg is Parafac2. Select --scaler MinMaxScaler '

    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _N_STREAMS_ = 2

    main()

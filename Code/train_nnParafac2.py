
from Code.Utils.timer import Timer
from Code.nnParafac2 import nnParafac2
from Code.cuParafac2 import cuParafac2
from Code.Utils.dataset_utils import read_parallel

import torch

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter

import matcouply.decomposition as decomposition


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_run_folder(exp_folder):
    run = 1
    run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    if not os.path.exists(run_folder):
        Path(run_folder).mkdir(parents=True, exist_ok=True)

        # os.makedirs(run_folder)
        print("Path {} created".format(run_folder))
        return run_folder

    while os.path.exists(run_folder):
        run += 1
        run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    # os.makedirs(run_folder)
    print("Path {} created".format(run_folder))
    return run_folder


def to_coo_tensor(mat, device):
    mat = mat.tocoo().astype(np.float32)
    values = mat.data
    indices = np.vstack((mat.row, mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mat.shape

    t = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)
    return t


def main():
    time = Timer()

    if args.cpu:
        device = "cpu"
    elif args.gpu:
        device = "gpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: {}'.format(device))

    run_folder = check_run_folder(os.path.join(args.root,
                                               'models', '-'.join(args.parallel_dataset),
                                               args.alg,
                                               'english-{}'.format(args.target_lang)))

    with open(os.path.join(run_folder, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    tknz_params = {'cased': False,
                   'use_tfidf': True,
                   'ngram_range': (args.ngram_range_min, args.ngram_range_max),
                   'tok_lib': args.tokenizer,
                   'norm': 'l2',
                   'analyzer': args.analyzer}

    with open(os.path.join(run_folder, 'tknz_params.json'), 'w') as fp:
        json.dump(tknz_params, fp)

    output_read_parallel = read_parallel(tknz_params=tknz_params,
                                         args=args,
                                         parallel_datasets=args.parallel_dataset,
                                         n_samples=args.parallel_n_samples,
                                         save_exp_folder=run_folder,
                                         device=device)

    parallel_data_dict, _, _, _ = output_read_parallel
    parallel_data = [parallel_data_dict[language] for language in ['english', args.target_lang]]

    time.start()
    for true_rank in range(args.minR, args.maxR, args.stepR):
        print('Training model rank: {}...'.format(true_rank))

        # Create log folder
        log_folder = os.path.join(run_folder, f'R{true_rank}', 'logs')

        Path(log_folder).mkdir(parents=True, exist_ok=True)

        # Create model folder
        save_model_folder = os.path.join(run_folder, f'R{true_rank}', 'models')

        Path(save_model_folder).mkdir(parents=True, exist_ok=True)

        parameters = {"rank": true_rank,
                      "seed": 45,
                      "log_folder": log_folder,
                      "max_m_iter": _N_EPOCHS_,
                      "error_tol": 1e-6,
                      'approx_fit_error': 200,
                      "verbose":True}

        # Save model parameters
        with open(os.path.join(run_folder, 'fit_parameters.json'), 'w') as fp:
            json.dump(parameters, fp, indent=4, sort_keys=True)

        tf_writer = SummaryWriter(log_folder)

        if args.alg == 'Parafac2' or args.alg == 'nnParafac2':
            if args.alg == 'Parafac2':
                model = cuParafac2(device=device, **parameters)
            elif args.alg == 'nnParafac2':
                # "random", "nndsvd", "nndsvda", "nndsvdar", "uniform"
                model = nnParafac2(device=device, u_init=args.u_init, w_init=args.w_init, **parameters)

            for epoch in range(1, _N_EPOCHS_ + 1):
                mean_loss, loss = model.partial_fit(parallel_data)
                if tf_writer:
                    tf_writer.add_scalars('Error', {'error': mean_loss}, epoch)
                    for k in range(_N_LANGUAGES_):
                        tf_writer.add_scalars('Error_X{}'.format(k), {'error': loss[k]}, epoch)

                epoch += 1
                if epoch % 1 == 0:
                    print('m_iter {} - model error {}'.format(epoch, mean_loss))

                save_cond = epoch % 2 == 0 if args.alg == 'Parafac2' else epoch % 5 == 0 and epoch > 40
                if save_cond:
                    torch.save(model.get_U(), os.path.join(save_model_folder, f"u{epoch}.pkl"))
                    torch.save(model.get_S(), os.path.join(save_model_folder, f"s{epoch}.pkl"))

        elif args.alg == 'aoadmm':
            parallel_data = [p.to_dense().detach().cpu().numpy() for p in parallel_data]  # First source, second target.
            cmf, diagnostics = decomposition.parafac2_aoadmm(
                parallel_data,
                rank=true_rank,
                non_negative=True,
                n_iter_max=45,
                l2_penalty=0.9,
                tol=1e-8,
                verbose=100,
                return_errors=True,
                random_state=45
            )
            weights, factors = cmf
            # A, B_is, C = factors, https://matcouply.readthedocs.io/en/latest/coupled_matrix_factorization.html
            U = factors[1]
    time.stop(tag='Training', verbose=True)



if __name__ == "__main__":
    """
    --u_init
uniform
--w_init
nndsvdar
--minR
1500
--maxR
4500
--stepR
500
--target_lang
spanish
--root
/media/blue/tsingalis/nnParafac2
--tokenizer
moses
--ngram_range_max
1
--ngram_range_min
1
--alg
nnParafac2
--analyzer
word
--parallel_dataset
WikiMatrix
--parallel_n_samples
100000
    
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--alg", default=None, type=str, required=True,
                        choices=['nnParafac2', 'Parafac2', 'aoadmm'],
                        help="The alg. to use: nnParafac2 or Parafac2 or aoadmm")

    parser.add_argument("--u_init", type=str, required=True,
                        choices=["uniform", "random", "nndsvd", "nndsvda", "nndsvdar"],
                        help="The initialization type of U in nnParafac2")

    parser.add_argument("--w_init", type=str, required=True,
                        choices=["uniform", "random", "nndsvd", "nndsvda", "nndsvdar"],
                        help="The initialization type of U in nnParafac2")

    parser.add_argument("--minR", default=None, type=int, required=True,
                        help="The min rank to start the evaluation of the model.")

    parser.add_argument("--maxR", default=None, type=int, required=True,
                        help="The min rank to start the evaluation of the model.")

    parser.add_argument("--stepR", default=500, type=int, required=False,
                        help="The step rank to start the evaluation of the model.")

    parser.add_argument("--target_lang", default=None, type=str, required=True,
                        help="The name of the target language.")

    parser.add_argument("--tokenizer", default='nltk', type=str, required=True,
                        choices=['nltk', 'spacy', 'moses', 'laser', 'BertTokenizer'],
                        help="The type of the tokenizer.")

    parser.add_argument("--masked", type=str2bool, nargs='?', const=True, default=False,
                        help="Masked data or not.")

    parser.add_argument("--root", default='./', type=str, required=True,
                        help="Root of the main folder")

    parser.add_argument("--ngram_range_max", default=1, type=int, required=False,
                        help="The ngram_range_max.")

    parser.add_argument("--ngram_range_min", default=1, type=int, required=False,
                        help="The ngram_range_min.")

    parser.add_argument("--parallel_dataset", default=None, type=str, required=True,
                        action='append',
                        choices=['WikiMatrix', 'WikiMedia', 'Bible', 'Bucc', 'Tanzil', 'GlobalVoices',
                                 'Europarlv8', 'CCAlignedv1', 'CCMatrixv1', 'Tatoeba2023',
                                 'NewsCommentary', 'OpenSubtitles2018', 'MultiUNv1'],
                        help="The parallel or comparable corpora to load.")

    parser.add_argument("--parallel_n_samples", default=-1, type=int, required=False,
                        help="The number of samples to train.")

    parser.add_argument("--analyzer", default='word', type=str, required=True, choices=['word', 'char'],
                        help="Root of the main folder")

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    if args.target_lang not in ['german', 'french', 'spanish', 'italian', 'japanese', 'russian', 'chinese']:
        raise ValueError('Target language should be one of german, french, spanish, italian, japanese, russian".')

    print('### Target language: {} - Number of samples {} '
          '- tokenizer: {} - masked: {} ###'.format(args.target_lang,
                                                    args.parallel_n_samples, args.tokenizer, args.masked))

    _N_EPOCHS_ = 20 if args.alg == 'Parafac2' else 100
    _N_LANGUAGES_ = 2

    main()

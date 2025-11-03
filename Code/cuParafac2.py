import numpy as np

import torch
from torch import Tensor
from torch import mm as ddmm
from torch.sparse import mm as sdmm

from scipy.sparse import diags
from scipy.sparse import csr_matrix

from tqdm import tqdm

from Code.Utils.timer import Timer

from tensorboardX import SummaryWriter


def dsmm(mat1: Tensor, mat2: Tensor) -> Tensor:
    return sdmm(mat2.t(), mat1.t()).t()


def dsewmm(d, s):
    i = s.indices()
    v = s.values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())


INIT_METHOD_U = 'uniform'
INIT_METHOD_W = 'uniform'

torch.set_printoptions(precision=10)


class cuParafac2:

    def __init__(self, rank, error_tol=1e-8, device='cpu', approx_fit_error=None,
                 max_m_iter=100, log_folder=None, seed=0, verbose=False):
        self.rank = rank
        self.seed = seed
        self.device = device
        self.max_m_iter = max_m_iter
        self.error_tol = error_tol
        self.approx_fit_error = approx_fit_error
        self.verbose = verbose

        if log_folder:
            self.tf_writer = SummaryWriter(log_folder)

        self.__n_samples, self.__n_languages, self._n_terms = 0, 0, []
        self._U, self._S, self._H, self._W = [], [], None, None

        self.eps = torch.tensor(1e-20, dtype=torch.float32, device=device)
        self.reg_eps = torch.tensor(1e-8, dtype=torch.float32, device=device)
        self.rng = np.random.default_rng(seed)

        self.unit_test = False

    def _check_init(self, X):

        if not self._U and not self._S and not self._H and not self._W:
            self.__initialize_decomposition(X)

    def get_U(self):
        return [self._U[k].clone() for k in range(self.__n_languages)]

    def get_H(self):
        return self._H.clone()

    def get_S(self):
        return [self._S[k].clone() for k in range(self.__n_languages)]

    def get_W(self):
        return self._W.clone()

    def __initialize_decomposition(self, X):
        print('Initialize cuParafac2')
        self.__init_U(X)  # projection matrices
        self.__init_H()  # factors
        self.__init_W()  # factors
        self.__init_S()  # factors

    def __init_U(self, X):

        for k in range(self.__n_languages):
            if self.verbose:
                print('Initialize U[{}] (dim: {} x {})...'.format(k, self._n_terms[k], self.rank))

            if INIT_METHOD_U == 'gaussian':
                avg = torch.sqrt(torch.mean(X[k].values()) / self.rank)
                rng = torch.Generator(device=self.device)
                rngs = rng.manual_seed(self.seed)
                U = avg * torch.randn(size=(self._n_terms[k], self.rank),
                                      dtype=X[k].dtype, generator=rngs,
                                      device=self.device)
                torch.abs(U, out=U)
            else:
                U = torch.tensor(self.rng.uniform(-0.01, 0.01, size=(self._n_terms[k], self.rank)),
                                 dtype=torch.float32, device=self.device)

            self._U.append(U)

    def __init_H(self):
        if self.verbose:
            print('Initialize H (dim: {} x {})...'.format(self.rank, self.rank))
        self._H = torch.eye(self.rank, self.rank, device=self.device)

    def __init_W(self):
        if self.verbose:
            print('Initialize W (dim: {} x {})...'.format(self.__n_samples, self.rank))
        self._W = torch.tensor(self.rng.uniform(-0.01, 0.01, size=(self.__n_samples, self.rank)),
                               dtype=torch.float32, device=self.device)

    def __init_S(self):
        for k in range(self.__n_languages):
            if self.verbose:
                print(f'Initialize S[{k}] (dim: {self.rank} x {self.rank})...')
            indices = torch.stack([torch.LongTensor([i, i]).to(device=self.device)
                                   for i in torch.arange(self.rank)], dim=1)
            values = torch.cat([torch.FloatTensor([1]).to(device=self.device) for _ in torch.arange(self.rank)])


            self._S.append(torch.sparse_coo_tensor(indices, values, torch.Size((self.rank, self.rank))).coalesce())

    def __partial_fit_U(self, X):

        for k in range(self.__n_languages):

            u, s, vt = torch.linalg.svd(dsmm(ddmm(dsmm(self._H, self._S[k]), self._W.T), X[k].t()),
                                        full_matrices=False)
            self._U[k] = ddmm(u, vt).T

            if self.unit_test:
                _W = self._W.detach().cpu().numpy()
                _H = self._H.to_dense().detach().cpu().numpy()
                _U = [self._U[k].detach().cpu().numpy() for k in range(self.__n_languages)]
                _X = [X[k].to_dense().detach().cpu().numpy() for k in range(self.__n_languages)]
                _S = [self._S[k].to_dense().detach().cpu().numpy() for k in range(self.__n_languages)]

                csr_matrix.dot(_H, _S[k])
                _u, _s, _vt = np.linalg.svd(_H.dot(_S[k]).dot(_W.T).dot(_X[k].T), full_matrices=False)
                _U[k] = (_u.dot(_vt)).T

                eq_test = np.allclose(self._U[k].detach().cpu().numpy(), _U[k], rtol=1e-4, atol=1.e-4)

                assert eq_test, '_U error'

    def __partial_fit_H(self, X):

        lhs = torch.stack([dsmm(ddmm(dsmm(self._U[k].T, X[k]), self._W), self._S[k])
                           for k in range(self.__n_languages)]).sum(0)

        rhs = torch.stack([dsmm(sdmm(self._S[k], ddmm(self._W.T, self._W)), self._S[k])
                           for k in range(self.__n_languages)]).sum(0)

        self._H = ddmm(lhs, torch.linalg.inv(rhs + self.reg_eps * torch.eye(*rhs.shape, device=self.device)))

        if self.unit_test:
            _W = self._W.detach().cpu().numpy()
            _H = self._H.to_dense().detach().cpu().numpy()
            _U = [self._U[k].detach().cpu().numpy() for k in range(self.__n_languages)]
            _X = [X[k].to_dense().detach().cpu().numpy() for k in range(self.__n_languages)]
            _S = [self._S[k].to_dense().detach().cpu().numpy() for k in range(self.__n_languages)]

            _lhs = np.sum([_U[k].T.dot(_X[k]).dot(_W).dot(_S[k]) for k in range(self.__n_languages)], axis=0)

            _rhs = np.sum([_S[k].dot(_W.T).dot(_W).dot(_S[k]) for k in range(self.__n_languages)], axis=0)
            H_np = _lhs.dot(np.linalg.inv(_rhs))

            eq_test = np.allclose(_H, H_np, rtol=1e-4, atol=1.e-4)

            assert eq_test, '_H error'

    def __partial_fit_W(self, X):

        lhs = torch.stack([dsmm(ddmm(sdmm(X[k].t(), self._U[k]), self._H), self._S[k])
                           for k in range(self.__n_languages)]).sum(0)

        rhs = torch.stack([dsmm(sdmm(self._S[k], ddmm(self._H.t(), self._H)), self._S[k])
                           for k in range(self.__n_languages)]).sum(0)

        self._W = ddmm(lhs, torch.linalg.inv(rhs + self.reg_eps * torch.eye(*rhs.shape, device=self.device)))

        if self.unit_test:
            _W = self._W.detach().cpu().numpy()
            _H = self._H.to_dense().detach().cpu().numpy()
            _S = [self._S[k].to_dense().detach().cpu().numpy() for k in range(self.__n_languages)]
            _X = [X[k].to_dense().detach().cpu().numpy() for k in range(self.__n_languages)]
            _U = [self._U[k].detach().cpu().numpy() for k in range(self.__n_languages)]

            _lhs = np.sum([_X[k].T.dot(_U[k]).dot(_H).dot(_S[k])
                           for k in range(self.__n_languages)], axis=0)
            _rhs = np.sum([_S[k].dot(_H.T).dot(_H).dot(_S[k]) for k in range(self.__n_languages)], axis=0)
            W_np = _lhs.dot(np.linalg.inv(_rhs))

            eq_test = np.allclose(_W, W_np, rtol=1e-4, atol=1.e-4)
            assert eq_test, '_W error'

    def __partial_fit_S(self, X):

        indices = torch.stack([torch.LongTensor([i, i]).to(device=self.device)
                               for i in torch.arange(self.rank)], dim=1)

        for k in range(self.__n_languages):
            values = torch.mv(
                torch.linalg.inv(torch.mul(ddmm(self._W.T, self._W),
                                           ddmm(self._H.T, self._H)) + self.reg_eps * torch.eye(self.rank,
                                                                                                device=self.device)),
                torch.diag(ddmm(dsmm(ddmm(self._H.T, self._U[k].T), X[k]), self._W)))

            self._S[k] = torch.sparse_coo_tensor(indices, values, torch.Size((self.rank, self.rank))).coalesce()

            if self.unit_test:
                _X = X[k].to_dense().detach().cpu().numpy()
                _S_k = self._S[k].to_dense().detach().cpu().numpy()
                _U_k = self._U[k].detach().cpu().numpy()
                _U = self._U[k].detach().cpu().numpy()
                _W = self._W.detach().cpu().numpy()
                _H = self._H.to_dense().detach().cpu().numpy()

                _S = diags(np.dot(np.linalg.inv(_W.T.dot(_W) * _H.T.dot(_H)),
                                  np.diag(_H.T.dot(_U.T).dot(_X).dot(_W))))

                eq_test = np.allclose(_S_k, _S.toarray(), rtol=1e-4, atol=1.e-4)

                assert eq_test, 'S_k error'

    def __rec_tensor(self, U, H, S, W):

        X_rec = [U[k] @ H @ (S[k] @ W.T) for k in range(self.__n_languages)]

        return X_rec

    def __check_convergence(self, error_old, error):

        check = abs(error_old - error) < self.error_tol * error_old

        return not check

    def __fit_error(self, X):

        rec_time = Timer()
        rec_time.start()

        loss = []
        if self.approx_fit_error:

            data_range = list(range(self.approx_fit_error))
            np.random.shuffle(data_range)
            sample_range = data_range[:self.approx_fit_error]

            for k in range(self.__n_languages):
                UHS = dsmm(self._U[k], dsewmm(self._H, self._S[k]))
                _error_nmr = torch.tensor(0, dtype=torch.float32, device=self.device)
                _error_dmr = torch.tensor(0, dtype=torch.float32, device=self.device)
                for n_col_ind in tqdm(sample_range, desc='lang {} fit_error'.format(k), disable=not self.verbose):
                    rec_tensor_n = torch.matmul(UHS, self._W[n_col_ind])
                    if X[k].is_sparse:
                        row_inds = X[k].indices()[0, :]
                        col_inds = X[k].indices()[1, :]
                        # Here we have the row and column indices of the n-th column
                        n_col_row_inds = row_inds[col_inds == n_col_ind]
                        # _values = X[k].values()[_n_row_inds]
                        values = X[k].values()[col_inds == n_col_ind]

                        X_rows = X[k].shape[0]
                        Xkn = torch.zeros([X_rows]).to(device=self.device)
                        Xkn[n_col_row_inds] = values
                    else:
                        Xkn = X[k][:, n_col_ind]
                    _error_nmr += torch.norm(Xkn - rec_tensor_n, p=2) ** 2
                    _error_dmr += torch.norm(rec_tensor_n, p=2) ** 2 + self.eps
                loss.append(torch.sqrt(_error_nmr / _error_dmr))
        else:
            for k in range(self.__n_languages):
                rec_tensor = ddmm(dsmm(self._U[k], self._H * self._S[k]), self._W.T).to_sparse()
                _error = torch.norm(X[k] - rec_tensor, p='fro') / torch.norm(rec_tensor, p='fro') + self.eps
                loss.append(_error)

        plt_mfcc_mel = False
        if plt_mfcc_mel:
            import librosa
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=2, sharex=True)
            S_dB = librosa.power_to_db(X[k].to_dense().detach().cpu().numpy(), ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time',
                                           y_axis='mel', sr=22050,
                                           fmax=8000, ax=ax[0])
            fig.colorbar(img, ax=ax[0], format='%+2.0f dB')
            ax[0].set(title='Mel-frequency spectrogram')
            S_dB = librosa.power_to_db(rec_tensor.to_dense().detach().cpu().numpy(), ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time',
                                           y_axis='mel', sr=22050,
                                           fmax=8000, ax=ax[1])
            fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
            ax[1].set(title='Mel-frequency spectrogram')
            plt.show()

        mean_loss = torch.sum(torch.FloatTensor(loss))

        rec_time.stop(tag='fit_error', verbose=self.verbose)

        return mean_loss, loss

    def partial_fit(self, X):

        self.__n_languages, self.__n_samples = len(X), X[0].shape[1]
        self._n_terms = [X_l.shape[0] for X_l in X]

        self._check_init(X)

        self.__partial_fit_U(X=X)
        self.__partial_fit_W(X=X)
        self.__partial_fit_H(X=X)
        self.__partial_fit_S(X=X)

        mean_loss, loss = self.__fit_error(X=X)

        return mean_loss, loss

    def fit(self, X):
        self.__n_languages, self.__n_samples = len(X), X[0].shape[1]
        self._n_terms = [X_l.shape[0] for X_l in X]

        m_iter, error_old, error = 0, 0, -1
        fit_time = Timer()
        fit_time.start()

        while self.__check_convergence(error_old=error_old, error=error) and m_iter < self.max_m_iter:
            error_old = error

            self.__partial_fit_U(X=X)
            self.__partial_fit_H(X=X)
            self.__partial_fit_W(X=X)
            self.__partial_fit_S(X=X)

            mean_loss, loss = self.__fit_error(X=X)

            if self.tf_writer:
                self.tf_writer.add_scalars('Error', {'error': mean_loss.item()}, m_iter)
                for k in range(self.__n_languages):
                    self.tf_writer.add_scalars('Error_X{}'.format(k), {'error': loss[k]}, m_iter)

            m_iter += 1
            if m_iter % 1 == 0:
                print('m_iter {} - model error {}'.format(m_iter, mean_loss))

        fit_time.stop(verbose=True)

        return self

import numpy as np
from tqdm import tqdm
from numpy.random import uniform
from Code.Utils.timer import Timer
from tensorboardX import SummaryWriter

from torch.sparse import mm as sdmm
from torch import mm as ddmm
import torch

from torch import Tensor
from scipy.sparse import issparse

from scipy.sparse import coo_matrix
from sklearn.decomposition._nmf import _initialize_nmf


def dsmm(mat1: Tensor, mat2: Tensor) -> Tensor:
    return sdmm(mat2.t(), mat1.t()).t()


def dsewmm(d, s):
    if not s.is_coalesced():
        s = s.coalesce()
    i = s.indices()
    v = s.values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix

    return torch.sparse_coo_tensor(i, v * dv, s.size()).coalesce()

    # return torch.sparse.FloatTensor(i, v * dv, s.size()).coalesce()


# INIT_METHOD_U = 'uniform'
# INIT_METHOD_W = 'uniform'

torch.set_printoptions(precision=10)


def sparse_eye(n, device):
    indices = torch.stack([torch.LongTensor([i, i]).to(device=device) for i in torch.arange(n)], dim=1)
    values = torch.cat([torch.FloatTensor([1]).to(device=device) for _ in torch.arange(n)])

    return torch.sparse_coo_tensor(indices, values, torch.Size((n, n))).coalesce()

    # return torch.sparse.FloatTensor(indices, values, torch.Size((n, n))).coalesce()


class nnParafac2:

    def __init__(self, rank, error_tol=1e-8, device='cpu', approx_fit_error=None,
                 max_m_iter=100,
                 log_folder=None,
                 u_init='random',
                 w_init='random',
                 reg_term_u=0,
                 reg_term_s=0,
                 reg_term_h=0,
                 reg_term_w=0,
                 verbose=False,
                 seed=0):
        self.rank = rank
        self.seed = seed
        self.device = device
        self.u_init = u_init
        self.w_init = w_init
        self.max_m_iter = max_m_iter
        self.error_tol = error_tol
        self.verbose = verbose
        self.approx_fit_error = approx_fit_error

        if log_folder:
            self.tf_writer = SummaryWriter(log_folder)

        self.__n_samples, self._n_languages, self._n_terms = 0, 0, []
        self._U, self._S, self._H, self._W = [], [], None, None

        self._n_languages = None
        self.unit_test = False
        self.reg_term_u = reg_term_u
        self.reg_term_s = reg_term_s
        self.reg_term_h = reg_term_h
        self.reg_term_w = reg_term_w
        self.rng = np.random.default_rng(seed)
        self.g_gpu = torch.Generator(device=self.device)
        self.g_gpu.manual_seed(seed)

        self.dtype = torch.float32

        self.eps = torch.finfo(self.dtype).eps

    def _check_init(self, X):
        # for k in range(self._n_languages):
        #     check_non_negative(X[k], "nnParafac2 initialization")

        if not self._U and not self._S and not self._H and not self._W:
            self.__initialize_decomposition(X)

    def get_U(self):
        return [self._U[k].clone().detach().cpu() for k in range(self._n_languages)]

    def get_H(self):
        return self._H.clone()

    def get_S(self):
        return [self._S[k].clone() for k in range(self._n_languages)]

    def get_W(self):
        return self._W.clone()

    def _non_negative_svd(self, X, q, u_init, niter=4):
        """
        References
        ----------
        C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for non-negative matrix factorization
        - Pattern Recognition, 2008
        http://tinyurl.com/nndsvd
        """
        # eps = torch.finfo(X.dtype).eps
        # check_non_negative(X, "NMF initialization")
        # Random initialization
        if u_init == "random":
            if X.is_sparse:
                avg = torch.sqrt(X.values().mean() / self.rank)
            else:
                avg = torch.sqrt(torch.mean(X) / self.rank)

            H = avg * torch.empty((self.rank, X.shape[0]),
                                  device=self.device).normal_(mean=0, std=1, generator=self.g_gpu)
            W = avg * torch.empty((X.shape[0], self.rank),
                                  device=self.device).normal_(mean=0, std=1, generator=self.g_gpu)
            torch.abs(H, out=H)
            torch.abs(W, out=W)
            return W, H, None
        else:

            # # NNDSVD initialization
            # _X = coo_matrix((X.values().detach().cpu().numpy(), X.indices().detach().cpu().numpy()), shape=X.shape)
            # U, S, Vt = randomized_svd(_X, self.rank, random_state=10)
            # U = torch.from_numpy(U).float().to(self.device)
            # S = torch.from_numpy(S).float().to(self.device)
            # Vt = torch.from_numpy(Vt).float().to(self.device)

            # NNDSVD initialization
            low_rank_matrices = torch.svd_lowrank(X, q=q, niter=niter)  # returns U, S, V
            U, S, Vt = low_rank_matrices[0], low_rank_matrices[1], low_rank_matrices[2].T

            # U, S, Vt = torch.linalg.svd(X.to_dense())
            # U, S, Vt = U[:, :self.rank], S[:self.rank], Vt[:self.rank, :]

            W = torch.zeros_like(U)
            H = torch.zeros_like(Vt)

            # The leading singular triplet is non-negative so it can be used as is for initialization.
            W[:, 0] = torch.sqrt(S[0]) * torch.abs(U[:, 0])
            H[0, :] = torch.sqrt(S[0]) * torch.abs(Vt[0, :])

            for j in range(1, self.rank):
                x, y = U[:, j], Vt[j, :]

                # extract positive and negative parts of column vectors
                x_p, y_p = torch.max(x, torch.zeros_like(x)), torch.max(y, torch.zeros_like(y))
                x_n, y_n = torch.abs(torch.min(x, torch.zeros_like(x))), torch.abs(torch.min(y, torch.zeros_like(y)))

                # and their norms
                x_p_nrm, y_p_nrm = torch.norm(x_p), torch.norm(y_p)
                x_n_nrm, y_n_nrm = torch.norm(x_n), torch.norm(y_n)

                m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

                # choose update
                if m_p > m_n:
                    u = x_p / x_p_nrm
                    v = y_p / y_p_nrm
                    sigma = m_p
                else:
                    u = x_n / x_n_nrm
                    v = y_n / y_n_nrm
                    sigma = m_n

                lbd = torch.sqrt(S[j] * sigma)
                W[:, j] = lbd * u
                H[j, :] = lbd * v

            W[W < self.eps] = 0
            H[H < self.eps] = 0

            if u_init == "nndsvd":
                return W, H, S
            elif u_init == "nndsvda":
                if X.is_sparse:
                    avg = torch.mean(X.values())
                else:
                    avg = torch.mean(X)

                W[W == 0] = avg
                H[H == 0] = avg
            elif u_init == "nndsvdar":
                if X.is_sparse:
                    avg = torch.mean(X.values())
                else:
                    avg = torch.mean(X)

                W[W == 0] = abs(avg * torch.empty(len(W[W == 0]),
                                                  device=self.device).normal_(mean=0,
                                                                              std=1, generator=self.g_gpu) / 100)
                H[H == 0] = abs(avg * torch.empty(len(H[H == 0]),
                                                  device=self.device).normal_(mean=0,
                                                                              std=1, generator=self.g_gpu) / 100)
            else:
                raise ValueError(
                    'Invalid u_init parameter: got %r instead of one of %r' %
                    (u_init, ('nndsvd', 'nndsvda', 'nndsvdar')))

            return W, H, S

    # def _initialise_wh(self, init_method):
    #     """
    #     Initialise basis and coefficient matrices according to `init_method`
    #     https://github.com/AlexandrovLab/SigProfilerExtractor/blob/master/SigProfilerExtractor/nmf_gpu.py#L69
    #     """
    #     if init_method == 'random':
    #         W = torch.from_numpy(
    #             self.rng.random((self._V.shape[0], self._V.shape[1], self._rank), dtype=np.float64)).cuda()
    #         H = torch.from_numpy(
    #             self.rng.random((self._V.shape[0], self._rank, self._V.shape[2]), dtype=np.float64)).cuda()
    #         if self.rng is np.float32:
    #             W = W.float()
    #             H = H.float()
    #         return W, H
    #
    #     elif init_method == 'nndsvd':
    #         W = np.zeros([self._V.shape[0], self._V.shape[1], self._rank])
    #         H = np.zeros([self._V.shape[0], self._rank, self._V.shape[2]])
    #         nv = nndsvd.Nndsvd()
    #         for i in range(self._V.shape[0]):
    #             vin = np.mat(self._V.cpu().numpy()[i])
    #             W[i, :, :], H[i, :, :] = nv.initialize(vin, self._rank, options={'flag': 0})
    #
    #     elif init_method == 'nndsvda':
    #         W = np.zeros([self._V.shape[0], self._V.shape[1], self._rank])
    #         H = np.zeros([self._V.shape[0], self._rank, self._V.shape[2]])
    #         nv = nndsvd.Nndsvd()
    #         for i in range(self._V.shape[0]):
    #             vin = np.mat(self._V.cpu().numpy()[i])
    #             W[i, :, :], H[i, :, :] = nv.initialize(vin, self._rank, options={'flag': 1})
    #
    #     elif init_method == 'nndsvdar':
    #         W = np.zeros([self._V.shape[0], self._V.shape[1], self._rank])
    #         H = np.zeros([self._V.shape[0], self._rank, self._V.shape[2]])
    #         nv = nndsvd.Nndsvd()
    #         for i in range(self._V.shape[0]):
    #             vin = np.mat(self._V.cpu().numpy()[i])
    #             W[i, :, :], H[i, :, :] = nv.initialize(vin, self._rank, options={'flag': 2})
    #     elif init_method == 'nndsvd_min':
    #         W = np.zeros([self._V.shape[0], self._V.shape[1], self._rank])
    #         H = np.zeros([self._V.shape[0], self._rank, self._V.shape[2]])
    #         nv = nndsvd.Nndsvd()
    #         for i in range(self._V.shape[0]):
    #             vin = np.mat(self._V.cpu().numpy()[i])
    #             w, h = nv.initialize(vin, self._rank, options={'flag': 2})
    #             min_X = np.min(vin[vin > 0])
    #             h[h <= min_X] = min_X
    #             w[w <= min_X] = min_X
    #             # W= np.expand_dims(W, axis=0)
    #             # H = np.expand_dims(H, axis=0)
    #             W[i, :, :] = w
    #             H[i, :, :] = h
    #     # W,H=initialize_nm(vin, nfactors, init=init, eps=1e-6,random_state=None)
    #     W = torch.from_numpy(W).type(self._tensor_type).cuda(self._gpu_id)
    #     H = torch.from_numpy(H).type(self._tensor_type).cuda(self._gpu_id)
    #     return W, H

    def __initialize_decomposition(self, X):
        if self.verbose:
            print('___Initialize nnPARAFAC2___')
        self.__init_U(X)  # projection matrices
        self.__init_H()  # factors
        self.__init_W(X)  # factors
        self.__init_S()  # factors

    def __init_U(self, X):
        init_time = Timer()
        init_time.start()

        if not self.u_init or self.u_init not in ("random", "nndsvd", "nndsvda", "nndsvdar", "uniform"):
            raise ValueError(
                "Invalid init parameter: got %r instead of one of %r"
                % (self.u_init, (None, "random", "nndsvd", "nndsvda", "nndsvdar", "uniform"))
            )

        for k in range(self._n_languages):
            if self.verbose:
                print('Initialize U[{}] (dim: {} x {})...'.format(k, self._n_terms[k], self.rank))

            # self._U.append(uniform(0, 0.1, size=(self._n_terms[k], self.rank)))

            # if self.u_init == 'random':
            #     U, _ = _initialize_nmf(coo_matrix((X[k].values().detach().cpu().numpy(),
            #                                        X[k].indices().detach().cpu().numpy()),
            #                                       shape=X[k].shape), self.rank, init=self.u_init)
            #     U = torch.from_numpy(U).float().to(self.device)
            # else:
            if self.u_init == "uniform":
                # np.random.seed(self.seed)
                U = torch.tensor(self.rng.uniform(0, 0.01, size=(self._n_terms[k], self.rank)),
                                 dtype=self.dtype, device=self.device)
            else:
                U, _, _ = self._non_negative_svd(X=X[k], q=self.rank, u_init=self.u_init)

            # np.random.seed(0)
            # U = torch.tensor(uniform(0, 0.01, size=(self._n_terms[k], self.rank)),
            #                  dtype=torch.float32, device=self.device)

            # if self.u_init == 'random':
            #     if INIT_METHOD_U == 'gaussian':
            #         avg = torch.sqrt(torch.mean(X[k].values()) / self.rank)
            #         rng = torch.Generator(device=self.device)
            #         rngs = rng.manual_seed(self.seed)
            #         U = avg * torch.randn(size=(self._n_terms[k], self.rank),
            #                               dtype=X[k].dtype, generator=rngs, device=self.device)
            #         torch.abs(U, out=U)
            #     else:
            #         U = torch.tensor(uniform(0, 1, size=(self._n_terms[k], self.rank)),
            #                          dtype=torch.float32, device=self.device)
            #
            #         # U, _ = torch.linalg.qr(U)
            #
            #         U, _, _ = torch.linalg.svd(U, full_matrices=False)
            #
            #         U = torch.clip(U, 0, float("inf"))
            # else:
            #     U, _ = _initialize_nmf(coo_matrix((X[k].values().detach().cpu().numpy(),
            #                                        X[k].indices().detach().cpu().numpy()),
            #                                       shape=X[k].shape), self.rank, init=self.u_init)
            #     U = torch.from_numpy(U).float().to(self.device)
            #     # U = self._non_negative_svd(X=X[k])
            #
            #     # I = len(matrices)
            #     # A = tl.ones((I, rank))
            #     # B_is = [svd_fun(matrix, n_eigenvecs=rank)[0] for matrix in matrices]
            #     # C = tl.transpose(svd_fun(tl.concatenate(matrices, 0), n_eigenvecs=rank)[2])
            #     # if init == "svd":
            #     #     return CoupledMatrixFactorization((None, [A, B_is, C]))
            #     #
            #     # A = tl.ones((I, rank))
            #     # B_is = [tl.clip(B_i, 0, float("inf")) for B_i in B_is]
            #     # C = tl.clip(C, 0, float("inf"))

            self._U.append(U)
        init_time.stop(tag=f'___Initialize U ({self.u_init})___', verbose=self.verbose)

    def __init_H(self):
        if self.verbose:
            print('Initialize H (dim: {} x {})...'.format(self.rank, self.rank))
        init_time = Timer()
        init_time.start()
        self._H = sparse_eye(self.rank, self.device)
        init_time.stop(tag='___Initialize H___', verbose=self.verbose)

    def __init_W(self, X):
        if self.verbose:
            print('Initialize W {} (dim: {} x {})...'.format(self.w_init, self.__n_samples, self.rank))
        init_time = Timer()
        init_time.start()

        # [torch.sparse.mm(X[k].t(), X[k]) for k in range(2)]

        # min_n_terms = min(self._n_terms)
        #
        # _X = []
        # for k in range(self._n_languages):
        #     # _x, _ = _initialize_nmf(coo_matrix((X[k].values().detach().cpu().numpy(),
        #     #                                     X[k].indices().detach().cpu().numpy()),
        #     #                                    shape=X[k].shape).T, min_n_terms, init=self.u_init)
        #
        #     _x, _ = self._non_negative_svd(X=X[k].t().coalesce(), q=self.rank)
        #
        #     # _x = torch.from_numpy(_x).float().to(self.device)
        #     _X.append(_x)
        #
        # self._W = torch.mean(torch.staMallonck(_X, dim=0), dim=0)

        # _w, _ = _initialize_nmf(_X.T, self.rank, init=self.u_init)
        # _w, _ = self._non_negative_svd(X=_X.T, q=self.rank)
        #
        # self._W = torch.from_numpy(_X @ _w).float().to(self.device)

        # a = torch.lobpcg(torch.sparse.mm(X[0].t(), X[0]), k=self.rank)[1]
        # L, Q = torch.linalg.eigh(torch.sparse.mm(X[0].t(), X[0]).to_dense())
        #
        # b = torch.lobpcg(torch.sparse.mm(X[0], X[0].t()), k=self.rank)[1]
        #
        # L1, Q1 = torch.linalg.eigh(torch.sparse.mm(X[0], X[0].t()).to_dense())
        #
        # C = torch.sparse.mm(X[0].T, Q1[:, :100]) == Q[:, :100]
        #
        # c = torch.sparse.mm(X[0].T, b) == a

        # # Steps
        if self.w_init == "random":
            W = []
            for k in range(self._n_languages):

                if X[k].is_sparse:
                    avg = torch.sqrt(X[k].values().mean() / self.rank)
                else:
                    avg = torch.sqrt(torch.mean(X[k]) / self.rank)

                w = avg * torch.empty((self.__n_samples, self.rank),
                                      device=self.device).normal_(mean=0, std=1, generator=self.g_gpu)
                torch.abs(w, out=w)

                W.append(w)
            self._W = torch.stack(W).mean(0)
        elif self.w_init == "uniform":
            # np.random.seed(self.seed)
            self._W = torch.tensor(self.rng.uniform(0, 0.001, size=(self.__n_samples, self.rank)),
                                   dtype=self.dtype, device=self.device)
        else:
            _, Vt1, S1 = self._non_negative_svd(X=X[1], q=self.rank, u_init=self.w_init, niter=4)
            hatV1 = Vt1.t() * (S1 + 1e-8).pow_(-1)
            _, hatV0, _ = self._non_negative_svd(X=sdmm(X[0], hatV1), q=self.rank, u_init=self.w_init)
            self._W = ddmm(hatV1, hatV0)

        init_time.stop(tag=f'___Initialize W ({self.w_init})___', verbose=self.verbose)

    def __init_S(self):
        init_time = Timer()
        init_time.start()
        for k in range(self._n_languages):
            if self.verbose:
                print(f'Initialize S[{k}] (dim: {self.rank} x {self.rank})...')
            self._S.append(sparse_eye(self.rank, self.device).coalesce())
        init_time.stop(tag='___Initialize S___', verbose=self.verbose)

    def __partial_fit_U(self, X):

        for k in range(self._n_languages):
            nrt = dsmm(
                dsmm(
                    sdmm(X[k], self._W) if X[k].is_sparse else ddmm(X[k], self._W),
                    self._S[k]), self._H.t())
            dnr = ddmm(self._U[k],
                       dsmm(
                           dsmm(
                               ddmm(
                                   dsmm(self._U[k].T, X[k]) if X[k].is_sparse else ddmm(self._U[k].T, X[k]),
                                   self._W),
                               self._S[k]
                           ),
                           self._H.t()
                       ) + self.reg_term_u * torch.eye(self.rank, device=self.device))

            if self.unit_test:
                _X_k = X[k].to_dense().detach().cpu().numpy()
                _S_k = self._S[k].to_dense().detach().cpu().numpy()
                _W = self._W.detach().cpu().numpy()
                _H = self._H.to_dense().detach().cpu().numpy()
                _U_k = self._U[k].detach().cpu().numpy()
                _nrt = _X_k.dot(_W).dot(_S_k).dot(_H.T)
                _dnr = _U_k.dot(_U_k.T).dot(_X_k).dot(_W).dot(_S_k).dot(_H.T)

                eq_test = np.allclose(nrt.detach().cpu().numpy(), _nrt) \
                          and np.allclose(dnr.detach().cpu().numpy(), _dnr)

                assert eq_test, '_U error'

            # dnr[dnr < self.eps] = self.eps
            self._U[k] = torch.mul(self._U[k], nrt / (dnr + 1e-8))

    def __partial_fit_H(self, X):

        nrt = torch.diag(torch.stack([sdmm(self._S[k], ddmm(self._W.t(), sdmm(X[k].t(), self._U[k]))) for k in
                                      range(self._n_languages)]).sum(0))

        dnr = (
                      torch.diag(
                          torch.stack(
                              [sdmm(self._S[k], dsmm(ddmm(self._W.t(), self._W), self._S[k])) for k in
                               range(self._n_languages)]
                          ).sum(0)
                      ) + self.reg_term_h * torch.ones(self.rank, device=self.device)
              ) * self._H.values()

        if self.unit_test:
            _W = self._W.detach().cpu().numpy()
            _H = self._H.to_dense().detach().cpu().numpy()

            _nrt = np.diag(np.sum([self._S[k].to_dense().detach().cpu().numpy().dot(self._W.t().detach().cpu().numpy())
                                  .dot(X[k].to_dense().detach().cpu().numpy().T)
                                  .dot(self._U[k].detach().cpu().numpy()) for k in
                                   range(self._n_languages)], axis=0))

            _dnr = np.diag(np.sum([self._S[k].to_dense().detach().cpu().numpy().dot(_W.T).dot(_W)
                                  .dot(self._S[k].to_dense().detach().cpu().numpy()) for k in
                                   range(self._n_languages)], axis=0)) * np.diag(_H) + self.eps.item()

            eq_test = np.allclose(nrt.detach().cpu().numpy(), _nrt, rtol=1e-4, atol=1.e-4) \
                      and np.allclose(dnr.detach().cpu().numpy(), _dnr, rtol=1e-4, atol=1.e-4)

            assert eq_test, '_H error'

        indices = torch.stack([torch.LongTensor([i, i]).to(device=self.device)
                               for i in torch.arange(self.rank)], dim=1)

        # dnr[dnr < self.eps] = self.eps

        nrt_div_dnr = torch.sparse_coo_tensor(indices, nrt / (dnr + 1e-8),
                                               torch.Size((self.rank, self.rank))).coalesce()
        # nrt_div_dnr = torch.sparse.FloatTensor(indices, nrt / (dnr + 1e-8),
        #                                        torch.Size((self.rank, self.rank))).coalesce()

        self._H = torch.mul(self._H, nrt_div_dnr)

    def __partial_fit_W(self, X):

        nrt = torch.stack([dsmm(dsmm(
            sdmm(X[k].t(), self._U[k]) if X[k].is_sparse else ddmm(X[k].T, self._U[k]),
            self._H), self._S[k])
            for k in range(2)]).sum(0)

        dnr = dsmm(
            self._W, torch.sparse.sum(
                torch.stack(
                    [self._S[k] * self._H.t() * self._H * self._S[k]
                     + 0.5 * self.reg_term_w * sparse_eye(self.rank, self.device)
                     for k in range(self._n_languages)]
                ),
                dim=0
            )
        )

        if self.unit_test:
            _W = self._W.detach().cpu().numpy()
            _H = self._H.to_dense().detach().cpu().numpy()

            _nrt = np.sum([X[k].to_dense().detach().cpu().numpy().T.dot(self._U[k].detach().cpu().numpy()).dot(_H).
                          dot(self._S[k].to_dense().detach().cpu().numpy())
                           for k in range(self._n_languages)], axis=0)

            _dnr = _W.dot(np.sum([self._S[k].to_dense().detach().cpu().numpy().dot(_H.T).dot(_H)
                                 .dot(self._S[k].to_dense().detach().cpu().numpy()) for k in
                                  range(self._n_languages)], axis=0)) + self.eps.item()

            eq_test = np.allclose(nrt.detach().cpu().numpy(), _nrt, rtol=1e-4, atol=1.e-4) \
                      and np.allclose(dnr.detach().cpu().numpy(), _dnr, rtol=1e-4, atol=1.e-4)
            assert eq_test, '_W error'

            # eq_test = (torch.dov(nrt, dnr).detach().cpu().numpy() - np.divide(_nrt, _dnr) < 1e-5).all()
            # assert eq_test, '_W error'

        # dnr[dnr < self.eps] = self.eps
        self._W = torch.mul(self._W, nrt / (dnr + 1e-8))

    def __partial_fit_S(self, X):

        for k in range(self._n_languages):
            nrt = torch.diag(
                ddmm(
                    dsmm(
                        sdmm(self._H.t(), self._U[k].T), X[k]
                    ) if X[k].is_sparse else
                    ddmm(
                        sdmm(self._H.t(), self._U[k].T), X[k]
                    ),
                    self._W)
            )
            dnr = (
                    (dsewmm(
                        ddmm(self._W.T, self._W),
                        self._H.t() * self._H) + self.reg_term_s * sparse_eye(self.rank, self.device)
                     ) * self._S[k]
            ).coalesce().values()

            indices = self._S[k].indices().to(torch.int64)
            shape = self._S[k].shape

            # dnr[dnr < self.eps] = self.eps

            nrt_div_dnr = torch.sparse_coo_tensor(indices, nrt / (dnr + 1e-8), torch.Size(shape)).coalesce()

            # nrt_div_dnr = torch.sparse.FloatTensor(indices, nrt / (dnr + 1e-8), torch.Size(shape)).coalesce()

            if self.unit_test:
                _X_k = X[k].to_dense().detach().cpu().numpy()
                _S_k = self._S[k].to_dense().detach().cpu().numpy()
                _U_k = self._U[k].detach().cpu().numpy()

                _W = self._W.detach().cpu().numpy()
                _H = self._H.to_dense().detach().cpu().numpy()

                _nrt = np.diag(_H.T.dot(_U_k.T).dot(_X_k).dot(_W))
                _dnr = ((_W.T.dot(_W)) * (_H.T.dot(_H))).dot(_S_k.diagonal()) + 1e-8

                eq_test = np.allclose(nrt.detach().cpu().numpy(), _nrt, rtol=1e-4, atol=1.e-4) \
                          and np.allclose(dnr.detach().cpu().numpy(), _dnr, rtol=1e-4, atol=1.e-4)

                assert eq_test, 'S_k error'

            self._S[k] = torch.mul(self._S[k], nrt_div_dnr)

    def __rec_tensor(self, U, H, S, W):
        X_rec = [U[k] @ H @ (S[k] @ W.T) for k in range(self._n_languages)]
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

            for k in range(self._n_languages):
                UHS = dsmm(self._U[k], (self._H * self._S[k]))
                _error_nmr = torch.tensor(0, dtype=self.dtype, device=self.device)
                _error_dmr = torch.tensor(0, dtype=self.dtype, device=self.device)
                for n_col_ind in tqdm(sample_range, desc='lang {} fit_error'.format(k), disable=not self.verbose):
                    # W_col = torch.index_select(self._W, dim=0,
                    #                            index=torch.tensor([n_col_ind],
                    #                                               dtype=torch.long, device=self.device))
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
            for k in range(self._n_languages):
                rec_tensor = ddmm(dsmm(self._U[k], self._H * self._S[k]), self._W.T).to_sparse()
                _error = torch.norm(X[k] - rec_tensor, p='fro') / (torch.norm(rec_tensor, p='fro') + self.eps)
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

        mean_loss = torch.mean(torch.FloatTensor(loss))

        rec_time.stop(tag='fit_error', verbose=self.verbose)

        return mean_loss, loss

    def partial_fit(self, X):

        if self._n_languages is None:
            self._n_languages, self.__n_samples = len(X), X[0].shape[1]
            self._n_terms = [X_l.shape[0] for X_l in X]

        self._check_init(X)

        self.__partial_fit_W(X=X)
        self.__partial_fit_H(X=X)
        self.__partial_fit_S(X=X)
        self.__partial_fit_U(X=X)

        mean_loss, loss = self.__fit_error(X=X)

        return mean_loss, loss

    def fit(self, X):
        self._n_languages, self.__n_samples = len(X), X[0].shape[1]
        self._n_terms = [X_l.shape[0] for X_l in X]

        m_iter, error_old, error = 0, 0, -1
        fit_time = Timer()
        fit_time.start()

        while self.__check_convergence(error_old=error_old, error=error) and m_iter < self.max_m_iter:
            error_old = error

            self.__partial_fit_H(X=X)
            self.__partial_fit_W(X=X)
            self.__partial_fit_S(X=X)
            self.__partial_fit_U(X=X)

            mean_loss, loss = self.__fit_error(X=X)

            if self.tf_writer:
                self.tf_writer.add_scalars('Error', {'error': mean_loss.item()}, m_iter)
                for k in range(self._n_languages):
                    self.tf_writer.add_scalars('Error_X{}'.format(k), {'error': loss[k]}, m_iter)

            m_iter += 1
            # if m_iter % 1 == 0:
            print('m_iter {} - model error {}'.format(m_iter, mean_loss))

        fit_time.stop(verbose=True)

        return self

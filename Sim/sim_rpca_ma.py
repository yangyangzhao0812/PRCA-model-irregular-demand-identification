import numpy as np
import pandas as pd

class StablePCP():
    """Stable principal component pursuit (stable version of Robust PCA)

    Dimensionality reduction using Accelerated Proximal Gradient (APG)
    to decompose the input 2D matrix M into a lower rank dense 2D matrix L and sparse
    but not low-rank 2D matrix S and a noise term Z. Here the noise matrix Z = M-L-S and
    satisfying Frobenius norm ||Z|| < detla.

    Parameters
    ----------
    lamb : positive float
        Sparse component coefficient.
        if user doesn't set it:
            lamb = 1/sqrt(max(M.shape))
        A effective default value from the reference.

    mu0 : positive float
        Coefficient for the singular value thresholding of M
        if user doesn't set it:
            mu0 = min([mu0_init*np.sqrt(2*max(M.shape)), 0.99*||M||2])
        namely, mu0 is chosen between manual value mu0_init*np.sqrt(2*max(M.shape)) and emprical value 0.99*||M||2

    mu0_init : positive float/int
        Coefficient for initial mu0

    mu_fixed : bool
        Flag for whether or not use a fixed mu for iterations

    mu_min : positive float
        minimum mu for thresholding

    sigma : positive float
        The standard deviation of the Gaussian noise N(0,sigma) for generating E

    eta : float
        Decay coefficient for thresholding, 0 < eta < 1

    max_rank : positive int
        The maximum rank allowed in the low rank matrix
        default is None --> no limit to the rank of the low
        rank matrix.

    tol : positive float
        Convergence criterion

    max_iter : positive int
        Maximum iterations for alternating updates

    use_fbpca : bool
        Determine if use fbpca for SVD. fbpca use Fast Randomized SVDself.
        default is False

    fbpca_rank_ratio : float, between (0, 1]
        If max_rank is not given, this sets the rank for fbpca.pca()
        fbpca_rank = int(fbpca_rank_ratio * min(M.shape))

    Attributes:
    -----------
    L : 2D array
            Lower rank dense 2D matrix

    S : 2D array
        Sparse but not low-rank 2D matrix

    converged : bool
        Flag shows if the fit is converged or not

    error : list
        list of errors from iterations


    References
    ----------
    Zhou, Zihan, et al. "Stable principal component pursuit."
        Information Theory Proceedings (ISIT), 2010 IEEE International Symposium on. IEEE, 2010.

    Lin, Zhouchen, et al. "Fast convex optimization algorithms for exact
    recovery of a corrupted low-rank matrix."
        Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP) 61.6 (2009).

    Wei Xiao "onlineRPCA"
        https://github.com/wxiao0421/onlineRPCA/tree/master/rpca
    """
    def __init__(self, lamb=None, mu0=None, mu0_init=1000, mu_fixed=False, mu_min=None, sigma=100, eta=0.9,
                 max_rank=None, tol=1e-6, max_iter=1000, use_fbpca=False, fbpca_rank_ratio=0.2, verbsome=False):
        self.lamb = lamb
        self.mu0 = mu0
        self.mu0_init = mu0_init
        self.mu_fixed = mu_fixed
        self.mu_min = mu_min
        self.sigma = sigma
        self.eta = eta
        self.max_rank = max_rank
        self.tol = tol
        self.max_iter = max_iter
        self.use_fbpca = use_fbpca
        self.fbpca_rank_ratio = fbpca_rank_ratio
        self.converged = None
        self.error = []

    def s_tau(self, X, tau):
        """Shrinkage operator
            Sτ [x] = sign(x) max(|x| − τ, 0)

        Parameters
        ----------
        X : 2D array
            Data for shrinking

        tau : positive float
            shrinkage threshold

        Returns
        -------
        shirnked 2D array
        """

        return np.sign(X)*np.maximum(np.abs(X)-tau,0)

    def fit(self, M):
        """Stable PCP fit.
        A Gaussian noise is assumed.

        Parameters
        ----------
        M : 2D array
            2D array for docomposing
        """

        size = M.shape

        # initialize L, S and t
        L0, L1 = np.zeros(size), np.zeros(size)
        S0, S1 = np.zeros(size), np.zeros(size)
        t0, t1 = 1, 1

        # if lamb and mu are not set, set with default values
        if self.mu_fixed:
            self.mu0 = np.sqrt(2*np.max(size))*self.sigma

        elif self.mu0==None:
            self.mu0 = np.min([self.mu0_init*np.sqrt(2*np.max(size)), 0.99*np.linalg.norm(M, 2)])
            if self.mu_min==None:
                self.mu_min = np.sqrt(2*np.max(size))*self.sigma

        mu = self.mu0 * 1

        if self.lamb==None:
            self.lamb = 1/np.sqrt(np.max(size))

        #
        for i in range(self.max_iter):
            YL = L1 + (t0-1)/t1*(L1-L0)
            YS = S1 + (t0-1)/t1*(S1-S0)

            # Thresdholding for updating L
            GL = YL - 0.5*(YL+YS-M)
            # singular value decomposition
            if self.use_fbpca:
                if self.max_rank:
                    (u, s, vh) = pca(GL, self.max_rank, True, n_iter = 5)
                else:
                    (u, s, vh) = pca(GL, int(np.min(X.shape)*self.fbpca_rank_ratio), True, n_iter = 5)
            else:
                u, s, vh = np.linalg.svd(GL, full_matrices=False)

            s = s[s>(mu/2)] - mu/2  # threshold by mu/2
            rank = len(s)

            # Max rank cut
            if self.max_rank:
                if rank > self.max_rank:
                    rank = self.max_rank*1
                    s = s[0:rank]

            # update L1, L0
            L0 = L1
            L1 = np.dot(u[:,0:rank]*s, vh[0:rank,:])

            # Thresdholding for updating S
            GS = YS - 0.5*(YL+YS-M)
            # update S0, SL
            S0 = S1
            S1 = self.s_tau(GS, self.lamb*mu/2) # threshold by lamb*mu/2

            # update t0, t1
            t0 = t1
            t1 = (1+np.sqrt(4*t1**2+1))/2

            if not self.mu_fixed:
                # update mu
                mu = np.max([self.eta*mu, self.mu_min])

            # Check Convergence
            EA = 2*(YL-L1)+(L1+S1-YL-YS)
            ES = 2*(YS-S1)+(L1+S1-YL-YS)
            Etot = np.sqrt(np.linalg.norm(EA)**2+np.linalg.norm(ES)**2)
            self.error.append(Etot)
            if Etot <= self.tol:
                break

        # Print if the fit is converged
        if Etot > self.tol:
            print('Not converged within %d iterations!'%self.max_iter)
            print('Total error: %f, allowed tolerance: %f'%(Etot, self.tol))
            self.converged = False
        else:
            print('Converged!')
            self.converged = True

        self.L, self.S, self.rank = L1, S1, rank

    def get_low_rank(self):
        '''Return the low rank matrix

        Returns:
        --------
        L : 2D array
            Lower rank dense 2D matrix
        '''
        return self.L

    def get_sparse(self):
        '''Return the sparse matrix

        Returns:
        --------
        S : 2D array
            Sparse but not low-rank 2D matrix
        '''
        return self.S

    def get_rank(self):
        '''Return the rank of low rank matrix

        Returns:
        rank : int
            The rank of low rank matrix
        '''
        return self.rank

def Synthetic_data1_generate():
    # L represents low-rank matrix; S represents sparse matrix.
    rep_time = 20
    # Gaussian standard deviation
    sigma = [5, 10, 20, 40, 60, 80, 100, 120]
    # corruption rate
    p = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    # row dimension of observed demand matrix
    dim = 100
    # rank of observed demand matrix
    rank = 7
    ##################
    MAPE_L_rpca = np.zeros((rep_time, len(sigma), len(p)))
    MAPE_S_rpca = np.zeros((rep_time, len(sigma), len(p)))
    RMSE_L_rpca = np.zeros((rep_time, len(sigma), len(p)))
    RMSE_S_rpca = np.zeros((rep_time, len(sigma), len(p)))
    MAPE_L_ma = np.zeros((rep_time, len(sigma), len(p)))
    MAPE_S_ma = np.zeros((rep_time, len(sigma), len(p)))
    RMSE_L_ma = np.zeros((rep_time, len(sigma), len(p)))
    RMSE_S_ma = np.zeros((rep_time, len(sigma), len(p)))
    ##################
    MAPE_L_ave_rpca = np.zeros((len(sigma), len(p)))
    MAPE_S_ave_rpca = np.zeros((len(sigma), len(p)))
    RMSE_L_ave_rpca = np.zeros((len(sigma), len(p)))
    RMSE_S_ave_rpca = np.zeros((len(sigma), len(p)))
    MAPE_L_ave_ma = np.zeros((len(sigma), len(p)))
    MAPE_S_ave_ma = np.zeros((len(sigma), len(p)))
    RMSE_L_ave_ma = np.zeros((len(sigma), len(p)))
    RMSE_S_ave_ma = np.zeros((len(sigma), len(p)))
    for seed in range(rep_time):
        print("seed", seed)
        for i_ord, i in enumerate(sigma):
            # Gaussian matrix generation
            Z = np.random.normal(0, i, (dim, dim))
            # low-rank matrix generation
            sub_mu = 15
            U = np.random.normal(sub_mu, 1, (dim, rank))
            V = np.random.normal(sub_mu, 1, (dim, rank))
            L = np.dot(U, V.transpose())
            for j_ord, j in enumerate(p):
                # sparse matrix generation
                S = np.random.uniform(-2000, 2000, (dim, dim))
                # select ids of elements equal to zero
                ids = np.random.choice(np.arange(S.size), replace=False, size=int(S.size * (1-j)))
                S.reshape(-1)[ids] = 0
                nonzero_ids = np.where(abs(S.reshape(-1))>2)[0]
                # nonzero_ids = [index for index in np.arange(S.size) if index not in ids]
                S = S.reshape(100, 100)
                # observed demand matrix generation
                M = L+S+Z
                # decomposition by SPCP
                APG = StablePCP(sigma=i)
                APG.fit(M)
                L_APG = APG.get_low_rank()
                S_APG = APG.get_sparse()
                RMSE_L = np.linalg.norm(L_APG-L, ord="fro")/100
                RMSE_S = np.linalg.norm(S_APG-S, ord="fro")/100
                mape_L = ((L_APG - L) / L).reshape(-1)
                MAPE_L = np.sum(abs(mape_L))/(100 * 100) * 100
                # select non-zero elements of sparse matrix S to avoid infinite MAPE value
                S_true_sel = S.reshape(-1)[nonzero_ids]
                S_APG_sel = S_APG.reshape(-1)[nonzero_ids]
                mape_s = abs((S_APG_sel - S_true_sel) / S_true_sel).reshape(-1)
                MAPE_S = np.sum(mape_s) / (len(nonzero_ids)) * 100
                # save and average MAPE and RMSE errors
                MAPE_L_rpca[seed][i_ord][j_ord] = MAPE_L
                MAPE_S_rpca[seed][i_ord][j_ord] = MAPE_S
                RMSE_L_rpca[seed][i_ord][j_ord] = RMSE_L
                RMSE_S_rpca[seed][i_ord][j_ord] = RMSE_S
                # moving average method
                M_obs = pd.DataFrame(M)
                M_ma = M_obs.rolling(window=50, axis=1).mean()
                M_ma[np.arange(49)] = M_ma[np.arange(49, 98)]
                L_ma = M_ma.to_numpy()
                S_ma = (M_obs - M_ma).to_numpy()
                RMSE_L = np.linalg.norm(L_ma - L, ord="fro") / 100
                RMSE_S = np.linalg.norm(S_ma - S, ord="fro") / 100
                mape_L = ((L_ma - L) / L).reshape(-1)
                MAPE_L = np.sum(abs(mape_L)) / (100 * 100) * 100
                S_ma_sel = S_ma.reshape(-1)[nonzero_ids]
                mape_s = abs((S_ma_sel - S_true_sel) / S_true_sel).reshape(-1)
                MAPE_S = np.sum(mape_s) / (len(nonzero_ids)) * 100
                MAPE_L_ma[seed][i_ord][j_ord] = MAPE_L
                MAPE_S_ma[seed][i_ord][j_ord] = MAPE_S
                RMSE_L_ma[seed][i_ord][j_ord] = RMSE_L
                RMSE_S_ma[seed][i_ord][j_ord] = RMSE_S
                if seed == rep_time - 1:
                    MAPE_L_ave_rpca = MAPE_L_rpca.mean(axis=0)
                    MAPE_S_ave_rpca = MAPE_S_rpca.mean(axis=0)
                    RMSE_L_ave_rpca = RMSE_L_rpca.mean(axis=0)
                    RMSE_S_ave_rpca = RMSE_S_rpca.mean(axis=0)
                    MAPE_L_ave_ma = MAPE_L_ma.mean(axis=0)
                    MAPE_S_ave_ma = MAPE_S_ma.mean(axis=0)
                    RMSE_L_ave_ma = RMSE_L_ma.mean(axis=0)
                    RMSE_S_ave_ma = RMSE_S_ma.mean(axis=0)
                    print("MAPE_L_ave_ma.shape", MAPE_L_ave_ma.shape)
                    print("MAPE_L_ave_rpca.shape", MAPE_L_ave_rpca.shape)
    error_rpca = np.concatenate((MAPE_L_ave_rpca, MAPE_S_ave_rpca, RMSE_L_ave_rpca, RMSE_S_ave_rpca), axis=0)
    error_ma = np.concatenate((MAPE_L_ave_ma, MAPE_S_ave_ma, RMSE_L_ave_ma, RMSE_S_ave_ma), axis=0)
    with open('error_rpca1.npy', 'wb') as f:
        np.save(f, error_rpca)
    with open('error_ma1.npy', 'wb') as f:
        np.save(f, error_ma)
    return None
        
def Synthetic_data2_generate():
    # L represents low-rank matrix; S represents sparse matrix.
    rep_time = 20
    # Gaussian standard deviation
    sigma = 100
    # Corruption rate
    p = 0.1
    # Row dimension of observed demand matrix
    dim = [10, 20, 40, 60, 80, 160, 320]
    # Rank of observed demand matrix
    rank = [1, 2, 3, 4, 5, 6, 7]
    MAPE_L_rpca = np.zeros((rep_time, len(dim), len(rank)))
    MAPE_S_rpca = np.zeros((rep_time, len(dim), len(rank)))
    RMSE_L_rpca = np.zeros((rep_time, len(dim), len(rank)))
    RMSE_S_rpca = np.zeros((rep_time, len(dim), len(rank)))
    MAPE_L_ma = np.zeros((rep_time, len(dim), len(rank)))
    MAPE_S_ma = np.zeros((rep_time, len(dim), len(rank)))
    RMSE_L_ma = np.zeros((rep_time, len(dim), len(rank)))
    RMSE_S_ma = np.zeros((rep_time, len(dim), len(rank)))
    ##################
    MAPE_L_ave_rpca = np.zeros((len(dim), len(rank)))
    MAPE_S_ave_rpca = np.zeros((len(dim), len(rank)))
    RMSE_L_ave_rpca = np.zeros((len(dim), len(rank)))
    RMSE_S_ave_rpca = np.zeros((len(dim), len(rank)))
    MAPE_L_ave_ma = np.zeros((len(dim), len(rank)))
    MAPE_S_ave_ma = np.zeros((len(dim), len(rank)))
    RMSE_L_ave_ma = np.zeros((len(dim), len(rank)))
    RMSE_S_ave_ma = np.zeros((len(dim), len(rank)))
    for seed in range(rep_time):
        print("seed", seed)
        for i_ord, i in enumerate(dim):
            for j_ord, j in enumerate(rank):
                # Gaussian matrix generation
                Z = np.random.normal(0, sigma, (i, i))
                # low-rank matrix generation
                sub_mu = 15
                U = np.random.normal(sub_mu, 1, (i, j))
                V = np.random.normal(sub_mu, 1, (i, j))
                L = np.dot(U, V.transpose())
                # sparse matrix generation
                S = np.random.uniform(-2000, 2000, (i, i))
                # select ids of elements equal zero
                ids = np.random.choice(np.arange(S.size), replace=False, size=int(S.size * (1-p)))
                S.reshape(-1)[ids] = 0
                nonzero_ids = np.where(abs(S.reshape(-1)) > 2)[0]
                # nonzero_ids = [index for index in np.arange(S.size) if index not in ids]
                S = S.reshape(i, i)
                # observed demand matrix generation
                M = L + S + Z
                # decomposition by SPCP
                APG = StablePCP(sigma=sigma)
                APG.fit(M)
                L_APG = APG.get_low_rank()
                S_APG = APG.get_sparse()
                RMSE_L = np.linalg.norm(L_APG - L, ord="fro") / i
                RMSE_S = np.linalg.norm(S_APG - S, ord="fro") / i
                mape_L = ((L_APG - L) / L).reshape(-1)
                MAPE_L = np.sum(abs(mape_L)) / (i * i) * 100
                # select non-zero elements of sparse matrix S to avoid infinite MAPE value
                S_true_sel = S.reshape(-1)[nonzero_ids]
                S_APG_sel = S_APG.reshape(-1)[nonzero_ids]
                mape_s = ((S_APG_sel - S_true_sel) / S_true_sel).reshape(-1)
                MAPE_S = np.sum(abs(mape_s)) / (len(nonzero_ids)) * 100
                MAPE_L_rpca[seed][i_ord][j_ord] = MAPE_L
                MAPE_S_rpca[seed][i_ord][j_ord] = MAPE_S
                RMSE_L_rpca[seed][i_ord][j_ord] = RMSE_L
                RMSE_S_rpca[seed][i_ord][j_ord] = RMSE_S
                # moving average method
                M_obs = pd.DataFrame(M)
                M_ma = M_obs.rolling(window=int((i - 1) / 2), axis=1).mean()
                M_ma[np.arange(0, int((i - 1) / 2))] = M_ma[np.arange(int((i - 1) / 2), int((i - 1) / 2) * 2)]
                L_ma = M_ma.to_numpy()
                S_ma = (M_obs - M_ma).to_numpy()
                RMSE_L = np.linalg.norm(L_ma - L, ord="fro") / i
                RMSE_S = np.linalg.norm(S_ma - S, ord="fro") / i
                mape_L = ((L_ma - L) / L).reshape(-1)
                MAPE_L = np.sum(abs(mape_L)) / (i * i) * 100
                S_ma_sel = S_ma.reshape(-1)[nonzero_ids]
                mape_s = ((S_ma_sel - S_true_sel) / S_true_sel).reshape(-1)
                MAPE_S = np.sum(abs(mape_s)) / len(nonzero_ids) * 100
                MAPE_L_ma[seed][i_ord][j_ord] = MAPE_L
                MAPE_S_ma[seed][i_ord][j_ord] = MAPE_S
                RMSE_L_ma[seed][i_ord][j_ord] = RMSE_L
                RMSE_S_ma[seed][i_ord][j_ord] = RMSE_S
                if seed == rep_time - 1:
                    MAPE_L_ave_rpca = MAPE_L_rpca.mean(axis=0)
                    MAPE_S_ave_rpca = MAPE_S_rpca.mean(axis=0)
                    RMSE_L_ave_rpca = RMSE_L_rpca.mean(axis=0)
                    RMSE_S_ave_rpca = RMSE_S_rpca.mean(axis=0)
                    MAPE_L_ave_ma = MAPE_L_ma.mean(axis=0)
                    MAPE_S_ave_ma = MAPE_S_ma.mean(axis=0)
                    RMSE_L_ave_ma = RMSE_L_ma.mean(axis=0)
                    RMSE_S_ave_ma = RMSE_S_ma.mean(axis=0)
                    print("MAPE_L_ave_rpca.shape", MAPE_L_ave_rpca.shape)
                    print("MAPE_L_ave_ma.shape", MAPE_L_ave_ma.shape)
    error_rpca = np.concatenate((MAPE_L_ave_rpca, MAPE_S_ave_rpca, RMSE_L_ave_rpca, RMSE_S_ave_rpca), axis=0)
    error_ma = np.concatenate((MAPE_L_ave_ma, MAPE_S_ave_ma, RMSE_L_ave_ma, RMSE_S_ave_ma), axis=0)
    with open('error_rpca2.npy', 'wb') as f:
        np.save(f, error_rpca)
    with open('error_ma2.npy', 'wb') as f:
        np.save(f, error_ma)
    return None
if __name__ == '__main__':
    # Synthetic_data1_generate()
    Synthetic_data2_generate()
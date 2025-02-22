import numpy as np
import typing as tp
#import numba

try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')


#@numba.njit(cache=True, parallel=True)
def frobenius_norm(M: np.ndarray) -> float:
    """ Compute the Frobenius norm of a matrix """
    return np.sqrt(np.sum(M**2))

#@numba.njit(cache=True, parallel=True)
def shrink(M: np.ndarray, tau: float) -> np.ndarray:
    """ Apply elementwise soft-thresholding """
    return np.sign(M) * np.maximum(np.abs(M) - tau, 0)

#@numba.njit(cache=True, parallel=True)
def svd_threshold(M: np.ndarray, tau: float) -> np.ndarray:
    """ Compute singular value thresholding (SVT) """
    U, S, V = np.linalg.svd(M, full_matrices=False)
    return U @ np.diag(shrink(S, tau)) @ V

#@numba.njit(cache=True, parallel=True)
def fit_numba(
        D: np.ndarray, 
        S: np.ndarray, 
        Y: np.ndarray, 
        mu: float,
        mu_inv: float, 
        lmbda: float, 
        tol: tp.Optional[float], 
        max_iter: int, 
        iter_print: int
    ) -> tuple[np.ndarray, np.ndarray]:
    """ JIT-compiled fit function to accelerate the iterative Principal Component Pursuit (PCP) algorithm """
    iter = 0
    err = np.inf
    L = np.zeros(D.shape)

    if tol is None:
        tol = 1e-7 * frobenius_norm(D)

    while (err > tol) and (iter < max_iter):
        L = svd_threshold(D - S + mu_inv * Y, mu_inv)
        S = shrink(D - L + mu_inv * Y, mu_inv * lmbda)
        Y = Y + mu * (D - L - S)

        err = frobenius_norm(D - L - S)
        iter += 1
        if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= tol:
            #with numba.objmode():
            print("Iteration: ", iter, "Error: ", err)  # w/o numba.objmode() print won't work properly

    return L, S

class R_pca:
    def __init__(self, D: np.ndarray, mu: float = None, lmbda: float = None) -> None:
        """ Initialize R_pca with data matrix D and optional parameters """
        self.D = D
        self.S = np.zeros(D.shape)
        self.Y = np.zeros(D.shape)

        if mu is not None:
            self.mu = mu
        else:
            self.mu = np.prod(D.shape) / (4 * np.linalg.norm(D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda is not None:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(D.shape))

    def fit(self, tol: tp.Optional[float] = None, max_iter: int = 1000, iter_print: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """ Fit the Robust PCA model using Principal Component Pursuit """
        self.L, self.S = fit_numba(self.D, self.S, self.Y, self.mu, self.mu_inv, self.lmbda, tol, max_iter, iter_print)
        return self.L, self.S

    def plot_fit(self, size: tp.Optional[tuple[int, int]] = None, tol: float = 0.1, axis_on: bool = True) -> None:
        """ Plot the low-rank and sparse decomposition """
        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows, ncols = int(sq), int(sq)

        ymin, ymax = np.nanmin(self.D), np.nanmax(self.D)
        print(f'ymin: {ymin}, ymax: {ymax}')

        numplots = min(n, nrows * ncols)
        plt.figure()

        for i in range(numplots):
            plt.subplot(nrows, ncols, i + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[i, :] + self.S[i, :], 'r')
            plt.plot(self.L[i, :], 'b')
            if not axis_on:
                plt.axis('off')

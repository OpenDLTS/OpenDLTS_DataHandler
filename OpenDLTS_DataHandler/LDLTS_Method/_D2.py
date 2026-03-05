"""
Module for implementing the D2 deconvolution algorithm.

This module defines the D2 class that extends L1 regularization to include
second-derivative (D2) regularization for solving Laplace DLTS inversion problems.
The D2 regularization promotes smoothness in the solution by penalizing large
curvatures in the spectral function.

Public Classes:
    D2: Main class implementing the D2 deconvolution algorithm.
"""
__all__ = ['D2']

import cvxpy as cp
import numpy as np

from ._L1 import L1
# LDLTS_METHOD_D2
class D2(L1):
    def _get_term2(self, polarity: str, f_scaling_by_C: bool, C_average: np.ndarray,
                   f: cp.Expression, reg1: float | cp.Parameter,
                   enable_irls_mode: bool, irls_weight: np.ndarray | cp.Parameter) -> cp.Expression:
        if polarity == 'both':
            raise NotImplementedError("D2 regularization with 'both' polarity is not implemented.")
        if enable_irls_mode:
            raise NotImplementedError("D2 regularization with IRLS mode is not implemented.")
        # L,(Ns-2,Ns)
        L = np.diag(np.ones(self.Ns-1), 1)+np.diag(np.ones(self.Ns-1), -1)+np.diag(-2*np.ones(self.Ns))
        L = L[1:-1,:]
        if f_scaling_by_C:
            term2 = reg1 * cp.sum_squares(L @ (f[1:]/C_average))
        else:
            term2 = reg1 * cp.sum_squares(L @ f[1:])
        return term2
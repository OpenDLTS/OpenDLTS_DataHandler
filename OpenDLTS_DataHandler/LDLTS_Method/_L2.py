
__all__ = ['L2']

import cvxpy as cp
import numpy as np
from .._config import *
from .._typing import *
from ._L1 import L1

# LDLTS_METHOD_L2
class L2(L1):
    def _get_term2(self, polarity: str, f_scaling_by_C: bool, C_average: np.ndarray,
                   f: cp.Expression, reg1: float | cp.Parameter,
                   enable_irls_mode: bool, irls_weight: np.ndarray | cp.Parameter) -> cp.Expression:
        if enable_irls_mode:
            raise NotImplementedError("L2 regularization with IRLS mode is not implemented.")
        if f_scaling_by_C:
            term2 = reg1 * cp.sum_squares(f[1:]/C_average)
        else:
            term2 = reg1 * cp.sum_squares(f[1:])
        return term2
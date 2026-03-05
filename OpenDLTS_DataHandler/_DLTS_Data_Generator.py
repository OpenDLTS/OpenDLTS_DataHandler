"""
Module for generating simulated DLTS (Deep Level Transient Spectroscopy) data.

This module provides a DLTS_Data_Generator class for simulating capacitance transients
based on trap parameters and material properties, with support for noise addition and
temperature shifting.
"""
__all__ = ["DLTS_Data_Generator"]

import numpy as np
from ._Trap import Trap
from ._typing import *
class DLTS_Data_Generator:
    def __init__(self, trap_list: list[Trap]) -> None:
        self.trap_list = trap_list
    def get_data(self, T: float | int | np.typing.ArrayLike, t: float | int | np.typing.ArrayLike,
                 sigma: float = 0.2, DC_fun: Callable[[float],float] = lambda T:200,
                 savefile: str | Path | None = None) -> np.ndarray:
        if type(T) in [float, int]:
            self._NT = 1
            self._T = np.array([T])
        else:
            try:
                self._NT = len(T)
                self._T = np.array(T)
            except Exception as e:
                raise ValueError(f'type(T): np.typing.ArrayLike')
        if type(t) in [float, int]:
            self._Nt = 1
            self._t = np.array([t])
        else:
            try:
                self._Nt = len(t)
                self._t = np.array(t)
            except Exception as e:
                raise ValueError(f'type(t): np.typing.ArrayLike')
        result = np.zeros((self._NT+1,self._Nt+1))
        result[1:,0] = self._T
        result[0,1:] = self._t
        for T_idx,iT in enumerate(self._T):
            temp_re = DC_fun(iT)
            for itrap in self.trap_list:
                # (Nt,)
                temp_term = itrap.amp(iT) * np.exp(-itrap.em(iT) * self._t)
                if itrap.trap_type == 'majority':
                    temp_re -= temp_term
                else:
                    temp_re += temp_term
            result[1+T_idx,1:] = temp_re + np.random.normal(0,sigma,size=self._Nt)
        def DLTS_format(x):
            formatted = f"{x:.7E}"
            if 'E-0' in formatted:
                formatted = formatted.replace('E-0', 'E-')
            elif 'E+0' in formatted:
                formatted = formatted.replace('E+0', 'E+')
            return formatted
        DLTS_format = np.vectorize(DLTS_format)
        if savefile:
            np.savetxt(Path(savefile).resolve(),DLTS_format(result),fmt='%s',delimiter='\t')
        return result
from typing import TypedDict, List, Dict, NewType, Any, Callable, Tuple
from pathlib import Path
import numpy as np
from scipy.sparse import csr_array

class _TransientDataType_dict(TypedDict):
    t: np.typing.ArrayLike      # (Nt,)
    T: np.typing.ArrayLike      # (NT,)
    C: np.typing.ArrayLike      # (NT, Nt)

TransientDataType = Path | str | None | _TransientDataType_dict | np.typing.ArrayLike  # (NT+1, Nt+1)

class TransientDataPlotType(TypedDict):
    C_unit: str                 # e.g. '$pF$' | '$\muA$' | '$V$'
    T_unit: str                 # e.g. '$K$' | '$V$' | '$a.u.$'
    t_unit: str                 # e.g. '$s$'
    em_unit: str                # e.g. '$s^{-1}$'
    tau_unit: str               # e.g. '$s$'
    t_label: str                # e.g. 'Time [$s$]'
    T_label: str                # e.g. 'Temperature [$K$]' | 'Voltage [$V$]'
    C_label: str                # e.g. 'Capacitance [$pF$]' | 'Current [$\muA$]' | 'Voltage [$V$]'
    em_label: str               # e.g. 'Emission Rate [$s^{-1}$]'
    tau_label: str              # e.g. 'Time Constant [$s$]'
    DLTS_label: str             # e.g. 'DLTS Signal [$pF$]' | 'DLTS Signal [$\muA$]' | 'DLTS Signal [$V$]'
    LDLTS_label: str            # e.g. 'LDLTS Signal [$pF$]' | 'LDLTS Signal [$\muA$]' | r'$\Delta V_{th}$ [$V$]'
    C_plot_factor: float        # e.g. data scaling for plotting

class _LDLTS_SolveResultTypeBase(TypedDict):
    input_params: Dict[str, Any]
    Ti_list: np.typing.ArrayLike            # (NT, ) List of temperature indices used in the fitting
    Kernel_A: np.ndarray                    # (Nt, Ns+1) Kernel matrix used in the fitting
    f: np.ndarray                           # (Ns+1, NT) LDLTS signal while f[0,:] is the dc (offset) term
    term1_error: float                      # value of term1 in the optimization problem
    rms_list: np.ndarray                    # (NT,) real rms value for each temperature
    reg: float                              # regularization parameter used in the fitting
    term2_error: float                      # value of term2 in the optimization problem
    irls_target: np.ndarray                 # Any Shape. target weight value for next irls fitting
    irls_eval_error_dict: Dict[str, float]  # Any evaluation error during irls fitting
    wall_time: float                        # wall time used in the fitting (seconds)

# SR
Enumerate_Args_Type = Dict[str, List | np.ndarray]
Unenumerate_Args_Type = Dict[str, Any]
class D_info_Type(TypedDict):
    word_fun: str                                                   # Word function name
    enumerate_args_list_dic: Enumerate_Args_Type
    unenumerate_args_list_dic: Unenumerate_Args_Type
    D_length: int                                                   # Number of atoms in the dictionary
    polarity: str                                                   # 'pos' | 'neg' | 'nonneg' | 'nonpos' | 'both'
    dc_mono: str                                                    # 'increase' | 'decrease' | 'both'

Dictionary_Type = csr_array | np.ndarray
D_info_List_Type = List[D_info_Type]
class LDLTS_SolveResultType(_LDLTS_SolveResultTypeBase, total=False):
    D: Dictionary_Type                      # Dictionary matrix, shape (Ns*NT, ND)
    X: np.ndarray                           # Sparse representation coefficients, shape (ND, )
    D_info_list: D_info_List_Type           # List of dictionary information

Enumerate_Args_List_Type = List[Enumerate_Args_Type]
Unenumerate_Args_List_Type = List[Unenumerate_Args_Type]
__all__ = [
    "TransientDataType",
    "Path",
    "TypedDict",
    "List",
    "Dict",
    "Tuple",
    "NewType",
    "Any",
    "Callable",
    "TransientDataPlotType",
    "LDLTS_SolveResultType",
    "Enumerate_Args_Type",
    "Unenumerate_Args_Type",
    "Dictionary_Type",
    "D_info_Type",
    "D_info_List_Type",
    "Enumerate_Args_List_Type",
    "Unenumerate_Args_List_Type",
]
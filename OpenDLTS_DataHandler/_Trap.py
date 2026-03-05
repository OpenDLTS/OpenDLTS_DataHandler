__all__ = ["Trap"]

from ._Material import Material 
import scipy.constants as sci_const
import numpy as np
from scipy.optimize import minimize_scalar as sci_opt_min_sca
from ._typing import *

class Trap:
    """
    Trap model for DLTS analysis.
    Args:
        Ea (float | None): Activation energy in eV for Arrhenius trap. Defaults to None.
        sigma (float | None): Capture cross-section in cm^2 for Arrhenius trap. Defaults to None.
        T_power (float | int): Temperature power factor for Arrhenius trap. Defaults to 2.
        constant_em (float | None): Constant emission rate in s^-1 for constant trap. Defaults to None.
        fun_amplitude_T (Callable[[float],float] | float | int): Function or constant for trap amplitude as a function of temperature. Defaults to 1.
        fun_em_T (Callable[[float],float] | None): User-defined function for emission rate as a function of temperature. Defaults to None.
        material (str): Material name for the trap. Defaults to 'Si'.
        material_doping_type (str): Doping type of the material ('N' or 'P'). Defaults to 'N'.
        trap_type (str): Type of trap ('majority' or 'minority'). Defaults to 'majority'.
    Attributes:
        mat (Material): Material object for the trap.
        mat_dop_type (str): Doping type of the material.
        trap_type (str): Type of trap.
        em (Callable[[float], float]): Emission rate function as a function of temperature.
        tau (Callable[[float], float]): Time constant function as a function of temperature.
        amp (Callable[[float], float]): Amplitude function as a function of temperature.
        trap_type (str): Type of trap ('majority' or 'minority').
    Methods:
        get_T_from_fixed_em: Get temperature from a fixed emission rate.
        get_T_from_fixed_tau: Get temperature from a fixed time constant.
    """
    def __init__(self, Ea: float | None = None, sigma: float | None = None, T_power: float | int = 2,
                 constant_em: float | None = None, fun_amplitude_T: Callable[[float],float] | float | int = 1,
                 fun_em_T: Callable[[float],float] | None = None, material: str = 'Si', material_doping_type: str = 'N',
                 trap_type: str = 'majority') -> None:
        try:
            self.mat = getattr(Material,material)()
        except:
            print('unknow material, add material in class Material')
        if material_doping_type in ['N','P','n','p']:
            self.mat_dop_type = str.upper(material_doping_type)
        if trap_type in ['majority','minority']:
            self.trap_type = trap_type
        if type(fun_amplitude_T) in [float,int]:
            self.amp = lambda T:fun_amplitude_T
        else:
            self.amp = fun_amplitude_T
        # arrh trap
        if all(v is not None for v in [Ea,sigma]):
            self._Ea = Ea
            self._sigma = sigma
            self._T_power = T_power
            def temp_em(T):
                self.mat.T = T
                if self.mat_dop_type == 'N':
                    temp_vth = self.mat.vth_n
                    if self.trap_type == 'majority':
                        temp_dos = self.mat.Nc
                    else:
                        temp_dos = self.mat.Nv
                else:
                    temp_vth = self.mat.vth_p
                    if self.trap_type == 'majority':
                        temp_dos = self.mat.Nv
                    else:
                        temp_dos = self.mat.Nc
                return self._sigma * temp_vth * temp_dos / T**2 * T**self._T_power * np.exp(-self._Ea*sci_const.e/sci_const.k/T)
            self.em = temp_em
        # constant time constant (emission rate) trap
        elif all(v is None for v in [Ea,sigma]) and constant_em is not None:
            self._constant_em = constant_em
            self.em = lambda T: self._constant_em
        # user-defined trap
        elif all(v is None for v in [Ea,sigma,constant_em]) and fun_em_T is not None:
            self.em = fun_em_T
        else:
            raise ValueError(f'InputError: Ea={Ea}, sigma={sigma}, T_power={T_power}, constant_em={constant_em},\
                              fun_amplitude_T={fun_amplitude_T}, fun_em_T={fun_em_T}, material={material},\
                              material_doping_type={material_doping_type}, trap_type={trap_type}')
        self.tau = lambda T: 1/self.em(T)
    def get_T_from_fixed_em(self, fixed_em: float | int, opt_bounds: tuple[float,float] = (0.1, 10000),
                            opt_method: str = 'bounded') -> float | None:
        def objective(T):
            return (fixed_em - self.em(T))**2
        result = sci_opt_min_sca(
            objective,
            bounds=opt_bounds,
            method=opt_method
        )
        if not result.success:
            return None
        else:
            return result.x
    def get_T_from_fixed_tau(self, fixed_tau: float | int, opt_bounds: tuple[float,float] = (0.1, 10000),
                             opt_method: str = 'bounded') -> float | None:
        fixed_em = 1/fixed_tau
        return self.get_T_from_fixed_em(fixed_em=fixed_em, opt_bounds=opt_bounds, opt_method=opt_method)
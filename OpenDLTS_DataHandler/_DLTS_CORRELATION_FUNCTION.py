"""
Deep Level Transient Spectroscopy (DLTS) correlation function implementations.

This module provides several DLTS signal processing techniques based on different 
correlation functions. Each class implements a specific weighting function optimized 
for detecting specific emission rate ranges.

Reference:
    [10.1063/1.1148038] - Original paper describing these optimized correlation functions

Classes:
    DLTS_CORRELATION_FUNCTION: Container class for different correlation methods
"""
__all__ = ["DLTS_CORRELATION_FUNCTION"]

import numpy as np
from scipy.optimize import minimize_scalar as sci_opt_min_sca
class DLTS_CORRELATION_FUNCTION:
    """Container class for various DLTS correlation function implementations."""
    class shifted_exponential:
        """
        Shifted exponential correlation function (1st order).
        
        Attributes:
            optimum_td_tc_ratio (float): Optimal time delay to time constant ratio
            optimum_pw (float): Optimal pulse width setting
            optimum_SNR (float): Optimal signal-to-noise ratio
            order (int): Order of the correlation function
        """
        optimum_td_tc_ratio = 0.082
        optimum_pw = 16.2
        optimum_SNR = 0.21
        order = 1
        def main_fun(self, t: np.ndarray, td: float, tc: float) -> np.ndarray:
            """
            Core correlation function calculation.
            
            Args:
                t (ndarray): Time values array
                td (float): Time delay parameter
                tc (float): Time constant parameter
                
            Returns:
                ndarray: Weight function values at given time points
            """
            t_star = (t-td)/tc
            return -(np.exp(-2*t_star)+(np.exp(-2)-1)/2)
        def find_em(self, tc0: float, tc1: float, em0: float=1e-1, em1: float=1e5, rel_tol: float=1e-3) -> tuple:
            """
            Find Rate Window (Find emission rate (em) that maximizes the correlation signal).
            
            Args:
                tc0 (float): Start time for rate window
                tc1 (float): End time for rate window
                em0 (float): Lower bound for emission rate search (default: 0.1)
                em1 (float): Upper bound for emission rate search (default: 1e5)
                rel_tol (float): Relative tolerance for optimization (default: 1e-3)
                
            Returns:
                tuple: 
                    em_opt (float): Optimal emission rate value
                    em_list (ndarray): Tested emission rate values
                    val_list (ndarray): Corresponding correlation signal values
                    
            Raises:
                RuntimeError: If optimization fails
            """
            # Convert to logarithmic space
            em0_log = np.log(em0)
            em1_log = np.log(em1)
            # Define the objective function (negative values in logarithmic space, because we want to maximize)
            def objective(log_em):
                em = np.exp(log_em)
                t_list = np.linspace(tc0,tc1,num=2000)
                c_list = 100 - np.exp(-em*t_list)
                return -np.trapezoid(self.main_fun(t_list, tc0, tc1-tc0)*c_list, t_list)
            # Calculates the tolerance in logarithmic space (rel_tol is converted to a multiplicative factor)
            tol_log = np.log(1 + rel_tol)
            # Solving using bounded optimization methods
            result = sci_opt_min_sca(
                objective,
                bounds=(em0_log, em1_log),
                method='bounded',
                options={'xatol': tol_log}
            )
            if not result.success:
                raise RuntimeError("solve failed: " + result.message)
            em_opt = np.exp(result.x)
            em_log_list = np.linspace(np.log(em0),np.log(em1),600)
            em_list = np.exp(em_log_list)
            val_list = np.zeros_like(em_log_list)
            for i,em in enumerate(em_log_list):
                val_list[i] = -objective(em)
            return em_opt,em_list,val_list
        def __once(self, t_list: np.ndarray, c_list: np.ndarray, tc0: float, tc1: float, interp_type: str = 'lin') -> float:
            """
            Compute single correlation integral (private helper).
            
            Args:
                t_list (ndarray): Time values
                c_list (ndarray): Capacitance values
                tc0 (float): Start time for integration
                tc1 (float): End time for integration
                interp_type (str): Interpolation type ('lin' or 'log', default: 'lin')
                
            Returns:
                float: Correlation integral value
            """
            len_interp = np.searchsorted(t_list, tc1)-np.searchsorted(t_list, tc0)
            if interp_type == 'lin':
                interp_t_list = np.linspace(tc0,tc1,num=2*len_interp)
            elif interp_type == 'log':
                interp_t_list = np.logspace(np.log10(tc0),np.log10(tc1),num=2*len_interp)
            interp_c_list = np.interp(interp_t_list, t_list, c_list)
            return np.trapezoid(self.main_fun(interp_t_list, tc0, tc1-tc0)*interp_c_list, interp_t_list)
        def __call__(self, t: np.ndarray, T: np.ndarray, C: np.ndarray, use_opt_ratio: bool = True,
                     tc0: float = -1, tc1: float = -1, interp_type: str = 'lin', em0: float = 1e-1,
                     em1: float = 1e5, rel_tol: float = 1e-3) -> tuple:
            """
            Compute DLTS signal for multiple temperature points.
            
            Args:
                t (ndarray): Time measurement points
                T (ndarray): Temperature values array
                C (ndarray): Capacitance transients matrix (shape: [n_temps, n_times])
                use_opt_ratio (bool): Use optimal time ratio if True (default: True)
                tc0 (float): Manual start time setting (used if use_opt_ratio=False)
                tc1 (float): Manual end time setting (default: last time point)
                interp_type (str): Interpolation type ('lin' or 'log', default: 'lin')
                em0 (float): Lower emission rate bound (default: 0.1)
                em1 (float): Upper emission rate bound (default: 1e5)
                rel_tol (float): Optimization tolerance (default: 1e-3)
                
            Returns:
                tuple:
                    T (ndarray): Temperature values (same as input)
                    val_list (ndarray): DLTS signal amplitudes at each temperature
                    rate_window (tuple): Optimal emission rate and evaluation data
                    
            Notes:
                Returns 1 if tc0 is before first measurement point
            """
            if tc1==-1:
                tc1 = t[-1]
            if use_opt_ratio:
                tc0 = self.optimum_td_tc_ratio * tc1 / (1 + self.optimum_td_tc_ratio)
            else:
                if tc0==-1:
                    tc0 = t[0]
            if tc0<=t[0]:
                return 1
            val_list = np.zeros_like(T)
            for i,_ in enumerate(T):
                val_list[i] = self.__once(t, C[i], tc0, tc1, interp_type=interp_type)
            # Find Rate Window
            rate_window = self.find_em(tc0, tc1, em0=em0, em1=em1, rel_tol=rel_tol)
            return T,val_list,rate_window
    class double_boxcar(shifted_exponential):
        """
        Double boxcar correlation function (1st order).
        
        Attributes:
            optimum_td_tc_ratio (float): Optimal time delay to time constant ratio
            optimum_pw (float): Optimal pulse width setting
            optimum_SNR (float): Optimal signal-to-noise ratio
            order (int): Order of the correlation function
        """
        optimum_td_tc_ratio = 0.131
        optimum_pw = 16.5
        optimum_SNR = 0.13
        order = 1
        def main_fun(self, t, td, tc):
            """
            Core correlation function calculation.
            
            Args:
                t (ndarray): Time values array
                td (float): Time delay parameter
                tc (float): Time constant parameter
                
            Returns:
                ndarray: Weight function values at given time points
            """
            t_star = (t-td)/tc
            mask1 = (t_star>=0) & (t_star<0.1)
            mask2 = (t_star>=0.1) & (t_star<=0.9)
            mask3 = (t_star>0.9) & (t_star<=1)
            result = np.zeros_like(t_star)
            result[mask1] = -1.0
            result[mask2] = 0.0
            result[mask3] = 1.0
            return result
    class triangular(shifted_exponential):
        """
        Triangular correlation function (2nd order).
        
        Attributes:
            optimum_td_tc_ratio (float): Optimal time delay to time constant ratio
            optimum_pw (float): Optimal pulse width setting
            optimum_SNR (float): Optimal signal-to-noise ratio
            order (int): Order of the correlation function
        """
        optimum_td_tc_ratio = 0.037
        optimum_pw = 8.8
        optimum_SNR = 0.092
        order = 2
        def main_fun(self, t, td, tc):
            """
            Core correlation function calculation.
            
            Args:
                t (ndarray): Time values array
                td (float): Time delay parameter
                tc (float): Time constant parameter
                
            Returns:
                ndarray: Weight function values at given time points
            """
            t_star = (t-td)/tc
            mask1 = (t_star>=0) & (t_star<0.5)
            mask2 = (t_star>=0.5) & (t_star<=1)
            result = np.zeros_like(t_star)
            result[mask1] = -(1-4*t_star[mask1])
            result[mask2] = -(4*t_star[mask2]-3)
            return result
    class HiRes_4(shifted_exponential):
        """
        HiRes_4 correlation function (3rd order).
        
        Attributes:
            optimum_td_tc_ratio (float): Optimal time delay to time constant ratio
            optimum_pw (float): Optimal pulse width setting
            optimum_SNR (float): Optimal signal-to-noise ratio
            order (int): Order of the correlation function
        """
        optimum_td_tc_ratio = 0.011
        optimum_pw = 6.7
        optimum_SNR = 0.029
        order = 3
        def main_fun(self, t, td, tc):
            """
            Core correlation function calculation.
            
            Args:
                t (ndarray): Time values array
                td (float): Time delay parameter
                tc (float): Time constant parameter
                
            Returns:
                ndarray: Weight function values at given time points
            """
            t_star = (t-td)/tc
            mask1 = (t_star>=0) & (t_star<0.25)
            mask2 = (t_star>=0.25) & (t_star<0.5)
            mask3 = (t_star>=0.5) & (t_star<0.75)
            mask4 = (t_star>=0.75) & (t_star<=1.0)
            result = np.zeros_like(t_star)
            result[mask1] = -1.0
            result[mask2] = 3.0
            result[mask3] = -3.0
            result[mask4] = 1.0
            return result
    class HiRes_5(shifted_exponential):
        """
        HiRes_5 correlation function (4th order).
        
        Attributes:
            optimum_td_tc_ratio (float): Optimal time delay to time constant ratio
            optimum_pw (float): Optimal pulse width setting
            optimum_SNR (float): Optimal signal-to-noise ratio
            order (int): Order of the correlation function
        """
        optimum_td_tc_ratio = 0.007
        optimum_pw = 5.9
        optimum_SNR = 0.013
        order = 4
        def main_fun(self, t, td, tc):
            """
            Core correlation function calculation.
            
            Args:
                t (ndarray): Time values array
                td (float): Time delay parameter
                tc (float): Time constant parameter
                
            Returns:
                ndarray: Weight function values at given time points
            """
            t_star = (t-td)/tc
            mask1 = (t_star>=0) & (t_star<0.2)
            mask2 = (t_star>=0.2) & (t_star<0.4)
            mask3 = (t_star>=0.4) & (t_star<0.6)
            mask4 = (t_star>=0.6) & (t_star<0.8)
            mask5 = (t_star>=0.8) & (t_star<=1.0)
            result = np.zeros_like(t_star)
            result[mask1] = -1.0
            result[mask2] = 4.0
            result[mask3] = -6.0
            result[mask4] = 4.0
            result[mask5] = -1.0
            return result
    class HiRes_6(shifted_exponential):
        """
        HiRes_6 correlation function (5th order).
        
        Attributes:
            optimum_td_tc_ratio (float): Optimal time delay to time constant ratio
            optimum_pw (float): Optimal pulse width setting
            optimum_SNR (float): Optimal signal-to-noise ratio
            order (int): Order of the correlation function
        """
        optimum_td_tc_ratio = 0.005
        optimum_pw = 5.4
        optimum_SNR = 0.0058
        order = 5
        def main_fun(self, t, td, tc):
            """
            Core correlation function calculation.
            
            Args:
                t (ndarray): Time values array
                td (float): Time delay parameter
                tc (float): Time constant parameter
                
            Returns:
                ndarray: Weight function values at given time points
            """
            t_star = (t-td)/tc
            mask1 = (t_star>=0.0) & (t_star<1/6)
            mask2 = (t_star>=1/6) & (t_star<2/6)
            mask3 = (t_star>=2/6) & (t_star<3/6)
            mask4 = (t_star>=3/6) & (t_star<4/6)
            mask5 = (t_star>=4/6) & (t_star<5/6)
            mask6 = (t_star>=5/6) & (t_star<=1.0)
            result = np.zeros_like(t_star)
            result[mask1] = -1.0
            result[mask2] = 5.0
            result[mask3] = -10.0
            result[mask4] = 10.0
            result[mask5] = -5.0
            result[mask6] = 1.0
            return result
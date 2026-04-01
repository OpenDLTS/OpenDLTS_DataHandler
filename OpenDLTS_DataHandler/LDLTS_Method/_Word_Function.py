__all__ = ['Word_Function']

import numpy as np
import scipy
from .._Material import Material

class Word_Function:
    """
    Provides base function calculations for DLTS data processing dictionary matrices.
    
    This class contains static methods for creating design matrix components that characterize
    different physical processes (e.g., Arrhenius processes, tunneling emission).
    Primary functionality includes emission rate interpolation, B-spline basis generation, 
    and sparse dictionary matrix construction.
    
    Attributes
    ----------
    (This class contains only static methods, no instance attributes)
    """
    @staticmethod
    def worker_for_get_dictionary(arg_pack):
        """
        Multiprocessing worker function for parallel dictionary matrix calculation.

        Parameters
        ----------
        arg_pack : tuple
            A tuple containing:
            - word_fun : callable
                The dictionary calculation function (e.g., get_dictionary_word_arrh_sparse)
            - unenumerate_args_list_dic : dict
                Dictionary of fixed arguments
            - arg : tuple
                Tuple containing current task index and dynamic parameters: (i, iarg)

        Returns
        -------
        ndarray or None
            Calculated non-zero elements array in format [[row_index, 0, value]], 
            or None if no data
        """
        word_fun, unenumerate_args_list_dic, arg = arg_pack
        i, iarg = arg
        tempsp, _ = word_fun(**iarg, **unenumerate_args_list_dic)
        if tempsp is not None:
            tempsp[:, 1] = i
        return tempsp
    @staticmethod
    def emission_rate_array_interpolation(s:np.ndarray,target_s):
        """
        Find interpolation indices and weights for target emission rate in sorted array.

        Uses linear interpolation to determine the position of target emission rate within
        a sorted emission rate array, returning nearest emission rate point indices and weights.

        Parameters
        ----------
        s : ndarray
            Sorted array of emission rates (ascending order)
        target_s : float
            Target emission rate value for interpolation

        Returns
        -------
        tuple (index_list, val_list)
            index_list : list[int]
                Indices of emission rate points participating in interpolation
            val_list : list[float]
                Corresponding interpolation weights (values between 0 and 1)

        Raises
        ------
        ValueError
            If input array length is less than 4
        """
        Ns = len(s)
        if Ns < 4:
            raise ValueError('emission_rate_array_interpolation Error: Ns too small')
        # Find the position of target_s from s_list
        pos = np.searchsorted(s, target_s)
        index_list = []
        val_list = []
        # Larger than the border
        if pos == Ns:
            if target_s < 2*s[-1]-s[-2]:
                # The last emission rate sequence
                val = (1 - (target_s - s[-1])/(s[-1]-s[-2]))
                if val != 0:
                    index_list.append(Ns - 1)
                    val_list.append(val)
            else:
                # target_s is outside the bounds of array s
                pass
        # Smaller than the border
        elif pos == 0 and target_s > 2*s[0]-s[1]:
            # The first emission rate sequence
            val = (target_s - (2*s[0]-s[1]))/(s[1]-s[0])
            if val != 0:
                index_list.append(0)
                val_list.append(val)
        elif s[pos] == target_s:
            # The (pos)th emission rate sequence
            index_list.append(pos)
            val_list.append(1.0)
        # in range
        elif pos > 0 and pos < Ns and s[pos] != target_s:
            # The (pos-1)th emission rate sequence
            val = (1 - (target_s - s[pos-1])/(s[pos]-s[pos-1]))
            if val != 0:
                index_list.append(pos - 1)
                val_list.append(val)
            # The (pos)th emission rate sequence
            val = 1-val
            if val != 0:
                index_list.append(pos)
                val_list.append(val)
        return index_list,val_list
    @staticmethod
    def get_B_spline_coeff_array(T_list, target_s_list, s, B_sp_n, B_sp_basis_i, B_sp_degree):
        """
        Generate B-spline basis function coefficient array.

        Computes B-spline basis function values for a temperature sequence given B-spline parameters.
        Returns all-ones array when parameters are invalid (no B-spline correction).

        Parameters
        ----------
        T_list : ndarray
            Temperature sequence (K)
        target_s_list : ndarray
            Target emission rate sequence (same length as T_list)
        s : ndarray
            Global emission rate array (defines valid range)
        B_sp_n : int
            Total number of B-spline basis functions
        B_sp_basis_i : int
            Index of current basis function (0 to B_sp_n-1)
        B_sp_degree : int
            B-spline degree (order = degree + 1)

        Returns
        -------
        ndarray
            Array of B-spline basis function values with same shape as T_list
        """
        if (B_sp_n>2) and (B_sp_basis_i>=0) and (B_sp_basis_i<B_sp_n):
            # Uniform nodes (can be adjusted according to data distribution)
            knots = np.linspace(T_list.min(), T_list.max(), B_sp_n - B_sp_degree + 1)
            knots = np.r_[(T_list.min(),)*B_sp_degree, knots, (T_list.max(),)*B_sp_degree]
            coefs = np.zeros(B_sp_n)
            coefs[B_sp_basis_i] = 1.0
            # B-spline basis functions
            B = np.zeros_like(T_list)
            mask = np.argwhere((s[0]<=target_s_list)&(target_s_list<=s[-1])).reshape(-1)
            B[mask] = scipy.interpolate.BSpline(knots, coefs, B_sp_degree)(T_list[mask])
            return B
        else:
            # No correction
            return np.ones_like(T_list)
    # Create a single sparse dictionary based on the Arrhenius equation, Ns, T, s, Ti_list are required inputs
    @staticmethod
    def get_dictionary_word_arrh_sparse(Ns, T, s, Ti_list, alpha, beta, ref_material='si', ref_doping_type='N', T_power=2, manual_T_list=None,
                                        B_sp_n=-1, B_sp_basis_i=-1, B_sp_degree=2):
        """
        Construct sparse dictionary matrix elements for Arrhenius processes.

        Based on the Arrhenius equation:
        s = β * T^T_power * vth * Nc/Nv * exp(-α/kT)
        Computes emission rates at each temperature point and generates non-zero matrix elements
        through linear interpolation.

        Parameters
        ----------
        Ns : int
            Emission rate array length
        T : ndarray
            Full temperature sequence (K)
        s : ndarray
            Emission rate array (sorted)
        Ti_list : list[int]
            Indices selecting temperature points from T
        alpha : float
            Activation energy (eV)
        beta : float
            Pre-exponential factor
        ref_material : str, optional
            Reference material type (default 'si'), options: 
            'si', 'Silicon', 'GaN', 'SiC', etc.
        ref_doping_type : str, optional
            Doping type (default 'N'), options: 'N' or 'P'
        T_power : int, optional
            Temperature power term (default 2)
        manual_T_list : ndarray, optional
            Manually specified temperature sequence (K) (overrides Ti_list)
        B_sp_n : int, optional
            Number of B-spline basis functions (default -1, not used)
        B_sp_basis_i : int, optional
            Current B-spline basis index (default -1, not used)
        B_sp_degree : int, optional
            B-spline degree (default 2)

        Returns
        -------
        tuple (W_ndarray, W_shape)
            W_ndarray : ndarray or None
                Non-zero elements array in format: [[row_index, 0, value]] or None
            W_shape : tuple
                Dictionary matrix shape: (Ns * num_temperature_points, 1)

        Notes
        -----
        W_ndarray format explanation:
        Column 0: Row indices (arranged by temperature point: T0_rate0, T0_rate1... T1_rate0...)
        Column 1: Always 0 (reserved)
        Column 2: Matrix element values at corresponding row indices
        """
        try:
            ref_mat = getattr(Material,ref_material)()
        except:
            print('unknow material, add material in class Material')
        
        if manual_T_list is None:
            W_shape = (Ns*len(Ti_list),1)
        else:
            W_shape = (Ns*len(manual_T_list),1)
        # Stores the non-zero value of W
        W_row_val_tuple_list = []
        if ref_doping_type=='N':
            if manual_T_list is None:
                target_s_list = beta * T[Ti_list]**T_power * ref_mat.vth_n * ref_mat.Nc / ref_mat.T**2 * np.exp(-alpha/T[Ti_list]/scipy.constants.k*scipy.constants.e)
            else:
                target_s_list = beta * manual_T_list**T_power * ref_mat.vth_n * ref_mat.Nc / ref_mat.T**2 * np.exp(-alpha/manual_T_list/scipy.constants.k*scipy.constants.e)
        elif ref_doping_type=='P':
            if manual_T_list is None:
                target_s_list = beta * T[Ti_list]**T_power * ref_mat.vth_p * ref_mat.Nv / ref_mat.T**2 * np.exp(-alpha/T[Ti_list]/scipy.constants.k*scipy.constants.e)
            else:
                target_s_list = beta * manual_T_list**T_power * ref_mat.vth_p * ref_mat.Nv / ref_mat.T**2 * np.exp(-alpha/manual_T_list/scipy.constants.k*scipy.constants.e)
        else:
            print('ref_doping_type?')
        # B_spline coeff.
        B = Word_Function.get_B_spline_coeff_array(T_list = T[Ti_list], target_s_list = target_s_list, s = s, B_sp_n = B_sp_n,
                                                   B_sp_basis_i = B_sp_basis_i, B_sp_degree = B_sp_degree)
        for index,target_s in enumerate(target_s_list):
            current_B = B[index]
            current_T = T[Ti_list[index]]
            current_start_idx = Ns * index
            # Linear interpolation
            temp_index_list,temp_val_list = Word_Function.emission_rate_array_interpolation(s=s, target_s=target_s)
            for temp_i,temp_v in zip(temp_index_list,temp_val_list):
                W_row_val_tuple_list.append((current_start_idx + temp_i, current_B * temp_v))
        if W_row_val_tuple_list:
            rows = []
            vals = []
            for (i, v) in W_row_val_tuple_list:
                rows.append(i)
                vals.append(v)
            W_ndarray = np.zeros((len(rows),3))
            W_ndarray[:,0] = np.array(rows)
            W_ndarray[:,2] = np.array(vals)
        else:
            W_ndarray = None
        # Return Tuple: (W_ndarray: np.ndarray, W_shape: Tuple)
        # W_ndarray: shape=(*,3)
        # W_ndarray[:,0] sorting: according to the index value of Ns when the amplitude is not zero at temperature T[0] + the index value of Ns when the amplitude is not zero at temperature T[1].
        # W_ndarray[:,1] is left blank, all 0
        # W_ndarray[:,2] sorting: according to the amplitude value of Ns when the amplitude is not zero at temperature T[0] + the amplitude value of Ns when the amplitude is not zero at temperature T[1].
        return W_ndarray,W_shape
    # Create a single sparse dictionary based on the Arrhenius equation, Ns, T, s, Ti_list are required inputs
    @staticmethod
    def calc_limited_arrh_given_em(target_em, fixed_em, fixed_T, Ea, T_power=2):
        from scipy.optimize import minimize_scalar as sci_opt_min_sca
        y_arrh = -np.log(fixed_em/fixed_T**T_power)
        x_arrh = 1/fixed_T/scipy.constants.k*scipy.constants.e
        def objective(T):
            return np.sum(np.square(target_em - T**T_power*np.exp(Ea*x_arrh-y_arrh)*np.exp(-Ea/T/scipy.constants.k*scipy.constants.e)))
        result = sci_opt_min_sca(
            objective,
            bounds=(1,10000),
            method='bounded'
        )
        if not result.success:
            raise RuntimeError("solve target_T failed")
        return result.x
            
    @staticmethod
    def get_dictionary_word_limited_arrh_sparse(Ns, T, s, Ti_list, Ea, Fixed_T, Fixed_em, T_power=2, manual_T_list=None,
                                                B_sp_n=-1, B_sp_basis_i=-1, B_sp_degree=2):
        if manual_T_list is None:
            W_shape = (Ns*len(Ti_list),1)
        else:
            W_shape = (Ns*len(manual_T_list),1)
        # Stores the non-zero value of W
        W_row_val_tuple_list = []
        # Given em,T,Ea, get sigma
        y_arrh = -np.log(Fixed_em/Fixed_T**T_power)
        x_arrh = 1/Fixed_T/scipy.constants.k*scipy.constants.e
        if manual_T_list is None:
            target_s_list = T[Ti_list]**T_power*np.exp(Ea*x_arrh-y_arrh)*np.exp(-Ea/T[Ti_list]/scipy.constants.k*scipy.constants.e)
        else:
            target_s_list = manual_T_list**T_power*np.exp(Ea*x_arrh-y_arrh)*np.exp(-Ea/manual_T_list/scipy.constants.k*scipy.constants.e)

        # B_spline coeff.
        B = Word_Function.get_B_spline_coeff_array(T_list = T[Ti_list], target_s_list = target_s_list, s = s, B_sp_n = B_sp_n,
                                                   B_sp_basis_i = B_sp_basis_i, B_sp_degree = B_sp_degree)
        for index,target_s in enumerate(target_s_list):
            current_B = B[index]
            current_T = T[Ti_list[index]]
            current_start_idx = Ns * index
            # Linear interpolation
            temp_index_list,temp_val_list = Word_Function.emission_rate_array_interpolation(s=s, target_s=target_s)
            for temp_i,temp_v in zip(temp_index_list,temp_val_list):
                W_row_val_tuple_list.append((current_start_idx + temp_i, current_B * temp_v))
        if W_row_val_tuple_list:
            rows = []
            vals = []
            for (i, v) in W_row_val_tuple_list:
                rows.append(i)
                vals.append(v)
            W_ndarray = np.zeros((len(rows),3))
            W_ndarray[:,0] = np.array(rows)
            W_ndarray[:,2] = np.array(vals)
        else:
            W_ndarray = None
        # Return Tuple: (W_ndarray: np.ndarray, W_shape: Tuple)
        # W_ndarray: shape=(*,3)
        # W_ndarray[:,0] sorting: according to the index value of Ns when the amplitude is not zero at temperature T[0] + the index value of Ns when the amplitude is not zero at temperature T[1].
        # W_ndarray[:,1] is left blank, all 0
        # W_ndarray[:,2] sorting: according to the amplitude value of Ns when the amplitude is not zero at temperature T[0] + the amplitude value of Ns when the amplitude is not zero at temperature T[1].
        return W_ndarray,W_shape

    @staticmethod
    def get_dictionary_word_cover_arrh_sparse(Ns, T, s, Ti_list, Ea, word_index, upper_em, lower_em, T_num, upper_T=-1, lower_T=-1, T_power=2, manual_T_list=None,
                                                B_sp_n=-1, B_sp_basis_i=-1, B_sp_degree=2):
        if manual_T_list is None:
            W_shape = (Ns*len(Ti_list),1)
        else:
            W_shape = (Ns*len(manual_T_list),1)
        # Stores the non-zero value of W
        W_row_val_tuple_list = []
        # get ref_T0, ref_T1
        if upper_T<=0 and lower_T>0:
            ref_T0 = lower_T if lower_T < T[-1] else T[0]
            ref_T1 = T[-1]
        elif upper_T>0 and lower_T<=0:
            ref_T0 = T[0]
            ref_T1 = upper_T if upper_T > T[0] else T[-1]
        elif upper_T>0 and lower_T>0:
            if upper_T>lower_T:
                ref_T0 = lower_T
                ref_T1 = upper_T
        else:
            ref_T0 = T[0]
            ref_T1 = T[-1]
            
        # Given upper_em, lower_em, ref_T0, ref_T1, Ea, get lower_x_arrh, lower_y_arrh, upper_x_arrh, upper_y_arrh
        lower_x_arrh = 1/ref_T1/scipy.constants.k*scipy.constants.e
        lower_y_arrh = -np.log(lower_em/ref_T1**T_power)
        upper_x_arrh = 1/ref_T0/scipy.constants.k*scipy.constants.e
        upper_y_arrh = -np.log(upper_em/ref_T0**T_power)

        x_arrh = np.linspace(lower_x_arrh,upper_x_arrh,T_num)[word_index]
        y_arrh = np.linspace(lower_y_arrh,upper_y_arrh,T_num)[word_index]
        
        if manual_T_list is None:
            target_s_list = T[Ti_list]**T_power*np.exp(Ea*x_arrh-y_arrh)*np.exp(-Ea/T[Ti_list]/scipy.constants.k*scipy.constants.e)
        else:
            target_s_list = manual_T_list**T_power*np.exp(Ea*x_arrh-y_arrh)*np.exp(-Ea/manual_T_list/scipy.constants.k*scipy.constants.e)

        # B_spline coeff.
        B = Word_Function.get_B_spline_coeff_array(T_list = T[Ti_list], target_s_list = target_s_list, s = s, B_sp_n = B_sp_n,
                                                   B_sp_basis_i = B_sp_basis_i, B_sp_degree = B_sp_degree)
        for index,target_s in enumerate(target_s_list):
            current_B = B[index]
            current_T = T[Ti_list[index]]
            current_start_idx = Ns * index
            # Linear interpolation
            temp_index_list,temp_val_list = Word_Function.emission_rate_array_interpolation(s=s, target_s=target_s)
            for temp_i,temp_v in zip(temp_index_list,temp_val_list):
                W_row_val_tuple_list.append((current_start_idx + temp_i, current_B * temp_v))
        if W_row_val_tuple_list:
            rows = []
            vals = []
            for (i, v) in W_row_val_tuple_list:
                rows.append(i)
                vals.append(v)
            W_ndarray = np.zeros((len(rows),3))
            W_ndarray[:,0] = np.array(rows)
            W_ndarray[:,2] = np.array(vals)
        else:
            W_ndarray = None
        # Return Tuple: (W_ndarray: np.ndarray, W_shape: Tuple)
        # W_ndarray: shape=(*,3)
        # W_ndarray[:,0] sorting: according to the index value of Ns when the amplitude is not zero at temperature T[0] + the index value of Ns when the amplitude is not zero at temperature T[1].
        # W_ndarray[:,1] is left blank, all 0
        # W_ndarray[:,2] sorting: according to the amplitude value of Ns when the amplitude is not zero at temperature T[0] + the amplitude value of Ns when the amplitude is not zero at temperature T[1].
        return W_ndarray,W_shape
    
    @staticmethod
    def get_dictionary_word_constant_emission_sparse(Ns, T, s, Ti_list, constant_emission_rate, manual_T_list=None,
                                                     B_sp_n=-1, B_sp_basis_i=-1, B_sp_degree=2):
        """
        Construct sparse dictionary matrix elements for constant emission processes.

        Generates matrix representation for temperature-independent emission processes
        (e.g., tunneling emission).

        Parameters
        ----------
        Ns : int
            Emission rate array length
        T : ndarray
            Full temperature sequence (K)
        s : ndarray
            Emission rate array (sorted)
        Ti_list : list[int]
            Indices selecting temperature points from T
        constant_emission_rate : float
            Constant emission rate value
        manual_T_list : ndarray, optional
            Manually specified temperature sequence (K) (overrides Ti_list)
        B_sp_n : int, optional
            Number of B-spline basis functions (default -1, not used)
        B_sp_basis_i : int, optional
            Current B-spline basis index (default -1, not used)
        B_sp_degree : int, optional
            B-spline degree (default 2)

        Returns
        -------
        tuple (W_ndarray, W_shape)
            W_ndarray : ndarray or None
                Non-zero elements array in format: [[row_index, 0, value]] or None
            W_shape : tuple
                Dictionary matrix shape: (Ns * num_temperature_points, 1)
        """
        if manual_T_list is None:
            W_shape = (Ns*len(Ti_list),1)
        else:
            W_shape = (Ns*len(manual_T_list),1)
        if manual_T_list is None:
            Ti_list2 = Ti_list
        else:
            Ti_list2 = manual_T_list
        W_row_val_tuple_list = []
        # B_spline coeff.
        target_s_list = constant_emission_rate*np.ones_like(T[Ti_list2])
        B = Word_Function.get_B_spline_coeff_array(T_list = T[Ti_list2], target_s_list = target_s_list, s = s, B_sp_n = B_sp_n,
                                                   B_sp_basis_i = B_sp_basis_i, B_sp_degree = B_sp_degree)
        for index,Ti in enumerate(Ti_list2):
            current_B = B[index]
            current_T = T[Ti]
            current_start_idx = Ns * index
            temp_index_list,temp_val_list = Word_Function.emission_rate_array_interpolation(s=s, target_s=constant_emission_rate)
            for temp_i,temp_v in zip(temp_index_list,temp_val_list):
                W_row_val_tuple_list.append((current_start_idx + temp_i, current_B * temp_v))
        if W_row_val_tuple_list:
            rows = []
            vals = []
            for (i, v) in W_row_val_tuple_list:
                rows.append(i)
                vals.append(v)
            W_ndarray = np.zeros((len(rows),3))
            W_ndarray[:,0] = np.array(rows)
            W_ndarray[:,2] = np.array(vals)
        else:
            W_ndarray = None
        return W_ndarray,W_shape
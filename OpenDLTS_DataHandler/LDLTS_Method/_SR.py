"""
DLTS Sparse Representation (SR) Solver Module.

Implements a sparse representation approach for Deep Level Transient Spectroscopy (DLTS) data analysis.
This module provides methods for dictionary generation, optimization problem formulation, 
and visualization of defect characterization results.

Classes:
    SR: Sparse solver for DLTS data based on sparse representation techniques.

Constants:
    all: Exportable modules (only 'SR' class)
"""
__all__ = ['SR']

import cvxpy as cp
import numpy as np
import scipy,time
from .._typing import *
from ._L1 import L1
from ._Word_Function import Word_Function

# LDLTS_METHOD_SR
class SR(L1):
    # generate_dictionary
    def generate_dictionary_stack_sparse(self, word_fun_list: list,
                                         enumerate_args_list_dic_list: list,
                                         unenumerate_args_list_dic_list: list,
                                         polarity_list: list, dc_mono_list: list) -> Tuple[Dictionary_Type, D_info_List_Type]:
        """
        Generate a stacked sparse dictionary matrix from multiple word functions.
        
        Combines multiple dictionary atoms into a single sparse matrix representation.

        Args:
            word_fun_list: List of word functions or function names to generate dictionary atoms.
            enumerate_args_list_dic_list: List of dictionaries with enumerable arguments for each word function.
            unenumerate_args_list_dic_list: List of dictionaries with common arguments for each word function.
            polarity_list: Optional list of polarity constraints for each dictionary segment.
            dc_mono_list: Optional list of monotonicity constraints for each dictionary segment.

        Returns:
            Tuple (D_mat, info):
                D_mat: scipy.sparse matrix representing the stacked dictionary
                info: List of dictionaries containing metadata for each dictionary segment

        Raises:
            ValueError: If input list dimensions are inconsistent.
        """
        if (len(word_fun_list)!=len(enumerate_args_list_dic_list)) or (len(word_fun_list)!=len(unenumerate_args_list_dic_list)) or (len(word_fun_list)!=len(polarity_list)) or (len(word_fun_list)!=len(dc_mono_list)):
            raise ValueError("input list dimension not match")
        # init D_matrix
        D_mat = None
        info = []
        for i,word_fun in enumerate(word_fun_list):
            enumerate_args_list_dic = enumerate_args_list_dic_list[i]
            unenumerate_args_list_dic = unenumerate_args_list_dic_list[i]
            polarity = polarity_list[i]
            dc_mono = dc_mono_list[i]
            tempD,tempinfo = self.get_dictionary_from_word_sparse(word_fun, enumerate_args_list_dic, unenumerate_args_list_dic, polarity, dc_mono)
            if D_mat is None:
                D_mat = tempD
            else:
                #D_mat = np.hstack((D_mat,tempD))
                D_mat = scipy.sparse.hstack((D_mat,tempD))
            info.append(tempinfo)
        return D_mat,info
    
    def get_dictionary_from_word_sparse(self, word_fun: Callable,
                                        enumerate_args_list_dic: dict,
                                        unenumerate_args_list_dic: dict,
                                        polarity: str, dc_mono: str) -> Tuple[Dictionary_Type, D_info_Type]:
        """
        Generate a sparse dictionary matrix for a single word function.
        
        Creates a dictionary matrix using multi processing to enumerate all parameter combinations.

        Args:
            word_fun: Word function object or string name of function to generate atoms.
            enumerate_args_list_dic: Dictionary of enumerable arguments with their value lists.
            unenumerate_args_list_dic: Dictionary of shared arguments for all enumerations.
            polarity: Optional polarity constraint for the dictionary segment ('nonneg', 'nonpos', etc.)
            dc_mono: Optional monotonicity constraint for the dictionary segment ('increase', 'decrease')

        Returns:
            Tuple (D_mat, info):
                D_mat: scipy.sparse matrix of the generated dictionary
                info: Metadata dictionary containing:
                    word_fun: Name of the word function used
                    enumerate_args_list_dic: Copied input of enumerable arguments
                    unenumerate_args_list_dic: Copied input of shared arguments
                    D_length: Number of atoms in the dictionary
                    polarity: Applied polarity constraint
                    dc_mono: Applied monotonicity constraint
        """
        import itertools
        from concurrent.futures import ProcessPoolExecutor
        # string recommanded
        if type(word_fun)==str:
            word_fun = getattr(Word_Function,word_fun)
        unenumerate_args_list_dic['Ns']=self.Ns
        unenumerate_args_list_dic['T']=self.T
        unenumerate_args_list_dic['s']=self.s
        # Get the size of the dictionary matrix to be created
        D_len = 1
        for temp_list in enumerate_args_list_dic.values():
            D_len = D_len * len(temp_list)
        temp_input_dic = {}
        for temp_key in enumerate_args_list_dic.keys():
            temp_input_dic[temp_key]=enumerate_args_list_dic[temp_key][0]
        _,temp_W_shape = word_fun(**unenumerate_args_list_dic, **temp_input_dic)
        D_width = temp_W_shape[0]
        # create D_mat
        #D_mat = np.zeros((D_width,D_len))
        D_mat_shape = (D_width,D_len)
        # enumerate
        args_list_list = list(enumerate_args_list_dic.values())
        args_key_list = list(enumerate_args_list_dic.keys())
        # Reserve space from D_len
        if 'Ti_list' in list(unenumerate_args_list_dic.keys()):
            reserved_length = len(unenumerate_args_list_dic['Ti_list'])*D_len*2
        else:
            self.logger.error(f'Ti_list needed')
        # Start enumerating all atoms
        time0 = time.time()
        self.logger.info(f'Start enumerating all atoms...')
        D_mat_row_col_val_array = np.zeros((reserved_length,3))
        time1 = time.time()
        self.logger.info(f'Creating the input parameter list...')
        # Multi-process acceleration
        # Get a list of all input parameters and their corresponding indices
        temp_worker_arg_list = []
        for i,temp_arg_tuple in enumerate(itertools.product(*args_list_list)):
            temp_worker_arg_list.append((i,dict(zip(args_key_list, temp_arg_tuple))))
        self.logger.info(f'Creating the input parameter list success. Wall time: {time.time()-time1:.2f}s')
        time2 = time.time()
        self.logger.info(f'Start multi-process running')
        worker_args = []
        for arg in temp_worker_arg_list:
            worker_args.append((word_fun, unenumerate_args_list_dic, arg))
        self.logger.info(f'Further creating parameter lists. Wall time: {time.time()-time2:.2f}s')
        with ProcessPoolExecutor() as executor:
            mp_result = executor.map(Word_Function.worker_for_get_dictionary, worker_args, chunksize=1024)
        self.logger.info(f'multi-process running success. Wall time: {time.time()-time2:.2f}s')
        time3 = time.time()
        self.logger.info(f'Start assembling Dictionary')
        total_len = 0
        start_idx = 0
        end_idx = 0
        for mp_result_i in mp_result:
            if mp_result_i is not None:
                total_len += mp_result_i.shape[0]
                end_idx = start_idx + mp_result_i.shape[0]
                D_mat_row_col_val_array[start_idx:end_idx] = mp_result_i
            start_idx = end_idx
        D_mat_row_col_val_array = D_mat_row_col_val_array[0:total_len]
        self.logger.info(f'Assembling dictionary success. Wall time: {time.time()-time3:.2f}s')
        self.logger.info(f'Total length of non-zero value={total_len}. Total wall time: {time.time()-time0:.2f}s')
        # Creating a sparse matrix(array)
        rows = list(D_mat_row_col_val_array[:,0])
        cols = list(D_mat_row_col_val_array[:,1])
        vals = list(D_mat_row_col_val_array[:,2])
        self.D_mat_row_col_val_array = D_mat_row_col_val_array
        D_mat = scipy.sparse.csr_array((vals, (rows, cols)), shape=D_mat_shape)
        # dictionary info
        info = {
            "word_fun": word_fun.__name__,
            "enumerate_args_list_dic": enumerate_args_list_dic,
            "unenumerate_args_list_dic": unenumerate_args_list_dic,
            "D_length": D_len,
            "polarity": polarity,
            "dc_mono": dc_mono
        }
        return D_mat,info
    def _if_problem_built(self, kwargs_for_filter_problems: dict, arg_lists_use_cp_parameter: list) -> int | None:
        # filter_problems
        for i,temp_problem in enumerate(self.problems):
            if temp_problem['kwargs_for_build_problem']['arg_lists_use_cp_parameter']==arg_lists_use_cp_parameter:
                da = {}
                for k,v in temp_problem['kwargs_for_build_problem'].items():
                    if k not in ['arg_lists_use_cp_parameter']+temp_problem['kwargs_for_build_problem']['arg_lists_use_cp_parameter']:
                        if k in ['enumerate_args_list_dic_list','unenumerate_args_list_dic_list']:
                            temp_v = v.copy()
                            for temp_v_idx,temp_vi in enumerate(temp_v):
                                if type(temp_vi)==dict:
                                    temp_temp_vi = temp_vi.copy()
                                    for tv_k,tv_v in temp_temp_vi.items():
                                        if type(tv_v)==np.ndarray:
                                            temp_temp_vi[tv_k]=tv_v.tolist()
                                    temp_v[temp_v_idx]=temp_temp_vi
                            da[k] = temp_v
                        else:
                            da[k] = v.tolist() if type(v)==np.ndarray else v
                db = {}
                for k,v in kwargs_for_filter_problems.items():
                    if k not in ['arg_lists_use_cp_parameter']+arg_lists_use_cp_parameter:
                        if k in ['enumerate_args_list_dic_list','unenumerate_args_list_dic_list']:
                            temp_v = v.copy()
                            for temp_v_idx,temp_vi in enumerate(temp_v):
                                if type(temp_vi)==dict:
                                    temp_temp_vi = temp_vi.copy()
                                    for tv_k,tv_v in temp_temp_vi.items():
                                        if type(tv_v)==np.ndarray:
                                            temp_temp_vi[tv_k]=tv_v.tolist()
                                    temp_v[temp_v_idx]=temp_temp_vi
                            db[k] = temp_v
                        else:
                            db[k] = v.tolist() if type(v)==np.ndarray else v
                #da = {k:v for k,v in temp_problem['kwargs_for_build_problem'].items() if k not in ['arg_lists_use_cp_parameter']+temp_problem['kwargs_for_build_problem']['arg_lists_use_cp_parameter']}
                #db = {k:v for k,v in kwargs_for_filter_problems.items() if k not in ['arg_lists_use_cp_parameter']+arg_lists_use_cp_parameter}
                if da == db and temp_problem['problem']:
                    return i
        return None
    def _if_D_builed(self, kwargs_for_build_problem) -> None | int:
        # Search created dictionaries
        param_list_for_filter_D = ['Ti_list','word_fun_list','enumerate_args_list_dic_list','unenumerate_args_list_dic_list','polarity_list','dc_mono_list']
        da = {}
        for k,v in kwargs_for_build_problem.items():
            if k in param_list_for_filter_D:
                temp_v = v.copy()
                for temp_v_idx,temp_vi in enumerate(temp_v):
                    if type(temp_vi)==dict:
                        temp_temp_vi = temp_vi.copy()
                        for tv_k,tv_v in temp_temp_vi.items():
                            if type(tv_v)==np.ndarray:
                                temp_temp_vi[tv_k]=tv_v.tolist()
                        temp_v[temp_v_idx]=temp_temp_vi
                da[k] = temp_v
        #da = {k:v for k,v in kwargs_for_build_problem.items() if k in ['word_fun_list','enumerate_args_list_dic_list','unenumerate_args_list_dic_list','polarity_list','dc_mono_list']}
        for i,tp in enumerate(self.problems):
            db = {}
            for k,v in tp['kwargs_for_build_problem'].items():
                if k in param_list_for_filter_D:
                    temp_v = v.copy()
                    for temp_v_idx,temp_vi in enumerate(temp_v):
                        if type(temp_vi)==dict:
                            temp_temp_vi = temp_vi.copy()
                            for tv_k,tv_v in temp_temp_vi.items():
                                if type(tv_v)==np.ndarray:
                                    temp_temp_vi[tv_k]=tv_v.tolist()
                            temp_v[temp_v_idx]=temp_temp_vi
                    db[k] = temp_v
            #db = {k:v for k,v in tp['kwargs_for_build_problem'].items() if k in ['word_fun_list','enumerate_args_list_dic_list','unenumerate_args_list_dic_list','polarity_list','dc_mono_list']}
            if da == db:
                return i
        return None
    def _create_polarity_constraint(self,D_len_list,dc_mn_list,polar_list,
                                    enumerate_args_list_dic_list,unenumerate_args_list_dic_list,
                                    polarity, X) -> List[Any]:
        # Create rows, columns, and values of the defect polarity sparse constraint matrix
        start_idx = 0
        constraints = []
        for seg_len, temp_dc_mono, temp_pol, temp_enumerate_args_list_dic, temp_unenumerate_args_list_dic in zip(D_len_list,dc_mn_list,polar_list,enumerate_args_list_dic_list,unenumerate_args_list_dic_list):
            end_idx = start_idx + seg_len
            # Check whether Bspline input parameters are correct
            if 'B_sp_basis_i' in list(temp_enumerate_args_list_dic.keys()):
                if list(temp_enumerate_args_list_dic.keys())[-1] != 'B_sp_basis_i':
                    self.logger.error(f"{self.method_name} solve: parameter 'B_sp_basis_i' should be placed at the end of the enumerate_args_dictionary")
                    break
                elif ('B_sp_n' not in list(temp_unenumerate_args_list_dic.keys())) or ('B_sp_degree' not in list(temp_unenumerate_args_list_dic.keys())):
                    self.logger.error(f"{self.method_name} solve: parameter 'B_sp_n' and 'B_sp_degree' should be placed at unenumerate_args_dictionary")
                    break
                elif len(temp_enumerate_args_list_dic['B_sp_basis_i']) != temp_unenumerate_args_list_dic['B_sp_n']:
                    self.logger.error(f"{self.method_name} solve: should be 'B_sp_basis_i = np.arange(B_sp_n)'")
                # If using spline interpolation. If monotonicity is defined, constrain X directly, otherwise constrain the polarity of the linear combination of the corresponding spline functions
                if (temp_dc_mono in ['increase','decrease']):
                    # Directly constrain X
                    segment = X[start_idx:end_idx]
                    if temp_pol in ['nonneg','nonnegative']:
                        constraints.append(segment >= 0)
                    elif temp_pol in ['nonpos','nonpositive']:
                        constraints.append(segment <= 0)
                    elif temp_pol in ['pos','positive']:
                        constraints.append(segment > 0)
                    elif temp_pol in ['neg','negative']:
                        constraints.append(segment < 0)
                    else:
                        if polarity == 'positive':
                            constraints.append(segment >= 0)
                        elif polarity == 'negative':
                            constraints.append(segment <= 0)
                else:
                    # For arbitrary monotonicity, the polarity of the corresponding B-spline function basis function itself needs to satisfy the polarity condition
                    # Generate B-spline basis functions with shape=(len(Ti_list),B_size)
                    B_size = temp_unenumerate_args_list_dic['B_sp_n']
                    len_x = len(temp_unenumerate_args_list_dic['Ti_list'])
                    temp_B_degree = temp_unenumerate_args_list_dic['B_sp_degree']
                    temp_x = np.linspace(0,1,len_x)
                    knots = np.linspace(0, 1, B_size - temp_B_degree + 1)
                    knots = np.r_[(0,)*temp_B_degree, knots, (1,)*temp_B_degree]
                    temp_B = np.zeros((len_x, B_size))
                    for j in range(B_size):
                        coefs = np.zeros(B_size)
                        coefs[j] = 1.0
                        temp_B[:, j] = scipy.interpolate.BSpline(knots, coefs, temp_B_degree)(temp_x)
                    if temp_pol in ['nonneg','nonnegative']:
                        constraints.append(temp_B @ cp.reshape(X[start_idx:end_idx], (B_size,-1), order='F') >= 0)
                    elif temp_pol in ['nonpos','nonpositive']:
                        constraints.append(temp_B @ cp.reshape(X[start_idx:end_idx], (B_size,-1), order='F') <= 0)
                    elif temp_pol in ['pos','positive']:
                        constraints.append(temp_B @ cp.reshape(X[start_idx:end_idx], (B_size,-1), order='F') > 0)
                    elif temp_pol in ['neg','negative']:
                        constraints.append(temp_B @ cp.reshape(X[start_idx:end_idx], (B_size,-1), order='F') < 0)
                    else:
                        if polarity == 'positive':
                            constraints.append(temp_B @ cp.reshape(X[start_idx:end_idx], (B_size,-1), order='F') >= 0)
                        elif polarity == 'negative':
                            constraints.append(temp_B @ cp.reshape(X[start_idx:end_idx], (B_size,-1), order='F') <= 0)
            # If do not use spline interpolation
            else:
                segment = X[start_idx:end_idx]
                if temp_pol in ['nonneg','nonnegative']:
                    constraints.append(segment >= 0)
                elif temp_pol in ['nonpos','nonpositive']:
                    constraints.append(segment <= 0)
                elif temp_pol in ['pos','positive']:
                    constraints.append(segment > 0)
                elif temp_pol in ['neg','negative']:
                    constraints.append(segment < 0)
                else:
                    if polarity == 'positive':
                        constraints.append(segment >= 0)
                    elif polarity == 'negative':
                        constraints.append(segment <= 0)
            start_idx = end_idx
        return constraints
    def _create_monotonicity_constraint(self,D_len_list,dc_mn_list,polar_list,
                                        enumerate_args_list_dic_list,unenumerate_args_list_dic_list,
                                        X) -> List[Any]:
        # dC monotonicity constraints (only when using Bspline)
        mono_constraints = []
        start_idx = 0
        for seg_len, temp_dc_mono, temp_pol, temp_enumerate_args_list_dic, temp_unenumerate_args_list_dic in zip(D_len_list,dc_mn_list,polar_list,enumerate_args_list_dic_list,unenumerate_args_list_dic_list):
            # If need to use spline interpolation
            if 'B_sp_basis_i' in list(temp_enumerate_args_list_dic.keys()):
                if list(temp_enumerate_args_list_dic.keys())[-1] != 'B_sp_basis_i':
                    self.logger.error(f"{self.method_name} solve: parameter 'B_sp_basis_i' should be placed at the end of the enumerate_args_dictionary")
                    break
                elif ('B_sp_n' not in list(temp_unenumerate_args_list_dic.keys())) or ('B_sp_degree' not in list(temp_unenumerate_args_list_dic.keys())):
                    self.logger.error(f"{self.method_name} solve: parameter 'B_sp_n' and 'B_sp_degree' should be placed at unenumerate_args_dictionary")
                    break
                elif len(temp_enumerate_args_list_dic['B_sp_basis_i']) != temp_unenumerate_args_list_dic['B_sp_n']:
                    self.logger.error(f"{self.method_name} solve: should be 'B_sp_basis_i = np.arange(B_sp_n)'")
                B_size = temp_unenumerate_args_list_dic['B_sp_n']
                end_idx = start_idx + seg_len
                for sub_i in range(int(seg_len/B_size)):
                    sub_start_idx = start_idx + sub_i * B_size
                    # Build matrix rows for each monotonicity constraint
                    for k in range(B_size-1):
                        idx1 = sub_start_idx + k
                        idx2 = sub_start_idx + k + 1
                        # Determine coefficients based on polarity and monotonicity
                        idx_of_constraints = len(mono_constraints)
                        if ((temp_pol in ['nonneg', 'pos', 'positive']) and (temp_dc_mono == 'increase')) or \
                           ((temp_pol in ['nonpos', 'neg', 'negative']) and (temp_dc_mono == 'decrease')):
                            # x_i - x_{i+1} <= 0
                            mono_constraints.append((idx_of_constraints, idx1, 1.0))
                            mono_constraints.append((idx_of_constraints, idx2, -1.0))
                        elif ((temp_pol in ['nonneg', 'pos', 'positive']) and (temp_dc_mono == 'decrease')) or \
                             ((temp_pol in ['nonpos', 'neg', 'negative']) and (temp_dc_mono == 'increase')):
                            # -x_i + x_{i+1} <= 0
                            mono_constraints.append((idx_of_constraints, idx1, -1.0))
                            mono_constraints.append((idx_of_constraints, idx2, 1.0))
            else:
                # If do not use spline interpolation
                # Do not make increasing or decreasing constraints (constant situation)
                break
            start_idx = end_idx
        # If there is a monotonicity constraint, construct the sparse matrix(array) form
        if mono_constraints:
            rows = []
            cols = []
            vals = []
            for (i, j, v) in mono_constraints:
                rows.append(i)
                cols.append(j)
                vals.append(v)
            n_mono_constr = max(rows) + 1 if rows else 0
            Cs = scipy.sparse.csr_array((vals, (rows, cols)), shape=(n_mono_constr, X.shape[0]))
            #constraints.append(Cs @ X <= 0)
            return [Cs @ X <= 0]
        else:
            return []
    def _create_trap_pattern_index_list(self,D_len_list,dc_mn_list,constraint_prefactor_list,
                                        enumerate_args_list_dic_list,unenumerate_args_list_dic_list,
                                        enable_irls_mode,irls_weight,X) -> tuple:
        # The 1-norm of B_spline should be the maximum value after the linear combination of basis functions
        max_dC_index_list = []
        all_constraint_prefactor_list = []
        x_one_term = 0
        start_idx = 0
        for seg_len, temp_dc_mono, temp_enumerate_args_list_dic, temp_unenumerate_args_list_dic, temp_constraint_prefactor in \
        zip(D_len_list,dc_mn_list,enumerate_args_list_dic_list,unenumerate_args_list_dic_list,constraint_prefactor_list):
            B_size = temp_unenumerate_args_list_dic['B_sp_n']
            end_idx = start_idx + seg_len
            if 'B_sp_basis_i' in list(temp_enumerate_args_list_dic.keys()):
                # If need to use B-spline
                first_index_list = np.arange(int(seg_len/B_size)) * B_size + start_idx
                last_index_list = first_index_list + B_size - 1
                # For Bspline, since the increase or decrease is restricted, its maximum value is exactly the coefficient of the boundary node
                if temp_dc_mono == 'increase':
                    # last term
                    max_dC_index_list.append(last_index_list)
                    all_constraint_prefactor_list.append(temp_constraint_prefactor*np.ones_like(last_index_list))
                elif temp_dc_mono == 'decrease':
                    # first term
                    max_dC_index_list.append(first_index_list)
                    all_constraint_prefactor_list.append(temp_constraint_prefactor*np.ones_like(last_index_list))
                else:
                    # Generate B-spline basis functions with shape=(len(Ti_list),B_size)
                    len_x = len(temp_unenumerate_args_list_dic['Ti_list'])
                    temp_B_degree = temp_unenumerate_args_list_dic['B_sp_degree']
                    temp_x = np.linspace(0,1,len_x)
                    knots = np.linspace(0, 1, B_size - temp_B_degree + 1)
                    knots = np.r_[(0,)*temp_B_degree, knots, (1,)*temp_B_degree]
                    temp_B = np.zeros((len_x, B_size))
                    for j in range(B_size):
                        coefs = np.zeros(B_size)
                        coefs[j] = 1.0
                        temp_B[:, j] = scipy.interpolate.BSpline(knots, coefs, temp_B_degree)(temp_x)
                    # The maximum value of each column -> axis=0
                    if enable_irls_mode:
                        x_one_term = x_one_term + cp.sum(cp.max(cp.abs(temp_B @ cp.reshape(cp.multiply(irls_weight[start_idx:end_idx],temp_constraint_prefactor*X[start_idx:end_idx]), (B_size,-1), order='F')), axis=0))
                    else:
                        x_one_term = x_one_term + cp.sum(cp.max(cp.abs(temp_B @ cp.reshape(temp_constraint_prefactor*X[start_idx:end_idx], (B_size,-1), order='F')), axis=0))
            else:
                # If do not use spline interpolation
                all_index_list = np.arange(int(seg_len)) + start_idx
                max_dC_index_list.append(all_index_list)
                all_constraint_prefactor_list.append(temp_constraint_prefactor*np.ones_like(all_index_list))
            start_idx = end_idx
        max_dC_index_list = np.concatenate(max_dC_index_list).reshape(-1)
        all_constraint_prefactor_list = np.concatenate(all_constraint_prefactor_list).reshape(-1)
        return max_dC_index_list,all_constraint_prefactor_list,x_one_term
    
    def _get_new_input_after_remove_large_irls_weight(self,irls_weight,large_irls_weight_threshold,
                                                      D_info_list,word_fun_list,enumerate_args_list_dic_list,
                                                      unenumerate_args_list_dic_list,constraint_prefactor_list,
                                                      D):
        # every trap pattern's irls weight should be same, so that delete them together here
        nonzero_mask_for_irls = np.where(np.array(irls_weight)<large_irls_weight_threshold)[0]
        D = D.tocsc()[:,nonzero_mask_for_irls]
        irls_weight = irls_weight[nonzero_mask_for_irls]
        # change D_info_list
        _new_D_info_list = D_info_list.copy()
        _new_word_fun_list = word_fun_list.copy()
        _new_enumerate_args_list_dic_list = enumerate_args_list_dic_list.copy()
        _new_unenumerate_args_list_dic_list = unenumerate_args_list_dic_list.copy()
        _new_constraint_prefactor_list = constraint_prefactor_list.copy()
        _start_idx = 0
        _del_idx_list = []
        for i,Di in enumerate(D_info_list):
            _end_idx = _start_idx + Di['D_length']
            _new_D_length = len(np.where((nonzero_mask_for_irls>=_start_idx)&(nonzero_mask_for_irls<_end_idx))[0])
            if _new_D_length == 0:
                _del_idx_list.append(i)
            else:
                _new_D_info_list[i]['D_length'] = _new_D_length
            _start_idx = _end_idx
        for i in _del_idx_list[::-1]:
            del _new_D_info_list[i]
            del _new_word_fun_list[i]
            del _new_enumerate_args_list_dic_list[i]
            del _new_unenumerate_args_list_dic_list[i]
            del _new_constraint_prefactor_list[i]
        return D,irls_weight,_new_D_info_list,_new_word_fun_list,_new_enumerate_args_list_dic_list,\
            _new_unenumerate_args_list_dic_list,_new_constraint_prefactor_list

    def _build_problem(self, kwargs_for_build_problem: dict, remove_large_irls_weight: bool = False,
                       large_irls_weight_threshold: float = 1e7) -> None:
        """
        Construct the convex optimization problem using CVXPY.
        
        Builds the convex problem with L1 regularization and constraint definitions.

        Args:
            kwargs_for_build_problem: Dictionary containing:
                Ti_list: List of temperature indices to analyze
                polarity: Global polarity constraint
                lambda1: Regularization strength
                enable_irls_mode: Flag for Iteratively Reweighted Least Squares mode
                irls_weight: Weight vector for IRLS
                word_fun_list: Dictionary word functions
                enumerate_args_list_dic_list: Enumerable arguments for dictionaries
                unenumerate_args_list_dic_list: Shared arguments for dictionaries
                polarity_list: Per-segment polarity constraints
                dc_mono_list: Per-segment monotonicity constraints
                data_weight_fun_T (Callable): weight function of data with different temperatures
                arg_lists_use_cp_parameter: List of parameter names to be treated as CVXPY parameters
        
        Update Attributes:
            problems
        """
        # extract params needed
        Ti_list = np.array(kwargs_for_build_problem['Ti_list'])
        polarity = kwargs_for_build_problem['polarity']
        lambda1 = kwargs_for_build_problem['lambda1']
        enable_irls_mode = kwargs_for_build_problem['enable_irls_mode']
        irls_weight = np.array(kwargs_for_build_problem['irls_weight'])
        word_fun_list = kwargs_for_build_problem['word_fun_list']
        enumerate_args_list_dic_list = kwargs_for_build_problem['enumerate_args_list_dic_list']
        unenumerate_args_list_dic_list = kwargs_for_build_problem['unenumerate_args_list_dic_list']
        polarity_list = kwargs_for_build_problem['polarity_list']
        dc_mono_list = kwargs_for_build_problem['dc_mono_list']
        constraint_prefactor_list = kwargs_for_build_problem['constraint_prefactor_list']
        if type(kwargs_for_build_problem['data_weight_fun_T']) == str:
            data_weight_fun_T = eval(kwargs_for_build_problem['data_weight_fun_T'])
        else:
            data_weight_fun_T = kwargs_for_build_problem['data_weight_fun_T']
        data_weight_fun_T = np.vectorize(data_weight_fun_T)
        term1_constraint_type = kwargs_for_build_problem['term1_constraint_type']
        arg_lists_use_cp_parameter = kwargs_for_build_problem['arg_lists_use_cp_parameter']
        constraint_max_rms = kwargs_for_build_problem['constraint_max_rms']

        temp_dic = {}
        for ivar in arg_lists_use_cp_parameter:
            if type(kwargs_for_build_problem[ivar])==np.ndarray:
                temp_dic[ivar] = cp.Parameter(kwargs_for_build_problem[ivar].shape,nonneg=True)
            elif type(kwargs_for_build_problem[ivar]) in [int,float]:
                temp_dic[ivar] = cp.Parameter(nonneg=True)
            temp_dic[ivar].value = kwargs_for_build_problem[ivar]

        if 'lambda1' in temp_dic.keys():
            lambda1 = temp_dic['lambda1']
        if 'irls_weight' in temp_dic.keys():
            irls_weight = temp_dic['irls_weight']
        
        C = self.C[Ti_list,:]
        NT = C.shape[0]
        C_average = np.tile(np.average(C.T,axis=0).reshape(1,NT),(self.Ns,1))
        C_weight = np.abs(data_weight_fun_T(self.T[Ti_list]))
        reg1 = lambda1

        # Search created dictionaries
        D_idx = self._if_D_builed(kwargs_for_build_problem)
        # create Dictionary
        #if D_built:
        if D_idx is not None:
            D = self.problems[D_idx]['D']
            D_info_list = self.problems[D_idx]['D_info_list']
        else:
            D,D_info_list = self.generate_dictionary_stack_sparse(
                word_fun_list=word_fun_list,
                enumerate_args_list_dic_list=enumerate_args_list_dic_list,
                unenumerate_args_list_dic_list=unenumerate_args_list_dic_list,
                polarity_list=polarity_list,
                dc_mono_list=dc_mono_list
            )

        if remove_large_irls_weight:
            if irls_weight.shape == ():
                self.logger.warning(f"{self.method_name} _build_problem: irls_weight is a single value, cannot remove large irls weight items.")
            else:
                D,irls_weight,D_info_list, word_fun_list, enumerate_args_list_dic_list, \
                    unenumerate_args_list_dic_list, constraint_prefactor_list = \
                        self._get_new_input_after_remove_large_irls_weight(
                            irls_weight=irls_weight,
                            large_irls_weight_threshold=large_irls_weight_threshold,
                            D_info_list=D_info_list,
                            word_fun_list=word_fun_list,
                            enumerate_args_list_dic_list=enumerate_args_list_dic_list,
                            unenumerate_args_list_dic_list=unenumerate_args_list_dic_list,
                            constraint_prefactor_list=constraint_prefactor_list,
                            D=D
                        )

        # (Nb,Na)
        X = cp.Variable(D.shape[1])
        # f
        f = cp.Variable((self.Ns+1,NT))
        residual = self.A_extended @ f - C.T
        D_times_X = D @ X
        constraints = [f[1:,:].T == cp.reshape(D_times_X, (NT,self.Ns), order='C')]
        D_len_list = []
        polar_list = []
        dc_mn_list = []
        for Di in D_info_list:
            D_len_list.append(Di['D_length'])
            polar_list.append(Di['polarity'])
            dc_mn_list.append(Di['dc_mono'])

        constraints += self._create_polarity_constraint(
            D_len_list=D_len_list,
            dc_mn_list=dc_mn_list,
            polar_list=polar_list,
            enumerate_args_list_dic_list=enumerate_args_list_dic_list,
            unenumerate_args_list_dic_list=unenumerate_args_list_dic_list,
            polarity=polarity,
            X=X
        )

        constraints += self._create_monotonicity_constraint(
            D_len_list=D_len_list,
            dc_mn_list=dc_mn_list,
            polar_list=polar_list,
            enumerate_args_list_dic_list=enumerate_args_list_dic_list,
            unenumerate_args_list_dic_list=unenumerate_args_list_dic_list,
            X=X
        )

        # This item may be scaled based on the SNR of each temperature???
        term1 = self._get_term1(residual, C_weight,constraint_max_rms=constraint_max_rms,
                                term1_constraint_type=term1_constraint_type, NT=NT)
        
        # The 1-norm of B_spline should be the maximum value after the linear combination of basis functions
        max_dC_index_list, all_constraint_prefactor_list, x_one_term = self._create_trap_pattern_index_list(
            D_len_list=D_len_list,
            dc_mn_list=dc_mn_list,
            enumerate_args_list_dic_list=enumerate_args_list_dic_list,
            unenumerate_args_list_dic_list=unenumerate_args_list_dic_list,
            constraint_prefactor_list=constraint_prefactor_list,
            enable_irls_mode=enable_irls_mode,
            irls_weight=irls_weight,
            X=X
        )

        # The one-norm of the X (when X is a matrix, cp.norm(X,1) is not 1 norm)
        if enable_irls_mode:
            term2 = reg1 * (cp.sum(cp.abs(cp.multiply(irls_weight[max_dC_index_list],cp.multiply(all_constraint_prefactor_list,X[max_dC_index_list])))) + x_one_term)
        else:
            term2 = reg1 * (cp.sum(cp.abs(cp.multiply(all_constraint_prefactor_list,X[max_dC_index_list]))) + x_one_term)
        objective = cp.Minimize(term1 + term2)
        problem = cp.Problem(objective, constraints)
        # clear problem with parameter=[]
        for i,tp in enumerate(self.problems):
            if tp['parameter']==[]:
                del self.problems[i]
        local_vars = locals()
        self.problems.append({
            'problem':problem,
            'parameter':{k:local_vars[k] for k in arg_lists_use_cp_parameter},
            'variable':{k:local_vars[k] for k in ['f','X','residual','term1','term2']},
            'kwargs_for_build_problem':kwargs_for_build_problem,
            'D_info_list':D_info_list,
            'D':D
        })
        del problem

    def solve(
        self, Ti_list: List[int], word_fun_list: list, enumerate_args_list_dic_list: list, polarity_list: list,
        unenumerate_args_list_dic_list: list, dc_mono_list: list, constraint_prefactor_list: list,
        polarity: str = 'positive', verbose: bool = False, lambda1: float = 1e-5,
        skip_solved: bool = True, enable_irls_mode: bool = False, irls_weight: np.ndarray | float | int = 1,
        solver: str = 'CVXOPT', solver_params: dict = {}, data_weight_fun_T: Callable | str = "lambda T: 1",
        arg_lists_use_cp_parameter: list = ['lambda1'],
        constraint_max_rms: bool = True, term1_constraint_type: str = 'L2',
        remove_large_irls_weight: bool = True, large_irls_weight_threshold: float = 1e7
    ) -> None | LDLTS_SolveResultType:
        """
        Solve the DLTS optimization problem with given parameters.
        
        Executes the optimization process using specified solver and parameters.

        Args:
            Ti_list: List of temperature indices to analyze.
            polarity: Global polarity constraint ('positive' or 'negative').
            verbose: Print solver progress messages.
            lambda1: Regularization strength.
            skip_solved: Skip solving if identical problem has been solved.
            enable_irls_mode: Enable Iteratively Reweighted Least Squares mode.
            irls_weight: Weight vector for IRLS.
            word_fun_list: List of word functions for dictionary generation.
            enumerate_args_list_dic_list: List of enumerable argument dictionaries.
            unenumerate_args_list_dic_list: List of shared argument dictionaries.
            polarity_list: Per-segment polarity constraints.
            dc_mono_list: Per-segment monotonicity constraints.
            solver: Name of CVXPY solver to use.
            solver_params: Additional parameters for the solver.
            data_weight_fun_T (Callable | str): weight function of data with different temperature.
                (lambda function is unpicklable, use str in this case like "lambda T: 1", this will be evaled inside).
            arg_lists_use_cp_parameter: Parameters to be treated as CVXPY parameters.

        Returns:
            None | LDLTS_SolveResultType

        Raises:
            Logs detailed errors through class logger.
        """
        local_vars = locals()
        input_kwargs = {k:v for k,v in local_vars.items() if k not in ['self']}
        kwargs_for_filter_solved = {}
        for k,v in input_kwargs.items():
            if k not in ['verbose','skip_solved','remove_large_irls_weight','large_irls_weight_threshold']:
                if type(v) == np.ndarray:
                    kwargs_for_filter_solved[k]=v.tolist()
                else:
                    kwargs_for_filter_solved[k]=v
        kwargs_for_filter_problems = {}
        for k,v in input_kwargs.items():
            if k not in ['verbose','skip_solved','solver','solver_params','remove_large_irls_weight','large_irls_weight_threshold']:
                if type(v) == np.ndarray:
                    kwargs_for_filter_problems[k]=v.tolist()
                else:
                    kwargs_for_filter_problems[k]=v
        # filter_solved
        if skip_solved and not enable_irls_mode:
            skipsolve = False
            for temp_solve_result in self.solve_history:
                if 'input_params' in temp_solve_result:
                    if self._dict_compare(temp_solve_result['input_params'], kwargs_for_filter_solved):
                        skipsolve = True
                        break
            if skipsolve:
                self.logger.info(f'{self.method_name} solve: Problem already solved, skip')
                return None

        if np.max(Ti_list)>=self.NT or np.min(Ti_list)<0:
            self.logger.error(f'{self.method_name} solve: Ti_list out of range')
            return None
        elif type(Ti_list)==list:
            Ti_list = np.array(Ti_list)
        # filter_problems
        problem_idx = self._if_problem_built(kwargs_for_filter_problems, arg_lists_use_cp_parameter)
        if problem_idx is not None:
            try:
                if verbose:
                    self.logger.info(f"{self.method_name} solve: problem already be built, problem_idx={problem_idx}")
                # update parameter
                for pi in arg_lists_use_cp_parameter:
                    self.problems[problem_idx]['parameter'][pi].value = input_kwargs[pi]
                    # update corresponding kwargs_for_build_problem
                    if pi in self.problems[problem_idx]['kwargs_for_build_problem']:
                        self.problems[problem_idx]['kwargs_for_build_problem'][pi] = input_kwargs[pi]
            except Exception as e:
                self.logger.error(f"{self.method_name} solve: update parameter failed:", e)
        else:
            try:
                if verbose:
                    self.logger.info(f"{self.method_name} solve: build new problem")
                self._build_problem(kwargs_for_filter_problems, remove_large_irls_weight=remove_large_irls_weight, large_irls_weight_threshold=large_irls_weight_threshold)
                problem_idx = -1
            except Exception as e:
                self.logger.error(f"{self.method_name} solve: build problem failed:", e)

        import time
        solve_start_time = time.time()
        with self._capture_all_logs(self.logger, pre_fix='#CVXPY Output#: '):
            try:
                # solve
                self.problems[problem_idx]['problem'].solve(solver=getattr(cp,solver), verbose=verbose, **solver_params)
            except Exception as e:
                self.logger.error(f"{self.method_name} solve: solve problem failed:", e)
        solve_end_time = time.time()

        if self.problems[problem_idx]['problem'].status == cp.OPTIMAL:
            _add_kwargs = {
                'input_params':kwargs_for_filter_solved,
                'wall_time':solve_end_time-solve_start_time
            }
            temp_result_dict = self._get_solve_result_by_problem_index(problem_idx, add_kwargs=_add_kwargs)
            if enable_irls_mode:
                return temp_result_dict
            else:
                self.solve_history.append(temp_result_dict)
        else:
            self.logger.error(f"{self.method_name} solve: fail to find solution, problem status:", self.problems[problem_idx]['problem'].status)
        return None
    
    def _get_solve_result_by_problem_index(self, problem_idx: int, add_kwargs: dict = {}) -> LDLTS_SolveResultType:
        f = self.problems[problem_idx]['variable']['f']
        X = self.problems[problem_idx]['variable']['X']
        residual = self.problems[problem_idx]['variable']['residual']
        term1 = self.problems[problem_idx]['variable']['term1']
        term2 = self.problems[problem_idx]['variable']['term2']
        Ti_list = np.array(self.problems[problem_idx]['kwargs_for_build_problem']['Ti_list'])
        lambda1 = self.problems[problem_idx]['kwargs_for_build_problem']['lambda1']
        enable_irls_mode = self.problems[problem_idx]['kwargs_for_build_problem']['enable_irls_mode']
        irls_weight = self.problems[problem_idx]['kwargs_for_build_problem']['irls_weight']
        enumerate_args_list_dic_list = self.problems[problem_idx]['kwargs_for_build_problem']['enumerate_args_list_dic_list']
        unenumerate_args_list_dic_list = self.problems[problem_idx]['kwargs_for_build_problem']['unenumerate_args_list_dic_list']
        f_current = f.value
        X_current = X.value
        # rms at each temperature
        rms_list = np.sqrt(np.sum(residual.value**2, axis=0)/self.Nt)
        term1_error = term1.value
        term2_error = term2.value
        # calc irls_target
        # irls_target depends on whether the B-spline function is used and the monotonicity setting
        D = self.problems[problem_idx]['D']
        D_info_list = self.problems[problem_idx]['D_info_list']
        irls_target = self._get_irls_target(
            D_info_list=D_info_list,
            enumerate_args_list_dic_list=enumerate_args_list_dic_list,
            unenumerate_args_list_dic_list=unenumerate_args_list_dic_list,
            X_current=X_current
        )
        temp_result_dict = {
            'input_params':{}, 'Ti_list':Ti_list, 'f':f_current, 'term1_error':term1_error, 'reg':lambda1,
            'term2_error':term2_error, 'X':X_current, 'D':D, 'D_info_list':D_info_list,
            'rms_list':rms_list, 'Kernel_A':self.A_extended, 'irls_target':irls_target,
            'irls_eval_error_dict':{'term1_error':term1_error,'term2_error':term2_error},
            'wall_time':0.0
        }
        # update input_params, wall_time and additional kwargs
        temp_result_dict.update(add_kwargs)
        if enable_irls_mode:
            temp_result_dict['irls_weight'] = irls_weight
        return temp_result_dict
    
    def _get_irls_target(self,D_info_list,enumerate_args_list_dic_list,unenumerate_args_list_dic_list,X_current):
        irls_target = X_current.copy()
        D_len_list = []
        polar_list = []
        dc_mn_list = []
        for Di in D_info_list:
            D_len_list.append(Di['D_length'])
            polar_list.append(Di['polarity'])
            dc_mn_list.append(Di['dc_mono'])
        start_idx = 0
        for seg_len, temp_dc_mono, temp_enumerate_args_list_dic, temp_unenumerate_args_list_dic in \
        zip(D_len_list,dc_mn_list,enumerate_args_list_dic_list,unenumerate_args_list_dic_list):
            B_size = temp_unenumerate_args_list_dic['B_sp_n']
            end_idx = start_idx + seg_len
            # Need to use B-spline interpolation
            if 'B_sp_basis_i' in list(temp_enumerate_args_list_dic.keys()):
                first_index_list = np.arange(int(seg_len/B_size)) * B_size + start_idx
                last_index_list = first_index_list + B_size - 1
                # For Bspline, since the increase or decrease is restricted, its maximum value is exactly the coefficient of the boundary node
                if temp_dc_mono == 'increase':
                    # last term
                    irls_target[start_idx:end_idx] = X_current[np.repeat(last_index_list, B_size)]
                elif temp_dc_mono == 'decrease':
                    # first term
                    irls_target[start_idx:end_idx] = X_current[np.repeat(first_index_list, B_size)]
                else:
                    # Generate B-spline basis functions with shape=(len(Ti_list),B_size)
                    len_x = len(temp_unenumerate_args_list_dic['Ti_list'])
                    temp_B_degree = temp_unenumerate_args_list_dic['B_sp_degree']
                    temp_x = np.linspace(0,1,len_x)
                    knots = np.linspace(0, 1, B_size - temp_B_degree + 1)
                    knots = np.r_[(0,)*temp_B_degree, knots, (1,)*temp_B_degree]
                    temp_B = np.zeros((len_x, B_size))
                    for j in range(B_size):
                        coefs = np.zeros(B_size)
                        coefs[j] = 1.0
                        temp_B[:, j] = scipy.interpolate.BSpline(knots, coefs, temp_B_degree)(temp_x)
                    # The maximum value of each column(axis=0)
                    irls_target[start_idx:end_idx] = np.repeat(np.max(np.abs(temp_B @ np.reshape(X_current[start_idx:end_idx], (B_size,-1), order='F')), axis=0), B_size)
            start_idx = end_idx
        return irls_target

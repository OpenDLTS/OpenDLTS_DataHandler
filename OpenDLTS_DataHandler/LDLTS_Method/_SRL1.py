__all__ = ['SRL1']

import cvxpy as cp
import numpy as np
from .._typing import *
from ._SR import SR
# LDLTS_METHOD_SR
class SRL1(SR):
    def _build_problem(self, kwargs_for_build_problem: dict, remove_large_irls_weight: bool = False,
                       large_irls_weight_threshold: float = 1e7) -> None:
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

        '''
        if remove_large_irls_weight:
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
        '''
        # (Nb,Na)
        X = cp.Variable(D.shape[1])
        # f
        f = cp.Variable((self.Ns+1,NT))
        # f_L1
        f_L1 = cp.Variable((self.Ns,NT))

        residual = self.A_extended @ f - C.T
        D_times_X = D @ X

        constraints = [f[1:,:].T == (cp.reshape(D_times_X, (NT,self.Ns), order='C') + f_L1.T)]
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

        if polarity in ['positive']:
            constraints += [f_L1>=0]
        elif polarity in ['negative']:
            constraints += [f_L1<=0]

        if polarity == 'both':
            if not enable_irls_mode:
                term2 = reg1 * cp.sum(cp.abs(f_L1))
            else:
                term2 = reg1 * cp.sum(cp.multiply(irls_weight,cp.abs(f_L1)))
        else:
            if not enable_irls_mode:
                if polarity in ['negative']:
                    term2 = reg1 * cp.sum(cp.abs(f_L1))
                else:
                    term2 = reg1 * cp.sum(f_L1)
            else:
                if polarity in ['negative']:
                    term2 = reg1 * cp.sum(cp.multiply(irls_weight,cp.abs(f_L1)))
                else:
                    term2 = reg1 * cp.sum(cp.multiply(irls_weight,f_L1))

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
            'variable':{k:local_vars[k] for k in ['f','f_L1','X','residual','term1','term2']},
            'kwargs_for_build_problem':kwargs_for_build_problem,
            'D_info_list':D_info_list,
            'D':D
        })
        del problem
    
    def _get_solve_result_by_problem_index(self, problem_idx: int, add_kwargs: dict = {}) -> LDLTS_SolveResultType:
        f = self.problems[problem_idx]['variable']['f']
        f_L1 = self.problems[problem_idx]['variable']['f_L1']
        X = self.problems[problem_idx]['variable']['X']
        residual = self.problems[problem_idx]['variable']['residual']
        term1 = self.problems[problem_idx]['variable']['term1']
        term2 = self.problems[problem_idx]['variable']['term2']
        Ti_list = np.array(self.problems[problem_idx]['kwargs_for_build_problem']['Ti_list'])
        lambda1 = self.problems[problem_idx]['kwargs_for_build_problem']['lambda1']
        enable_irls_mode = self.problems[problem_idx]['kwargs_for_build_problem']['enable_irls_mode']
        irls_weight = self.problems[problem_idx]['kwargs_for_build_problem']['irls_weight']
        #enumerate_args_list_dic_list = self.problems[problem_idx]['kwargs_for_build_problem']['enumerate_args_list_dic_list']
        #unenumerate_args_list_dic_list = self.problems[problem_idx]['kwargs_for_build_problem']['unenumerate_args_list_dic_list']
        f_current = f.value
        f_L1_current = f_L1.value
        X_current = X.value
        # rms at each temperature
        rms_list = np.sqrt(np.sum(residual.value**2, axis=0)/self.Nt)
        term1_error = term1.value
        term2_error = term2.value
        # calc irls_target
        # irls_target depends on whether the B-spline function is used and the monotonicity setting
        D = self.problems[problem_idx]['D']
        D_info_list = self.problems[problem_idx]['D_info_list']
        irls_target = f_L1_current.copy()
        temp_result_dict = {
            'input_params':{}, 'Ti_list':Ti_list, 'f':f_current, 'f_L1':f_L1_current, 'term1_error':term1_error, 'reg':lambda1,
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
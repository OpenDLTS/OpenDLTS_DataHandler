__all__ = ['L1']

import cvxpy as cp
import numpy as np
from .._Data_Loader import Data_Loader
from .._config import *
from .._typing import *
from ._Plotter_Base import PlotterBase

class L1(PlotterBase):
    def __init__(self, dlts_data: Data_Loader, Ns: int = 500, s0: float = 1e-1, s1: float = 1e5) -> None:
        """
        Init LDLTS_Method

        Args:
            dlts_data: OpenDLTS.DataHandler.Data_Loader object.
            Ns: Number of emission rate.
            s0: Minimum emission rate.
            s1: Maximum emission rate.
                
        Return:
            None

        Update Attributes:
            method_name: Method name from self.__class__.__name__
            _data_dict_for_ldlts: Data dictionary for LDLTS from dlts_data and method self._gen_kernel_mat.
                t: time array.
                T: temperature array.
                C: defect density array.
                Nt: Number of time points.
                NT: Number of temperature points.
                data_plot_dict: Data dictionary for plotting.
                data_scaling_factor: Data scaling factor.
                s: Emission rate array.
                Ns: Number of emission rate.
                A: Kernel matrix.
                A_extended: Extended kernel matrix.
                rms_limit: RMS limit.
            solve_history: Init solve history.
            lcurve_history: Init L-curve history.
            irls_solve_history: Init IRLS solve history.
            prsse_solve_history: Init PRSSE solve history.
            trap_group_rebuild_data: Init Trap group rebuild data.
            trap_group_rebuild_data_arrh: Init Trap group rebuild data for Arrhenius.
            temp_irls_solve_result: Init Temporary IRLS solve result.
            problems: Init CVXPY problems.
            logger: Logger from LOGGER_ODDH.getChild(self.method_name).
        """
        self.method_name = self.__class__.__name__
        # load data from Data_Loader
        self._data_dict_for_ldlts = {
            't':dlts_data.t,
            'T':dlts_data.T,
            'C':dlts_data.C,
            'Nt':len(dlts_data.t),
            'NT':len(dlts_data.T),
            'data_plot_dict':dlts_data.data_plot_dict,
            'data_scaling_factor':dlts_data.data_scaling_factor
        }
        for dk,dv in self._data_dict_for_ldlts.items():
            setattr(self,dk,dv)
        # gen_kernel_mat
        # self.s, self.Ns, self.A, self.A_extended
        self._gen_kernel_mat(self.t, Ns=Ns, s0=s0, s1=s1)
        self.rms_limit = 0.0
        self._data_dict_for_ldlts.update({
            's':self.s,
            'Ns':self.Ns,
            'A':self.A,
            'A_extended':self.A_extended,
            'rms_limit':self.rms_limit
        })
        self.solve_history = []
        self.lcurve_history = []
        self.irls_solve_history = []
        self.prsse_solve_history = []
        self.trap_group_rebuild_data = None
        self.trap_group_rebuild_data_arrh = None
        self.temp_irls_solve_result = None
        self.problems = []
        self._attr_names_for_save = ['method_name','_data_dict_for_ldlts','solve_history',\
                                     'lcurve_history','irls_solve_history','prsse_solve_history',\
                                     'trap_group_rebuild_data','trap_group_rebuild_data_arrh']
        self.logger = LOGGER_ODDH.getChild(self.method_name)

    def _gen_kernel_mat(self, t: np.ndarray, Ns: int, s0: float, s1: float) -> None:
        """
        Generate kernel matrix.

        Args:
            t: Temperature array.
            Ns: Number of emission rate.
            s0: Minimum emission rate.
            s1: Maximum emission rate.

        Return:
            None

        Update Attributes:
            s: Emission rate array.
            Ns: Number of emission rate.
            A: Kernel matrix.
            A_extended: Extended kernel matrix.
        """
        self.s = np.logspace(np.log10(s0), np.log10(s1), Ns, base=10.0)
        self.Ns = len(self.s)
        self.A = -np.exp(-t.reshape(-1, 1) * self.s.reshape(1, -1))
        self.A_extended = np.hstack([np.ones((len(t), 1)), self.A])

    def _del_unpicklable_attr(self) -> None:
        """
        Delete unpicklable attributes.
        """
        for tempproblem in self.problems:
            tempproblem['problem']=None
        for attr in ['arrh_plotter', 'ldlts_plotter', 'ldlts_T_plotter', 'trans_plotter']:
            if hasattr(self, attr):
                delattr(self, attr)

    def _if_problem_built(self, kwargs_for_filter_problems, arg_lists_use_cp_parameter) -> int | None:
        """
        Check if problem is already built.

        Args:
            kwargs_for_filter_problems (dict): Problem configuration including:
            Ti_list (ndarray): Temperature indices to process
        
        Return:
            int | None:
            if problem is already built, return index of problem in self.problems.
            else, return None.

        """
        # filter_problems
        for i,temp_problem in enumerate(self.problems):
            if temp_problem['kwargs_for_build_problem']['arg_lists_use_cp_parameter']==arg_lists_use_cp_parameter:
                da = {}
                for k,v in temp_problem['kwargs_for_build_problem'].items():
                    if k not in ['arg_lists_use_cp_parameter']+temp_problem['kwargs_for_build_problem']['arg_lists_use_cp_parameter']:
                        da[k] = v.tolist() if type(v)==np.ndarray else v
                db = {}
                for k,v in kwargs_for_filter_problems.items():
                    if k not in ['arg_lists_use_cp_parameter']+arg_lists_use_cp_parameter:
                        da[k] = v.tolist() if type(v)==np.ndarray else v
                #da = {k:v for k,v in temp_problem['kwargs_for_build_problem'].items() if k not in ['arg_lists_use_cp_parameter']+temp_problem['kwargs_for_build_problem']['arg_lists_use_cp_parameter']}
                #db = {k:v for k,v in kwargs_for_filter_problems.items() if k not in ['arg_lists_use_cp_parameter']+arg_lists_use_cp_parameter}
                if da == db and temp_problem['problem']:
                    return i
        return None
    def _dict_compare(self, dict1, dict2):
        """递归比较两个字典，支持np.ndarray的比较"""
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        for key in dict1:
            val1 = dict1[key]
            val2 = dict2[key]
            # 如果两个值都是np.ndarray
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.array_equal(val1, val2):
                    return False
            # 如果都是字典，递归比较
            elif isinstance(val1, dict) and isinstance(val2, dict):
                if not self._dict_compare(val1, val2):
                    return False
            # 如果都是列表
            elif isinstance(val1, list) and isinstance(val2, list):
                if not self._list_compare(val1, val2):
                    return False
            # 其他类型直接比较
            else:
                if val1 != val2:
                    return False
        return True
    def _list_compare(self, list1, list2):
        """递归比较两个字典，支持np.ndarray的比较"""
        if len(list1) != len(list2):
            return False
        for index,_ in enumerate(list1):
            val1 = list1[index]
            val2 = list2[index]
            # 如果两个值都是np.ndarray
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.array_equal(val1, val2):
                    return False
            # 如果都是字典
            elif isinstance(val1, dict) and isinstance(val2, dict):
                if not self._dict_compare(val1, val2):
                    return False
            # 如果都是列表，递归比较
            elif isinstance(val1, list) and isinstance(val2, list):
                if not self._list_compare(val1, val2):
                    return False
            # 其他类型直接比较
            else:
                if val1 != val2:
                    return False
        return True
            

    from contextlib import contextmanager
    @staticmethod
    @contextmanager
    def _capture_all_logs(target_logger, pre_fix='#CVXPY Output#: '):
        import io,sys
        class _LoggerStream(io.TextIOBase):
            def __init__(self, logger, log_level='info', pre_fix=''):
                self.logger = logger
                self.log_level = log_level
                self.pre_fix = pre_fix
                self.buffer = []
            def write(self, message):
                if message.strip():
                    for line in message.rstrip().splitlines():
                        if line:
                            getattr(self.logger,str.lower(self.log_level))(self.pre_fix + line)
                return len(message)
        logger_redir_out = _LoggerStream(target_logger, log_level='info', pre_fix=pre_fix)
        logger_redir_err = _LoggerStream(target_logger, log_level='error', pre_fix=pre_fix)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = logger_redir_out
        sys.stderr = logger_redir_err
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    def _get_term1(self, residual, C_weight, constraint_max_rms, term1_constraint_type, NT) -> cp.Expression:
        """
        Get term1 expression.

        Args:
            residual (cp.Expression): Residual expression.
            C_weight (ndarray): Weight array for capacitance data.
            constraint_max_rms (bool): Whether to constrain maximum RMS.
            term1_constraint_type (str): Type of term1 constraint ('L2' or others).
            NT (int): Number of temperature points.

        Returns:
            cp.Expression: Term1 expression.
        """
        if constraint_max_rms:
            if term1_constraint_type in ['L2', 'l2', 'rms']:
                # argmin {|Exp-Fit|^2}_T
                term1 = cp.max(cp.sum(cp.square(cp.multiply(np.sqrt(C_weight),residual)),axis=0))/self.Nt
            elif term1_constraint_type in ['L1', 'l1']:
                # argmin {|Exp-Fit|}_T
                term1 = cp.max(cp.sum(cp.abs(cp.multiply(C_weight,residual)),axis=0))/self.Nt
        else:
            if term1_constraint_type in ['L2', 'l2', 'rms']:
                # argmin {|Exp-Fit|^2}
                term1 = cp.sum_squares(cp.multiply(np.sqrt(C_weight),residual))/self.Nt/NT
            elif term1_constraint_type in ['L1', 'l1']:
                # argmin {|Exp-Fit|}
                term1 = cp.sum(cp.abs(cp.multiply(C_weight,residual)))/self.Nt/NT
        return term1

    def _get_term2(self, polarity: str, f_scaling_by_C: bool, C_average: np.ndarray,
                   f: cp.Expression, reg1: float | cp.Parameter,
                   enable_irls_mode: bool, irls_weight: np.ndarray | cp.Parameter) -> cp.Expression:
        '''
        if polarity == 'both':
            if f_scaling_by_C and not enable_irls_mode:
                term2 = reg1 * cp.sum(cp.abs(f[1:]/C_average))
            elif f_scaling_by_C and enable_irls_mode:
                term2 = reg1 * cp.sum(cp.multiply(irls_weight,cp.abs(f[1:]/C_average)))
            elif not f_scaling_by_C and not enable_irls_mode:
                term2 = reg1 * cp.sum(cp.abs(f[1:]))
            elif not f_scaling_by_C and enable_irls_mode:
                term2 = reg1 * cp.sum(cp.multiply(irls_weight,cp.abs(f[1:])))
        else:
            if f_scaling_by_C and not enable_irls_mode:
                if polarity == 'negative':
                    term2 = reg1 * cp.sum(cp.abs(f[1:])/C_average)
                else:
                    term2 = reg1 * cp.sum(f[1:]/C_average)
            elif f_scaling_by_C and enable_irls_mode:
                if polarity == 'negative':
                    term2 = reg1 * cp.sum(cp.multiply(irls_weight,cp.abs(f[1:])/C_average))
                else:
                    term2 = reg1 * cp.sum(cp.multiply(irls_weight,f[1:]/C_average))
            elif not f_scaling_by_C and not enable_irls_mode:
                if polarity == 'negative':
                    term2 = reg1 * cp.sum(cp.abs(f[1:]))
                else:
                    term2 = reg1 * cp.sum(f[1:])
            elif not f_scaling_by_C and enable_irls_mode:
                if polarity == 'negative':
                    term2 = reg1 * cp.sum(cp.multiply(irls_weight,cp.abs(f[1:])))
                else:
                    term2 = reg1 * cp.sum(cp.multiply(irls_weight,f[1:]))
        '''
        if f_scaling_by_C and not enable_irls_mode:
            term2 = reg1 * cp.sum(cp.abs(f[1:]/C_average))
        elif f_scaling_by_C and enable_irls_mode:
            term2 = reg1 * cp.sum(cp.multiply(irls_weight,cp.abs(f[1:]/C_average)))
        elif not f_scaling_by_C and not enable_irls_mode:
            term2 = reg1 * cp.sum(cp.abs(f[1:]))
        elif not f_scaling_by_C and enable_irls_mode:
            term2 = reg1 * cp.sum(cp.multiply(irls_weight,cp.abs(f[1:])))
        return term2


    def _build_problem(self, kwargs_for_build_problem: dict) -> None:
        """
        Construct CVXPY optimization problem for L1-regularized inversion.

        Args:
            kwargs_for_build_problem (dict): Problem configuration including:
                Ti_list (ndarray): Temperature indices to process
                data_weight_fun_T (Callable): weight function of data with different temperature
                constraint_max_rms (bool): Whether to constrain maximum RMS
                lambda1 (float): Regularization parameter
                polarity (str): Defect polarity constraint ('positive','negative','both')
                f_scaling_by_C (bool): Scale defect density by averaged capacitance
                enable_irls_mode (bool): Enable Iteratively Reweighted Least Squares mode
                irls_weight (ndarray): Initial weights for IRLS
                arg_lists_use_cp_parameter (list): Parameters to treat as CVXPY parameters
        Return:
            None
        
        Update Attribute:
            self.problems (list): new problem will be appended to self.problems.
        """
        # extract params needed
        Ti_list = np.array(kwargs_for_build_problem['Ti_list'])
        if type(kwargs_for_build_problem['data_weight_fun_T']) == str:
            data_weight_fun_T = eval(kwargs_for_build_problem['data_weight_fun_T'])
        else:
            data_weight_fun_T = kwargs_for_build_problem['data_weight_fun_T']
        data_weight_fun_T = np.vectorize(data_weight_fun_T)
        term1_constraint_type = kwargs_for_build_problem['term1_constraint_type']
        constraint_max_rms = kwargs_for_build_problem['constraint_max_rms']
        lambda1 = kwargs_for_build_problem['lambda1']
        polarity = kwargs_for_build_problem['polarity']
        f_scaling_by_C = kwargs_for_build_problem['f_scaling_by_C']
        enable_irls_mode = kwargs_for_build_problem['enable_irls_mode']
        irls_weight = np.array(kwargs_for_build_problem['irls_weight'])
        arg_lists_use_cp_parameter = kwargs_for_build_problem['arg_lists_use_cp_parameter']
        
        temp_dic = {}
        for ivar in arg_lists_use_cp_parameter:
            if type(kwargs_for_build_problem[ivar])==np.ndarray:
                temp_dic[ivar] = cp.Parameter(kwargs_for_build_problem[ivar].shape,nonneg=True)
            elif type(kwargs_for_build_problem[ivar]) in [int,float]:
                temp_dic[ivar] = cp.Parameter(nonneg=True)
            temp_dic[ivar].value = kwargs_for_build_problem[ivar]

        if 'lambda1' in temp_dic.keys():
            lambda1 = temp_dic['lambda1']
        elif 'irls_weight' in temp_dic.keys():
            irls_weight = temp_dic['irls_weight']
        
        C = self.C[Ti_list,:]
        NT = C.shape[0]
        # (Ns,NT)
        C_average = np.tile(np.average(C.T,axis=0).reshape(1,NT),(self.Ns,1))
        C_weight = np.abs(data_weight_fun_T(self.T[Ti_list]))
        reg1 = lambda1
        f = cp.Variable((self.Ns+1,NT))
        if polarity in ['positive']:
            constraints = [f[1:,:]>=0]
        elif polarity in ['negative']:
            constraints = [f[1:,:]<=0]
        else:
            constraints = []
        residual = self.A_extended @ f - C.T
        term1 = self._get_term1(residual, C_weight,constraint_max_rms=constraint_max_rms,
                                term1_constraint_type=term1_constraint_type, NT=NT)
        term2 = self._get_term2(polarity, f_scaling_by_C, C_average, f, reg1,
                                enable_irls_mode, irls_weight)
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
            'variable':{k:local_vars[k] for k in ['f','term1','term2','residual']},
            'kwargs_for_build_problem':kwargs_for_build_problem
        })
        del problem,local_vars
    
    def solve(
        self, Ti_list: np.ndarray, polarity: str = 'positive', verbose: bool = False,
        lambda1: float = 1e-5, skip_solved: bool = True, enable_irls_mode: bool = False,
        irls_weight: int | float | np.ndarray = 1, solver: str = 'CVXOPT', solver_params: dict = {},
        data_weight_fun_T: Callable | str = "lambda T: 1", f_scaling_by_C: bool = False,
        constraint_max_rms: bool = True, term1_constraint_type: str = 'L2',
        arg_lists_use_cp_parameter: list = ['lambda1']
    ) -> None | LDLTS_SolveResultType:
        """
        Solve L1-regularized inverse problem for given temperature indices.

        Args:
            Ti_list (ndarray): Temperature indices to process
            polarity (str): Defect polarity constraint ('positive','negative','both')
            verbose (bool): Show log.
            lambda1 (float): Regularization parameter.
            skip_solved (bool): Skip solved problem.
            enable_irls_mode (bool): Enable Iteratively Reweighted Least Squares mode, will return dict in this case
            irls_weight (int | float | ndarray): Initial weights for IRLS.
            solver (str): Solver name.
            solver_params (dict): kwargs for solver.
            data_weight_fun_T (Callable | str): weight function of data with different temperature.
                (lambda function is unpicklable, use str in this case like "lambda T: 1", this will be evaled inside)
            f_scaling_by_C (bool): Scale defect density by averaged capacitance.
            constraint_max_rms (bool): Whether to constrain maximum RMS.
            arg_lists_use_cp_parameter (list): String list of parameter name to treat as CVXPY parameters.

        Returns:
            None | LDLTS_SolveResultType

        Update Attributes:
            solve_history: if not in irls mode

        Raises:
            RuntimeError: If solver fails to converge
        """
        local_vars = locals()
        input_kwargs = {k:v for k,v in local_vars.items() if k not in ['self']}
        kwargs_for_filter_solved = {}
        for k,v in input_kwargs.items():
            if k not in ['verbose','skip_solved']:
                if type(v) == np.ndarray:
                    kwargs_for_filter_solved[k]=v.tolist()
                else:
                    kwargs_for_filter_solved[k]=v
        kwargs_for_filter_problems = {}
        for k,v in input_kwargs.items():
            if k not in ['verbose','skip_solved','solver','solver_params']:
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
                self.logger.warning(f'{self.method_name} solve: Problem already solved, skip')
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
            except Exception as e:
                self.logger.error(f"{self.method_name} solve: update parameter failed:", e)
        else:
            try:
                if verbose:
                    self.logger.info(f"{self.method_name} solve: build new problem")
                self._build_problem(kwargs_for_filter_problems)
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
        residual = self.problems[problem_idx]['variable']['residual']
        term1 = self.problems[problem_idx]['variable']['term1']
        term2 = self.problems[problem_idx]['variable']['term2']
        Ti_list = np.array(self.problems[problem_idx]['kwargs_for_build_problem']['Ti_list'])
        lambda1 = self.problems[problem_idx]['kwargs_for_build_problem']['lambda1']
        enable_irls_mode = self.problems[problem_idx]['kwargs_for_build_problem']['enable_irls_mode']
        irls_weight = self.problems[problem_idx]['kwargs_for_build_problem']['irls_weight']
        f_current = f.value
        # rms at each temperature
        rms_list = np.sqrt(np.sum(residual.value**2, axis=0)/self.Nt)
        term1_error = term1.value
        term2_error = term2.value
        irls_target = f_current[1:]
        temp_result_dict = {
            'input_params':{}, 'Ti_list':Ti_list, 'f':f_current, 'term1_error':term1_error, 'reg':lambda1, 'term2_error':term2_error,
            'rms_list':rms_list,
            'irls_target':irls_target, 'irls_eval_error_dict':{'term1_error':term1_error,'term2_error':term2_error}, 'wall_time':0.0
        }
        # update input_params, wall_time and additional kwargs
        temp_result_dict.update(add_kwargs)
        if enable_irls_mode:
            temp_result_dict['irls_weight'] = irls_weight
        return temp_result_dict
    
    def irls(
        self, solver: str = 'CVXOPT', solver_params: dict = {}, solve_index: int = -1,
        irls_max_iter: int = 20, irls_p_norm: float = 0.5, irls_rel_tol: float = 1e-7,
        irls_max_weight: float = 1e8, irls_min_weight: float = 1e-8, irls_oscillate_rel_tol: float = 1e-7,
        verbose: bool = False, arg_lists_use_cp_parameter: list = [], save_unconverged_data: bool = True,
        overwrite_input_params: dict = {}
    ) -> None:
        """
        Perform Iteratively Reweighted Least Squares (IRLS) optimization.

        Args:
            solver (str, optional): CVXPY solver name. Default 'CVXOPT'
            solver_params (dict, optional): Solver parameters. Default {}
            solve_index (int, optional): Solution index to process. Default -1 (latest)
            irls_max_iter (int, optional): Maximum IRLS iterations. Default 20
            irls_p_norm (float, optional): L-p norm exponent. Default 0.5
            irls_rel_tol (float, optional): Relative error tolerance. Default 1e-7
            irls_max_weight (float, optional): Maximum weight threshold. Default 1e8
            irls_min_weight (float, optional): Minimum weight threshold. Default 1e-8
            irls_oscillate_rel_tol (float, optional): Oscillation detection tolerance. Default 1e-7
            verbose (bool, optional): Print iteration details. Default False
            arg_lists_use_cp_parameter (list, optional): Parameters to treat as CVXPY parameters. Default []
            save_unconverged_data (bool, optional): Save unconverged data. Default True
            overwrite_input_params (dict, optional): Overwrite input parameters. Default {}

        Returns:
            None

        Update Attributes:
            irls_solve_history if converged

        Raises:
            ValueError: Invalid solve index
            RuntimeError: If solve fails during IRLS iteration
        """
        self.logger.info(f"{self.method_name} IRLS Method Started with params: ")
        self.logger.info(f"solver={solver}; solver_params={solver_params}; solve_index={solve_index}; arg_lists_use_cp_parameter={arg_lists_use_cp_parameter}")
        self.logger.info(f"irls_max_iter={irls_max_iter}; irls_p_norm={irls_p_norm}; irls_rel_tol={irls_rel_tol};")
        self.logger.info(f"irls_max_weight={irls_max_weight}; irls_min_weight={irls_min_weight}; irls_oscillate_rel_tol={irls_oscillate_rel_tol};")
        # extract solve input_params and irls params
        try:
            irls_target = self.solve_history[solve_index]['irls_target']
            if self.method_name in ['SR']:
                _origin_irls_target = np.zeros_like(irls_target)
            irls_eval_error_dict = self.solve_history[solve_index]['irls_eval_error_dict']
            irls_input_params = self.solve_history[solve_index]['input_params']
            for k,v in overwrite_input_params.items():
                irls_input_params[k] = v
            irls_input_params['solver'] = solver
            irls_input_params['solver_params'] = solver_params
            irls_input_params['enable_irls_mode'] = True
            # important, set problem parameter to irls_weight only, so that build a dpp problem
            # howerer, for a large problem, compilation of dpp is too slow
            #irls_input_params['arg_lists_use_cp_parameter'] = ['irls_weight']
            irls_input_params['arg_lists_use_cp_parameter'] = arg_lists_use_cp_parameter
        except Exception as e:
            raise ValueError(f"{self.method_name} irls: Get irls params error: {str(e)}")
        irls_error_history_dict = {}
        for error_name in irls_eval_error_dict:
            irls_error_history_dict[error_name] = [irls_eval_error_dict[error_name]]
        first_line_flag = True
        print_len = 20
        # Start iterating
        for iter in range(irls_max_iter):
            # Processing irls target
            try:
                if self.method_name in ['SR']:
                    if iter==0:
                        irls_weight = np.clip(1/(np.abs(irls_target))**(1-irls_p_norm), irls_min_weight, irls_max_weight)
                        irls_target_mask = irls_weight<irls_max_weight
                    else:
                        _origin_irls_target[irls_target_mask] = irls_target
                        irls_weight = np.clip(1/(np.abs(_origin_irls_target))**(1-irls_p_norm), irls_min_weight, irls_max_weight)
                        # update mask
                        irls_target_mask = irls_weight<irls_max_weight
                else:
                    irls_weight = np.clip(1/(np.abs(irls_target))**(1-irls_p_norm), irls_min_weight, irls_max_weight)
                    irls_target_mask = None
                irls_input_params['irls_weight'] = irls_weight
            except:
                pass
            verbose_input = True if verbose == 2 else False
            temp_input = irls_input_params
            temp_input['verbose'] = verbose_input
            if self.method_name in ['SR']:
                temp_input['remove_large_irls_weight']=True
                temp_input['large_irls_weight_threshold']=irls_max_weight
            
            solve_converged = False
            try:
                self._del_unpicklable_attr()
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.solve,**temp_input)
                    self.temp_irls_solve_result = future.result()
                solve_converged = True
            except Exception as e:
                raise RuntimeError(f"{self.method_name} irls: Solve Failed: {e}")
            
            if not solve_converged:
                self.logger.info(f"{self.method_name} irls: Solve Failed")
                break
            else:
                # update value
                irls_eval_error_dict = self.temp_irls_solve_result['irls_eval_error_dict']
                if verbose:
                    if first_line_flag:
                        first_line_flag=False
                        temp_line = 'Iter'
                        temp_line += ' '*(print_len-len('Iter'))
                        for error_name in irls_eval_error_dict:
                            temp_line += f"{error_name}"+' '*(print_len-len(error_name))
                        for error_name in irls_eval_error_dict:
                            if error_name != list(irls_eval_error_dict.keys())[-1]:
                                temp_line += f"{error_name}_rel"+ ' '*(print_len-len(f"{error_name}_rel"))
                            else:
                                temp_line += f"{error_name}_rel"
                        self.logger.info(temp_line)
                    temp_line = f"{iter+1}"+' '*(print_len-len(str(iter+1)))
                    for error_val in irls_eval_error_dict.values():
                        temp_line += f"{error_val:.5e}"+' '*(print_len-len(f"{error_val:.5e}"))
                irls_target = self.temp_irls_solve_result['irls_target']
                # Error added to history
                for error_name in irls_eval_error_dict:
                    irls_error_history_dict[error_name].append(irls_eval_error_dict[error_name])
                # Judging convergence
                converged = False
                for i,error_name in enumerate(irls_eval_error_dict):
                    error_history = irls_error_history_dict[error_name]
                    # Convergence detection, either of the two conditions is met
                    # Relative error detection
                    if len(error_history) > 1:
                        rel_change = abs(error_history[-1] - error_history[-2]) / abs(error_history[-1])
                        temp_line += f"{rel_change:.5e}"+' '*(print_len-len(f"{rel_change:.5e}"))
                        # Convergence condition: relative change is less than the threshold
                        if rel_change < irls_rel_tol:
                            # If converged, determine if this is the last error in the list. This means all errors have converged, so break directly.
                            # Otherwise, the current error has converged, and the loop for the next error needs to be continued.
                            if i+1 == len(irls_eval_error_dict):
                                converged=True
                                self.logger.info(temp_line)
                                self.logger.info(f"{self.method_name} irls: Success! Relative Error < {irls_rel_tol}")
                                break
                            else:
                                continue
                        else:
                            # If it does not converge, jump out of the loop directly
                            break
                    # Error oscillation detection (requires at least 4 iterations)
                    if len(error_history) > 3:
                        # Calculate the oscillation indicator: Check whether the error value is oscillating alternately
                        if abs(error_history[-1]-error_history[-3])/abs(error_history[-1]) < irls_oscillate_rel_tol and \
                        abs(error_history[-2]-error_history[-4])/abs(error_history[-2]) < irls_oscillate_rel_tol:
                            # If converged, determine if this is the last error in the list. This means all errors have converged, so break directly.
                            # Otherwise, the current error has converged, and the loop for the next error needs to continue.
                            if i+1 == len(irls_eval_error_dict):
                                converged=True
                                self.logger.info(temp_line)
                                self.logger.info(f"{self.method_name} irls: Success! Oscillate Relative Error < {irls_oscillate_rel_tol}")
                                #print()
                                #print(f"{self.method_name} irls: Success! Oscillate Relative Error < {irls_oscillate_rel_tol}")
                                break
                            else:
                                continue
                        else:
                            # If it does not converge, jump out of the loop directly
                            break
                if converged:
                    # End of iteration
                    self.temp_irls_solve_result['if_converged'] = True
                    self.irls_solve_history.append(self.temp_irls_solve_result)
                    self.irls_solve_history[-1]['input_params'] = irls_input_params
                    # save mask info
                    if irls_target_mask is not None:
                        self.irls_solve_history[-1]['irls_target_mask'] = irls_target_mask
                    break
                else:
                    if save_unconverged_data:
                        self.temp_irls_solve_result['if_converged'] = False
                        self.irls_solve_history.append(self.temp_irls_solve_result)
                        self.irls_solve_history[-1]['input_params'] = irls_input_params
                        # save mask info
                        if irls_target_mask is not None:
                            self.irls_solve_history[-1]['irls_target_mask'] = irls_target_mask
                    self.logger.info(temp_line)
        if not converged:
            self.logger.info(f"{self.method_name} irls stopped: attempt times >= {irls_max_iter}")
    
    # input input_params dictionary
    def filtering_solve_history(self, input_dic: dict = {}) -> list:
        """
        Filter solve history by input parameters.

        Args:
            input_dic (dict, optional): Search parameters. Default {} (all matches)

        Returns:
            list: Indices of matching solutions in solve_history
        """
        filtered_index_list=[]
        for solve_index,solve_result in enumerate(self.solve_history):
            if input_dic.items()<=solve_result['input_params'].items():
                filtered_index_list.append(solve_index)
        return filtered_index_list

    def lcurve(
        self, solve_arg: dict = {}, ramp_param_name: str = 'lambda1', auto_update_rms_limit: bool = True,
        lcurve_init_lambda: float = 1e-8, lcurve_target_rms_point: float = 1e-4,
        lcurve_target_rms_range: float = 2e-5, lcurve_max_attempt: float = 20, lcurve_multiplier: float = 10
    ):
        """
        Perform L-curve regularization parameter optimization.

        Args:
            solve_arg (dict, optional): Solve parameters for L-curve search. Default {}
            ramp_param_name (str, optional): Parameter to optimize. Default 'lambda1'
            auto_update_rms_limit (bool, optional): Update RMS tolerance. Default True
            lcurve_init_lambda (float, optional): Initial regularization value. Default 1e-8
            lcurve_target_rms_point (float, optional): Target relative RMS. Default 1e-4
            lcurve_target_rms_range (float, optional): Acceptable RMS range. Default 2e-5
            lcurve_max_attempt (int, optional): Maximum search iterations. Default 20
            lcurve_multiplier (float, optional): Regularization adjustment factor. Default 10

        Returns:
            bool
        
        Update Attributes:
            solve_history by calling self.solve
            lcurve_history if converged
        """
        self.logger.info(f"{self.method_name} Lcurve Method Started with params: ")
        self.logger.info(f"solve_arg={solve_arg};")
        self.logger.info(f"ramp_param_name={ramp_param_name};")
        self.logger.info(f"auto_update_rms_limit={auto_update_rms_limit}; lcurve_init_lambda={lcurve_init_lambda}; lcurve_target_rms_point={lcurve_target_rms_point};")
        self.logger.info(f"lcurve_target_rms_range={lcurve_target_rms_range}; lcurve_max_attempt={lcurve_max_attempt}; lcurve_multiplier={lcurve_multiplier};")
        import inspect
        # Get the input parameters of the self.solve method
        solve_all_param_list = list(inspect.signature(self.solve).parameters.keys())
        # Check if all the keys in solve_arg are in solve_need_param_list
        for spi in list(solve_arg.keys()):
            if spi not in solve_all_param_list:
                raise ValueError(f"{self.method_name} L-Curve: wrong input: solve_arg")
        # Filter out the parameter names that need to be ramped, that is, the regularization coefficient and the verbose parameter
        solve_input_params = {}
        for spi in solve_all_param_list:
            if spi not in [ramp_param_name,'verbose','word_fun_list','Ti_list']:
                solve_input_params[spi] = solve_arg[spi]
            elif spi in ['word_fun_list']:
                word_fun_list_str = []
                for temp_wf in solve_arg[spi]:
                    if type(temp_wf)!=str:
                        word_fun_list_str.append(temp_wf.__name__)
                    else:
                        word_fun_list_str.append(temp_wf)
                solve_input_params[spi] = word_fun_list_str
            elif spi in ['Ti_list']:
                solve_input_params[spi] = list(solve_arg[spi])
        # get target_rms
        if self.rms_limit!=0:
            target_rms = self.rms_limit*(1+lcurve_target_rms_point)
        else:
            if auto_update_rms_limit:
                reg_input_kwarg = {ramp_param_name:lcurve_init_lambda}
                self.solve(**reg_input_kwarg, **solve_input_params, verbose=False)
                self.rms_limit=self.solve_history[-1]['rms']
                target_rms = self.rms_limit*(1+lcurve_target_rms_point)
            if self.rms_limit==0:
                self.logger.error(f"{self.method_name} L-Curve: rms_limit unset")
                return 1
        # Remove unnecessary input parameters to filter solve results
        feature_solve_input_params = {}
        for spi in list(solve_input_params.keys()):
            if spi not in ['solver_params','skip_solved']:
                feature_solve_input_params[spi] = solve_input_params[spi]
        # Search for points have already run
        solved_index_list = self.filtering_solve_history(input_dic=feature_solve_input_params)
        if len(solved_index_list)==0:
            # init run
            reg_input_kwarg = {ramp_param_name:lcurve_init_lambda}
            if self.solve(**reg_input_kwarg, **solve_input_params, verbose=False):
                self.logger.error(f"{self.method_name} L-Curve: initial solve failed")
                return 1
            # update solved_index_list
            solved_index_list = self.filtering_solve_history(input_dic=feature_solve_input_params)
        # Target rms calculation
        target_rms0 = target_rms*(1-lcurve_target_rms_range)
        target_rms1 = target_rms*(1+lcurve_target_rms_range)
        self.logger.info(f'{self.method_name} L-Curve: Target rms is {round(target_rms,8)}, allow range=({round(target_rms0,8)},{round(target_rms1,8)})')
        converged = False
        rms_limit_too_small = False
        solve_failed = False
        found_solve_index = None
        for iter in range(lcurve_max_attempt):
            # update reg_list & rms_list
            reg_list = [self.solve_history[i]['lambda'] for i in solved_index_list]
            rms_list = [self.solve_history[i]['rms'] for i in solved_index_list]
            # lambda too small
            if max(rms_list)<target_rms0:
                if min(reg_list)<1e-12:
                    rms_limit_too_small=True
                    break
                else:
                    next_lambda = max(reg_list) * lcurve_multiplier
                    self.logger.info(f'{self.method_name} L-Curve: current max rms={round(max(rms_list),8)} too small, set next lambda={round(next_lambda,8)}')
            # lambda too big
            elif min(rms_list)>target_rms1:
                next_lambda = min(reg_list) / lcurve_multiplier
                self.logger.info(f'{self.method_name} L-Curve: current min rms={round(min(rms_list),8)} too big, set next lambda={round(next_lambda,8)}')
            else:
                for i,_ in enumerate(solved_index_list):
                    temp_rms = rms_list[i]
                    temp_lambda = reg_list[i]
                    if temp_rms>=target_rms0 and temp_rms<=target_rms1:
                        found_solve_index = solved_index_list[i]
                        converged = True
                        break
                if converged:
                    self.logger.info(f'{self.method_name} L-Curve: Task Successed. Find Solve_index={found_solve_index} with rms={temp_rms} and lambda={temp_lambda}')
                    break
                # Find the largest lambda smaller than target_rms0 and the smallest lambda larger than target_rms1
                temp_sort = np.argsort(rms_list)
                rms_bound0_index = temp_sort[0]
                rms_bound1_index = temp_sort[-1]
                for i,_ in enumerate(solved_index_list):
                    temp_rms = rms_list[i]
                    temp_reg = reg_list[i]
                    if temp_rms<target_rms0:
                        if temp_reg>reg_list[rms_bound0_index]:
                            rms_bound0_index=i
                    if temp_rms>target_rms1:
                        if temp_reg<reg_list[rms_bound1_index]:
                            rms_bound1_index=i
                # The corresponding rms boundary
                xp = [rms_list[rms_bound0_index],rms_list[rms_bound1_index]]
                # The corresponding reg boundary
                yp = [reg_list[rms_bound0_index],reg_list[rms_bound1_index]]
                # Linear interpolation -> change to logarithmic interpolation?
                next_lambda = np.interp(target_rms, xp, yp)
                self.logger.info(f"{self.method_name} L-Curve: current rms={round(self.solve_history[-1]['rms'],8)}, set next lambda={round(next_lambda,8)}")
            # solve next_lambda
            reg_input_kwarg = {ramp_param_name:next_lambda}
            if self.solve(**reg_input_kwarg, **solve_input_params, verbose=False):
                solve_failed = True
                break
            # update solved_index_list
            solved_index_list = self.filtering_solve_history(input_dic=feature_solve_input_params)
        # save date
        if converged:
            self.lcurve_history.append({
                'lcurve_solved_flag':True, 'associated_solve_index_list':solved_index_list, 'lcurve_solve_index':found_solve_index,
                'rms_limit':self.rms_limit, 'f':self.solve_history[found_solve_index]['f'], 'rms':self.solve_history[found_solve_index]['rms'],
                'lambda':self.solve_history[found_solve_index]['lambda'],
                'lcurve_target_rms_point':lcurve_target_rms_point, 'lcurve_target_rms_range':lcurve_target_rms_range
            })
            return False
        else:
            self.lcurve_history.append({
                'lcurve_solved_flag':False, 'associated_solve_index_list':solved_index_list, 'lcurve_solve_index':None,
                'rms_limit':self.rms_limit, 'f':None, 'rms':None,
                'lambda':None, 'lcurve_target_rms_point':lcurve_target_rms_point, 'lcurve_target_rms_range':lcurve_target_rms_range
            })
            if rms_limit_too_small:
                self.logger.error(f'{self.method_name} L-Curve: Task Failed. Rms_limit too small')
            elif solve_failed:
                self.logger.error(f'{self.method_name} L-Curve: Task Failed. Solve Failed')
            else:
                self.logger.error(f'{self.method_name} L-Curve: Task Failed. L-curve Attempt too many times={lcurve_max_attempt}')
            return True

    # Perturbation Response Spectral Sensitivity Estimation
    def prsse(self, solve_index: int = -1, si_list: list | None = None, solver: str = 'CVXOPT',
              solver_params: dict = {}, fwhm_height: float = 0.5, perturbation_factor: float = 1) -> None:
        """
        Perform Perturbation Response Spectral Sensitivity Estimation (PRSSE).

        Args:
            solve_index (int, optional): Solution index to process. Default -1 (latest)
            si_list (list | None, optional): Emission rate index list to process. Default None
            solver (str, optional): CVXPY solver name. Default 'CVXOPT'
            solver_params (dict, optional): kwargs for solver. Default {}
            fwhm_height (float, optional): full width at half maximum height. Default 0.5 (half)
            perturbation_factor (float, optional): Perturbation factor. Default 1
        
        Returns:
            None
        
        Update Attributes:
            prsse_solve_history if converged

        """
        target_data = self.solve_history[solve_index]
        Ti_list = list(target_data['Ti_list'])
        if si_list is None:
            step = int(self.Ns/100)
            if step <= 1:
                si_list = np.arange(0,self.Ns)
            else:
                si_list = np.arange(0,self.Ns,step)
        s_list = self.s[si_list]
        tau_list = 1/s_list
        total_rms_list = []
        rms_limit_list = []
        # build problem
        for i,Ti in enumerate(Ti_list):
            Nt = self.Nt
            # (Nt,)
            temp_C = self.C[Ti]
            # (Ns,)
            temp_f = target_data['f'][:,i]
            temp_C_fit = self.A_extended @ temp_f
            temp_rms = np.sqrt(((temp_C-temp_C_fit)**2).mean())
            rms_limit_list.append(temp_rms)
            # built problem
            temp_const = cp.Variable(1)
            delta_f = cp.Parameter(len(temp_f))
            temp_delta_f = np.zeros_like(temp_f)
            temp_delta_f[0] = temp_rms
            delta_f.value = temp_delta_f
            term0 = cp.sum_squares(temp_C - self.A_extended@(temp_f+delta_f) - temp_const)/Nt
            temp_obj = cp.Minimize(term0)
            problem = cp.Problem(temp_obj)
            # solve
            rms_list = []
            for si in si_list:
                temp_delta_f = np.zeros_like(temp_f)
                # skip DC
                temp_delta_f[1+si] = perturbation_factor*temp_rms
                delta_f.value = temp_delta_f
                if solver is not None:
                    problem.solve(solver=getattr(cp,solver),**solver_params)
                else:
                    problem.solve()
                rms_list.append(np.sqrt(term0.value))
            total_rms_list.append(rms_list)
        total_rms_array = np.array(total_rms_list)
        fwhm_list = []
        try:
            for i,Ti in enumerate(Ti_list):
                x = s_list
                y = total_rms_array[i]
                rms0 = rms_limit_list[i]
                rms1 = np.max(y)
                fwhm_x0,fwhm_x1 = self._find_fwhm(x, y, rms0, rms1, ref_height=fwhm_height)
                fwhm_list.append([fwhm_x0,fwhm_x1])
            fwhm_list = np.array(fwhm_list)
        finally:
            self.prsse_solve_history.append({
                'Ti_list':Ti_list,
                'T_list':self.T[np.array(Ti_list)],
                's_list':s_list,
                'tau_list':tau_list,
                'rms_array':total_rms_array,
                'fwhm_s0_s1_array':fwhm_list,
                'fwhm_height':fwhm_height
            })
            

    @staticmethod
    def _find_fwhm(x: np.ndarray, y: np.ndarray, y_0: float, y_peak: float, ref_height: float = 0.5) -> tuple[float, float]:
        """
        Calculates the FWHM (full width at half maximum) boundary points of a peak.
        Parameters:
            x: 1D array of x values
            y: 1D array of y values
        Returns:
            fwhm_x0: x-coordinate of the left boundary
            fwhm_x1: x-coordinate of the right boundary
        """
        half_max = (y_0 + y_peak) * ref_height
        cross_points = []
        for i in range(len(y) - 1):
            y0, y1 = y[i], y[i + 1]
            if (y0 < half_max < y1) or (y1 < half_max < y0):
                dx = x[i + 1] - x[i]
                dy = y[i + 1] - y[i]
                slope = dy / dx
                x_cross = x[i] + (half_max - y[i]) / slope
                cross_points.append(x_cross)
        cross_points.sort()
        if len(cross_points) < 2:
            raise ValueError("Unable to find two intersection points: Please check whether the data is valid unimodal")
        fwhm_x0, fwhm_x1 = cross_points[0], cross_points[-1]
        return fwhm_x0, fwhm_x1

    def save_result(self, filepath: str | Path | None = None, save_problem: bool = True) -> None:
        """
        Save solver state to file.

        Args:
            filepath (str | Path | None, optional): Output file path. Default None
            save_problem (bool, optional): Save problem data. Default True
        """
        if filepath is None:
            filepath = Path(f'{self.method_name}_temp_ldlts_result.npy').resolve()
        else:
            filepath = Path(filepath).resolve()
        if not filepath.suffix == ".npy":
            filepath = filepath.with_suffix(".npy")
        config = {}
        for an in self._attr_names_for_save:
            config[an] = getattr(self,an)
        if save_problem:
            # deal problem
            temp_problems = self.problems.copy()
            for tp in temp_problems:
                tp['problem']=None
            config['problems'] = temp_problems
        try:
            np.save(filepath, config)
        except Exception as e:
            self.logger.error(f'save config error: {str(e)}')
    def load_result(self, filepath: str | Path | None = None, load_problem: bool = True) -> None:
        """
        Load solver state from file.

        Args:
            filepath (str | Path | None, optional): Input file path. Default None
            load_problem (bool, optional): Load problem data. Default True
        """
        if filepath is None:
            filepath = Path(f'{self.method_name}_temp_ldlts_result.npy').resolve()
        else:
            filepath = Path(filepath).resolve()
        if not filepath.suffix == ".npy":
            filepath = filepath.with_suffix(".npy")
        temp = np.load(filepath, allow_pickle=True).item()
        for ak,av in temp.items():
            setattr(self,ak,av)
        for dk,dv in self._data_dict_for_ldlts.items():
            setattr(self,dk,dv)
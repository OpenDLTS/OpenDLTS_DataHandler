__all__ = ["Data_Loader"]

from dataclasses import dataclass, field
import numpy as np
from ._typing import *
from ._error import *
from ._config import *
from ._get_data_plot_dict import get_data_plot_dict

@dataclass
class Data_Loader:
    """
    Core class for handling DLTS experimental datasets.

    Attributes:
        t (np.ndarray): Time measurement points [s]
        T (np.ndarray): Measurement conditions (temperature/voltage) for each transient
        C (np.ndarray): Signal data (capacitance/current/voltage) for each condition
        data_plot_dict (TransientDataPlotType): Plot configuration (labels, units, plot scaling factors)
        data_scaling_factor (float): Scaling factor for data to fit

    Methods allow data loading from files/arrays
    """
    transient_data: TransientDataType = None
    raw_data_scaling_factor: float = 1.0
    time_shift: float | np.ndarray = 0
    T_shift: float | np.ndarray = 0
    data_scaling_factor: float = 1e12
    data_type: str = 'C'
    data_x_type: str = 'Time'
    condition_type: str = 'Temperature'
    logging_level: str = 'info'
    logging_file: None | Path | str = None
    logging_file_clear: bool = False
    
    t: np.ndarray = field(init=False)
    T: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    data_plot_dict: TransientDataPlotType = field(init=False)
    transient_data_full_path: Path = field(init=False)
    rawdata: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize DLTS dataset loader.
        """
        # init logging file
        if self.logging_file is not None:
            INIT_LOG_FILE(self.logging_file, self.logging_file_clear)
        # init logging level
        import logging
        LOGGER_CVXPY.setLevel(getattr(logging, str.upper(self.logging_level)))
        LOGGER_ODDH.setLevel(getattr(logging, str.upper(self.logging_level)))
        
        if isinstance(self.transient_data, (str, Path)):
            self.transient_data_full_path = Path(self.transient_data).resolve()
            tempdata = np.loadtxt(self.transient_data_full_path, delimiter=None)
            t = tempdata[0,1:]
            T = tempdata[1:,0]
            C = tempdata[1:,1:]
            LOGGER_ODDH.info(f"Data loaded from file: {self.transient_data_full_path}")
        elif isinstance(self.transient_data, dict):
            self.transient_data_full_path = Path("temp_transient_data.transdata").resolve()
            t = np.array(self.transient_data['t'])
            T = np.array(self.transient_data['T'])
            C = np.array(self.transient_data['C'])
            if t.shape[0]!=C.shape[1] or T.shape[0]!=C.shape[0]:
                raise ValueError('Input Wrong')
            LOGGER_ODDH.info(f"Data loaded from provided dictionary.")
        elif isinstance(self.transient_data, np.ndarray):
            self.transient_data_full_path = Path("temp_transient_data.transdata").resolve()
            t = self.transient_data[0,1:]
            T = self.transient_data[1:,0]
            C = self.transient_data[1:,1:]
            LOGGER_ODDH.info(f"Data loaded from provided ndarray.")
        else:
            raise TransientDataTypeError
        
        self._rawdata_init(t, T, C, self.time_shift, self.T_shift, self.raw_data_scaling_factor)
        self._data_preprocessing()
        self._sub_T_tol = 0.2

    def _rawdata_init(self, t: np.ndarray, T: np.ndarray, C: np.ndarray,
                      time_shift: float | np.ndarray, T_shift: float | np.ndarray,
                      raw_data_scaling_factor: float) -> None:
        if isinstance(time_shift, (float, int)):
            time_shift = np.ones_like(t) * time_shift
        if isinstance(T_shift, (float, int)):
            T_shift = np.ones_like(T) * T_shift
        if time_shift.shape != t.shape:
            raise ValueError('time_shift shape mismatch')
        if T_shift.shape != T.shape:
            raise ValueError('T_shift shape mismatch')
        self.t = t + time_shift
        self.T = T + T_shift
        self.C = C * raw_data_scaling_factor
        self.rawdata = np.zeros((len(self.T)+1,len(self.t)+1))
        self.rawdata[1:,0] = self.T
        self.rawdata[1:,1:] = self.C
        self.rawdata[0,1:] = self.t

    def _data_preprocessing(self) -> None:
        """
        Preprocess data for analysis and visualization.
        """
        # data scaling
        self.C = self.C * self.data_scaling_factor
        # get data plot dict
        self.data_plot_dict = get_data_plot_dict(
            data_scaling_factor=self.data_scaling_factor,
            data_type=self.data_type,
            data_x_type=self.data_x_type,
            condition_type=self.condition_type
        )

    def data_space(self, num: int, space: str = 'log', lin_interp: bool = False) -> 'Data_Loader':
        """
        Resample data points in linear or logarithmic time space.
        
        Args:
            num: Number of points in new time axis
            space: 'lin' (linear) or 'log' (logarithmic)
            lin_interp: If True, use linear interpolation for resampling
        Returns:
            Data_Loader: A new instance with resampled data.
        """
        if lin_interp:
            if space == 'lin':
                t_new = np.linspace(self.t[0], self.t[-1], num)
            elif space == 'log':
                t_new = np.logspace(np.log10(self.t[0]), np.log10(self.t[-1]), num)
            C_new = np.zeros((len(self.T),len(t_new)))
            for Ti in range(len(self.T)):
                C_new[Ti] = np.interp(t_new, self.t, self.C[Ti])
        else:
            from ._ReSampleFromTimeArray import ReSampleFromTimeArray
            t_new_indices = ReSampleFromTimeArray(self.t, num, space=space)
            t_new = self.t[t_new_indices]
            C_new = self.C[:, t_new_indices]

        new_transient_data = {
            't': t_new,
            'T': self.T,
            'C': C_new / self.data_scaling_factor
        }
        new_loader = Data_Loader(
            transient_data=new_transient_data,
            raw_data_scaling_factor=1.0,
            time_shift=0.0,
            T_shift=0.0,
            data_scaling_factor=self.data_scaling_factor,
            data_type=self.data_type,
            data_x_type=self.data_x_type,
            condition_type=self.condition_type,
            logging_level=self.logging_level,
            logging_file=None,
            logging_file_clear=False
        )
        new_loader.transient_data_full_path = self.transient_data_full_path
        LOGGER_ODDH.info(f"New Data resampled to {num} points in {space} space.")
        return new_loader
    
    def savedata(self, newdatafile: Path | str = 'temp_output_data.transdata', ignore_scaling: bool = False) -> None:
        """
        Export data in DLTS_format.
        
        Args:
            newdatafile: Output filename or Path
            ignore_scaling: If True, omit final unit scaling. Recommended for raw data export. e.g. unit in F instead of pF.
        """
        # (NT+1,Nt+1)
        newdltsdata = np.zeros((self.T.shape[0]+1,self.t.shape[0]+1))
        # Data Assembly
        newdltsdata[0,0] = 0.0
        newdltsdata[0,1:] = self.t
        newdltsdata[1:,0] = self.T
        if ignore_scaling:
            newdltsdata[1:,1:] = self.C
        else:
            newdltsdata[1:,1:] = self.C / self.data_scaling_factor
        # Format (force E- notation without +/zeros)
        def DLTS_format(x):
            formatted = f"{x:.7E}"
            if 'E-0' in formatted:
                formatted = formatted.replace('E-0', 'E-')
            elif 'E+0' in formatted:
                formatted = formatted.replace('E+0', 'E+')
            return formatted
        # Vectorize the formatting function
        DLTS_format = np.vectorize(DLTS_format)
        newdatafile = Path(newdatafile).resolve()
        np.savetxt(newdatafile, DLTS_format(newdltsdata), fmt='%s', delimiter='\t')
        LOGGER_ODDH.info(f"Transient data saved to: {newdatafile}")

    def __sub__(self, other: 'Data_Loader') -> 'Data_Loader':
        """
        Subtract two Data_Loader instances.
        
        Args:
            other (Data_Loader): The instance to subtract from self.
            
        Returns:
            Data_Loader: A new instance with C = self.C - other.C
        """
        if not isinstance(other, Data_Loader):
            raise TypeError(f"Unsupported operand type(s) for -: 'Data_Loader' and '{type(other).__name__}'")
            
        # merge t
        new_t = np.intersect1d(self.t, other.t)
        if not new_t.any():
            raise ValueError("Time axes (t) mismatch between Data_Loader instances.")
        
        t_indices1 = np.searchsorted(self.t, new_t)
        t_indices2 = np.searchsorted(other.t, new_t)
        C_1 = self.C[:, t_indices1]
        C_2 = other.C[:, t_indices2]
        
        # merge T
        sub_T_tol = self._sub_T_tol
        new_T = []
        T_indices1 = []
        T_indices2 = []
        
        for si,sT in enumerate(self.T):
            for oi,oT in enumerate(other.T):
                if np.abs(sT - oT) < sub_T_tol:
                    new_T.append((sT+oT)/2)
                    T_indices1.append(si)
                    T_indices2.append(oi)
                    break
        new_T = np.array(new_T)
        if not new_T.any():
            raise ValueError("Condition axes (T) mismatch between Data_Loader instances.")
        
        T_indices1 = np.array(T_indices1)
        T_indices2 = np.array(T_indices2)
        C_1 = C_1[T_indices1,:]
        C_2 = C_2[T_indices2,:]
            
        new_C = C_1 / self.data_scaling_factor - C_2 / other.data_scaling_factor
        
        new_transient_data = {
            't': new_t,
            'T': new_T,
            'C': new_C
        }
        new_loader = Data_Loader(
            transient_data=new_transient_data,
            raw_data_scaling_factor=1.0,
            time_shift=0.0,
            T_shift=0.0,
            data_scaling_factor=self.data_scaling_factor,
            data_type=self.data_type,
            data_x_type=self.data_x_type,
            condition_type=self.condition_type,
            logging_level=self.logging_level,
            logging_file=None,
            logging_file_clear=False
        )
        new_loader.transient_data_full_path = self.transient_data_full_path
        return new_loader

    def __sub__(self, other: 'Data_Loader') -> 'Data_Loader':
        """
        Subtract two Data_Loader instances.
        
        Args:
            other (Data_Loader): The instance to subtract from self.
            
        Returns:
            Data_Loader: A new instance with C = self.C - other.C
        """
        if not isinstance(other, Data_Loader):
            raise TypeError(f"Unsupported operand type(s) for -: 'Data_Loader' and '{type(other).__name__}'")
            
        # merge t
        new_t = np.intersect1d(self.t, other.t)
        if not new_t.any():
            raise ValueError("Time axes (t) mismatch between Data_Loader instances.")
        
        t_indices1 = np.searchsorted(self.t, new_t)
        t_indices2 = np.searchsorted(other.t, new_t)
        C_1 = self.C[:, t_indices1]
        C_2 = other.C[:, t_indices2]
        
        # merge T
        sub_T_tol = self._sub_T_tol
        new_T = []
        T_indices1 = []
        T_indices2 = []
        
        for si,sT in enumerate(self.T):
            for oi,oT in enumerate(other.T):
                if np.abs(sT - oT) < sub_T_tol:
                    new_T.append((sT+oT)/2)
                    T_indices1.append(si)
                    T_indices2.append(oi)
                    break
        new_T = np.array(new_T)
        if not new_T.any():
            raise ValueError("Condition axes (T) mismatch between Data_Loader instances.")
        
        T_indices1 = np.array(T_indices1)
        T_indices2 = np.array(T_indices2)
        C_1 = C_1[T_indices1,:]
        C_2 = C_2[T_indices2,:]
            
        new_C = C_1 / self.data_scaling_factor - C_2 / other.data_scaling_factor
        
        new_transient_data = {
            't': new_t,
            'T': new_T,
            'C': new_C
        }
        new_loader = Data_Loader(
            transient_data=new_transient_data,
            raw_data_scaling_factor=1.0,
            time_shift=0.0,
            T_shift=0.0,
            data_scaling_factor=self.data_scaling_factor,
            data_type=self.data_type,
            data_x_type=self.data_x_type,
            condition_type=self.condition_type,
            logging_level=self.logging_level,
            logging_file=None,
            logging_file_clear=False
        )
        new_loader.transient_data_full_path = self.transient_data_full_path
        return new_loader
    
    def __neg__(self) -> 'Data_Loader':
        """
        Negate the Data_Loader instance.
        
        Returns:
            Data_Loader: A new instance with C = -self.C
        """
        new_C = -self.C
        new_transient_data = {
            't': self.t,
            'T': self.T,
            'C': new_C
        }
        new_loader = Data_Loader(
            transient_data=new_transient_data,
            raw_data_scaling_factor=1.0,
            time_shift=0.0,
            T_shift=0.0,
            data_scaling_factor=self.data_scaling_factor,
            data_type=self.data_type,
            data_x_type=self.data_x_type,
            condition_type=self.condition_type,
            logging_level=self.logging_level,
            logging_file=None,
            logging_file_clear=False
        )
        new_loader.transient_data_full_path = self.transient_data_full_path
        return new_loader
    
    def __add__(self, other: 'Data_Loader') -> 'Data_Loader':
        if not isinstance(other, Data_Loader):
            raise TypeError(f"Unsupported operand type(s) for +: 'Data_Loader' and '{type(other).__name__}'")
        neg_other = -other
        return self - neg_other
    
    def __and__(self, other: 'Data_Loader') -> 'Data_Loader':
        """
        Merge two Data_Loader instances based on attribute T (Temperature).
        
        Args:
            other (Data_Loader): The instance to merge with self.
            
        Returns:
            Data_Loader: A new instance with Merged t,T,C
        """
        if not isinstance(other, Data_Loader):
            raise TypeError(f"Unsupported operand type(s) for &: 'Data_Loader' and '{type(other).__name__}'")
        # merge t
        new_t = np.intersect1d(self.t, other.t)
        if not new_t.any():
            raise ValueError("Time axes (t) mismatch between Data_Loader instances.")
        t_indices1 = np.searchsorted(self.t, new_t)
        t_indices2 = np.searchsorted(other.t, new_t)
        C_1 = self.C[:, t_indices1]
        C_2 = other.C[:, t_indices2]
        # merge T
        new_T = np.unique(np.concatenate((self.T,other.T)), sorted=True)
        new_C = np.zeros((len(new_T),len(new_t)))
        for si,sT in enumerate(new_T):
            if sT in self.T:
                new_C[si] = C_1[np.where(self.T==sT)[0][0]] / self.data_scaling_factor
            elif sT in other.T:
                new_C[si] = C_2[np.where(other.T==sT)[0][0]] / other.data_scaling_factor
            else:
                pass
        new_transient_data = {
            't': new_t,
            'T': new_T,
            'C': new_C
        }
        new_loader = Data_Loader(
            transient_data=new_transient_data,
            raw_data_scaling_factor=1.0,
            time_shift=0.0,
            T_shift=0.0,
            data_scaling_factor=self.data_scaling_factor,
            data_type=self.data_type,
            data_x_type=self.data_x_type,
            condition_type=self.condition_type,
            logging_level=self.logging_level,
            logging_file=None,
            logging_file_clear=False
        )
        new_loader.transient_data_full_path = self.transient_data_full_path
        return new_loader
    def copy(self) -> 'Data_Loader':
        new_transient_data = {
            't': self.t,
            'T': self.T,
            'C': self.C / self.data_scaling_factor
        }
        new_loader = Data_Loader(
            transient_data=new_transient_data,
            raw_data_scaling_factor=1.0,
            time_shift=0.0,
            T_shift=0.0,
            data_scaling_factor=self.data_scaling_factor,
            data_type=self.data_type,
            data_x_type=self.data_x_type,
            condition_type=self.condition_type,
            logging_level=self.logging_level,
            logging_file=None,
            logging_file_clear=False
        )
        new_loader.transient_data_full_path = self.transient_data_full_path
        return new_loader

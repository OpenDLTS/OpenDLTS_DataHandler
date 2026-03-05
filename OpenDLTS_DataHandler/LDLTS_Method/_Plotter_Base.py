
__all__ = ["PlotterBase"]

import numpy as np
import scipy.constants as sci_const
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .._typing import *

class PlotterBase:
    def plot_arrh(self, solve_index: int = -1, c_abs_min: float = 1e-5, c_polarity: str = 'both',
                  material: str = 'si', dopant_species: str = 'N', colorbar_scale: str = 'log',
                  point_scale: float = 30, lw: float = 0.8, peak_point_scale: float = 30,
                  peak_lw: float = 0.8, line_lw: float = 1.0, power_T: int = 2, cmap: str = 'coolwarm',
                  data_from_irls: bool = False, use_time_constant: bool = True, figsize: tuple = (12,8),
                  return_widgets: bool = False, fs: int = 14) -> None | Any:
        Ti_list = self.solve_history[solve_index]['Ti_list'] if not data_from_irls else self.irls_solve_history[solve_index]['Ti_list']
        s = self.s
        f = self.solve_history[solve_index]['f'] if not data_from_irls else self.irls_solve_history[solve_index]['f']
        all_x = []
        all_y = []
        all_c = []
        for rTi,Ti in enumerate(Ti_list):
            # -ln(em/T**{power_T})
            y = -np.log(s/self.T[Ti]**power_T)
            # 1/kT (eV^{-1})
            x = 1/self.T[Ti]/sci_const.k*sci_const.e*np.ones(len(y))
            c = f[1:,rTi].flatten()
            all_x.append(x)
            all_y.append(y)
            all_c.append(c)
        arrh_x = np.concatenate(all_x)
        arrh_y = np.concatenate(all_y)
        arrh_c = np.concatenate(all_c)
        sort_idx = np.argsort(np.abs(arrh_c))
        arrh_x = arrh_x[sort_idx]
        arrh_y = arrh_y[sort_idx]
        arrh_c = arrh_c[sort_idx]
        # kwargs for plotting
        self._kwargs_arrh_plotter = {
            'x': arrh_x,
            'y': arrh_y,
            'c': arrh_c,
            'cmap': cmap,
            'point_scale': point_scale,
            'colorbar_scale': colorbar_scale,
            'lw': lw,
            'peak_point_scale': peak_point_scale,
            'peak_lw': peak_lw,
            'material': material,
            'dopant_species': dopant_species,
            'line_lw': line_lw,
            'power_T': power_T,
            'data_plot_dict': self.data_plot_dict,
            'sec_ax_max_n_locator': 7,
            'use_time_constant': use_time_constant,
            'c_abs_min': c_abs_min,
            'c_polarity': c_polarity,
            'figsize': figsize,
            'fs': fs
        }
        from ..Widgets import ARRH_PLOTTER
        if return_widgets:
            return ARRH_PLOTTER(**self._kwargs_arrh_plotter)
        with plt.ion():
            self.arrh_plotter = ARRH_PLOTTER(**self._kwargs_arrh_plotter)
    def plot_ldlts(self, solve_index: int = -1, Ti: int = 0, use_time_constant: bool = True,
                   figsize: tuple = (8,6), fs: int = 14, data_from_irls: bool = False,
                   return_widgets: bool = False) -> None | Any:
        Ti_list = self.solve_history[solve_index]['Ti_list'] if not data_from_irls else self.irls_solve_history[solve_index]['Ti_list']
        if Ti in Ti_list:
            rTi = np.argwhere(Ti_list==Ti).reshape(-1)[0]
        else:
            self.logger.error(f'Ti out of range, Ti_list={Ti_list}')
            return None
        s = self.s
        f = self.solve_history[solve_index]['f'][1:,rTi] if not data_from_irls else self.irls_solve_history[solve_index]['f'][1:,rTi]
        self._kwargs_ldlts_plotter = {
            's': s,
            'f': f,
            'figsize': figsize,
            'data_plot_dict': self.data_plot_dict,
            'use_time_constant': use_time_constant,
            'fs': fs
        }
        from ..Widgets import LDLTS_PLOTTER
        if return_widgets:
            return LDLTS_PLOTTER(**self._kwargs_ldlts_plotter)
        with plt.ion():
            self.ldlts_plotter = LDLTS_PLOTTER(**self._kwargs_ldlts_plotter)
    def plot_ldlts_T(self, solve_index: int = -1, use_time_constant: bool = True,
                     figsize: tuple = (12,8), fs: int = 14, data_from_irls: bool = False,
                     point_scale: float = 30, lw: float = 0.8, c_polarity: str = 'both',
                     c_abs_min: float = 1e-4, colorbar_scale: str = 'log', cmap: str = 'coolwarm',
                     return_widgets: bool = False, ax: Axes | None = None) -> None | Any:
        Ti_list = self.solve_history[solve_index]['Ti_list'] if not data_from_irls else self.irls_solve_history[solve_index]['Ti_list']
        s = self.s
        f = self.solve_history[solve_index]['f'] if not data_from_irls else self.irls_solve_history[solve_index]['f']
        all_x = []
        all_y = []
        all_c = []
        for rTi,Ti in enumerate(Ti_list):
            if use_time_constant:
                y = 1/s
            else:
                y = s
            x = self.T[Ti]*np.ones(len(y))
            c = f[1:, rTi].flatten()
            all_x.append(x)
            all_y.append(y)
            all_c.append(c)
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        all_c = np.concatenate(all_c)
        self._kwargs_ldlts_T_plotter = {
            'x': all_x,
            'y': all_y,
            'c': all_c,
            'figsize': figsize,
            'fs': fs,
            'data_plot_dict': self.data_plot_dict,
            'use_time_constant': use_time_constant,
            'point_scale': point_scale,
            'lw': lw,
            'c_polarity': c_polarity,
            'c_abs_min': c_abs_min,
            'colorbar_scale': colorbar_scale,
            'cmap': cmap,
            'ax': ax
        }
        from ..Widgets import LDLTS_T_PLOTTER
        if return_widgets:
            return LDLTS_T_PLOTTER(**self._kwargs_ldlts_T_plotter)
        with plt.ion():
            self.ldlts_T_plotter = LDLTS_T_PLOTTER(**self._kwargs_ldlts_T_plotter)

    def plot_trans(self, solve_index: int = -1, Ti: int = 0, show_log_time: bool = True,
                   figsize: tuple = (8,6), fs: int = 14, data_from_irls: bool = False,
                   return_widgets: bool = False) -> None | Any:
        Ti_list = self.solve_history[solve_index]['Ti_list'] if not data_from_irls else self.irls_solve_history[solve_index]['Ti_list']
        if Ti in Ti_list:
            rTi = np.argwhere(Ti_list==Ti).reshape(-1)[0]
        else:
            self.logger.error(f'Ti out of range, Ti_list={Ti_list}')
            return None
        t = self.t
        c = self.A_extended @ (self.solve_history[solve_index]['f'][:,rTi] if not data_from_irls else self.irls_solve_history[solve_index]['f'][:,rTi])
        rms = self.solve_history[solve_index]['rms_list'][rTi] if not data_from_irls else self.irls_solve_history[solve_index]['rms_list'][rTi]
        if show_log_time:
            c_exp = self.C[rTi]
        else:
            c_exp = None
        self._kwargs_trans_plotter = {
            't': t,
            'c': c,
            'c_exp': c_exp,
            'figsize': figsize,
            'rms': rms,
            'data_plot_dict': self.data_plot_dict,
            'fs': fs
        }
        from ..Widgets import TRANS_PLOTTER
        if return_widgets:
            return TRANS_PLOTTER(**self._kwargs_trans_plotter)
        with plt.ion():
            self.trans_plotter = TRANS_PLOTTER(**self._kwargs_trans_plotter)
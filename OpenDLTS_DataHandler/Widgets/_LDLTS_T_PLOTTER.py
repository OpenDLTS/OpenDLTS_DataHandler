__all__ = ["LDLTS_T_PLOTTER"]
from .._typing import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.axes import Axes
import numpy as np
class LDLTS_T_PLOTTER:
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray, figsize: tuple = (12,8), fs: int = 14,
                 data_plot_dict: TransientDataPlotType | None = None, use_time_constant: bool = True,
                 point_scale: float | int = 30, lw: float | int = 0.8, c_polarity: str = 'both',
                 c_abs_min: float = 1e-4, colorbar_scale: str = 'log', cmap: str = 'coolwarm',
                 ax: Axes | None = None) -> None:
        if data_plot_dict is None:
            self.data_plot_dict = {}
            self.data_plot_dict['C_plot_factor'] = 1.0
            self.data_plot_dict['T_label'] = 'Temperature [$K$]'
            self.data_plot_dict['tau_label'] = 'Time Constant [$s$]'
            self.data_plot_dict['em_label'] = 'Emission Rate [$s^{-1}$]'
            self.data_plot_dict['LDLTS_label'] = 'LDLTS Signal [$a.u.$]'
        else:
            self.data_plot_dict = data_plot_dict
        fontdict={'family':'DejaVu Sans','size':fs}
        # data preprocessing
        if c_polarity == 'positive':
            x = x[c>=c_abs_min]
            y = y[c>=c_abs_min]
            c = c[c>=c_abs_min]
            colormap = plt.colormaps.get_cmap(cmap)
            cmap = colors.LinearSegmentedColormap.from_list('positive_part', colormap(np.linspace(0.5,1,256)))
            if colorbar_scale == 'lin':
                norm = colors.Normalize(
                    vmin = np.min(c)*data_plot_dict['C_plot_factor'],
                    vmax = np.max(c)*data_plot_dict['C_plot_factor']
                )
            else:
                norm = colors.SymLogNorm(
                    vmin = np.min(c)*data_plot_dict['C_plot_factor'],
                    vmax = np.max(c)*data_plot_dict['C_plot_factor'],
                    linthresh = c_abs_min
                )
        elif c_polarity == 'negative':
            x = x[c<=-c_abs_min]
            y = y[c<=-c_abs_min]
            c = c[c<=-c_abs_min]
            colormap = plt.colormaps.get_cmap(cmap)
            cmap = colors.LinearSegmentedColormap.from_list('negative_part', colormap(np.linspace(0,0.5,256)))
            if colorbar_scale == 'lin':
                norm = colors.Normalize(
                    vmin = np.min(c)*data_plot_dict['C_plot_factor'],
                    vmax = np.max(c)*data_plot_dict['C_plot_factor']
                )
            else:
                norm = colors.SymLogNorm(
                    vmin = np.min(c)*data_plot_dict['C_plot_factor'],
                    vmax = np.max(c)*data_plot_dict['C_plot_factor'],
                    linthresh = c_abs_min
                )
        elif c_polarity == 'both':
            # When the signal is basically positive
            if c[c<=-c_abs_min].size == 0:
                if colorbar_scale == 'lin':
                    norm = colors.Normalize(
                        vmin = -np.max(c)*data_plot_dict['C_plot_factor'],
                        vmax = np.max(c)*data_plot_dict['C_plot_factor']
                    )
                else:
                    norm = colors.SymLogNorm(
                        vmin = -np.max(c)*data_plot_dict['C_plot_factor'],
                        vmax = np.max(c)*data_plot_dict['C_plot_factor'],
                        linthresh = c_abs_min
                    )
            # When the signal is basically negative
            elif c[c>=c_abs_min].size == 0:
                if colorbar_scale == 'lin':
                    norm = colors.Normalize(
                        vmin = np.min(c)*data_plot_dict['C_plot_factor'],
                        vmax = -np.min(c)*data_plot_dict['C_plot_factor']
                    )
                else:
                    norm = colors.SymLogNorm(
                        vmin = np.min(c)*data_plot_dict['C_plot_factor'],
                        vmax = -np.min(c)*data_plot_dict['C_plot_factor'],
                        linthresh = c_abs_min
                    )
            else:
                if colorbar_scale == 'lin':
                    norm = colors.Normalize(
                        vmin = -np.max(np.abs(c))*data_plot_dict['C_plot_factor'],
                        vmax = np.max(np.abs(c))*data_plot_dict['C_plot_factor']
                    )
                else:
                    norm = colors.SymLogNorm(
                        vmin = -np.max(np.abs(c))*data_plot_dict['C_plot_factor'],
                        vmax = np.max(np.abs(c))*data_plot_dict['C_plot_factor'],
                        linthresh = c_abs_min
                    )
        else:
            raise ValueError("polarity = 'positive' | 'negative' | 'both'")
        

        if ax is None:
            self.fig = plt.figure(figsize=figsize, layout='tight')
            self.ax = plt.gca()
            try:
                self.fig.canvas.header_visible = False
                self.fig.canvas.resizable = False
            except AttributeError:
                pass
        else:
            self.ax = ax
            self.fig = ax.figure

        # The point with the largest c is drawn in the front, and all_x, all_y, all_c are sorted.
        sort_idx = np.argsort(np.abs(c))
        all_x = x[sort_idx]
        all_y = y[sort_idx]
        all_c = c[sort_idx]
        self.ax.scatter(
            x=all_x,
            y=all_y,
            c=all_c*data_plot_dict['C_plot_factor'],
            cmap=cmap,
            norm=norm,
            marker='o',
            edgecolors='face',
            facecolors='none',
            s=point_scale,  # Control point size
            linewidths=lw  # Control edge line width
        )
        self.ax.set_yscale('log')
        self.ax.set_xlabel(data_plot_dict['T_label'],fontdict=fontdict)
        if use_time_constant:
            self.ax.set_ylabel(data_plot_dict['tau_label'],fontdict=fontdict)
        else:
            self.ax.set_ylabel(data_plot_dict['em_label'],fontdict=fontdict)
        # Add a color bar to show the temperature map
        self.cbar = self.fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            label=data_plot_dict['LDLTS_label'],
            ax=self.ax,
            format='%.0e'
        )
        self.cbar.ax.set_ylabel(data_plot_dict['LDLTS_label'],fontdict=fontdict)
        for tt in self.ax.get_xticklabels():
            tt.set_fontsize(fs)
            tt.set_fontfamily(fontdict['family'])
        for tt in self.ax.get_yticklabels():
            tt.set_fontsize(fs)
            tt.set_fontfamily(fontdict['family'])
        for tt in self.cbar.ax.get_yticklabels():
            tt.set_fontsize(fs)
            tt.set_fontfamily(fontdict['family'])
        self.box = self.fig.canvas
__all__ = ["LDLTS_PLOTTER"]
from .._typing import *
import matplotlib.pyplot as plt
import numpy as np

class LDLTS_PLOTTER:
    def __init__(self, s: np.ndarray, f: np.ndarray, figsize: tuple = (8,6), fs: int = 14,
                 data_plot_dict: TransientDataPlotType | None = None, use_time_constant: bool = True) -> None:
        if data_plot_dict is None:
            self.data_plot_dict = {}
            self.data_plot_dict['C_plot_factor'] = 1.0
            self.data_plot_dict['em_label'] = 'Emission Rate [$s^{-1}$]'
            self.data_plot_dict['tau_label'] = 'Time Constant [$s$]'
            self.data_plot_dict['LDLTS_label'] = 'LDLTS Signal [$a.u.$]'
        else:
            self.data_plot_dict = data_plot_dict
        fontdict={'family':'DejaVu Sans','size':fs}
        self.fig = plt.figure(figsize=figsize, layout='tight')
        self.ax = plt.gca()
        self.fig.canvas.header_visible = False
        self.fig.canvas.resizable = False
        self.s = s
        self.f = f
        self.use_time_constant = use_time_constant
        if self.use_time_constant:
            self.ax.plot(1/self.s, self.f * self.data_plot_dict['C_plot_factor'])
            self.ax.set_xlabel(self.data_plot_dict['tau_label'], fontdict=fontdict)
        else:
            self.ax.plot(self.s, self.f * self.data_plot_dict['C_plot_factor'])
            self.ax.set_xlabel(self.data_plot_dict['em_label'], fontdict=fontdict)
        self.ax.set_ylabel(self.data_plot_dict['LDLTS_label'], fontdict=fontdict)
        self.ax.set_xscale('log')
        for tt in self.ax.get_xticklabels():
            tt.set_fontsize(fs)
            tt.set_fontfamily(fontdict['family'])
        for tt in self.ax.get_yticklabels():
            tt.set_fontsize(fs)
            tt.set_fontfamily(fontdict['family'])
        self.box = self.fig.canvas
    def _update_data(self, s: np.ndarray, f: np.ndarray) -> None:
        self.s = s
        self.f = f
        if self.use_time_constant:
            self.ax.lines[0].set_data(1/self.s, self.f * self.data_plot_dict['C_plot_factor'])
        else:
            self.ax.lines[0].set_data(self.s, self.f * self.data_plot_dict['C_plot_factor'])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        
__all__ = ["TRANS_PLOTTER"]
from .._typing import *
import matplotlib.pyplot as plt
import numpy as np

class TRANS_PLOTTER:
    def __init__(self, t: np.ndarray, c: np.ndarray, t_exp: np.ndarray | None = None, c_exp: np.ndarray | None = None,
                 figsize: tuple = (8,6), data_plot_dict: TransientDataPlotType | None = None, fs: int = 18,
                 rms: float | None = None) -> None:
        if data_plot_dict is None:
            self.data_plot_dict = {}
            self.data_plot_dict['C_plot_factor'] = 1.0
            self.data_plot_dict['t_label'] = 'Time [$s$]'
            self.data_plot_dict['C_label'] = 'Transient Signal [$a.u.$]'
        else:
            self.data_plot_dict = data_plot_dict
        fontdict={'family':'DejaVu Sans','size':fs}
        self.fig = plt.figure(figsize=figsize, layout='tight')
        self.ax = plt.gca()
        self.ax2 = self.ax.twiny()
        self.fig.canvas.header_visible = False
        self.fig.canvas.resizable = False
        self.t = t
        self.c = c
        self.c_exp = c_exp
        self.t_exp = t_exp
        self._colorlist = list(plt.cm.colors.TABLEAU_COLORS.values())
        if self.c_exp is not None and self.t_exp is not None:
            self.ax.plot(self.t_exp, self.c_exp * self.data_plot_dict['C_plot_factor'], 'x', label='Exp.',
                         color=self._colorlist[0])
            self.ax2.plot(self.t_exp, self.c_exp * self.data_plot_dict['C_plot_factor'], 'x', label='Exp.',
                          color=self._colorlist[0])
        elif self.t_exp is None and self.c_exp is not None:
            self.ax.plot(self.t, self.c_exp * self.data_plot_dict['C_plot_factor'], 'x', label='Exp.', color=self._colorlist[0])
            self.ax2.plot(self.t, self.c_exp * self.data_plot_dict['C_plot_factor'], 'x', label='Exp.', color=self._colorlist[0])
        self._ax_fit_color = 'black'
        self.ax.plot(self.t, self.c * self.data_plot_dict['C_plot_factor'], label=f'Fit, rms={rms:.5f}',
                     color=self._ax_fit_color)
        self.ax2.plot(self.t, self.c * self.data_plot_dict['C_plot_factor'], label=f'Fit, rms={rms:.5f}',
                      color=self._colorlist[1])

        self.ax.set_xlabel(self.data_plot_dict['t_label'],fontdict=fontdict, color=self._ax_fit_color)
        self.ax.set_ylabel(self.data_plot_dict['C_label'],fontdict=fontdict, color=self._ax_fit_color)
        self.ax2.set_xlabel(self.data_plot_dict['t_label'],fontdict=fontdict, color=self._colorlist[1])
        self.ax2.set_ylabel(self.data_plot_dict['C_label'],fontdict=fontdict, color=self._colorlist[1])
        self.ax2.set_xscale('log')
        self.ax2.tick_params(axis='x', colors=self._colorlist[1])
        self.ax2.spines["top"].set_color(self._colorlist[1])
        self.ax2.spines["right"].set_color(self._colorlist[1])
        h1,l1 = self.ax.get_legend_handles_labels()
        h2,l2 = self.ax2.get_legend_handles_labels()
        new_handles = []
        for th1,th2 in zip(h1,h2):
            if not new_handles:
                new_handles.append(th1)
            else:
                new_handles.append((th1, th2))
        from matplotlib.legend_handler import HandlerTuple
        self.ax.legend(new_handles,l1,prop=fontdict,handler_map={tuple:HandlerTuple(ndivide=None)})
        for tt in self.ax.get_xticklabels():
            tt.set_fontsize(fs)
            tt.set_fontfamily(fontdict['family'])
        for tt in self.ax.get_yticklabels():
            tt.set_fontsize(fs)
            tt.set_fontfamily(fontdict['family'])
        for tt in self.ax2.get_xticklabels():
            tt.set_fontsize(fs)
            tt.set_fontfamily(fontdict['family'])
        self.box = self.fig.canvas
    def _update_data(self, t: np.ndarray, c: np.ndarray, rms: float | None = None,
                     t_exp: np.ndarray | None = None, c_exp: np.ndarray | None = None) -> None:
        self.t = t
        self.c = c
        self.c_exp = c_exp
        self.t_exp = t_exp
        if self.c_exp is not None and self.t_exp is not None:
            self.ax.lines[0].set_data(self.t_exp, self.c_exp * self.data_plot_dict['C_plot_factor'])
            self.ax2.lines[0].set_data(self.t_exp, self.c_exp * self.data_plot_dict['C_plot_factor'])
        elif self.t_exp is None and self.c_exp is not None:
            self.ax.lines[0].set_data(self.t, self.c_exp * self.data_plot_dict['C_plot_factor'])
            self.ax2.lines[0].set_data(self.t, self.c_exp * self.data_plot_dict['C_plot_factor'])
        self.ax.lines[1].set_data(self.t, self.c * self.data_plot_dict['C_plot_factor'])
        self.ax2.lines[1].set_data(self.t, self.c * self.data_plot_dict['C_plot_factor'])
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # update legend text
        self.ax.lines[1].set_label(f'Fit, rms={rms:.5f}')
        self.ax2.lines[1].set_label(f'Fit, rms={rms:.5f}')
        leg = self.ax.get_legend()
        if leg:
            leg.get_texts()[1].set_text(f'Fit, rms={rms:.5f}')
        
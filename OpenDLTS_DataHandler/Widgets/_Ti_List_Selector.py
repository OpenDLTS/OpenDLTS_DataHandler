__all__ = ['Ti_list_selector']

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

class Ti_list_selector:
    def __init__(self, T, Ti0=0, Ti1=None, min_step_coeff=1, init_showui=True):
        self.T = T
        self.Ti_list_selected=[]
        self.T_list_selected=[]
        if Ti1 is None:
            Ti1 = len(self.T)-1
        self._Ti = widgets.IntRangeSlider(
            value=[Ti0, Ti1],
            min=0,
            max=len(self.T)-1,
            step=1,
            description='Ti Range:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=False,
            readout_format='d',
            layout=widgets.Layout(width='40%')
        )
        self._T_label = widgets.Label(
            value=f"{T[Ti0]}K-{T[Ti1]}K"
        )
        self._selected_T_label = widgets.Label(
            value=f"0K-0K"
        )
        self._min_step_coeff = widgets.FloatSlider(
            value=min_step_coeff,
            min=0.01,
            max=10.0,
            step=0.01,
            description='Step Coeff.:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(width='40%')
        )
        plt.ioff()
        self.fig = plt.figure(figsize=(4,4),layout='tight')
        self.fig.canvas.header_visible = False
        self.fig.canvas.resizable = False
        self.fig.canvas.footer_visible = False
        self.ax = plt.gca()
        self.ax.plot(T,1/T,'o')
        self.ax.set_yticks([])
        self.ax.set_xlabel('T')
        self.ax.set_ylabel('1/T')
        self.box = widgets.VBox([widgets.HBox([self._Ti,self._T_label]),widgets.HBox([self._min_step_coeff,self._selected_T_label]),self.fig.canvas])
        self._set_event()
        T0 = self.T[self._Ti.value[0]]
        T1 = self.T[self._Ti.value[1]]
        self._min_inv_dT_step = 1/T0 - 1/self.T[self._Ti.value[0]+1]
        self._sample_inv_dT = self._min_step_coeff.value*self._min_inv_dT_step
        self._sample_T = 1/np.arange(1/T0,1/T1,-self._sample_inv_dT)
        self._update_plot()
        if init_showui:
            self.show_ui()
        # Ti_selector_box
        self._iTi_selector = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.Ti_list_selected)-1,
            step=1,
            description='Ti selector:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=False,
            readout_format='d',
            layout=widgets.Layout(width='40%')
        )
        self._Ti_selector_label = widgets.Label(
            value=f"Ti={self.selected_Ti}(T={self.T_list_selected[self._iTi_selector.value]:.2f}K)"
        )
        self.Ti_selector_box = widgets.HBox([self._iTi_selector, self._Ti_selector_label])
        self._iTi_selector.observe(self._obs_iTi_selector, names='value')
    def show_ui(self):
        display(self.box)
    def show_ui_Ti_selector(self):
        display(self.Ti_selector_box)
    def _set_event(self):
        self._Ti.observe(self._obs_Ti, names='value')
        self._min_step_coeff.observe(self._obs_min_step_coeff, names='value')
    @property
    def selected_Ti(self):
        return int(self.Ti_list_selected[self._iTi_selector.value])
    @property
    def selected_T(self):
        return self.T_list_selected[self._iTi_selector.value]
    def _obs_iTi_selector(self, change):
        self._Ti_selector_label.value=f"Ti={self.Ti_list_selected[change['new']]}(T={self.T_list_selected[change['new']]:.2f}K)"
    def _update_iTi_selector(self):
        self._iTi_selector.max = len(self.Ti_list_selected)-1
        self._Ti_selector_label.value = f"Ti={self.selected_Ti}(T={self.T_list_selected[self._iTi_selector.value]:.2f}K)"
    def _obs_Ti(self, change):
        T0 = self.T[change['new'][0]]
        T1 = self.T[change['new'][1]]
        self._min_inv_dT_step = 1/T0 - 1/self.T[change['new'][0]+1]
        self._sample_inv_dT = self._min_step_coeff.value*self._min_inv_dT_step
        self._sample_T = 1/np.arange(1/T0,1/T1,-self._sample_inv_dT)
        self._update_plot()
        self._update_iTi_selector()
    def _obs_min_step_coeff(self, change):
        T0 = self.T[self._Ti.value[0]]
        T1 = self.T[self._Ti.value[1]]
        self._min_inv_dT_step = 1/T0 - 1/self.T[self._Ti.value[0]+1]
        self._sample_inv_dT = change['new']*self._min_inv_dT_step
        self._sample_T = 1/np.arange(1/T0,1/T1,-self._sample_inv_dT)
        self._update_plot()
        self._update_iTi_selector()
    def _update_plot(self):
        self.Ti_list_selected = []
        for isT in self._sample_T:
            if len(self.Ti_list_selected)>0:
                temp_Ti = np.searchsorted(self.T,isT)
                if self.Ti_list_selected[-1] != temp_Ti:
                    self.Ti_list_selected.append(temp_Ti)
            else:
                temp_Ti = np.searchsorted(self.T,isT)
                self.Ti_list_selected.append(temp_Ti)
        self.T_list_selected = self.T[np.array(self.Ti_list_selected)]
        if len(self.ax.lines)==1:
            self.ax.plot(self.T_list_selected,1/self.T_list_selected,'x')
        else:
            self.ax.lines[1].set_data(self.T_list_selected,1/self.T_list_selected)

        if len(self.ax.lines)==2:
            self.ax.plot(self.T[0]*np.ones_like(self.T_list_selected),1/self.T_list_selected,'s',markersize=2)
        else:
            self.ax.lines[2].set_data(self.T[0]*np.ones_like(self.T_list_selected),1/self.T_list_selected)
            
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        T0 = self.T[self._Ti.value[0]]
        T1 = self.T[self._Ti.value[1]]
        self._T_label.value = f"{T0:.2f}K(idx={self._Ti.value[0]})-{T1:.2f}K(idx={self._Ti.value[1]}) ($N_{{T}}$={len(self.Ti_list_selected)})"
        self._selected_T_label.value = f"selected T: [{self.T_list_selected[0]:.2f}K({self.Ti_list_selected[0]}), {self.T_list_selected[1]:.2f}K({self.Ti_list_selected[1]}), ..."+f"{self.T_list_selected[-2]:.2f}K({self.Ti_list_selected[-2]}), {self.T_list_selected[-1]:.2f}K({self.Ti_list_selected[-1]})]"

"""
Interactive DLTS (Deep Level Transient Spectroscopy) visualization tool.

This module provides an IPython widget-based interface for exploring DLTS data 
with different analysis modes, correlation functions, and visualization options.

Public Classes:
    DLTS_Viewer_Box: Main visualization container providing interactive DLTS plotting.

Dependencies:
    numpy, matplotlib, ipywidgets, threading, scipy.optimize, OpenDLTS_DataHandler
"""
__all__ = ['DLTS_Viewer_Box']

import numpy as np
import ipywidgets as widgets
import threading
import matplotlib.pyplot as plt
from scipy.optimize import fsolve as sci_opt_fsolve
from scipy.optimize import minimize_scalar as sci_opt_min_sca
from .._DLTS_CORRELATION_FUNCTION import DLTS_CORRELATION_FUNCTION
from .._Data_Loader import Data_Loader
from .._config import *

# DLTS_Viewer_Box
class DLTS_Viewer_Box:
    """
    Interactive widget container for DLTS data visualization and analysis.

    Provides four operation modes:
    - Fix t2/t1 ratio: Automatically computes t2 based on fixed ratio to t1
    - Fix rate window: Solves for t2 that produces constant emission rate
    - Manually: Full manual control of both t1 and t2
    - Correlation function: Uses specialized DLTS correlation functions

    Attributes:
        parent: Parent object containing DLTS data
        logger_name (str): Name for logging operations
        logging_level (str): Verbosity level for logging
        fig1 (Figure): Primary matplotlib figure object
        plot_out_image1 (FigureCanvas): Canvas for embedded plot display

    Args:
        parent: Data container object with DLTS data arrays
        logging_level: Verbosity level ('DEBUG', 'INFO', 'WARNING', etc.)
    """
    def __init__(self, dlts_data: Data_Loader, show_ui: bool = True):
        """
        Initialize DLTS viewer box and UI components.

        Args:
            parent: Data container object with DLTS data arrays
            logging_level: Verbosity level for logging system
        """
        self.parent = dlts_data
        self.plot_dlts_lock = threading.Lock()
        self.plot_x = np.array([0])
        self.plot_y = np.array([0])
        self.plot_em = 0.0
        self._create_ui()
        self._set_event()
        self.init_dlts_mode0()
        self.logger = LOGGER_ODDH.getChild('DLTS_Viewer_Box')
        if show_ui:
            display(self.box)

    def _create_ui(self):
        """
        Construct interactive widget interface components.
        
        Creates:
        - Plot mode selector radio buttons
        - Time constant ratio/rate window controls
        - Temperature/time selectors with validation
        - Correlation function dropdown and parameters
        - Matplotlib figure with constrained layout
        - Data export controls
        """
        self.dlts_plot_mode = widgets.RadioButtons(
            options=['Fix t2/t1 ratio', 'Fix rate window', 'Manually', 'Correalation function'],
            description='DLTS Plot Mode:',
            disabled=False
        )
        self.t2t1_ratio = widgets.BoundedFloatText(
            value=10.0,
            min=1.001,
            max=100000,
            disabled=False,
            description='t2/t1 ratio:',
            layout=widgets.Layout(width='30%'),
            style={'description_width': 'initial'}
        )
        self.rate_window = widgets.BoundedFloatText(
            value=600.0,
            min=1.00,
            max=100000,
            disabled=False,
            description='rate window:',
            layout=widgets.Layout(width='30%'),
            style={'description_width': 'initial'}
        )
        self.log_box = widgets.Label('')
        self.t1 = widgets.IntSlider(
            value=0,
            min=0,
            max=1000,
            step=1,
            description='Set t1:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            layout=widgets.Layout(width='90%'),
            readout=False,
            readout_format='d'
        )
        self.t2 = widgets.IntSlider(
            value=0,
            min=0,
            max=1000,
            step=1,
            description='Set t2:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            layout=widgets.Layout(width='90%'),
            readout=False,
            readout_format='d'
        )
        def get_class_method_names(cls):
            method_names = []
            for name, value in cls.__dict__.items():
                if callable(value) or isinstance(value, (classmethod, staticmethod)):
                    method_names.append(name)
            return method_names
        self.cf = widgets.Dropdown(
            options=get_class_method_names(DLTS_CORRELATION_FUNCTION),
            value=get_class_method_names(DLTS_CORRELATION_FUNCTION)[2],
            description='Function:',
            disabled=False,
        )
        self.cf_tc0 = widgets.BoundedFloatText(
            value=self.parent.t[0],
            min=self.parent.t[0],
            max=self.parent.t[-1],
            disabled=False,
            description='tc0:',
            layout=widgets.Layout(width='10%'),
            style={'description_width': 'initial'}
        )
        self.cf_tc1 = widgets.BoundedFloatText(
            value=self.parent.t[1],
            min=self.parent.t[0],
            max=self.parent.t[-1],
            disabled=False,
            description='tc1:',
            layout=widgets.Layout(width='10%'),
            style={'description_width': 'initial'}
        )
        self.cf_target_em = widgets.BoundedFloatText(
            value=100,
            min=10,
            max=1000,
            disabled=False,
            description='Target em:',
            layout=widgets.Layout(width='10%'),
            style={'description_width': 'initial'}
        )
        self.cf_target_em_btn = widgets.Button(
            description='Find em',
            disabled=False,
            button_style='danger',
            tooltip='Find target em by changing tc1',
            icon='bullseye',
            layout=widgets.Layout(width='8%')
        )
        self.cf_replot_btn = widgets.Button(
            description='Replot',
            disabled=False,
            button_style='info',
            tooltip='Replot',
            icon='reply',
            layout=widgets.Layout(width='7%')
        )
        self.cf_use_opt_tc0 = widgets.Checkbox(
            value=True,
            description='Use Opt. tc0',
            disabled=False
        )
        #self.plot_out = widgets.Output()
        plt.ioff()
        self.fig1 = plt.figure(figsize = (12,3), layout='tight')
        #self.fig1 = plt.figure(figsize = (12,2.5))
        self.fig1.canvas.header_visible = False
        self.fig1.canvas.resizable = False
        self.fig1.canvas.footer_visible = False
        x=np.array([0])
        y=np.array([0])
        self.ax1 = plt.gca()
        self.ax1.grid()
        self.im1 = self.ax1.plot(x,y)
        self.plot_out_image1 = self.fig1.canvas
        self.plot_out_image1.layout = widgets.Layout(width='1200px',height='300px')
        self.ax1.text(-0.01,0,'Plot DLTS spectra here', fontsize=20)
        self.save_file_name = widgets.Text(
            #value=f'{str(self.parent.original_data_file_name)}.em_{round(self.plot_em,2)}_dltsplot',
            value = self.parent.transient_data_full_path.with_suffix('.dltsplot').name,
            placeholder='Output File Path',
            disabled=False,
            layout=widgets.Layout(width='40%')
        )
        self.save_file_btn = widgets.Button(
            description='Save File',
            disabled=False,
            button_style='info',
            tooltip='Save current plot data',
            icon='file',
            layout=widgets.Layout(width='10%')
        )
        self.box = widgets.VBox([
            widgets.HBox([self.dlts_plot_mode, widgets.VBox([widgets.HBox([self.t2t1_ratio, self.rate_window]), self.t1, self.t2,],layout=widgets.Layout(width='70%'))]),
            widgets.HBox([self.cf, self.cf_use_opt_tc0, self.cf_tc0, self.cf_tc1, self.cf_target_em, self.cf_target_em_btn, self.cf_replot_btn]),
            self.log_box,
            #self.plot_out,
            self.plot_out_image1,
            widgets.HBox([self.save_file_name, self.save_file_btn])
        ])
    def _set_event(self):
        """Register event handlers for all interactive widgets."""
        self.dlts_plot_mode.observe(self._observe_dlts_plot_mode, names='value')
        self.t2t1_ratio.observe(self._observe_t2t1_ratio, names='value')
        self.rate_window.observe(self._observe_rate_window, names='value')
        self.t1.observe(self._observe_t1, names='value')
        self.t2.observe(self._observe_t2, names='value')
        self.cf.observe(self._observe_cf, names='value')
        self.cf_use_opt_tc0.observe(self._observe_cf_use_opt_tc0, names='value')
        self.cf_tc0.observe(self._observe_cf_tc0, names='value')
        self.cf_tc1.observe(self._observe_cf_tc1, names='value')
        self.save_file_btn.on_click(self._click_save_file_btn)
        self.cf_target_em_btn.on_click(self._click_cf_target_em_btn)
        self.cf_replot_btn.on_click(self._click_cf_replot_btn)
        
    def _observe_dlts_plot_mode(self, change):
        """
        Handle plot mode selection changes.
        
        Routes to appropriate initialization method based on selected mode.
        
        Args:
            change: Widget change event object
        """
        if self.dlts_plot_mode.value == 'Fix t2/t1 ratio':
            self.init_dlts_mode0()
        elif self.dlts_plot_mode.value == 'Fix rate window':
            self.init_dlts_mode1()
        elif self.dlts_plot_mode.value == 'Manually':
            self.init_dlts_mode2()
        elif self.dlts_plot_mode.value == 'Correalation function':
            self.init_dlts_mode3()
    def _observe_t2t1_ratio(self, change):
        """
        Handle changes to t2/t1 ratio parameter.
        
        Triggers recalculation and plotting for fixed t2/t1 mode.
        
        Args:
            change: Widget change event object
        """
        self.plot_dlts_fix_t2t1_ratio()
    def _observe_rate_window(self, change):
        """
        Handle changes to rate window parameter.
        
        Triggers recalculation and plotting for fixed rate window mode.
        
        Args:
            change: Widget change event object
        """
        self.plot_dlts_fix_rate_window()
    def _observe_t1(self, change):
        """
        Handle manual t1 slider changes.
        
        Routes recalculation to appropriate mode (ratio, rate window, or manual).
        
        Args:
            change: Widget change event object
        """
        if self.dlts_plot_mode.value == 'Fix t2/t1 ratio':
            self.plot_dlts_fix_t2t1_ratio()
        elif self.dlts_plot_mode.value == 'Fix rate window':
            self.plot_dlts_fix_rate_window()
        elif self.dlts_plot_mode.value == 'Manually':
            self.plot_dlts_manually()
    def _observe_t2(self, change):
        """
        Handle manual t2 slider changes.
        
        Triggers manual mode recalculation when applicable.
        
        Args:
            change: Widget change event object
        """
        if self.dlts_plot_mode.value == 'Fix t2/t1 ratio':
            pass
        elif self.dlts_plot_mode.value == 'Fix rate window':
            pass
        elif self.dlts_plot_mode.value == 'Manually':
            self.plot_dlts_manually()
    def _observe_cf(self, change):
        """
        Handle correlation function selection changes.
        
        Triggers replotting with selected correlation function.
        
        Args:
            change: Widget change event object
        """
        self.plot_dlts_cf()
        self._update_cf_target_em_max_min()
    def _observe_cf_use_opt_tc0(self, change):
        """
        Handle optimized tc0 checkbox changes.
        
        Enables/disables tc0 input based on checkbox state.
        
        Args:
            change: Widget change event object
        """
        if self.cf_use_opt_tc0.value:
            self.cf_tc0.disabled = True
        else:
            self.cf_tc0.disabled = False
        self.plot_dlts_cf()
    def _observe_cf_tc0(self, change):
        """
        Handle tc0 parameter changes.
        
        Triggers correlation function replotting.
        
        Args:
            change: Widget change event object
        """
        self.plot_dlts_cf()
    def _observe_cf_tc1(self, change):
        """
        Handle tc1 parameter changes.
        
        Triggers correlation function replotting.
        
        Args:
            change: Widget change event object
        """
        self.plot_dlts_cf()
    def _click_save_file_btn(self, b):
        """
        Handle data export button click.
        
        Saves current plot data to text file using numpy.savetxt.
        
        Args:
            b: Button click event object
        """
        data_need_save = np.concatenate((self.plot_x.reshape(-1,1),self.plot_y.reshape(-1,1)),axis=1)
        #np.savetxt(f'{self.parent.original_data_file_path}{self.save_file_name.value}', data_need_save,fmt='%s',delimiter='\t')
        np.savetxt(self.parent.transient_data_full_path.parent / self.save_file_name.value, data_need_save, fmt='%s',delimiter='\t')
    def init_dlts_mode0(self):
        """
        Initialize UI for 'Fix t2/t1 ratio' mode.
        
        Configures:
        - Enables t1 control and ratio input
        - Disables t2 control and correlation parameters
        - Sets slider boundaries
        """
        self.t1.value = 0
        self.t1.min = 0
        self.t1.max = len(self.parent.t)-1
        self.cf.disabled = True
        self.cf_use_opt_tc0.disabled = True
        self.cf_tc0.disabled = True
        self.cf_tc1.disabled = True
        self.cf_target_em.disabled = True
        self.cf_target_em_btn.disabled = True
        self.t1.disabled = False
        self.t2.disabled = True
        self.t2t1_ratio.disabled = False
        self.rate_window.disabled = True
    def init_dlts_mode1(self):
        """
        Initialize UI for 'Fix rate window' mode.
        
        Configures:
        - Enables t1 control and rate window input
        - Disables t2 control and correlation parameters
        - Sets slider boundaries
        """
        self.t1.value = 0
        self.t1.min = 0
        self.t1.max = len(self.parent.t)-1
        self.cf.disabled = True
        self.cf_use_opt_tc0.disabled = True
        self.cf_tc0.disabled = True
        self.cf_tc1.disabled = True
        self.cf_target_em.disabled = True
        self.cf_target_em_btn.disabled = True
        self.t1.disabled = False
        self.t2.disabled = True
        self.t2t1_ratio.disabled = True
        self.rate_window.disabled = False
    def init_dlts_mode2(self):
        """
        Initialize UI for manual mode.
        
        Configures:
        - Enables both t1 and t2 controls
        - Disables ratio/rate window and correlation parameters
        - Sets slider boundaries
        """
        self.t1.value = 0
        self.t1.min = 0
        self.t1.max = len(self.parent.t)-1
        self.t2.value = 0
        self.t2.min = 0
        self.t2.max = len(self.parent.t)-1
        self.cf.disabled = True
        self.cf_use_opt_tc0.disabled = True
        self.cf_tc0.disabled = True
        self.cf_tc1.disabled = True
        self.cf_target_em.disabled = True
        self.cf_target_em_btn.disabled = True
        self.t1.disabled = False
        self.t2.disabled = False
        self.t2t1_ratio.disabled = True
        self.rate_window.disabled = True
    def init_dlts_mode3(self):
        """
        Initialize UI for correlation function mode.
        
        Configures:
        - Disables time selectors
        - Enables correlation function parameters
        - Initializes time constants to valid ranges
        - Configures optimized tc0 state
        """
        self.t1.value = 0
        self.t1.min = 0
        self.t1.max = len(self.parent.t)-1
        self.t2.value = 0
        self.t2.min = 0
        self.t2.max = len(self.parent.t)-1
        self.cf.disabled = False
        self.cf_use_opt_tc0.disabled = False
        self.cf_tc0.value = self.parent.t[0]
        self.cf_tc0.min = self.parent.t[0]
        self.cf_tc0.max = self.parent.t[-1]
        self.cf_tc1.value = self.parent.t[-1]
        self.cf_tc1.min = self.parent.t[0]
        self.cf_tc1.max = self.parent.t[-1]
        self.cf_target_em.disabled = False
        self.cf_target_em_btn.disabled = False
        if self.cf_use_opt_tc0:
            self.cf_tc0.disabled = True
        else:
            self.cf_tc0.disabled = False
        self.cf_tc1.disabled = False
        self._update_cf_target_em_max_min()
        self.t1.disabled = True
        self.t2.disabled = True
        self.t2t1_ratio.disabled = True
        self.rate_window.disabled = True
    def _update_cf_target_em_max_min(self):
        dlts_cf = getattr(DLTS_CORRELATION_FUNCTION, self.cf.value)()
        max_tc1 = self.parent.t[-1]
        max_tc0 = max_tc1 * dlts_cf.optimum_td_tc_ratio / (1+dlts_cf.optimum_td_tc_ratio)
        min_tc0 = self.parent.t[1]
        min_tc1 = min_tc0 / dlts_cf.optimum_td_tc_ratio * (1+dlts_cf.optimum_td_tc_ratio)
        _,_,rw_min = dlts_cf(self.parent.t, self.parent.T, self.parent.C, use_opt_ratio=True, tc0=max_tc0, tc1=max_tc1)
        rw_min = rw_min[0]
        _,_,rw_max = dlts_cf(self.parent.t, self.parent.T, self.parent.C, use_opt_ratio=True, tc0=min_tc0, tc1=min_tc1)
        rw_max = rw_max[0]
        self.cf_target_em.min = rw_min
        self.cf_target_em.max = rw_max
    def _click_cf_replot_btn(self, b):
        self.plot_dlts_cf()
    def _click_cf_target_em_btn(self, b):
        with self.plot_dlts_lock:
            self.plot_dlts_thread = threading.Thread(
                target=self._click_cf_target_em_btn_thread_fun
            )
            self.plot_dlts_thread.start()
    def _click_cf_target_em_btn_thread_fun(self):
        dlts_cf = getattr(DLTS_CORRELATION_FUNCTION, self.cf.value)()
        def objective(tc1):
            _,_,temp_rw = dlts_cf(self.parent.t, self.parent.T, self.parent.C, use_opt_ratio=self.cf_use_opt_tc0.value, tc0=-1, tc1=tc1)
            return (self.cf_target_em.value -temp_rw[0])**2
        result = sci_opt_min_sca(
            objective,
            bounds=(self.cf_tc1.min, self.cf_tc1.max),
            method='bounded'
        )
        if not result.success:
            self.log_box.value = f"Can't find tc1 when em={self.cf_target_em.value:.1f} $s^{{-1}}$"
        else:
            # plot
            for texti in self.ax1.texts:
                texti.remove()
            self.log_box.value = ''
            tc1 = result.x
            tc0 = tc1 * dlts_cf.optimum_td_tc_ratio / (1+dlts_cf.optimum_td_tc_ratio)
            self.cf_tc1.value = tc1
            self.cf_tc0.value = tc0
            self.plot_x = self.parent.T
            _,self.plot_y,self.plot_rw = dlts_cf(self.parent.t, self.parent.T, self.parent.C, use_opt_ratio=self.cf_use_opt_tc0.value, tc0=tc0, tc1=tc1)
            self.plot_em = self.plot_rw[0]
            self.save_file_name.value = self.parent.transient_data_full_path.with_suffix('').name + f'_em_{round(self.plot_em,2)}_cf_{self.cf.value}'+ '.dltsplot'
            for li in self.ax1.lines:
                li.remove()
            self.ax1.plot(self.plot_x,self.plot_y*self.parent.data_plot_dict['C_plot_factor'],'-x',color='black', label=rf'em={str(round(self.plot_em,2))}$s^{-1}$')
            self.ax1.grid(True)
            self.ax1.set_xlabel(self.parent.data_plot_dict['T_label'],fontsize=14)
            self.ax1.set_ylabel(self.parent.data_plot_dict['DLTS_label'],fontsize=14)
            self.ax1.legend(loc='lower right')
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.plot_out_image1.draw()
            self.plot_out_image1.flush_events()
            f_tc0 = '{:.3e}'.format(tc0)
            f_tc1 = '{:.3e}'.format(tc1)
            if self.cf_use_opt_tc0.value:
                self.log_box.value = f'{self.cf.value}(order={dlts_cf.order}). Current tc0={f_tc0}(Calculated from opt. td/tc ratio={dlts_cf.optimum_td_tc_ratio}), tc1={f_tc1}'
            else:
                self.log_box.value = f'{self.cf.value}(order={dlts_cf.order}). Current tc0={f_tc0}, tc1={f_tc1}'
    def plot_dlts_cf(self):
        """
        Launch thread for correlation function plotting.
        
        Uses threading lock to prevent concurrent plot operations.
        """
        with self.plot_dlts_lock:
            self.plot_dlts_thread = threading.Thread(
                target=self.plot_dlts_cf_thread_fun
            )
            self.plot_dlts_thread.start()
    def plot_dlts_cf_thread_fun(self):
        """
        Worker function for correlation function plotting.
        
        Core operations:
        - Validates time constant parameters
        - Calculates DLTS response using selected function
        - Updates plot with emission rate information
        - Manages logging and error messages
        """
        for texti in self.ax1.texts:
            texti.remove()
        self.log_box.value = ''
        tc0 = self.cf_tc0.value
        tc1 = self.cf_tc1.value
        if self.cf_use_opt_tc0.value == False and tc1<=tc0:
            self.log_box.value = 'Should be tc0<tc1'
        else:
            dlts_cf = getattr(DLTS_CORRELATION_FUNCTION, self.cf.value)()
            if self.cf_use_opt_tc0.value:
                tc0 = tc1 * dlts_cf.optimum_td_tc_ratio / (1+dlts_cf.optimum_td_tc_ratio)
            if tc0 < self.parent.t[0]:
                f_tc0 = '{:.2e}'.format(tc0)
                self.log_box.value = f'tc0={f_tc0} too small'
            else:
                self.plot_x = self.parent.T
                _,self.plot_y,self.plot_rw = dlts_cf(self.parent.t, self.parent.T, self.parent.C, use_opt_ratio=self.cf_use_opt_tc0.value, tc0=tc0, tc1=tc1)
                self.plot_em = self.plot_rw[0]
                self.save_file_name.value = self.parent.transient_data_full_path.with_suffix('').name + f'_em_{round(self.plot_em,2)}_cf_{self.cf.value}'+ '.dltsplot'
                for li in self.ax1.lines:
                    li.remove()
                self.ax1.plot(self.plot_x,self.plot_y*self.parent.data_plot_dict['C_plot_factor'],'-x',color='black', label=rf'em={str(round(self.plot_em,2))}$s^{-1}$')
                self.ax1.grid(True)
                self.ax1.set_xlabel(self.parent.data_plot_dict['T_label'],fontsize=14)
                self.ax1.set_ylabel(self.parent.data_plot_dict['DLTS_label'],fontsize=14)
                self.ax1.legend(loc='lower right')
                self.ax1.relim()
                self.ax1.autoscale_view()
                self.plot_out_image1.draw()
                self.plot_out_image1.flush_events()
                f_tc0 = '{:.3e}'.format(tc0)
                f_tc1 = '{:.3e}'.format(tc1)
                if self.cf_use_opt_tc0.value:
                    self.log_box.value = f'{self.cf.value}(order={dlts_cf.order}). Current tc0={f_tc0}(Calculated from opt. td/tc ratio={dlts_cf.optimum_td_tc_ratio}), tc1={f_tc1}'
                else:
                    self.log_box.value = f'{self.cf.value}(order={dlts_cf.order}). Current tc0={f_tc0}, tc1={f_tc1}'

    def plot_dlts_manually(self):
        """
        Launch thread for manual mode plotting.
        
        Uses threading lock to prevent concurrent plot operations.
        """
        with self.plot_dlts_lock:
            self.plot_dlts_thread = threading.Thread(
                target=self.plot_dlts_manually_thread_fun
            )
            self.plot_dlts_thread.start()
    def plot_dlts_manually_thread_fun(self):
        """
        Worker function for manual mode plotting.
        
        Core operations:
        - Validates t1 < t2
        - Computes ΔC = C(t2) - C(t1)
        - Calculates emission rate from time points
        - Updates plot and filename fields
        """
        self.log_box.value = ''
        t1i = self.t1.value
        t2i = self.t2.value
        C1 = self.parent.C[:,t1i]
        C2 = self.parent.C[:,t2i]
        t1 = self.parent.t[t1i]
        t2 = self.parent.t[t2i]
        if t2>t1:
            DelC = C2-C1
            self.plot_x = self.parent.T
            self.plot_y = DelC
            self.plot_em = (np.log(t2)-np.log(t1))/(t2-t1)
            self.save_file_name.value = self.parent.transient_data_full_path.with_suffix('').name + f'_em_{round(self.plot_em,2)}_cf_{self.cf.value}'+ '.dltsplot'
            for li in self.ax1.lines:
                li.remove()
            self.ax1.plot(self.plot_x,self.plot_y*self.parent.data_plot_dict['C_plot_factor'],'-x',color='black', label=rf'em={str(round(self.plot_em,2))}$s^{-1}$')
            self.ax1.grid(True)
            self.ax1.set_xlabel(self.parent.data_plot_dict['T_label'],fontsize=14)
            self.ax1.set_ylabel(self.parent.data_plot_dict['DLTS_label'],fontsize=14)
            self.ax1.legend(loc='lower right')
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.plot_out_image1.draw()
            self.plot_out_image1.flush_events()
        else:
            self.log_box.value = 'Should be t1<t2'
        
    def plot_dlts_fix_t2t1_ratio(self):
        """
        Launch thread for fixed t2/t1 ratio plotting.
        
        Uses threading lock to prevent concurrent plot operations.
        """
        with self.plot_dlts_lock:
            self.plot_dlts_thread = threading.Thread(
                target=self.plot_dlts_fix_t2t1_ratio_thread_fun
            )
            self.plot_dlts_thread.start()
    def plot_dlts_fix_t2t1_ratio_thread_fun(self):
        """
        Worker function for fixed t2/t1 ratio plotting.
        
        Core operations:
        - Validates t2 within data range
        - Interpolates C(t2) at calculated time
        - Computes ΔC = C(t2) - C(t1)
        - Calculates emission rate from ratio
        - Updates plot and filename fields
        """
        self.log_box.value = ''
        t1i = self.t1.value
        C1 = self.parent.C[:,t1i].reshape(-1)
        C2 = np.zeros(len(C1))
        t1 = self.parent.t[t1i]
        t2 = t1*self.t2t1_ratio.value
        if t2>self.parent.t[-1]:
            self.log_box.value = 't1*t2t1_ratio too big!'
        else:
            for i in range(len(C2)):
                C2[i] = np.interp(t2, self.parent.t, self.parent.C[i,:].reshape(-1))
            DelC = C2-C1
            #DelCC = DelC/np.average(self.parent.C, axis=1)
            self.plot_x = self.parent.T
            self.plot_y = DelC
            self.plot_em = np.log(self.t2t1_ratio.value)/((self.t2t1_ratio.value-1)*self.parent.t[t1i])
            self.save_file_name.value = self.parent.transient_data_full_path.with_suffix('').name + f'_em_{round(self.plot_em,2)}_cf_{self.cf.value}'+ '.dltsplot'
            for li in self.ax1.lines:
                li.remove()
            self.ax1.plot(self.plot_x,self.plot_y*self.parent.data_plot_dict['C_plot_factor'],'-x',color='black', label=rf'em={str(round(self.plot_em,2))}$s^{-1}$')
            self.ax1.grid(True)
            self.ax1.set_xlabel(self.parent.data_plot_dict['T_label'],fontsize=14)
            self.ax1.set_ylabel(self.parent.data_plot_dict['DLTS_label'],fontsize=14)
            self.ax1.legend(loc='lower right')
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.plot_out_image1.draw()
            self.plot_out_image1.flush_events()
    def plot_dlts_fix_rate_window(self):
        """
        Launch thread for fixed rate window plotting.
        
        Uses threading lock to prevent concurrent plot operations.
        """
        with self.plot_dlts_lock:
            self.plot_dlts_thread = threading.Thread(
                target=self.plot_dlts_fix_rate_window_thread_fun
            )
            self.plot_dlts_thread.start()
    def plot_dlts_fix_rate_window_thread_fun(self):
        """
        Worker function for fixed rate window plotting.
        
        Core operations:
        - Solves for t2 using root-finding algorithm
        - Validates t2 within data range
        - Interpolates C(t2) at calculated time
        - Computes ΔC = C(t2) - C(t1)
        - Updates plot and filename fields
        """
        self.log_box.value = ''
        t1i = self.t1.value
        C1 = self.parent.C[:,t1i].reshape(-1)
        C2 = np.zeros(len(C1))
        t1 = self.parent.t[t1i]
        def tempfunc(x):
            return self.rate_window.value*t1-np.log(t1)-self.rate_window.value*x+np.log(x)
        try:
            root = sci_opt_fsolve(tempfunc, self.parent.t[-1])
        except Exception as e:
            return 0
            self.log_box.value = f'Solve Failed:{str(e)}'
        t2 = root[0]
        if t2>self.parent.t[-1]:
            self.log_box.value = 'Solved t2 too big!'
        else:
            for i in range(len(C2)):
                C2[i] = np.interp(t2, self.parent.t, self.parent.C[i,:].reshape(-1))
            DelC = C2-C1
            #DelCC = DelC/np.average(self.parent.C, axis=1)
            self.plot_x = self.parent.T
            self.plot_y = DelC
            self.plot_em = np.log(t2/t1)/(t2-t1)
            self.save_file_name.value = self.parent.transient_data_full_path.with_suffix('').name + f'_em_{round(self.plot_em,2)}_cf_{self.cf.value}'+ '.dltsplot'
            for li in self.ax1.lines:
                li.remove()
            self.ax1.plot(self.plot_x,self.plot_y*self.parent.data_plot_dict['C_plot_factor'],'-x',color='black', label=rf'em={str(round(self.plot_em,2))}$s^{-1}$')
            self.ax1.grid(True)
            self.ax1.set_xlabel(self.parent.data_plot_dict['T_label'],fontsize=14)
            self.ax1.set_ylabel(self.parent.data_plot_dict['DLTS_label'],fontsize=14)
            self.ax1.legend(loc='lower right')
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.plot_out_image1.draw()
            self.plot_out_image1.flush_events()

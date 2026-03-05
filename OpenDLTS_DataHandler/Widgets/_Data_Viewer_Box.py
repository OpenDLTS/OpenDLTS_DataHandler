"""
Interactive transient data visualization and manipulation widget for spectroscopic/time-series data.

Class:
    Data_Viewer_Box - Main visualization container with interactive controls

Attributes:
    parent (object): Parent dataset container object
    logger (Logger): Custom logger instance for debug/info tracking

Methods:
    __init__: Initialize UI elements and interactive controls
    _init_logger: Configure logging handlers and formatters
    _create_ui: Build widget layout and plotting components
    _set_event: Connect widget handlers to callback functions
    _observe_Tno: Handle transient selection index changes
    _observe_neighbor: Handle neighbor count changes
    _click_del_trans_btn: Remove selected transient
    _click_del_time_btn: Remove leading time points
    _click_undo_del_btn: Restore last deleted item
    _click_save_file_btn: Save modified dataset
    _init_trans: Initialize data tracking structures
    _del_trans: Execute data deletion (transients or timepoints)
    _undo_del_trans: Recover last deleted data segment
    plot_trans: Thread-safe plot triggering
    plot_trans_thread_fun: Background plotting operation
"""
__all__ = ['Data_Viewer_Box']

import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import threading
from .._Data_Loader import Data_Loader
from .._config import *

# Data_Viewer_Box
class Data_Viewer_Box:
    """
    Interactive viewer for transient spectroscopy data with deletion/correction tools.
    
    Provides dual-panel visualization:
    - Top plot: Time-domain transients
    - Bottom plot: Transient-averaged spectrum
    Features real-time interaction, data pruning, and undo functionality.
    
    Requires parent container with:
        T: Transient values (1D array)
        t: Time points (1D array)
        C: Concentration matrix (transients x time)
        data_plot_dict: Plot formatting parameters
        original_data_file_name: Default save filename
        savedata(data): Data export method
        
    Args:
        parent: Parent data container object
        logging_level: Logger verbosity ('DEBUG','INFO','WARNING','ERROR')
    """
    def __init__(self, dlts_data: Data_Loader, show_ui: bool = True):
        """Initialize UI, event handlers, and plotting systems."""
        self.parent = dlts_data
        self._create_ui()
        self._init_trans()
        self._set_event()
        self.plot_trans_lock = threading.Lock()
        self.logger = LOGGER_ODDH.getChild('Data_Viewer_Box')
        if show_ui:
            display(self.box)
    

    def _create_ui(self):
        """Construct widget interface components and plot panels."""
        self.Tno_label = widgets.Label('Transient Index: ')
        self.Tno_text =  widgets.Label('0')
        self.Tno = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.parent.T)-1,
            step=1,
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=False,
            readout_format='d',
            layout=widgets.Layout(width='40%')
        )
        self.del_trans_btn = widgets.Button(
            description='Delete This Trans',
            disabled=False,
            button_style='danger',
            tooltip='Delete This Trans',
            icon='trash'
        )
        self.undo_del_trans_btn = widgets.Button(
            description='Undo Delete',
            disabled=False,
            button_style='warning',
            tooltip='Undo Delete',
            icon='rotate-left'
        )
        self.neighbor = widgets.IntSlider(
            value=1,
            min=0,
            max=5,
            step=1,
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=False,
            readout_format='d',
            layout=widgets.Layout(width='20%')
        )
        self.neighbor_label = widgets.Label('NeighborNumber:')
        self.neighbor_text = widgets.Label('1')
        self.del_points_num = widgets.BoundedIntText(
            value=1,
            min=1,
            max=10,
            step=1,
            disabled=False,
            layout=widgets.Layout(width='5%')
        )
        self.del_points_label = widgets.Label('Delete Time Points From Beginning:')
        self.del_points_btn = widgets.Button(
            description='Delete Time Points From Beginning',
            disabled=False,
            button_style='danger',
            tooltip='Delete Time Points From Beginning',
            icon='trash',
            layout=widgets.Layout(width='20%')
        )
        plt.ioff()
        self.fig1 = plt.figure(figsize = (12,3), layout='tight')
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
        self.ax1.text(-0.035,0,r'Drag the $\mathbf{Transient\ Index\ Slider}$ to plot transient data here', fontsize=20)

        self.fig2 = plt.figure(figsize = (12,3), layout='tight')
        self.fig2.canvas.header_visible = False
        self.fig2.canvas.resizable = False
        self.fig2.canvas.footer_visible = False
        x=np.array([0])
        y=np.array([0])
        self.ax2 = plt.gca()
        self.ax2.grid()
        self.im2 = self.ax2.plot(x,y)
        self.plot_out_image2 = self.fig2.canvas
        self.plot_out_image2.layout = widgets.Layout(width='1200px',height='300px')
        self.ax2.text(-0.04,0,r'Drag the $\mathbf{Transient\ Index\ Slider}$ to plot averaged transient data here', fontsize=20)

        self.save_file_name = widgets.Text(
            #value=str(self.parent.original_data_file_name)+'.cut',
            value = self.parent.transient_data_full_path.with_suffix('').name + '_cut' + self.parent.transient_data_full_path.suffix,
            placeholder='Output File Path',
            disabled=False,
            layout=widgets.Layout(width='40%')
        )
        self.save_file_btn = widgets.Button(
            description='Save File',
            disabled=False,
            button_style='info',
            tooltip='Save File',
            icon='file',
            layout=widgets.Layout(width='10%')
        )
        self.box = widgets.VBox([
            widgets.HBox([self.Tno_label, self.Tno_text, self.Tno, self.del_trans_btn, self.undo_del_trans_btn]),
            widgets.HBox([self.neighbor_label, self.neighbor_text, self.neighbor, self.del_points_label, self.del_points_num, self.del_points_btn]),
            self.plot_out_image1,
            self.plot_out_image2,
            widgets.HBox([self.save_file_name, self.save_file_btn])
        ])
    
    def _set_event(self):
        """Bind UI events to handler methods."""
        self.Tno.observe(self._observe_Tno, names='value')
        self.neighbor.observe(self._observe_neighbor, names='value')
        self.del_trans_btn.on_click(self._click_del_trans_btn)
        self.del_points_btn.on_click(self._click_del_time_btn)
        self.undo_del_trans_btn.on_click(self._click_undo_del_btn)
        self.save_file_btn.on_click(self._click_save_file_btn)
    def _observe_Tno(self, change):
        """Handle transient index slider change (update label and plot)."""
        self.Tno_text.value = str(change['new'])
        self.plot_trans()
    def _observe_neighbor(self, change):
        """Handle neighbor count slider change (update label and plot)."""
        self.neighbor_text.value = str(change['new'])
        self.plot_trans()
    def _click_del_trans_btn(self, b):
        """Delete currently selected transient and update display."""
        self._del_trans('T', self.Tno.value)
        self.Tno.max = self.parent.T.shape[0]-1
        self.plot_trans()
    def _click_del_time_btn(self, b):
        """Delete leading time points and update display."""
        self._del_trans('t', np.arange(self.del_points_num.value))
        self.plot_trans()
    def _click_undo_del_btn(self, b):
        """Restore last deleted transient/time segment and update display."""
        self._undo_del_trans()
        self.Tno.max = self.parent.T.shape[0]-1
        self.plot_trans()
    # after click Save File
    def _click_save_file_btn(self, b):
        """Trigger parent data export with current filename."""
        self.parent.savedata(self.parent.transient_data_full_path.parent / self.save_file_name.value)
    def _init_trans(self):
        """Initialize data tracking indices and UI state."""
        #index to 0
        self.Tno.value=0
        #Tno max update
        self.Tno.max=len(self.parent.T)-1
        #init live index
        self.live_index = {'t': np.arange(self.parent.t.shape[0]), 'T': np.arange(self.parent.T.shape[0])}
        #del dead index
        self.dead_index = []
    # del trans
    def _del_trans(self, deltype, index):
        """
        Delete specified data segments and track modifications.
        
        Args:
            deltype: 't' (time) or 'T' (transient)
            index: Indices to delete in time/transient dimension
        """
        if deltype=='t':
            self.parent.t = np.delete(self.parent.t, index, axis=0)
            self.parent.C = np.delete(self.parent.C, index, axis=1)
        elif deltype=='T':
            self.parent.T = np.delete(self.parent.T, index, axis=0)
            self.parent.C = np.delete(self.parent.C, index, axis=0)
        self.dead_index.append([deltype, self.live_index[deltype][index]])
        self.live_index[deltype] = np.delete(self.live_index[deltype], index, axis=0)
    def _undo_del_trans(self):
        """Restore last deleted items from undo stack and recalc dataset."""
        [rectype,recarray] = self.dead_index.pop(-1)
        self.live_index[rectype] = np.sort(np.append(self.live_index[rectype],np.array(recarray)))
        self.parent.t = self.parent.rawdata[0,1:][self.live_index['t']]
        self.parent.T = self.parent.rawdata[1:,0][self.live_index['T']]
        self.parent.C = self.parent.rawdata[1:,1:][self.live_index['T'],:]
        self.parent.C = self.parent.C[:,self.live_index['t']]
    def plot_trans(self):
        """Thread-safe plot update trigger."""
        with self.plot_trans_lock:
            self.plot_trans_thread = threading.Thread(
                target=self.plot_trans_thread_fun
            )
            self.plot_trans_thread.start()
    def plot_trans_thread_fun(self):
        """
        Background plotting function.
        
        - Top plot: Shows current transient + neighboring traces
        - Bottom plot: Shows full transient-averaged spectrum with active point
        """
        for texti in self.ax1.texts:
            texti.remove()
        for texti in self.ax2.texts:
            texti.remove()
        Tn = self.Tno.value+1
        Nb = self.neighbor.value
        Tn0 = 0 if Tn-Nb<1 else Tn-Nb-1
        Tn1 = self.parent.T.shape[0] if Tn+Nb>self.parent.T.shape[0] else Tn+Nb
        Tnlist = np.arange(Tn0,Tn1,1)

        if len(Tnlist)!=len(self.ax1.lines):
            # lines don't match
            # del all lines and replot
            for li in self.ax1.lines:
                li.remove()
            for i in Tnlist:
                if i==Tn-1:
                    self.ax1.plot(self.parent.t, self.parent.C[i,:]*self.parent.data_plot_dict['C_plot_factor'],
                                  color='black', label='T='+str(format(self.parent.T[i],'.2f'))+f" {self.parent.data_plot_dict['T_unit']}"+'(selected)')
                elif i<Tn-1:
                    self.ax1.plot(self.parent.t, self.parent.C[i,:]*self.parent.data_plot_dict['C_plot_factor'], ':',
                                  color='blue', label='T='+str(format(self.parent.T[i],'.2f'))+f" {self.parent.data_plot_dict['T_unit']}")
                else:
                    self.ax1.plot(self.parent.t, self.parent.C[i,:]*self.parent.data_plot_dict['C_plot_factor'], ':',
                                  color='red', label='T='+str(format(self.parent.T[i],'.2f'))+f" {self.parent.data_plot_dict['T_unit']}")
        else:
            for idx,i in enumerate(Tnlist):
                li = self.ax1.lines[idx]
                li.set_data(self.parent.t, self.parent.C[i,:]*self.parent.data_plot_dict['C_plot_factor'])
                if i==Tn-1:
                    li.set_color('black')
                    li.set_linestyle('-')
                    li.set_label('T='+str(format(self.parent.T[i],'.2f'))+f" {self.parent.data_plot_dict['T_unit']}"+'(selected)')
                elif i<Tn-1:
                    li.set_color('blue')
                    li.set_linestyle(':')
                    li.set_label('T='+str(format(self.parent.T[i],'.2f'))+f" {self.parent.data_plot_dict['T_unit']}")
                else:
                    li.set_color('red')
                    li.set_linestyle(':')
                    li.set_label('T='+str(format(self.parent.T[i],'.2f'))+f" {self.parent.data_plot_dict['T_unit']}")
        
        self.ax1.grid(True)
        self.ax1.set_xlabel(self.parent.data_plot_dict['t_label'],fontsize=14)
        self.ax1.set_ylabel(self.parent.data_plot_dict['C_label'],fontsize=14)
        self.ax1.legend(loc='lower right')


        avrageC = np.nan_to_num(np.average(self.parent.C, axis=1),nan=-1)
        if len(self.ax1.lines)!=2:
            for li in self.ax2.lines:
                li.remove()
            # lines don't match
            # del all lines and replot
            self.ax2.plot(self.parent.T,avrageC*self.parent.data_plot_dict['C_plot_factor'],'.',color='black')
            self.ax2.plot(self.parent.T[Tn-1],avrageC[Tn-1]*self.parent.data_plot_dict['C_plot_factor'],'x',markersize=12,color='red',label='selected')
        else:
            self.ax2.lines[0].set_data(self.parent.T,avrageC*self.parent.data_plot_dict['C_plot_factor'])
            self.ax2.lines[0].set_marker('.')
            self.ax2.lines[0].set_color('black')
            self.ax2.lines[1].set_data([self.parent.T[Tn-1]],[avrageC[Tn-1]*self.parent.data_plot_dict['C_plot_factor']])
            self.ax2.lines[1].set_marker('x')
            self.ax2.lines[1].set_markersize(12)
            self.ax2.lines[1].set_color('red')
            self.ax2.lines[1].set_label('selected')
        
        self.ax2.set_xlabel(self.parent.data_plot_dict['T_label'],fontsize=14)
        self.ax2.set_ylabel('Averaged '+self.parent.data_plot_dict['C_label'],fontsize=14)
        self.ax2.grid(True)
        self.ax2.legend(loc='lower right')
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.plot_out_image1.draw()
        self.plot_out_image1.flush_events()
        delta = np.max(self.parent.T) - np.min(self.parent.T)
        self.ax2.set_xlim(np.min(self.parent.T)-0.1*delta,np.max(self.parent.T)+0.1*delta)
        delta = np.max(avrageC*self.parent.data_plot_dict['C_plot_factor']) - np.min(avrageC*self.parent.data_plot_dict['C_plot_factor'])
        self.ax2.set_ylim(np.min(avrageC*self.parent.data_plot_dict['C_plot_factor'])-0.1*delta,np.max(avrageC*self.parent.data_plot_dict['C_plot_factor'])+0.1*delta)
        self.ax2.autoscale_view()
        self.plot_out_image2.draw()
        self.plot_out_image2.flush_events()
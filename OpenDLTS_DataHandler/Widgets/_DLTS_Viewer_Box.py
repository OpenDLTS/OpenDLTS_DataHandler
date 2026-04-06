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
import scipy.constants as sci_const
import scipy.interpolate
import scipy.signal
from scipy.optimize import fsolve as sci_opt_fsolve
from scipy.optimize import minimize_scalar as sci_opt_min_sca
from ._ARRH_PLOTTER import ARRH_PLOTTER_2D
from .._DLTS_CORRELATION_FUNCTION import DLTS_CORRELATION_FUNCTION
from .._Data_Loader import Data_Loader
from .._config import *

# DLTS_Viewer_Box
class DLTS_Viewer_Box:
    def __init__(self, dlts_data: Data_Loader, show_ui: bool = True):
        self.parent = dlts_data
        self.plot_dlts_lock = threading.Lock()
        self.plot_x = np.array([0])
        self.plot_y = np.array([0])
        self.plot_em = 0.0
        
        # 新增：用于存储已保存的 DLTS 谱图数据
        self.saved_dlts_data = {}
        
        self._create_ui()
        self._set_event()
        self.init_dlts_mode0()
        self.logger = LOGGER_ODDH.getChild('DLTS_Viewer_Box')
        if show_ui:
            display(self.box)

    def _create_ui(self):
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
            value=0, min=0, max=1000, step=1,
            description='Set t1:', disabled=False,
            continuous_update=True, orientation='horizontal',
            layout=widgets.Layout(width='90%'), readout=False
        )
        self.t2 = widgets.IntSlider(
            value=0, min=0, max=1000, step=1,
            description='Set t2:', disabled=False,
            continuous_update=True, orientation='horizontal',
            layout=widgets.Layout(width='90%'), readout=False
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
            description='Function:', disabled=False,
        )
        self.cf_tc0 = widgets.BoundedFloatText(
            value=self.parent.t[0], min=self.parent.t[0], max=self.parent.t[-1],
            description='tc0:', layout=widgets.Layout(width='10%'), style={'description_width': 'initial'}
        )
        self.cf_tc1 = widgets.BoundedFloatText(
            value=self.parent.t[1], min=self.parent.t[0], max=self.parent.t[-1],
            description='tc1:', layout=widgets.Layout(width='10%'), style={'description_width': 'initial'}
        )
        self.cf_target_em = widgets.BoundedFloatText(
            value=100, min=10, max=1000,
            description='Target em:', layout=widgets.Layout(width='10%'), style={'description_width': 'initial'}
        )
        self.cf_target_em_btn = widgets.Button(
            description='Find em', button_style='danger', icon='bullseye', layout=widgets.Layout(width='8%')
        )
        self.cf_replot_btn = widgets.Button(
            description='Replot', button_style='info', icon='reply', layout=widgets.Layout(width='7%')
        )
        self.cf_use_opt_tc0 = widgets.Checkbox(value=True, description='Use Opt. tc0')

        plt.ioff()
        self.fig1 = plt.figure(figsize = (12,3), layout='tight')
        self.fig1.canvas.header_visible = False
        self.fig1.canvas.resizable = False
        self.fig1.canvas.footer_visible = False
        self.ax1 = plt.gca()
        self.ax1.grid()
        self.im1 = self.ax1.plot([0],[0])
        self.plot_out_image1 = self.fig1.canvas
        self.plot_out_image1.layout = widgets.Layout(width='1200px',height='300px')
        self.ax1.text(-0.01,0,'Plot DLTS spectra here', fontsize=20)
        
        self.save_file_name = widgets.Text(
            value = self.parent.transient_data_full_path.with_suffix('.dltsplot').name,
            placeholder='Output File Path', layout=widgets.Layout(width='40%')
        )
        self.save_file_btn = widgets.Button(
            description='Save File', button_style='info', icon='file', layout=widgets.Layout(width='10%')
        )
        
        # 新增按钮与列表
        self.dlts_add_list_btn = widgets.Button(
            description='Add to List', button_style='info', tooltip='Add current DLTS to List', layout=widgets.Layout(width='10%')
        )
        self.dlts_list_selector = widgets.SelectMultiple(
            options=[], value=[], description='DLTS List', layout=widgets.Layout(width='95%')
        )
        self.dlts_del_list_btn = widgets.Button(
            description='Delete data', button_style='danger', tooltip='Delete selected from List', layout=widgets.Layout(width='95%')
        )
        
        # 新增：导出 List 数据与峰值的按钮与文本框
        self.dlts_save_list_name = widgets.Text(
            value='dlts_list_export.txt', placeholder='Export List Name', layout=widgets.Layout(width='95%')
        )
        self.dlts_save_list_btn = widgets.Button(
            description='Export List & Peaks', button_style='success', icon='download', layout=widgets.Layout(width='95%')
        )
        
        # 峰值提取参数
        self.peak_or_valley = widgets.Select(
            options=['peak', 'valley'], value='peak', description='extract type:', layout=widgets.Layout(width='95%')
        )
        # 【修改】将 IntSlider 修改为 FloatLogSlider
        self.smoothing_factor = widgets.FloatLogSlider(
            value=10, base=10, min=-2, max=4, step=0.1, description='Smoothing factor:',
            continuous_update=False, orientation='horizontal', layout=widgets.Layout(width='70%'),
            style={'description_width': 'initial'}
        )
        
        self.fig2 = plt.figure(figsize = (10,3), layout='tight')
        self.fig2.canvas.header_visible = False
        self.fig2.canvas.resizable = False
        self.fig2.canvas.footer_visible = False
        self.ax2 = plt.gca()
        self.ax2.grid()
        self.plot_out_image2 = self.fig2.canvas
        self.plot_out_image2.layout = widgets.Layout(width='1000px',height='300px')
        self.ax2.text(-0.01,0,'Plot List here', fontsize=20)
        
        self.arrh_mat = widgets.Dropdown(
            options=['SiC', 'Si', 'GaN'], value='SiC', description='Material:', layout=widgets.Layout(width='15%'), style={'description_width': 'initial'}
        )
        self.arrh_doping_type = widgets.Dropdown(
            options=['N', 'P'], value='N', description='Doping Type:', layout=widgets.Layout(width='15%'), style={'description_width': 'initial'}
        )
        self.arrh_widget = widgets.VBox()
        self.arrh_wwidget = widgets.VBox([self.arrh_widget])

        # -------------------------------------
        # 组装 UI
        # -------------------------------------
        row1 = widgets.HBox([self.dlts_plot_mode, widgets.VBox([widgets.HBox([self.t2t1_ratio, self.rate_window]), self.t1, self.t2,],layout=widgets.Layout(width='70%'))])
        row2 = widgets.HBox([self.cf, self.cf_use_opt_tc0, self.cf_tc0, self.cf_tc1, self.cf_target_em, self.cf_target_em_btn, self.cf_replot_btn])
        row_file = widgets.HBox([self.save_file_name, self.save_file_btn, self.dlts_add_list_btn])
        
        # 将新增加的导出功能加入到左侧的控件列中
        list_controls = widgets.VBox([
            self.dlts_list_selector, 
            self.dlts_del_list_btn, 
            self.peak_or_valley,
            widgets.HTML("<hr>"), # 分隔线让 UI 更好看点
            self.dlts_save_list_name,
            self.dlts_save_list_btn
        ], layout=widgets.Layout(width='20%'))
        
        fig2_controls = widgets.HBox([self.smoothing_factor])
        fig2_area = widgets.VBox([fig2_controls, self.plot_out_image2], layout=widgets.Layout(width='80%'))
        row_list_fig2 = widgets.HBox([list_controls, fig2_area])
        
        row_arrh = widgets.HBox([self.arrh_mat, self.arrh_doping_type])
        arrh_area = widgets.VBox([row_arrh, self.arrh_wwidget])

        self.box = widgets.VBox([
            row1, row2, self.log_box, self.plot_out_image1,
            row_file, 
            row_list_fig2, 
            arrh_area
        ])

    def _set_event(self):
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
        
        # 列表与峰值提取控件的事件绑定
        self.dlts_add_list_btn.on_click(self._click_add_list_btn)
        self.dlts_del_list_btn.on_click(self._click_del_list_btn)
        self.dlts_save_list_btn.on_click(self._click_save_list_btn) # 新增的导出事件
        self.smoothing_factor.observe(self._observe_peak_extraction_params, names='value')
        self.peak_or_valley.observe(self._observe_peak_extraction_params, names='value')
        self.arrh_mat.observe(self._observe_peak_extraction_params, names='value')
        self.arrh_doping_type.observe(self._observe_peak_extraction_params, names='value')

    # ==========================================
    # List 和 Peak Extraction 的联动逻辑与导出
    # ==========================================
    def _click_add_list_btn(self, b):
        if len(self.plot_x) <= 1:
            self.log_box.value = "No valid DLTS data to add."
            return
        
        base_label = f"em={self.plot_em:.2f} s^-1"
        label = base_label
        count = 1
        while label in self.saved_dlts_data:
            label = f"{base_label} ({count})"
            count += 1
            
        factor = 1.0
        if hasattr(self.parent, 'data_plot_dict') and 'C_plot_factor' in self.parent.data_plot_dict:
            factor = self.parent.data_plot_dict['C_plot_factor']
            
        self.saved_dlts_data[label] = {
            'x': self.plot_x.copy(),
            'y': self.plot_y.copy() * factor,
            'em': self.plot_em,
            'pkx': [], # 预留位置保存峰值坐标
            'pky': []
        }
        self.dlts_list_selector.options = list(self.saved_dlts_data.keys())
        self._update_list_and_arrh_plot()

    def _click_del_list_btn(self, b):
        if not self.dlts_list_selector.value:
            return
        for item in self.dlts_list_selector.value:
            if item in self.saved_dlts_data:
                del self.saved_dlts_data[item]
        self.dlts_list_selector.options = list(self.saved_dlts_data.keys())
        self._update_list_and_arrh_plot()

    def _click_save_list_btn(self, b):
        """新增：将 List 中的所有谱图数据和对应提取出的峰值保存到文本文件中"""
        if not self.saved_dlts_data:
            self.log_box.value = "List is empty, nothing to export."
            return
            
        try:
            save_path = self.parent.transient_data_full_path.parent / self.dlts_save_list_name.value
            with open(save_path, 'w', encoding='utf-8') as f:
                for label, data in self.saved_dlts_data.items():
                    f.write(f"--- Curve: {label} ---\n")
                    f.write(f"Emission Rate: {data['em']:.4e} s^-1\n\n")
                    
                    f.write(">>> Extracted Peaks (Temperature [K], DLTS Signal):\n")
                    pkx_list = data.get('pkx', [])
                    pky_list = data.get('pky', [])
                    if len(pkx_list) == 0:
                        f.write("None\n")
                    else:
                        for px, py in zip(pkx_list, pky_list):
                            f.write(f"{px:.4f}\t{py:.6e}\n")
                    
                    f.write("\n>>> Raw Data (Temperature [K], DLTS Signal):\n")
                    for dx, dy in zip(data['x'], data['y']):
                        f.write(f"{dx:.4f}\t{dy:.6e}\n")
                    f.write("\n" + "="*40 + "\n\n")
                    
            self.log_box.value = f"Successfully exported to: {self.dlts_save_list_name.value}"
        except Exception as e:
            self.log_box.value = f"Export failed: {str(e)}"

    def _observe_peak_extraction_params(self, change):
        self._update_list_and_arrh_plot()

    def _update_list_and_arrh_plot(self):
        # 1. 清理 ax2
        self.ax2.clear()
        self.ax2.grid(True)
        if hasattr(self.parent, 'data_plot_dict'):
            self.ax2.set_xlabel(self.parent.data_plot_dict.get('T_label', r'$Temperature\ (K)$'), fontsize=14)
            self.ax2.set_ylabel(self.parent.data_plot_dict.get('DLTS_label', r'$DLTS\ signal\ (a.u.)$'), fontsize=14)

        if not self.saved_dlts_data:
            self.plot_out_image2.draw()
            self.plot_out_image2.flush_events()
            self.arrh_widget.children = []
            return

        arrh_x = []
        arrh_y = []
        arrh_c = []
        
        lam = self.smoothing_factor.value
        is_peak = (self.peak_or_valley.value == 'peak')

        # 2. 绘制 Spline 和 Peaks
        cmap = plt.colormaps.get_cmap('tab10')
        
        for label, data in self.saved_dlts_data.items():
            x = data['x']
            y = data['y']
            emi = data['em']
            
            self.ax2.plot(x, y, label=label)
            
            # 清空旧的峰值坐标记录，以便下方写入最新的值
            self.saved_dlts_data[label]['pkx'] = []
            self.saved_dlts_data[label]['pky'] = []
            
            if len(x) < 4:
                continue
            
            try:
                spl = scipy.interpolate.make_smoothing_spline(x, y, lam=lam)
                xs = np.linspace(x[0], x[-1], 1000)
                ys = spl(xs)
                
                # 寻找峰值
                pi, _ = scipy.signal.find_peaks(ys if is_peak else -ys)
                pkx = xs[pi]
                pky = ys[pi]
                
                # 保存峰值，供导出功能使用
                self.saved_dlts_data[label]['pkx'] = pkx
                self.saved_dlts_data[label]['pky'] = pky
                
                c_array = [cmap(i % 10) for i in range(len(pkx))]
                self.ax2.scatter(pkx, pky, color=c_array, marker='o', zorder=3)
                
                # 提取 Arrhenius 数据
                for i in range(len(pkx)):
                    pkxi = pkx[i]
                    x_arrh = 1 / sci_const.k / pkxi * sci_const.e
                    y_arrh = -np.log(emi / (pkxi**2))
                    
                    arrh_x.append(x_arrh)
                    arrh_y.append(y_arrh)
                    arrh_c.append(i)
                    
            except Exception as e:
                self.logger.warning(f"Failed to fit spline for {label}: {e}")
                pass

        self.ax2.legend(loc='lower right')
        self.plot_out_image2.draw()
        self.plot_out_image2.flush_events()

        # 3. 渲染 Arrhenius Plot
        self.arrh_widget.children = []
        if len(arrh_x) > 0:
            plt.ioff()
            self.arrh_plotter_instance = ARRH_PLOTTER_2D(
                x=np.array(arrh_x),
                y=np.array(arrh_y),
                c=np.array(arrh_c),
                material=self.arrh_mat.value,
                dopant_species=self.arrh_doping_type.value,
                figsize=(12, 6) 
            )
            
            self.arrh_widget.close()
            self.arrh_widget = widgets.VBox()
            self.arrh_wwidget.children = [self.arrh_widget]
            self.arrh_widget.children = [self.arrh_plotter_instance.fig.canvas]

    # ==========================================
    # 原有逻辑保留区
    # ==========================================
    def _observe_dlts_plot_mode(self, change):
        if self.dlts_plot_mode.value == 'Fix t2/t1 ratio':
            self.init_dlts_mode0()
        elif self.dlts_plot_mode.value == 'Fix rate window':
            self.init_dlts_mode1()
        elif self.dlts_plot_mode.value == 'Manually':
            self.init_dlts_mode2()
        elif self.dlts_plot_mode.value == 'Correalation function':
            self.init_dlts_mode3()

    def _observe_t2t1_ratio(self, change):
        self.plot_dlts_fix_t2t1_ratio()

    def _observe_rate_window(self, change):
        self.plot_dlts_fix_rate_window()

    def _observe_t1(self, change):
        if self.dlts_plot_mode.value == 'Fix t2/t1 ratio':
            self.plot_dlts_fix_t2t1_ratio()
        elif self.dlts_plot_mode.value == 'Fix rate window':
            self.plot_dlts_fix_rate_window()
        elif self.dlts_plot_mode.value == 'Manually':
            self.plot_dlts_manually()

    def _observe_t2(self, change):
        if self.dlts_plot_mode.value == 'Manually':
            self.plot_dlts_manually()

    def _observe_cf(self, change):
        self.plot_dlts_cf()
        self._update_cf_target_em_max_min()

    def _observe_cf_use_opt_tc0(self, change):
        if self.cf_use_opt_tc0.value:
            self.cf_tc0.disabled = True
        else:
            self.cf_tc0.disabled = False
        self.plot_dlts_cf()

    def _observe_cf_tc0(self, change):
        self.plot_dlts_cf()

    def _observe_cf_tc1(self, change):
        self.plot_dlts_cf()

    def _click_save_file_btn(self, b):
        data_need_save = np.concatenate((self.plot_x.reshape(-1,1),self.plot_y.reshape(-1,1)),axis=1)
        np.savetxt(self.parent.transient_data_full_path.parent / self.save_file_name.value, data_need_save, fmt='%s',delimiter='\t')

    def init_dlts_mode0(self):
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
            objective, bounds=(self.cf_tc1.min, self.cf_tc1.max), method='bounded'
        )
        if not result.success:
            self.log_box.value = f"Can't find tc1 when em={self.cf_target_em.value:.1f} $s^{{-1}}$"
        else:
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
        with self.plot_dlts_lock:
            self.plot_dlts_thread = threading.Thread(target=self.plot_dlts_cf_thread_fun)
            self.plot_dlts_thread.start()

    def plot_dlts_cf_thread_fun(self):
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
        with self.plot_dlts_lock:
            self.plot_dlts_thread = threading.Thread(target=self.plot_dlts_manually_thread_fun)
            self.plot_dlts_thread.start()

    def plot_dlts_manually_thread_fun(self):
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
        with self.plot_dlts_lock:
            self.plot_dlts_thread = threading.Thread(target=self.plot_dlts_fix_t2t1_ratio_thread_fun)
            self.plot_dlts_thread.start()

    def plot_dlts_fix_t2t1_ratio_thread_fun(self):
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
        with self.plot_dlts_lock:
            self.plot_dlts_thread = threading.Thread(target=self.plot_dlts_fix_rate_window_thread_fun)
            self.plot_dlts_thread.start()

    def plot_dlts_fix_rate_window_thread_fun(self):
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
        t2 = root[0]
        if t2>self.parent.t[-1]:
            self.log_box.value = 'Solved t2 too big!'
        else:
            for i in range(len(C2)):
                C2[i] = np.interp(t2, self.parent.t, self.parent.C[i,:].reshape(-1))
            DelC = C2-C1
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
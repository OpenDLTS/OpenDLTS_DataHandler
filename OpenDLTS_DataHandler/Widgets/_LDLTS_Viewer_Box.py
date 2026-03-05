__all__ = ['LDLTS_Viewer_Box']

import ipywidgets as widgets
import threading
import numpy as np
import matplotlib.pyplot as plt
from .._Data_Loader import Data_Loader
from .._config import *
from ..LDLTS_Method import *
from .._typing import *

def create_widgets_from_function(input_fun: Callable, name_blacklist: list = []) -> Dict[str, widgets.Widget]:
    widgets_dict = {}
    import inspect
    for name,param in inspect.signature(input_fun).parameters.items():
        if name.startswith('__') or name=='self' or name in name_blacklist:
            continue
        default = param.default if param.default != inspect.Parameter.empty else 0
        desc = f"{name}:"
        if param.annotation == bool:
            widgets_dict[name] = widgets.Checkbox(
                value=default,
                description=desc
            )
        elif param.annotation == float:
            widgets_dict[name] = widgets.FloatText(
                value=default,
                description=desc
            )
        elif param.annotation == int:
            widgets_dict[name] = widgets.IntText(
                value=default,
                description=desc
            )
        else:
            widgets_dict[name] = widgets.Text(
                value=str(default),
                description=desc
            )
    return widgets_dict

class Ti_Selector:
    def __init__(self, Ti_list: list, T_list: list):
        self.Ti_list = Ti_list
        self.T_list = T_list
        self.selector = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.Ti_list)-1,
            step=1,
            description='',
            disabled=False,
            continuous_update=True,
            readout=False
        )
        self.selector_label_widget = widgets.Label(f'Ti={self.Ti_list[0]} (T={self.T_list[0]:.2f}K)')
        self.selector.observe(self._obs_selector, names='value')
        self.box = widgets.HBox([self.selector,self.selector_label_widget])
    def _obs_selector(self, change):
        self.selector_label_widget.value = f"Ti={self.Ti_list[change['new']]} (T={self.T_list[change['new']]:.2f}K)"

# LDLTS_Viewer_Box
class LDLTS_Viewer_Box:
    def __init__(self, dlts_data: Data_Loader, Ns: int = 500, s0: float = 1e-1, s1: float = 1e5, show_ui: bool = True) -> None:
        self.parent = dlts_data
        self.logger = LOGGER_ODDH.getChild('LDLTS_Viewer_Box')
        self._init_ldlts_method(Ns=Ns, s0=s0, s1=s1)
        self.plot_fig1_fig2_lock = threading.Lock()
        self._create_ui()
        self._set_event()
        if show_ui:
            display(self.box)
    def _init_ldlts_method(self, Ns: int = 500, s0: float = 1e-1, s1: float = 1e5):
        for method_name in METHOD_NAME_LIST:
            setattr(self, method_name, globals()[method_name](dlts_data=self.parent, Ns=Ns, s0=s0, s1=s1))
        self.logger.info(f'LDLTS Method Initialized with Ns={Ns}, s0={s0:.1E}, s1={s1:.1E}')
    def _create_ui(self):
        self._create_init_ldlts_method_ui()
        self._create_result_viewer_box()
        self.box = widgets.VBox([self._init_ldlts_method_box, self.result_viewer_box])
    def _create_init_ldlts_method_ui(self):
        self._init_ldlts_method_widgets_dict = create_widgets_from_function(
            self._init_ldlts_method
        )
        temp_widgets_list = []
        for _,widget in self._init_ldlts_method_widgets_dict.items():
            temp_widgets_list.append(widget)
        self._init_ldlts_method_btn = widgets.Button(
            description='Init LDLTS Method',
            button_style='info',
            disabled=False
        )
        temp_widgets_list.append(self._init_ldlts_method_btn)
        temp_init_ldlts_method_widgets = []
        widgets_per_row = 4
        for i in range(0, len(temp_widgets_list), widgets_per_row):
            temp = widgets.HBox(temp_widgets_list[i:i+widgets_per_row])
            temp_init_ldlts_method_widgets.append(temp)
        self._init_ldlts_method_box = widgets.VBox(
            temp_init_ldlts_method_widgets,
            layout=widgets.Layout(width='100%', display='flex', align_items='center')
        )
    def _create_result_viewer_box(self):
        result_viewer_label = widgets.Label('Result Viewer:')
        result_viewer_tab_refresh_btn = widgets.Button(
            description='Refresh',
            button_style='info',
            disabled=False
        )
        result_viewer_intro_box = widgets.HBox([result_viewer_label, result_viewer_tab_refresh_btn])
        result_viewer_tab = widgets.Tab()
        temp_result_viewer_widgets_dict={}
        for method_name in METHOD_NAME_LIST:
            result_viewer_solve_index = widgets.IntText(
                value=-1,
                description='solve_index:',
                disabled=False,
                layout=widgets.Layout(width='15%'),
                style={'description_width': 'initial'}
            )
            data_from_irls_widget = widgets.Checkbox(
                value=False,
                description='data_from_irls',
                disabled=False
            )
            material_widget = widgets.Dropdown(
                options=['SiC', 'Si', 'GaN'],
                value='SiC',
                description='Material:',
                disabled=False,
                layout=widgets.Layout(width='15%'),
                style={'description_width': 'initial'}
            )
            material_doping_type_widget = widgets.Dropdown(
                options=['N', 'P'],
                value='N',
                description='Doping Type:',
                disabled=False,
                layout=widgets.Layout(width='15%'),
                style={'description_width': 'initial'}
            )
            Ti_selector_widget = widgets.VBox()
            unchanged_part_box = widgets.VBox([
                widgets.HBox([result_viewer_solve_index, data_from_irls_widget,
                              material_widget, material_doping_type_widget]),
                Ti_selector_widget
            ])
            fig1_widget = widgets.VBox()
            fig2_widget = widgets.VBox()
            fig3_widget = widgets.VBox()
            fig4_widget = widgets.VBox()
            fig1_wwidget = widgets.VBox([fig1_widget])
            fig2_wwidget = widgets.VBox([fig2_widget])
            fig3_wwidget = widgets.VBox([fig3_widget])
            fig4_wwidget = widgets.VBox([fig4_widget])
            updated_part_box = widgets.VBox([
                widgets.HBox([fig1_wwidget, fig2_wwidget]),
                widgets.HBox([fig3_wwidget, fig4_wwidget])
            ])
            unchanged_part = {
                'main_box': unchanged_part_box,
                'sub_box_solve_index': result_viewer_solve_index,
                'sub_box_data_from_irls': data_from_irls_widget,
                'sub_box_material': material_widget,
                'sub_box_material_doping_type': material_doping_type_widget,
                'sub_box_Ti_selector': Ti_selector_widget
            }
            updated_part = {
                'main_box': updated_part_box,
                'sub_box_fig1_ww': fig1_wwidget,
                'sub_box_fig2_ww': fig2_wwidget,
                'sub_box_fig3_ww': fig3_wwidget,
                'sub_box_fig4_ww': fig4_wwidget,
                'sub_box_fig1_w': fig1_widget,
                'sub_box_fig2_w': fig2_widget,
                'sub_box_fig3_w': fig3_widget,
                'sub_box_fig4_w': fig4_widget
            }
            temp_result_viewer_widgets_dict[method_name] = {
                'main_box': widgets.VBox([unchanged_part_box, updated_part_box]),
                'sub_dict_unchanged_part': unchanged_part,
                'sub_dict_updated_part': updated_part
            }
        result_viewer_tab.children = [temp_result_viewer_widgets_dict[m_n]['main_box'] for m_n in METHOD_NAME_LIST]
        result_viewer_tab.titles = METHOD_NAME_LIST
        self.result_viewer_box = widgets.VBox([result_viewer_intro_box, result_viewer_tab])
        self._result_viewer_widgets_dict = {
            'main_box': self.result_viewer_box,
            'sub_dict_intro': {
                'main_box': result_viewer_intro_box,
                'sub_box_label': result_viewer_label,
                'sub_box_refresh_btn': result_viewer_tab_refresh_btn
            },
            'sub_dict_tab': {
                'main_box': result_viewer_tab,
                'sub_dict_tab': temp_result_viewer_widgets_dict
            }
        }
    def _update_fig1_fig2(self, method_name: str):
        current_method = getattr(self, method_name)
        if 'Ti_selector' in self._result_viewer_widgets_dict['sub_dict_tab']['sub_dict_tab'][method_name]\
            ['sub_dict_unchanged_part'].keys():
                Ti_selector = self._result_viewer_widgets_dict['sub_dict_tab']['sub_dict_tab'][method_name]\
                    ['sub_dict_unchanged_part']['Ti_selector']
        iTi = Ti_selector.selector.value
        result_viewer_solve_index = self._result_viewer_widgets_dict['sub_dict_tab']['sub_dict_tab'][method_name]\
            ['sub_dict_unchanged_part']['sub_box_solve_index']
        data_from_irls_widget = self._result_viewer_widgets_dict['sub_dict_tab']['sub_dict_tab'][method_name]\
            ['sub_dict_unchanged_part']['sub_box_data_from_irls']
        if data_from_irls_widget.value:
            target_solve_history = current_method.irls_solve_history
        else:
            target_solve_history = current_method.solve_history
        if method_name=='L1':
            self.temp_target_solve_history = target_solve_history
        Ti_list = np.array(target_solve_history[result_viewer_solve_index.value]['Ti_list'])
        trans_t = current_method.t
        trans_c = current_method.A_extended @ \
            target_solve_history[result_viewer_solve_index.value]['f'][:,iTi]
        trans_c_exp = current_method.C[Ti_list[iTi],:]
        trans_rms = target_solve_history[result_viewer_solve_index.value]['rms_list'][iTi]
        trans_plotter = self._result_viewer_widgets_dict['sub_dict_tab']['sub_dict_tab'][method_name]\
            ['sub_dict_updated_part']['trans_plotter']
        trans_plotter._update_data(t=trans_t, c=trans_c, c_exp=trans_c_exp, rms=trans_rms)
        ldlts_s = current_method.s
        ldlts_f = target_solve_history[result_viewer_solve_index.value]['f'][1:,iTi]
        ldlts_plotter = self._result_viewer_widgets_dict['sub_dict_tab']['sub_dict_tab'][method_name]\
            ['sub_dict_updated_part']['ldlts_plotter']
        ldlts_plotter._update_data(s=ldlts_s, f=ldlts_f)
    def _set_event_Ti_selector(self) -> None:
        for method_name in METHOD_NAME_LIST:
            if 'Ti_selector' in self._result_viewer_widgets_dict['sub_dict_tab']['sub_dict_tab'][method_name]\
                        ['sub_dict_unchanged_part'].keys():
                Ti_selector = self._result_viewer_widgets_dict['sub_dict_tab']['sub_dict_tab'][method_name]\
                        ['sub_dict_unchanged_part']['Ti_selector']
            else:
                self.logger.info(f'no Ti_selector for {method_name}')
                continue
            # Early Binding
            Ti_selector.selector.observe(
                lambda change, m=method_name, ts=Ti_selector: self._handle_ti_selector_change(change, m, ts),
                names='value'
            )
    def _set_event_refresh_btn(self) -> None:
        result_viewer_tab_refresh_btn = self._result_viewer_widgets_dict['sub_dict_intro']['sub_box_refresh_btn']
        def click(b):
            self._bind_fig_widgets()
            self._set_event_Ti_selector()
        result_viewer_tab_refresh_btn.on_click(click)
    def _handle_ti_selector_change(self, change, method_name, ti_selector):
        ti_selector._obs_selector(change)
        self._update_fig1_fig2(method_name)
    def _bind_fig_widgets(self, all_figsize: tuple = (7,5), all_fs: int = 10) -> None:
        for method_name in METHOD_NAME_LIST:
            current_tab = self._result_viewer_widgets_dict['sub_dict_tab']['sub_dict_tab'][method_name]
            result_viewer_solve_index = current_tab['sub_dict_unchanged_part']['sub_box_solve_index']
            data_from_irls_widget = current_tab['sub_dict_unchanged_part']['sub_box_data_from_irls']
            material_widget = current_tab['sub_dict_unchanged_part']['sub_box_material']
            material_doping_type_widget = current_tab['sub_dict_unchanged_part']['sub_box_material_doping_type']
            # try to get Ti_list from solve_history
            if getattr(self,method_name).solve_history:
                Ti_list = np.array(getattr(self,method_name).solve_history[result_viewer_solve_index.value]['Ti_list'])
                T_list = np.array(getattr(self,method_name).T)[Ti_list]
                result_viewer_Ti_selector = Ti_Selector(Ti_list, T_list)
                # bind Ti selector
                current_tab['sub_dict_unchanged_part']['sub_box_Ti_selector'].children = [result_viewer_Ti_selector.box]
                current_tab['sub_dict_unchanged_part']['Ti_selector']= result_viewer_Ti_selector
                # bind fig widgets
                plt.ioff()
                # close pre widget
                current_tab['sub_dict_updated_part']['sub_box_fig1_w'].close()
                current_tab['sub_dict_updated_part']['sub_box_fig2_w'].close()
                current_tab['sub_dict_updated_part']['sub_box_fig3_w'].close()
                current_tab['sub_dict_updated_part']['sub_box_fig4_w'].close()
                # create new widget
                current_tab['sub_dict_updated_part']['sub_box_fig1_w'] = widgets.VBox()
                current_tab['sub_dict_updated_part']['sub_box_fig2_w'] = widgets.VBox()
                current_tab['sub_dict_updated_part']['sub_box_fig3_w'] = widgets.VBox()
                current_tab['sub_dict_updated_part']['sub_box_fig4_w'] = widgets.VBox()
                # bind to ww
                current_tab['sub_dict_updated_part']['sub_box_fig1_ww'].children = \
                    [current_tab['sub_dict_updated_part']['sub_box_fig1_w']]
                current_tab['sub_dict_updated_part']['sub_box_fig2_ww'].children = \
                    [current_tab['sub_dict_updated_part']['sub_box_fig2_w']]
                current_tab['sub_dict_updated_part']['sub_box_fig3_ww'].children = \
                    [current_tab['sub_dict_updated_part']['sub_box_fig3_w']]
                current_tab['sub_dict_updated_part']['sub_box_fig4_ww'].children = \
                    [current_tab['sub_dict_updated_part']['sub_box_fig4_w']]
                # input canvas
                trans_plotter = getattr(self,method_name).plot_trans(
                    solve_index = result_viewer_solve_index.value, data_from_irls = data_from_irls_widget.value,
                    Ti = Ti_list[result_viewer_Ti_selector.selector.value], figsize = all_figsize, fs = all_fs,
                    return_widgets = True
                )
                current_tab['sub_dict_updated_part']['sub_box_fig1_w'].children = [trans_plotter.fig.canvas]
                ldlts_plotter = getattr(self,method_name).plot_ldlts(
                    solve_index = result_viewer_solve_index.value, data_from_irls = data_from_irls_widget.value,
                    Ti = Ti_list[result_viewer_Ti_selector.selector.value], figsize = all_figsize, fs = all_fs,
                    return_widgets = True
                )
                current_tab['sub_dict_updated_part']['sub_box_fig2_w'].children = [ldlts_plotter.fig.canvas]
                ldlts_T_plotter = getattr(self,method_name).plot_ldlts_T(
                    solve_index = result_viewer_solve_index.value, data_from_irls = data_from_irls_widget.value,
                    figsize = all_figsize, fs = all_fs, c_abs_min = 1e-2, return_widgets = True
                )
                current_tab['sub_dict_updated_part']['sub_box_fig3_w'].children = [ldlts_T_plotter.fig.canvas]
                arrh_plotter = getattr(self,method_name).plot_arrh(
                    solve_index = result_viewer_solve_index.value, data_from_irls = data_from_irls_widget.value,
                    figsize = all_figsize, fs = all_fs, c_abs_min = 1e-2, return_widgets = True,
                    material = material_widget.value, dopant_species = material_doping_type_widget.value
                )
                current_tab['sub_dict_updated_part']['sub_box_fig4_w'].children = [arrh_plotter.fig.canvas]
                current_tab['sub_dict_updated_part']['trans_plotter'] = trans_plotter
                current_tab['sub_dict_updated_part']['ldlts_plotter'] = ldlts_plotter
                current_tab['sub_dict_updated_part']['ldlts_T_plotter'] = ldlts_T_plotter
                current_tab['sub_dict_updated_part']['arrh_plotter'] = arrh_plotter
            else:
                self.logger.info(f'Bind {method_name} fig widgets failed, no solve_history')
                continue


    def press_init_ldlts_method_btn(self, b):
        # get current input value
        temp_kwargs = {}
        for name,widget in self._init_ldlts_method_widgets_dict.items():
            temp_kwargs[name] = widget.value
        # init
        self._init_ldlts_method(**temp_kwargs)
        
    def _set_event(self):
        self._init_ldlts_method_btn.on_click(self.press_init_ldlts_method_btn)
        self._set_event_refresh_btn()
    
        
__all__ = ["ARRH_PLOTTER","ARRH_PLOTTER_2D"]

from .._Material import Material
from .._Trap import Trap
from .._typing import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy.constants as sci_const
import matplotlib.patches as patches

class ARRH_PLOTTER:
    """
    Interactive Arrhenius plot fitter for deep-level defect analysis.
    
    This class creates an interactive plot for analyzing semiconductor defects 
    using Arrhenius plots. Requires the ipympl module for interactive functionality.
    
    Key Features:
    1. Visualizes data on an Arrhenius plot with semiconductor materials information
    2. Allows selection/deselection of points using left/right mouse buttons
    3. Calculates centroid positions based on amplitude (c) for points at same temperature (x)
    4. Fits linear relationships using centroid positions to compute activation energy and capture cross-section
    5. Groups selected points for defect classification (stored in trap_groups attribute)
    6. Saves/loads current point selections (excluding grouped points)
    
    Attributes:
        trap_groups (list): Stores grouped defect information
        selected (set): Indices of currently selected points
        grouped_points (set): Indices of points that have been grouped
    
    Methods:
        get_group(): Creates a new group from currently selected points
        save_selected(): Saves indices of selected points to file
        load_selected(): Loads indices of selected points from file
    """
    @staticmethod
    def sec_ax_forward(x):
        return 1 / (x + 1e-20) / sci_const.k * sci_const.e
    @staticmethod
    def sec_ax_inverse(T):
        return 1 / (T + 1e-20) / sci_const.k * sci_const.e
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray, cmap: str = 'coolwarm',
                 point_scale: int | float = 10, lw: int | float = 0.8, peak_point_scale: int | float = 10,
                 peak_lw: int | float = 0.8, line_lw: int | float = 1.0, colorbar_scale: str = 'log',
                 sec_ax_max_n_locator: int = 7, material: str = 'Silicon', dopant_species: str = 'N',
                 power_T: int | float = 2, data_plot_dict: TransientDataPlotType | None = None, c_abs_min: float = 1e-4,
                 c_polarity: str = 'both', use_time_constant: bool = True, figsize: tuple = (12, 8),
                 fs: int = 14) -> None:
        """
        Initialize the Arrhenius plot fitter instance.
        
        Parameters:
            x (array-like): 1/kT values [eV^{-1}]
            y (array-like): -ln(e_m/T^p) values where p = power_T
            c (array-like): LDLTS amplitude signal values
            cmap (str): Colormap for visualization (default: 'coolwarm')
            point_scale (float): Size of unselected points (default: 10)
            lw (float): Line width for unselected points (default: 0.8)
            peak_point_scale (float): Size of selected points (default: 10)
            peak_lw (float): Line width for selected points (default: 0.8)
            line_lw (float): Line width for fitted lines (default: 1.0)
            colorbar_scale (str): Colorbar scaling type ('log' or 'linear') (default: 'log')
            sec_ax_max_n_locator (int): Number of ticks on secondary temperature axis (default: 7)
            material (str): Semiconductor material type for defect parameter calculation 
                            ('Silicon', 'GaN', 'SiC') (default: 'Silicon')
            dopant_species (str): Semiconductor doping type for defect parameter calculation 
                                  ('N' for n-type, 'P' for p-type) (default: 'N')
            power_T (int|float): Power_T value used for y-axis data in defect parameter calculation (default: 2)
            data_plot_dict (TransientDataPlotType | None): Dictionary of plot configuration parameters. Default includes:
                {
                    'C_plot_factor': 1.0,     # Amplitude scaling factor
                    'T_label': 'Temperature [$K$]',        # Secondary temperature axis label
                    'LDLTS_label': 'LDLTS Signal [$a.u.$]' # Colorbar label
                }
            c_abs_min (float): Minimum absolute value for amplitude cutoff (default: 1e-4)
            c_polarity (str): Amplitude polarity display option: 
                              'positive'/'negative' to show only positive/negative peaks, 
                              'both' for both (default: 'both')
        """
        if data_plot_dict is None:
            self.data_plot_dict = {}
            self.data_plot_dict['C_plot_factor'] = 1.0
            self.data_plot_dict['T_label'] = 'Temperature [$K$]'
            self.data_plot_dict['em_label'] = 'Emission Rate [$s^{-1}$]'
            self.data_plot_dict['DLTS_label'] = 'DLTS Signal [$a.u.$]'
            self.data_plot_dict['LDLTS_label'] = 'LDLTS Signal [$a.u.$]'
            self.data_plot_dict['C_unit'] = r'$pF$'
        else:
            self.data_plot_dict = data_plot_dict
        try:
            self.mat = getattr(Material,material)()
        except:
            print('unknow material, add material in class Material')
        self.fontdict={'family':'DejaVu Sans','size':fs}
        self.fs = fs
        self.dopant_species = str.upper(dopant_species)
        print(f'Cross Section & Activation Energy Calculation Based On Material: {self.dopant_species}-type {self.mat.name} with Temperature Coeff.: {power_T}')
        self.power_T = power_T
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.c = np.asarray(c)
        self.colorbar_scale = colorbar_scale
        self.c_abs_min = c_abs_min
        self.c_polarity = c_polarity
        self.cmap = cmap
        # data preprocessing
        if self.c_polarity == 'both':
            abs_max_c = np.max(np.abs(self.c))
            if self.colorbar_scale == 'log':
                self.norm = colors.SymLogNorm(vmin=-abs_max_c*self.data_plot_dict['C_plot_factor'], vmax=abs_max_c*self.data_plot_dict['C_plot_factor'], linthresh=self.c_abs_min)
            else:
                self.norm = colors.Normalize(vmin=-abs_max_c*self.data_plot_dict['C_plot_factor'], vmax=abs_max_c*self.data_plot_dict['C_plot_factor'])
        elif self.c_polarity == 'positive':
            colormap = plt.colormaps.get_cmap(self.cmap)
            self.cmap = colors.LinearSegmentedColormap.from_list('positive_part', colormap(np.linspace(0.5, 1, 256)))
            self.x = self.x[self.c > self.c_abs_min]
            self.y = self.y[self.c > self.c_abs_min]
            self.c = self.c[self.c > self.c_abs_min]
            if self.colorbar_scale == 'log':
                self.norm = colors.SymLogNorm(vmin=np.min(self.c)*self.data_plot_dict['C_plot_factor'], vmax=np.max(self.c)*self.data_plot_dict['C_plot_factor'], linthresh=self.c_abs_min)
            else:
                self.norm = colors.Normalize(vmin=np.min(self.c)*self.data_plot_dict['C_plot_factor'], vmax=np.max(self.c)*self.data_plot_dict['C_plot_factor'])
        elif self.c_polarity == 'negative':
            colormap = plt.colormaps.get_cmap(self.cmap)
            self.cmap = colors.LinearSegmentedColormap.from_list('negative_part', colormap(np.linspace(0, 0.5, 256)))
            self.x = self.x[self.c < -self.c_abs_min]
            self.y = self.y[self.c < -self.c_abs_min]
            self.c = self.c[self.c < -self.c_abs_min]
            if self.colorbar_scale == 'log':
                self.norm = colors.SymLogNorm(vmin=np.min(self.c)*self.data_plot_dict['C_plot_factor'], vmax=np.max(self.c)*self.data_plot_dict['C_plot_factor'], linthresh=self.c_abs_min)
            else:
                self.norm = colors.Normalize(vmin=np.min(self.c)*self.data_plot_dict['C_plot_factor'], vmax=np.max(self.c)*self.data_plot_dict['C_plot_factor'])
        else:
            raise ValueError('c_polarity?')
        from scipy.spatial import cKDTree
        self.points_tree = cKDTree(np.column_stack((self.x, self.y)))
        self.selected_mask = np.zeros(len(self.x), dtype=bool)
        self.last_update_mask = self.selected_mask.copy()
        self.last_grouped_mask = np.zeros(len(self.x), dtype=bool)
        self.fit_needed = False
        self.selected = set()
        self.trap_groups = []
        self.grouped_points = set()
        self.line_artists = []
        self.line_labels = []
        self.fig = plt.figure(figsize=figsize, layout='tight')
        self.ax = plt.gca()
        self.point_scale = point_scale
        self.lw = lw
        self.peak_point_scale = peak_point_scale
        self.peak_lw = peak_lw
        self.line_lw = line_lw
        self.sec_ax_max_n_locator = sec_ax_max_n_locator
        self.grouped_scatter = self.ax.scatter(
            np.empty((0, 2)), [],
            c=[], cmap=self.cmap, norm=self.norm,
            marker='s',
            s=self.point_scale,
            linewidths=self.lw
        )
        self.unselected_scatter = self.ax.scatter(
            np.empty((0, 2)), [],
            c=[], cmap=self.cmap, norm=self.norm,
            marker='o',
            s=self.point_scale,
            linewidths=self.lw
        )
        self.selected_scatter = self.ax.scatter(
            np.empty((0, 2)), [],
            c=[], cmap=self.cmap, norm=self.norm,
            marker='x',
            s=self.peak_point_scale,
            linewidths=self.peak_lw
        )
        self.ax.set_xlabel(r'$1/kT\ [eV^{-1}]$',fontdict=self.fontdict)
        if use_time_constant:
            self.ax.set_ylabel(fr'$ln(\tau \cdot T^{{{power_T}}})$',fontdict=self.fontdict)
        else:
            self.ax.set_ylabel(fr'$-ln(e_m/T^{{{power_T}}})$',fontdict=self.fontdict)
        self.sc = self.ax.scatter(self.x, self.y, c=self.c * self.data_plot_dict['C_plot_factor'],
                                 cmap=self.cmap, s=self.point_scale, linewidths=self.lw)
        self.cmin, self.cmax = self.sc.get_clim()
        self.sc.remove()
        self.colorbar = self.fig.colorbar(
            plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
            label=self.data_plot_dict['LDLTS_label'],
            norm=self.norm,
            ax=self.ax,
            format='%.0e'
        )
        self.colorbar.ax.set_ylabel(self.data_plot_dict['LDLTS_label'],fontdict=self.fontdict)
        self.sec_ax = self.ax.secondary_xaxis('top', functions=(self.sec_ax_forward, self.sec_ax_inverse))
        self.sec_ax.set_xlabel(self.data_plot_dict['T_label'],fontdict=self.fontdict)
        self.sec_ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        self.sec_ax.xaxis.set_major_locator(plt.MaxNLocator(self.sec_ax_max_n_locator))
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.is_drawing = False
        self.temp_line_artist = None
        self.temp_line_labels = None
        self.button = None
        self.line = None
        self.legend = None
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.header_visible = False
        self.fig.canvas.resizable = False
        self.fixed_xlim = self.ax.get_xlim()
        self.fixed_ylim = self.ax.get_ylim()
        for tt in self.ax.get_xticklabels():
            tt.set_fontsize(self.fs)
            tt.set_fontfamily(self.fontdict['family'])
        for tt in self.ax.get_yticklabels():
            tt.set_fontsize(self.fs)
            tt.set_fontfamily(self.fontdict['family'])
        for tt in self.colorbar.ax.get_yticklabels():
            tt.set_fontsize(self.fs)
            tt.set_fontfamily(self.fontdict['family'])
        for tt in self.sec_ax.get_xticklabels():
            tt.set_fontsize(self.fs)
            tt.set_fontfamily(self.fontdict['family'])
        # init update
        self._update_layers(force_full_update=True)
        self.box = self.fig.canvas
    def _update_layers(self, force_full_update=False):
        # get all indices
        all_indices = np.arange(len(self.x))
        grouped_mask = np.isin(all_indices, list(self.grouped_points))
        # judge if need update
        needs_update = force_full_update
        needs_update |= not np.array_equal(self.selected_mask, self.last_update_mask)
        needs_update |= not np.array_equal(grouped_mask, self.last_grouped_mask)
        if not needs_update:
            return
        # create masks
        grouped_mask = np.isin(all_indices, list(self.grouped_points))
        ungrouped_mask = ~grouped_mask
        selected_mask = np.isin(all_indices, list(self.selected))
        unsel_ungrouped_mask = ungrouped_mask & ~selected_mask
        if force_full_update:
            self.grouped_scatter.set_offsets(np.empty((0, 2)))
            self.grouped_scatter.set_array(np.array([]))
            self.unselected_scatter.set_offsets(np.empty((0, 2)))
            self.unselected_scatter.set_array(np.array([]))
            self.selected_scatter.set_offsets(np.empty((0, 2)))
            self.selected_scatter.set_array(np.array([]))
        # update grouped_scatter
        if np.any(grouped_mask):
            grouped_points = np.column_stack((self.x[grouped_mask], self.y[grouped_mask]))
            self.grouped_scatter.set_offsets(grouped_points)
            self.grouped_scatter.set_array(self.c[grouped_mask] * self.data_plot_dict['C_plot_factor'])
        else:
            self.grouped_scatter.set_offsets(np.empty((0, 2)))
            self.grouped_scatter.set_array(np.array([]))
        # update last_grouped_mask
        self.last_grouped_mask = grouped_mask
        # update selected_scatter
        sel_points = np.column_stack((self.x[selected_mask], self.y[selected_mask]))
        self.selected_scatter.set_offsets(sel_points)
        self.selected_scatter.set_array(self.c[selected_mask] * self.data_plot_dict['C_plot_factor'])
        # update unselected_scatter
        unselected_points = np.column_stack((self.x[unsel_ungrouped_mask], self.y[unsel_ungrouped_mask]))
        self.unselected_scatter.set_offsets(unselected_points)
        self.unselected_scatter.set_array(self.c[unsel_ungrouped_mask] * self.data_plot_dict['C_plot_factor'])
        self.last_update_mask = self.selected_mask.copy()
        self.fig.canvas.draw_idle()
    def _fit_and_draw_line(self):
        if self.temp_line_artist:
            self.temp_line_artist.remove()
            self.temp_line_artist = None
            self.temp_line_labels = None
        if not self.selected:
            return
        indices = list(self.selected)
        x_sel = self.x[indices]
        y_sel = self.y[indices]
        c_sel = self.c[indices]
        unique_x = np.unique(x_sel)
        y_fit = []
        c_fit = []
        for x_val in unique_x:
            mask = x_sel == x_val
            group_y = y_sel[mask]
            group_c = c_sel[mask]
            c_sum = np.sum(group_c)
            y_ave = np.sum(group_y * group_c) / c_sum
            y_fit.append(y_ave)
            c_fit.append(c_sum)
        # check if has enough points to fit
        if len(unique_x) < 2:
            return
        try:
            # liner fit
            coeffs = np.polyfit(unique_x, y_fit, 1)
        except np.linalg.LinAlgError:
            return
        a, b = coeffs
        c_avg = np.mean(c_fit)
        color = self.selected_scatter.cmap(self.norm(c_avg * self.data_plot_dict['C_plot_factor']))
        # generate line data (fixed range)
        x_min, x_max = self.fixed_xlim
        x_line = np.linspace(x_min, x_max, 100)
        y_line = a * x_line + b
        # plot line & add label
        activation_energy = a
        cross_section = None
        if self.dopant_species == 'N':
            cross_section = np.exp(-b) * self.mat.T**2 / self.mat.Nc / self.mat.vth_n
            self.temp_line_artist, = self.ax.plot(x_line, y_line, color=color, lw=self.line_lw)
        elif self.dopant_species == 'P':
            cross_section = np.exp(-b) * self.mat.T**2 / self.mat.Nv / self.mat.vth_p
            self.temp_line_artist, = self.ax.plot(x_line, y_line, color=color, lw=self.line_lw)
        # create an oddh.Trap instance
        temp_T = 1/np.array(unique_x)/sci_const.k*sci_const.e
        temp_amp = np.array(c_fit)
        temp_argsort = np.argsort(temp_T)
        temp_T = temp_T[temp_argsort]
        temp_amp = temp_amp[temp_argsort]
        temp_trap = Trap(Ea=activation_energy, sigma=cross_section, T_power=self.power_T, fun_amplitude_T=lambda T:np.interp(T,temp_T,temp_amp),
                         material=self.mat.name, material_doping_type=self.dopant_species, trap_type='majority')
        if cross_section is not None:
            # add label
            plot_c = c_avg * self.data_plot_dict['C_plot_factor']
            try:
                label = fr"$E_a={activation_energy:.3f}eV$"+'\n'+\
                fr"$\sigma_{str.lower(self.dopant_species)}={cross_section:.2e}cm^2$"+'\n'+\
                fr"$T_{{\tau = 1ms}}={temp_trap.get_T_from_fixed_tau(0.001):.2f}K$"+'\n'+\
                fr"$T_{{\tau = 10ms}}={temp_trap.get_T_from_fixed_tau(0.01):.2f}K$"+'\n'+\
                fr"$\Delta C={plot_c:.3f}${self.data_plot_dict['C_unit']}"
            except Exception as e:
                print(f"Error in _fit_and_draw_line: {e}, may be the wrong material input")
                label = fr"$E_a={activation_energy:.3f}eV$"
            self.temp_line_labels = label
            self._update_line_legend()
            self.ax.set_xlim(self.fixed_xlim)
            self.ax.set_ylim(self.fixed_ylim)
    def _update_line_legend(self):
        if self.legend:
            self.legend.remove()
            self.legend = None
        # if has any line, create legend
        if self.line_artists or self.temp_line_artist:
            handles = []
            labels = []
            # add grouped line
            for i, artist in enumerate(self.line_artists):
                handles.append(artist)
                labels.append(self.line_labels[i])
            # add temp_line
            if self.temp_line_artist and self.temp_line_labels:
                handles.append(self.temp_line_artist)
                labels.append(self.temp_line_labels)
            # create legend
            self.legend = self.ax.legend(handles, labels)
            self.legend.set_visible(True)
    def _on_press(self, event):
        if event.inaxes != self.ax or event.button not in (1, 3):
            return
        self.button = event.button
        self.start_x = event.xdata
        self.start_y = event.ydata
        self.is_drawing = True
        # init rectangle
        color = 'red' if self.button == 1 else 'blue'
        self.rect = patches.Rectangle(
            (self.start_x, self.start_y), 0, 0,
            edgecolor=color, facecolor=color, alpha=0.3
        )
        self.ax.add_patch(self.rect)
        self.fig.canvas.draw_idle()
    def _on_motion(self, event):
        if not self.is_drawing or event.inaxes != self.ax:
            return
        # update rectangle
        current_x = event.xdata
        current_y = event.ydata
        width = current_x - self.start_x
        height = current_y - self.start_y
        self.rect.set_width(width)
        self.rect.set_height(height)
        self.fig.canvas.draw_idle()
    def _on_release(self, event):
        if not self.is_drawing:
            return
        # clear rectangle
        self.rect.remove()
        self.rect = None
        self.is_drawing = False
        # release action should be in the ax
        if event.xdata is None or event.ydata is None:
            return
        # calc region
        x0, y0 = self.start_x, self.start_y
        x1, y1 = event.xdata, event.ydata
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        # calc ungrouped points' indices in the region
        ungrouped_mask = ~self.last_grouped_mask
        in_rect_x = (self.x >= x_min) & (self.x <= x_max) & ungrouped_mask
        in_rect_y = (self.y >= y_min) & (self.y <= y_max) & ungrouped_mask
        in_rect_indices = np.where(in_rect_x & in_rect_y)[0]
        # left click
        if self.button == 1:
            self.selected |= set(in_rect_indices)
        # right click
        elif self.button == 3:
            self.selected -= set(in_rect_indices)
        # update mask
        self.selected_mask = np.zeros(len(self.x), dtype=bool)
        for idx in self.selected:
            if idx < len(self.selected_mask):
                self.selected_mask[idx] = True
        self._update_layers()
        # fit
        self._fit_and_draw_line()
    def get_group(self):
        """
        Group the currently selected scatter points
        """
        if not self.selected:
            print("No points selected to form a group.")
            return
        try:
            # get indices
            valid_indices = [i for i in self.selected if i < len(self.x)]
            if not valid_indices:
                return
            # get data
            indices = list(valid_indices)
            x_sel = self.x[indices]
            y_sel = self.y[indices]
            c_sel = self.c[indices]
            # group in same x value
            group_dict = {}
            for i in range(len(x_sel)):
                x_val = x_sel[i]
                if x_val not in group_dict:
                    group_dict[x_val] = {'x': x_val, 'y_vals': [], 'c_vals': []}
                group_dict[x_val]['y_vals'].append(y_sel[i])
                group_dict[x_val]['c_vals'].append(c_sel[i])
            # if don't have enough points
            if len(group_dict) < 2:
                print("Insufficient distinct x-values for fitting (need at least 2)")
                return
            # fit
            x_fit = []
            y_fit = []
            c_fit = []
            for key, group in group_dict.items():
                y_vals = group['y_vals']
                c_vals = group['c_vals']
                c_sum = sum(c_vals)
                # calc center of mass (c value)
                y_weighted = sum(y * c for y, c in zip(y_vals, c_vals))
                y_fit.append(y_weighted / c_sum)
                c_fit.append(c_sum)
                x_fit.append(key)
            # liner fit
            try:
                coeffs = np.polyfit(x_fit, y_fit, 1)
            except np.linalg.LinAlgError as e:
                print(f"Linear fit error: {e}")
                return
            a, b = coeffs
            c_avg = np.mean(c_fit) if c_fit else 0
            norm_value = self.norm(c_avg * self.data_plot_dict['C_plot_factor'])
            color = self.selected_scatter.cmap(norm_value)
            x_min, x_max = self.fixed_xlim
            x_line = np.linspace(x_min, x_max, 100)
            y_line = a * x_line + b
            activation_energy = a
            cross_section = None
            if self.dopant_species == 'N':
                cross_section = np.exp(-b) * self.mat.T**2 / self.mat.Nc / self.mat.vth_n
            elif self.dopant_species == 'P':
                cross_section = np.exp(-b) * self.mat.T**2 / self.mat.Nv / self.mat.vth_p
            else:
                return
            # create an oddh.Trap instance
            temp_T = 1/np.array(x_fit)/sci_const.k*sci_const.e
            temp_amp = np.array(c_fit)
            temp_argsort = np.argsort(temp_T)
            temp_T = temp_T[temp_argsort]
            temp_amp = temp_amp[temp_argsort]
            temp_trap = Trap(Ea=activation_energy, sigma=cross_section, T_power=self.power_T, fun_amplitude_T=lambda T:np.interp(T,temp_T,temp_amp),
                             material=self.mat.name, material_doping_type=self.dopant_species, trap_type='majority')
            # save data to self.trap_groups
            group_data = {
                'members': np.column_stack((x_sel, y_sel, c_sel)),
                'x_fit': x_fit,
                'y_fit': y_fit,
                'c_fit': c_fit,
                'T_power': self.power_T,
                'material': self.mat.name,
                'Ea': activation_energy,
                'sigma0': cross_section,
                'dopant_species': self.dopant_species,
                'C_plot_factor': self.data_plot_dict['C_plot_factor'],
                'intercept':b,
                'trap_instance':temp_trap
            }
            self.trap_groups.append(group_data)
            # clear selected points
            self.grouped_points.update(indices)
            self.selected = set()
            # create grouped lines
            num_g = len(self.line_artists)
            ls_list = ['-',':','--','-.',(0,(1,10)),(0,(1,5)),(0,(1,1)),(5,(10,3)),(0,(5,10)),(0,(5,5)),(0,(5,1))]
            linestyle = ls_list[num_g] if num_g < len(ls_list) else ls_list[-1]
            # create new line
            new_line, = self.ax.plot(x_line, y_line, color=color, 
                                    lw=self.line_lw, linestyle=linestyle)
            self.line_artists.append(new_line)
            # create grouped label
            plot_c = c_avg * self.data_plot_dict['C_plot_factor']
            label = fr"$Group\ {num_g+1}:$"+'\n'+\
            fr"$E_a={activation_energy:.3f}eV$"+'\n'+\
            fr"$\sigma_{str.lower(self.dopant_species)}={cross_section:.2e}cm^2$"+'\n'+\
            fr"$T_{{\tau = 1ms}}={temp_trap.get_T_from_fixed_tau(0.001):.2f}K$"+'\n'+\
            fr"$T_{{\tau = 10ms}}={temp_trap.get_T_from_fixed_tau(0.01):.2f}K$"+'\n'+\
            fr"$\Delta C={plot_c:.3f}${self.data_plot_dict['C_unit']}"
            self.line_labels.append(label)
            # remove temp_line
            if self.temp_line_artist:
                self.temp_line_artist.remove()
                self.temp_line_artist = None
            self._update_line_legend()
            self._update_layers()
        except Exception as e:
            import traceback
            print(f"Error in get_group: {e}")
            traceback.print_exc()
    def save_selected(self, filepath='./', filename='temp_trap_map_selected'):
        """
        Save the index of the currently selected scatter point to a file
        """
        temp = list(self.selected)
        if not temp:
            print("No points selected to save.")
            return
        np.savetxt(filepath + filename, temp)
    def load_selected(self, filepath='./', filename='temp_trap_map_selected'):
        """
        Load the indices of the selected scatter points from a file
        """
        try:
            temp = np.loadtxt(filepath + filename, dtype=np.int64)
            temp = set(temp)
            self.selected = temp
            self.selected_mask = np.zeros(len(self.x), dtype=bool)
            for idx in self.selected:
                if idx < len(self.selected_mask):
                    self.selected_mask[idx] = True
            self._update_layers()
            self._fit_and_draw_line()
        except Exception as e:
            print(f"Error loading selected points: {e}")


import matplotlib.lines as mlines
class ARRH_PLOTTER_2D(ARRH_PLOTTER):
    """
    分类交互式阿伦尼乌斯图拟合器。
    继承自 ARRH_PLOTTER，处理离散分类并完美处理生命周期冲突。
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray, **kwargs):
        # 强制允许所有标签通过过滤，并使用线性缩放以绕过对数报错
        kwargs['c_polarity'] = 'both'
        kwargs['colorbar_scale'] = 'linear'
        
        # 提前计算好颜色索引，供后续使用
        self.unique_labels = np.unique(c)
        self.label_to_color_idx = {label: idx % 10 for idx, label in enumerate(self.unique_labels)}
        self.c_color_idx = np.array([self.label_to_color_idx[val] for val in c])
        
        # 执行父类初始化 (此时父类内部调用的 _update_layers 会被我们底下的 if 拦截掉)
        super().__init__(x, y, c, **kwargs)

        # 1. 移除父类生成的连续 Colorbar
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        # 2. 建立离散的分类颜色映射 (Categorical Colormap)
        self.cmap = plt.colormaps.get_cmap('tab10')
        self.norm = colors.Normalize(vmin=0, vmax=9)

        # 重新应用离散颜色映射到散点图
        self.grouped_scatter.set_cmap(self.cmap)
        self.grouped_scatter.set_norm(self.norm)
        self.unselected_scatter.set_cmap(self.cmap)
        self.unselected_scatter.set_norm(self.norm)
        self.selected_scatter.set_cmap(self.cmap)
        self.selected_scatter.set_norm(self.norm)

        # 3. 添加分类图例 (Category Legend)
        self._add_discrete_legend()

        # 4. 子类属性全部就绪，手动触发完整的图层刷新！
        self._subclass_ready = True 
        self._update_layers(force_full_update=True)

    def _add_discrete_legend(self):
        """添加一个静态图例来表示不同的峰值分类，并放置在图表右侧外部填补空白"""
        handles = []
        for label in self.unique_labels:
            idx = self.label_to_color_idx[label]
            color = self.cmap(self.norm(idx))
            handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', 
                                         markersize=8, label=f'Peak {int(label)}'))
        
        # 【修改这里】：利用 bbox_to_anchor 将图例移出主绘图区，放到坐标轴右侧
        self.category_legend = self.ax.legend(
            handles=handles, 
            title="Categories", 
            loc='center left',            # 图例的左侧对齐
            bbox_to_anchor=(1.02, 0.5),   # X坐标1.02（图表外部），Y坐标0.5（垂直居中）
            borderaxespad=0.
        )
        self.ax.add_artist(self.category_legend)
        
        # 强制刷新一下画布的排版引擎，确保留白被完美利用
        self.fig.canvas.draw_idle()
        
    def _update_layers(self, force_full_update=False):
        """重写图层更新，使用分类颜色索引"""
        # 【核心修复】：如果子类还没有完全准备好（处于 super().__init__ 阶段），直接跳过
        if not getattr(self, '_subclass_ready', False):
            return

        all_indices = np.arange(len(self.x))
        grouped_mask = np.isin(all_indices, list(self.grouped_points))
        
        needs_update = force_full_update
        needs_update |= not np.array_equal(self.selected_mask, self.last_update_mask)
        needs_update |= not np.array_equal(grouped_mask, self.last_grouped_mask)
        
        if not needs_update:
            return

        ungrouped_mask = ~grouped_mask
        selected_mask = np.isin(all_indices, list(self.selected))
        unsel_ungrouped_mask = ungrouped_mask & ~selected_mask

        if force_full_update:
            self.grouped_scatter.set_offsets(np.empty((0, 2)))
            self.unselected_scatter.set_offsets(np.empty((0, 2)))
            self.selected_scatter.set_offsets(np.empty((0, 2)))

        if np.any(grouped_mask):
            grouped_points = np.column_stack((self.x[grouped_mask], self.y[grouped_mask]))
            self.grouped_scatter.set_offsets(grouped_points)
            self.grouped_scatter.set_array(self.c_color_idx[grouped_mask]) 
        else:
            self.grouped_scatter.set_offsets(np.empty((0, 2)))

        self.last_grouped_mask = grouped_mask

        sel_points = np.column_stack((self.x[selected_mask], self.y[selected_mask]))
        self.selected_scatter.set_offsets(sel_points)
        self.selected_scatter.set_array(self.c_color_idx[selected_mask]) 

        unselected_points = np.column_stack((self.x[unsel_ungrouped_mask], self.y[unsel_ungrouped_mask]))
        self.unselected_scatter.set_offsets(unselected_points)
        self.unselected_scatter.set_array(self.c_color_idx[unsel_ungrouped_mask]) 

        self.last_update_mask = self.selected_mask.copy()
        self.fig.canvas.draw_idle()

    def _fit_and_draw_line(self):
        """算术平均线性拟合"""
        if self.temp_line_artist:
            self.temp_line_artist.remove()
            self.temp_line_artist = None
            self.temp_line_labels = None
            
        if not self.selected:
            return
            
        indices = list(self.selected)
        x_sel = self.x[indices]
        y_sel = self.y[indices]
        c_sel = self.c[indices] 

        unique_x = np.unique(x_sel)
        y_fit = []
        
        for x_val in unique_x:
            mask = x_sel == x_val
            group_y = y_sel[mask]
            y_fit.append(np.mean(group_y)) 
            
        if len(unique_x) < 2:
            return
            
        try:
            coeffs = np.polyfit(unique_x, y_fit, 1)
        except np.linalg.LinAlgError:
            return
            
        a, b = coeffs
        
        values, counts = np.unique(c_sel, return_counts=True)
        dominant_label = values[np.argmax(counts)]
        idx = self.label_to_color_idx[dominant_label]
        color = self.cmap(self.norm(idx))

        x_min, x_max = self.fixed_xlim
        x_line = np.linspace(x_min, x_max, 100)
        y_line = a * x_line + b

        activation_energy = a
        cross_section = None
        
        try:
            if hasattr(self, 'mat') and self.mat is not None:
                if self.dopant_species == 'N':
                    cross_section = np.exp(-b) * self.mat.T**2 / self.mat.Nc / self.mat.vth_n
                elif self.dopant_species == 'P':
                    cross_section = np.exp(-b) * self.mat.T**2 / self.mat.Nv / self.mat.vth_p
        except AttributeError:
            pass

        self.temp_line_artist, = self.ax.plot(x_line, y_line, color=color, lw=self.line_lw)
        
        if cross_section is not None:
            label = fr"$E_a={activation_energy:.3f}eV$"+'\n'+ fr"$\sigma={cross_section:.2e}cm^2$"
        else:
            label = fr"$E_a={activation_energy:.3f}eV$"
            
        self.temp_line_labels = label
        self._update_line_legend()
        self.ax.set_xlim(self.fixed_xlim)
        self.ax.set_ylim(self.fixed_ylim)

    def get_group(self):
        """保存算术平均拟合数据"""
        if not self.selected:
            print("No points selected to form a group.")
            return
            
        try:
            valid_indices = [i for i in self.selected if i < len(self.x)]
            if not valid_indices:
                return
                
            indices = list(valid_indices)
            x_sel = self.x[indices]
            y_sel = self.y[indices]
            c_sel = self.c[indices]

            group_dict = {}
            for i in range(len(x_sel)):
                x_val = x_sel[i]
                if x_val not in group_dict:
                    group_dict[x_val] = []
                group_dict[x_val].append(y_sel[i])

            if len(group_dict) < 2:
                print("Insufficient distinct x-values for fitting (need at least 2)")
                return

            x_fit = []
            y_fit = []
            for key, y_vals in group_dict.items():
                y_fit.append(np.mean(y_vals))
                x_fit.append(key)

            try:
                coeffs = np.polyfit(x_fit, y_fit, 1)
            except np.linalg.LinAlgError as e:
                print(f"Linear fit error: {e}")
                return
                
            a, b = coeffs
            
            values, counts = np.unique(c_sel, return_counts=True)
            dominant_label = values[np.argmax(counts)]
            idx = self.label_to_color_idx[dominant_label]
            color = self.cmap(self.norm(idx))

            x_min, x_max = self.fixed_xlim
            x_line = np.linspace(x_min, x_max, 100)
            y_line = a * x_line + b
            activation_energy = a
            cross_section = None
            
            if hasattr(self, 'mat') and self.mat is not None:
                if self.dopant_species == 'N':
                    cross_section = np.exp(-b) * self.mat.T**2 / self.mat.Nc / self.mat.vth_n
                elif self.dopant_species == 'P':
                    cross_section = np.exp(-b) * self.mat.T**2 / self.mat.Nv / self.mat.vth_p

            group_data = {
                'members': np.column_stack((x_sel, y_sel, c_sel)),
                'x_fit': x_fit,
                'y_fit': y_fit,
                'Ea': activation_energy,
                'sigma0': cross_section,
                'intercept': b
            }
            self.trap_groups.append(group_data)
            
            self.grouped_points.update(indices)
            self.selected = set()

            num_g = len(self.line_artists)
            ls_list = ['-',':','--','-.',(0,(1,10)),(0,(1,5)),(0,(1,1)),(5,(10,3))]
            linestyle = ls_list[num_g] if num_g < len(ls_list) else ls_list[-1]

            new_line, = self.ax.plot(x_line, y_line, color=color, lw=self.line_lw, linestyle=linestyle)
            self.line_artists.append(new_line)

            if cross_section is not None:
                label = fr"$Group\ {num_g+1}: E_a={activation_energy:.3f}eV, \sigma={cross_section:.2e}cm^2$"
            else:
                label = fr"$Group\ {num_g+1}: E_a={activation_energy:.3f}eV$"
                
            self.line_labels.append(label)

            if self.temp_line_artist:
                self.temp_line_artist.remove()
                self.temp_line_artist = None
                
            self._update_line_legend()
            self._update_layers()

        except Exception as e:
            import traceback
            print(f"Error in get_group: {e}")
            traceback.print_exc()
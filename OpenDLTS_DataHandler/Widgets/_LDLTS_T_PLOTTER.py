__all__ = ["LDLTS_T_PLOTTER"]
from .._typing import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.axes import Axes

class LDLTS_T_PLOTTER:
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray, figsize: tuple = (12,8), fs: int = 14,
                 data_plot_dict: dict | None = None, use_time_constant: bool = True,
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
            self.data_plot_dict['C_unit'] = 'a.u.'
        else:
            self.data_plot_dict = data_plot_dict
            if 'C_unit' not in self.data_plot_dict:
                self.data_plot_dict['C_unit'] = 'a.u.'
                
        fontdict={'family':'DejaVu Sans','size':fs}
        self.point_scale = point_scale
        self.lw = lw
        
        # data preprocessing
        if c_polarity == 'positive':
            mask = c >= c_abs_min
            colormap = plt.colormaps.get_cmap(cmap)
            cmap = colors.LinearSegmentedColormap.from_list('positive_part', colormap(np.linspace(0.5,1,256)))
        elif c_polarity == 'negative':
            mask = c <= -c_abs_min
            colormap = plt.colormaps.get_cmap(cmap)
            cmap = colors.LinearSegmentedColormap.from_list('negative_part', colormap(np.linspace(0,0.5,256)))
        elif c_polarity == 'both':
            mask = np.ones_like(c, dtype=bool) # 保留所有（后续逻辑处理 norm）
        else:
            raise ValueError("polarity = 'positive' | 'negative' | 'both'")
            
        x_filtered = x[mask]
        y_filtered = y[mask]
        c_filtered = c[mask]
        
        # Norm logic
        if c_filtered.size > 0:
            if c_polarity == 'both' and c_filtered[c_filtered <= -c_abs_min].size == 0:
                vmin, vmax = -np.max(c_filtered), np.max(c_filtered)
            elif c_polarity == 'both' and c_filtered[c_filtered >= c_abs_min].size == 0:
                vmin, vmax = np.min(c_filtered), -np.min(c_filtered)
            elif c_polarity == 'both':
                max_abs = np.max(np.abs(c_filtered))
                vmin, vmax = -max_abs, max_abs
            else:
                vmin, vmax = np.min(c_filtered), np.max(c_filtered)
                
            vmin *= self.data_plot_dict['C_plot_factor']
            vmax *= self.data_plot_dict['C_plot_factor']
            
            if colorbar_scale == 'lin':
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=c_abs_min)
        else:
            norm = colors.Normalize(vmin=-1, vmax=1)

        # 排序使得信号大的点画在最上面，并保存为类属性供交互使用
        sort_idx = np.argsort(np.abs(c_filtered))
        self.x = x_filtered[sort_idx]
        self.y = y_filtered[sort_idx]
        self.c = c_filtered[sort_idx]
        
        self.cmap = cmap
        self.norm = norm
        self.selected = set()
        
        # 交互控制变量
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.is_drawing = False
        self.button = None
        self.legend = None

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

        self.ax.set_yscale('log')
        
        # 初始化散点图（分为未选中和选中两层）
        self.unselected_scatter = self.ax.scatter(
            x=self.x, y=self.y, c=self.c * self.data_plot_dict['C_plot_factor'],
            cmap=self.cmap, norm=self.norm, marker='o',
            edgecolors='face', facecolors='none',
            s=self.point_scale, linewidths=self.lw
        )
        
        # 选中的点：增加红色边框和更大的尺寸以便于观察
        self.selected_scatter = self.ax.scatter(
            np.empty((0,)), np.empty((0,)), c=[],
            cmap=self.cmap, norm=self.norm, marker='o',
            edgecolors='red', facecolors='none',
            s=self.point_scale * 1.5, linewidths=self.lw * 1.5
        )

        self.ax.set_xlabel(self.data_plot_dict['T_label'], fontdict=fontdict)
        if use_time_constant:
            self.ax.set_ylabel(self.data_plot_dict['tau_label'], fontdict=fontdict)
        else:
            self.ax.set_ylabel(self.data_plot_dict['em_label'], fontdict=fontdict)
            
        # Colorbar
        self.cbar = self.fig.colorbar(
            plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
            label=self.data_plot_dict['LDLTS_label'], ax=self.ax, format='%.0e'
        )
        self.cbar.ax.set_ylabel(self.data_plot_dict['LDLTS_label'], fontdict=fontdict)
        
        for tt in self.ax.get_xticklabels() + self.ax.get_yticklabels() + self.cbar.ax.get_yticklabels():
            tt.set_fontsize(fs)
            tt.set_fontfamily(fontdict['family'])
            
        # 绑定鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        
        self.box = self.fig.canvas

    def _update_layers(self):
        """更新散点图图层，区分选中与未选中的点"""
        selected_mask = np.zeros(len(self.x), dtype=bool)
        if self.selected:
            selected_mask[list(self.selected)] = True
            
        unselected_mask = ~selected_mask
        
        # 更新未选中的点
        unselected_points = np.column_stack((self.x[unselected_mask], self.y[unselected_mask]))
        self.unselected_scatter.set_offsets(unselected_points)
        self.unselected_scatter.set_array(self.c[unselected_mask] * self.data_plot_dict['C_plot_factor'])
        
        # 更新选中的点
        if np.any(selected_mask):
            selected_points = np.column_stack((self.x[selected_mask], self.y[selected_mask]))
            self.selected_scatter.set_offsets(selected_points)
            self.selected_scatter.set_array(self.c[selected_mask] * self.data_plot_dict['C_plot_factor'])
        else:
            self.selected_scatter.set_offsets(np.empty((0, 2)))
            self.selected_scatter.set_array(np.array([]))
            
        self.fig.canvas.draw_idle()

    def _update_legend(self):
        """计算选中点 c 值总和，并在 Legend 中显示"""
        if self.legend:
            self.legend.remove()
            self.legend = None
            
        if not self.selected:
            self.fig.canvas.draw_idle()
            return
            
        # 提取选中点的 c 值并求和
        selected_indices = list(self.selected)
        c_sum = np.sum(self.c[selected_indices]) * self.data_plot_dict['C_plot_factor']
        
        # 格式化文本
        label = fr"$\sum \Delta C = {c_sum:.3e}$ {self.data_plot_dict['C_unit']}"
        
        # 创建一个 Dummy Artist 用于在 Legend 中显示文本，而不影响现有图形
        dummy_line = plt.Line2D([0], [0], color='none', marker='o', 
                                markeredgecolor='red', markerfacecolor='none', 
                                markersize=8, label=label)
        
        self.legend = self.ax.legend(handles=[dummy_line], loc='best', 
                                     framealpha=0.8, edgecolor='black')
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button not in (1, 3):
            return
        self.button = event.button
        self.start_x = event.xdata
        self.start_y = event.ydata
        self.is_drawing = True
        
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
        self.rect.remove()
        self.rect = None
        self.is_drawing = False
        
        if event.xdata is None or event.ydata is None:
            return
            
        # 计算框选范围
        x0, y0 = self.start_x, self.start_y
        x1, y1 = event.xdata, event.ydata
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        
        # 查找框内的点（由于 y 轴是对数坐标，event.ydata 已经是实际的指数数值，直接比较即可）
        in_rect_mask = (self.x >= x_min) & (self.x <= x_max) & (self.y >= y_min) & (self.y <= y_max)
        in_rect_indices = np.where(in_rect_mask)[0]
        
        # 左键选中，右键取消
        if self.button == 1:
            self.selected |= set(in_rect_indices)
        elif self.button == 3:
            self.selected -= set(in_rect_indices)
            
        # 更新显示层和图注
        self._update_layers()
        self._update_legend()
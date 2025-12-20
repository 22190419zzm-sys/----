from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QSizePolicy
import matplotlib.pyplot as plt


class PlotStyleConfig:
    """通用的绘图样式配置类，用于统一管理样式参数"""
    def __init__(self, parent_dialog=None):
        self.parent_dialog = parent_dialog
        self.settings = QSettings("GTLab", "SpectraPro_v4")
        
    def get_default_style_params(self):
        """获取默认样式参数"""
        return {
            # Figure
            'fig_width': 10.0,
            'fig_height': 6.0,
            'fig_dpi': 300,
            'aspect_ratio': 0.6,
            
            # Font
            'font_family': 'Times New Roman',
            'axis_title_fontsize': 20,
            'tick_label_fontsize': 16,
            'legend_fontsize': 10,
            'title_fontsize': 18,
            
            # Lines
            'line_width': 1.2,
            'line_style': '-',
            'marker_size': 4,
            'marker_style': 'o',
            
            # Ticks
            'tick_direction': 'in',
            'tick_len_major': 8,
            'tick_len_minor': 4,
            'tick_width': 1.0,
            
            # Grid
            'show_grid': True,
            'grid_alpha': 0.2,
            'grid_linestyle': '-',
            
            # Spines
            'spine_top': True,
            'spine_bottom': True,
            'spine_left': True,
            'spine_right': True,
            'spine_width': 2.0,
            
            # Legend
            'show_legend': True,
            'legend_frame': True,
            'legend_loc': 'best',
            
            # Colors
            'color_raw': 'gray',
            'color_fit': 'blue',
            'color_residual': 'black',
            
            # Text labels
            'title_text': '',
            'validation_title_fontsize': 18,
            'validation_title_pad': 10.0,
            'validation_title_show': True,
            'xlabel_text': 'Wavenumber (cm⁻¹)',
            'validation_xlabel_fontsize': 20,
            'validation_xlabel_pad': 10.0,
            'validation_xlabel_show': True,
            'ylabel_main_text': 'Intensity',
            'ylabel_residual_text': 'Residuals',
            'validation_ylabel_fontsize': 20,
            'validation_ylabel_pad': 10.0,
            'validation_ylabel_show': True,
            'legend_raw_label': 'Raw Low-Conc. Spectrum',
            'legend_fit_label': 'Fitted Organic Contribution',
            'show_label_a': True,
            'show_label_b': True,
            'label_a_text': '(A)',
            'label_b_text': '(B)',
        }
    
    def load_style_params(self, window_name):
        """从QSettings加载样式参数"""
        params = self.get_default_style_params()
        prefix = f"{window_name}/style/"
        
        for key in params.keys():
            value = self.settings.value(f"{prefix}{key}", params[key])
            # 类型转换
            if isinstance(params[key], bool):
                params[key] = value == 'true' if isinstance(value, str) else bool(value)
            elif isinstance(params[key], int):
                params[key] = int(value) if value is not None else params[key]
            elif isinstance(params[key], float):
                params[key] = float(value) if value is not None else params[key]
            else:
                params[key] = value if value is not None else params[key]
        
        return params
    
    def save_style_params(self, window_name, params):
        """保存样式参数到QSettings"""
        prefix = f"{window_name}/style/"
        for key, value in params.items():
            self.settings.setValue(f"{prefix}{key}", value)
        self.settings.sync()
    
    def apply_style_to_axes(self, ax, params):
        """将样式参数应用到matplotlib axes（发表级别质量）"""
        # 强制使用 Times New Roman 字体（发表级别要求）
        font_family = 'Times New Roman'
        axis_title_fontsize = params.get('axis_title_fontsize', 20)
        tick_label_fontsize = params.get('tick_label_fontsize', 16)
        
        # 启用 LaTeX 数学格式支持
        plt.rcParams['text.usetex'] = False  # 如果系统有 LaTeX，可以设为 True
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
        
        # 设置标签字体（强制 Times New Roman）
        ax.xaxis.label.set_fontsize(axis_title_fontsize)
        ax.yaxis.label.set_fontsize(axis_title_fontsize)
        ax.title.set_fontsize(params.get('title_fontsize', 18))
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
        ax.title.set_fontfamily(font_family)
        
        # 发表级别刻度设置：direction='in', top=True, right=True
        ax.tick_params(axis='both', which='major', 
                      direction='in',  # 强制向内
                      length=params.get('tick_len_major', 8),
                      width=params.get('tick_width', 1.0),
                      labelsize=tick_label_fontsize,
                      top=True,  # 顶部刻度
                      right=True)  # 右侧刻度
        ax.tick_params(axis='both', which='minor',
                      direction='in',  # 强制向内
                      length=params.get('tick_len_minor', 4),
                      width=params.get('tick_width', 1.0),
                      top=True,  # 顶部刻度
                      right=True)  # 右侧刻度
        
        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(tick_label_fontsize)
        
        # 发表级别边框设置：linewidth=1.5，所有边框可见
        ax.spines['top'].set_visible(True)  # 强制显示顶部边框
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)  # 强制显示右侧边框
        
        spine_width = 1.5  # 发表级别标准：1.5
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        
        # 网格设置
        if params.get('show_grid', True):
            ax.grid(True, alpha=params.get('grid_alpha', 0.2),
                   linestyle=params.get('grid_linestyle', '-'))
        else:
            ax.grid(False)
        
        # 图例设置（强制 Times New Roman）
        legend = ax.get_legend()
        if legend:
            legend_fontsize = params.get('legend_fontsize', 10)
            try:
                legend.set_fontsize(legend_fontsize)
            except AttributeError:
                for text in legend.get_texts():
                    text.set_fontsize(legend_fontsize)
            
            # 强制使用 Times New Roman
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            legend_font.set_family(font_family)
            legend_font.set_size(legend_fontsize)
            for text in legend.get_texts():
                text.set_fontproperties(legend_font)
            legend.set_frame_on(params.get('legend_frame', True))
            if params.get('legend_loc'):
                legend.set_loc(params.get('legend_loc', 'best'))


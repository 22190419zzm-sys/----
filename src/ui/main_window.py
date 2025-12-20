
# flake8: noqa
import os
import sys
import glob
import json
import pickle
import math
import random
import itertools
import traceback
import warnings
import re
from collections import defaultdict
from pathlib import Path
from importlib import util
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from scipy.optimize import nnls

from PyQt6.QtCore import Qt, QPoint, QSize, QSettings, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, 
    QGroupBox, QCheckBox, QLineEdit, QSpinBox, 
    QDoubleSpinBox, QPushButton, QFormLayout, 
    QLabel, QMessageBox, QTextEdit, QWidget,
    QToolButton, QSizePolicy, QHBoxLayout, QScrollArea, QSpacerItem,
    QComboBox, QFileDialog, QTabWidget, QGridLayout, QFrame,
    QListWidget, QListWidgetItem, QAbstractItemView, QMenu,
    QRadioButton, QButtonGroup, QColorDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QInputDialog
)

from src.config.constants import C_H, C_C, C_K, C_CM_TO_HZ
from src.config.plot_config import PlotStyleConfig
from src.utils.fonts import setup_matplotlib_fonts
from src.utils.helpers import natural_sort_key, group_files_by_name
from src.utils.lazy_import import lazy_import
from src.utils.cache import get_cache_manager
# 延迟导入非必需的模块
from src.core.preprocessor import DataPreProcessor
# 以下模块延迟导入
# from src.core.generators import SyntheticDataGenerator
# from src.core.matcher import SpectralMatcher
# from src.core.transformers import AutoencoderTransformer, AdaptiveMineralFilter, TORCH_AVAILABLE
# from src.core.rruff_loader import RRUFFLibraryLoader, PeakMatcher
# NonNegativeTransformer 需要立即导入，因为NMF分析中会用到
from src.core.transformers import NonNegativeTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from src.ui.widgets.custom_widgets import (
    CollapsibleGroupBox,
    SmartDoubleSpinBox,
    UnlimitedNumericInput,
)
from src.ui.canvas import MplCanvas
from src.ui.windows.nmf_window import NMFResultWindow
from src.ui.windows.plot_window import MplPlotWindow

from src.ui.controllers import DataController

from src.ui.windows.quantitative_window import QuantitativeResultWindow, QuantitativeAnalysisDialog
from src.ui.windows.nmf_validation_window import NMFFitValidationWindow
from src.ui.windows.two_dcos_window import TwoDCOSWindow, TwoDCOSMarginalPlotWindow
from src.ui.windows.classification_window import ClassificationResultWindow
from src.ui.windows.dae_window import DAEComparisonWindow
from src.ui.windows.batch_plot_window import BatchPlotWindow
from src.ui.windows.function_windows import FunctionWindow
from src.ui.panels.nmf_panel import NMFPanelMixin
from src.ui.panels.cos_panel import COSPanelMixin
from src.ui.panels.classify_panel import ClassifyPanelMixin
from src.ui.panels.publication_style_panel import PublicationStylePanel
from src.ui.panels.spectrum_scan_panel import SpectrumScanPanel
from src.ui.panels.peak_matching_panel import PeakMatchingPanel
from src.ui.windows.style_matching_window import StyleMatchingWindow
# 注意：多子图配置已删除

# 延迟导入（避免循环依赖）
try:
    from src.core.matcher import SpectralMatcher
except ImportError:
    SpectralMatcher = None

try:
    from src.core.rruff_loader import RRUFFLibraryLoader
except ImportError:
    RRUFFLibraryLoader = None

try:
    from src.core.generators import SyntheticDataGenerator
except ImportError:
    SyntheticDataGenerator = None

try:
    from src.core.transformers import TORCH_AVAILABLE, AutoencoderTransformer
except ImportError:
    TORCH_AVAILABLE = False
    AutoencoderTransformer = None

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

class SpectraConfigDialog(QDialog, NMFPanelMixin, COSPanelMixin, ClassifyPanelMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("光谱数据处理工作站（GTzhou组 - Pro版）")
        
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        
        self.resize(1200, 900)
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        self.settings = QSettings("GTLab", "SpectraPro_v4") # 更新版本号
        
        self.main_layout = QVBoxLayout(self)
        # 减小整体垂直间距，让各个区域更加紧凑
        self.main_layout.setSpacing(2) 
        self.main_layout.setContentsMargins(10, 10, 10, 10) 
        
        self.individual_control_widgets = {} 
        self.nmf_component_control_widgets = {}  # NMF组分的独立Y轴控制
        self.nmf_component_rename_widgets = {}  # NMF组分的图例重命名
        self.legend_rename_widgets = {}
        self.group_waterfall_control_widgets = {}  # 组瀑布图的独立堆叠位移控制
        self.last_fixed_H = None  # 存储上一次标准NMF运行得到的H矩阵，用于组分回归模式（预滤波空间）
        self.last_fixed_H_original = None  # 存储原始空间的H矩阵，用于绘图和验证
        
        # 初始化数据缓存
        from src.core.plot_data_cache import PlotDataCache
        self.plot_data_cache = PlotDataCache(max_cache_size=200)
        
        # 缓存配置管理器实例（避免重复创建）
        self._config_manager = None
        self.last_pca_model = None  # 存储训练好的 PCA 模型实例
        self.last_common_x = None  # 存储NMF分析时的波数轴，用于定量分析
        self.nmf_target_component_index = 0  # 存储NMF目标组分索引，默认选择Component 1

        # 数据增强与光谱匹配相关
        self.library_matcher = None  # 存储 SpectralMatcher 实例
        self.library_folder_path = ""  # 存储标准库路径
        self.data_generator = None  # 存储 SyntheticDataGenerator 实例
        self.dae_window = None  # Deep Autoencoder 可视化窗口
        self.batch_plot_window = None  # 批量绘图窗口
        self.data_controller = DataController()  # 将数据逻辑托管到可复用控制器

        # 绘图与功能窗口管理
        self.plot_windows = {}          # 所有绘图窗口
        self.nmf_window = None          # NMF 结果窗口
        # 存储当前激活的绘图窗口引用，用于叠加分析
        self.active_plot_window = None
        # 功能配置窗口缓存，避免重复创建
        self.function_windows = {}
        
        # RRUFF匹配相关（延迟初始化）
        self.rruff_loader = None  # RRUFF库加载器
        self._rruff_peak_matcher = None  # 峰值匹配器（延迟初始化）
        self.rruff_match_results = {}  # {group_name: [match_results]} 存储匹配结果
        self.selected_rruff_spectra = {}  # {group_name: set([rruff_names])} 存储已选中的RRUFF光谱名称
        self.rruff_match_window = None  # RRUFF匹配结果窗口
        
        # 缓存管理器
        self.cache_manager = get_cache_manager()
        
        # 字体设置标志（延迟到首次绘图时）
        self._fonts_setup = False
        
        self.setup_ui()
        self._ensure_nmf_marker_defaults()

        # 连接所有样式参数的自动更新信号
        self._connect_all_style_update_signals()

        # 在所有初始化完成后加载设置
        self.load_settings()
    
    def _ensure_fonts_setup(self):
        """确保字体已设置（延迟到首次绘图时）"""
        if not self._fonts_setup:
            from src.utils.fonts import setup_matplotlib_fonts
            setup_matplotlib_fonts()
            self._fonts_setup = True

    def update_nmf_target_component(self, index):
        """更新NMF目标组分索引（由NMFResultWindow调用）"""
        self.nmf_target_component_index = index
    
    def get_nmf_target_component_index(self):
        """获取当前NMF目标组分索引"""
        # 如果NMF窗口存在，优先从窗口获取
        if hasattr(self, 'nmf_window') and self.nmf_window is not None:
            if hasattr(self.nmf_window, 'get_target_component_index'):
                return self.nmf_window.get_target_component_index()
        return self.nmf_target_component_index
    
    def open_quantitative_dialog(self):
        """打开定量校准分析对话框"""
        try:
            dialog = QuantitativeAnalysisDialog(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开定量校准分析窗口失败: {e}")

    def open_batch_plot_window(self):
        """打开批量绘图窗口"""
        try:
            if not hasattr(self, 'batch_plot_window') or self.batch_plot_window is None:
                self.batch_plot_window = BatchPlotWindow(self)
            # 如果RRUFF库已加载，同步到批量绘图窗口
            if self.rruff_loader and hasattr(self.batch_plot_window, 'rruff_loader'):
                # 更新批量绘图窗口的RRUFF库预处理参数
                self.batch_plot_window.update_rruff_preprocessing()
            self.batch_plot_window.show()
            self.batch_plot_window.raise_()
            self.batch_plot_window.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开批量绘图窗口失败: {e}")
            import traceback
            traceback.print_exc()

    def _parse_optional_float(self, text):
        text = text.strip()
        if not text: return None
        try: return float(text)
        except ValueError: raise ValueError(f"输入 '{text}' 必须是数字。")

    def _ensure_nmf_marker_defaults(self):
        """
        确保与 NMF 标记样式相关的控件属性存在。
        在部分代码路径中，即便界面未完整构建，也需要这些属性避免 AttributeError。
        """
        if not hasattr(self, 'nmf_marker_size'):
            sb = QSpinBox()
            sb.setRange(-999999999, 999999999)
            sb.setValue(8)
            self.nmf_marker_size = sb
        if not hasattr(self, 'nmf_marker_style'):
            combo = QComboBox()
            combo.addItems(['o', 'x', 's', 'D', '^'])
            combo.setCurrentText('o')
            self.nmf_marker_style = combo

    def _create_h_layout(self, widgets):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(5)
        for wid in widgets: l.addWidget(wid)
        return w
    
    def _create_color_picker_button(self, color_input):
        """创建颜色选择器按钮的辅助方法"""
        color_button = QPushButton("颜色")
        color_button.setFixedSize(30, 25)
        color_button.setToolTip("点击选择颜色")
        
        # 根据当前颜色设置按钮背景
        def update_button_color():
            color_str = color_input.text().strip()
            if color_str:
                try:
                    # 尝试将颜色字符串转换为QColor
                    if color_str.startswith('#'):
                        qcolor = QColor(color_str)
                    else:
                        # 使用matplotlib颜色名称
                        import matplotlib.colors as mcolors
                        rgba = mcolors.to_rgba(color_str)
                        qcolor = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                    color_button.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #999;")
                except:
                    color_button.setStyleSheet("background-color: #CCCCCC; border: 1px solid #999;")
            else:
                color_button.setStyleSheet("background-color: #CCCCCC; border: 1px solid #999;")
        
        # 初始设置
        update_button_color()
        
        # 当颜色输入改变时更新按钮颜色
        color_input.textChanged.connect(update_button_color)
        
        # 点击按钮时打开颜色选择器
        def pick_color():
            color_str = color_input.text().strip()
            initial_color = QColor(128, 128, 128)  # 默认灰色
            
            if color_str:
                try:
                    if color_str.startswith('#'):
                        initial_color = QColor(color_str)
                    else:
                        # 使用matplotlib颜色名称
                        import matplotlib.colors as mcolors
                        rgba = mcolors.to_rgba(color_str)
                        initial_color = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                except:
                    pass
            
            color = QColorDialog.getColor(initial_color, self, "选择颜色")
            if color.isValid():
                # 将QColor转换为十六进制字符串
                color_input.setText(color.name())
        
        color_button.clicked.connect(pick_color)
        return color_button
    
    def _clear_layout_recursively(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None: widget.deleteLater()
                elif item.layout() is not None: self._clear_layout_recursively(item.layout())
    
    def _connect_all_style_update_signals(self):
        """连接所有样式参数控件的自动更新信号（通用方法）"""
        # 初始化更新定时器（防抖）
        if not hasattr(self, '_style_update_timer'):
            self._style_update_timer = QTimer()
            self._style_update_timer.setSingleShot(True)
            self._style_update_timer.timeout.connect(self._auto_update_all_plots)
        
        # 连接所有样式参数控件的信号
        # 注意：只连接样式参数，不连接数据相关参数（如文件夹、文件选择等）
        
        # 字体和标题参数
        if hasattr(self, 'font_family_combo'):
            self.font_family_combo.currentTextChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'axis_title_font_spin'):
            self.axis_title_font_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'tick_label_font_spin'):
            self.tick_label_font_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'legend_font_spin'):
            self.legend_font_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 标题和标签参数
        if hasattr(self, 'title_input'):
            self.title_input.textChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'title_font_spin'):
            self.title_font_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'title_pad_spin'):
            self.title_pad_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'title_show_check'):
            self.title_show_check.stateChanged.connect(self._on_style_param_changed)
        
        if hasattr(self, 'xlabel_input'):
            self.xlabel_input.textChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'xlabel_font_spin'):
            self.xlabel_font_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'xlabel_pad_spin'):
            self.xlabel_pad_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'xlabel_show_check'):
            self.xlabel_show_check.stateChanged.connect(self._on_style_param_changed)
        
        if hasattr(self, 'ylabel_input'):
            self.ylabel_input.textChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'ylabel_font_spin'):
            self.ylabel_font_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'ylabel_pad_spin'):
            self.ylabel_pad_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'ylabel_show_check'):
            self.ylabel_show_check.stateChanged.connect(self._on_style_param_changed)
        
        # 图例参数
        if hasattr(self, 'show_legend_check'):
            self.show_legend_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'legend_frame_check'):
            self.legend_frame_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'legend_loc_combo'):
            self.legend_loc_combo.currentTextChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'legend_column_spin'):
            self.legend_column_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'legend_columnspacing_spin'):
            self.legend_columnspacing_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'legend_labelspacing_spin'):
            self.legend_labelspacing_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'legend_handlelength_spin'):
            self.legend_handlelength_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 刻度样式参数
        if hasattr(self, 'tick_direction_combo'):
            self.tick_direction_combo.currentTextChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'tick_len_major_spin'):
            self.tick_len_major_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'tick_len_minor_spin'):
            self.tick_len_minor_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'tick_width_spin'):
            self.tick_width_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 边框参数
        if hasattr(self, 'spine_width_spin'):
            self.spine_width_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'spine_top_check'):
            self.spine_top_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'spine_right_check'):
            self.spine_right_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'spine_bottom_check'):
            self.spine_bottom_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'spine_left_check'):
            self.spine_left_check.stateChanged.connect(self._on_style_param_changed)
        
        # 网格参数
        if hasattr(self, 'show_grid_check'):
            self.show_grid_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'grid_alpha_spin'):
            self.grid_alpha_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 图尺寸和DPI
        if hasattr(self, 'fig_width_spin'):
            self.fig_width_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'fig_height_spin'):
            self.fig_height_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'fig_dpi_spin'):
            self.fig_dpi_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 垂直线样式参数（参考线，应用于所有谱图）
        if hasattr(self, 'vertical_lines_input'):
            self.vertical_lines_input.textChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'vertical_line_color_input'):
            self.vertical_line_color_input.textChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'vertical_line_width_spin'):
            self.vertical_line_width_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'vertical_line_style_combo'):
            self.vertical_line_style_combo.currentTextChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'vertical_line_alpha_spin'):
            self.vertical_line_alpha_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'rruff_ref_lines_enabled_check'):
            self.rruff_ref_lines_enabled_check.stateChanged.connect(self._on_style_param_changed)
        
        # 匹配线样式参数（快速响应）
        if hasattr(self, 'match_line_color_input'):
            self.match_line_color_input.textChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'match_line_width_spin'):
            self.match_line_width_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'match_line_style_combo'):
            self.match_line_style_combo.currentTextChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'match_line_alpha_spin'):
            self.match_line_alpha_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 预处理参数（需要强制重新加载数据，因为会影响预处理结果）
        # 这些参数改变时需要清空预处理缓存并重新读取数据
        preprocess_params = [
            'qc_check', 'qc_threshold_spin', 'be_check', 'be_temp_spin',
            'smoothing_check', 'smoothing_window_spin', 'smoothing_poly_spin',
            'baseline_als_check', 'lam_spin', 'p_spin',
            'baseline_poly_check', 'baseline_points_spin', 'baseline_poly_spin',
            'normalization_combo', 'global_transform_combo',
            'global_log_base_combo', 'global_log_offset_spin',
            'global_sqrt_offset_spin',
            # 注意：derivative_check、quadratic_fit_check已删除
            'global_y_offset_spin', 'x_min_phys_input', 'x_max_phys_input'
        ]
        
        for param_name in preprocess_params:
            widget = getattr(self, param_name, None)
            if widget:
                if hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(lambda: self._on_style_param_changed(force_data_reload=True))
                elif hasattr(widget, 'textChanged'):
                    widget.textChanged.connect(lambda: self._on_style_param_changed(force_data_reload=True))
                elif hasattr(widget, 'currentTextChanged'):
                    widget.currentTextChanged.connect(lambda: self._on_style_param_changed(force_data_reload=True))
                elif hasattr(widget, 'stateChanged'):
                    widget.stateChanged.connect(lambda: self._on_style_param_changed(force_data_reload=True))
        
        # 预处理参数（需要强制重新加载数据，因为会影响预处理结果）
        # 这些参数改变时需要清空预处理缓存并重新读取数据
        def connect_preprocess_param(widget_name, signal_type='stateChanged'):
            widget = getattr(self, widget_name, None)
            if widget:
                if signal_type == 'stateChanged' and hasattr(widget, 'stateChanged'):
                    widget.stateChanged.connect(lambda: self._on_style_param_changed(force_data_reload=True))
                elif signal_type == 'valueChanged' and hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(lambda: self._on_style_param_changed(force_data_reload=True))
                elif signal_type == 'textChanged' and hasattr(widget, 'textChanged'):
                    widget.textChanged.connect(lambda: self._on_style_param_changed(force_data_reload=True))
                elif signal_type == 'currentTextChanged' and hasattr(widget, 'currentTextChanged'):
                    widget.currentTextChanged.connect(lambda: self._on_style_param_changed(force_data_reload=True))
        
        # 预处理参数（也需要触发自动更新，包括RRUFF库的预处理）
        if hasattr(self, 'qc_check'):
            connect_preprocess_param('qc_check', 'stateChanged')
        if hasattr(self, 'qc_threshold_spin'):
            # UnlimitedNumericInput 使用 textChanged 信号
            if isinstance(self.qc_threshold_spin, UnlimitedNumericInput):
                connect_preprocess_param('qc_threshold_spin', 'textChanged')
            else:
                connect_preprocess_param('qc_threshold_spin', 'valueChanged')
        if hasattr(self, 'be_check'):
            connect_preprocess_param('be_check', 'stateChanged')
        if hasattr(self, 'be_temp_spin'):
            # UnlimitedNumericInput 使用 textChanged 信号
            if isinstance(self.be_temp_spin, UnlimitedNumericInput):
                connect_preprocess_param('be_temp_spin', 'textChanged')
            else:
                connect_preprocess_param('be_temp_spin', 'valueChanged')
        if hasattr(self, 'smoothing_check'):
            connect_preprocess_param('smoothing_check', 'stateChanged')
        if hasattr(self, 'smoothing_window_spin'):
            connect_preprocess_param('smoothing_window_spin', 'valueChanged')
        if hasattr(self, 'smoothing_poly_spin'):
            connect_preprocess_param('smoothing_poly_spin', 'valueChanged')
        if hasattr(self, 'baseline_als_check'):
            connect_preprocess_param('baseline_als_check', 'stateChanged')
        if hasattr(self, 'lam_spin'):
            # UnlimitedNumericInput 使用 textChanged 信号
            if isinstance(self.lam_spin, UnlimitedNumericInput):
                connect_preprocess_param('lam_spin', 'textChanged')
            else:
                connect_preprocess_param('lam_spin', 'valueChanged')
        if hasattr(self, 'p_spin'):
            # UnlimitedNumericInput 使用 textChanged 信号
            if isinstance(self.p_spin, UnlimitedNumericInput):
                connect_preprocess_param('p_spin', 'textChanged')
            else:
                connect_preprocess_param('p_spin', 'valueChanged')
        if hasattr(self, 'normalization_combo'):
            connect_preprocess_param('normalization_combo', 'currentTextChanged')
        if hasattr(self, 'global_transform_combo'):
            connect_preprocess_param('global_transform_combo', 'currentTextChanged')
        if hasattr(self, 'global_log_base_combo'):
            connect_preprocess_param('global_log_base_combo', 'currentTextChanged')
        if hasattr(self, 'global_log_offset_spin'):
            # UnlimitedNumericInput 使用 textChanged 信号
            if isinstance(self.global_log_offset_spin, UnlimitedNumericInput):
                connect_preprocess_param('global_log_offset_spin', 'textChanged')
            else:
                connect_preprocess_param('global_log_offset_spin', 'valueChanged')
        if hasattr(self, 'global_sqrt_offset_spin'):
            # UnlimitedNumericInput 使用 textChanged 信号
            if isinstance(self.global_sqrt_offset_spin, UnlimitedNumericInput):
                connect_preprocess_param('global_sqrt_offset_spin', 'textChanged')
            else:
                connect_preprocess_param('global_sqrt_offset_spin', 'valueChanged')
        if hasattr(self, 'global_y_offset_spin'):
            connect_preprocess_param('global_y_offset_spin', 'valueChanged')
        # 注意：derivative_check已删除，二次导数在预处理流程中应用
        # X轴物理截断也需要重新加载数据
        if hasattr(self, 'x_min_phys_input'):
            connect_preprocess_param('x_min_phys_input', 'textChanged')
        if hasattr(self, 'x_max_phys_input'):
            connect_preprocess_param('x_max_phys_input', 'textChanged')
        
        # 峰值检测参数（也需要触发自动更新）
        if hasattr(self, 'peak_check'):
            self.peak_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_height_spin'):
            self.peak_height_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_distance_spin'):
            self.peak_distance_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_prominence_spin'):
            self.peak_prominence_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_width_spin'):
            self.peak_width_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_wlen_spin'):
            self.peak_wlen_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_rel_height_spin'):
            self.peak_rel_height_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_show_label_check'):
            self.peak_show_label_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_label_font_combo'):
            self.peak_label_font_combo.currentTextChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_label_size_spin'):
            self.peak_label_size_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_label_color_input'):
            self.peak_label_color_input.textChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_label_bold_check'):
            self.peak_label_bold_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_label_rotation_spin'):
            self.peak_label_rotation_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_marker_shape_combo'):
            self.peak_marker_shape_combo.currentTextChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_marker_size_spin'):
            self.peak_marker_size_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'peak_marker_color_input'):
            self.peak_marker_color_input.textChanged.connect(self._on_style_param_changed)

    def apply_style_preset(self, preset_name: str):
        """
        应用出版质量样式预设。
        - 默认：保持用户当前设置
        - Icarus 单栏：适合单栏图 (约 8.5 cm 宽)
        - Icarus 双栏：适合双栏通栏图 (约 17 cm 宽)
        """
        # 仅在核心控件已创建后才应用
        if not hasattr(self, 'fig_width_spin'):
            return

        if preset_name == "默认":
            # 恢复为启动时的默认参数（不会影响用户后来手动保存的样式）
            defaults = getattr(self, "_style_default_params", None)
            if defaults:
                self.fig_width_spin.setValue(defaults.get("fig_width", self.fig_width_spin.value()))
                self.fig_height_spin.setValue(defaults.get("fig_height", self.fig_height_spin.value()))
                self.fig_dpi_spin.setValue(defaults.get("fig_dpi", self.fig_dpi_spin.value()))
                self.aspect_ratio_spin.setValue(defaults.get("aspect_ratio", self.aspect_ratio_spin.value()))
            # 不再强制改动字体/线宽等，其它样式沿用当前值

        elif preset_name == "Icarus 单栏":
            # 单栏图（大致 8.5 cm × 6 cm，对应英寸约 3.35 × 2.4）
            self.fig_width_spin.setValue(3.4)   # inches
            self.fig_height_spin.setValue(2.6)  # inches
            self.fig_dpi_spin.setValue(300)
            self.aspect_ratio_spin.setValue(2.6 / 3.4)

            self.font_family_combo.setCurrentText("Times New Roman")
            self.axis_title_font_spin.setValue(10)   # 轴标题
            self.tick_label_font_spin.setValue(8)    # 刻度
            self.legend_font_spin.setValue(8)        # 图例

            self.line_width_spin.setValue(1.0)
            self.tick_direction_combo.setCurrentText("in")
            self.tick_len_major_spin.setValue(6)
            self.tick_len_minor_spin.setValue(3)
            self.tick_width_spin.setValue(1.0)

            self.spine_width_spin.setValue(1.0)
            self.grid_alpha_spin.setValue(0.25)
            self.shadow_alpha_spin.setValue(0.15)

            self.legend_fontsize_spin.setValue(8)
            self.legend_column_spin.setValue(1)
            self.legend_columnspacing_spin.setValue(1.5)
            self.legend_labelspacing_spin.setValue(0.4)
            self.legend_handlelength_spin.setValue(1.5)

        elif preset_name == "Icarus 双栏":
            # 双栏通栏图（大致 17 cm × 6–7 cm，对应英寸约 6.7 × 2.6）
            self.fig_width_spin.setValue(6.7)   # inches
            self.fig_height_spin.setValue(2.8)  # inches
            self.fig_dpi_spin.setValue(300)
            self.aspect_ratio_spin.setValue(2.8 / 6.7)

            self.font_family_combo.setCurrentText("Times New Roman")
            self.axis_title_font_spin.setValue(10)
            self.tick_label_font_spin.setValue(8)
            self.legend_font_spin.setValue(8)

            self.line_width_spin.setValue(1.0)
            self.tick_direction_combo.setCurrentText("in")
            self.tick_len_major_spin.setValue(6)
            self.tick_len_minor_spin.setValue(3)
            self.tick_width_spin.setValue(1.0)

            self.spine_width_spin.setValue(1.0)
            self.grid_alpha_spin.setValue(0.25)
            self.shadow_alpha_spin.setValue(0.15)

            self.legend_fontsize_spin.setValue(8)
            self.legend_column_spin.setValue(1)
            self.legend_columnspacing_spin.setValue(1.5)
            self.legend_labelspacing_spin.setValue(0.4)
            self.legend_handlelength_spin.setValue(1.5)
        
        else:
            # 自定义预设：从保存的设置中加载
            custom_presets_json = self.settings.value("custom_style_presets", "{}")
            try:
                custom_presets = json.loads(custom_presets_json) if custom_presets_json else {}
                if preset_name in custom_presets:
                    self._apply_style_params(custom_presets[preset_name])
            except Exception as e:
                print(f"加载自定义预设失败: {e}")

        # 预设应用后，自动更新当前打开的所有图
        self._auto_update_all_plots()

        # 堆叠距离（也需要触发自动更新）
        # 注意：堆叠偏移已移至"样式与匹配"窗口的谱线扫描面板，不再需要连接
        # 注意：show_x_val_check、show_y_val_check已移至样式面板
    
    def _load_custom_presets(self):
        """加载自定义风格预设到下拉框"""
        # 基础预设
        presets = ["默认", "Icarus 单栏", "Icarus 双栏"]
        
        # 加载自定义预设
        custom_presets_json = self.settings.value("custom_style_presets", "{}")
        try:
            custom_presets = json.loads(custom_presets_json) if custom_presets_json else {}
            presets.extend(custom_presets.keys())
        except:
            pass
        
        self.style_preset_combo.clear()
        self.style_preset_combo.addItems(presets)
    
    def _get_font_family(self):
        """获取当前字体家族（从面板获取）"""
        if hasattr(self, 'publication_style_panel'):
            config = self.publication_style_panel.get_config()
            return config.publication_style.font_family
        # 向后兼容
        if hasattr(self, 'font_family_combo'):
            return self.font_family_combo.currentText()
        return 'Times New Roman'
    
    def _get_current_style_params(self):
        """获取当前所有样式参数（从面板获取）"""
        # 如果面板已创建，从面板获取配置
        if hasattr(self, 'publication_style_panel'):
            config = self.publication_style_panel.get_config()
            ps = config.publication_style
            
            return {
                'fig_width': ps.fig_width,
                'fig_height': ps.fig_height,
                'fig_dpi': ps.fig_dpi,
                'aspect_ratio': ps.aspect_ratio,
                'font_family': ps.font_family,
                'axis_title_font': ps.axis_title_fontsize,
                'tick_label_font': ps.tick_label_fontsize,
                'legend_font': ps.legend_fontsize,
                'line_width': ps.line_width,
                'line_style': ps.line_style,
                'tick_direction': ps.tick_direction,
                'tick_len_major': ps.tick_len_major,
                'tick_len_minor': ps.tick_len_minor,
                'tick_width': ps.tick_width,
                'show_grid': ps.show_grid,
                'grid_alpha': ps.grid_alpha,
                'shadow_alpha': ps.shadow_alpha,
                'show_legend': ps.show_legend,
                'legend_frame': ps.legend_frame,
                'legend_loc': ps.legend_loc,
                'legend_fontsize': ps.legend_fontsize,
                'legend_column': ps.legend_ncol,
                'legend_columnspacing': ps.legend_columnspacing,
                'legend_labelspacing': ps.legend_labelspacing,
                'legend_handlelength': ps.legend_handlelength,
                'spine_top': ps.spine_top,
                'spine_bottom': ps.spine_bottom,
                'spine_left': ps.spine_left,
                'spine_right': ps.spine_right,
                'spine_width': ps.spine_width,
                # 标题控制（新增）
                'xlabel_text': ps.xlabel_text,
                'xlabel_show': ps.xlabel_show,
                'xlabel_fontsize': ps.xlabel_fontsize,
                'xlabel_pad': ps.xlabel_pad,
                'ylabel_text': ps.ylabel_text,
                'ylabel_show': ps.ylabel_show,
                'ylabel_fontsize': ps.ylabel_fontsize,
                'ylabel_pad': ps.ylabel_pad,
                'title_text': ps.title_text,
                'title_show': ps.title_show,
                'title_fontsize': ps.title_fontsize,
                'title_pad': ps.title_pad,
            }
        
        # 向后兼容：如果面板未创建，返回空字典
        return {}
    
    def _apply_style_params(self, params):
        """应用样式参数到控件"""
        if not hasattr(self, 'fig_width_spin'):
            return
        
        self.fig_width_spin.setValue(params.get('fig_width', 10.0))
        self.fig_height_spin.setValue(params.get('fig_height', 6.0))
        self.fig_dpi_spin.setValue(params.get('fig_dpi', 300))
        self.aspect_ratio_spin.setValue(params.get('aspect_ratio', 0.6))
        self.font_family_combo.setCurrentText(params.get('font_family', 'Times New Roman'))
        self.axis_title_font_spin.setValue(params.get('axis_title_font', 20))
        self.tick_label_font_spin.setValue(params.get('tick_label_font', 16))
        self.legend_font_spin.setValue(params.get('legend_font', 10))
        self.line_width_spin.setValue(params.get('line_width', 1.2))
        self.line_style_combo.setCurrentText(params.get('line_style', '-'))
        self.tick_direction_combo.setCurrentText(params.get('tick_direction', 'in'))
        self.tick_len_major_spin.setValue(params.get('tick_len_major', 8))
        self.tick_len_minor_spin.setValue(params.get('tick_len_minor', 4))
        self.tick_width_spin.setValue(params.get('tick_width', 1.0))
        self.show_grid_check.setChecked(params.get('show_grid', False))
        self.grid_alpha_spin.setValue(params.get('grid_alpha', 0.2))
        self.shadow_alpha_spin.setValue(params.get('shadow_alpha', 0.25))
        self.show_legend_check.setChecked(params.get('show_legend', True))
        self.legend_frame_check.setChecked(params.get('legend_frame', True))
        self.legend_loc_combo.setCurrentText(params.get('legend_loc', 'best'))
        if hasattr(self, 'legend_fontsize_spin'):
            self.legend_fontsize_spin.setValue(params.get('legend_fontsize', 10))
        if hasattr(self, 'legend_column_spin'):
            self.legend_column_spin.setValue(params.get('legend_column', 1))
        if hasattr(self, 'legend_columnspacing_spin'):
            self.legend_columnspacing_spin.setValue(params.get('legend_columnspacing', 2.0))
        if hasattr(self, 'legend_labelspacing_spin'):
            self.legend_labelspacing_spin.setValue(params.get('legend_labelspacing', 0.5))
        if hasattr(self, 'legend_handlelength_spin'):
            self.legend_handlelength_spin.setValue(params.get('legend_handlelength', 2.0))
        self.spine_top_check.setChecked(params.get('spine_top', True))
        self.spine_bottom_check.setChecked(params.get('spine_bottom', True))
        self.spine_left_check.setChecked(params.get('spine_left', True))
        self.spine_right_check.setChecked(params.get('spine_right', True))
        self.spine_width_spin.setValue(params.get('spine_width', 2.0))
    
    def _manage_style_presets(self):
        """管理自定义风格预设对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("管理风格预设")
        dialog.setMinimumSize(600, 500)
        layout = QVBoxLayout(dialog)
        
        # 预设列表
        layout.addWidget(QLabel("自定义预设列表:"))
        preset_list = QListWidget()
        
        # 加载自定义预设
        custom_presets_json = self.settings.value("custom_style_presets", "{}")
        try:
            custom_presets = json.loads(custom_presets_json) if custom_presets_json else {}
        except:
            custom_presets = {}
        
        for name in custom_presets.keys():
            preset_list.addItem(name)
        
        layout.addWidget(preset_list)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        btn_save_current = QPushButton("保存当前设置为新预设")
        btn_save_current.clicked.connect(lambda: self._save_current_as_preset(dialog, preset_list, custom_presets))
        
        btn_load = QPushButton("加载选中预设")
        btn_load.clicked.connect(lambda: self._load_selected_preset(dialog, preset_list, custom_presets))
        
        btn_delete = QPushButton("删除选中预设")
        btn_delete.clicked.connect(lambda: self._delete_selected_preset(dialog, preset_list, custom_presets))
        
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dialog.accept)
        
        btn_layout.addWidget(btn_save_current)
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_delete)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        
        dialog.exec()
    
    def _save_current_as_preset(self, dialog, preset_list, custom_presets):
        """保存当前设置为新预设"""
        name, ok = QInputDialog.getText(dialog, "保存预设", "请输入预设名称:")
        if not ok or not name.strip():
            return
        
        name = name.strip()
        
        # 检查名称是否已存在
        if name in custom_presets or name in ["默认", "Icarus 单栏", "Icarus 双栏"]:
            QMessageBox.warning(dialog, "错误", f"预设名称 '{name}' 已存在，请使用其他名称。")
            return
        
        # 保存当前设置
        params = self._get_current_style_params()
        custom_presets[name] = params
        
        # 保存到设置
        self.settings.setValue("custom_style_presets", json.dumps(custom_presets))
        self.settings.sync()
        
        # 更新列表
        preset_list.addItem(name)
        QMessageBox.information(dialog, "成功", f"预设 '{name}' 已保存。")
        
        # 更新下拉框
        self._load_custom_presets()
    
    def _load_selected_preset(self, dialog, preset_list, custom_presets):
        """加载选中的预设"""
        selected_items = preset_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(dialog, "提示", "请先选择一个预设。")
            return
        
        name = selected_items[0].text()
        if name not in custom_presets:
            QMessageBox.warning(dialog, "错误", f"预设 '{name}' 不存在。")
            return
        
        # 应用预设
        self._apply_style_params(custom_presets[name])
        
        # 更新下拉框选择
        self.style_preset_combo.blockSignals(True)
        self.style_preset_combo.setCurrentText(name)
        self.style_preset_combo.blockSignals(False)
        
        # 触发自动更新
        self._auto_update_all_plots()
        
        QMessageBox.information(dialog, "成功", f"预设 '{name}' 已加载。")
    
    def _delete_selected_preset(self, dialog, preset_list, custom_presets):
        """删除选中的预设"""
        selected_items = preset_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(dialog, "提示", "请先选择一个预设。")
            return
        
        name = selected_items[0].text()
        
        reply = QMessageBox.question(dialog, "确认删除", f"确定要删除预设 '{name}' 吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 删除预设
        del custom_presets[name]
        
        # 保存到设置
        self.settings.setValue("custom_style_presets", json.dumps(custom_presets))
        self.settings.sync()
        
        # 更新列表
        preset_list.takeItem(preset_list.row(selected_items[0]))
        
        # 更新下拉框
        self._load_custom_presets()
        
        QMessageBox.information(dialog, "成功", f"预设 '{name}' 已删除。")
    
    def _on_style_param_changed(self, force_data_reload=False):
        """
        样式参数变化时的回调函数（防抖 + 自动更新）
        
        Args:
            force_data_reload: 如果为True，强制重新加载数据（用于预处理参数改变时）
        """
        # 检查是否启用自动更新
        if hasattr(self, 'auto_update_check') and not self.auto_update_check.isChecked():
            return  # 如果未启用自动更新，直接返回
        
        # 如果预处理参数改变，需要重新加载数据
        if force_data_reload:
            # 清空相关缓存
            if hasattr(self, 'plot_data_cache'):
                self.plot_data_cache.clear_preprocess_cache()
            # 使用完整更新流程
            if not hasattr(self, '_style_update_timer'):
                from PyQt6.QtCore import QTimer
                self._style_update_timer = QTimer()
                self._style_update_timer.setSingleShot(True)
                self._style_update_timer.timeout.connect(self._auto_update_all_plots)
            self._style_update_timer.start(300)
            return
        
        # 样式参数改变：只更新样式，不重新读取数据
        # 重置定时器，100ms后执行更新（更快响应，防抖）
        if hasattr(self, '_style_update_timer'):
            # 确保连接到样式更新方法（不重新加载数据）
            try:
                self._style_update_timer.timeout.disconnect()
            except:
                pass
            self._style_update_timer.timeout.connect(self._auto_update_current_plot)
            self._style_update_timer.stop()
            self._style_update_timer.start(100)  # 从300ms减少到100ms，更快响应
        else:
            # 如果定时器不存在，创建它
            from PyQt6.QtCore import QTimer
            self._style_update_timer = QTimer()
            self._style_update_timer.setSingleShot(True)
            self._style_update_timer.timeout.connect(self._auto_update_current_plot)
            self._style_update_timer.start(100)  # 从300ms减少到100ms，更快响应
    
    def _on_file_color_changed(self):
        """文件颜色改变时的回调函数（自动更新图表）"""
        # 颜色改变时立即更新所有打开的绘图窗口
        self._on_style_param_changed()
    
    def _on_scan_spectra_requested(self):
        """处理谱线扫描请求"""
        # 获取最后一次绘图的数据
        if self.active_plot_window and hasattr(self.active_plot_window, 'current_plot_data'):
            plot_data = []
            for key, data in self.active_plot_window.current_plot_data.items():
                if 'x' in data and 'y' in data:
                    plot_data.append({
                        'x': data['x'],
                        'y': data['y'],
                        'label': data.get('label', key),
                        'color': data.get('color', 'blue')
                    })
            if plot_data:
                self.spectrum_scan_panel.scan_last_plot(plot_data)
            else:
                QMessageBox.warning(self, "警告", "当前绘图窗口没有可扫描的数据")
        else:
            # 尝试从所有绘图窗口获取数据
            found_data = False
            for window_name, window in self.plot_windows.items():
                if window and window.isVisible() and hasattr(window, 'current_plot_data'):
                    plot_data = []
                    for key, data in window.current_plot_data.items():
                        if 'x' in data and 'y' in data:
                            plot_data.append({
                                'x': data['x'],
                                'y': data['y'],
                                'label': data.get('label', key),
                                'color': data.get('color', 'blue')
                            })
                    if plot_data:
                        self.spectrum_scan_panel.scan_last_plot(plot_data)
                        found_data = True
                        break
            if not found_data:
                QMessageBox.warning(self, "警告", "请先运行绘图，然后再扫描谱线")
    
    def _auto_update_current_plot(self):
        """自动更新当前活动绘图窗口（仅样式更新，不重新读取数据）"""
        # 防止重复更新
        if hasattr(self, '_is_updating_plots') and self._is_updating_plots:
            return
        
        # 如果启用了自动更新，更新当前活动窗口
        if hasattr(self, 'auto_update_check') and not self.auto_update_check.isChecked():
            return
        
        self._is_updating_plots = True
        try:
            # 更新活动绘图窗口
            if self.active_plot_window and self.active_plot_window.isVisible():
                try:
                    # 尝试从当前窗口获取已有数据
                    grouped_files_data = None
                    control_data_list = None
                    
                    # 尝试从窗口的 current_plot_data 重建 grouped_files_data
                    if hasattr(self.active_plot_window, 'current_plot_data') and self.active_plot_window.current_plot_data:
                        # 从 current_plot_data 重建数据（如果可能）
                        # 但更可靠的方法是重新读取数据
                        pass
                    
                    # 如果无法从窗口获取数据，重新读取（但使用缓存）
                    if grouped_files_data is None:
                        # 使用_prepare_plot_params重新读取数据（会使用缓存）
                        params = self._prepare_plot_params()
                    else:
                        # 使用已有数据
                        params = self._prepare_plot_params(grouped_files_data=grouped_files_data, control_data_list=control_data_list)
                    
                    if params and 'grouped_files_data' in params:
                        # 更新绘图（会使用params中的样式参数和数据）
                        self.active_plot_window.update_plot(params)
                    else:
                        # 如果无法准备参数，重新运行完整绘图逻辑
                        self.run_plot_logic()
                except Exception as e:
                    print(f"自动更新当前绘图窗口失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 如果自动更新失败，尝试重新运行完整绘图逻辑
                    try:
                        self.run_plot_logic()
                    except:
                        pass
            else:
                # 如果没有活动窗口，尝试更新所有可见窗口
                self._auto_update_all_plots()
        finally:
            self._is_updating_plots = False
    
    def _auto_update_all_plots(self):
        """自动更新所有打开的绘图窗口（包括预处理参数改变时更新RRUFF库）"""
        # 防止重复更新：使用标志位
        if hasattr(self, '_is_updating_plots') and self._is_updating_plots:
            return
        self._is_updating_plots = True
        
        try:
            # 更新批量绘图窗口（包括RRUFF库预处理参数和绘图）
            # 注意：这里只更新批量绘图窗口，不更新主窗口的RRUFF库（避免重复处理）
            if hasattr(self, 'batch_plot_window') and self.batch_plot_window and self.batch_plot_window.isVisible():
                try:
                    # 更新RRUFF库预处理参数（这个方法会检查参数是否改变，只在改变时才重新处理）
                    self.batch_plot_window.update_rruff_preprocessing()
                    # update_rruff_preprocessing 内部已经会调用 _update_plots_with_rruff，不需要重复调用
                except Exception as e:
                    print(f"更新批量绘图窗口失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 更新所有主绘图窗口（只更新一次，避免重复）
            updated = False
            for group_name, plot_window in self.plot_windows.items():
                if plot_window and plot_window.isVisible():
                    try:
                        # 重新运行绘图逻辑（会使用当前参数）
                        if not updated:
                            self.run_plot_logic()
                            updated = True
                        break  # 只更新一次，因为run_plot_logic会更新所有窗口
                    except Exception as e:
                        print(f"自动更新绘图窗口 {group_name} 失败: {e}")
        finally:
            self._is_updating_plots = False
        
        # 更新组瀑布图窗口（如果存在）
        if "GroupComparison" in self.plot_windows:
            group_comparison_window = self.plot_windows["GroupComparison"]
            if group_comparison_window and group_comparison_window.isVisible():
                try:
                    # 重新运行组瀑布图逻辑（会使用当前参数，包括颜色和位移）
                    self.run_group_average_waterfall()
                except Exception as e:
                    print(f"自动更新组瀑布图窗口失败: {e}")
        
        # 更新NMF窗口（如果存在）
        if hasattr(self, 'nmf_window') and self.nmf_window is not None and self.nmf_window.isVisible():
            try:
                # 重新获取样式参数并更新NMF窗口
                style_params = self._get_current_style_params()
                # 合并NMF特定参数
                nmf_style_params = {
                    **style_params,
                    'comp1_color': getattr(self, 'comp1_color', '#034DFB'),
                    'comp2_color': getattr(self, 'comp2_color', '#FF0000'),
                    'comp_line_width': style_params.get('line_width', 1.2),
                    'comp_line_style': style_params.get('line_style', '-'),
                    'is_derivative': False,  # 二次导数在预处理流程中应用
                    'global_stack_offset': self._get_stack_offset_from_panel(),
                    'global_scale_factor': self.global_y_scale_factor_spin.value() if hasattr(self, 'global_y_scale_factor_spin') else 1.0,
                    'x_axis_invert': style_params.get('x_axis_invert', False),
                    'tick_font_size': style_params.get('tick_label_font', 16),
                    'label_font_size': style_params.get('axis_title_font', 20),
                    'title_font_size': style_params.get('title_fontsize', 20),
                    'legend_font_size': style_params.get('legend_fontsize', 10),
                    'weight_marker_style': 'o',
                    'weight_marker_size': 5,
                    'weight_line_style': '-',
                    'weight_line_width': 1.0,
                }
                # 更新NMF窗口的样式参数并重新绘制
                if hasattr(self.nmf_window, 'style_params'):
                    self.nmf_window.style_params.update(nmf_style_params)
                    self.nmf_window.plot_results(self.nmf_window.style_params)
            except Exception as e:
                print(f"自动更新NMF窗口失败: {e}")
                import traceback
                traceback.print_exc()

    # --- 核心：数据读取 (新增物理截断 + 多段截断) ---
    def _parse_segment_ranges(self, text):
        """
        解析多段截断字符串，格式如："600-800, 1000-1200"。
        返回 [(600, 800), (1000, 1200)] 这样的列表。
        """
        segments = []
        text = text.strip()
        if not text:
            return segments
        for part in text.split(','):
            part = part.strip()
            if not part:
                continue
            try:
                min_s, max_s = part.split('-', 1)
                min_v = float(min_s.strip())
                max_v = float(max_s.strip())
                if min_v > max_v:
                    min_v, max_v = max_v, min_v
                segments.append((min_v, max_v))
            except Exception:
                raise ValueError(f"多段截断格式错误: '{part}'，应为 '起始-终止' 例如 600-800")
        return segments

    def _apply_segment_ranges(self, x, y, segments):
        """在 DataController 物理截断之后，再按多段截断进一步裁剪。"""
        if not segments:
            return x, y
        import numpy as np
        mask = np.zeros_like(x, dtype=bool)
        for min_v, max_v in segments:
            mask |= (x >= min_v) & (x <= max_v)
        if not np.any(mask):
            raise ValueError("多段截断范围内无数据，请检查输入。")
        return x[mask], y[mask]

    def read_data(self, file_path, skip_rows, x_min_phys=None, x_max_phys=None):
        """委托 DataController 读取光谱数据，并根据 UI 进行多段截断。"""
        import numpy as np
        # 检查文件缓存
        if hasattr(self, 'plot_data_cache'):
            cached_data = self.plot_data_cache.get_file_data(file_path)
            if cached_data is not None:
                x, y = cached_data
                # 应用物理截断（如果指定）
                if x_min_phys is not None or x_max_phys is not None:
                    mask = np.ones(len(x), dtype=bool)
                    if x_min_phys is not None:
                        mask &= (x >= x_min_phys)
                    if x_max_phys is not None:
                        mask &= (x <= x_max_phys)
                    x = x[mask]
                    y = y[mask]
                # 应用多段截断
                if hasattr(self, "x_segments_input") and self.x_segments_input is not None:
                    text = self.x_segments_input.text().strip()
                    if text:
                        segments = self._parse_segment_ranges(text)
                        x, y = self._apply_segment_ranges(x, y, segments)
                return x, y
        
        # 如果skip_rows为-1，使用缓存的检测结果或自动检测
        if skip_rows == -1:
            # 优先使用缓存的检测结果
            if hasattr(self, 'skip_rows_detection_results') and file_path in self.skip_rows_detection_results:
                skip_rows = self.skip_rows_detection_results[file_path]['skip_rows']
            else:
                # 如果缓存中没有，尝试快速检测（只检测前20行）
                try:
                    from src.utils.skip_rows_detector import SkipRowsDetector
                    skip_rows = SkipRowsDetector.detect_skip_rows(file_path, max_check_lines=10)  # 减少检测行数
                    # 缓存结果
                    if not hasattr(self, 'skip_rows_detection_results'):
                        self.skip_rows_detection_results = {}
                    try:
                        self.skip_rows_detection_results[file_path] = {
                            'skip_rows': skip_rows,
                            'mtime': os.path.getmtime(file_path)
                        }
                    except:
                        self.skip_rows_detection_results[file_path] = {'skip_rows': skip_rows, 'mtime': 0}
                except:
                    # 如果检测失败，使用默认值0
                    skip_rows = 0
        
        x, y = self.data_controller.read_data(file_path, skip_rows, x_min_phys, x_max_phys)
        
        # 如果存在多段截断输入，则进一步裁剪
        if hasattr(self, "x_segments_input") and self.x_segments_input is not None:
            text = self.x_segments_input.text().strip()
            if text:
                segments = self._parse_segment_ranges(text)
                x, y = self._apply_segment_ranges(x, y, segments)
        
        # 缓存最终数据（应用所有截断后）- 使用numpy数组
        if hasattr(self, 'plot_data_cache'):
            import numpy as np
            # 确保是numpy数组
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            self.plot_data_cache.cache_file_data(file_path, (x.copy(), y.copy()))
        
        return x, y

    def parse_region_weights(self, weights_str, wavenumbers):
        """
        解析区域权重字符串并生成权重向量
        
        Args:
            weights_str: 权重字符串，格式如 "800-1000:0.1, 1000-1200:1.0"
            wavenumbers: 波数数组
        
        Returns:
            weight_vector: 权重向量，长度与 wavenumbers 相同
        """
        return self.data_controller.parse_region_weights(weights_str, wavenumbers)

    def load_and_average_data(self, file_list, n_chars, skip_rows, x_min_phys=None, x_max_phys=None):
        """
        加载并平均数据：将重复样本（如 sample-1, sample-2）分组并计算平均光谱
        
        Args:
            file_list: 文件路径列表
            n_chars: 用于分组的文件名前缀字符数
            skip_rows: 跳过的行数
            x_min_phys: X轴最小值（物理截断）
            x_max_phys: X轴最大值（物理截断）
        
        Returns:
            averaged_data: 字典，键为组名，值为 {'x': x_array, 'y': y_averaged, 'label': group_name, 'files': file_list}
            common_x: 公共的X轴（波数轴）
        """
        return self.data_controller.load_and_average_data(file_list, n_chars, skip_rows, x_min_phys, x_max_phys)

    # --- GUI 布局 ---
    def setup_ui(self):
        # --- 顶部全局控制 (文件 & 数据读取) ---
        top_bar = QFrame()
        top_bar.setFrameShape(QFrame.Shape.Panel)
        top_bar.setFrameShadow(QFrame.Shadow.Raised)
        top_bar_layout = QHBoxLayout(top_bar)
        
        # A. 文件夹选择
        folder_group = QGroupBox("数据文件夹")
        h_file = QHBoxLayout(folder_group)
        self.folder_input = QLineEdit()
        self.folder_input.setMinimumWidth(300)  # 设置最小宽度以显示完整路径
        self.btn_browse = QPushButton("...")
        self.btn_browse.setFixedWidth(40)
        self.btn_browse.clicked.connect(self.browse_folder)
        h_file.addWidget(self.folder_input)
        h_file.addWidget(self.btn_browse)
        
        # B. 文件分组配置
        group_group = QGroupBox("文件分组")
        group_layout = QFormLayout(group_group)
        group_layout.setSpacing(5)
        
        # 分组前缀长度
        self.n_chars_spin = QSpinBox()
        self.n_chars_spin.setRange(-999999999, 999999999)
        self.n_chars_spin.setValue(3)
        self.n_chars_spin.setToolTip("分组前缀长度：取文件名前n个字符作为组名（0=使用完整文件名）")
        group_layout.addRow("分组前缀长度:", self.n_chars_spin)
        
        # 指定组别
        self.groups_input = QLineEdit()
        self.groups_input.setPlaceholderText("例如: ant, mpt (留空则全选)")
        self.groups_input.setToolTip("指定要处理的组别，多个组用逗号分隔，留空则处理所有组")
        group_layout.addRow("指定组别 (可选):", self.groups_input)
        
        # 对照文件
        self.control_files_input = QTextEdit()
        self.control_files_input.setFixedHeight(40)
        self.control_files_input.setPlaceholderText("例如: His (自动识别.txt/.csv等后缀，多个文件用逗号或换行分隔)")
        self.control_files_input.setToolTip("对照文件：这些文件会优先绘制，可用于对比分析")
        group_layout.addRow("对照文件 (优先绘制):", self.control_files_input)
        
        # NMF解混平均选项
        self.nmf_average_check = QCheckBox("启用分组平均 (NMF分析时对重复样本求平均)")
        self.nmf_average_check.setChecked(True)  # 默认启用
        self.nmf_average_check.setToolTip("启用后，NMF分析会将相同前缀的文件（如sample-1, sample-2）分组并计算平均光谱，提高信噪比")
        group_layout.addRow(self.nmf_average_check)
        
        # C. 数据读取 / 跳过行数（自动检测，显示检测结果）
        read_group = QGroupBox("数据读取")
        read_layout = QVBoxLayout(read_group)
        read_layout.setSpacing(5)
        
        # 跳过行数控制（紧凑）
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("跳过行数:"))
        self.skip_rows_spin = QSpinBox()
        self.skip_rows_spin.setRange(-1, 999999999)  # -1表示自动检测
        self.skip_rows_spin.setValue(-1)  # 默认自动检测
        self.skip_rows_spin.setSpecialValueText("自动检测")
        self.skip_rows_spin.setMaximumWidth(100)
        self.skip_rows_spin.setToolTip("跳过行数：-1=自动检测，0或正数=手动指定")
        self.skip_rows_spin.valueChanged.connect(self._on_skip_rows_changed)
        skip_layout.addWidget(self.skip_rows_spin)
        skip_layout.addStretch()
        read_layout.addLayout(skip_layout)
        
        # 检测结果显示（前中后）
        self.skip_rows_info_label = QLabel("检测状态: 等待扫描文件...")
        self.skip_rows_info_label.setStyleSheet("font-size: 9pt; color: #666; padding: 3px;")
        self.skip_rows_info_label.setWordWrap(True)
        read_layout.addWidget(self.skip_rows_info_label)
        
        # 检测按钮
        detect_btn = QPushButton("检测跳过行数")
        detect_btn.setMaximumWidth(120)
        detect_btn.setStyleSheet("font-size: 9pt; padding: 4px;")
        detect_btn.clicked.connect(self._detect_skip_rows_for_all_files)
        skip_layout.addWidget(detect_btn)
        
        # 文件夹改变时自动检测
        self.folder_input.textChanged.connect(self._on_folder_changed)
        
        # 初始化跳过行数检测结果
        self.skip_rows_detection_results = {}
        
        # 将控件添加到top_bar_layout
        top_bar_layout.addWidget(folder_group)
        top_bar_layout.addWidget(group_group)
        top_bar_layout.addWidget(read_group)
        
        # 将top_bar添加到main_layout
        self.main_layout.addWidget(top_bar)
        
        # --- 主按钮区：两列布局（左侧参数配置，右侧运行绘图）---
        buttons_container = QFrame()
        buttons_container.setFrameShape(QFrame.Shape.Panel)
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setSpacing(20)
        buttons_layout.setContentsMargins(20, 20, 20, 20)
        
        # 初始化功能标签页内容（用于独立窗口）
        try:
            self._init_function_tabs()
        except Exception as e:
            import traceback
            print(f"警告: 初始化功能标签页失败: {e}")
            traceback.print_exc()
            self.plotting_tab_content = None
            self.file_tab_content = None
            self.peak_tab_content = None
            self.nmf_tab_content = None
            self.physics_tab_content = None
        
        # 第一列：参数配置按钮区（包含样式配置）
        left_buttons_group = QGroupBox("参数配置")
        left_buttons_layout = QVBoxLayout(left_buttons_group)
        left_buttons_layout.setSpacing(10)
        
        # 参数配置按钮
        self.btn_plotting = QPushButton("📊 绘图与预处理")
        self.btn_plotting.setStyleSheet("font-size: 12pt; padding: 12px; text-align: left;")
        self.btn_plotting.clicked.connect(lambda: self.open_function_window('plotting'))
        left_buttons_layout.addWidget(self.btn_plotting)
        
        self.btn_file = QPushButton("📁 文件扫描与独立Y轴")
        self.btn_file.setStyleSheet("font-size: 12pt; padding: 12px; text-align: left;")
        self.btn_file.clicked.connect(lambda: self.open_function_window('file'))
        left_buttons_layout.addWidget(self.btn_file)
        
        self.btn_peak = QPushButton("📈 波峰检测")
        self.btn_peak.setStyleSheet("font-size: 12pt; padding: 12px; text-align: left;")
        self.btn_peak.clicked.connect(lambda: self.open_function_window('peak'))
        left_buttons_layout.addWidget(self.btn_peak)
        
        self.btn_nmf = QPushButton("🔬 NMF分析")
        self.btn_nmf.setStyleSheet("font-size: 12pt; padding: 12px; text-align: left;")
        self.btn_nmf.clicked.connect(lambda: self.open_function_window('nmf'))
        left_buttons_layout.addWidget(self.btn_nmf)
        
        self.btn_physics = QPushButton("⚛️ 物理验证")
        self.btn_physics.setStyleSheet("font-size: 12pt; padding: 12px; text-align: left;")
        self.btn_physics.clicked.connect(lambda: self.open_function_window('physics'))
        left_buttons_layout.addWidget(self.btn_physics)
        
        # 注意：多子图配置已删除，功能已整合到样式与匹配窗口中
        
        self.btn_style_matching = QPushButton("🎨 样式与匹配")
        self.btn_style_matching.setStyleSheet("font-size: 12pt; padding: 12px; text-align: left;")
        self.btn_style_matching.clicked.connect(self.open_style_matching_window)
        left_buttons_layout.addWidget(self.btn_style_matching)
        
        left_buttons_layout.addStretch()
        buttons_layout.addWidget(left_buttons_group)
        
        # 第二列：运行绘图按钮区
        right_buttons_group = QGroupBox("运行绘图")
        right_buttons_layout = QVBoxLayout(right_buttons_group)
        right_buttons_layout.setSpacing(10)
        
        # 主要运行按钮
        self.run_button = QPushButton("▶️ 运行绘图 (Plot Group Spectra)")
        self.run_button.setStyleSheet("font-size: 14pt; padding: 12px; background-color: #4CAF50; color: white; font-weight: bold;")
        self.run_button.clicked.connect(self.run_plot_logic)
        right_buttons_layout.addWidget(self.run_button)
        
        self.btn_run_nmf = QPushButton("▶️ 运行 NMF 解混分析")
        self.btn_run_nmf.setStyleSheet("font-size: 14pt; padding: 12px; background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_run_nmf.clicked.connect(self.run_nmf_button_handler)
        right_buttons_layout.addWidget(self.btn_run_nmf)
        
        self.btn_rerun_nmf_plot = QPushButton("🔄 重新绘制 NMF 图")
        self.btn_rerun_nmf_plot.setStyleSheet("font-size: 12pt; padding: 10px; background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_rerun_nmf_plot.clicked.connect(self._rerun_nmf_plot_handler)
        self.btn_rerun_nmf_plot.setToolTip("使用当前设置重新绘制NMF图，不重新运行NMF分析")
        right_buttons_layout.addWidget(self.btn_rerun_nmf_plot)
        
        # 其他功能按钮
        self.btn_batch_plot = QPushButton("📸 光谱+镜下图绘制")
        self.btn_batch_plot.setStyleSheet("font-size: 12pt; padding: 10px;")
        self.btn_batch_plot.clicked.connect(self.open_batch_plot_window)
        right_buttons_layout.addWidget(self.btn_batch_plot)
        
        self.btn_quantitative = QPushButton("📊 定量校准分析")
        self.btn_quantitative.setStyleSheet("font-size: 12pt; padding: 10px; background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_quantitative.clicked.connect(self.open_quantitative_dialog)
        right_buttons_layout.addWidget(self.btn_quantitative)
        
        self.btn_compare = QPushButton("📉 绘制组间平均对比 (瀑布图)")
        self.btn_compare.setStyleSheet("font-size: 12pt; padding: 10px; background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_compare.clicked.connect(self.run_group_average_waterfall)
        right_buttons_layout.addWidget(self.btn_compare)
        
        self.btn_2dcos = QPushButton("📐 运行 2D-COS (组梯度分析)")
        self.btn_2dcos.setStyleSheet("font-size: 12pt; padding: 10px; background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_2dcos.clicked.connect(self.run_2d_cos_analysis)
        self.btn_2dcos.setToolTip("2D-COS分析：基于浓度梯度数据解析重叠峰（如1100 vs 1107 cm⁻¹）")
        right_buttons_layout.addWidget(self.btn_2dcos)
        
        self.btn_export = QPushButton("💾 导出预处理后数据")
        self.btn_export.setStyleSheet("font-size: 12pt; padding: 10px;")
        self.btn_export.clicked.connect(self.export_processed_data)
        right_buttons_layout.addWidget(self.btn_export)
        
        right_buttons_layout.addStretch()
        buttons_layout.addWidget(right_buttons_group)
        
        # 将按钮容器添加到主布局
        self.main_layout.addWidget(buttons_container)
        
        # 设置主布局
        self.setLayout(self.main_layout)
    
    def _on_folder_changed(self):
        """文件夹改变时自动检测跳过行数（优化版：延迟更长，避免频繁触发）"""
        folder = self.folder_input.text()
        if not folder or not os.path.isdir(folder):
            self.skip_rows_info_label.setText("检测状态: 文件夹无效")
            # 清空检测结果缓存（文件夹改变时）
            if hasattr(self, 'skip_rows_detection_results'):
                self.skip_rows_detection_results.clear()
            # 清空文件缓存（文件夹改变时）
            if hasattr(self, 'plot_data_cache'):
                self.plot_data_cache.clear_file_cache()
            return
        
        # 延迟检测（避免频繁检测）- 增加到1秒，减少卡顿
        if not hasattr(self, '_skip_rows_detection_timer'):
            from PyQt6.QtCore import QTimer
            self._skip_rows_detection_timer = QTimer()
            self._skip_rows_detection_timer.setSingleShot(True)
            self._skip_rows_detection_timer.timeout.connect(self._detect_skip_rows_for_all_files)
        
        # 停止之前的定时器，重新开始（防抖）
        self._skip_rows_detection_timer.stop()
        self._skip_rows_detection_timer.start(1500)  # 1.5秒后检测，减少卡顿
    
    def _detect_skip_rows_for_all_files(self):
        """检测所有文件的跳过行数（优化版：使用缓存，减少检测次数）"""
        folder = self.folder_input.text()
        if not folder or not os.path.isdir(folder):
            return
        
        try:
            from src.utils.skip_rows_detector import SkipRowsDetector
            
            # 获取所有CSV和TXT文件（os已在文件顶部import）
            csv_files = glob.glob(os.path.join(folder, '*.csv'))
            txt_files = glob.glob(os.path.join(folder, '*.txt'))
            all_files = csv_files + txt_files
            
            if not all_files:
                self.skip_rows_info_label.setText("检测状态: 未找到数据文件")
                return
            
            # 优化：只检测前3个文件（减少检测时间）
            sample_files = all_files[:3]
            
            # 检查缓存：只检测修改时间改变的文件
            files_to_detect = []
            if not hasattr(self, 'skip_rows_detection_results'):
                self.skip_rows_detection_results = {}
            
            for file_path in sample_files:
                if file_path not in self.skip_rows_detection_results:
                    files_to_detect.append(file_path)
                else:
                    # 检查文件是否改变
                    try:
                        cached_mtime = self.skip_rows_detection_results[file_path].get('mtime', 0)
                        current_mtime = os.path.getmtime(file_path)
                        if current_mtime != cached_mtime:
                            files_to_detect.append(file_path)
                    except:
                        files_to_detect.append(file_path)
            
            # 只检测需要检测的文件
            if files_to_detect:
                new_results = SkipRowsDetector.detect_multiple_files(files_to_detect)
                # 更新缓存（os已在文件顶部import）
                for file_path, info in new_results.items():
                    try:
                        info['mtime'] = os.path.getmtime(file_path)
                    except:
                        info['mtime'] = 0
                    self.skip_rows_detection_results[file_path] = info
                results = self.skip_rows_detection_results
            else:
                # 使用缓存结果
                results = {k: v for k, v in self.skip_rows_detection_results.items() if k in sample_files}
            
            if not results:
                self.skip_rows_info_label.setText("检测状态: 检测失败")
                return
            
            # 统计跳过行数
            skip_rows_list = [info['skip_rows'] for info in results.values()]
            most_common_skip = max(set(skip_rows_list), key=skip_rows_list.count) if skip_rows_list else 0
            
            # 获取第一个文件的预览
            first_file = list(results.keys())[0]
            first_info = results[first_file]
            
            # 格式化显示信息
            preview_text = first_info.get('preview', '')[:50].replace('\n', ' ') if first_info.get('preview') else "N/A"
            middle_text = first_info.get('middle', '')[:50].replace('\n', ' ') if first_info.get('middle') else "N/A"
            end_text = first_info.get('end', '')[:50].replace('\n', ' ') if first_info.get('end') else "N/A"
            
            detected_count = len(files_to_detect) if files_to_detect else 0
            cached_count = len(sample_files) - detected_count
            
            info_text = (
                f"检测结果: 跳过 {most_common_skip} 行 (检测 {detected_count} 个文件，使用缓存 {cached_count} 个)\n"
                f"前: {preview_text}...\n"
                f"中: {middle_text}...\n"
                f"后: {end_text}..."
            )
            
            self.skip_rows_info_label.setText(info_text)
            # 注意：results已经是self.skip_rows_detection_results的引用，不需要重新赋值
            
            # 如果当前是自动检测模式，更新跳过行数
            if self.skip_rows_spin.value() == -1:
                # 不改变值，但更新显示
                pass
                
        except Exception as e:
            self.skip_rows_info_label.setText(f"检测状态: 检测失败 - {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_skip_rows_changed(self, value):
        """跳过行数改变时更新显示"""
        if value == -1:
            # 自动检测模式，显示检测结果
            if self.skip_rows_detection_results:
                self._detect_skip_rows_for_all_files()
            else:
                self.skip_rows_info_label.setText("检测状态: 自动检测模式（等待扫描文件）")
        else:
            self.skip_rows_info_label.setText(f"检测状态: 手动指定跳过 {value} 行")


    def _init_function_tabs(self):
        """初始化功能标签页内容（用于独立窗口）"""
        # 创建各个功能标签页的内容，但不添加到标签页中
        self.plotting_tab_content = self._create_plotting_tab_content()
        self.file_tab_content = self._create_file_tab_content()
        self.peak_tab_content = self._create_peak_tab_content()
        self.nmf_tab_content = self._create_nmf_tab_content()
        self.physics_tab_content = self._create_physics_tab_content()
    
    def open_function_window(self, function_name):
        """打开功能配置窗口"""
        if function_name not in self.function_windows:
            if function_name == 'plotting':
                window = FunctionWindow("绘图与预处理", self)
                window.add_content(self.plotting_tab_content)
            elif function_name == 'file':
                window = FunctionWindow("文件扫描与独立Y轴", self)
                window.add_content(self.file_tab_content)
            elif function_name == 'peak':
                window = FunctionWindow("波峰检测", self)
                window.add_content(self.peak_tab_content)
            elif function_name == 'nmf':
                window = FunctionWindow("NMF 分析", self)
                window.add_content(self.nmf_tab_content)
            elif function_name == 'physics':
                window = FunctionWindow("物理验证", self)
                window.add_content(self.physics_tab_content)
            else:
                return
            
            self.function_windows[function_name] = window
        
        # 获取窗口
        window = self.function_windows[function_name]
        
        # 如果窗口最小化了，先还原
        if window.isMinimized():
            window.showNormal()
        
        # 显示窗口
        window.show()
        window.raise_()
        window.activateWindow()
    
    def open_style_matching_window(self):
        """打开样式与匹配窗口"""
        if not hasattr(self, '_style_matching_window') or self._style_matching_window is None:
            self._style_matching_window = StyleMatchingWindow(self)
            # 保存面板引用，以便主窗口访问
            self.publication_style_panel = self._style_matching_window.get_publication_style_panel()
            self.peak_matching_panel = self._style_matching_window.get_peak_matching_panel()
            self.spectrum_scan_panel = self._style_matching_window.get_spectrum_scan_panel()
            
            # 连接谱线扫描信号
            if hasattr(self.spectrum_scan_panel, 'scan_requested'):
                self.spectrum_scan_panel.scan_requested.connect(self._on_scan_spectra_requested)
        
        window = self._style_matching_window
        
        # 如果窗口最小化了，先还原
        if window.isMinimized():
            window.showNormal()
        
        # 显示窗口
        window.show()
        window.raise_()
        window.activateWindow()
    
    # 注意：多子图配置窗口已删除，功能已整合到样式与匹配窗口中
    def _on_multi_subplot_config_changed(self, window_id, subplot_index):
        """多子图配置改变时自动更新绘图（已废弃，保留以兼容旧代码）"""
        # 自动更新对应的窗口
        if hasattr(self, 'nmf_window') and self.nmf_window and self.nmf_window.isVisible():
            if hasattr(self.nmf_window, 'style_params') and self.nmf_window.style_params:
                self.nmf_window.plot_results(self.nmf_window.style_params)
    
    def _create_plotting_tab_content(self):
        """创建绘图与预处理标签页内容"""
        # 直接复用原有的 tab 构建函数，返回其内部的 widget
        return self.setup_plotting_tab()
    
    def _create_file_tab_content(self):
        """创建文件扫描与独立Y轴标签页内容"""
        return self.setup_file_controls_tab()
    
    def _create_peak_tab_content(self):
        """创建波峰检测标签页内容"""
        return self.setup_peak_detection_tab()
    
    def _create_nmf_tab_content(self):
        """创建NMF分析标签页内容"""
        # NMF 页由 NMFPanelMixin 构建
        return self.setup_nmf_tab()
    
    def _create_physics_tab_content(self):
        """创建物理验证标签页内容"""
        return self.setup_physics_tab()
    
    # --- Tab 1: 绘图设置 ---
    def setup_plotting_tab(self):
        tab1 = QWidget()
        grid_layout = QGridLayout(tab1)
        grid_layout.setSpacing(10)

        # --- 1. 左侧：X轴截断 + 预处理 ---
        left_vbox = QVBoxLayout()

        # 1.1 X 轴截断（物理 + 多段）
        x_trunc_group = CollapsibleGroupBox("1. X 轴截断", is_expanded=True)
        x_trunc_layout = QFormLayout()

        # 物理 Min / Max 截断（从主菜单迁移到这里）
        self.x_min_phys_input = QLineEdit()
        self.x_min_phys_input.setPlaceholderText("例如: 600")
        self.x_max_phys_input = QLineEdit()
        self.x_max_phys_input.setPlaceholderText("例如: 4000")
        x_trunc_layout.addRow("物理截断 Min:", self.x_min_phys_input)
        x_trunc_layout.addRow("物理截断 Max:", self.x_max_phys_input)

        # 多段截断：如 600-800, 1000-1200
        self.x_segments_input = QLineEdit()
        self.x_segments_input.setPlaceholderText("多段截断: 例如 600-800, 1000-1200（留空则只用 Min/Max 或全范围）")
        x_trunc_layout.addRow("多段截断 (可选):", self.x_segments_input)

        x_trunc_group.setContentLayout(x_trunc_layout)
        left_vbox.addWidget(x_trunc_group)
        
        # 1.2 数据预处理（文件及分组配置已移到主菜单）
        preprocess_group = CollapsibleGroupBox("2. 数据预处理 (AsLS / QC / BE / SNV)", is_expanded=True)
        prep_layout = QFormLayout()
        
        # 跳过行数已移至主菜单，这里不再显示
        
        self.qc_check = QCheckBox("启用 QC (剔除弱信号)")
        
        # 改为无限长度的数字输入，避免小数位/位数被限制
        self.qc_threshold_spin = UnlimitedNumericInput(default_value="5.0")
        
        prep_layout.addRow(self._create_h_layout([self.qc_check, QLabel("阈值:"), self.qc_threshold_spin]))
        
        # --- Bose-Einstein 修正：整合到预处理 ---
        self.be_check = QCheckBox("启用 Bose-Einstein 校正")
        self.be_temp_spin = UnlimitedNumericInput(default_value="300.0")
        prep_layout.addRow(self.be_check)
        prep_layout.addRow("BE 温度 T (K):", self.be_temp_spin)
        # ----------------------------------------
        
        self.baseline_als_check = QCheckBox("启用 AsLS 基线校正 (推荐)")
        
        self.lam_spin = UnlimitedNumericInput(default_value="10000")
        
        self.p_spin = UnlimitedNumericInput(default_value="0.005")
        
        prep_layout.addRow(self.baseline_als_check)
        prep_layout.addRow("Lambda (平滑度):", self.lam_spin)
        prep_layout.addRow("P (非对称度):", self.p_spin)

        # 多点多项式基线校正（兼容旧版）
        self.baseline_poly_check = QCheckBox("启用多项式基线 (备选)")
        self.baseline_points_spin = QSpinBox()
        self.baseline_points_spin.setRange(1, 1000000)
        self.baseline_points_spin.setValue(50)
        self.baseline_poly_spin = QSpinBox()
        self.baseline_poly_spin.setRange(1, 10)
        self.baseline_poly_spin.setValue(3)
        prep_layout.addRow(self.baseline_poly_check)
        prep_layout.addRow("采样点 / 多项式阶数:", self._create_h_layout([
            self.baseline_points_spin, QLabel("阶数:"), self.baseline_poly_spin
        ]))
        
        self.smoothing_check = QCheckBox("启用 SG 平滑")
        
        self.smoothing_window_spin = QSpinBox()
        self.smoothing_window_spin.setRange(-999999999, 999999999)
        self.smoothing_window_spin.setValue(15)
        
        self.smoothing_poly_spin = QSpinBox()
        self.smoothing_poly_spin.setRange(-999999999, 999999999)
        self.smoothing_poly_spin.setValue(3)
        
        prep_layout.addRow(self.smoothing_check)
        prep_layout.addRow("窗口 / 阶数:", self._create_h_layout([self.smoothing_window_spin, QLabel("阶数:"), self.smoothing_poly_spin]))
        
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(['None', 'snv', 'max', 'area'])
        prep_layout.addRow("归一化模式:", self.normalization_combo)
        
        # 注意：SVD 去噪已移至NMF分析配置中，不再在全局预处理中使用
        
        # 全局动态范围压缩预处理
        self.global_transform_combo = QComboBox()
        self.global_transform_combo.addItems(['无', '对数变换 (Log)', '平方根变换 (Sqrt)'])
        self.global_transform_combo.setCurrentText('无')
        
        self.global_log_base_combo = QComboBox()
        self.global_log_base_combo.addItems(['10', 'e'])
        self.global_log_base_combo.setCurrentText('10')
        
        self.global_log_offset_spin = UnlimitedNumericInput(default_value="1.0")
        
        self.global_sqrt_offset_spin = UnlimitedNumericInput(default_value="0.0")
        
        transform_layout = QVBoxLayout()
        transform_layout.addWidget(QLabel("全局动态范围压缩:"))
        transform_layout.addWidget(self.global_transform_combo)
        
        log_params_layout = QHBoxLayout()
        log_params_layout.addWidget(QLabel("对数底数:"))
        log_params_layout.addWidget(self.global_log_base_combo)
        log_params_layout.addWidget(QLabel("偏移:"))
        log_params_layout.addWidget(self.global_log_offset_spin)
        log_params_widget = QWidget()
        log_params_widget.setLayout(log_params_layout)
        
        sqrt_params_layout = QHBoxLayout()
        sqrt_params_layout.addWidget(QLabel("平方根偏移:"))
        sqrt_params_layout.addWidget(self.global_sqrt_offset_spin)
        sqrt_params_widget = QWidget()
        sqrt_params_widget.setLayout(sqrt_params_layout)
        
        transform_layout.addWidget(log_params_widget)
        transform_layout.addWidget(sqrt_params_widget)
        
        transform_group = QGroupBox()
        transform_group.setLayout(transform_layout)
        prep_layout.addRow(transform_group)
        
        # 注意：二次函数拟合已删除（用户只需要二次导数，已在预处理流程中应用）
        
        preprocess_group.setContentLayout(prep_layout)
        left_vbox.addWidget(preprocess_group)
        
        grid_layout.addLayout(left_vbox, 0, 0, 1, 1) # 左侧布局

        # --- 2. 右侧：绘图样式 (出版质量控制) ---
        right_vbox = QVBoxLayout()
        
        # 2.0 自动更新开关
        auto_update_group = CollapsibleGroupBox("⚙️ 自动更新设置", is_expanded=False)
        auto_update_layout = QFormLayout()
        
        self.auto_update_check = QCheckBox("启用自动更新（参数改变时自动重新绘制当前谱图）")
        self.auto_update_check.setChecked(True)  # 默认启用
        self.auto_update_check.setToolTip("启用后，调整参数时当前谱图会自动重新绘制")
        auto_update_layout.addRow(self.auto_update_check)
        
        auto_update_group.setContentLayout(auto_update_layout)
        right_vbox.addWidget(auto_update_group)
        
        # 2.1 绘图模式与标签
        plot_style_group = CollapsibleGroupBox("📈 4. 绘图模式与全局设置", is_expanded=True)
        style_layout = QFormLayout()
        
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(['Normal Overlay', 'Mean + Shadow'])
        style_layout.addRow("绘图模式:", self.plot_mode_combo)
        
        # 注意：X轴翻转、显示X/Y轴数值已移至"样式与匹配"窗口的出版质量样式面板
        # 注意：二次导数已在预处理流程中应用，不再需要单独的控件
        
        # 整体Y轴偏移（预处理最后一步，在二次导数之后）
        self.global_y_offset_spin = QDoubleSpinBox()
        self.global_y_offset_spin.setRange(-999999999.0, 999999999.0)
        self.global_y_offset_spin.setDecimals(15)
        self.global_y_offset_spin.setValue(0.0)
        self.global_y_offset_spin.setSingleStep(0.1)
        self.global_y_offset_spin.setToolTip("整体Y轴偏移（预处理最后一步，在二次导数之后应用）")
        style_layout.addRow("整体Y轴偏移（预处理）:", self.global_y_offset_spin)
        
        self.plot_style_combo = QComboBox()
        self.plot_style_combo.addItems(['line', 'scatter'])
        style_layout.addRow("绘制风格:", self.plot_style_combo)

        
        # FIX: 修正 QDoubleSpinBox 实例化错误
        self.global_stack_offset_spin = QDoubleSpinBox()
        self.global_stack_offset_spin.setRange(-999999999.0, 999999999.0)
        self.global_stack_offset_spin.setDecimals(15)
        self.global_stack_offset_spin.setValue(0.5)
        
        self.global_y_scale_factor_spin = QDoubleSpinBox()
        self.global_y_scale_factor_spin.setRange(-999999999.0, 999999999.0)
        self.global_y_scale_factor_spin.setDecimals(15)
        self.global_y_scale_factor_spin.setValue(1.0)
        self.global_y_scale_factor_spin.setSingleStep(0.1)
        
        # 注意：堆叠偏移已移至"样式与匹配"窗口的谱线扫描面板
        style_layout.addRow("Y缩放:", self.global_y_scale_factor_spin)
        
        # 注意：X/Y轴标题、主标题控制、出版质量样式、峰值匹配、谱线扫描已移至"样式与匹配"窗口
        # 注意：浓度梯度图相关设置已移至样式配置，统一管理

        plot_style_group.setContentLayout(style_layout)
        right_vbox.addWidget(plot_style_group)
        
        # 注意：出版质量样式控制、峰值匹配、谱线扫描已移至"样式与匹配"窗口
        # 请点击主菜单的"样式与匹配"按钮进行配置
        
        right_vbox.addStretch(1) # 撑开
        grid_layout.addLayout(right_vbox, 0, 1, 1, 1) # 右侧布局
        return tab1
    
    # --- Tab 2: 文件扫描与独立Y轴 ---
    def setup_file_controls_tab(self):
        tab2 = QWidget()
        layout = QVBoxLayout(tab2)
        layout.setSpacing(10)
        
        # 注意：谱线扫描与堆叠偏移面板已移至"样式与匹配"窗口
        # 请点击主菜单的"样式与匹配"按钮进行配置
        
        # 1. 文件扫描与独立Y轴控制（保留原有功能）
        file_controls_group = CollapsibleGroupBox("文件扫描与独立Y轴控制", is_expanded=True)
        file_controls_layout = QVBoxLayout()
        
        self.scan_button = QPushButton("扫描文件并加载调整项")
        self.scan_button.setStyleSheet("font-size: 12pt; padding: 8px; background-color: #4CAF50; color: white; font-weight: bold;")
        self.scan_button.clicked.connect(self.scan_and_load_file_controls)
        file_controls_layout.addWidget(self.scan_button)
        
        self.dynamic_controls_layout = QVBoxLayout()
        self.dynamic_controls_widget = QWidget()
        self.dynamic_controls_widget.setLayout(self.dynamic_controls_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.dynamic_controls_widget)
        scroll.setFixedHeight(400)
        file_controls_layout.addWidget(scroll)
        
        file_controls_group.setContentLayout(file_controls_layout)
        layout.addWidget(file_controls_group)
        
        # 2. NMF组分独立Y轴控制和重命名
        nmf_controls_group = CollapsibleGroupBox("NMF组分独立Y轴控制和图例重命名", is_expanded=True)
        nmf_controls_layout = QVBoxLayout()
        
        nmf_info_label = QLabel("提示：运行NMF分析后，会自动为每个组分创建独立Y轴控制和图例重命名选项。")
        nmf_info_label.setWordWrap(True)
        nmf_controls_layout.addWidget(nmf_info_label)
        
        self.nmf_component_controls_layout = QVBoxLayout()
        self.nmf_component_controls_widget = QWidget()
        self.nmf_component_controls_widget.setLayout(self.nmf_component_controls_layout)
        nmf_scroll = QScrollArea()
        nmf_scroll.setWidgetResizable(True)
        nmf_scroll.setWidget(self.nmf_component_controls_widget)
        nmf_scroll.setFixedHeight(300)
        nmf_controls_layout.addWidget(nmf_scroll)
        
        nmf_controls_group.setContentLayout(nmf_controls_layout)
        layout.addWidget(nmf_controls_group)
        
        # 3. 组瀑布图独立堆叠位移控制
        waterfall_controls_group = CollapsibleGroupBox("组瀑布图独立堆叠位移控制", is_expanded=True)
        waterfall_controls_layout = QVBoxLayout()
        
        waterfall_info_label = QLabel("提示：扫描组后，可以为每组设置独立的堆叠位移值。")
        waterfall_info_label.setWordWrap(True)
        waterfall_controls_layout.addWidget(waterfall_info_label)
        
        # 扫描组按钮
        scan_groups_button = QPushButton("扫描组并加载位移控制")
        scan_groups_button.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #2196F3; color: white; font-weight: bold;")
        scan_groups_button.clicked.connect(self.scan_and_load_group_waterfall_controls)
        waterfall_controls_layout.addWidget(scan_groups_button)
        
        # 导出平均值谱线按钮
        export_avg_button = QPushButton("导出平均值谱线")
        export_avg_button.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #FF9800; color: white; font-weight: bold;")
        export_avg_button.clicked.connect(self.export_group_averages)
        waterfall_controls_layout.addWidget(export_avg_button)
        
        self.group_waterfall_controls_layout = QVBoxLayout()
        self.group_waterfall_controls_widget = QWidget()
        self.group_waterfall_controls_widget.setLayout(self.group_waterfall_controls_layout)
        waterfall_scroll = QScrollArea()
        waterfall_scroll.setWidgetResizable(True)
        waterfall_scroll.setWidget(self.group_waterfall_controls_widget)
        waterfall_scroll.setFixedHeight(300)
        waterfall_controls_layout.addWidget(waterfall_scroll)
        
        waterfall_controls_group.setContentLayout(waterfall_controls_layout)
        layout.addWidget(waterfall_controls_group)
        
        # 4. 合成数据与标准库配置
        aug_lib_group = CollapsibleGroupBox("合成数据与标准库配置", is_expanded=True)
        aug_lib_layout = QFormLayout()
        
        # 数据增强部分
        aug_header = QLabel("数据增强 (Data Augmentation)")
        aug_header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        aug_lib_layout.addRow(aug_header)
        
        # 纯组分文件夹
        aug_folder_layout = QHBoxLayout()
        self.aug_folder_input = QLineEdit()
        self.aug_folder_input.setPlaceholderText("选择包含纯组分光谱的文件夹")
        self.aug_browse_button = QPushButton("浏览...")
        self.aug_browse_button.clicked.connect(self._browse_aug_folder)
        aug_folder_layout.addWidget(self.aug_folder_input)
        aug_folder_layout.addWidget(self.aug_browse_button)
        aug_lib_layout.addRow("纯组分文件夹:", aug_folder_layout)
        
        # 噪音和基线漂移参数
        self.aug_noise_spin = QDoubleSpinBox()
        self.aug_noise_spin.setRange(-999999999.0, 999999999.0)
        self.aug_noise_spin.setDecimals(15)
        self.aug_noise_spin.setValue(0.01)
        self.aug_noise_spin.setToolTip("高斯噪声水平（相对于最大强度）")
        
        self.aug_drift_spin = QDoubleSpinBox()
        self.aug_drift_spin.setRange(-999999999.0, 999999999.0)
        self.aug_drift_spin.setDecimals(15)
        self.aug_drift_spin.setValue(0.0)
        self.aug_drift_spin.setToolTip("基线漂移幅度")
        
        # 复杂度参数（控制高级增强强度）
        self.aug_complexity_spin = QDoubleSpinBox()
        self.aug_complexity_spin.setRange(-999999999.0, 999999999.0)
        self.aug_complexity_spin.setDecimals(15)
        self.aug_complexity_spin.setValue(0.5)
        self.aug_complexity_spin.setToolTip("复杂度因子（0-1）：控制偏移/拉伸/抑制等高级增强的强度")
        
        # 高级增强开关
        self.aug_advanced_check = QCheckBox("启用高级增强 (偏移/拉伸/峰抑制)")
        self.aug_advanced_check.setChecked(True)
        self.aug_advanced_check.setToolTip("启用后，将应用光谱偏移、拉伸和选择性峰抑制等高级增强技术")
        
        aug_lib_layout.addRow("噪声水平:", self.aug_noise_spin)
        aug_lib_layout.addRow("基线漂移:", self.aug_drift_spin)
        aug_lib_layout.addRow("复杂度因子:", self.aug_complexity_spin)
        aug_lib_layout.addRow(self.aug_advanced_check)
        
        # 生成合成数据按钮
        self.generate_synthetic_button = QPushButton("生成合成数据 (1000条)")
        self.generate_synthetic_button.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #9C27B0; color: white; font-weight: bold;")
        self.generate_synthetic_button.clicked.connect(self._run_data_augmentation)
        aug_lib_layout.addRow(self.generate_synthetic_button)
        
        # 标准库匹配部分
        lib_header = QLabel("标准库匹配 (Library Matching)")
        lib_header.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 10px;")
        aug_lib_layout.addRow(lib_header)
        
        # 标准库文件夹
        lib_folder_layout = QHBoxLayout()
        self.library_folder_input = QLineEdit()
        self.library_folder_input.setPlaceholderText("选择标准库文件夹（RRUFF或有机物标准库）")
        self.library_browse_button = QPushButton("浏览...")
        self.library_browse_button.clicked.connect(self._browse_library_folder)
        lib_folder_layout.addWidget(self.library_folder_input)
        lib_folder_layout.addWidget(self.library_browse_button)
        aug_lib_layout.addRow("标准库文件夹:", lib_folder_layout)
        
        # 加载标准库按钮
        self.load_library_button = QPushButton("加载标准库")
        self.load_library_button.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #2196F3; color: white; font-weight: bold;")
        self.load_library_button.clicked.connect(self._load_library_matcher)
        aug_lib_layout.addRow(self.load_library_button)
        
        # 标准库状态标签
        self.library_status_label = QLabel("状态: 未加载")
        self.library_status_label.setStyleSheet("color: gray; font-size: 9pt;")
        aug_lib_layout.addRow("", self.library_status_label)
        
        # RRUFF匹配部分
        rruff_header = QLabel("RRUFF数据库匹配")
        rruff_header.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 10px;")
        aug_lib_layout.addRow(rruff_header)
        
        # RRUFF匹配容差参数
        self.rruff_match_tolerance_spin = SmartDoubleSpinBox()
        self.rruff_match_tolerance_spin.setRange(0.1, 100.0)
        self.rruff_match_tolerance_spin.setDecimals(1)
        self.rruff_match_tolerance_spin.setValue(5.0)
        self.rruff_match_tolerance_spin.setToolTip("峰值匹配容差（cm⁻¹）：两个峰值位置的距离小于此值时认为匹配。值越大匹配的峰值越多。对于自身匹配，建议设置为较大值（如10-20）以确保100%匹配。默认5.0 cm⁻¹")
        self.rruff_match_tolerance_spin.valueChanged.connect(self._on_rruff_tolerance_changed)
        aug_lib_layout.addRow("匹配容差 (cm⁻¹):", self.rruff_match_tolerance_spin)
        
        # RRUFF匹配按钮
        self.btn_rruff_match = QPushButton("🔍 匹配RRUFF光谱")
        self.btn_rruff_match.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #FF5722; color: white; font-weight: bold;")
        self.btn_rruff_match.clicked.connect(self._match_rruff_spectra)
        self.btn_rruff_match.setEnabled(False)
        aug_lib_layout.addRow("", self.btn_rruff_match)
        
        # RRUFF匹配结果列表
        self.rruff_match_list = QListWidget()
        self.rruff_match_list.setMaximumHeight(200)
        self.rruff_match_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # 支持Ctrl+点击多选
        self.rruff_match_list.itemDoubleClicked.connect(self._on_rruff_item_double_clicked)
        self.rruff_match_list.itemClicked.connect(self._on_rruff_item_clicked)  # 使用itemClicked检测Ctrl键
        self.rruff_match_list.itemSelectionChanged.connect(self._on_rruff_selection_changed)
        aug_lib_layout.addRow("匹配结果 (双击添加，Ctrl+点击叠加):", self.rruff_match_list)
        
        # 清除选中按钮
        self.btn_clear_rruff = QPushButton("清除已选RRUFF光谱")
        self.btn_clear_rruff.setStyleSheet("font-size: 10pt; padding: 4px; background-color: #9E9E9E; color: white;")
        self.btn_clear_rruff.clicked.connect(self._clear_selected_rruff)
        self.btn_clear_rruff.setEnabled(False)
        aug_lib_layout.addRow("", self.btn_clear_rruff)
        
        aug_lib_group.setContentLayout(aug_lib_layout)
        layout.addWidget(aug_lib_group)
        
        layout.addStretch(1)
        return tab2
    
    def _browse_aug_folder(self):
        """浏览纯组分文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择纯组分文件夹")
        if folder:
            self.aug_folder_input.setText(folder)
    
    def _browse_library_folder(self):
        """浏览标准库文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择标准库文件夹")
        if folder:
            self.library_folder_input.setText(folder)
            self.library_folder_path = folder
    
    def _load_library_matcher(self):
        """加载标准库匹配器（同时加载RRUFF库）"""
        folder = self.library_folder_input.text()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "错误", "请先选择有效的标准库文件夹")
            return
        
        try:
            # 加载标准库匹配器
            self.library_matcher = SpectralMatcher(folder)
            n_spectra = len(self.library_matcher.library_spectra)
            
            # 同时加载RRUFF库（使用相同的预处理参数和峰值检测参数）
            preprocess_params = self._get_preprocess_params()
            self.rruff_loader = RRUFFLibraryLoader(folder, preprocess_params)
            
            # 获取峰值检测参数并更新RRUFF库的峰值检测
            peak_detection_params = {
                'peak_height_threshold': self.peak_height_spin.value() if hasattr(self, 'peak_height_spin') else 0.0,
                'peak_distance_min': self.peak_distance_spin.value() if hasattr(self, 'peak_distance_spin') else 10,
                'peak_prominence': self.peak_prominence_spin.value() if hasattr(self, 'peak_prominence_spin') else None,
                'peak_width': self.peak_width_spin.value() if hasattr(self, 'peak_width_spin') else None,
                'peak_wlen': self.peak_wlen_spin.value() if hasattr(self, 'peak_wlen_spin') else None,
                'peak_rel_height': self.peak_rel_height_spin.value() if hasattr(self, 'peak_rel_height_spin') else None,
            }
            # 更新峰值检测参数并重新检测峰值
            for name, spectrum in self.rruff_loader.library_spectra.items():
                if 'y_raw' in spectrum:
                    spectrum['peaks'] = self.rruff_loader._detect_peaks(
                        spectrum['x'], spectrum['y'], 
                        peak_detection_params=peak_detection_params
                    )
            self.rruff_loader.peak_detection_params = peak_detection_params
            
            n_rruff = len(self.rruff_loader.library_spectra)
            
            # 启用RRUFF匹配按钮
            if hasattr(self, 'btn_rruff_match'):
                self.btn_rruff_match.setEnabled(n_rruff > 0)
            
            self.library_status_label.setText(f"状态: 已加载 {n_spectra} 条标准光谱, {n_rruff} 条RRUFF光谱")
            self.library_status_label.setStyleSheet("color: green; font-size: 9pt;")
            # 不再弹出“加载成功”的提示框，仅通过状态标签显示
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载标准库失败：{str(e)}")
            self.library_status_label.setText("状态: 加载失败")
            self.library_status_label.setStyleSheet("color: red; font-size: 9pt;")
            traceback.print_exc()
    
    def _get_preprocess_params(self):
        """获取当前预处理参数"""
        return {
            'qc_enabled': self.qc_check.isChecked(),
            'qc_threshold': self.qc_threshold_spin.value(),
            'is_be_correction': self.be_check.isChecked(),
            'be_temp': self.be_temp_spin.value(),
            'is_smoothing': self.smoothing_check.isChecked(),
            'smoothing_window': self.smoothing_window_spin.value(),
            'smoothing_poly': self.smoothing_poly_spin.value(),
            'is_baseline_als': self.baseline_als_check.isChecked(),
            'als_lam': self.lam_spin.value(),
            'als_p': self.p_spin.value(),
            'normalization_mode': self.normalization_combo.currentText(),
            'global_transform_mode': self.global_transform_combo.currentText(),
            'global_log_base': self.global_log_base_combo.currentText(),
            'global_log_offset': self.global_log_offset_spin.value(),
            'global_sqrt_offset': self.global_sqrt_offset_spin.value(),
            'global_y_offset': self.global_y_offset_spin.value() if hasattr(self, 'global_y_offset_spin') else 0.0,
            'is_derivative': False,  # 二次导数在预处理流程中应用
        }
    
    def _run_data_augmentation(self):
        """
        运行数据增强：生成合成数据
        
        纯组分文件夹使用说明：
        1. 文件夹应包含纯组分光谱文件（.txt 或 .csv 格式）
        2. 每个文件应包含两列数据：第一列为波数（cm⁻¹），第二列为强度
        3. 文件可以有头部（会自动跳过），也可以没有头部（直接是数据）
        4. 支持的格式：
           - 无头部：直接两列数据
           - 有头部：自动检测并跳过头部行（最多2行）
        5. 波数轴会自动对齐到当前数据的波数范围
        """
        folder = self.aug_folder_input.text()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "错误", 
                              "请先选择纯组分文件夹\n\n"
                              "使用说明：\n"
                              "1. 文件夹应包含纯组分光谱文件（.txt 或 .csv）\n"
                              "2. 每个文件包含两列：波数（cm⁻¹）和强度\n"
                              "3. 文件可以有头部（会自动跳过），也可以没有头部")
            return
        
        # 检查是否有 common_x（需要先运行一次NMF或绘图）
        if self.last_common_x is None:
            QMessageBox.warning(self, "错误", 
                              "请先运行一次NMF分析或绘图以初始化波数轴\n\n"
                              "数据增强需要知道当前数据的波数范围，以便将纯组分光谱对齐到相同的波数轴。")
            return
        
        try:
            # 初始化数据生成器
            self.data_generator = SyntheticDataGenerator(self.last_common_x)
            
            # 加载纯组分光谱
            files = glob.glob(os.path.join(folder, '*.txt')) + glob.glob(os.path.join(folder, '*.csv'))
            if not files:
                QMessageBox.warning(self, "错误", 
                                  f"纯组分文件夹中未找到光谱文件\n\n"
                                  f"文件夹路径: {folder}\n"
                                  f"请确保文件夹中包含 .txt 或 .csv 格式的光谱文件")
                return
            
            print(f"找到 {len(files)} 个文件，开始加载...")
            loaded_count = 0
            failed_files = []
            
            for file_path in files:
                name = os.path.splitext(os.path.basename(file_path))[0]
                if self.data_generator.load_pure_spectrum(file_path, name):
                    loaded_count += 1
                else:
                    failed_files.append(os.path.basename(file_path))
            
            if loaded_count == 0:
                error_msg = (f"未能加载任何纯组分光谱\n\n"
                           f"尝试加载了 {len(files)} 个文件，但都失败了。\n\n"
                           f"可能的原因：\n"
                           f"1. 文件格式不正确（需要两列数据：波数，强度）\n"
                           f"2. 文件包含非数值数据\n"
                           f"3. 文件为空或损坏\n\n"
                           f"失败的文件：\n" + "\n".join(failed_files[:5]))
                if len(failed_files) > 5:
                    error_msg += f"\n... 还有 {len(failed_files) - 5} 个文件失败"
                QMessageBox.warning(self, "错误", error_msg)
                return
            
            if loaded_count < 2:
                QMessageBox.warning(self, "警告", 
                                  f"仅加载了 {loaded_count} 个纯组分，建议至少2个\n\n"
                                  f"成功加载的组分：{list(self.data_generator.pure_spectra.keys())}\n"
                                  f"失败的文件数：{len(failed_files)}")
                if failed_files:
                    print(f"失败的文件：{failed_files}")
            
            # 获取参数
            noise_level = self.aug_noise_spin.value()
            baseline_drift = self.aug_drift_spin.value()
            complexity = self.aug_complexity_spin.value()
            use_advanced = self.aug_advanced_check.isChecked()
            
            # 设置比例范围（假设所有组分比例在0.1-0.9之间）
            component_names = list(self.data_generator.pure_spectra.keys())
            ratio_ranges = {name: (0.1, 0.9) for name in component_names}
            
            # 生成1000条合成数据（使用高级增强方法）
            n_samples = 1000
            X_synthetic, ratios_used = self.data_generator.generate_batch(
                n_samples, ratio_ranges, noise_level, baseline_drift, complexity, use_advanced
            )
            
            # 保存到文件
            save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
            if not save_dir:
                return
            
            saved_count = 0
            for i, (spectrum, ratios) in enumerate(zip(X_synthetic, ratios_used)):
                # 构建文件名（包含比例信息）
                ratio_str = "_".join([f"{name}_{ratios[name]:.2f}" for name in component_names])
                filename = f"synthetic_{i+1:04d}_{ratio_str}.txt"
                filepath = os.path.join(save_dir, filename)
                
                # 保存为两列格式（波数，强度）
                data = np.column_stack([self.last_common_x, spectrum])
                np.savetxt(filepath, data, fmt='%.6f', delimiter='\t', header='Wavenumber\tIntensity', comments='')
                saved_count += 1
            
            # 不再弹出“成功”提示框，仅在控制台打印简单日志
            print(f"[Data Augmentation] 合成数据生成完成: 纯组分 {loaded_count} 个, 生成样本 {n_samples} 条, 保存 {saved_count} 个文件, 目录: {save_dir}")
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据增强失败：{str(e)}")
            traceback.print_exc()
    
    # --- Tab 3: 波峰检测 ---
    def setup_peak_detection_tab(self):
        tab3 = QWidget()
        layout = QVBoxLayout(tab3)
        layout.setSpacing(10)
        
        # 波峰检测配置
        advanced_group = CollapsibleGroupBox("波峰检测与垂直参考线", is_expanded=True)
        adv_layout = QFormLayout()
        
        # 波峰检测开关
        self.peak_check = QCheckBox("启用自动波峰检测")
        adv_layout.addRow(self.peak_check)
        
        # 波峰检测参数组
        peak_params_group = QGroupBox("波峰检测参数")
        peak_params_layout = QFormLayout(peak_params_group)
        
        # 基础参数（使用合理的默认值，代码会自动根据数据范围调整）
        # height: 峰高阈值，0表示自动（使用数据最大值的2%）
        # 注意：SpecialValueText只在值为最小值时显示，所以将最小值设为0
        self.peak_height_spin = QDoubleSpinBox()
        self.peak_height_spin.setRange(-999999999.0, 999999999.0)  # 允许负数，0值会显示SpecialValueText
        self.peak_height_spin.setDecimals(15)
        self.peak_height_spin.setValue(0.0)  # 0表示自动（代码会使用数据最大值的0.01%）
        self.peak_height_spin.setSpecialValueText("自动 (0.01% of max)")
        self.peak_height_spin.setToolTip("峰高阈值：0=自动(0.01%)，可设置为极小值(如-999999)或负数以检测所有峰值，包括负峰")
        self.peak_height_spin.setSingleStep(0.1)
        
        # distance: 峰值之间的最小距离（数据点数量），0表示自动（使用数据点数的0.1%）
        self.peak_distance_spin = QSpinBox()
        self.peak_distance_spin.setRange(0, 999999999)  # 最小值设为0
        self.peak_distance_spin.setValue(0)  # 0表示自动，代码会自动计算
        self.peak_distance_spin.setSpecialValueText("自动 (0.1% of points)")
        self.peak_distance_spin.setToolTip("最小间距：0=自动(0.1%)，设置为1可检测所有相邻峰值，设置为更大值可过滤假峰")
        
        # prominence: 峰值突出度，0表示不使用此参数（推荐保持为0，除非需要更精确的峰值筛选）
        self.peak_prominence_spin = QDoubleSpinBox()
        self.peak_prominence_spin.setRange(0.0, 999999999.0)  # 最小值设为0
        self.peak_prominence_spin.setDecimals(15)
        self.peak_prominence_spin.setValue(0.0)  # 0表示不使用此参数（推荐保持为0）
        self.peak_prominence_spin.setSpecialValueText("禁用 (推荐)")
        
        # width: 峰值的最小宽度（数据点数量），通常不需要设置
        self.peak_width_spin = QDoubleSpinBox()
        self.peak_width_spin.setRange(0.0, 999999999.0)  # 最小值设为0
        self.peak_width_spin.setDecimals(15)
        self.peak_width_spin.setValue(0.0)  # 0表示不使用此参数（推荐保持为0）
        self.peak_width_spin.setSpecialValueText("禁用 (推荐)")
        
        # wlen: 用于计算prominence的窗口长度，通常不需要设置
        self.peak_wlen_spin = QSpinBox()
        self.peak_wlen_spin.setRange(0, 999999999)  # 最小值设为0
        self.peak_wlen_spin.setValue(0)  # 0表示不使用此参数（推荐保持为0）
        self.peak_wlen_spin.setSpecialValueText("禁用 (推荐)")
        
        # rel_height: 用于width计算的相对高度，通常不需要设置
        self.peak_rel_height_spin = QDoubleSpinBox()
        self.peak_rel_height_spin.setRange(0.0, 999999999.0)  # 最小值设为0
        self.peak_rel_height_spin.setDecimals(15)
        self.peak_rel_height_spin.setValue(0.0)  # 0表示不使用此参数（推荐保持为0）
        self.peak_rel_height_spin.setSpecialValueText("禁用 (推荐)")
        
        peak_params_layout.addRow("峰高阈值 (height):", self.peak_height_spin)
        peak_params_layout.addRow("最小间距 (distance):", self.peak_distance_spin)
        peak_params_layout.addRow("突出度 (prominence):", self.peak_prominence_spin)
        peak_params_layout.addRow("最小宽度 (width):", self.peak_width_spin)
        peak_params_layout.addRow("窗口长度 (wlen):", self.peak_wlen_spin)
        peak_params_layout.addRow("相对高度 (rel_height):", self.peak_rel_height_spin)
        
        # 添加详细说明标签
        info_text = """参数调整指南：
        
【基础参数 - 建议调整】
• height (峰高阈值): 保持"自动"即可，代码会自动使用数据最大值的0.01%
  - 如果检测到太多小峰：可以设置一个较大的值（如数据最大值的1-5%）
  - 如果检测不到峰值：保持"自动"或设置为极小值（如0.0001）以检测所有峰值
  - 要检测所有峰值：设置为极小值（如0.0001）或负数（检测负峰）
  
• distance (最小间距): 保持"自动"即可，代码会自动使用数据点数的0.1%
  - 如果两个峰太近被合并：设置为1以检测所有相邻峰值
  - 如果检测到太多假峰：可以设置一个较大的值（如10-20个数据点）
  - 要检测所有峰值：设置为1（最小间距）

【高级参数 - 通常不需要调整】
• prominence (突出度): 保持"禁用"，除非需要更精确的峰值筛选
• width (最小宽度): 保持"禁用"，除非需要过滤窄峰
• wlen (窗口长度): 保持"禁用"，除非使用prominence参数
• rel_height (相对高度): 保持"禁用"，除非使用width参数

【调整建议】
1. 首先保持所有参数为默认值（自动/禁用）
2. 如果检测不到峰值：检查数据是否经过预处理（平滑、基线校正等）
3. 如果检测到太多假峰：适当增加height或distance的值
4. 如果两个峰太近：减小distance的值"""
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #333; font-size: 9pt; padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        peak_params_layout.addRow("", info_label)
        
        adv_layout.addRow(peak_params_group)
        
        # 标记样式设置
        peak_marker_group = QGroupBox("峰值标记样式")
        peak_marker_layout = QFormLayout(peak_marker_group)
        
        self.peak_marker_shape_combo = QComboBox()
        self.peak_marker_shape_combo.addItems(['x', 'o', 's', 'D', '^', 'v', '*', '+', '.'])
        self.peak_marker_shape_combo.setCurrentText('x')
        
        self.peak_marker_size_spin = QSpinBox()
        self.peak_marker_size_spin.setRange(-999999999, 999999999)
        self.peak_marker_size_spin.setValue(10)
        
        self.peak_marker_color_input = QLineEdit("")
        self.peak_marker_color_input.setPlaceholderText("留空=使用线条颜色，例如: red, #FF0000")
        
        peak_marker_layout.addRow("标记形状:", self.peak_marker_shape_combo)
        peak_marker_layout.addRow("标记大小:", self.peak_marker_size_spin)
        peak_marker_layout.addRow("标记颜色:", self._create_h_layout([self.peak_marker_color_input, self._create_color_picker_button(self.peak_marker_color_input)]))
        
        adv_layout.addRow(peak_marker_group)
        
        # 波数显示设置
        peak_label_group = QGroupBox("波数标签显示")
        peak_label_layout = QFormLayout(peak_label_group)
        
        self.peak_show_label_check = QCheckBox("显示波数值", checked=True)
        
        self.peak_label_font_combo = QComboBox()
        self.peak_label_font_combo.addItems(['Times New Roman', 'Arial', 'SimHei', 'Courier New'])
        
        self.peak_label_size_spin = QSpinBox()
        self.peak_label_size_spin.setRange(-999999999, 999999999)
        self.peak_label_size_spin.setValue(10)
        
        self.peak_label_color_input = QLineEdit("black")
        self.peak_label_color_input.setPlaceholderText("例如: red, #FF0000")
        
        self.peak_label_bold_check = QCheckBox("字体加粗")
        
        self.peak_label_rotation_spin = QDoubleSpinBox()
        self.peak_label_rotation_spin.setRange(-999999999.0, 999999999.0)
        self.peak_label_rotation_spin.setDecimals(15)
        self.peak_label_rotation_spin.setValue(0.0)
        self.peak_label_rotation_spin.setSuffix("°")
        
        peak_label_layout.addRow(self.peak_show_label_check)
        peak_label_layout.addRow("字体:", self.peak_label_font_combo)
        peak_label_layout.addRow("字体大小:", self.peak_label_size_spin)
        peak_label_layout.addRow("颜色:", self._create_h_layout([self.peak_label_color_input, self._create_color_picker_button(self.peak_label_color_input)]))
        peak_label_layout.addRow(self.peak_label_bold_check)
        peak_label_layout.addRow("旋转角度:", self.peak_label_rotation_spin)
        
        adv_layout.addRow(peak_label_group)
        
        # 垂直参考线设置
        vertical_lines_group = QGroupBox("垂直参考线")
        vertical_lines_layout = QFormLayout(vertical_lines_group)
        
        self.vertical_lines_input = QTextEdit()
        self.vertical_lines_input.setFixedHeight(40)
        self.vertical_lines_input.setPlaceholderText("垂直参考线 (逗号分隔)")
        
        self.vertical_line_color_input = QLineEdit("gray")
        self.vertical_line_color_input.setPlaceholderText("例如: red, #FF0000")
        
        self.vertical_line_width_spin = QDoubleSpinBox()
        self.vertical_line_width_spin.setRange(-999999999.0, 999999999.0)
        self.vertical_line_width_spin.setDecimals(15)
        self.vertical_line_width_spin.setValue(0.8)
        
        self.vertical_line_style_combo = QComboBox()
        self.vertical_line_style_combo.addItems(['-', '--', '-.', ':', ''])
        self.vertical_line_style_combo.setCurrentText(':')
        
        self.vertical_line_alpha_spin = QDoubleSpinBox()
        self.vertical_line_alpha_spin.setRange(-999999999.0, 999999999.0)
        self.vertical_line_alpha_spin.setDecimals(15)
        self.vertical_line_alpha_spin.setValue(0.7)
        
        vertical_lines_layout.addRow("波数位置:", self.vertical_lines_input)
        vertical_lines_layout.addRow("颜色:", self._create_h_layout([self.vertical_line_color_input, self._create_color_picker_button(self.vertical_line_color_input)]))
        vertical_lines_layout.addRow("线宽:", self.vertical_line_width_spin)
        vertical_lines_layout.addRow("线型:", self.vertical_line_style_combo)
        vertical_lines_layout.addRow("透明度:", self.vertical_line_alpha_spin)
        
        adv_layout.addRow(vertical_lines_group)
        
        # 匹配线样式设置（单独设置，放在通用样式栏中间）
        match_lines_group = QGroupBox("匹配线样式")
        match_lines_layout = QFormLayout(match_lines_group)
        
        self.match_line_color_input = QLineEdit("red")
        self.match_line_color_input.setPlaceholderText("例如: red, #FF0000")
        
        self.match_line_width_spin = QDoubleSpinBox()
        self.match_line_width_spin.setRange(-999999999.0, 999999999.0)
        self.match_line_width_spin.setDecimals(15)
        self.match_line_width_spin.setValue(1.0)
        
        self.match_line_style_combo = QComboBox()
        self.match_line_style_combo.addItems(['-', '--', '-.', ':', ''])
        self.match_line_style_combo.setCurrentText('-')
        
        self.match_line_alpha_spin = QDoubleSpinBox()
        self.match_line_alpha_spin.setRange(-999999999.0, 999999999.0)
        self.match_line_alpha_spin.setDecimals(15)
        self.match_line_alpha_spin.setValue(0.8)
        
        match_lines_layout.addRow("颜色:", self._create_h_layout([self.match_line_color_input, self._create_color_picker_button(self.match_line_color_input)]))
        match_lines_layout.addRow("线宽:", self.match_line_width_spin)
        match_lines_layout.addRow("线型:", self.match_line_style_combo)
        match_lines_layout.addRow("透明度:", self.match_line_alpha_spin)
        
        adv_layout.addRow(match_lines_group)
        
        # RRUFF参考线设置
        rruff_ref_lines_group = QGroupBox("RRUFF匹配参考线")
        rruff_ref_lines_layout = QFormLayout(rruff_ref_lines_group)
        
        self.rruff_ref_lines_enabled_check = QCheckBox("启用RRUFF匹配参考线", checked=True)
        rruff_ref_lines_layout.addRow(self.rruff_ref_lines_enabled_check)
        
        adv_layout.addRow(rruff_ref_lines_group)
        
        # 图例重命名
        rename_group = QGroupBox("图例重命名")
        rename_group_layout = QVBoxLayout()
        self.rename_scan_button = QPushButton("扫描文件并加载重命名选项")
        self.rename_scan_button.clicked.connect(self.scan_and_load_legend_rename)
        rename_group_layout.addWidget(self.rename_scan_button)
        
        self.rename_area = QScrollArea(widgetResizable=True)
        self.rename_area.setFixedHeight(150)
        self.rename_container = QWidget()
        self.rename_layout = QVBoxLayout(self.rename_container)
        self.rename_area.setWidget(self.rename_container)
        rename_group_layout.addWidget(self.rename_area)
        
        rename_group.setLayout(rename_group_layout)
        adv_layout.addRow(rename_group)
        
        advanced_group.setContentLayout(adv_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch(1)
        # 独立窗口模式下，不再添加到 tab_widget，直接返回内容 widget
        return tab3

    # --- Tab 2: NMF 分析 ---
    # 新实现：委托给 nmf_panel.NMFPanelMixin，保留旧实现为 _setup_nmf_tab_internal_legacy
    def _setup_nmf_tab_internal(self):
        tab2 = NMFPanelMixin._setup_nmf_tab_internal(self)
        # 独立窗口模式下，直接返回 NMF 配置页
        return tab2

    def _setup_nmf_tab_internal_legacy(self):
        tab2 = QWidget()
        layout = QVBoxLayout(tab2)
        
        # --- A. NMF 参数设置 ---
        nmf_group = QGroupBox("非负矩阵分解 (NMF) 设置")
        nmf_layout = QFormLayout(nmf_group)
        
        # FIX: 修正 QSpinBox 实例化错误
        self.nmf_comp_spin = QSpinBox()
        self.nmf_comp_spin.setRange(-999999999, 999999999)
        self.nmf_comp_spin.setValue(2)
        
        self.nmf_max_iter = QSpinBox()
        self.nmf_max_iter.setRange(-999999999, 999999999)
        self.nmf_max_iter.setValue(200)
        
        nmf_layout.addRow("组件数量 (k):", self.nmf_comp_spin)
        nmf_layout.addRow("最大迭代次数:", self.nmf_max_iter)
        
        # --- 在 NMF Group 中新增预滤波控制 ---
        # 预滤波开关
        self.nmf_pca_filter_check = QCheckBox("启用预滤波/降维 (Pre-filtering)")
        self.nmf_pca_filter_check.setChecked(True)  # 默认启用
        
        # 降维算法选择（Modified NMF Algorithm Selection）
        self.nmf_filter_algo_combo = QComboBox()
        algo_options = ['PCA (主成分分析)', 'NMF (非负矩阵分解)']
        # 如果PyTorch可用，只显示Deep Autoencoder；否则显示sklearn版本
        if TORCH_AVAILABLE:
            algo_options.append('Deep Autoencoder (PyTorch)')
        else:
            algo_options.append('Autoencoder (AE - sklearn)')
        self.nmf_filter_algo_combo.addItems(algo_options)
        self.nmf_filter_algo_combo.setCurrentText('NMF (非负矩阵分解)')
        
        # 预滤波成分数（通用，适用于PCA和NMF）
        self.nmf_pca_comp_spin = QSpinBox()
        self.nmf_pca_comp_spin.setRange(-999999999, 999999999)
        self.nmf_pca_comp_spin.setValue(6)  # 默认值 6 (根据成功经验)
        
        # 随机种子（用于Deep Autoencoder，可通过滚轮切换）
        self.nmf_random_seed_spin = QSpinBox()
        self.nmf_random_seed_spin.setRange(-999999999, 999999999)
        self.nmf_random_seed_spin.setValue(42)  # 默认种子
        self.nmf_random_seed_spin.setToolTip("随机种子（用于Deep Autoencoder）\n"
                                            "使用鼠标滚轮切换种子，自动更新NMF结果\n"
                                            "不同种子会产生不同的训练结果，可手动筛选最优解")
        
        # 连接滚轮事件和值改变事件，自动重新运行NMF
        self.nmf_random_seed_spin.valueChanged.connect(self._on_seed_changed)
        
        # 将控件添加到 nmf_layout
        nmf_layout.addRow(self.nmf_pca_filter_check)
        nmf_layout.addRow(QLabel("预滤波/降维算法:"), self.nmf_filter_algo_combo)
        nmf_layout.addRow("预滤波成分数 (N_Filter):", self.nmf_pca_comp_spin)
        nmf_layout.addRow("随机种子 (Random Seed):", self.nmf_random_seed_spin)
        
        # 新增：区域权重输入（用于特征加权 NMF）
        self.nmf_region_weights_input = QLineEdit()
        self.nmf_region_weights_input.setPlaceholderText("例如: 800-1000:0.1, 1000-1200:1.0, 1200-1800:0.5")
        self.nmf_region_weights_input.setToolTip("区域权重格式：波数范围1:权重1, 波数范围2:权重2, ...\n"
                                                 "例如：800-1000:0.1 表示800-1000 cm⁻¹区域的权重为0.1\n"
                                                 "留空则所有区域权重为1.0（无加权）")
        nmf_layout.addRow("区域权重 (Region Weights):", self.nmf_region_weights_input)
        
        layout.addWidget(nmf_group)
        
        # --- A1. NMF 运行模式选择 ---
        mode_group = QGroupBox("NMF 运行模式")
        mode_layout = QVBoxLayout(mode_group)
        
        self.nmf_mode_button_group = QButtonGroup()
        self.nmf_mode_standard = QRadioButton("A. 标准 NMF (学习 H 和 W)")
        self.nmf_mode_regression = QRadioButton("B. 组分回归 (固定 H，仅计算 W)")
        self.nmf_mode_standard.setChecked(True)  # 默认选择标准模式
        
        self.nmf_mode_button_group.addButton(self.nmf_mode_standard, 0)
        self.nmf_mode_button_group.addButton(self.nmf_mode_regression, 1)
        
        mode_layout.addWidget(self.nmf_mode_standard)
        mode_layout.addWidget(self.nmf_mode_regression)
        
        mode_info_label = QLabel("提示：标准模式会同时更新H和W矩阵；组分回归模式使用上一次标准NMF得到的H矩阵，仅计算新数据的W权重。")
        mode_info_label.setWordWrap(True)
        mode_layout.addWidget(mode_info_label)
        
        layout.addWidget(mode_group)
        
        # --- B. NMF 结果绘图样式 (新增) ---
        style_group = CollapsibleGroupBox("NMF 结果绘图样式", is_expanded=True)
        style_layout = QFormLayout()
        
        # 标题和轴标签设置
        title_group = QGroupBox("标题和轴标签")
        title_layout = QFormLayout(title_group)
        
        self.nmf_top_title_input = QLineEdit("Extracted Spectra (Components)")
        self.nmf_bottom_title_input = QLineEdit("Concentration Weights (vs. Sample)")
        
        self.nmf_xlabel_top_input = QLineEdit("Wavenumber ($\\mathrm{cm^{-1}}$)")
        self.nmf_ylabel_top_input = QLineEdit("Intensity (Arb. Unit)")
        
        self.nmf_xlabel_bottom_input = QLineEdit("Sample Name")
        self.nmf_ylabel_bottom_input = QLineEdit("Weight (Arb. Unit)")
        
        title_layout.addRow("上图标题:", self.nmf_top_title_input)
        title_layout.addRow("下图标题:", self.nmf_bottom_title_input)
        title_layout.addRow("上图X轴标签:", self.nmf_xlabel_top_input)
        
        # NMF上图X轴标题控制：大小、间距、显示/隐藏
        self.nmf_top_xlabel_font_spin = QSpinBox()
        self.nmf_top_xlabel_font_spin.setRange(-999999999, 999999999)
        self.nmf_top_xlabel_font_spin.setValue(16)  # 默认值
        
        self.nmf_top_xlabel_pad_spin = QDoubleSpinBox()
        self.nmf_top_xlabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.nmf_top_xlabel_pad_spin.setDecimals(15)
        self.nmf_top_xlabel_pad_spin.setValue(10.0)  # 默认值
        
        self.nmf_top_xlabel_show_check = QCheckBox("显示上图X轴标题")
        self.nmf_top_xlabel_show_check.setChecked(True)  # 默认显示
        
        title_layout.addRow("上图X轴标题控制:", self._create_h_layout([self.nmf_top_xlabel_show_check, QLabel("大小:"), self.nmf_top_xlabel_font_spin, QLabel("间距:"), self.nmf_top_xlabel_pad_spin]))
        
        title_layout.addRow("上图Y轴标签:", self.nmf_ylabel_top_input)
        
        # NMF上图Y轴标题控制：大小、间距、显示/隐藏
        self.nmf_top_ylabel_font_spin = QSpinBox()
        self.nmf_top_ylabel_font_spin.setRange(-999999999, 999999999)
        self.nmf_top_ylabel_font_spin.setValue(16)  # 默认值
        
        self.nmf_top_ylabel_pad_spin = QDoubleSpinBox()
        self.nmf_top_ylabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.nmf_top_ylabel_pad_spin.setDecimals(15)
        self.nmf_top_ylabel_pad_spin.setValue(10.0)  # 默认值
        
        self.nmf_top_ylabel_show_check = QCheckBox("显示上图Y轴标题")
        self.nmf_top_ylabel_show_check.setChecked(True)  # 默认显示
        
        title_layout.addRow("上图Y轴标题控制:", self._create_h_layout([self.nmf_top_ylabel_show_check, QLabel("大小:"), self.nmf_top_ylabel_font_spin, QLabel("间距:"), self.nmf_top_ylabel_pad_spin]))
        
        title_layout.addRow("下图X轴标签:", self.nmf_xlabel_bottom_input)
        
        # NMF下图X轴标题控制：大小、间距、显示/隐藏
        self.nmf_bottom_xlabel_font_spin = QSpinBox()
        self.nmf_bottom_xlabel_font_spin.setRange(-999999999, 999999999)
        self.nmf_bottom_xlabel_font_spin.setValue(16)  # 默认值
        
        self.nmf_bottom_xlabel_pad_spin = QDoubleSpinBox()
        self.nmf_bottom_xlabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.nmf_bottom_xlabel_pad_spin.setDecimals(15)
        self.nmf_bottom_xlabel_pad_spin.setValue(10.0)  # 默认值
        
        self.nmf_bottom_xlabel_show_check = QCheckBox("显示下图X轴标题")
        self.nmf_bottom_xlabel_show_check.setChecked(True)  # 默认显示
        
        title_layout.addRow("下图X轴标题控制:", self._create_h_layout([self.nmf_bottom_xlabel_show_check, QLabel("大小:"), self.nmf_bottom_xlabel_font_spin, QLabel("间距:"), self.nmf_bottom_xlabel_pad_spin]))
        
        title_layout.addRow("下图Y轴标签:", self.nmf_ylabel_bottom_input)
        
        # NMF下图Y轴标题控制：大小、间距、显示/隐藏
        self.nmf_bottom_ylabel_font_spin = QSpinBox()
        self.nmf_bottom_ylabel_font_spin.setRange(-999999999, 999999999)
        self.nmf_bottom_ylabel_font_spin.setValue(16)  # 默认值
        
        self.nmf_bottom_ylabel_pad_spin = QDoubleSpinBox()
        self.nmf_bottom_ylabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.nmf_bottom_ylabel_pad_spin.setDecimals(15)
        self.nmf_bottom_ylabel_pad_spin.setValue(10.0)  # 默认值
        
        self.nmf_bottom_ylabel_show_check = QCheckBox("显示下图Y轴标题")
        self.nmf_bottom_ylabel_show_check.setChecked(True)  # 默认显示
        
        title_layout.addRow("下图Y轴标题控制:", self._create_h_layout([self.nmf_bottom_ylabel_show_check, QLabel("大小:"), self.nmf_bottom_ylabel_font_spin, QLabel("间距:"), self.nmf_bottom_ylabel_pad_spin]))
        
        style_layout.addRow(title_group)
        
        # 字体设置
        self.nmf_title_font_spin = QSpinBox()
        self.nmf_title_font_spin.setRange(-999999999, 999999999)
        self.nmf_title_font_spin.setValue(16)
        
        self.nmf_tick_font_spin = QSpinBox()
        self.nmf_tick_font_spin.setRange(-999999999, 999999999)
        self.nmf_tick_font_spin.setValue(10)
        
        style_layout.addRow("标题 / 刻度字体:", self._create_h_layout([self.nmf_title_font_spin, self.nmf_tick_font_spin]))
        
        # NMF上图标题控制：大小、间距、显示/隐藏
        self.nmf_top_title_font_spin = QSpinBox()
        self.nmf_top_title_font_spin.setRange(-999999999, 999999999)
        self.nmf_top_title_font_spin.setValue(16)  # 默认值
        
        self.nmf_top_title_pad_spin = QDoubleSpinBox()
        self.nmf_top_title_pad_spin.setRange(-999999999.0, 999999999.0)
        self.nmf_top_title_pad_spin.setDecimals(15)
        self.nmf_top_title_pad_spin.setValue(10.0)  # 默认值
        
        self.nmf_top_title_show_check = QCheckBox("显示上图标题")
        self.nmf_top_title_show_check.setChecked(True)  # 默认显示
        
        # NMF下图标题控制：大小、间距、显示/隐藏
        self.nmf_bottom_title_font_spin = QSpinBox()
        self.nmf_bottom_title_font_spin.setRange(-999999999, 999999999)
        self.nmf_bottom_title_font_spin.setValue(16)  # 默认值
        
        self.nmf_bottom_title_pad_spin = QDoubleSpinBox()
        self.nmf_bottom_title_pad_spin.setRange(-999999999.0, 999999999.0)
        self.nmf_bottom_title_pad_spin.setDecimals(15)
        self.nmf_bottom_title_pad_spin.setValue(10.0)  # 默认值
        
        self.nmf_bottom_title_show_check = QCheckBox("显示下图标题")
        self.nmf_bottom_title_show_check.setChecked(True)  # 默认显示
        
        style_layout.addRow("上图标题控制:", self._create_h_layout([self.nmf_top_title_show_check, QLabel("大小:"), self.nmf_top_title_font_spin, QLabel("间距:"), self.nmf_top_title_pad_spin]))
        style_layout.addRow("下图标题控制:", self._create_h_layout([self.nmf_bottom_title_show_check, QLabel("大小:"), self.nmf_bottom_title_font_spin, QLabel("间距:"), self.nmf_bottom_title_pad_spin]))
        
        # H (Spectra) 样式
        self.nmf_comp_line_width = QDoubleSpinBox()
        self.nmf_comp_line_width.setRange(-999999999.0, 999999999.0)
        self.nmf_comp_line_width.setDecimals(15)
        self.nmf_comp_line_width.setValue(2.0)
        
        self.nmf_comp_line_style = QComboBox()
        self.nmf_comp_line_style.addItems(['-', '--', ':', '-.'])
        self.nmf_comp_line_style.setCurrentText('-')
        
        style_layout.addRow("光谱线宽 / 线型:", self._create_h_layout([self.nmf_comp_line_width, self.nmf_comp_line_style]))
        
        self.comp1_color_input = QLineEdit("blue")
        self.comp2_color_input = QLineEdit("red")
        style_layout.addRow("Comp 1 颜色:", self._create_h_layout([self.comp1_color_input, self._create_color_picker_button(self.comp1_color_input)]))
        style_layout.addRow("Comp 2 颜色:", self._create_h_layout([self.comp2_color_input, self._create_color_picker_button(self.comp2_color_input)]))
        
        # 连接颜色控件到自动更新
        self.comp1_color_input.textChanged.connect(self._on_nmf_color_changed)
        self.comp2_color_input.textChanged.connect(self._on_nmf_color_changed)

        # W (Weights) 样式
        self.nmf_weight_line_width = QDoubleSpinBox()
        self.nmf_weight_line_width.setRange(-999999999.0, 999999999.0)
        self.nmf_weight_line_width.setDecimals(15)
        self.nmf_weight_line_width.setValue(1.0)
        
        self.nmf_weight_line_style = QComboBox()
        self.nmf_weight_line_style.addItems(['-', '--', ':', ''])
        self.nmf_weight_line_style.setCurrentText('-')
        
        self.nmf_marker_size = QSpinBox()
        self.nmf_marker_size.setRange(-999999999, 999999999)
        self.nmf_marker_size.setValue(8)

        self.nmf_marker_style = QComboBox()
        self.nmf_marker_style.addItems(['o', 'x', 's', 'D', '^'])
        self.nmf_marker_style.setCurrentText('o')
        
        style_layout.addRow("权重线宽 / 线型:", self._create_h_layout([self.nmf_weight_line_width, self.nmf_weight_line_style]))
        style_layout.addRow("标记大小 / 样式:", self._create_h_layout([self.nmf_marker_size, self.nmf_marker_style]))
        
        style_group.setContentLayout(style_layout)
        layout.addWidget(style_group)
        
        # --- C. NMF 文件排序设置 ---
        sort_group = CollapsibleGroupBox("📋 NMF 文件排序设置", is_expanded=True)
        sort_layout = QFormLayout()
        
        self.nmf_sort_method_combo = QComboBox()
        self.nmf_sort_method_combo.addItems(['按文件名排序', '按修改时间排序', '按文件大小排序', '自定义顺序'])
        self.nmf_sort_method_combo.setCurrentText('按文件名排序')
        self.nmf_sort_method_combo.currentTextChanged.connect(self._update_nmf_sort_preview)
        
        self.nmf_sort_reverse_check = QCheckBox("降序（Z→A）")
        
        self.nmf_file_preview_list = QListWidget()
        self.nmf_file_preview_list.setMaximumHeight(150)
        self.nmf_file_preview_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)  # 允许拖拽排序
        self.nmf_file_preview_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)  # 允许多选
        
        # 添加右键菜单用于删除文件
        self.nmf_file_preview_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.nmf_file_preview_list.customContextMenuRequested.connect(self._show_nmf_file_context_menu)
        
        self.nmf_refresh_preview_btn = QPushButton("刷新预览")
        self.nmf_refresh_preview_btn.clicked.connect(self._update_nmf_sort_preview)
        
        self.nmf_remove_selected_btn = QPushButton("删除选中文件（不参与NMF）")
        self.nmf_remove_selected_btn.clicked.connect(self._remove_selected_nmf_files)
        
        sort_layout.addRow("排序方式:", self.nmf_sort_method_combo)
        sort_layout.addRow(self.nmf_sort_reverse_check)
        sort_layout.addRow("文件顺序预览（可拖拽调整，右键删除）:", self.nmf_file_preview_list)
        sort_layout.addRow(self._create_h_layout([self.nmf_refresh_preview_btn, self.nmf_remove_selected_btn]))
        
        sort_group.setContentLayout(sort_layout)
        layout.addWidget(sort_group)
        
        # --- D. NMF 对照组设置 ---
        control_group = CollapsibleGroupBox("NMF 对照组设置", is_expanded=True)
        control_layout = QFormLayout()
        
        self.nmf_include_control_check = QCheckBox("对照组参与NMF解混分析")
        self.nmf_include_control_check.setChecked(False)  # 默认不参与
        control_layout.addRow(self.nmf_include_control_check)
        
        control_info_label = QLabel("提示：如果勾选，对照文件将参与NMF解混；否则仅用于绘图对比。")
        control_info_label.setWordWrap(True)
        control_layout.addRow(control_info_label)
        
        control_group.setContentLayout(control_layout)
        layout.addWidget(control_group)
        
        # --- C. 运行按钮 ---
        # NMF运行按钮已移到主界面左侧按钮区，这里不再需要
        layout.addStretch(1)
        
        # 添加 NMF 提示
        info_label = QLabel("提示：NMF 分析将使用GUI中设置的所有预处理选项（QC、BE校正、平滑、基线校正、归一化等）。\n最终会将负值置零以满足NMF的非负要求。请确保在 'X 轴物理截断' 中设置了范围（例如 > 600 cm⁻¹）。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 独立窗口模式下，直接返回 NMF 设置页
        return tab2

    # --- Tab 3: 物理验证 ---
    def setup_physics_tab(self):
        tab3 = QWidget()
        layout = QVBoxLayout(tab3)
        
        # 3.1 Bose-Einstein 校正 (移除，已整合到预处理)
        
        # 3.2 瑞利散射尾拟合 (修改为叠加模式)
        fit_group = CollapsibleGroupBox("📈 散射尾部拟合 (叠加到当前图)", is_expanded=True)
        fit_layout = QFormLayout()
        
        self.fit_cutoff_spin = QDoubleSpinBox()
        self.fit_cutoff_spin.setRange(-999999999.0, 999999999.0)
        self.fit_cutoff_spin.setDecimals(15)
        self.fit_cutoff_spin.setValue(400.0)
        
        self.fit_model_combo = QComboBox()
        self.fit_model_combo.addItems(['Lorentzian', 'Gaussian'])
        fit_layout.addRow("拟合截止波数 (cm⁻¹):", self.fit_cutoff_spin)
        fit_layout.addRow("拟合模型:", self.fit_model_combo)
        
        # 拟合曲线样式控制
        self.fit_line_color_input = QLineEdit("magenta")
        self.fit_line_style_combo = QComboBox()
        self.fit_line_style_combo.addItems(['-', '--', '-.', ':'])
        self.fit_line_style_combo.setCurrentText('--')
        
        self.fit_line_width_spin = QDoubleSpinBox()
        self.fit_line_width_spin.setRange(-999999999.0, 999999999.0)
        self.fit_line_width_spin.setDecimals(15)
        self.fit_line_width_spin.setValue(2.5)
        
        self.fit_marker_combo = QComboBox()
        self.fit_marker_combo.addItems(['无', 'o', 's', '^', 'D', 'x', '+', '*'])
        self.fit_marker_combo.setCurrentText('无')
        
        self.fit_marker_size_spin = QDoubleSpinBox()
        self.fit_marker_size_spin.setRange(-999999999.0, 999999999.0)
        self.fit_marker_size_spin.setDecimals(15)
        self.fit_marker_size_spin.setValue(5.0)
        
        fit_layout.addRow("拟合线颜色:", self._create_h_layout([self.fit_line_color_input, self._create_color_picker_button(self.fit_line_color_input)]))
        fit_layout.addRow("拟合线型 / 线宽:", self._create_h_layout([self.fit_line_style_combo, self.fit_line_width_spin]))
        fit_layout.addRow("标记样式 / 大小:", self._create_h_layout([self.fit_marker_combo, self.fit_marker_size_spin]))
        
        # 拟合曲线图例控制
        self.fit_legend_label_input = QLineEdit("")
        self.fit_legend_label_input.setPlaceholderText("留空则自动生成，例如: Fit: 文件名")
        
        self.fit_show_legend_check = QCheckBox("显示拟合曲线图例")
        self.fit_show_legend_check.setChecked(True)
        self.fit_show_legend_check.setToolTip("遵循主菜单的图例显示设置，但可以单独控制拟合曲线的图例")
        
        fit_layout.addRow("图例标签:", self.fit_legend_label_input)
        fit_layout.addRow("", self.fit_show_legend_check)
        
        # 支持多条拟合曲线
        self.fit_curve_count_spin = QSpinBox()
        self.fit_curve_count_spin.setRange(-999999999, 999999999)
        self.fit_curve_count_spin.setValue(1)
        self.fit_curve_count_spin.setToolTip("可以多次运行拟合，每次生成一条曲线，最多支持10条")
        
        self.btn_clear_fits = QPushButton("清除所有拟合曲线")
        self.btn_clear_fits.setStyleSheet("background-color: #FF5722; color: white; font-weight: bold;")
        self.btn_clear_fits.clicked.connect(self.clear_all_fit_curves)
        
        fit_layout.addRow("拟合曲线数量:", self.fit_curve_count_spin)
        fit_layout.addRow("", self.btn_clear_fits)
        
        self.btn_run_fit = QPushButton("运行拟合并叠加到当前图")
        self.btn_run_fit.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        self.btn_run_fit.clicked.connect(self.run_scattering_fit_overlay)
        fit_layout.addRow("", self.btn_run_fit)
        
        self.fit_output_text = QTextEdit()
        self.fit_output_text.setReadOnly(True)
        self.fit_output_text.setFixedHeight(150)
        fit_layout.addRow("拟合结果:", self.fit_output_text)
        
        fit_group.setContentLayout(fit_layout)
        layout.addWidget(fit_group)
        
        # 存储拟合曲线信息（用于清除和样式管理）
        self.fit_curves_info = []  # 存储拟合曲线的信息列表
        layout.addStretch(1)
        
        # 独立窗口模式下，直接返回物理验证页
        return tab3

    # --- 辅助逻辑 (文件扫描和重命名) ---
    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据文件夹")
        if d: 
            self.folder_input.setText(d)
            self.scan_and_load_legend_rename() # 扫描并加载重命名
            # 自动检测跳过行数
            self._on_folder_changed()

    def scan_and_load_legend_rename(self):
        # 扫描文件，为图例重命名做准备（包括瀑布图的组名）
        try:
            folder_path = self.folder_input.text()
            if not os.path.isdir(folder_path): return

            self.legend_rename_widgets.clear()
            self._clear_layout_recursively(self.rename_layout)
            
            # 1. 扫描文件（用于主图）
            csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
            txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
            file_list_full = sorted(csv_files + txt_files) 
            
            # 2. 扫描分组（用于瀑布图）
            n_chars = self.n_chars_spin.value()
            groups = group_files_by_name(file_list_full, n_chars)
            
            # 筛选指定组（如果设置了）
            target_gs = [x.strip() for x in self.groups_input.text().split(',') if x.strip()]
            if target_gs:
                groups = {k: v for k, v in groups.items() if k in target_gs}
            
            # 3. 先收集所有组中的文件，避免重复添加
            files_in_groups = set()
            for g_files in groups.values():
                files_in_groups.update(g_files)
                
            # 4. 为组名创建重命名选项（用于瀑布图）- 包括平均线和标准方差
            for g_name in sorted(groups.keys()):
                # 4.1 基础组名（用于重命名基础名称）
                h1 = QHBoxLayout()
                lbl1 = QLabel(f"{g_name} (组-基础)")
                lbl1.setFixedWidth(150)
                lbl1.setStyleSheet("color: #2196F3; font-weight: bold;")
                rename_input_base = QLineEdit(placeholderText="新的组名（影响所有图例）")
                delete_btn1 = QPushButton("删除")
                delete_btn1.setFixedWidth(50)
                delete_btn1.setStyleSheet("background-color: #f44336; color: white;")
                widget_container1 = QWidget()
                widget_container1.setLayout(h1)
                
                def create_delete_handler1(widget, key):
                    def delete_handler():
                        if key in self.legend_rename_widgets:
                            del self.legend_rename_widgets[key]
                        widget.setParent(None)
                        widget.deleteLater()
                    return delete_handler
                
                delete_btn1.clicked.connect(create_delete_handler1(widget_container1, g_name))
                h1.addWidget(lbl1)
                h1.addWidget(QLabel("→"))
                h1.addWidget(rename_input_base)
                h1.addWidget(delete_btn1)
                h1.addStretch(1)
                self.rename_layout.addWidget(widget_container1)
                self.legend_rename_widgets[g_name] = rename_input_base
                
                # 4.2 平均线图例 (Avg)
                h2 = QHBoxLayout()
                lbl2 = QLabel(f"{g_name} (Avg)")
                lbl2.setFixedWidth(150)
                lbl2.setStyleSheet("color: #4CAF50;")
                rename_input_avg = QLineEdit(placeholderText="新的平均线图例名称")
                delete_btn2 = QPushButton("删除")
                delete_btn2.setFixedWidth(50)
                delete_btn2.setStyleSheet("background-color: #f44336; color: white;")
                widget_container2 = QWidget()
                widget_container2.setLayout(h2)
                
                def create_delete_handler2(widget, key):
                    def delete_handler():
                        if key in self.legend_rename_widgets:
                            del self.legend_rename_widgets[key]
                        widget.setParent(None)
                        widget.deleteLater()
                    return delete_handler
                
                delete_btn2.clicked.connect(create_delete_handler2(widget_container2, f"{g_name} (Avg)"))
                h2.addWidget(lbl2)
                h2.addWidget(QLabel("→"))
                h2.addWidget(rename_input_avg)
                h2.addWidget(delete_btn2)
                h2.addStretch(1)
                self.rename_layout.addWidget(widget_container2)
                self.legend_rename_widgets[f"{g_name} (Avg)"] = rename_input_avg
                
                # 4.3 标准方差图例 (± Std)
                h3 = QHBoxLayout()
                lbl3 = QLabel(f"{g_name} ± Std")
                lbl3.setFixedWidth(150)
                lbl3.setStyleSheet("color: #FF9800;")
                rename_input_std = QLineEdit(placeholderText="新的标准方差图例名称")
                delete_btn3 = QPushButton("删除")
                delete_btn3.setFixedWidth(50)
                delete_btn3.setStyleSheet("background-color: #f44336; color: white;")
                widget_container3 = QWidget()
                widget_container3.setLayout(h3)
                
                def create_delete_handler3(widget, key):
                    def delete_handler():
                        if key in self.legend_rename_widgets:
                            del self.legend_rename_widgets[key]
                        widget.setParent(None)
                        widget.deleteLater()
                    return delete_handler
                
                delete_btn3.clicked.connect(create_delete_handler3(widget_container3, f"{g_name} ± Std"))
                h3.addWidget(lbl3)
                h3.addWidget(QLabel("→"))
                h3.addWidget(rename_input_std)
                h3.addWidget(delete_btn3)
                h3.addStretch(1)
                self.rename_layout.addWidget(widget_container3)
                self.legend_rename_widgets[f"{g_name} ± Std"] = rename_input_std
            
            # 5. 为组名添加Mean + Shadow模式的图例项（如果组中有多个文件）
            for g_name in sorted(groups.keys()):
                g_files = groups[g_name]
                # 如果组中有多个文件，会使用Mean + Shadow模式
                if len(g_files) > 1:
                    # 5.1 Mean图例
                    mean_key = f"{g_name} Mean"
                    if mean_key not in self.legend_rename_widgets:
                        h_mean = QHBoxLayout()
                        lbl_mean = QLabel(mean_key)
                        lbl_mean.setFixedWidth(150)
                        lbl_mean.setStyleSheet("color: #4CAF50;")
                        rename_input_mean = QLineEdit(placeholderText="新的平均线图例名称")
                        delete_btn_mean = QPushButton("删除")
                        delete_btn_mean.setFixedWidth(50)
                        delete_btn_mean.setStyleSheet("background-color: #f44336; color: white;")
                        widget_container_mean = QWidget()
                        widget_container_mean.setLayout(h_mean)
                        
                        def create_delete_handler_mean(widget, key):
                            def delete_handler():
                                if key in self.legend_rename_widgets:
                                    del self.legend_rename_widgets[key]
                                widget.setParent(None)
                                widget.deleteLater()
                            return delete_handler
                        
                        delete_btn_mean.clicked.connect(create_delete_handler_mean(widget_container_mean, mean_key))
                        h_mean.addWidget(lbl_mean)
                        h_mean.addWidget(QLabel("→"))
                        h_mean.addWidget(rename_input_mean)
                        h_mean.addWidget(delete_btn_mean)
                        h_mean.addStretch(1)
                        self.rename_layout.addWidget(widget_container_mean)
                        self.legend_rename_widgets[mean_key] = rename_input_mean
                    
                    # 5.2 Std Dev图例
                    std_key = f"{g_name} Std Dev"
                    if std_key not in self.legend_rename_widgets:
                        h_std = QHBoxLayout()
                        lbl_std = QLabel(std_key)
                        lbl_std.setFixedWidth(150)
                        lbl_std.setStyleSheet("color: #FF9800;")
                        rename_input_std = QLineEdit(placeholderText="新的标准方差图例名称")
                        delete_btn_std = QPushButton("删除")
                        delete_btn_std.setFixedWidth(50)
                        delete_btn_std.setStyleSheet("background-color: #f44336; color: white;")
                        widget_container_std = QWidget()
                        widget_container_std.setLayout(h_std)
                        
                        def create_delete_handler_std(widget, key):
                            def delete_handler():
                                if key in self.legend_rename_widgets:
                                    del self.legend_rename_widgets[key]
                                widget.setParent(None)
                                widget.deleteLater()
                            return delete_handler
                        
                        delete_btn_std.clicked.connect(create_delete_handler_std(widget_container_std, std_key))
                        h_std.addWidget(lbl_std)
                        h_std.addWidget(QLabel("→"))
                        h_std.addWidget(rename_input_std)
                        h_std.addWidget(delete_btn_std)
                        h_std.addStretch(1)
                        self.rename_layout.addWidget(widget_container_std)
                        self.legend_rename_widgets[std_key] = rename_input_std
            
            # 6. 为柱状图添加图例项（定量校准结果）
            bar_legend_items = [
                '原始权重 ($w_{low}$)',
                '原始权重',  # 简化版本
                '校准后权重 ($w_{calibrated}$)',
                '校准后权重',  # 简化版本
                '空白偏差'
            ]
            for item in bar_legend_items:
                if item not in self.legend_rename_widgets:
                    h = QHBoxLayout()
                    lbl = QLabel(f"{item} (柱状图)")
                    lbl.setFixedWidth(150)
                    lbl.setStyleSheet("color: #9C27B0; font-weight: bold;")
                    rename_input = QLineEdit(placeholderText="新的图例名称")
                    delete_btn = QPushButton("删除")
                    delete_btn.setFixedWidth(50)
                    delete_btn.setStyleSheet("background-color: #f44336; color: white;")
                    widget_container = QWidget()
                    widget_container.setLayout(h)
                    
                    def create_delete_handler(widget, key):
                        def delete_handler():
                            if key in self.legend_rename_widgets:
                                del self.legend_rename_widgets[key]
                            widget.setParent(None)
                            widget.deleteLater()
                        return delete_handler
                    
                    delete_btn.clicked.connect(create_delete_handler(widget_container, item))
                    h.addWidget(lbl)
                    h.addWidget(QLabel("→"))
                    h.addWidget(rename_input)
                    h.addWidget(delete_btn)
                    h.addStretch(1)
                    self.rename_layout.addWidget(widget_container)
                    self.legend_rename_widgets[item] = rename_input
            
            # 7. 为NMF解谱图添加图例项（如果NMF窗口存在）
            if hasattr(self, 'nmf_window') and self.nmf_window is not None:
                if hasattr(self.nmf_window, 'n_components') and self.nmf_window.n_components > 0:
                    for i in range(self.nmf_window.n_components):
                        nmf_label = f"NMF Component {i+1}"
                        if nmf_label not in self.legend_rename_widgets:
                            h = QHBoxLayout()
                            lbl = QLabel(f"{nmf_label} (NMF)")
                            lbl.setFixedWidth(150)
                            lbl.setStyleSheet("color: #FF5722; font-weight: bold;")
                            rename_input = QLineEdit(placeholderText="新的图例名称")
                            delete_btn = QPushButton("删除")
                            delete_btn.setFixedWidth(50)
                            delete_btn.setStyleSheet("background-color: #f44336; color: white;")
                            widget_container = QWidget()
                            widget_container.setLayout(h)
                            
                            def create_delete_handler_nmf(widget, key):
                                def delete_handler():
                                    if key in self.legend_rename_widgets:
                                        del self.legend_rename_widgets[key]
                                    widget.setParent(None)
                                    widget.deleteLater()
                                return delete_handler
                            
                            delete_btn.clicked.connect(create_delete_handler_nmf(widget_container, nmf_label))
                            h.addWidget(lbl)
                            h.addWidget(QLabel("→"))
                            h.addWidget(rename_input)
                            h.addWidget(delete_btn)
                            h.addStretch(1)
                            self.rename_layout.addWidget(widget_container)
                            self.legend_rename_widgets[nmf_label] = rename_input
            
            # 8. 为拟合验证图添加图例项
            fit_legend_items = [
                '原始光谱',
                '拟合结果',
                '残差'
            ]
            for item in fit_legend_items:
                if item not in self.legend_rename_widgets:
                    h = QHBoxLayout()
                    lbl = QLabel(f"{item} (拟合验证)")
                    lbl.setFixedWidth(150)
                    lbl.setStyleSheet("color: #607D8B; font-weight: bold;")
                    rename_input = QLineEdit(placeholderText="新的图例名称")
                    delete_btn = QPushButton("删除")
                    delete_btn.setFixedWidth(50)
                    delete_btn.setStyleSheet("background-color: #f44336; color: white;")
                    widget_container = QWidget()
                    widget_container.setLayout(h)
                    
                    def create_delete_handler_fit(widget, key):
                        def delete_handler():
                            if key in self.legend_rename_widgets:
                                del self.legend_rename_widgets[key]
                            widget.setParent(None)
                            widget.deleteLater()
                        return delete_handler
                    
                    delete_btn.clicked.connect(create_delete_handler_fit(widget_container, item))
                    h.addWidget(lbl)
                    h.addWidget(QLabel("→"))
                    h.addWidget(rename_input)
                    h.addWidget(delete_btn)
                    h.addStretch(1)
                    self.rename_layout.addWidget(widget_container)
                    self.legend_rename_widgets[item] = rename_input
            
            # 9. 为文件创建重命名选项（用于主图）
            for file_path in file_list_full:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # 检查是否已经在组中（如果是组的一部分，跳过，因为组名已经添加）
                file_group = None
                for g_name, g_files in groups.items():
                    if file_path in g_files:
                        file_group = g_name
                        break
                
                # 如果文件属于某个组，且组名已添加，则跳过
                if file_group and file_group in self.legend_rename_widgets:
                    continue
                
                h = QHBoxLayout()
                lbl = QLabel(base_name)
                lbl.setFixedWidth(150)
                
                rename_input = QLineEdit(placeholderText="新的图例名称")
                
                # 删除按钮
                delete_btn = QPushButton("删除")
                delete_btn.setFixedWidth(50)
                delete_btn.setStyleSheet("background-color: #f44336; color: white;")
                
                # 存储widget引用以便删除
                widget_container = QWidget()
                widget_container.setLayout(h)
                
                def create_delete_handler(widget, key):
                    def delete_handler():
                        # 从字典中删除
                        if key in self.legend_rename_widgets:
                            del self.legend_rename_widgets[key]
                        # 从布局中删除widget
                        widget.setParent(None)
                        widget.deleteLater()
                    return delete_handler
                
                delete_btn.clicked.connect(create_delete_handler(widget_container, base_name))
                
                h.addWidget(lbl)
                h.addWidget(QLabel("→"))
                h.addWidget(rename_input)
                h.addWidget(delete_btn)
                h.addStretch(1)
                
                self.rename_layout.addWidget(widget_container)
                self.legend_rename_widgets[base_name] = rename_input

            self.rename_layout.addStretch(1)
        except Exception:
            traceback.print_exc()

    def scan_and_load_file_controls(self):
        # 扫描文件，为独立 Y 轴控制和预处理做准备
        try:
            folder_path = self.folder_input.text()
            if not os.path.isdir(folder_path): return

            self.individual_control_widgets.clear()
            self._clear_layout_recursively(self.dynamic_controls_layout)
            
            csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
            txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
            file_list_full = sorted(csv_files + txt_files) 
            
            if not file_list_full: 
                # 不再弹出“未找到文件”提示框
                return

            for file_path in file_list_full:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # 创建文件控制容器（使用垂直布局，包含多行）
                file_widget = QWidget()
                file_vbox = QVBoxLayout(file_widget)
                file_vbox.setContentsMargins(5, 5, 5, 5)
                file_vbox.setSpacing(5)
                
                # 文件名标签
                name_label = QLabel(f"📄 {base_name}")
                name_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
                file_vbox.addWidget(name_label)
                
                # 第一行：独立Y轴控制
                h1 = QHBoxLayout()
                h1.addWidget(QLabel("Y轴控制:"))
                
                scale_sb = QDoubleSpinBox()
                scale_sb.setRange(-999999999.0, 999999999.0)
                scale_sb.setDecimals(15)
                scale_sb.setValue(1.0)
                scale_sb.setToolTip("Y轴缩放因子")
                
                offset_sb = QDoubleSpinBox()
                offset_sb.setRange(-999999999.0, 999999999.0)
                offset_sb.setDecimals(15)
                offset_sb.setValue(0.0)
                offset_sb.setToolTip("Y轴偏移量")
                
                h1.addWidget(QLabel("Scale:"))
                h1.addWidget(scale_sb)
                h1.addWidget(QLabel("Offset:"))
                h1.addWidget(offset_sb)
                
                # 添加颜色选择
                h1.addWidget(QLabel("颜色:"))
                color_input = QLineEdit()
                # 使用默认颜色序列
                default_colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal', 'darkred']
                color_idx = len(self.individual_control_widgets) % len(default_colors)
                color_input.setText(default_colors[color_idx])
                color_input.setToolTip("线条颜色（支持颜色名称如'red'、'blue'或十六进制如'#FF0000'）")
                color_input.setMaximumWidth(100)
                h1.addWidget(color_input)
                # 添加颜色选择器按钮
                color_button = self._create_color_picker_button(color_input)
                h1.addWidget(color_button)
                
                h1.addStretch(1)
                file_vbox.addLayout(h1)
                
                # 第二行：预处理选项（对数/平方根变换）
                h2 = QHBoxLayout()
                h2.addWidget(QLabel("动态范围压缩:"))
                
                transform_combo = QComboBox()
                transform_combo.addItems(['无', '对数变换 (Log)', '平方根变换 (Sqrt)'])
                transform_combo.setCurrentText('无')
                transform_combo.setToolTip("压缩高强度信号动态范围，凸显微弱峰值")
                
                # 对数变换参数
                log_base_combo = QComboBox()
                log_base_combo.addItems(['10', 'e'])
                log_base_combo.setCurrentText('10')
                log_base_combo.setToolTip("对数底数")
                
                log_offset_spin = QDoubleSpinBox()
                log_offset_spin.setRange(-999999999.0, 999999999.0)
                log_offset_spin.setDecimals(15)
                log_offset_spin.setValue(1.0)
                log_offset_spin.setToolTip("对数变换偏移量")
                
                # 平方根变换参数
                sqrt_offset_spin = QDoubleSpinBox()
                sqrt_offset_spin.setRange(-999999999.0, 999999999.0)
                sqrt_offset_spin.setDecimals(15)
                sqrt_offset_spin.setValue(0.0)
                sqrt_offset_spin.setToolTip("平方根变换偏移量")
                
                # 参数容器（根据选择的变换类型显示/隐藏）
                params_widget = QWidget()
                params_layout = QHBoxLayout(params_widget)
                params_layout.setContentsMargins(0, 0, 0, 0)
                
                log_params_label = QLabel("底数:")
                log_params_label.hide()
                log_base_combo.hide()
                log_offset_label = QLabel("偏移:")
                log_offset_label.hide()
                log_offset_spin.hide()
                
                sqrt_params_label = QLabel("偏移:")
                sqrt_params_label.hide()
                sqrt_offset_spin.hide()
                
                params_layout.addWidget(log_params_label)
                params_layout.addWidget(log_base_combo)
                params_layout.addWidget(log_offset_label)
                params_layout.addWidget(log_offset_spin)
                params_layout.addWidget(sqrt_params_label)
                params_layout.addWidget(sqrt_offset_spin)
                params_layout.addStretch(1)
                
                # 使用lambda闭包确保每个文件的控件独立绑定
                def make_update_func(log_lbl, log_base, log_off_lbl, log_off_spin, sqrt_lbl, sqrt_spin):
                    """创建更新函数，确保每个文件的控件独立"""
                    def update_transform_params(index):
                        """根据选择的变换类型显示/隐藏相应参数"""
                        if index == 0:  # 无
                            log_lbl.hide()
                            log_base.hide()
                            log_off_lbl.hide()
                            log_off_spin.hide()
                            sqrt_lbl.hide()
                            sqrt_spin.hide()
                        elif index == 1:  # 对数变换
                            log_lbl.show()
                            log_base.show()
                            log_off_lbl.show()
                            log_off_spin.show()
                            sqrt_lbl.hide()
                            sqrt_spin.hide()
                        elif index == 2:  # 平方根变换
                            log_lbl.hide()
                            log_base.hide()
                            log_off_lbl.hide()
                            log_off_spin.hide()
                            sqrt_lbl.show()
                            sqrt_spin.show()
                    return update_transform_params
                
                # 为当前文件创建独立的更新函数
                update_transform_params = make_update_func(
                    log_params_label, log_base_combo, log_offset_label, log_offset_spin,
                    sqrt_params_label, sqrt_offset_spin
                )
                transform_combo.currentIndexChanged.connect(update_transform_params)
                
                h2.addWidget(transform_combo)
                h2.addWidget(params_widget)
                h2.addStretch(1)
                file_vbox.addLayout(h2)
                
                # 添加分隔线
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.HLine)
                separator.setFrameShadow(QFrame.Shadow.Sunken)
                file_vbox.addWidget(separator)
                
                self.dynamic_controls_layout.addWidget(file_widget)
                
                self.individual_control_widgets[base_name] = {
                    'scale': scale_sb,
                    'offset': offset_sb,
                    'color': color_input,  # 添加颜色控件
                    'transform': transform_combo,
                    'log_base': log_base_combo,
                    'log_offset': log_offset_spin,
                    'sqrt_offset': sqrt_offset_spin
                }
                
                # 连接颜色输入框的信号，颜色改变时自动更新图表
                color_input.textChanged.connect(self._on_file_color_changed)

            self.dynamic_controls_layout.addStretch(1)
            # 不再弹出“已加载独立控制项”的完成提示框
        except Exception as e:
            traceback.print_exc()
    
    def scan_and_load_group_waterfall_controls(self):
        """扫描组并为组瀑布图创建独立的堆叠位移控制"""
        try:
            folder_path = self.folder_input.text()
            if not os.path.isdir(folder_path):
                QMessageBox.warning(self, "警告", "请先设置数据文件夹路径")
                return
            
            # 获取分组参数
            n_chars = self.n_chars_spin.value()
            
            # 扫描文件并分组
            files = sorted(glob.glob(os.path.join(folder_path, '*.csv')) + glob.glob(os.path.join(folder_path, '*.txt')))
            if not files:
                # 不再弹出“未找到文件”提示框
                return
            
            groups = group_files_by_name(files, n_chars)
            
            # 筛选指定组
            target_gs = [x.strip() for x in self.groups_input.text().split(',') if x.strip()]
            if target_gs:
                groups = {k: v for k, v in groups.items() if k in target_gs}
            
            if not groups:
                QMessageBox.warning(self, "警告", "未找到有效的组")
                return
            
            # 清除旧的控件
            self.group_waterfall_control_widgets.clear()
            self._clear_layout_recursively(self.group_waterfall_controls_layout)
            
            # 获取全局默认偏移值（从谱线扫描面板）
            default_offset = self._get_stack_offset_from_panel()
            
            # 对组名进行排序
            sorted_group_names = sorted(groups.keys())
            
            # 为每组创建控制项
            for i, group_name in enumerate(sorted_group_names):
                group_widget = QWidget()
                group_vbox = QVBoxLayout(group_widget)
                group_vbox.setContentsMargins(5, 5, 5, 5)
                group_vbox.setSpacing(5)
                
                # 组名标签
                name_label = QLabel(f"{group_name} (共 {len(groups[group_name])} 个文件)")
                name_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
                group_vbox.addWidget(name_label)
                
                # 堆叠位移控制
                h_layout = QHBoxLayout()
                h_layout.addWidget(QLabel("堆叠位移:"))
                
                offset_sb = QDoubleSpinBox()
                offset_sb.setRange(-999999999.0, 999999999.0)
                offset_sb.setDecimals(15)
                offset_sb.setValue(default_offset * i)  # 使用默认偏移值乘以索引
                offset_sb.setToolTip("该组在瀑布图中的垂直堆叠位移值")
                
                h_layout.addWidget(offset_sb)
                h_layout.addStretch(1)
                group_vbox.addLayout(h_layout)
                
                # 颜色控制
                color_layout = QHBoxLayout()
                color_layout.addWidget(QLabel("颜色:"))
                
                # 使用默认颜色序列
                default_colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal', 'darkred']
                color_idx = i % len(default_colors)
                
                color_input = QLineEdit()
                color_input.setText(default_colors[color_idx])
                color_input.setToolTip("线条颜色（支持颜色名称如'red'、'blue'或十六进制如'#FF0000'）")
                color_input.setMaximumWidth(100)
                
                color_layout.addWidget(color_input)
                # 添加颜色选择器按钮
                color_button = self._create_color_picker_button(color_input)
                color_layout.addWidget(color_button)
                color_layout.addStretch(1)
                group_vbox.addLayout(color_layout)
                
                # 添加分隔线
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.HLine)
                separator.setFrameShadow(QFrame.Shadow.Sunken)
                group_vbox.addWidget(separator)
                
                self.group_waterfall_controls_layout.addWidget(group_widget)
                
                self.group_waterfall_control_widgets[group_name] = {
                    'offset': offset_sb,
                    'color': color_input  # 添加颜色控件
                }
                
                # 连接颜色输入框的信号，颜色改变时自动更新图表
                color_input.textChanged.connect(self._on_file_color_changed)
            
            self.group_waterfall_controls_layout.addStretch(1)
            # 不再弹出“已加载组控制”的完成提示框
        except Exception as e:
            QMessageBox.critical(self, "错误", f"扫描组时出错: {str(e)}")
            traceback.print_exc()
    
    def run_2d_cos_analysis(self):
        """运行2D-COS分析"""
        return self._run_2d_cos_analysis_internal()
    
    def _run_2d_cos_analysis_internal(self):
        # 由 COSPanelMixin 提供实现
        return COSPanelMixin._run_2d_cos_analysis_internal(self)
    
    def _rerun_nmf_plot_handler(self):
        """重新绘制NMF图的处理函数"""
        if hasattr(self, 'nmf_window') and self.nmf_window is not None and self.nmf_window.isVisible():
            # 如果NMF窗口存在且可见，重新运行NMF分析以更新图表
            QMessageBox.information(self, "提示", "请使用'运行 NMF 解混分析'按钮重新运行NMF分析以更新图表。")
        else:
            QMessageBox.warning(self, "警告", "请先运行NMF分析。")
    
    def _export_group_averages_internal(self):
        # 由 COSPanelMixin 提供实现
        return COSPanelMixin._export_group_averages_internal(self)
    
    def _create_nmf_component_controls(self, n_components, preserve_values=True):
        """为NMF组分创建独立Y轴控制项、预处理选项和图例重命名
        
        Args:
            n_components: 组分数量
            preserve_values: 如果为True，且控件已存在且组分数量相同，则保留现有值
        """
        # 检查是否已有控件且组分数量相同
        if preserve_values and hasattr(self, 'nmf_component_control_widgets') and len(self.nmf_component_control_widgets) == n_components:
            # 保留现有控件，不重新创建
            return
        
        # 保存现有值（如果存在）
        old_values = {}
        old_rename_values = {}
        if preserve_values and hasattr(self, 'nmf_component_control_widgets'):
            for comp_label, widgets in self.nmf_component_control_widgets.items():
                old_values[comp_label] = {
                    'scale': widgets['scale'].value(),
                    'offset': widgets['offset'].value(),
                    'transform': widgets['transform'].currentText(),
                    'log_base': widgets['log_base'].currentText(),
                    'log_offset': widgets['log_offset'].value(),
                    'sqrt_offset': widgets['sqrt_offset'].value()
                }
        if preserve_values and hasattr(self, 'nmf_component_rename_widgets'):
            for comp_label, rename_widget in self.nmf_component_rename_widgets.items():
                old_rename_values[comp_label] = rename_widget.text()
        
        # 清除旧的NMF组分控制项
        self.nmf_component_control_widgets.clear()
        self.nmf_component_rename_widgets.clear()
        self._clear_layout_recursively(self.nmf_component_controls_layout)
        
        # 为每个组分创建控制项
        for i in range(n_components):
            comp_label = f"Component {i+1}"
            
            # 创建文件控制容器（使用垂直布局，包含多行）
            comp_widget = QWidget()
            comp_vbox = QVBoxLayout(comp_widget)
            comp_vbox.setContentsMargins(5, 5, 5, 5)
            comp_vbox.setSpacing(5)
            
            # 组分名称标签
            name_label = QLabel(f"{comp_label}")
            name_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
            comp_vbox.addWidget(name_label)
            
            # 第一行：独立Y轴控制
            h1 = QHBoxLayout()
            h1.addWidget(QLabel("Y轴控制:"))
            
            scale_sb = QDoubleSpinBox()
            scale_sb.setRange(-999999999.0, 999999999.0)
            scale_sb.setDecimals(15)
            scale_sb.setValue(1.0)
            scale_sb.setToolTip("Y轴缩放因子")
            
            offset_sb = QDoubleSpinBox()
            offset_sb.setRange(-999999999.0, 999999999.0)
            offset_sb.setDecimals(15)
            offset_sb.setValue(0.0)
            offset_sb.setToolTip("Y轴偏移量")
            
            h1.addWidget(QLabel("Scale:"))
            h1.addWidget(scale_sb)
            h1.addWidget(QLabel("Offset:"))
            h1.addWidget(offset_sb)
            h1.addStretch(1)
            comp_vbox.addLayout(h1)
            
            # 第二行：预处理选项（对数/平方根变换）
            h2 = QHBoxLayout()
            h2.addWidget(QLabel("动态范围压缩:"))
            
            transform_combo = QComboBox()
            transform_combo.addItems(['无', '对数变换 (Log)', '平方根变换 (Sqrt)'])
            transform_combo.setCurrentText('无')
            transform_combo.setToolTip("压缩高强度信号动态范围，凸显微弱峰值")
            
            # 对数变换参数
            log_base_combo = QComboBox()
            log_base_combo.addItems(['10', 'e'])
            log_base_combo.setCurrentText('10')
            log_base_combo.setToolTip("对数底数")
            
            log_offset_spin = QDoubleSpinBox()
            log_offset_spin.setRange(-999999999.0, 999999999.0)
            log_offset_spin.setDecimals(15)
            log_offset_spin.setValue(1.0)
            log_offset_spin.setToolTip("对数变换偏移量")
            
            # 平方根变换参数
            sqrt_offset_spin = QDoubleSpinBox()
            sqrt_offset_spin.setRange(-999999999.0, 999999999.0)
            sqrt_offset_spin.setDecimals(15)
            sqrt_offset_spin.setValue(0.0)
            sqrt_offset_spin.setToolTip("平方根变换偏移量")
            
            # 参数容器（根据选择的变换类型显示/隐藏）
            params_widget = QWidget()
            params_layout = QHBoxLayout(params_widget)
            params_layout.setContentsMargins(0, 0, 0, 0)
            
            log_params_label = QLabel("底数:")
            log_params_label.hide()
            log_base_combo.hide()
            log_offset_label = QLabel("偏移:")
            log_offset_label.hide()
            log_offset_spin.hide()
            
            sqrt_params_label = QLabel("偏移:")
            sqrt_params_label.hide()
            sqrt_offset_spin.hide()
            
            params_layout.addWidget(log_params_label)
            params_layout.addWidget(log_base_combo)
            params_layout.addWidget(log_offset_label)
            params_layout.addWidget(log_offset_spin)
            params_layout.addWidget(sqrt_params_label)
            params_layout.addWidget(sqrt_offset_spin)
            params_layout.addStretch(1)
            
            # 使用lambda闭包确保每个组分的控件独立绑定
            def make_update_func(log_lbl, log_base, log_off_lbl, log_off_spin, sqrt_lbl, sqrt_spin):
                """创建更新函数，确保每个组分的控件独立"""
                def update_transform_params(index):
                    """根据选择的变换类型显示/隐藏相应参数"""
                    if index == 0:  # 无
                        log_lbl.hide()
                        log_base.hide()
                        log_off_lbl.hide()
                        log_off_spin.hide()
                        sqrt_lbl.hide()
                        sqrt_spin.hide()
                    elif index == 1:  # 对数变换
                        log_lbl.show()
                        log_base.show()
                        log_off_lbl.show()
                        log_off_spin.show()
                        sqrt_lbl.hide()
                        sqrt_spin.hide()
                    elif index == 2:  # 平方根变换
                        log_lbl.hide()
                        log_base.hide()
                        log_off_lbl.hide()
                        log_off_spin.hide()
                        sqrt_lbl.show()
                        sqrt_spin.show()
                return update_transform_params
            
            # 为当前组分创建独立的更新函数
            update_transform_params = make_update_func(
                log_params_label, log_base_combo, log_offset_label, log_offset_spin,
                sqrt_params_label, sqrt_offset_spin
            )
            transform_combo.currentIndexChanged.connect(update_transform_params)
            
            h2.addWidget(transform_combo)
            h2.addWidget(params_widget)
            h2.addStretch(1)
            comp_vbox.addLayout(h2)
            
            # 第三行：图例重命名
            h3 = QHBoxLayout()
            h3.addWidget(QLabel("图例名称:"))
            
            rename_input = QLineEdit(placeholderText="新的图例名称（留空则使用默认名称）")
            
            h3.addWidget(rename_input)
            h3.addStretch(1)
            comp_vbox.addLayout(h3)
            
            # 添加分隔线
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            comp_vbox.addWidget(separator)
            
            self.nmf_component_controls_layout.addWidget(comp_widget)
            
            self.nmf_component_control_widgets[comp_label] = {
                'scale': scale_sb,
                'offset': offset_sb,
                'transform': transform_combo,
                'log_base': log_base_combo,
                'log_offset': log_offset_spin,
                'sqrt_offset': sqrt_offset_spin
            }
            
            self.nmf_component_rename_widgets[comp_label] = rename_input
            
            # 恢复旧值（如果存在）
            if preserve_values and comp_label in old_values:
                old_val = old_values[comp_label]
                scale_sb.setValue(old_val['scale'])
                offset_sb.setValue(old_val['offset'])
                transform_combo.setCurrentText(old_val['transform'])
                log_base_combo.setCurrentText(old_val['log_base'])
                log_offset_spin.setValue(old_val['log_offset'])
                sqrt_offset_spin.setValue(old_val['sqrt_offset'])
                # 触发参数显示/隐藏更新
                transform_combo.currentIndexChanged.emit(transform_combo.currentIndex())
            
            if preserve_values and comp_label in old_rename_values:
                rename_input.setText(old_rename_values[comp_label])
        
        self.nmf_component_controls_layout.addStretch(1)

    def _get_stack_offset_from_panel(self):
        """从谱线扫描面板获取堆叠偏移值"""
        try:
            if hasattr(self, 'spectrum_scan_panel'):
                from src.core.plot_config_manager import PlotConfigManager
                config_manager = PlotConfigManager()
                config = config_manager.get_config()
                ss = config.spectrum_scan
                return ss.stack_offset if ss else 0.5
        except:
            pass
        return 0.5  # 默认值
    
    def _prepare_plot_params(self, grouped_files_data=None, control_data_list=None):
        """
        准备绘图参数（提取自run_plot_logic，用于避免重复代码）
        
        Args:
            grouped_files_data: 分组文件数据，如果为None则从文件夹读取
            control_data_list: 对照数据列表，如果为None则从输入读取
            
        Returns:
            参数字典，如果无法准备则返回None
        """
        try:
            folder = self.folder_input.text()
            if not os.path.isdir(folder):
                return None

            # 获取样式参数
            style_params = self._get_current_style_params()

            # 物理截断值
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())

            # 从面板获取配置（如果可用）
            config = None
            ps = None
            pm = None
            ss = None
            if hasattr(self, 'publication_style_panel'):
                config = self.publication_style_panel.get_config()
                ps = config.publication_style
            if hasattr(self, 'peak_matching_panel'):
                if config is None:
                    from src.core.plot_config_manager import PlotConfigManager
                    config_manager = PlotConfigManager()
                    config = config_manager.get_config()
                pm = config.peak_matching
            if hasattr(self, 'spectrum_scan_panel'):
                if config is None:
                    from src.core.plot_config_manager import PlotConfigManager
                    config_manager = PlotConfigManager()
                    config = config_manager.get_config()
                ss = config.spectrum_scan
            
            # 收集参数
            params = {
                # 模式与全局
                'plot_mode': self.plot_mode_combo.currentText(),
                # 从样式面板获取坐标轴显示控制
                'show_y_values': ps.show_y_values if ps else True,
                'show_x_values': ps.show_x_values if ps else True,
                'x_axis_invert': ps.x_axis_invert if ps else False,
                # 注意：二次导数已在预处理流程中应用，这里不再需要
                'is_derivative': False,  # 二次导数在预处理中应用
                # 堆叠偏移从谱线扫描面板获取（见下方）
                'global_scale_factor': self.global_y_scale_factor_spin.value(),
                'plot_style': self.plot_style_combo.currentText(), # 新增：绘制风格 
                
                
                # 预处理
                'qc_enabled': self.qc_check.isChecked(),
                'qc_threshold': self.qc_threshold_spin.value(),
                'is_baseline_als': self.baseline_als_check.isChecked(),
                'als_lam': self.lam_spin.value(),
                'als_p': self.p_spin.value(),
                'is_baseline': False, # 旧版基线默认关闭，以免冲突
                'is_baseline_poly': self.baseline_poly_check.isChecked() if hasattr(self, 'baseline_poly_check') else False,
                'baseline_points': self.baseline_points_spin.value() if hasattr(self, 'baseline_points_spin') else 50,
                'baseline_poly': self.baseline_poly_spin.value() if hasattr(self, 'baseline_poly_spin') else 3,
                'is_smoothing': self.smoothing_check.isChecked(),
                'smoothing_window': self.smoothing_window_spin.value(),
                'smoothing_poly': self.smoothing_poly_spin.value(),
                'normalization_mode': self.normalization_combo.currentText(),
                
                # Bose-Einstein
                'is_be_correction': self.be_check.isChecked(),
                'be_temp': self.be_temp_spin.value(),
                
                # 全局动态变换和整体Y轴偏移
                'global_transform_mode': self.global_transform_combo.currentText(),
                'global_log_base': self.global_log_base_combo.currentText(),
                'global_log_offset': self.global_log_offset_spin.value(),
                'global_sqrt_offset': self.global_sqrt_offset_spin.value(),
                'global_y_offset': self.global_y_offset_spin.value() if hasattr(self, 'global_y_offset_spin') else 0.0,
                
                # 注意：二次函数拟合已删除
                'is_quadratic_fit': False,
                'quadratic_degree': 2,
                
                # 高级/波峰检测（增强版）
                'peak_detection_enabled': self.peak_check.isChecked(),
                'peak_height_threshold': self.peak_height_spin.value(),
                'peak_distance_min': self.peak_distance_spin.value(),
                'peak_prominence': self.peak_prominence_spin.value(),
                'peak_width': self.peak_width_spin.value(),
                'peak_wlen': self.peak_wlen_spin.value(),
                'peak_rel_height': self.peak_rel_height_spin.value(),
                'peak_show_label': self.peak_show_label_check.isChecked(),
                'peak_label_font': self.peak_label_font_combo.currentText(),
                'peak_label_size': self.peak_label_size_spin.value(),
                'peak_label_color': self.peak_label_color_input.text().strip() or 'black',
                'peak_label_bold': self.peak_label_bold_check.isChecked(),
                'peak_label_rotation': self.peak_label_rotation_spin.value(),
                'peak_marker_shape': self.peak_marker_shape_combo.currentText(),
                'peak_marker_size': self.peak_marker_size_spin.value(),
                'peak_marker_color': self.peak_marker_color_input.text().strip() or '',  # 空字符串表示使用线条颜色
                'vertical_lines': self.parse_list_input(self.vertical_lines_input.toPlainText()),
                'vertical_line_color': self.vertical_line_color_input.text().strip() or 'gray',
                'vertical_line_width': self.vertical_line_width_spin.value(),
                'vertical_line_style': self.vertical_line_style_combo.currentText(),
                'vertical_line_alpha': self.vertical_line_alpha_spin.value(),
                # 匹配线样式参数（单独设置）
                'match_line_color': self.match_line_color_input.text().strip() or 'red' if hasattr(self, 'match_line_color_input') else 'red',
                'match_line_width': self.match_line_width_spin.value() if hasattr(self, 'match_line_width_spin') else 1.0,
                'match_line_style': self.match_line_style_combo.currentText() if hasattr(self, 'match_line_style_combo') else '-',
                'match_line_alpha': self.match_line_alpha_spin.value() if hasattr(self, 'match_line_alpha_spin') else 0.8,
                'rruff_ref_lines_enabled': self.rruff_ref_lines_enabled_check.isChecked() if hasattr(self, 'rruff_ref_lines_enabled_check') else True,
                
                # 出版质量样式（从面板获取）
                'fig_width': ps.fig_width if ps else 10.0,
                'fig_height': ps.fig_height if ps else 6.0,
                'fig_dpi': ps.fig_dpi if ps else 300,
                'font_family': ps.font_family if ps else 'Times New Roman',
                'axis_title_fontsize': ps.axis_title_fontsize if ps else 20,
                'tick_label_fontsize': ps.tick_label_fontsize if ps else 16,
                'legend_fontsize': ps.legend_fontsize if ps else 10,
                'line_width': ps.line_width if ps else 1.2,
                'line_style': ps.line_style if ps else '-',
                'tick_direction': ps.tick_direction if ps else 'in',
                'tick_len_major': ps.tick_len_major if ps else 8,
                'tick_len_minor': ps.tick_len_minor if ps else 4,
                'tick_width': ps.tick_width if ps else 1.0,
                'show_grid': ps.show_grid if ps else True,
                'grid_alpha': ps.grid_alpha if ps else 0.2,
                'shadow_alpha': ps.shadow_alpha if ps else 0.25,
                'show_legend': ps.show_legend if ps else True,
                'legend_frame': ps.legend_frame if ps else True,
                'legend_loc': ps.legend_loc if ps else 'best',
                'legend_ncol': ps.legend_ncol if ps else 1,
                'legend_columnspacing': ps.legend_columnspacing if ps else 2.0,
                'legend_labelspacing': ps.legend_labelspacing if ps else 0.5,
                'legend_handlelength': ps.legend_handlelength if ps else 2.0,
                'border_sides': self._get_border_sides_from_config(ps) if ps else self.get_checked_border_sides(),
                'border_linewidth': ps.spine_width if ps else 2.0,
                'aspect_ratio': ps.aspect_ratio if ps else 0.6,
                
                # 标题和轴标签（从面板获取）
                'xlabel_text': ps.xlabel_text if ps else r"Wavenumber ($\mathrm{cm^{-1}}$)",
                'xlabel_show': ps.xlabel_show if ps else True,
                'xlabel_fontsize': ps.xlabel_fontsize if ps else 20,
                'xlabel_pad': ps.xlabel_pad if ps else 10.0,
                'ylabel_text': ps.ylabel_text if ps else "Intensity",
                'ylabel_show': ps.ylabel_show if ps else True,
                'ylabel_fontsize': ps.ylabel_fontsize if ps else 20,
                'ylabel_pad': ps.ylabel_pad if ps else 10.0,
                'main_title_text': ps.title_text if ps else "",
                'main_title_show': ps.title_show if ps else True,
                'main_title_fontsize': ps.title_fontsize if ps else 18,
                'main_title_pad': ps.title_pad if ps else 10.0,
                
                # 峰值匹配参数（从面板获取）
                'peak_matching_enabled': pm.enabled if pm else False,
                'peak_matching_mode': pm.mode if pm else 'all_matched',
                'peak_matching_tolerance': pm.tolerance if pm else 5.0,
                'peak_matching_reference_index': pm.reference_index if pm else -1,
                # 峰值匹配样式参数
                'peak_matching_marker_shape': pm.marker_shape if pm else 'v',
                'peak_matching_marker_size': pm.marker_size if pm else 8.0,
                'peak_matching_marker_distance': pm.marker_distance if pm else 0.0,
                'peak_matching_marker_rotation': pm.marker_rotation if pm else 0.0,
                'peak_matching_show_connection_lines': pm.show_connection_lines if pm else False,
                'peak_matching_use_spectrum_color_for_connection': pm.use_spectrum_color_for_connection if pm else True,
                'peak_matching_connection_line_color': pm.connection_line_color if pm else 'red',
                'peak_matching_connection_line_width': pm.connection_line_width if pm else 1.0,
                'peak_matching_connection_line_style': pm.connection_line_style if pm else '-',
                'peak_matching_connection_line_alpha': pm.connection_line_alpha if pm else 0.8,
                'peak_matching_show_peak_labels': pm.show_peak_labels if pm else False,
                'peak_matching_label_fontsize': pm.label_fontsize if pm else 10.0,
                'peak_matching_label_color': pm.label_color if pm else 'black',
                'peak_matching_label_rotation': pm.label_rotation if pm else 0.0,
                'peak_matching_label_distance': pm.label_distance if pm else 5.0,
                
                # 谱线扫描参数（从面板获取）
                'spectrum_scan_enabled': ss.enabled if ss else False,
                'stack_offset': ss.stack_offset if ss else 0.5,
                'global_stack_offset': ss.stack_offset if ss else 0.5,  # 使用谱线扫描的堆叠偏移
                'individual_offsets': ss.individual_offsets if ss else {},
                'custom_mappings': ss.custom_mappings if ss else [],
                
                # 从样式面板获取坐标轴显示控制
                'show_x_values': ps.show_x_values if ps else True,
                'show_y_values': ps.show_y_values if ps else True,
                'x_axis_invert': ps.x_axis_invert if ps else False,
            }
            
            # 读取独立控件值（包括颜色）
            ind_params = {}
            group_colors = {}  # 存储组颜色（用于Mean + Shadow模式）
            for k, v in self.individual_control_widgets.items():
                transform_type = v['transform'].currentText()
                transform_mode = 'none'
                transform_params = {}
                
                if transform_type == '对数变换 (Log)':
                    transform_mode = 'log'
                    transform_params = {
                        'base': float(v['log_base'].currentText()) if v['log_base'].currentText() == '10' else np.e,
                        'offset': v['log_offset'].value()
                    }
                elif transform_type == '平方根变换 (Sqrt)':
                    transform_mode = 'sqrt'
                    transform_params = {
                        'offset': v['sqrt_offset'].value()
                    }
                
                ind_params[k] = {
                    'scale': v['scale'].value(),
                    'offset': v['offset'].value(),
                    'color': v.get('color', None),  # 添加颜色信息
                    'transform': transform_mode,
                    'transform_params': transform_params
                }
                
                # 收集组颜色（用于Mean + Shadow模式）
                # 从文件名提取组名（使用分组前缀长度）
                n_chars = self.n_chars_spin.value()
                if n_chars > 0:
                    group_name = k[:n_chars] if len(k) >= n_chars else k
                else:
                    group_name = k  # 使用完整文件名作为组名
                
                # 如果该组还没有颜色，使用当前文件的颜色
                if group_name not in group_colors:
                    color_text = v.get('color', None)
                    if color_text and hasattr(color_text, 'text'):
                        color_value = color_text.text().strip() or None
                        if color_value:
                            group_colors[group_name] = color_value
            
            params['individual_y_params'] = ind_params
            params['group_colors'] = group_colors  # 传递组颜色
            
            # 构建文件颜色映射（用于绘图时获取颜色）
            file_colors = {}
            for k, v in self.individual_control_widgets.items():
                color_widget = v.get('color')
                if color_widget and hasattr(color_widget, 'text'):
                    color_text = color_widget.text().strip()
                    if color_text:
                        file_colors[k] = color_text
            params['file_colors'] = file_colors
            
            # 读取重命名
            rename_map = {k: v.text().strip() for k, v in self.legend_rename_widgets.items() if v.text().strip()}
            params['legend_names'] = rename_map

            # 如果提供了grouped_files_data和control_data_list，直接使用
            if grouped_files_data is not None:
                params['grouped_files_data'] = grouped_files_data
            if control_data_list is not None:
                params['control_data_list'] = control_data_list
            
            # 如果两者都没有提供，需要读取文件
            if grouped_files_data is None or control_data_list is None:
                # 读取文件列表
                skip = self.skip_rows_spin.value()
                all_files = sorted(glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt')))
                
                if not all_files:
                    print(f"警告：在文件夹 {folder} 中未找到任何 .csv 或 .txt 文件")
                    # 不设置grouped_files_data，让调用者知道没有文件
                    if grouped_files_data is None:
                        params['grouped_files_data'] = []
                    if control_data_list is None:
                        params['control_data_list'] = []
                    return params
                
                # 提取对照文件（自动识别后缀）
                c_text = self.control_files_input.toPlainText()
                c_names = [x.strip() for x in c_text.replace('\n', ',').split(',') if x.strip()]
                
                if control_data_list is None:
                    control_data_list = []
                    files_to_remove = []
                    for c_name_base in c_names:
                        # 自动识别后缀（.txt, .csv等）
                        found_file = None
                        for ext in ['.txt', '.csv', '.TXT', '.CSV']:
                            c_name = c_name_base + ext if not c_name_base.endswith(ext) else c_name_base
                            full_p = os.path.join(folder, c_name)
                            if full_p in all_files:
                                found_file = full_p
                                break
                        
                        if found_file:
                            try:
                                x, y = self.read_data(found_file, skip, x_min_phys, x_max_phys) # 使用物理截断
                                control_data_list.append({
                                    'df': pd.DataFrame({'Wavenumber': x, 'Intensity': y}),
                                    'label': rename_map.get(os.path.splitext(os.path.basename(found_file))[0], os.path.splitext(os.path.basename(found_file))[0]),
                                    'filename': os.path.basename(found_file)
                                })
                                files_to_remove.append(found_file)
                            except ValueError as ve:
                                QMessageBox.warning(self, "警告", f"对照文件 {c_name_base} 读取失败: {ve}")
                            except: pass
                        else:
                            QMessageBox.warning(self, "警告", f"对照文件 {c_name_base} 未找到（已尝试 .txt 和 .csv 后缀）")
                    
                    params['control_data_list'] = control_data_list
                else:
                    # 如果control_data_list已提供，需要找出哪些文件是对照文件
                    files_to_remove = []
                    for c_name_base in c_names:
                        for ext in ['.txt', '.csv', '.TXT', '.CSV']:
                            c_name = c_name_base + ext if not c_name_base.endswith(ext) else c_name_base
                            full_p = os.path.join(folder, c_name)
                            if full_p in all_files:
                                files_to_remove.append(full_p)
                                break
                
                # 读取分组文件数据（如果未提供）
                if grouped_files_data is None:
                    plot_files = [f for f in all_files if f not in files_to_remove]
                    
                    # 分组
                    n_chars = self.n_chars_spin.value()
                    groups = group_files_by_name(plot_files, n_chars)
                    
                    # 筛选组别
                    target_g_text = self.groups_input.text()
                    target_gs = [x.strip() for x in target_g_text.split(',') if x.strip()]
                    if target_gs:
                        groups = {k: v for k, v in groups.items() if k in target_gs}
                    
                    # 读取第一个组的数据（用于自动更新）
                    # 注意：自动更新时通常只更新当前活动窗口对应的组
                    if groups:
                        # 获取第一个组的数据（如果活动窗口有组名，优先使用该组）
                        g_name = None
                        if self.active_plot_window and hasattr(self.active_plot_window, 'group_name'):
                            g_name = self.active_plot_window.group_name
                            if g_name not in groups:
                                g_name = None
                        
                        if g_name is None:
                            g_name = list(groups.keys())[0]
                        
                        g_files = groups[g_name]
                        g_data = []
                        for f in g_files:
                            try:
                                x, y = self.read_data(f, skip, x_min_phys, x_max_phys)
                                g_data.append((f, x, y))
                            except ValueError as ve:
                                QMessageBox.warning(self, "警告", f"文件 {os.path.basename(f)} 读取失败: {ve}")
                            except: pass
                        
                        params['grouped_files_data'] = g_data
                    else:
                        # 如果没有找到匹配的组，设置为空列表
                        params['grouped_files_data'] = []
            
            return params
        except Exception as e:
            print(f"准备绘图参数失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # --- 核心：运行绘图逻辑 ---
    def run_plot_logic(self):
        try:
            # 延迟设置字体（首次绘图时）
            self._ensure_fonts_setup()
            
            folder = self.folder_input.text()
            if not os.path.isdir(folder): return
            
            # 物理截断值
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())
            
            # 从面板获取配置（如果可用）
            config = None
            ps = None
            pm = None
            ss = None
            if hasattr(self, 'publication_style_panel'):
                config = self.publication_style_panel.get_config()
                ps = config.publication_style
            if hasattr(self, 'peak_matching_panel'):
                if config is None:
                    from src.core.plot_config_manager import PlotConfigManager
                    config_manager = PlotConfigManager()
                    config = config_manager.get_config()
                pm = config.peak_matching
            if hasattr(self, 'spectrum_scan_panel'):
                if config is None:
                    from src.core.plot_config_manager import PlotConfigManager
                    config_manager = PlotConfigManager()
                    config = config_manager.get_config()
                ss = config.spectrum_scan
            
            # 收集参数（复用_prepare_plot_params的逻辑）
            # 注意：run_plot_logic需要读取所有组的数据，所以不传入grouped_files_data
            params = self._prepare_plot_params(grouped_files_data=None, control_data_list=None)
            if params is None:
                return
            
            # 读取文件列表和分组（run_plot_logic总是需要重新读取所有组的数据）
            if True:  # 总是重新读取，确保读取所有组
                skip = self.skip_rows_spin.value()
                all_files = sorted(glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt')))
                
                # 提取对照文件（自动识别后缀）
                c_text = self.control_files_input.toPlainText()
                c_names = [x.strip() for x in c_text.replace('\n', ',').split(',') if x.strip()]
                
                control_data_list = []
                files_to_remove = []
                rename_map = params.get('legend_names', {})
                
                for c_name_base in c_names:
                    # 自动识别后缀（.txt, .csv等）
                    found_file = None
                    for ext in ['.txt', '.csv', '.TXT', '.CSV']:
                        c_name = c_name_base + ext if not c_name_base.endswith(ext) else c_name_base
                        full_p = os.path.join(folder, c_name)
                        if full_p in all_files:
                            found_file = full_p
                            break
                    
                    if found_file:
                        try:
                            x, y = self.read_data(found_file, skip, x_min_phys, x_max_phys)
                            control_data_list.append({
                                'df': pd.DataFrame({'Wavenumber': x, 'Intensity': y}),
                                'label': rename_map.get(os.path.splitext(os.path.basename(found_file))[0], os.path.splitext(os.path.basename(found_file))[0]),
                                'filename': os.path.basename(found_file)
                            })
                            files_to_remove.append(found_file)
                        except ValueError as ve:
                            QMessageBox.warning(self, "警告", f"对照文件 {c_name_base} 读取失败: {ve}")
                        except: pass
                    else:
                        QMessageBox.warning(self, "警告", f"对照文件 {c_name_base} 未找到（已尝试 .txt 和 .csv 后缀）")
                
                plot_files = [f for f in all_files if f not in files_to_remove]
                params['control_data_list'] = control_data_list
                
                # 分组
                n_chars = self.n_chars_spin.value()
                groups = group_files_by_name(plot_files, n_chars)
                
                # 筛选组别
                target_g_text = self.groups_input.text()
                target_gs = [x.strip() for x in target_g_text.split(',') if x.strip()]
                if target_gs:
                    groups = {k: v for k, v in groups.items() if k in target_gs}
                
                if not groups and not control_data_list:
                    QMessageBox.warning(self, "警告", "无数据可绘图")
                    return
                
                # 遍历组并绘图
                for g_name, g_files in groups.items():
                    g_data = []
                    for f in g_files:
                        try:
                            x, y = self.read_data(f, skip, x_min_phys, x_max_phys)
                            g_data.append((f, x, y))
                        except ValueError as ve:
                             QMessageBox.warning(self, "警告", f"文件 {os.path.basename(f)} 读取失败: {ve}")
                        except: pass
                    
                    params['grouped_files_data'] = g_data
                    
                    # 添加RRUFF光谱数据（如果已选中）
                    params['rruff_spectra'] = []
                    params['rruff_match_results'] = []
                    if self.rruff_loader and g_name in self.selected_rruff_spectra:
                        for rruff_name in self.selected_rruff_spectra[g_name]:
                            rruff_data = self.rruff_loader.get_spectrum(rruff_name)
                            if rruff_data:
                                # 找到对应的匹配结果
                                match_result = None
                                if g_name in self.rruff_match_results:
                                    for match in self.rruff_match_results[g_name]:
                                        if match['name'] == rruff_name:
                                            match_result = match
                                            break
                                params['rruff_spectra'].append({
                                    'name': rruff_name,
                                    'x': rruff_data['x'],
                                    'y': rruff_data['y'],
                                    'matches': match_result['matches'] if match_result else []
                                })
                                if match_result:
                                    params['rruff_match_results'].append(match_result)
                    
                    if g_name not in self.plot_windows:
                        # 创建新窗口（不指定位置，让窗口自动计算远离主菜单的位置）
                        self.plot_windows[g_name] = MplPlotWindow(g_name, initial_geometry=None, parent=self)
                    
                    win = self.plot_windows[g_name]
                    # 更新活动绘图窗口引用
                    self.active_plot_window = win
                    # 更新绘图（会自动保持窗口位置和大小）
                    win.update_plot(params)
                    # 确保窗口显示
                    if not win.isVisible():
                        win.show()
                    
                    # 记录当前激活的绘图窗口
                    self.active_plot_window = win
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            traceback.print_exc()

    def _get_border_sides_from_config(self, ps):
        """从配置获取边框设置"""
        if ps is None:
            return self.get_checked_border_sides()
        sides = []
        if ps.spine_top:
            sides.append('top')
        if ps.spine_bottom:
            sides.append('bottom')
        if ps.spine_left:
            sides.append('left')
        if ps.spine_right:
            sides.append('right')
        return sides
    
    def get_checked_border_sides(self):
        # 收集边框可见性 - 优先从 publication_style_panel 获取
        if hasattr(self, 'publication_style_panel') and self.publication_style_panel:
            config = self.publication_style_panel.get_config()
            ps = config.publication_style
            sides = []
            if ps.spine_top: sides.append('top')
            if ps.spine_bottom: sides.append('bottom')
            if ps.spine_left: sides.append('left')
            if ps.spine_right: sides.append('right')
            return sides
        
        # 回退：如果面板不存在，尝试直接访问控件（向后兼容）
        sides = []
        if hasattr(self, 'spine_top_check') and self.spine_top_check.isChecked(): 
            sides.append('top')
        if hasattr(self, 'spine_bottom_check') and self.spine_bottom_check.isChecked(): 
            sides.append('bottom')
        if hasattr(self, 'spine_left_check') and self.spine_left_check.isChecked(): 
            sides.append('left')
        if hasattr(self, 'spine_right_check') and self.spine_right_check.isChecked(): 
            sides.append('right')
        
        # 如果都没有，返回默认值（全部显示）
        if not sides:
            return ['top', 'bottom', 'left', 'right']
        return sides
        
    # --- 核心：NMF 分析 ---
    def _on_seed_changed(self):
        """当随机种子改变时，如果使用Deep Autoencoder且已运行过NMF，自动重新运行"""
        # 检查是否使用Deep Autoencoder
        if hasattr(self, 'nmf_filter_algo_combo'):
            filter_algorithm = self.nmf_filter_algo_combo.currentText()
            if filter_algorithm == 'Deep Autoencoder (PyTorch)':
                # 检查是否有文件夹路径（说明已经设置过）
                if hasattr(self, 'folder_input') and self.folder_input.text().strip():
                    # 延迟执行，避免滚轮快速滚动时频繁触发
                    if not hasattr(self, '_seed_change_timer'):
                        self._seed_change_timer = QTimer()
                        self._seed_change_timer.setSingleShot(True)
                        self._seed_change_timer.timeout.connect(self._auto_rerun_nmf)
                    self._seed_change_timer.stop()  # 停止之前的计时器
                    self._seed_change_timer.start(500)  # 500ms延迟后执行
    
    def _auto_rerun_nmf(self):
        """自动重新运行NMF（当种子改变时）"""
        try:
            # 检查是否在标准NMF模式且已设置文件夹
            if (hasattr(self, 'nmf_mode_standard') and self.nmf_mode_standard.isChecked() and
                hasattr(self, 'folder_input') and self.folder_input.text().strip()):
                self.run_nmf_analysis()
        except Exception as e:
            # 如果出错，不显示错误（避免干扰用户）
            pass
    
    def run_nmf_button_handler(self):
        """
        处理NMF按钮点击事件，根据单选按钮状态调用标准NMF或组分回归模式
        """
        # 检查运行模式
        if self.nmf_mode_regression.isChecked():
            # 组分回归模式：使用固定的H矩阵
            if self.last_fixed_H is None:
                QMessageBox.warning(self, "NMF 警告", "请先运行标准NMF分析以获取固定的H矩阵。")
                return
            
            # 调用组分回归函数
            self.run_nmf_regression_mode()
        else:
            # 标准NMF模式
            self.run_nmf_analysis()

    def _run_nmf_regression_mode_legacy(self):
        """
        组分回归模式的完整流程：收集文件、调用run_nmf_regression、显示结果
        """
        try:
            folder = self.folder_input.text()
            files = glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt'))
            if not files:
                QMessageBox.warning(self, "NMF 警告", "未找到数据文件。")
                return
            
            # 获取预览列表中保留的文件（排除已删除的文件）
            included_files = set()
            for i in range(self.nmf_file_preview_list.count()):
                item = self.nmf_file_preview_list.item(i)
                if item and item.data(256):
                    included_files.add(item.data(256))
            
            # 如果预览列表为空，则包含所有文件；否则只包含预览列表中的文件
            if included_files:
                files = [f for f in files if f in included_files]
            
            # 处理对照组（如果设置了）
            control_files_to_exclude = []
            control_data_list = []
            if hasattr(self, 'control_files_input'):
                c_text = self.control_files_input.toPlainText()
                c_names = [x.strip() for x in c_text.replace('\n', ',').split(',') if x.strip()]
                
                for c_name_base in c_names:
                    # 自动识别后缀
                    found_file = None
                    for ext in ['.txt', '.csv', '.TXT', '.CSV']:
                        c_name = c_name_base + ext if not c_name_base.endswith(ext) else c_name_base
                        full_p = os.path.join(folder, c_name)
                        if full_p in files:
                            found_file = full_p
                            break
                    
                    if found_file:
                        # 如果选项是"不参与NMF"，则从NMF分析中排除
                        if not (hasattr(self, 'nmf_include_control_check') and self.nmf_include_control_check.isChecked()):
                            control_files_to_exclude.append(found_file)
                        else:
                            # 如果参与NMF，则添加到数据收集列表
                            control_data_list.append(found_file)
            
            # 排除对照组文件（如果它们不参与NMF）
            files = [f for f in files if f not in control_files_to_exclude]
            
            # 应用文件排序
            files = self._apply_nmf_file_sort(files)
            
            # 收集对照组数据（用于绘图，但不参与NMF）
            control_data_for_plot = []
            skip = self.skip_rows_spin.value()
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())
            
            for c_file in control_files_to_exclude:
                try:
                    x, y = self.read_data(c_file, skip, x_min_phys, x_max_phys)
                    # 应用预处理（与NMF数据一致，使用主菜单的所有预处理参数）
                    y_proc = y.astype(float)
                    
                    # 1. QC 检查（如果启用）
                    if self.qc_check.isChecked() and np.max(y_proc) < self.qc_threshold_spin.value():
                        continue
                    
                    # 2. BE 校正（如果启用）
                    if self.be_check.isChecked():
                        y_proc = DataPreProcessor.apply_bose_einstein_correction(x, y_proc, self.be_temp_spin.value())
                    
                    # 3. 平滑（如果启用）
                    if self.smoothing_check.isChecked():
                        y_proc = DataPreProcessor.apply_smoothing(y_proc, self.smoothing_window_spin.value(), self.smoothing_poly_spin.value())
                    
                    # 4. AsLS 基线校正（如果启用）
                    if self.baseline_als_check.isChecked():
                        b = DataPreProcessor.apply_baseline_als(y_proc, self.lam_spin.value(), self.p_spin.value())
                        y_proc = y_proc - b
                        y_proc[y_proc < 0] = 0
                    elif self.baseline_poly_check.isChecked():
                        y_proc = DataPreProcessor.apply_baseline_correction(x, y_proc, self.baseline_points_spin.value(), self.baseline_poly_spin.value())
                    
                    # 5. 归一化（如果启用）
                    normalization_mode = self.normalization_combo.currentText()
                    if normalization_mode == 'max':
                        y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
                    elif normalization_mode == 'area':
                        y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
                    elif normalization_mode == 'snv':
                        y_proc = DataPreProcessor.apply_snv(y_proc)
                    y_proc[y_proc < 0] = 0
                    
                    # 6. 全局动态范围压缩（如果启用）- 在归一化之后
                    global_transform_mode = self.global_transform_combo.currentText()
                    if global_transform_mode == '对数变换 (Log)':
                        base = float(self.global_log_base_combo.currentText()) if self.global_log_base_combo.currentText() == '10' else np.e
                        y_proc = DataPreProcessor.apply_log_transform(y_proc, base=base, offset=self.global_log_offset_spin.value())
                    elif global_transform_mode == '平方根变换 (Sqrt)':
                        y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=self.global_sqrt_offset_spin.value())
                    
                    # 7. 二次导数（如果启用）- 在全局动态变换之后
                    # 注意：二次导数在预处理流程中应用，这里不再需要
                    # if False:  # 二次导数已在预处理中应用
                    #     d1 = np.gradient(y_proc, x)
                    #     y_proc = np.gradient(d1, x)
                    
                    # 8. 整体Y轴偏移（预处理最后一步，在二次导数之后）
                    global_y_offset = self.global_y_offset_spin.value() if hasattr(self, 'global_y_offset_spin') else 0.0
                    y_proc = y_proc + global_y_offset
                    
                    control_data_for_plot.append({
                        'x': x,
                        'y': y_proc,
                        'label': os.path.splitext(os.path.basename(c_file))[0]
                    })
                except:
                    pass
            
            # 收集数据（包括参与NMF的对照组）
            all_nmf_files = files + control_data_list
            
            # 调用run_nmf_regression函数
            W, H, common_x, sample_labels = self.run_nmf_regression(all_nmf_files, self.last_fixed_H)
            
            if W is None or H is None or common_x is None:
                return
            
            # 为NMF组分创建独立Y轴控制项（如果还没有创建，保留现有值）
            n_components = H.shape[0]
            self._create_nmf_component_controls(n_components, preserve_values=True)
            
            # 收集独立Y轴参数和预处理选项（用于NMF组分绘图）
            individual_y_params = {}
            if hasattr(self, 'nmf_component_control_widgets'):
                for comp_label, widgets in self.nmf_component_control_widgets.items():
                    transform_type = widgets['transform'].currentText()
                    transform_mode = 'none'
                    transform_params = {}
                    
                    if transform_type == '对数变换 (Log)':
                        transform_mode = 'log'
                        transform_params = {
                            'base': float(widgets['log_base'].currentText()) if widgets['log_base'].currentText() == '10' else np.e,
                            'offset': widgets['log_offset'].value()
                        }
                    elif transform_type == '平方根变换 (Sqrt)':
                        transform_mode = 'sqrt'
                        transform_params = {
                            'offset': widgets['sqrt_offset'].value()
                        }
                    
                    individual_y_params[comp_label] = {
                        'scale': widgets['scale'].value(),
                        'offset': widgets['offset'].value(),
                        'transform': transform_mode,
                        'transform_params': transform_params
                    }
            
            # 收集NMF组分图例重命名
            # 从主窗口的legend_rename_widgets获取NMF图例重命名
            nmf_legend_names = {}
            # 首先从NMF组件重命名控件获取
            if hasattr(self, 'nmf_component_rename_widgets'):
                for comp_label, rename_widget in self.nmf_component_rename_widgets.items():
                    new_name = rename_widget.text().strip()
                    if new_name:
                        nmf_legend_names[comp_label] = new_name
            # 然后从主窗口的legend_rename_widgets获取（优先级更高）
            if hasattr(self, 'legend_rename_widgets'):
                for key, widget in self.legend_rename_widgets.items():
                    if hasattr(widget, 'text'):
                        renamed = widget.text().strip()
                        if renamed and key.startswith('NMF Component'):
                            # 提取组件编号
                            comp_num = key.replace('NMF Component ', '')
                            comp_label = f"Component {comp_num}"
                            nmf_legend_names[comp_label] = renamed
            
            # 为对照组数据添加独立Y轴参数（如果存在）
            for ctrl_data in control_data_for_plot:
                ctrl_label = ctrl_data['label']
                # 检查组回归模式中是否有对应的独立Y轴控制项
                if hasattr(self, 'individual_control_widgets') and ctrl_label in self.individual_control_widgets:
                    widgets = self.individual_control_widgets[ctrl_label]
                    individual_y_params[ctrl_label] = {
                        'scale': widgets['scale'].value(),
                        'offset': widgets['offset'].value(),
                        'transform': 'none',  # 对照组不使用变换
                        'transform_params': {}
                    }
            
            # 获取垂直参考线参数（从主菜单）
            vertical_lines = []
            if hasattr(self, 'vertical_lines_input'):
                vlines_text = self.vertical_lines_input.toPlainText().strip()
                if vlines_text:
                    try:
                        import re
                        vlines_str = re.split(r'[,;\s\n]+', vlines_text)
                        vertical_lines = [float(x.strip()) for x in vlines_str if x.strip()]
                    except:
                        pass

            # 获取样式参数
            style_params = self._get_current_style_params()

            # 收集 NMF 业务参数（不包含主窗口的样式参数，让窗口使用自己的默认设置）
            nmf_style_params = {
                # NMF特定业务参数
                'comp1_color': self.comp1_color_input.text().strip() if self.comp1_color_input.text().strip() else 'blue',
                'comp2_color': self.comp2_color_input.text().strip() if self.comp2_color_input.text().strip() else 'red',
                'comp_line_width': self.nmf_comp_line_width.value(),
                'comp_line_style': self.nmf_comp_line_style.currentText(),
                'weight_line_width': self.nmf_weight_line_width.value(),
                'weight_line_style': self.nmf_weight_line_style.currentText(),
                'weight_marker_size': getattr(self, 'nmf_marker_size', QSpinBox()).value() if hasattr(self, 'nmf_marker_size') else 8,
                'weight_marker_style': getattr(self, 'nmf_marker_style', QComboBox()).currentText() if hasattr(self, 'nmf_marker_style') else 'o',
                'title_font_size': self.nmf_title_font_spin.value(),
                'label_font_size': self.nmf_title_font_spin.value() - 2,
                'tick_font_size': self.nmf_tick_font_spin.value(),
                'legend_font_size': self.nmf_tick_font_spin.value() + 2,
                'x_axis_invert': style_params.get('x_axis_invert', False),
                'peak_detection_enabled': self.peak_check.isChecked(),
                'nmf_top_title': self.nmf_top_title_input.text().strip(),
                'nmf_bottom_title': self.nmf_bottom_title_input.text().strip(),
                'nmf_top_title_fontsize': self.nmf_top_title_font_spin.value(),
                'nmf_top_title_pad': self.nmf_top_title_pad_spin.value(),
                'nmf_top_title_show': self.nmf_top_title_show_check.isChecked(),
                'nmf_bottom_title_fontsize': self.nmf_bottom_title_font_spin.value(),
                'nmf_bottom_title_pad': self.nmf_bottom_title_pad_spin.value(),
                'nmf_bottom_title_show': self.nmf_bottom_title_show_check.isChecked(),
                'nmf_top_xlabel': self.nmf_xlabel_top_input.text().strip(),
                'nmf_top_xlabel_fontsize': self.nmf_top_xlabel_font_spin.value(),
                'nmf_top_xlabel_pad': self.nmf_top_xlabel_pad_spin.value(),
                'nmf_top_xlabel_show': self.nmf_top_xlabel_show_check.isChecked(),
                'nmf_top_ylabel': self.nmf_ylabel_top_input.text().strip(),
                'nmf_top_ylabel_fontsize': self.nmf_top_ylabel_font_spin.value(),
                'nmf_top_ylabel_pad': self.nmf_top_ylabel_pad_spin.value(),
                'nmf_top_ylabel_show': self.nmf_top_ylabel_show_check.isChecked(),
                'nmf_bottom_xlabel': self.nmf_xlabel_bottom_input.text().strip(),
                'nmf_bottom_xlabel_fontsize': self.nmf_bottom_xlabel_font_spin.value(),
                'nmf_bottom_xlabel_pad': self.nmf_bottom_xlabel_pad_spin.value(),
                'nmf_bottom_xlabel_show': self.nmf_bottom_xlabel_show_check.isChecked(),
                'nmf_bottom_ylabel': self.nmf_ylabel_bottom_input.text().strip(),
                'nmf_bottom_ylabel_fontsize': self.nmf_bottom_ylabel_font_spin.value(),
                'nmf_bottom_ylabel_pad': self.nmf_bottom_ylabel_pad_spin.value(),
                'nmf_bottom_ylabel_show': self.nmf_bottom_ylabel_show_check.isChecked(),
                'is_derivative': False,  # 二次导数在预处理流程中应用
                'global_stack_offset': self._get_stack_offset_from_panel(),
                'global_scale_factor': self.global_y_scale_factor_spin.value(),
                'individual_y_params': individual_y_params,
                'nmf_legend_names': nmf_legend_names,
                'control_data_list': control_data_for_plot,
                # 添加主菜单的出版质量样式控制参数
                'font_family': self.font_family_combo.currentText(),
                'axis_title_fontsize': self.axis_title_font_spin.value(),
                'tick_label_fontsize': self.tick_label_font_spin.value(),
                'legend_fontsize': self.legend_font_spin.value(),
                'line_width': self.line_width_spin.value(),
                'line_style': self.line_style_combo.currentText(),
                'tick_direction': self.tick_direction_combo.currentText(),
                'tick_len_major': self.tick_len_major_spin.value(),
                'tick_len_minor': self.tick_len_minor_spin.value(),
                'tick_width': self.tick_width_spin.value(),
                'show_grid': self.show_grid_check.isChecked(),
                'grid_alpha': self.grid_alpha_spin.value(),
                'spine_top': self.spine_top_check.isChecked(),
                'spine_bottom': self.spine_bottom_check.isChecked(),
                'spine_left': self.spine_left_check.isChecked(),
                'spine_right': self.spine_right_check.isChecked(),
                'spine_width': self.spine_width_spin.value(),
                'show_legend': self.show_legend_check.isChecked(),
                'legend_frame': self.legend_frame_check.isChecked(),
                'legend_loc': self.legend_loc_combo.currentText(),
                'legend_ncol': self.legend_column_spin.value() if hasattr(self, 'legend_column_spin') else 1,
                'legend_columnspacing': self.legend_columnspacing_spin.value() if hasattr(self, 'legend_columnspacing_spin') else 2.0,
                'legend_labelspacing': self.legend_labelspacing_spin.value() if hasattr(self, 'legend_labelspacing_spin') else 0.5,
                'legend_handlelength': self.legend_handlelength_spin.value() if hasattr(self, 'legend_handlelength_spin') else 2.0,
                'aspect_ratio': self.aspect_ratio_spin.value(),  # 纵横比
                'vertical_lines': vertical_lines,  # 垂直参考线
                'vertical_line_color': '#034DFB',  # 默认蓝色
                'vertical_line_style': '--',  # 默认虚线
                'vertical_line_width': 0.8,  # 默认线宽
                'vertical_line_alpha': 0.8,  # 默认透明度
            }
            
            # 准备 NMF 结果窗口
            if hasattr(self, 'nmf_window') and self.nmf_window is not None and self.nmf_window.isVisible():
                self.nmf_window.set_data(W, H, common_x, nmf_style_params, sample_labels)
                # 恢复之前选择的目标组分索引
                if hasattr(self.nmf_window, 'target_component_index'):
                    self.nmf_window.target_component_index = self.nmf_target_component_index
                    self.nmf_window._update_target_component_radios()
                self.nmf_window.raise_()
            else:
                win = NMFResultWindow("NMF Analysis Result (Component Regression)", self)
                win.target_component_index = self.nmf_target_component_index  # 设置初始选择
                win.set_data(W, H, common_x, nmf_style_params, sample_labels)
                self.nmf_window = win
                win.show()
            
        except Exception as e:
            QMessageBox.critical(self, "NMF-CR Error", f"非负组分回归运行失败: {str(e)}")
            traceback.print_exc()

    def _run_nmf_analysis_legacy(self):
        import numpy as np  # 确保在方法内部导入
        try:
            folder = self.folder_input.text()
            files = glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt'))
            if not files: return
            
            # 获取预览列表中保留的文件（排除已删除的文件）
            included_files = set()
            for i in range(self.nmf_file_preview_list.count()):
                item = self.nmf_file_preview_list.item(i)
                if item and item.data(256):
                    included_files.add(item.data(256))
            
            # 如果预览列表为空，则包含所有文件；否则只包含预览列表中的文件
            if included_files:
                files = [f for f in files if f in included_files]
            
            # 处理对照组（如果设置了）
            control_files_to_exclude = []
            control_data_list = []
            if hasattr(self, 'control_files_input'):
                c_text = self.control_files_input.toPlainText()
                c_names = [x.strip() for x in c_text.replace('\n', ',').split(',') if x.strip()]
                
                for c_name_base in c_names:
                    # 自动识别后缀
                    found_file = None
                    for ext in ['.txt', '.csv', '.TXT', '.CSV']:
                        c_name = c_name_base + ext if not c_name_base.endswith(ext) else c_name_base
                        full_p = os.path.join(folder, c_name)
                        if full_p in files:
                            found_file = full_p
                            break
                    
                    if found_file:
                        # 如果选项是"不参与NMF"，则从NMF分析中排除
                        if not (hasattr(self, 'nmf_include_control_check') and self.nmf_include_control_check.isChecked()):
                            control_files_to_exclude.append(found_file)
                        else:
                            # 如果参与NMF，则添加到数据收集列表
                            control_data_list.append(found_file)
            
            # 排除对照组文件（如果它们不参与NMF）
            files = [f for f in files if f not in control_files_to_exclude]
            
            # 应用文件排序
            files = self._apply_nmf_file_sort(files)
            
            skip = self.skip_rows_spin.value()
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())
            
            data_matrix = []
            common_x = None
            sample_labels = []
            control_data_for_plot = []  # 用于绘图的对照组数据
            
            # 收集对照组数据（用于绘图，但不参与NMF）
            for c_file in control_files_to_exclude:
                try:
                    x, y = self.read_data(c_file, skip, x_min_phys, x_max_phys)
                    # 应用预处理（与NMF数据一致，使用主菜单的所有预处理参数）
                    y_proc = y.astype(float)
                    
                    # 1. QC 检查（如果启用）
                    if self.qc_check.isChecked() and np.max(y_proc) < self.qc_threshold_spin.value():
                        continue
                    
                    # 2. BE 校正（如果启用）
                    if self.be_check.isChecked():
                        y_proc = DataPreProcessor.apply_bose_einstein_correction(x, y_proc, self.be_temp_spin.value())
                    
                    # 3. 平滑（如果启用）
                    if self.smoothing_check.isChecked():
                        y_proc = DataPreProcessor.apply_smoothing(y_proc, self.smoothing_window_spin.value(), self.smoothing_poly_spin.value())
                    
                    # 4. AsLS 基线校正（如果启用）
                    if self.baseline_als_check.isChecked():
                        b = DataPreProcessor.apply_baseline_als(y_proc, self.lam_spin.value(), self.p_spin.value())
                        y_proc = y_proc - b
                        y_proc[y_proc < 0] = 0
                    elif self.baseline_poly_check.isChecked():
                        y_proc = DataPreProcessor.apply_baseline_correction(x, y_proc, self.baseline_points_spin.value(), self.baseline_poly_spin.value())
                    
                    # 5. 归一化（如果启用）
                    normalization_mode = self.normalization_combo.currentText()
                    if normalization_mode == 'max':
                        y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
                    elif normalization_mode == 'area':
                        y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
                    elif normalization_mode == 'snv':
                        y_proc = DataPreProcessor.apply_snv(y_proc)
                    y_proc[y_proc < 0] = 0
                    
                    # 6. 全局动态范围压缩（如果启用）- 在归一化之后
                    global_transform_mode = self.global_transform_combo.currentText()
                    if global_transform_mode == '对数变换 (Log)':
                        base = float(self.global_log_base_combo.currentText()) if self.global_log_base_combo.currentText() == '10' else np.e
                        y_proc = DataPreProcessor.apply_log_transform(y_proc, base=base, offset=self.global_log_offset_spin.value())
                    elif global_transform_mode == '平方根变换 (Sqrt)':
                        y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=self.global_sqrt_offset_spin.value())
                    
                    # 7. 二次导数（如果启用）- 在全局动态变换之后
                    # 注意：二次导数在预处理流程中应用，这里不再需要
                    # if False:  # 二次导数已在预处理中应用
                    #     d1 = np.gradient(y_proc, x)
                    #     y_proc = np.gradient(d1, x)
                    
                    # 8. 整体Y轴偏移（预处理最后一步，在二次导数之后）
                    global_y_offset = self.global_y_offset_spin.value() if hasattr(self, 'global_y_offset_spin') else 0.0
                    y_proc = y_proc + global_y_offset
                    
                    control_data_for_plot.append({
                        'x': x,
                        'y': y_proc,
                        'label': os.path.splitext(os.path.basename(c_file))[0]
                    })
                except: pass
            
            # 收集数据（包括参与NMF的对照组）
            all_nmf_files = files + control_data_list
            
            # 检查是否启用分组平均
            use_averaging = hasattr(self, 'nmf_average_check') and self.nmf_average_check.isChecked()
            n_chars = self.n_chars_spin.value() if hasattr(self, 'n_chars_spin') else 5
            
            if use_averaging:
                # 使用分组平均方法
                averaged_data, common_x_avg = self.load_and_average_data(
                    all_nmf_files, n_chars, skip, x_min_phys, x_max_phys
                )
                
                if not averaged_data or common_x_avg is None:
                    QMessageBox.warning(self, "NMF 警告", "分组平均后无有效数据")
                    return
                
                # 对每个分组应用预处理
                for group_key, group_data in averaged_data.items():
                    x = group_data['x']
                    y_proc = group_data['y'].astype(float)
                    
                    # 应用所有预处理步骤（与单个文件处理相同）
                    # 1. QC 检查
                    if self.qc_check.isChecked() and np.max(y_proc) < self.qc_threshold_spin.value():
                        continue
                    
                    # 2. BE 校正
                    if self.be_check.isChecked():
                        y_proc = DataPreProcessor.apply_bose_einstein_correction(x, y_proc, self.be_temp_spin.value())
                    
                    # 3. 平滑
                    if self.smoothing_check.isChecked():
                        y_proc = DataPreProcessor.apply_smoothing(y_proc, self.smoothing_window_spin.value(), self.smoothing_poly_spin.value())
                    
                    # 4. 基线校正
                    if self.baseline_als_check.isChecked():
                        b = DataPreProcessor.apply_baseline_als(y_proc, self.lam_spin.value(), self.p_spin.value())
                        y_proc = y_proc - b
                        y_proc[y_proc < 0] = 0
                    elif self.baseline_poly_check.isChecked():
                        y_proc = DataPreProcessor.apply_baseline_correction(x, y_proc, self.baseline_points_spin.value(), self.baseline_poly_spin.value())
                    
                    # 5. 归一化
                    normalization_mode = self.normalization_combo.currentText()
                    if normalization_mode == 'max':
                        y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
                    elif normalization_mode == 'area':
                        y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
                    elif normalization_mode == 'snv':
                        y_proc = DataPreProcessor.apply_snv(y_proc)
                    y_proc[y_proc < 0] = 0
                    
                    # 6. 全局动态范围压缩
                    global_transform_mode = self.global_transform_combo.currentText()
                    if global_transform_mode == '对数变换 (Log)':
                        base = float(self.global_log_base_combo.currentText()) if self.global_log_base_combo.currentText() == '10' else np.e
                        y_proc = DataPreProcessor.apply_log_transform(y_proc, base=base, offset=self.global_log_offset_spin.value())
                    elif global_transform_mode == '平方根变换 (Sqrt)':
                        y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=self.global_sqrt_offset_spin.value())
                    
                    # 7. 二次导数
                    # 注意：二次导数在预处理流程中应用，这里不再需要
                    # if False:  # 二次导数已在预处理中应用
                    #     d1 = np.gradient(y_proc, x)
                    #     y_proc = np.gradient(d1, x)
                    
                    # 8. 整体Y轴偏移
                    global_y_offset = self.global_y_offset_spin.value() if hasattr(self, 'global_y_offset_spin') else 0.0
                    y_proc = y_proc + global_y_offset
                    
                    # 9. 确保非负
                    y_proc[y_proc < 0] = 0
                    
                    if common_x is None:
                        common_x = x
                    elif len(x) != len(common_x):
                        # 需要插值对齐
                        from scipy.interpolate import interp1d
                        f_interp = interp1d(x, y_proc, kind='linear', fill_value=0, bounds_error=False)
                        y_proc = f_interp(common_x)
                    
                    data_matrix.append(y_proc)
                    sample_labels.append(group_key)
            else:
                # 原有逻辑：逐个文件处理
                for f in all_nmf_files:
                    try:
                        x, y = self.read_data(f, skip, x_min_phys, x_max_phys) # 物理截断
                        
                        # NMF 预处理：使用GUI中设置的所有预处理选项
                        y_proc = y.astype(float)
                        
                        # 1. QC 检查（如果启用）
                        if self.qc_check.isChecked() and np.max(y_proc) < self.qc_threshold_spin.value():
                            continue
                        
                        # 2. BE 校正（如果启用）
                        if self.be_check.isChecked():
                            y_proc = DataPreProcessor.apply_bose_einstein_correction(x, y_proc, self.be_temp_spin.value())
                        
                        # 3. 平滑（如果启用）
                        if self.smoothing_check.isChecked():
                            y_proc = DataPreProcessor.apply_smoothing(y_proc, self.smoothing_window_spin.value(), self.smoothing_poly_spin.value())
                        
                        # 4. 基线校正（优先 AsLS，如果启用）
                        if self.baseline_als_check.isChecked():
                            b = DataPreProcessor.apply_baseline_als(y_proc, self.lam_spin.value(), self.p_spin.value())
                            y_proc = y_proc - b
                            y_proc[y_proc < 0] = 0  # 去负（基线校正后可能为负）
                        
                        # 5. 归一化（如果启用）
                        normalization_mode = self.normalization_combo.currentText()
                        if normalization_mode == 'max':
                            y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
                        elif normalization_mode == 'area':
                            y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
                        elif normalization_mode == 'snv':
                            y_proc = DataPreProcessor.apply_snv(y_proc)
                        
                        # 6. 全局动态范围压缩（如果启用）- 在归一化之后
                        global_transform_mode = self.global_transform_combo.currentText()
                        if global_transform_mode == '对数变换 (Log)':
                            base = float(self.global_log_base_combo.currentText()) if self.global_log_base_combo.currentText() == '10' else np.e
                            y_proc = DataPreProcessor.apply_log_transform(y_proc, base=base, offset=self.global_log_offset_spin.value())
                        elif global_transform_mode == '平方根变换 (Sqrt)':
                            y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=self.global_sqrt_offset_spin.value())
                        
                        # 7. 二次导数（如果启用）- 在全局动态变换之后
                        # 注意：二次导数在预处理流程中应用，这里不再需要
                        # if False:  # 二次导数已在预处理中应用
                        #     d1 = np.gradient(y_proc, x)
                        #     y_proc = np.gradient(d1, x)
                        
                        # 8. 整体Y轴偏移（预处理最后一步，在二次导数之后）
                        global_y_offset = self.global_y_offset_spin.value() if hasattr(self, 'global_y_offset_spin') else 0.0
                        y_proc = y_proc + global_y_offset
                        
                        # 9. NMF 输入必须非负（最终确保）
                        y_proc[y_proc < 0] = 0
                        
                        # 4. 检查并设置 common_x/数据长度
                        if common_x is None: 
                            common_x = x
                        elif len(x) != len(common_x):
                            QMessageBox.warning(self, "NMF 警告", f"文件 {os.path.basename(f)} 波数点数 ({len(x)}) 与基准 ({len(common_x)}) 不一致，跳过。")
                            continue
                            
                        data_matrix.append(y_proc)
                        sample_labels.append(os.path.splitext(os.path.basename(f))[0])
                    except Exception: 
                        # 忽略读取/处理失败的文件
                        continue 
            
            if not data_matrix or common_x is None: # NMF 调试修正 3B
                QMessageBox.warning(self, "NMF 警告", "有效数据不足或波数范围为空 (检查 QC 阈值或 X 轴物理截断)")
                return

            X = np.array(data_matrix)
            
            # 应用 SVD 去噪（从NMF面板获取，如果启用）
            if hasattr(self, 'nmf_svd_denoise_check') and self.nmf_svd_denoise_check.isChecked():
                k_components = self.nmf_svd_components_spin.value() if hasattr(self, 'nmf_svd_components_spin') else 5
                X = DataPreProcessor.svd_denoise(X, k_components)
                print(f"已应用 SVD 去噪（NMF专用），保留 {k_components} 个主成分")
            
            # 解析和应用区域权重（加权 NMF）
            region_weights = None
            if hasattr(self, 'nmf_region_weights_input'):
                weights_str = self.nmf_region_weights_input.text().strip()
                if weights_str:
                    region_weights = self.parse_region_weights(weights_str, common_x)
                    # 应用权重：X_weighted = X * w
                    X_weighted = X * region_weights[np.newaxis, :]
                    X_original = X.copy()  # 保存原始数据用于后续恢复
                    X = X_weighted
                    print(f"已应用区域权重，加权 NMF 模式")
            
            # 读取预滤波参数
            pca_filter_enabled = self.nmf_pca_filter_check.isChecked()
            filter_algorithm = self.nmf_filter_algo_combo.currentText()  # 新增：读取降维算法
            filter_components = self.nmf_pca_comp_spin.value()  # 预滤波成分数
            nmf_components = self.nmf_comp_spin.value()  # 最终 NMF 组件数
            max_iter = self.nmf_max_iter.value()
            
            # 检查成分数合法性
            if pca_filter_enabled and filter_components < nmf_components:
                QMessageBox.warning(self, "警告", "预滤波成分数必须大于或等于 NMF 组件数。请检查输入。")
                return
            
            # 检查 NMF 组件数是否超过数据维度限制
            n_samples, n_features = X.shape
            max_components = min(n_samples, n_features)
            if nmf_components > max_components:
                QMessageBox.warning(self, "警告", 
                                  f"NMF 组件数 ({nmf_components}) 超过数据维度限制 (min(样本数={n_samples}, 特征数={n_features})={max_components})。\n"
                                  f"已自动调整为 {max_components}。")
                nmf_components = max_components
            
            # 如果使用预滤波，也要检查预滤波组件数
            if pca_filter_enabled:
                # 对于预滤波，限制基于原始数据维度
                if filter_components > max_components:
                    QMessageBox.warning(self, "警告",
                                      f"预滤波组件数 ({filter_components}) 超过数据维度限制 ({max_components})。\n"
                                      f"已自动调整为 {max_components}。")
                    filter_components = max_components
                
                # 确保 filter_components >= nmf_components
                if filter_components < nmf_components:
                    filter_components = nmf_components
                    # 不再弹出提示框，仅在控制台打印一条日志
                    print(f"[NMF Filter] 已自动调整预滤波组件数为 {filter_components} 以匹配 NMF 组件数。")
            
            # 确定 NMF 初始化方法：如果组件数超过限制，使用 'random' 而不是 'nndsvd'
            nmf_init = 'nndsvd' if nmf_components <= max_components else 'random'
            filter_init = 'nndsvd' if not pca_filter_enabled or filter_components <= max_components else 'random'
            
            # --- 构建 Pipeline ---
            if pca_filter_enabled:
                if filter_algorithm == 'PCA (主成分分析)':
                    pipeline = Pipeline([
                        ('filter', PCA(n_components=filter_components)),
                        ('nonneg', NonNegativeTransformer()),
                        ('nmf', NMF(n_components=nmf_components, init=nmf_init, random_state=42, max_iter=max_iter))
                    ])
                elif filter_algorithm == 'Deep Autoencoder (PyTorch)':
                    # Use the new PyTorch-based Transformer with user-specified random seed
                    random_seed = self.nmf_random_seed_spin.value()  # 获取用户设置的随机种子
                    pipeline = Pipeline([
                        ('filter', AutoencoderTransformer(n_components=filter_components, use_deep=True, 
                                                         max_iter=max_iter, random_state=random_seed)),
                        ('nonneg', NonNegativeTransformer()), # Double check for non-negativity
                        ('nmf', NMF(n_components=nmf_components, init=nmf_init, random_state=42, max_iter=max_iter))
                    ])
                elif 'Autoencoder' in filter_algorithm: # Fallback sklearn AE
                     pipeline = Pipeline([
                        ('filter', AutoencoderTransformer(n_components=filter_components, use_deep=False, 
                                                         max_iter=max_iter, random_state=42)),
                        ('nonneg', NonNegativeTransformer()),
                        ('nmf', NMF(n_components=nmf_components, init=nmf_init, random_state=42, max_iter=max_iter))
                    ])
                else: # NMF -> NMF
                    pipeline = Pipeline([
                        ('filter', NMF(n_components=filter_components, init=filter_init, random_state=42, max_iter=max_iter)),
                        ('nonneg', NonNegativeTransformer()),
                        ('nmf', NMF(n_components=nmf_components, init=nmf_init, random_state=42, max_iter=max_iter))
                    ])
                
                # 训练 Pipeline（在加权数据上）
                W = pipeline.fit_transform(X)
                H_filtered = pipeline.named_steps['nmf'].components_  # 在预滤波空间中的 H (用于回归)
                
                # Deep Autoencoder 可视化（如果使用）
                if filter_algorithm == 'Deep Autoencoder (PyTorch)':
                    try:
                        # 获取第一个样本的原始输入和重构输出
                        ae_model = pipeline.named_steps['filter']
                        if hasattr(ae_model, 'model') and ae_model.model is not None:
                            # 获取第一个样本
                            sample_input = X[0:1, :]  # 保持2D形状
                            
                            # 归一化（如果AE使用了归一化）
                            if ae_model.normalize and ae_model.mean_ is not None:
                                sample_normalized = (sample_input - ae_model.mean_) / ae_model.std_
                            else:
                                sample_normalized = sample_input
                            
                            # 通过AE模型获取重构输出
                            ae_model.model.eval()
                            import torch
                            with torch.no_grad():
                                sample_tensor = torch.tensor(sample_normalized, dtype=torch.float32)
                                y_recon, _ = ae_model.model(sample_tensor)
                                y_clean = y_recon.numpy()
                                
                                # 反归一化
                                if ae_model.normalize and ae_model.mean_ is not None:
                                    y_clean = y_clean * ae_model.std_ + ae_model.mean_
                                
                                # 准备可视化数据
                                y_raw_viz = sample_input.flatten()
                                y_clean_viz = y_clean.flatten()
                                
                                # 创建或更新 DAE 对比窗口
                                if self.dae_window is None:
                                    self.dae_window = DAEComparisonWindow(self)
                                
                                self.dae_window.set_data(common_x, y_raw_viz, y_clean_viz)
                                self.dae_window.show()
                                self.dae_window.raise_()
                    except Exception as e:
                        print(f"Deep Autoencoder 可视化失败: {e}")
                        traceback.print_exc()
                
                # 如果使用了区域权重，需要恢复 H 的物理形状
                if region_weights is not None:
                    # H 在加权空间中，需要除以权重恢复物理形状
                    # 但 H_filtered 是在预滤波空间中，需要先转换回原始空间
                    pass  # 将在下面处理
                
                # 将 H 矩阵转换回原始空间，以便绘图
                if filter_algorithm == 'PCA (主成分分析)':
                    # PCA: 使用 inverse_transform 将 H 转换回原始空间
                    pca_model = pipeline.named_steps['filter']
                    H = pca_model.inverse_transform(H_filtered)  # (nmf_components, n_features_original)
                    
                    # 如果使用了区域权重，恢复 H 的物理形状
                    if region_weights is not None:
                        # H 在加权空间中，除以权重恢复物理形状
                        H = H / region_weights[np.newaxis, :]
                        H[H < 0] = 0  # 确保非负
                elif filter_algorithm in ['Autoencoder (AE - sklearn)', 'Deep Autoencoder (PyTorch)']:
                    # AE: 使用 inverse_transform 将 H 转换回原始空间
                    ae_model = pipeline.named_steps['filter']
                    # H_filtered 形状: (nmf_components, filter_components)
                    # inverse_transform 返回: (nmf_components, n_features_original)
                    H = ae_model.inverse_transform(H_filtered)  # (nmf_components, n_features_original)
                    
                    # 如果使用了区域权重，恢复 H 的物理形状
                    if region_weights is not None:
                        H = H / region_weights[np.newaxis, :]
                        H[H < 0] = 0  # 确保非负
                    
                    # 确保 H 的维度正确，如果维度不匹配，进行插值对齐
                    if H.shape[1] != len(common_x):
                        # 维度不匹配：使用插值将H对齐到common_x
                        from scipy.interpolate import interp1d
                        # 获取训练时的特征维度（应该在fit时已保存）
                        n_features_train = ae_model.n_features if hasattr(ae_model, 'n_features') and ae_model.n_features is not None else H.shape[1]
                        
                        # 创建训练时的x轴（假设是均匀分布的，与common_x范围一致）
                        # 注意：这里假设训练时的x轴与common_x的范围相同，只是点数不同
                        x_train = np.linspace(common_x[0], common_x[-1], n_features_train)
                        
                        # 对每个组分进行插值对齐
                        H_aligned = np.zeros((H.shape[0], len(common_x)))
                        for i in range(H.shape[0]):
                            f_interp = interp1d(x_train, H[i, :], kind='linear', 
                                              fill_value=0, bounds_error=False)
                            H_aligned[i, :] = f_interp(common_x)
                        H = H_aligned
                        
                        print(f"信息：H矩阵维度已从 {n_features_train} 插值对齐到 {len(common_x)}")
                else:  # NMF (非负矩阵分解)
                    # NMF -> NMF: H_final = H_filtered @ H_filter (矩阵乘法)
                    # H_filtered 是第二个 NMF 的 components_ (nmf_components, filter_components)
                    # H_filter 是第一个 NMF 的 components_ (filter_components, n_features_original)
                    # 结果: H_final (nmf_components, n_features_original)
                    nmf_filter_model = pipeline.named_steps['filter']
                    H_filter = nmf_filter_model.components_  # (filter_components, n_features_original)
                    H = H_filtered @ H_filter  # (nmf_components, filter_components) @ (filter_components, n_features_original) = (nmf_components, n_features_original)
                    
                    # 如果使用了区域权重，恢复 H 的物理形状
                    if region_weights is not None:
                        H = H / region_weights[np.newaxis, :]
                        H[H < 0] = 0  # 确保非负
                
                # 保存预滤波模型供回归使用
                self.last_pca_model = pipeline.named_steps['filter']  # 无论 PCA 还是 NMF，都保存为 filter
                
                # 保存预滤波空间中的 H 用于回归（重要：回归时需要在预滤波空间中进行）
                self.last_fixed_H = H_filtered.copy()
                # 保存原始空间的 H 用于绘图和验证
                self.last_fixed_H_original = H.copy()
                # 保存波数轴，用于定量分析
                self.last_common_x = common_x.copy()
            else:
                # 标准 NMF (不启用预滤波)
                model = NMF(n_components=nmf_components, init=nmf_init, random_state=42, max_iter=max_iter)
                W = model.fit_transform(X)
                H = model.components_
                
                # 如果使用了区域权重，恢复 H 的物理形状
                if region_weights is not None:
                    H = H / region_weights[np.newaxis, :]
                    H[H < 0] = 0  # 确保非负
                
                self.last_pca_model = None  # 清除预滤波模型引用
                # 标准 NMF：H 矩阵直接用于回归和绘图
                self.last_fixed_H = H.copy()
                self.last_fixed_H_original = H.copy()
                # 保存波数轴，用于定量分析
                self.last_common_x = common_x.copy()
            
            # 为NMF组分创建独立Y轴控制项（如果还没有创建，保留现有值）
            self._create_nmf_component_controls(nmf_components, preserve_values=True)
            
            # 收集独立Y轴参数和预处理选项（用于NMF组分绘图）- 只使用NMF组分的控制项
            individual_y_params = {}
            if hasattr(self, 'nmf_component_control_widgets'):
                for comp_label, widgets in self.nmf_component_control_widgets.items():
                    transform_type = widgets['transform'].currentText()
                    transform_mode = 'none'
                    transform_params = {}
                    
                    if transform_type == '对数变换 (Log)':
                        transform_mode = 'log'
                        transform_params = {
                            'base': float(widgets['log_base'].currentText()) if widgets['log_base'].currentText() == '10' else np.e,
                            'offset': widgets['log_offset'].value()
                        }
                    elif transform_type == '平方根变换 (Sqrt)':
                        transform_mode = 'sqrt'
                        transform_params = {
                            'offset': widgets['sqrt_offset'].value()
                        }
                    
                    individual_y_params[comp_label] = {
                        'scale': widgets['scale'].value(),
                        'offset': widgets['offset'].value(),
                        'transform': transform_mode,
                        'transform_params': transform_params
                    }
            
            # 收集NMF组分图例重命名
            nmf_legend_names = {}
            if hasattr(self, 'nmf_component_rename_widgets'):
                for comp_label, rename_widget in self.nmf_component_rename_widgets.items():
                    new_name = rename_widget.text().strip()
                    if new_name:  # 如果输入了新名称，使用新名称；否则使用默认名称
                        nmf_legend_names[comp_label] = new_name
            
            # 为对照组数据添加独立Y轴参数（如果存在）
            for ctrl_data in control_data_for_plot:
                ctrl_label = ctrl_data['label']
                # 检查是否有对应的独立Y轴控制项
                if hasattr(self, 'individual_control_widgets') and ctrl_label in self.individual_control_widgets:
                    widgets = self.individual_control_widgets[ctrl_label]
                    individual_y_params[ctrl_label] = {
                        'scale': widgets['scale'].value(),
                        'offset': widgets['offset'].value(),
                        'transform': 'none',  # 对照组不使用变换
                        'transform_params': {}
                    }
            
            # 获取垂直参考线参数（从主菜单）
            vertical_lines = []
            if hasattr(self, 'vertical_lines_input'):
                vlines_text = self.vertical_lines_input.toPlainText().strip()
                if vlines_text:
                    try:
                        import re
                        vlines_str = re.split(r'[,;\s\n]+', vlines_text)
                        vertical_lines = [float(x.strip()) for x in vlines_str if x.strip()]
                    except:
                        pass

            # 获取样式参数
            style_params = self._get_current_style_params()

            # 收集 NMF 样式参数（包括标题和轴标签，以及所有绘图参数）
            nmf_style_params = {
                # NMF特定业务参数（不包含主窗口的样式参数）
                'comp1_color': self.comp1_color_input.text().strip() if self.comp1_color_input.text().strip() else 'blue',
                'comp2_color': self.comp2_color_input.text().strip() if self.comp2_color_input.text().strip() else 'red',
                'comp_line_width': self.nmf_comp_line_width.value(),
                'comp_line_style': self.nmf_comp_line_style.currentText(),
                'weight_line_width': self.nmf_weight_line_width.value(),
                'weight_line_style': self.nmf_weight_line_style.currentText(),
                'weight_marker_size': self.nmf_marker_size.value(),
                'weight_marker_style': self.nmf_marker_style.currentText(),
                'title_font_size': self.nmf_title_font_spin.value(),
                'label_font_size': self.nmf_title_font_spin.value() - 2,
                'tick_font_size': self.nmf_tick_font_spin.value(),
                'legend_font_size': self.nmf_tick_font_spin.value() + 2,
                'x_axis_invert': style_params.get('x_axis_invert', False),
                'peak_detection_enabled': self.peak_check.isChecked(),
                'nmf_top_title': self.nmf_top_title_input.text().strip(),
                'nmf_bottom_title': self.nmf_bottom_title_input.text().strip(),
                'nmf_top_title_fontsize': self.nmf_top_title_font_spin.value(),
                'nmf_top_title_pad': self.nmf_top_title_pad_spin.value(),
                'nmf_top_title_show': self.nmf_top_title_show_check.isChecked(),
                'nmf_bottom_title_fontsize': self.nmf_bottom_title_font_spin.value(),
                'nmf_bottom_title_pad': self.nmf_bottom_title_pad_spin.value(),
                'nmf_bottom_title_show': self.nmf_bottom_title_show_check.isChecked(),
                'nmf_top_xlabel': self.nmf_xlabel_top_input.text().strip(),
                'nmf_top_xlabel_fontsize': self.nmf_top_xlabel_font_spin.value(),
                'nmf_top_xlabel_pad': self.nmf_top_xlabel_pad_spin.value(),
                'nmf_top_xlabel_show': self.nmf_top_xlabel_show_check.isChecked(),
                'nmf_top_ylabel': self.nmf_ylabel_top_input.text().strip(),
                'nmf_top_ylabel_fontsize': self.nmf_top_ylabel_font_spin.value(),
                'nmf_top_ylabel_pad': self.nmf_top_ylabel_pad_spin.value(),
                'nmf_top_ylabel_show': self.nmf_top_ylabel_show_check.isChecked(),
                'nmf_bottom_xlabel': self.nmf_xlabel_bottom_input.text().strip(),
                'nmf_bottom_xlabel_fontsize': self.nmf_bottom_xlabel_font_spin.value(),
                'nmf_bottom_xlabel_pad': self.nmf_bottom_xlabel_pad_spin.value(),
                'nmf_bottom_xlabel_show': self.nmf_bottom_xlabel_show_check.isChecked(),
                'nmf_bottom_ylabel': self.nmf_ylabel_bottom_input.text().strip(),
                'nmf_bottom_ylabel_fontsize': self.nmf_bottom_ylabel_font_spin.value(),
                'nmf_bottom_ylabel_pad': self.nmf_bottom_ylabel_pad_spin.value(),
                'nmf_bottom_ylabel_show': self.nmf_bottom_ylabel_show_check.isChecked(),
                'is_derivative': False,  # 二次导数在预处理流程中应用
                'global_stack_offset': self._get_stack_offset_from_panel(),
                'global_scale_factor': self.global_y_scale_factor_spin.value(),
                'individual_y_params': individual_y_params,
                'nmf_legend_names': nmf_legend_names,
                'control_data_list': control_data_for_plot,
                'vertical_lines': vertical_lines,  # 垂直参考线
                'vertical_line_color': '#034DFB',  # 默认蓝色
                'vertical_line_style': '--',  # 默认虚线
                'vertical_line_width': 0.8,  # 默认线宽
                'vertical_line_alpha': 0.8,  # 默认透明度
            }

            # 准备 NMF 结果窗口（如果已存在则更新，否则创建）
            if hasattr(self, 'nmf_window') and self.nmf_window is not None and self.nmf_window.isVisible():
                # 更新现有窗口
                self.nmf_window.set_data(W, H, common_x, nmf_style_params, sample_labels)
                # 恢复之前选择的目标组分索引
                if hasattr(self.nmf_window, 'target_component_index'):
                    self.nmf_window.target_component_index = self.nmf_target_component_index
                    self.nmf_window._update_target_component_radios()
                self.nmf_window.raise_()  # 将窗口置于最前
            else:
                # 创建新窗口
                win = NMFResultWindow("NMF Analysis Result", self)
                win.target_component_index = self.nmf_target_component_index  # 设置初始选择
                win.set_data(W, H, common_x, nmf_style_params, sample_labels)
                self.nmf_window = win
                win.show()
            
        except Exception as e:
            QMessageBox.critical(self, "NMF Error", f"NMF 运行失败: {str(e)}")
            traceback.print_exc()
    
    def run_nmf_regression(self, target_files, fixed_H):
        """
        非负组分回归 (NMF-CR)：使用固定的H矩阵计算新数据的W权重
        
        参数:
            target_files: 目标文件列表（完整路径）
            fixed_H: 固定的组分光谱矩阵 H (n_components, n_features)
        
        返回:
            W: 权重矩阵 (n_samples, n_components)
            H: 固定的组分矩阵（与输入相同）
            common_x: 波数轴
            sample_labels: 样本标签列表
        """
        try:
            skip = self.skip_rows_spin.value()
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())
            
            data_matrix = []
            common_x = None
            sample_labels = []
            
            # 收集目标文件的数据（target_files已经是完整路径）
            for f in target_files:
                try:
                    x, y = self.read_data(f, skip, x_min_phys, x_max_phys)  # 物理截断
                    
                    # NMF 预处理：使用GUI中设置的所有预处理选项
                    y_proc = y.astype(float)
                    
                    # 1. QC 检查（如果启用）
                    if self.qc_check.isChecked() and np.max(y_proc) < self.qc_threshold_spin.value():
                        continue
                    
                    # 2. BE 校正（如果启用）
                    if self.be_check.isChecked():
                        y_proc = DataPreProcessor.apply_bose_einstein_correction(x, y_proc, self.be_temp_spin.value())
                    
                    # 3. 平滑（如果启用）
                    if self.smoothing_check.isChecked():
                        y_proc = DataPreProcessor.apply_smoothing(y_proc, self.smoothing_window_spin.value(), self.smoothing_poly_spin.value())
                    
                    # 4. 基线校正（优先 AsLS，如果启用）
                    if self.baseline_als_check.isChecked():
                        b = DataPreProcessor.apply_baseline_als(y_proc, self.lam_spin.value(), self.p_spin.value())
                        y_proc = y_proc - b
                        y_proc[y_proc < 0] = 0  # 去负（基线校正后可能为负）
                    
                    # 5. 归一化（如果启用）
                    normalization_mode = self.normalization_combo.currentText()
                    if normalization_mode == 'max':
                        y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
                    elif normalization_mode == 'area':
                        y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
                    elif normalization_mode == 'snv':
                        y_proc = DataPreProcessor.apply_snv(y_proc)
                    
                    # 6. 全局动态范围压缩（如果启用）- 在归一化之后
                    global_transform_mode = self.global_transform_combo.currentText()
                    if global_transform_mode == '对数变换 (Log)':
                        base = float(self.global_log_base_combo.currentText()) if self.global_log_base_combo.currentText() == '10' else np.e
                        y_proc = DataPreProcessor.apply_log_transform(y_proc, base=base, offset=self.global_log_offset_spin.value())
                    elif global_transform_mode == '平方根变换 (Sqrt)':
                        y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=self.global_sqrt_offset_spin.value())
                    
                    # 7. 二次导数（如果启用）- 在全局动态变换之后
                    # 注意：二次导数在预处理流程中应用，这里不再需要
                    # if False:  # 二次导数已在预处理中应用
                    #     d1 = np.gradient(y_proc, x)
                    #     y_proc = np.gradient(d1, x)
                    
                    # 8. 整体Y轴偏移（预处理最后一步，在二次导数之后）
                    global_y_offset = self.global_y_offset_spin.value() if hasattr(self, 'global_y_offset_spin') else 0.0
                    y_proc = y_proc + global_y_offset
                    
                    # 9. NMF 输入必须非负（最终确保）
                    y_proc[y_proc < 0] = 0
                    
                    # 检查并设置 common_x/数据长度
                    if common_x is None:
                        common_x = x
                    elif len(x) != len(common_x):
                        QMessageBox.warning(self, "NMF 警告", f"文件 {os.path.basename(f)} 波数点数 ({len(x)}) 与基准 ({len(common_x)}) 不一致，跳过。")
                        continue
                    
                    # 检查数据长度是否与H矩阵匹配
                    # 如果使用了预滤波，fixed_H.shape[1]是预滤波成分数，需要在预滤波转换后检查
                    # 如果未使用预滤波，fixed_H.shape[1]是原始特征数，需要在这里检查
                    if self.last_pca_model is None:
                        # 未使用预滤波：检查原始数据长度
                        if len(y_proc) != fixed_H.shape[1]:
                            QMessageBox.warning(self, "NMF 警告", f"文件 {os.path.basename(f)} 数据长度 ({len(y_proc)}) 与固定H矩阵的特征数 ({fixed_H.shape[1]}) 不匹配，跳过。")
                            continue
                    # 如果使用了预滤波，数据长度检查将在预滤波转换后进行
                    
                    data_matrix.append(y_proc)
                    sample_labels.append(os.path.splitext(os.path.basename(f))[0])
                except Exception as e:
                    print(f"处理文件 {f} 时出错: {e}")
                    continue
            
            if not data_matrix or common_x is None:
                QMessageBox.warning(self, "NMF 警告", "有效数据不足或波数范围为空 (检查 QC 阈值或 X 轴物理截断)")
                return None, None, None, None
            
            X = np.array(data_matrix)  # (n_samples, n_features)
            
            # 核心修正：如果存在训练好的预滤波模型，必须先对 X 进行转换
            if self.last_pca_model is not None:
                try:
                    X_filtered = self.last_pca_model.transform(X)
                    # 确保非负（PCA 输出可能包含负值，NMF 需要非负输入）
                    X_filtered[X_filtered < 0] = 0
                    X_target = X_filtered
                    
                    # 确保 fixed_H 是在滤波空间中提取的 H 矩阵 (n_components, n_features_filtered)
                    n_samples, n_features_filtered = X_target.shape
                    n_components = fixed_H.shape[0]
                    
                    # 检查预滤波转换后的特征数是否与fixed_H匹配
                    if n_features_filtered != fixed_H.shape[1]:
                        QMessageBox.warning(self, "NMF 警告", f"预滤波转换后的特征数 ({n_features_filtered}) 与固定H矩阵的特征数 ({fixed_H.shape[1]}) 不匹配。请确保使用相同的预滤波设置。")
                        return None, None, None, None
                    
                    # NMF 回归现在在预滤波空间中进行
                    W = np.zeros((n_samples, n_components))
                    H_T = fixed_H.T  # (n_features_filtered, n_components)
                    
                    for i in range(n_samples):
                        x_i_filtered = X_target[i, :]  # 在预滤波空间中的行向量
                        w_i_T, _ = nnls(H_T, x_i_filtered)
                        W[i, :] = w_i_T
                        
                except Exception as e:
                    QMessageBox.critical(self, "回归错误", f"预滤波转换或 NNLS 求解失败: {e}")
                    traceback.print_exc()
                    return None, None, None, None
            else:
                # 无预滤波：标准 NMF 回归
                X_target = X
                n_samples, n_features = X_target.shape
                n_components = fixed_H.shape[0]
                
                # 使用非负最小二乘求解 W
                # 对于每条光谱 x_i（行向量），求解 H^T * w_i^T ≈ x_i^T
                # 即求解 w_i^T = nnls(H^T, x_i^T)[0]
                W = np.zeros((n_samples, n_components))
                H_T = fixed_H.T  # (n_features, n_components)
                
                for i in range(n_samples):
                    x_i = X_target[i, :]  # 第i条光谱 (n_features,)
                    # 求解 H^T * w_i^T ≈ x_i^T，即 w_i^T = nnls(H^T, x_i^T)[0]
                    w_i_T, _ = nnls(H_T, x_i)
                    W[i, :] = w_i_T  # w_i^T 已经是列向量，直接赋值
            
            return W, fixed_H, common_x, sample_labels
            
        except Exception as e:
            QMessageBox.critical(self, "NMF-CR Error", f"非负组分回归运行失败: {str(e)}")
            traceback.print_exc()
            return None, None, None, None
    
    def _on_nmf_color_changed(self):
        """NMF颜色变化时的回调函数（自动更新图表）"""
        # 只有在NMF窗口已存在时才自动更新
        if hasattr(self, 'nmf_window') and self.nmf_window is not None and hasattr(self.nmf_window, 'H'):
            # 注意：已删除 rerun_nmf_plot 方法，NMF窗口更新需要重新运行NMF分析
            pass

    # --- 核心：拉曼散射拟合叠加到图上 ---
    def run_scattering_fit_overlay(self):
        if self.active_plot_window is None or not self.active_plot_window.isVisible():
            QMessageBox.warning(self, "警告", "请先运行一次绘图，打开一个光谱图窗口。")
            return
            
        win = self.active_plot_window
        ax = win.current_ax
        plot_data = win.current_plot_data
        
        if not plot_data:
            QMessageBox.warning(self, "警告", "当前图中没有可用于拟合的数据。")
            return
        
        # 检查是否已达到最大拟合曲线数量
        current_fit_count = len(self.fit_curves_info)
        max_fit_count = self.fit_curve_count_spin.value()
        if current_fit_count >= max_fit_count:
            QMessageBox.warning(self, "警告", f"已达到最大拟合曲线数量 ({max_fit_count})。请先清除部分拟合曲线或增加最大数量。")
            return
            
        # 1. 定义散射拟合模型
        def lorentzian(x, A, x0, gamma):
            return A * (gamma**2 / ((x - x0)**2 + gamma**2))

        def gaussian(x, A, x0, sigma):
            return A * np.exp(-((x - x0)**2) / (2 * sigma**2))

        try:
            cutoff = self.fit_cutoff_spin.value()
            model_name = self.fit_model_combo.currentText()
            model_func = lorentzian if model_name == 'Lorentzian' else gaussian
            
            # 2. 选择第一个有效的个体光谱进行拟合
            data_key = next((k for k, v in plot_data.items() if v['type'] in ['Individual', 'Mean']), None)
            
            if not data_key:
                self.fit_output_text.append("未找到个体或平均光谱数据进行拟合。")
                return

            item = plot_data[data_key]
            x_raw, y_raw = item['x'], item['y']
            original_color = item['color']
            
            # 仅使用截止波数以下的数据
            mask = x_raw <= cutoff
            x_fit = x_raw[mask]
            y_fit = y_raw[mask] # 使用经过主绘图管道预处理/偏移后的 Y 值
            
            if len(x_fit) < 4:
                self.fit_output_text.append(f"数据 {data_key} 在截止点 {cutoff} 以下数据不足。")
                return
                
            # 移除数据上的偏移 (Fit 必须在接近零基线上进行)
            min_y_fit = np.min(y_fit)
            y_fit_zeroed = y_fit - min_y_fit 
            y_fit_zeroed[y_fit_zeroed < 0] = 0 # 保证非负
            
            # 初始参数猜测 (基于零基线数据)
            A_guess = np.max(y_fit_zeroed)
            x0_guess = x_fit[np.argmax(y_fit_zeroed)]
            gamma_sigma_guess = 10 
            
            p0 = [A_guess, x0_guess, gamma_sigma_guess]
            bounds = ([0, x_fit.min(), 0], [np.inf, x_fit.max(), np.inf])
            
            popt, pcov = curve_fit(model_func, x_fit, y_fit_zeroed, p0=p0, bounds=bounds)
            
            # 3. 报告结果
            if model_name == 'Lorentzian':
                params_str = f"A={popt[0]:.2f}, x0={popt[1]:.2f}, $\\gamma$={popt[2]:.2f}"
            else:
                params_str = f"A={popt[0]:.2f}, x0={popt[1]:.2f}, $\\sigma$={popt[2]:.2f}"
            
            fit_index = current_fit_count + 1
            self.fit_output_text.append(f"✅ 拟合曲线 #{fit_index}: {data_key} ({model_name} 拟合)\n参数: {params_str}\n---")
            
            # 4. 获取拟合曲线样式参数
            fit_color = self.fit_line_color_input.text().strip() or 'magenta'
            fit_line_style = self.fit_line_style_combo.currentText()
            fit_line_width = self.fit_line_width_spin.value()
            fit_marker = self.fit_marker_combo.currentText()
            fit_marker_size = self.fit_marker_size_spin.value()
            
            # 验证颜色
            try:
                from matplotlib.colors import to_rgba
                to_rgba(fit_color)
            except:
                fit_color = 'magenta'
                self.fit_output_text.append(f"⚠️ 颜色 '{self.fit_line_color_input.text()}' 无效，使用默认颜色 'magenta'\n")
            
            # 生成图例标签
            legend_label = self.fit_legend_label_input.text().strip()
            if not legend_label:
                legend_label = f"Fit #{fit_index}: {data_key}"
            
            # 5. 计算拟合曲线 Y 值并绘制
            y_fit_curve = model_func(x_fit, *popt)
            y_fit_final = y_fit_curve + min_y_fit
            
            # 准备绘图参数
            plot_kwargs = {
                'color': fit_color,
                'linewidth': fit_line_width,
                'label': legend_label
            }
            
            # 如果有标记，添加标记参数
            if fit_marker != '无':
                plot_kwargs['marker'] = fit_marker
                plot_kwargs['markersize'] = fit_marker_size
                plot_kwargs['markevery'] = max(1, len(x_fit) // 50)  # 每50个点显示一个标记，避免太密集
            
            # 绘制拟合线
            line_obj = ax.plot(x_fit, y_fit_final, fit_line_style, **plot_kwargs)[0]
            
            # 6. 存储拟合曲线信息（用于清除和样式管理）
            fit_info = {
                'line_obj': line_obj,
                'data_key': data_key,
                'model_name': model_name,
                'params': popt,
                'x_data': x_fit,
                'y_data': y_fit_final,
                'cutoff': cutoff,
                'legend_label': legend_label,
                'color': fit_color,
                'line_style': fit_line_style,
                'line_width': fit_line_width,
                'marker': fit_marker,
                'marker_size': fit_marker_size
            }
            self.fit_curves_info.append(fit_info)
            
            # 将拟合曲线添加到plot_data中，以便可以被扫描到图例中
            fit_data_key = f"Fit_{fit_index}_{data_key}"
            plot_data[fit_data_key] = {
                'x': x_fit,
                'y': y_fit_final,
                'label': legend_label,
                'color': fit_color,
                'type': 'Fit'
            }
            
            # 7. 更新图例（遵循主菜单的设置）
            # 获取主菜单的图例显示设置
            show_legend_main = self.show_legend_check.isChecked() if hasattr(self, 'show_legend_check') else True
            show_legend_fit = self.fit_show_legend_check.isChecked()
            
            # 只有当主菜单显示图例且拟合曲线图例也启用时才显示图例
            if show_legend_main and show_legend_fit:
                # 获取主菜单的图例样式参数（从UI控件获取）
                from matplotlib.font_manager import FontProperties
                legend_font = FontProperties()
                font_family = self._get_font_family()
                if font_family != 'SimHei':
                    legend_font.set_family(font_family)
                else:
                    legend_font.set_family('sans-serif')
                legend_fontsize = self.legend_font_spin.value() if hasattr(self, 'legend_font_spin') else 12
                legend_font.set_size(legend_fontsize)
                
                legend_loc = self.legend_loc_combo.currentText() if hasattr(self, 'legend_loc_combo') else 'best'
                legend_frame = self.legend_frame_check.isChecked() if hasattr(self, 'legend_frame_check') else True
                legend_ncol = self.legend_column_spin.value() if hasattr(self, 'legend_column_spin') else 1
                legend_columnspacing = self.legend_columnspacing_spin.value() if hasattr(self, 'legend_columnspacing_spin') else 0.8
                legend_labelspacing = self.legend_labelspacing_spin.value() if hasattr(self, 'legend_labelspacing_spin') else 0.5
                legend_handlelength = self.legend_handlelength_spin.value() if hasattr(self, 'legend_handlelength_spin') else 2.0
                
                ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                         ncol=legend_ncol, columnspacing=legend_columnspacing, 
                         labelspacing=legend_labelspacing, handlelength=legend_handlelength)
            elif not show_legend_main:
                # 如果主菜单不显示图例，移除图例
                ax.legend().remove()
            
            win.canvas.draw()

        except Exception as e:
            self.fit_output_text.append(f"❌ 拟合失败: {str(e)}\n---")
            QMessageBox.critical(self, "拟合错误", f"拟合失败: {str(e)}")
            traceback.print_exc()
    
    def clear_all_fit_curves(self):
        """清除所有拟合曲线"""
        if self.active_plot_window is None or not self.active_plot_window.isVisible():
            QMessageBox.warning(self, "警告", "请先运行一次绘图，打开一个光谱图窗口。")
            return
        
        win = self.active_plot_window
        ax = win.current_ax
        plot_data = win.current_plot_data
        
        # 移除所有拟合曲线
        for fit_info in self.fit_curves_info:
            try:
                fit_info['line_obj'].remove()
            except:
                pass
        
        # 从plot_data中移除拟合曲线数据
        fit_keys_to_remove = [k for k in plot_data.keys() if k.startswith('Fit_')]
        for key in fit_keys_to_remove:
            plot_data.pop(key, None)
        
        self.fit_curves_info.clear()
        self.fit_output_text.append("已清除所有拟合曲线。\n")
        
        # 更新图例（遵循主菜单的设置）
        show_legend_main = self.show_legend_check.isChecked() if hasattr(self, 'show_legend_check') else True
        if show_legend_main:
            # 重新绘制图例（只包含原始数据的图例）
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            font_family = self.font_family_combo.currentText() if hasattr(self, 'font_family_combo') else 'SimHei'
            if font_family != 'SimHei':
                legend_font.set_family(font_family)
            else:
                legend_font.set_family('sans-serif')
            legend_fontsize = self.legend_font_spin.value() if hasattr(self, 'legend_font_spin') else 12
            legend_font.set_size(legend_fontsize)
            
            legend_loc = self.legend_loc_combo.currentText() if hasattr(self, 'legend_loc_combo') else 'best'
            legend_frame = self.legend_frame_check.isChecked() if hasattr(self, 'legend_frame_check') else True
            legend_ncol = self.legend_column_spin.value() if hasattr(self, 'legend_column_spin') else 1
            legend_columnspacing = self.legend_columnspacing_spin.value() if hasattr(self, 'legend_columnspacing_spin') else 0.8
            legend_labelspacing = self.legend_labelspacing_spin.value() if hasattr(self, 'legend_labelspacing_spin') else 0.5
            legend_handlelength = self.legend_handlelength_spin.value() if hasattr(self, 'legend_handlelength_spin') else 2.0
            
            ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                     ncol=legend_ncol, columnspacing=legend_columnspacing, 
                     labelspacing=legend_labelspacing, handlelength=legend_handlelength)
        else:
            ax.legend().remove()
        
        win.canvas.draw()


    # --- 核心：组间平均值瀑布图 (保留原功能) ---
    def run_group_average_waterfall(self):
        try:
            folder = self.folder_input.text()
            if not os.path.isdir(folder): return

            # 物理截断值
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())

            # 1. 读取基础参数
            skip = self.skip_rows_spin.value()
            n_chars = self.n_chars_spin.value()
            offset_step = self._get_stack_offset_from_panel()
            scale = self.global_y_scale_factor_spin.value()
            
            # 2. 获取文件并分组
            files = sorted(glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt')))
            groups = group_files_by_name(files, n_chars)
            
            # 筛选指定组
            target_gs = [x.strip() for x in self.groups_input.text().split(',') if x.strip()]
            if target_gs:
                groups = {k: v for k, v in groups.items() if k in target_gs}
            
            # 3. 准备绘图窗口 - 保留窗口位置
            if "GroupComparison" not in self.plot_windows:
                # 创建新窗口（不指定位置，让窗口自动计算远离主菜单的位置）
                self.plot_windows["GroupComparison"] = MplPlotWindow("Group Comparison (Averages)", initial_geometry=None, parent=self)
            
            win = self.plot_windows["GroupComparison"]
            ax = win.canvas.axes
            ax.cla()
            
            colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'teal', 'darkred']
            
            # 对组名进行排序 (尝试按数字逻辑排序，否则按字母)
            sorted_keys = sorted(groups.keys())
            
            # 获取重命名映射（在循环外计算一次）
            rename_map = {k: v.text().strip() for k, v in self.legend_rename_widgets.items() if v.text().strip()}
            
            # 4. 循环处理每一组
            for i, g_name in enumerate(sorted_keys):
                g_files = groups[g_name]
                y_list = []
                common_x = None
                
                # 组内处理：收集所有有效光谱
                for f in g_files:
                    try:
                        x, y = self.read_data(f, skip, x_min_phys, x_max_phys) # 使用物理截断
                        if common_x is None: common_x = x
                        
                        # --- 预处理流程 (复用配置) ---
                        # A. QC
                        if self.qc_check.isChecked() and np.max(y) < self.qc_threshold_spin.value(): continue
                        
                        # B. BE 校正
                        if self.be_check.isChecked(): 
                            y = DataPreProcessor.apply_bose_einstein_correction(x, y, self.be_temp_spin.value())
                            
                        # C. 平滑
                        if self.smoothing_check.isChecked():
                            y = DataPreProcessor.apply_smoothing(y, self.smoothing_window_spin.value(), self.smoothing_poly_spin.value())
                            
                        # D. 基线 (AsLS优先)
                        if self.baseline_als_check.isChecked():
                            b = DataPreProcessor.apply_baseline_als(y, self.lam_spin.value(), self.p_spin.value())
                            y = y - b
                            y[y<0] = 0
                        
                        # E. 归一化 (SNV推荐)
                        if self.normalization_combo.currentText() == 'snv':
                            y = DataPreProcessor.apply_snv(y)
                        elif self.normalization_combo.currentText() == 'max':
                            y = DataPreProcessor.apply_normalization(y, 'max')
                            
                        y_list.append(y)
                    except: pass
                
                if not y_list: continue
                
                # 5. 计算该组平均值和标准差
                y_array = np.array(y_list)
                y_avg = np.mean(y_array, axis=0)
                y_std = np.std(y_array, axis=0)
                
                # 6. 堆叠绘图
                y_plot = y_avg * scale
                y_std_plot = y_std * scale
                
                # 是否求导
                if self.derivative_check.isChecked():
                    d1 = np.gradient(y_plot, common_x)
                    y_plot = np.gradient(d1, common_x)
                    # 求导模式下不绘制阴影
                    y_std_plot = None
                
                # 使用组的独立堆叠位移（如果存在），否则使用全局默认值
                if g_name in self.group_waterfall_control_widgets:
                    group_offset = self.group_waterfall_control_widgets[g_name]['offset'].value()
                else:
                    group_offset = i * offset_step  # 回退到全局默认值
                
                final_y = y_plot + group_offset
                final_y_upper = (y_plot + y_std_plot) + group_offset if y_std_plot is not None else None
                final_y_lower = (y_plot - y_std_plot) + group_offset if y_std_plot is not None else None
                
                # 优先使用组瀑布图的独立颜色（如果存在）
                color = colors[i % len(colors)]  # 默认颜色
                
                # 1. 首先检查组瀑布图的独立颜色控件
                if g_name in self.group_waterfall_control_widgets:
                    color_widget = self.group_waterfall_control_widgets[g_name].get('color')
                    if color_widget and hasattr(color_widget, 'text'):
                        color_text = color_widget.text().strip()
                        if color_text:
                            try:
                                import matplotlib.colors as mcolors
                                mcolors.to_rgba(color_text)  # 验证颜色
                                color = color_text
                            except (ValueError, AttributeError):
                                pass  # 如果颜色无效，继续尝试其他颜色源
                
                # 2. 如果组瀑布图没有独立颜色，则从individual_control_widgets中获取该组第一个文件的颜色
                if color == colors[i % len(colors)] and g_files and hasattr(self, 'individual_control_widgets'):
                    first_file_base = os.path.splitext(os.path.basename(g_files[0]))[0]
                    if first_file_base in self.individual_control_widgets:
                        color_widget = self.individual_control_widgets[first_file_base].get('color')
                        if color_widget and hasattr(color_widget, 'text'):
                            color_text = color_widget.text().strip()
                            if color_text:
                                # 验证颜色有效性
                                try:
                                    import matplotlib.colors as mcolors
                                    mcolors.to_rgba(color_text)  # 验证颜色
                                    color = color_text
                                except (ValueError, AttributeError):
                                    pass  # 如果颜色无效，使用默认颜色
                
                # 使用重命名后的组名（如果有）
                base_display_name = rename_map.get(g_name, g_name)
                
                # 获取完整的图例名称（包括后缀的重命名）
                avg_label_key = f"{g_name} (Avg)"
                std_label_key = f"{g_name} ± Std"
                
                # 如果基础名称被重命名，构建新的图例名称
                if base_display_name != g_name:
                    # 基础名称被重命名，检查是否有单独的后缀重命名
                    avg_label = rename_map.get(avg_label_key, f"{base_display_name} (Avg)")
                    std_label = rename_map.get(std_label_key, f"{base_display_name} ± Std")
                else:
                    # 基础名称未重命名，使用单独的后缀重命名或默认
                    avg_label = rename_map.get(avg_label_key, f"{g_name} (Avg)")
                    std_label = rename_map.get(std_label_key, f"{g_name} ± Std")
                
                # 绘制阴影（如果启用）- 使用线条颜色，确保阴影、线条、图例颜色完全一致
                # 从 publication_style_panel 获取阴影设置
                if hasattr(self, 'publication_style_panel') and self.publication_style_panel:
                    config = self.publication_style_panel.get_config()
                    ps = config.publication_style
                    shadow_alpha = ps.shadow_alpha if ps else 0.25
                    show_shadow = True  # 默认显示阴影
                else:
                    shadow_alpha = 0.25  # 默认值
                    show_shadow = True
                
                # 确保 alpha 值在 0-1 范围内
                safe_alpha = max(0.0, min(1.0, shadow_alpha))
                
                if show_shadow and final_y_upper is not None and final_y_lower is not None:
                    # 阴影颜色与线条颜色完全一致
                    ax.fill_between(common_x, final_y_lower, final_y_upper, 
                                   color=color, alpha=safe_alpha, label=std_label)
                
                # 绘制平均线 - 使用出版质量样式参数（线宽、线型）
                # 从面板获取，如果没有面板则使用默认值
                if hasattr(self, 'publication_style_panel') and self.publication_style_panel:
                    config = self.publication_style_panel.get_config()
                    ps = config.publication_style
                    line_width = ps.line_width if ps else 1.2
                    line_style = ps.line_style if ps else '-'
                else:
                    # 从配置管理器获取
                    from src.core.plot_config_manager import PlotConfigManager
                    config_manager = PlotConfigManager()
                    config = config_manager.get_config()
                    ps = config.publication_style
                    line_width = ps.line_width
                    line_style = ps.line_style
                plot_style = self.plot_style_combo.currentText()  # line 或 scatter
                
                label_text = avg_label
                
                if plot_style == 'line':
                    ax.plot(common_x, final_y, label=label_text, color=color, 
                           linewidth=line_width, linestyle=line_style)
                else:  # scatter
                    ax.plot(common_x, final_y, label=label_text, color=color, 
                           marker='.', linestyle='', markersize=line_width*3)

            # 7. 样式修饰 - 使用主菜单的出版样式参数
            # 设置字体
            font_family = self._get_font_family()
            current_font = 'Times New Roman' if font_family == 'Times New Roman' else font_family
            
            # 坐标轴翻转
            # 从样式面板获取坐标轴显示控制
            if hasattr(self, 'publication_style_panel') and self.publication_style_panel:
                config = self.publication_style_panel.get_config()
                ps = config.publication_style
                if ps.x_axis_invert: ax.invert_xaxis()
                if not ps.show_y_values: ax.set_yticks([])
            else:
                # 向后兼容
                if hasattr(self, 'x_axis_invert_check') and self.x_axis_invert_check.isChecked(): ax.invert_xaxis()
                if hasattr(self, 'show_y_val_check') and not self.show_y_val_check.isChecked(): ax.set_yticks([])
            
            # 使用GUI中的浓度梯度图X轴标题控制参数
            # 优先从 publication_style_panel 获取
            if hasattr(self, 'publication_style_panel') and self.publication_style_panel:
                config = self.publication_style_panel.get_config()
                ps = config.publication_style
                xlabel_text = ps.xlabel_text if ps else r"Wavenumber ($\mathrm{cm^{-1}}$)"
                ylabel_text = ps.ylabel_text if ps else "Intensity"
                xlabel_fontsize = ps.xlabel_fontsize if ps else 20
                xlabel_pad = ps.xlabel_pad if ps else 10.0
                ylabel_fontsize = ps.ylabel_fontsize if ps else 20
                ylabel_pad = ps.ylabel_pad if ps else 10.0
                xlabel_show = ps.xlabel_show if ps else True
                ylabel_show = ps.ylabel_show if ps else True
            else:
                # 回退：从直接控件获取（向后兼容）
                xlabel_text = self.xlabel_input.text() if hasattr(self, 'xlabel_input') else r"Wavenumber ($\mathrm{cm^{-1}}$)"
                ylabel_text = self.ylabel_input.text() if hasattr(self, 'ylabel_input') else "Intensity"
                xlabel_fontsize = self.gradient_xlabel_font_spin.value() if hasattr(self, 'gradient_xlabel_font_spin') else 20
                xlabel_pad = self.gradient_xlabel_pad_spin.value() if hasattr(self, 'gradient_xlabel_pad_spin') else 10.0
                ylabel_fontsize = self.gradient_ylabel_font_spin.value() if hasattr(self, 'gradient_ylabel_font_spin') else 20
                ylabel_pad = self.gradient_ylabel_pad_spin.value() if hasattr(self, 'gradient_ylabel_pad_spin') else 10.0
                xlabel_show = self.gradient_xlabel_show_check.isChecked() if hasattr(self, 'gradient_xlabel_show_check') else True
                ylabel_show = self.gradient_ylabel_show_check.isChecked() if hasattr(self, 'gradient_ylabel_show_check') else True
            
            if xlabel_show:
                ax.set_xlabel(xlabel_text, fontsize=xlabel_fontsize, 
                            labelpad=xlabel_pad, fontfamily=current_font)
            
            # 使用GUI中的浓度梯度图Y轴标题控制参数
            # 注意：二次导数在预处理流程中应用，ylabel不再需要根据derivative_check改变
            ylabel = ylabel_text
            if ylabel_show:
                ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, 
                            labelpad=ylabel_pad, fontfamily=current_font)
            
            # 使用GUI中的标题控制参数
            if self.gradient_title_show_check.isChecked():
                gradient_title_text = self.gradient_title_input.text().strip() or "Concentration Gradient (Group Averages)"
                ax.set_title(gradient_title_text, fontsize=self.gradient_title_font_spin.value(), 
                           pad=self.gradient_title_pad_spin.value(), fontfamily=current_font)
            
            # Ticks 样式（使用主菜单的样式参数）
            style_params = self._get_current_style_params()
            tick_direction = style_params.get('tick_direction', 'in')
            tick_len_major = style_params.get('tick_len_major', 8)
            tick_len_minor = style_params.get('tick_len_minor', 4)
            tick_width = style_params.get('tick_width', 1.0)
            tick_label_fontsize = style_params.get('tick_label_font', 16)
            
            ax.tick_params(labelsize=tick_label_fontsize, direction=tick_direction, width=tick_width, labelfontfamily=current_font)
            ax.tick_params(which='major', length=tick_len_major)
            ax.tick_params(which='minor', length=tick_len_minor)
            
            # 边框设置 (Spines) - 使用主菜单的样式参数
            border_sides = self.get_checked_border_sides()
            border_linewidth = style_params.get('spine_width', 2.0)
            for side in ['top', 'right', 'left', 'bottom']:
                if side in border_sides:
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_linewidth(border_linewidth)
                else:
                    ax.spines[side].set_visible(False)
            
            # 网格设置 - 使用主菜单的样式参数
            if style_params.get('show_grid', False):
                ax.grid(True, alpha=style_params.get('grid_alpha', 0.2))
            
            # 图例设置 - 使用主菜单的样式参数
            if style_params.get('show_legend', True):
                legend_fontsize = style_params.get('legend_fontsize', 10)
                legend_frame = style_params.get('legend_frame', True)
                legend_loc = style_params.get('legend_loc', 'best')
                
                # 设置图例字体（支持中文）
                from matplotlib.font_manager import FontProperties
                legend_font = FontProperties()
                # 检测是否包含中文，如果包含则使用sans-serif
                def contains_chinese(text):
                    if not isinstance(text, str):
                        return False
                    return any('\u4e00' <= char <= '\u9fff' for char in text)
                
                # 检查图例标签是否包含中文
                has_chinese_in_legend = False
                if hasattr(ax, 'get_legend'):
                    legend = ax.get_legend()
                    if legend:
                        for text in legend.get_texts():
                            if contains_chinese(text.get_text()):
                                has_chinese_in_legend = True
                                break
                
                # 如果包含中文或字体是SimHei，使用sans-serif
                if has_chinese_in_legend or font_family == 'SimHei':
                    legend_font.set_family('sans-serif')
                else:
                    legend_font.set_family(font_family)
                legend_font.set_size(legend_fontsize)
                
                # 图例列数和间距控制
                legend_ncol = self.legend_column_spin.value() if hasattr(self, 'legend_column_spin') else 1
                legend_columnspacing = self.legend_columnspacing_spin.value() if hasattr(self, 'legend_columnspacing_spin') else 2.0
                legend_labelspacing = self.legend_labelspacing_spin.value() if hasattr(self, 'legend_labelspacing_spin') else 0.5
                legend_handlelength = self.legend_handlelength_spin.value() if hasattr(self, 'legend_handlelength_spin') else 2.0
                
                ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                         ncol=legend_ncol, columnspacing=legend_columnspacing, 
                         labelspacing=legend_labelspacing, handlelength=legend_handlelength)
            
            # 垂直参考线（使用可自定义的样式）
            lines = self.parse_list_input(self.vertical_lines_input.toPlainText())
            line_color = self.vertical_line_color_input.text().strip() or 'gray'
            line_width = self.vertical_line_width_spin.value()
            line_style = self.vertical_line_style_combo.currentText()
            line_alpha = self.vertical_line_alpha_spin.value()
            for lx in lines: 
                ax.axvline(lx, color=line_color, linestyle=line_style, linewidth=line_width, alpha=line_alpha)

            # 坐标轴范围由matplotlib自动设置（与数据处理.py保持一致）
            
            # 添加纵横比控制（使用主菜单的出版质量样式控制参数）
            # 优先从 publication_style_panel 获取
            if hasattr(self, 'publication_style_panel') and self.publication_style_panel:
                config = self.publication_style_panel.get_config()
                ps = config.publication_style
                aspect_ratio = ps.aspect_ratio if ps else 0.6
            else:
                # 回退：从直接控件获取（向后兼容）
                aspect_ratio = self.aspect_ratio_spin.value() if hasattr(self, 'aspect_ratio_spin') else 0.6
            
            if aspect_ratio > 0:
                ax.set_box_aspect(aspect_ratio)
            else:
                ax.set_aspect('auto')
            
            # 强制布局更新 (解决裁切)
            win.canvas.figure.subplots_adjust(left=0.15, right=0.95, bottom=0.22, top=0.90)

            win.canvas.draw()
            # 确保窗口显示（如果已存在则保持位置）
            if not win.isVisible():
                win.show()
            else:
                win.raise_()  # 将窗口置于最前

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            traceback.print_exc()
    
    # --- 核心：导出数据 (保留原功能) ---
    def export_processed_data(self):
        try:
            folder = self.folder_input.text()
            save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
            if not save_dir: return

            skip = self.skip_rows_spin.value()
            files = sorted(glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt')))
            
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())

            count = 0
            # BE 参数
            is_be = self.be_check.isChecked()
            be_temp = self.be_temp_spin.value()

            for f in files:
                try:
                    x, y = self.read_data(f, skip, x_min_phys, x_max_phys) # 使用物理截断
                    
                    # 预处理流程 (与主绘图一致)
                    if is_be:
                        y = DataPreProcessor.apply_bose_einstein_correction(x, y, be_temp)

                    if self.baseline_als_check.isChecked():
                        b = DataPreProcessor.apply_baseline_als(y, self.lam_spin.value(), self.p_spin.value())
                        y = y - b
                        y[y < 0] = 0
                    
                    if self.normalization_combo.currentText() == 'snv':
                        y = DataPreProcessor.apply_snv(y)
                    
                    df = pd.DataFrame({'Wavenumber': x, 'Intensity': y})
                    out_name = "proc_" + os.path.basename(f)
                    df.to_csv(os.path.join(save_dir, out_name), index=False)
                    count += 1
                except: pass
            # 不再弹出“导出完成”的提示框
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def _safe_set_widget_value(self, widget_attr, setter_func):
        """安全地设置控件值，避免C++对象删除错误和属性不存在错误"""
        try:
            widget = getattr(self, widget_attr, None)
            if widget is not None:
                setter_func(widget)
        except (AttributeError, RuntimeError):
            # 如果属性不存在或控件已被删除，跳过设置
            pass
    
    def _safe_get_widget_value(self, widget_attr, getter_func, default=None):
        """安全地获取控件值，避免C++对象删除错误和属性不存在错误"""
        try:
            widget = getattr(self, widget_attr, None)
            if widget is not None:
                return getter_func(widget)
        except (AttributeError, RuntimeError):
            # 如果属性不存在或控件已被删除，返回默认值
            pass
        return default

    # --- 核心：参数保存与加载 ---
    def load_settings(self):
        # 1. 通用和预处理参数
        self._safe_set_widget_value('folder_input', lambda w: w.setText(self.settings.value("folder", "")))
        self._safe_set_widget_value('n_chars_spin', lambda w: w.setValue(int(self.settings.value("n_chars", 3))))
        self._safe_set_widget_value('skip_rows_spin', lambda w: w.setValue(int(self.settings.value("skip_rows", -1))))

        self._safe_set_widget_value('qc_check', lambda w: w.setChecked(self.settings.value("qc", False, type=bool)))
        self._safe_set_widget_value('qc_threshold_spin', lambda w: w.setValue(float(self.settings.value("qc_threshold", 5.0))))

        self._safe_set_widget_value('be_check', lambda w: w.setChecked(self.settings.value("be_check", False, type=bool)))
        self._safe_set_widget_value('be_temp_spin', lambda w: w.setValue(float(self.settings.value("be_temp", 300.0))))

        self._safe_set_widget_value('baseline_als_check', lambda w: w.setChecked(self.settings.value("asls", False, type=bool)))
        self._safe_set_widget_value('lam_spin', lambda w: w.setValue(float(self.settings.value("lam", 10000))))
        self._safe_set_widget_value('p_spin', lambda w: w.setValue(float(self.settings.value("p", 0.005))))
        self._safe_set_widget_value('baseline_poly_check', lambda w: w.setChecked(self.settings.value("baseline_poly_check", False, type=bool)))
        self._safe_set_widget_value('baseline_points_spin', lambda w: w.setValue(int(self.settings.value("baseline_points", 50))))
        self._safe_set_widget_value('baseline_poly_spin', lambda w: w.setValue(int(self.settings.value("baseline_poly", 3))))

        self._safe_set_widget_value('smoothing_check', lambda w: w.setChecked(self.settings.value("smooth_check", False, type=bool)))
        self._safe_set_widget_value('smoothing_window_spin', lambda w: w.setValue(int(self.settings.value("smooth_window", 15))))
        self._safe_set_widget_value('smoothing_poly_spin', lambda w: w.setValue(int(self.settings.value("smooth_poly", 3))))

        self._safe_set_widget_value('normalization_combo', lambda w: w.setCurrentText(self.settings.value("norm", "None")))
        
        # 二次函数拟合参数
        # 注意：二次函数拟合、二次导数、X轴翻转、显示X/Y轴数值已移至样式面板或删除

        # 2. 绘图模式和全局设置
        self._safe_set_widget_value('plot_mode_combo', lambda w: w.setCurrentText(self.settings.value("mode", "Normal Overlay")))
        self._safe_set_widget_value('plot_style_combo', lambda w: w.setCurrentText(self.settings.value("plot_style", "line")))
        # 注意：堆叠偏移已移至谱线扫描面板，不再需要单独加载
        self._safe_set_widget_value('global_y_scale_factor_spin', lambda w: w.setValue(float(self.settings.value("y_scale", 1.0))))
        
        # 标题控制（从面板加载，如果面板存在）
        # 注意：面板会在初始化时自动加载配置，这里只需要向后兼容旧控件
        self._safe_set_widget_value('main_title_input', lambda w: w.setText(self.settings.value("main_title", "")))
        self._safe_set_widget_value('main_title_font_spin', lambda w: w.setValue(int(self.settings.value("main_title_fontsize", 20))))
        self._safe_set_widget_value('main_title_pad_spin', lambda w: w.setValue(float(self.settings.value("main_title_pad", 10.0))))
        self._safe_set_widget_value('main_title_show_check', lambda w: w.setChecked(self.settings.value("main_title_show", True, type=bool)))

        self._safe_set_widget_value('xlabel_input', lambda w: w.setText(self.settings.value("xlabel_text", "Wavenumber ($\\mathrm{cm^{-1}}$)")))
        self._safe_set_widget_value('xlabel_font_spin', lambda w: w.setValue(int(self.settings.value("xlabel_fontsize", 20))))
        self._safe_set_widget_value('xlabel_pad_spin', lambda w: w.setValue(float(self.settings.value("xlabel_pad", 10.0))))
        self._safe_set_widget_value('xlabel_show_check', lambda w: w.setChecked(self.settings.value("xlabel_show", True, type=bool)))
        self._safe_set_widget_value('ylabel_input', lambda w: w.setText(self.settings.value("ylabel_text", "Transmittance")))
        self._safe_set_widget_value('ylabel_font_spin', lambda w: w.setValue(int(self.settings.value("ylabel_fontsize", 20))))
        self._safe_set_widget_value('ylabel_pad_spin', lambda w: w.setValue(float(self.settings.value("ylabel_pad", 10.0))))
        self._safe_set_widget_value('ylabel_show_check', lambda w: w.setChecked(self.settings.value("ylabel_show", True, type=bool)))

        # 注意：浓度梯度图相关设置已移至样式配置

        # 3. 物理截断
        self._safe_set_widget_value('x_min_phys_input', lambda w: w.setText(self.settings.value("x_min_phys", "")))
        self._safe_set_widget_value('x_max_phys_input', lambda w: w.setText(self.settings.value("x_max_phys", "")))

        # 4. 文件选择相关
        self._safe_set_widget_value('control_files_input', lambda w: w.setPlainText(self.settings.value("control_files", "")))
        self._safe_set_widget_value('groups_input', lambda w: w.setText(self.settings.value("groups_input", "")))
        self._safe_set_widget_value('nmf_average_check', lambda w: w.setChecked(self.settings.value("nmf_average_enabled", True, type=bool)))
        
        # 5. 出版质量样式（从面板加载，如果面板存在）
        # 注意：面板会在初始化时自动加载配置，这里只需要向后兼容旧控件
        self._safe_set_widget_value('fig_width_spin', lambda w: w.setValue(float(self.settings.value("fig_width", 10.0))))
        self._safe_set_widget_value('fig_height_spin', lambda w: w.setValue(float(self.settings.value("fig_height", 6.0))))
        self._safe_set_widget_value('fig_dpi_spin', lambda w: w.setValue(int(self.settings.value("fig_dpi", 300))))
        self._safe_set_widget_value('aspect_ratio_spin', lambda w: w.setValue(float(self.settings.value("aspect_ratio", 0.6))))  # 默认0.6

        # 加载风格预设选择（在加载完所有样式参数后）
        # 注意：如果面板存在，这些设置会通过面板自动加载
        if hasattr(self, 'style_preset_combo'):
            saved_preset = self.settings.value("style_preset", "默认")
            if saved_preset and self.style_preset_combo.findText(saved_preset) >= 0:
                # 临时断开信号，避免触发apply_style_preset
                self.style_preset_combo.blockSignals(True)
                self._safe_set_widget_value('style_preset_combo', lambda w: w.setCurrentText(saved_preset))
                self.style_preset_combo.blockSignals(False)

        self._safe_set_widget_value('axis_title_font_spin', lambda w: w.setValue(int(self.settings.value("axis_title_font", 20))))
        self._safe_set_widget_value('tick_label_font_spin', lambda w: w.setValue(int(self.settings.value("tick_label_font", 16))))
        self._safe_set_widget_value('legend_font_spin', lambda w: w.setValue(int(self.settings.value("legend_font", 10))))
        self._safe_set_widget_value('line_width_spin', lambda w: w.setValue(float(self.settings.value("line_width", 1.2))))
        self._safe_set_widget_value('line_style_combo', lambda w: w.setCurrentText(self.settings.value("line_style", "-")))
        self._safe_set_widget_value('font_family_combo', lambda w: w.setCurrentText(self.settings.value("font_family", "Times New Roman")))
        self._safe_set_widget_value('tick_direction_combo', lambda w: w.setCurrentText(self.settings.value("tick_direction", "in")))
        self._safe_set_widget_value('tick_len_major_spin', lambda w: w.setValue(int(self.settings.value("tick_len_major", 8))))
        self._safe_set_widget_value('tick_len_minor_spin', lambda w: w.setValue(int(self.settings.value("tick_len_minor", 4))))
        self._safe_set_widget_value('tick_width_spin', lambda w: w.setValue(float(self.settings.value("tick_width", 1.0))))
        self._safe_set_widget_value('show_grid_check', lambda w: w.setChecked(self.settings.value("show_grid", False, type=bool)))
        self._safe_set_widget_value('grid_alpha_spin', lambda w: w.setValue(float(self.settings.value("grid_alpha", 0.2))))
        self._safe_set_widget_value('shadow_alpha_spin', lambda w: w.setValue(float(self.settings.value("shadow_alpha", 0.25))))
        self._safe_set_widget_value('show_legend_check', lambda w: w.setChecked(self.settings.value("show_legend", True, type=bool)))
        self._safe_set_widget_value('legend_frame_check', lambda w: w.setChecked(self.settings.value("legend_frame", True, type=bool)))
        self._safe_set_widget_value('legend_loc_combo', lambda w: w.setCurrentText(self.settings.value("legend_loc", "best")))
        
        # 图例大小和间距控制
        self._safe_set_widget_value('legend_fontsize_spin', lambda w: w.setValue(int(self.settings.value("legend_fontsize", 10))))
        self._safe_set_widget_value('legend_column_spin', lambda w: w.setValue(int(self.settings.value("legend_column", 1))))
        self._safe_set_widget_value('legend_columnspacing_spin', lambda w: w.setValue(float(self.settings.value("legend_columnspacing", 2.0))))
        self._safe_set_widget_value('legend_labelspacing_spin', lambda w: w.setValue(float(self.settings.value("legend_labelspacing", 0.5))))
        self._safe_set_widget_value('legend_handlelength_spin', lambda w: w.setValue(float(self.settings.value("legend_handlelength", 2.0))))
        # 边框设置（向后兼容）
        self._safe_set_widget_value('spine_top_check', lambda w: w.setChecked(self.settings.value("spine_top", True, type=bool)))
        self._safe_set_widget_value('spine_bottom_check', lambda w: w.setChecked(self.settings.value("spine_bottom", True, type=bool)))
        self._safe_set_widget_value('spine_left_check', lambda w: w.setChecked(self.settings.value("spine_left", True, type=bool)))
        self._safe_set_widget_value('spine_right_check', lambda w: w.setChecked(self.settings.value("spine_right", True, type=bool)))
        self._safe_set_widget_value('spine_width_spin', lambda w: w.setValue(float(self.settings.value("spine_width", 2.0))))
        
        # 6. 高级设置（波峰检测、垂直参考线）
        self._safe_set_widget_value('peak_check', lambda w: w.setChecked(self.settings.value("peak_check", False, type=bool)))
        self._safe_set_widget_value('peak_height_spin', lambda w: w.setValue(float(self.settings.value("peak_height", 0.0))))  # 默认0表示自动
        self._safe_set_widget_value('peak_distance_spin', lambda w: w.setValue(int(self.settings.value("peak_distance", 10))))  # 减小默认值
        self._safe_set_widget_value('peak_prominence_spin', lambda w: w.setValue(float(self.settings.value("peak_prominence", 0.0))))  # 默认0表示禁用
        self._safe_set_widget_value('peak_width_spin', lambda w: w.setValue(float(self.settings.value("peak_width", 1.0))))
        self._safe_set_widget_value('peak_wlen_spin', lambda w: w.setValue(int(self.settings.value("peak_wlen", 200))))
        self._safe_set_widget_value('peak_rel_height_spin', lambda w: w.setValue(float(self.settings.value("peak_rel_height", 0.5))))
        self._safe_set_widget_value('peak_show_label_check', lambda w: w.setChecked(self.settings.value("peak_show_label", True, type=bool)))
        self._safe_set_widget_value('peak_label_font_combo', lambda w: w.setCurrentText(self.settings.value("peak_label_font", "Times New Roman")))
        self._safe_set_widget_value('peak_label_size_spin', lambda w: w.setValue(int(self.settings.value("peak_label_size", 10))))
        self._safe_set_widget_value('peak_label_color_input', lambda w: w.setText(self.settings.value("peak_label_color", "black")))
        self._safe_set_widget_value('peak_label_bold_check', lambda w: w.setChecked(self.settings.value("peak_label_bold", False, type=bool)))
        self._safe_set_widget_value('peak_label_rotation_spin', lambda w: w.setValue(float(self.settings.value("peak_label_rotation", 0.0))))
        self._safe_set_widget_value('peak_marker_shape_combo', lambda w: w.setCurrentText(self.settings.value("peak_marker_shape", "x")))
        self._safe_set_widget_value('peak_marker_size_spin', lambda w: w.setValue(int(self.settings.value("peak_marker_size", 10))))
        self._safe_set_widget_value('peak_marker_color_input', lambda w: w.setText(self.settings.value("peak_marker_color", "")))
        self._safe_set_widget_value('vertical_lines_input', lambda w: w.setPlainText(self.settings.value("vertical_lines", "")))
        self._safe_set_widget_value('vertical_line_color_input', lambda w: w.setText(self.settings.value("vertical_line_color", "gray")))
        self._safe_set_widget_value('vertical_line_width_spin', lambda w: w.setValue(float(self.settings.value("vertical_line_width", 0.8))))
        self._safe_set_widget_value('vertical_line_style_combo', lambda w: w.setCurrentText(self.settings.value("vertical_line_style", ":")))
        self._safe_set_widget_value('vertical_line_alpha_spin', lambda w: w.setValue(float(self.settings.value("vertical_line_alpha", 0.7))))

        # 7. NMF和物理拟合参数
        # NMF SVD去噪设置
        self._safe_set_widget_value('nmf_svd_denoise_check', lambda w: w.setChecked(self.settings.value("nmf_svd_denoise_enabled", False, type=bool)))
        self._safe_set_widget_value('nmf_svd_components_spin', lambda w: w.setValue(int(self.settings.value("nmf_svd_components", 5))))
        self._safe_set_widget_value('nmf_comp_spin', lambda w: w.setValue(int(self.settings.value("nmf_comp", 2))))
        self._safe_set_widget_value('nmf_max_iter', lambda w: w.setValue(int(self.settings.value("nmf_max_iter", 200))))
        self._safe_set_widget_value('nmf_top_title_input', lambda w: w.setText(self.settings.value("nmf_top_title", "Extracted Spectra (Components)")))
        self._safe_set_widget_value('nmf_bottom_title_input', lambda w: w.setText(self.settings.value("nmf_bottom_title", "Concentration Weights (vs. Sample)")))
        self._safe_set_widget_value('nmf_top_title_font_spin', lambda w: w.setValue(int(self.settings.value("nmf_top_title_fontsize", 16))))
        self._safe_set_widget_value('nmf_top_title_pad_spin', lambda w: w.setValue(float(self.settings.value("nmf_top_title_pad", 10.0))))
        self._safe_set_widget_value('nmf_top_title_show_check', lambda w: w.setChecked(self.settings.value("nmf_top_title_show", True, type=bool)))
        self._safe_set_widget_value('nmf_bottom_title_font_spin', lambda w: w.setValue(int(self.settings.value("nmf_bottom_title_fontsize", 16))))
        self._safe_set_widget_value('nmf_bottom_title_pad_spin', lambda w: w.setValue(float(self.settings.value("nmf_bottom_title_pad", 10.0))))
        self._safe_set_widget_value('nmf_bottom_title_show_check', lambda w: w.setChecked(self.settings.value("nmf_bottom_title_show", True, type=bool)))
        self._safe_set_widget_value('nmf_xlabel_top_input', lambda w: w.setText(self.settings.value("nmf_top_xlabel", "Wavenumber ($\\mathrm{cm^{-1}}$)")))
        self._safe_set_widget_value('nmf_top_xlabel_font_spin', lambda w: w.setValue(int(self.settings.value("nmf_top_xlabel_fontsize", 16))))
        self._safe_set_widget_value('nmf_top_xlabel_pad_spin', lambda w: w.setValue(float(self.settings.value("nmf_top_xlabel_pad", 10.0))))
        self._safe_set_widget_value('nmf_top_xlabel_show_check', lambda w: w.setChecked(self.settings.value("nmf_top_xlabel_show", True, type=bool)))

        self._safe_set_widget_value('nmf_ylabel_top_input', lambda w: w.setText(self.settings.value("nmf_top_ylabel", "Intensity (Arb. Unit)")))
        self._safe_set_widget_value('nmf_top_ylabel_font_spin', lambda w: w.setValue(int(self.settings.value("nmf_top_ylabel_fontsize", 16))))
        self._safe_set_widget_value('nmf_top_ylabel_pad_spin', lambda w: w.setValue(float(self.settings.value("nmf_top_ylabel_pad", 10.0))))
        self._safe_set_widget_value('nmf_top_ylabel_show_check', lambda w: w.setChecked(self.settings.value("nmf_top_ylabel_show", True, type=bool)))

        self._safe_set_widget_value('nmf_xlabel_bottom_input', lambda w: w.setText(self.settings.value("nmf_bottom_xlabel", "Sample Name")))
        self._safe_set_widget_value('nmf_bottom_xlabel_font_spin', lambda w: w.setValue(int(self.settings.value("nmf_bottom_xlabel_fontsize", 16))))
        self._safe_set_widget_value('nmf_bottom_xlabel_pad_spin', lambda w: w.setValue(float(self.settings.value("nmf_bottom_xlabel_pad", 10.0))))
        self._safe_set_widget_value('nmf_bottom_xlabel_show_check', lambda w: w.setChecked(self.settings.value("nmf_bottom_xlabel_show", True, type=bool)))

        self._safe_set_widget_value('nmf_ylabel_bottom_input', lambda w: w.setText(self.settings.value("nmf_bottom_ylabel", "Weight (Arb. Unit)")))
        self._safe_set_widget_value('nmf_bottom_ylabel_font_spin', lambda w: w.setValue(int(self.settings.value("nmf_bottom_ylabel_fontsize", 16))))
        self._safe_set_widget_value('nmf_bottom_ylabel_pad_spin', lambda w: w.setValue(float(self.settings.value("nmf_bottom_ylabel_pad", 10.0))))
        self._safe_set_widget_value('nmf_bottom_ylabel_show_check', lambda w: w.setChecked(self.settings.value("nmf_bottom_ylabel_show", True, type=bool)))
        self._safe_set_widget_value('nmf_sort_method_combo', lambda w: w.setCurrentText(self.settings.value("nmf_sort_method", "按文件名排序")))
        self._safe_set_widget_value('nmf_sort_reverse_check', lambda w: w.setChecked(self.settings.value("nmf_sort_reverse", False, type=bool)))
        self._safe_set_widget_value('nmf_include_control_check', lambda w: w.setChecked(self.settings.value("nmf_include_control", False, type=bool)))
        self._safe_set_widget_value('nmf_mode_standard', lambda w: w.setChecked(self.settings.value("nmf_mode_standard", True, type=bool)))
        self._safe_set_widget_value('nmf_mode_regression', lambda w: w.setChecked(self.settings.value("nmf_mode_regression", False, type=bool)))
        self.nmf_target_component_index = int(self.settings.value("nmf_target_component_index", 0))
        self._safe_set_widget_value('fit_cutoff_spin', lambda w: w.setValue(float(self.settings.value("fit_cutoff", 400.0))))
        self._safe_set_widget_value('fit_model_combo', lambda w: w.setCurrentText(self.settings.value("fit_model", "Lorentzian")))
        
        # 全局变换设置
        self._safe_set_widget_value('global_transform_combo', lambda w: w.setCurrentText(self.settings.value("global_transform", "无")))
        self._safe_set_widget_value('global_log_base_combo', lambda w: w.setCurrentText(self.settings.value("global_log_base", "10")))
        self._safe_set_widget_value('global_log_offset_spin', lambda w: w.setValue(float(self.settings.value("global_log_offset", 1.0))))
        self._safe_set_widget_value('global_sqrt_offset_spin', lambda w: w.setValue(float(self.settings.value("global_sqrt_offset", 0.0))))

        # 自动更新设置
        self._safe_set_widget_value('auto_update_check', lambda w: w.setChecked(self.settings.value("auto_update_enabled", True, type=bool)))


    def closeEvent(self, event):
        # 退出时保存所有参数（使用安全方法避免C++对象删除错误）
        
        # 1. 通用和预处理参数
        folder = self._safe_get_widget_value('folder_input', lambda w: w.text(), "")
        self.settings.setValue("folder", folder)
        self.settings.setValue("n_chars", self._safe_get_widget_value('n_chars_spin', lambda w: w.value(), 3))
        self.settings.setValue("skip_rows", self._safe_get_widget_value('skip_rows_spin', lambda w: w.value(), -1))
        self.settings.setValue("qc", self._safe_get_widget_value('qc_check', lambda w: w.isChecked(), False))
        self.settings.setValue("qc_threshold", self._safe_get_widget_value('qc_threshold_spin', lambda w: w.value(), 5.0))
        
        self.settings.setValue("be_check", self._safe_get_widget_value('be_check', lambda w: w.isChecked(), False))
        self.settings.setValue("be_temp", self._safe_get_widget_value('be_temp_spin', lambda w: w.value(), 300.0))

        self.settings.setValue("asls", self._safe_get_widget_value('baseline_als_check', lambda w: w.isChecked(), False))
        self.settings.setValue("lam", self._safe_get_widget_value('lam_spin', lambda w: w.value(), 1e5))
        self.settings.setValue("p", self._safe_get_widget_value('p_spin', lambda w: w.value(), 0.01))
        self.settings.setValue("baseline_poly_check", self._safe_get_widget_value('baseline_poly_check', lambda w: w.isChecked(), False))
        self.settings.setValue("baseline_points", self._safe_get_widget_value('baseline_points_spin', lambda w: w.value(), 10))
        self.settings.setValue("baseline_poly", self._safe_get_widget_value('baseline_poly_spin', lambda w: w.value(), 3))
        
        self.settings.setValue("smooth_check", self._safe_get_widget_value('smoothing_check', lambda w: w.isChecked(), False))
        self.settings.setValue("smooth_window", self._safe_get_widget_value('smoothing_window_spin', lambda w: w.value(), 5))
        self.settings.setValue("smooth_poly", self._safe_get_widget_value('smoothing_poly_spin', lambda w: w.value(), 3))

        self.settings.setValue("norm", self._safe_get_widget_value('normalization_combo', lambda w: w.currentText(), "None"))
        
        # 二次函数拟合参数
        # 注意：二次函数拟合相关设置已删除
        
        # 2. 绘图模式和全局设置
        self.settings.setValue("mode", self._safe_get_widget_value('plot_mode_combo', lambda w: w.currentText(), "Individual"))
        self.settings.setValue("plot_style", self._safe_get_widget_value('plot_style_combo', lambda w: w.currentText(), "Line"))
        # 注意：二次导数相关设置已删除
        self.settings.setValue("x_invert", self._safe_get_widget_value('x_axis_invert_check', lambda w: w.isChecked(), False))
        self.settings.setValue("show_y", self._safe_get_widget_value('show_y_val_check', lambda w: w.isChecked(), False))
        self.settings.setValue("show_x", self._safe_get_widget_value('show_x_val_check', lambda w: w.isChecked(), True))
        # 注意：堆叠偏移已移至谱线扫描面板，不再需要单独保存
        
        # 自动更新设置
        self.settings.setValue("auto_update_enabled", self._safe_get_widget_value('auto_update_check', lambda w: w.isChecked(), True))
        self.settings.setValue("y_scale", self._safe_get_widget_value('global_y_scale_factor_spin', lambda w: w.value(), 1.0))
        self.settings.setValue("main_title", self._safe_get_widget_value('main_title_input', lambda w: w.text(), ""))
        self.settings.setValue("main_title_fontsize", self._safe_get_widget_value('main_title_font_spin', lambda w: w.value(), 16))
        self.settings.setValue("main_title_pad", self._safe_get_widget_value('main_title_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("main_title_show", self._safe_get_widget_value('main_title_show_check', lambda w: w.isChecked(), True))
        
        # 浓度梯度图标题控制
        self.settings.setValue("gradient_title", self._safe_get_widget_value('gradient_title_input', lambda w: w.text(), ""))
        self.settings.setValue("gradient_title_fontsize", self._safe_get_widget_value('gradient_title_font_spin', lambda w: w.value(), 16))
        self.settings.setValue("gradient_title_pad", self._safe_get_widget_value('gradient_title_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("gradient_title_show", self._safe_get_widget_value('gradient_title_show_check', lambda w: w.isChecked(), True))
        
        # 3. X/Y 标签和物理截断
        self.settings.setValue("xlabel_text", self._safe_get_widget_value('xlabel_input', lambda w: w.text(), ""))
        self.settings.setValue("xlabel_fontsize", self._safe_get_widget_value('xlabel_font_spin', lambda w: w.value(), 14))
        self.settings.setValue("xlabel_pad", self._safe_get_widget_value('xlabel_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("xlabel_show", self._safe_get_widget_value('xlabel_show_check', lambda w: w.isChecked(), True))
        
        self.settings.setValue("ylabel_text", self._safe_get_widget_value('ylabel_input', lambda w: w.text(), ""))
        self.settings.setValue("ylabel_fontsize", self._safe_get_widget_value('ylabel_font_spin', lambda w: w.value(), 14))
        self.settings.setValue("ylabel_pad", self._safe_get_widget_value('ylabel_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("ylabel_show", self._safe_get_widget_value('ylabel_show_check', lambda w: w.isChecked(), True))
        
        # 注意：浓度梯度图相关设置已移至样式配置
        
        # 注意：浓度梯度图相关设置已移至样式配置
        self.settings.setValue("x_min_phys", self._safe_get_widget_value('x_min_phys_input', lambda w: w.text(), ""))
        self.settings.setValue("x_max_phys", self._safe_get_widget_value('x_max_phys_input', lambda w: w.text(), ""))
        
        # 4. 文件选择相关
        self.settings.setValue("control_files", self._safe_get_widget_value('control_files_input', lambda w: w.toPlainText(), ""))
        self.settings.setValue("groups_input", self._safe_get_widget_value('groups_input', lambda w: w.text(), ""))
        self.settings.setValue("nmf_average_enabled", self._safe_get_widget_value('nmf_average_check', lambda w: w.isChecked(), True))
        
        # 5. 出版质量样式参数（从面板保存）
        if hasattr(self, 'publication_style_panel'):
            config = self.publication_style_panel.get_config()
            ps = config.publication_style
            self.settings.setValue("fig_width", ps.fig_width)
            self.settings.setValue("fig_height", ps.fig_height)
            self.settings.setValue("fig_dpi", ps.fig_dpi)
            self.settings.setValue("aspect_ratio", ps.aspect_ratio)
            self.settings.setValue("axis_title_font", ps.axis_title_fontsize)
            self.settings.setValue("tick_label_font", ps.tick_label_fontsize)
            self.settings.setValue("legend_font", ps.legend_fontsize)
            self.settings.setValue("line_width", ps.line_width)
            self.settings.setValue("line_style", ps.line_style)
            self.settings.setValue("font_family", ps.font_family)
            self.settings.setValue("tick_direction", ps.tick_direction)
            self.settings.setValue("tick_len_major", ps.tick_len_major)
            self.settings.setValue("tick_len_minor", ps.tick_len_minor)
            self.settings.setValue("tick_width", ps.tick_width)
            self.settings.setValue("show_grid", ps.show_grid)
            self.settings.setValue("grid_alpha", ps.grid_alpha)
            self.settings.setValue("shadow_alpha", ps.shadow_alpha)
            self.settings.setValue("show_legend", ps.show_legend)
            self.settings.setValue("legend_frame", ps.legend_frame)
            self.settings.setValue("legend_loc", ps.legend_loc)
            self.settings.setValue("legend_fontsize", ps.legend_fontsize)
            self.settings.setValue("legend_column", ps.legend_ncol)
            self.settings.setValue("legend_columnspacing", ps.legend_columnspacing)
            self.settings.setValue("legend_labelspacing", ps.legend_labelspacing)
            self.settings.setValue("legend_handlelength", ps.legend_handlelength)
            self.settings.setValue("spine_top", ps.spine_top)
            self.settings.setValue("spine_bottom", ps.spine_bottom)
            self.settings.setValue("spine_left", ps.spine_left)
            self.settings.setValue("spine_right", ps.spine_right)
            self.settings.setValue("spine_width", ps.spine_width)
        else:
            # 向后兼容：如果面板不存在，尝试从旧控件保存
            self.settings.setValue("fig_width", self._safe_get_widget_value('fig_width_spin', lambda w: w.value(), 10.0))
            self.settings.setValue("fig_height", self._safe_get_widget_value('fig_height_spin', lambda w: w.value(), 6.0))
            self.settings.setValue("fig_dpi", self._safe_get_widget_value('fig_dpi_spin', lambda w: w.value(), 100))
            self.settings.setValue("aspect_ratio", self._safe_get_widget_value('aspect_ratio_spin', lambda w: w.value(), 1.0))
            self.settings.setValue("axis_title_font", self._safe_get_widget_value('axis_title_font_spin', lambda w: w.value(), 16))
            self.settings.setValue("tick_label_font", self._safe_get_widget_value('tick_label_font_spin', lambda w: w.value(), 12))
            self.settings.setValue("legend_font", self._safe_get_widget_value('legend_font_spin', lambda w: w.value(), 12))
            self.settings.setValue("line_width", self._safe_get_widget_value('line_width_spin', lambda w: w.value(), 1.5))
            self.settings.setValue("line_style", self._safe_get_widget_value('line_style_combo', lambda w: w.currentText(), "-"))
            self.settings.setValue("font_family", self._safe_get_widget_value('font_family_combo', lambda w: w.currentText(), "Arial"))
            self.settings.setValue("tick_direction", self._safe_get_widget_value('tick_direction_combo', lambda w: w.currentText(), "in"))
            self.settings.setValue("tick_len_major", self._safe_get_widget_value('tick_len_major_spin', lambda w: w.value(), 5.0))
            self.settings.setValue("tick_len_minor", self._safe_get_widget_value('tick_len_minor_spin', lambda w: w.value(), 3.0))
            self.settings.setValue("tick_width", self._safe_get_widget_value('tick_width_spin', lambda w: w.value(), 0.5))
            self.settings.setValue("show_grid", self._safe_get_widget_value('show_grid_check', lambda w: w.isChecked(), False))
            self.settings.setValue("grid_alpha", self._safe_get_widget_value('grid_alpha_spin', lambda w: w.value(), 0.3))
            self.settings.setValue("shadow_alpha", self._safe_get_widget_value('shadow_alpha_spin', lambda w: w.value(), 0.2))
            self.settings.setValue("show_legend", self._safe_get_widget_value('show_legend_check', lambda w: w.isChecked(), True))
            self.settings.setValue("legend_frame", self._safe_get_widget_value('legend_frame_check', lambda w: w.isChecked(), True))
            self.settings.setValue("legend_loc", self._safe_get_widget_value('legend_loc_combo', lambda w: w.currentText(), "best"))
            self.settings.setValue("legend_fontsize", self._safe_get_widget_value('legend_fontsize_spin', lambda w: w.value(), 12))
            self.settings.setValue("legend_column", self._safe_get_widget_value('legend_column_spin', lambda w: w.value(), 1))
            self.settings.setValue("legend_columnspacing", self._safe_get_widget_value('legend_columnspacing_spin', lambda w: w.value(), 1.0))
            self.settings.setValue("legend_labelspacing", self._safe_get_widget_value('legend_labelspacing_spin', lambda w: w.value(), 0.5))
            self.settings.setValue("legend_handlelength", self._safe_get_widget_value('legend_handlelength_spin', lambda w: w.value(), 2.0))
            self.settings.setValue("style_preset", self._safe_get_widget_value('style_preset_combo', lambda w: w.currentText(), "Default"))
            self.settings.setValue("spine_top", self._safe_get_widget_value('spine_top_check', lambda w: w.isChecked(), False))
            self.settings.setValue("spine_bottom", self._safe_get_widget_value('spine_bottom_check', lambda w: w.isChecked(), True))
            self.settings.setValue("spine_left", self._safe_get_widget_value('spine_left_check', lambda w: w.isChecked(), True))
            self.settings.setValue("spine_right", self._safe_get_widget_value('spine_right_check', lambda w: w.isChecked(), False))
            self.settings.setValue("spine_width", self._safe_get_widget_value('spine_width_spin', lambda w: w.value(), 0.5))
        
        # 6. 高级设置（波峰检测、垂直参考线）
        self.settings.setValue("peak_check", self._safe_get_widget_value('peak_check', lambda w: w.isChecked(), False))
        self.settings.setValue("peak_height", self._safe_get_widget_value('peak_height_spin', lambda w: w.value(), 0.0))
        self.settings.setValue("peak_distance", self._safe_get_widget_value('peak_distance_spin', lambda w: w.value(), 1))
        self.settings.setValue("peak_prominence", self._safe_get_widget_value('peak_prominence_spin', lambda w: w.value(), 0.0))
        self.settings.setValue("peak_width", self._safe_get_widget_value('peak_width_spin', lambda w: w.value(), 0))
        self.settings.setValue("peak_wlen", self._safe_get_widget_value('peak_wlen_spin', lambda w: w.value(), 1))
        self.settings.setValue("peak_rel_height", self._safe_get_widget_value('peak_rel_height_spin', lambda w: w.value(), 0.5))
        self.settings.setValue("peak_show_label", self._safe_get_widget_value('peak_show_label_check', lambda w: w.isChecked(), True))
        self.settings.setValue("peak_label_font", self._safe_get_widget_value('peak_label_font_combo', lambda w: w.currentText(), "Arial"))
        self.settings.setValue("peak_label_size", self._safe_get_widget_value('peak_label_size_spin', lambda w: w.value(), 10))
        self.settings.setValue("peak_label_color", self._safe_get_widget_value('peak_label_color_input', lambda w: w.text(), "#000000"))
        self.settings.setValue("peak_label_bold", self._safe_get_widget_value('peak_label_bold_check', lambda w: w.isChecked(), False))
        self.settings.setValue("peak_label_rotation", self._safe_get_widget_value('peak_label_rotation_spin', lambda w: w.value(), 0))
        self.settings.setValue("peak_marker_shape", self._safe_get_widget_value('peak_marker_shape_combo', lambda w: w.currentText(), "o"))
        self.settings.setValue("peak_marker_size", self._safe_get_widget_value('peak_marker_size_spin', lambda w: w.value(), 8))
        self.settings.setValue("peak_marker_color", self._safe_get_widget_value('peak_marker_color_input', lambda w: w.text(), "#ff0000"))
        self.settings.setValue("vertical_lines", self._safe_get_widget_value('vertical_lines_input', lambda w: w.toPlainText(), ""))
        self.settings.setValue("vertical_line_color", self._safe_get_widget_value('vertical_line_color_input', lambda w: w.text(), "#000000"))
        self.settings.setValue("vertical_line_width", self._safe_get_widget_value('vertical_line_width_spin', lambda w: w.value(), 1.0))
        self.settings.setValue("vertical_line_style", self._safe_get_widget_value('vertical_line_style_combo', lambda w: w.currentText(), "-"))
        self.settings.setValue("vertical_line_alpha", self._safe_get_widget_value('vertical_line_alpha_spin', lambda w: w.value(), 1.0))
        
        # 7. NMF和物理拟合参数
        # NMF SVD去噪设置
        self.settings.setValue("nmf_svd_denoise_enabled", self._safe_get_widget_value('nmf_svd_denoise_check', lambda w: w.isChecked(), False))
        self.settings.setValue("nmf_svd_components", self._safe_get_widget_value('nmf_svd_components_spin', lambda w: w.value(), 5))
        self.settings.setValue("nmf_comp", self._safe_get_widget_value('nmf_comp_spin', lambda w: w.value(), 3))
        self.settings.setValue("nmf_max_iter", self._safe_get_widget_value('nmf_max_iter', lambda w: w.value(), 200))
        # 保存NMF目标组分索引（如果窗口存在，从窗口获取最新值）
        if hasattr(self, 'nmf_window') and self.nmf_window is not None:
            try:
                if hasattr(self.nmf_window, 'get_target_component_index'):
                    self.nmf_target_component_index = self.nmf_window.get_target_component_index()
            except RuntimeError:
                pass  # 窗口可能已被删除
        self.settings.setValue("nmf_target_component_index", self.nmf_target_component_index)
        self.settings.setValue("nmf_top_title", self._safe_get_widget_value('nmf_top_title_input', lambda w: w.text(), ""))
        self.settings.setValue("nmf_bottom_title", self._safe_get_widget_value('nmf_bottom_title_input', lambda w: w.text(), ""))
        self.settings.setValue("nmf_top_title_fontsize", self._safe_get_widget_value('nmf_top_title_font_spin', lambda w: w.value(), 16))
        self.settings.setValue("nmf_top_title_pad", self._safe_get_widget_value('nmf_top_title_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("nmf_top_title_show", self._safe_get_widget_value('nmf_top_title_show_check', lambda w: w.isChecked(), True))
        self.settings.setValue("nmf_bottom_title_fontsize", self._safe_get_widget_value('nmf_bottom_title_font_spin', lambda w: w.value(), 16))
        self.settings.setValue("nmf_bottom_title_pad", self._safe_get_widget_value('nmf_bottom_title_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("nmf_bottom_title_show", self._safe_get_widget_value('nmf_bottom_title_show_check', lambda w: w.isChecked(), True))
        self.settings.setValue("nmf_top_xlabel", self._safe_get_widget_value('nmf_xlabel_top_input', lambda w: w.text(), ""))
        self.settings.setValue("nmf_top_xlabel_fontsize", self._safe_get_widget_value('nmf_top_xlabel_font_spin', lambda w: w.value(), 14))
        self.settings.setValue("nmf_top_xlabel_pad", self._safe_get_widget_value('nmf_top_xlabel_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("nmf_top_xlabel_show", self._safe_get_widget_value('nmf_top_xlabel_show_check', lambda w: w.isChecked(), True))
        
        self.settings.setValue("nmf_top_ylabel", self._safe_get_widget_value('nmf_ylabel_top_input', lambda w: w.text(), ""))
        self.settings.setValue("nmf_top_ylabel_fontsize", self._safe_get_widget_value('nmf_top_ylabel_font_spin', lambda w: w.value(), 14))
        self.settings.setValue("nmf_top_ylabel_pad", self._safe_get_widget_value('nmf_top_ylabel_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("nmf_top_ylabel_show", self._safe_get_widget_value('nmf_top_ylabel_show_check', lambda w: w.isChecked(), True))
        
        self.settings.setValue("nmf_bottom_xlabel", self._safe_get_widget_value('nmf_xlabel_bottom_input', lambda w: w.text(), ""))
        self.settings.setValue("nmf_bottom_xlabel_fontsize", self._safe_get_widget_value('nmf_bottom_xlabel_font_spin', lambda w: w.value(), 14))
        self.settings.setValue("nmf_bottom_xlabel_pad", self._safe_get_widget_value('nmf_bottom_xlabel_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("nmf_bottom_xlabel_show", self._safe_get_widget_value('nmf_bottom_xlabel_show_check', lambda w: w.isChecked(), True))
        
        self.settings.setValue("nmf_bottom_ylabel", self._safe_get_widget_value('nmf_ylabel_bottom_input', lambda w: w.text(), ""))
        self.settings.setValue("nmf_bottom_ylabel_fontsize", self._safe_get_widget_value('nmf_bottom_ylabel_font_spin', lambda w: w.value(), 14))
        self.settings.setValue("nmf_bottom_ylabel_pad", self._safe_get_widget_value('nmf_bottom_ylabel_pad_spin', lambda w: w.value(), 10.0))
        self.settings.setValue("nmf_bottom_ylabel_show", self._safe_get_widget_value('nmf_bottom_ylabel_show_check', lambda w: w.isChecked(), True))
        self.settings.setValue("nmf_sort_method", self._safe_get_widget_value('nmf_sort_method_combo', lambda w: w.currentText(), "按文件名排序"))
        self.settings.setValue("nmf_sort_reverse", self._safe_get_widget_value('nmf_sort_reverse_check', lambda w: w.isChecked(), False))
        # 保存NMF目标组分索引（如果窗口存在，从窗口获取最新值）
        if hasattr(self, 'nmf_window') and self.nmf_window is not None:
            try:
                if hasattr(self.nmf_window, 'get_target_component_index'):
                    self.nmf_target_component_index = self.nmf_window.get_target_component_index()
            except RuntimeError:
                pass  # 窗口可能已被删除
        self.settings.setValue("nmf_target_component_index", self.nmf_target_component_index)
        self.settings.setValue("fit_cutoff", self._safe_get_widget_value('fit_cutoff_spin', lambda w: w.value(), 400.0))
        self.settings.setValue("fit_model", self._safe_get_widget_value('fit_model_combo', lambda w: w.currentText(), "Lorentzian"))
        
        # 全局变换设置
        self.settings.setValue("global_transform", self._safe_get_widget_value('global_transform_combo', lambda w: w.currentText(), "无"))
        self.settings.setValue("global_log_base", self._safe_get_widget_value('global_log_base_combo', lambda w: w.currentText(), "10"))
        self.settings.setValue("global_log_offset", self._safe_get_widget_value('global_log_offset_spin', lambda w: w.value(), 1.0))
        self.settings.setValue("global_sqrt_offset", self._safe_get_widget_value('global_sqrt_offset_spin', lambda w: w.value(), 0.0))
        self.settings.setValue("global_y_offset", self._safe_get_widget_value('global_y_offset_spin', lambda w: w.value(), 0.0))
        
        super().closeEvent(event)
    
    def _update_nmf_sort_preview(self):
        """更新NMF文件排序预览"""
        folder = self.folder_input.text()
        if not folder or not os.path.isdir(folder):
            self.nmf_file_preview_list.clear()
            return
        
        files = glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt'))
        if not files:
            self.nmf_file_preview_list.clear()
            return
        
        # 获取当前已排除的文件列表（从列表中移除的项目）
        excluded_files = set()
        for i in range(self.nmf_file_preview_list.count()):
            item = self.nmf_file_preview_list.item(i)
            if item and item.data(256):  # 如果存在完整路径
                excluded_files.add(item.data(256))
        
        # 应用排序
        sorted_files = self._apply_nmf_file_sort(files)
        
        # 更新预览列表（保留已排除的文件标记）
        current_items = {}
        for i in range(self.nmf_file_preview_list.count()):
            item = self.nmf_file_preview_list.item(i)
            if item:
                full_path = item.data(256)
                if full_path:
                    current_items[full_path] = item
        
        self.nmf_file_preview_list.clear()
        for f in sorted_files:
            item = QListWidgetItem(os.path.basename(f))
            item.setData(256, f)  # 存储完整路径
            # 如果文件之前被标记为排除，可以在这里添加标记（可选）
            self.nmf_file_preview_list.addItem(item)
    
    def _show_nmf_file_context_menu(self, position):
        """显示NMF文件列表的右键菜单"""
        item = self.nmf_file_preview_list.itemAt(position)
        if item is None:
            return
        
        menu = QMenu(self)
        delete_action = menu.addAction("删除（不参与NMF）")
        action = menu.exec(self.nmf_file_preview_list.mapToGlobal(position))
        
        if action == delete_action:
            self._remove_selected_nmf_files()
    
    def _remove_selected_nmf_files(self):
        """从NMF文件预览列表中删除选中的文件"""
        selected_items = self.nmf_file_preview_list.selectedItems()
        if not selected_items:
            # 如果没有选中项，尝试删除当前项
            current_item = self.nmf_file_preview_list.currentItem()
            if current_item:
                selected_items = [current_item]
        
        if selected_items:
            for item in selected_items:
                row = self.nmf_file_preview_list.row(item)
                self.nmf_file_preview_list.takeItem(row)
    
    def _apply_nmf_file_sort(self, files):
        """应用NMF文件排序"""
        if not files:
            return files
        
        sort_method = self.nmf_sort_method_combo.currentText()
        reverse = self.nmf_sort_reverse_check.isChecked()
        
        if sort_method == '按文件名排序':
            # 使用自然排序（Windows风格），考虑数字的数值大小
            def natural_sort_key(filename):
                import re
                name = os.path.basename(filename).lower()
                # 将文件名分割成数字和非数字部分
                parts = re.split(r'(\d+)', name)
                # 将数字部分转换为整数，非数字部分保持原样
                return [int(part) if part.isdigit() else part for part in parts]
            sorted_files = sorted(files, key=natural_sort_key, reverse=reverse)
        elif sort_method == '按修改时间排序':
            sorted_files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=reverse)
        elif sort_method == '按文件大小排序':
            sorted_files = sorted(files, key=lambda x: os.path.getsize(x), reverse=reverse)
        elif sort_method == '自定义顺序':
            # 使用预览列表中的顺序
            sorted_files = []
            for i in range(self.nmf_file_preview_list.count()):
                item = self.nmf_file_preview_list.item(i)
                if item:
                    full_path = item.data(256)
                    if full_path and full_path in files:
                        sorted_files.append(full_path)
            # 添加预览列表中没有的文件
            for f in files:
                if f not in sorted_files:
                    sorted_files.append(f)
        else:
            sorted_files = sorted(files)
        
        return sorted_files
        
    def parse_list_input(self, text, data_type=float):
        """解析列表输入（逗号、换行或空格分隔的数字）"""
        text = text.strip()
        if not text: return []
        items = []
        # 使用正则表达式更稳健地分割
        for item in re.split(r'[,\n\s]+', text):
            item = item.strip()
            if item: 
                try: items.append(data_type(item))
                except: pass
        return items
    
    def _on_rruff_tolerance_changed(self, value):
        """RRUFF匹配容差改变时更新匹配器"""
        self.rruff_peak_matcher.tolerance = value
    
    def _match_rruff_spectra(self):
        """匹配RRUFF光谱"""
        if not self.rruff_loader or not self.rruff_loader.library_spectra:
            QMessageBox.warning(self, "警告", "请先加载RRUFF库")
            return
        
        # 更新匹配容差
        tolerance = self.rruff_match_tolerance_spin.value()
        self.rruff_peak_matcher.tolerance = tolerance
        
        # 检查是否有打开的绘图窗口
        if not self.plot_windows:
            QMessageBox.warning(self, "警告", "请先运行绘图以获取当前光谱数据")
            return
        
        try:
            # 获取当前绘图窗口中的第一个光谱数据作为查询光谱
            # 这里我们需要从绘图窗口获取数据，但绘图窗口可能没有存储原始数据
            # 所以我们需要从文件重新读取
            folder = self.folder_input.text()
            if not os.path.isdir(folder):
                QMessageBox.warning(self, "警告", "请先选择数据文件夹")
                return
            
            # 获取第一个分组的数据
            n_chars = self.n_chars_spin.value()
            files = self._get_sorted_files(folder)
            if not files:
                QMessageBox.warning(self, "警告", "未找到数据文件")
                return
            
            # 读取第一个文件作为查询光谱
            query_file = files[0]
            skip_rows = self.skip_rows_spin.value()
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())
            
            x_query, y_query = self.read_data(query_file, skip_rows, x_min_phys, x_max_phys)
            
            # 应用预处理
            y_proc = self._preprocess_spectrum(x_query, y_query)
            
            # 检测峰值（使用主菜单的峰值检测参数，降低阈值以检测更多小峰）
            from scipy.signal import find_peaks
            
            # 使用主菜单的峰值检测参数（允许极小的值以检测所有峰值）
            peak_height = self.peak_height_spin.value() if hasattr(self, 'peak_height_spin') else 0.0
            peak_distance = self.peak_distance_spin.value() if hasattr(self, 'peak_distance_spin') else 10
            peak_prominence = self.peak_prominence_spin.value() if hasattr(self, 'peak_prominence_spin') else None
            
            # 计算智能阈值
            y_max = np.max(y_proc) if len(y_proc) > 0 else 0
            y_min = np.min(y_proc) if len(y_proc) > 0 else 0
            y_range = y_max - y_min
            
            peak_kwargs = {}
            # 如果height为0，使用极低阈值（0.1%）；否则使用用户设置的值
            if peak_height == 0:
                if y_max > 0:
                    peak_height = y_max * 0.001  # 降低到0.1%以检测所有小峰
                else:
                    peak_height = 0
            # 只有当height明显不合理时才调整
            if peak_height > y_range * 2 and y_range > 0:
                peak_height = y_max * 0.001
            if peak_height != 0:
                peak_kwargs['height'] = peak_height
            
            # 如果distance为0，使用极低阈值（0.1%）；否则使用用户设置的值
            if peak_distance == 0:
                peak_distance = max(1, int(len(y_proc) * 0.001))  # 降低到0.1%，最小为1
            # 只有当distance明显不合理时才调整
            if peak_distance > len(y_proc) * 0.5:
                peak_distance = max(1, int(len(y_proc) * 0.001))
            peak_distance = max(1, peak_distance)  # 确保至少为1
            peak_kwargs['distance'] = peak_distance
            
            # 如果prominence不为0，使用用户设置的值
            if peak_prominence is not None and peak_prominence != 0:
                if peak_prominence > y_range * 2 and y_range > 0:
                    peak_prominence = y_range * 0.001  # 只有在明显不合理时才调整
                peak_kwargs['prominence'] = peak_prominence
            
            try:
                peaks, properties = find_peaks(y_proc, **peak_kwargs)
            except:
                # 如果参数错误，使用默认参数
                peaks, properties = find_peaks(y_proc, 
                                            height=y_max * 0.01 if y_max > 0 else 0,
                                            distance=max(1, int(len(y_proc) * 0.01)))
            
            peak_wavenumbers = x_query[peaks] if len(peaks) > 0 else np.array([])
            
            # 确保RRUFF库使用相同的峰值检测参数（在匹配前更新）
            peak_detection_params = {
                'peak_height_threshold': self.peak_height_spin.value() if hasattr(self, 'peak_height_spin') else 0.0,
                'peak_distance_min': self.peak_distance_spin.value() if hasattr(self, 'peak_distance_spin') else 10,
                'peak_prominence': self.peak_prominence_spin.value() if hasattr(self, 'peak_prominence_spin') else None,
                'peak_width': self.peak_width_spin.value() if hasattr(self, 'peak_width_spin') else None,
                'peak_wlen': self.peak_wlen_spin.value() if hasattr(self, 'peak_wlen_spin') else None,
                'peak_rel_height': self.peak_rel_height_spin.value() if hasattr(self, 'peak_rel_height_spin') else None,
            }
            # 如果峰值检测参数已改变，重新检测RRUFF库的峰值
            if hasattr(self.rruff_loader, 'peak_detection_params') and self.rruff_loader.peak_detection_params != peak_detection_params:
                for name, spectrum in self.rruff_loader.library_spectra.items():
                    if 'y_raw' in spectrum:
                        spectrum['peaks'] = self.rruff_loader._detect_peaks(
                            spectrum['x'], spectrum['y'], 
                            peak_detection_params=peak_detection_params
                        )
                self.rruff_loader.peak_detection_params = peak_detection_params
            
            # 匹配RRUFF光谱
            matches = self.rruff_peak_matcher.find_best_matches(
                x_query, y_proc, peak_wavenumbers, self.rruff_loader, top_k=20
            )
            
            # 保存匹配结果到当前分组
            if files:
                n_chars = self.n_chars_spin.value()
                first_file_basename = os.path.splitext(os.path.basename(files[0]))[0]
                current_group = first_file_basename[:n_chars] if len(first_file_basename) >= n_chars else first_file_basename
            else:
                current_group = "default"
            
            if not hasattr(self, 'rruff_match_results') or not isinstance(self.rruff_match_results, dict):
                self.rruff_match_results = {}
            self.rruff_match_results[current_group] = matches
            
            # 更新列表
            self.rruff_match_list.clear()
            # 获取当前分组名称（使用第一个文件的分组）
            if files:
                n_chars = self.n_chars_spin.value()
                first_file_basename = os.path.splitext(os.path.basename(files[0]))[0]
                current_group = first_file_basename[:n_chars] if len(first_file_basename) >= n_chars else first_file_basename
            else:
                current_group = "default"
            
            for i, match in enumerate(matches):
                name = match['name']
                score = match['match_score']
                item = QListWidgetItem(f"{i+1}. {name} (匹配度: {score:.2%})")
                item.setData(Qt.ItemDataRole.UserRole, name)
                # 检查是否在当前分组中已选中
                if current_group in self.selected_rruff_spectra and name in self.selected_rruff_spectra[current_group]:
                    item.setSelected(True)
                self.rruff_match_list.addItem(item)
            
            # 保存当前分组名称，用于后续操作
            self.current_rruff_group = current_group
            
            self.btn_rruff_match.setEnabled(True)
            # 不再弹出“找到多少匹配光谱”的提示框
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"匹配RRUFF光谱失败：{str(e)}")
            traceback.print_exc()
    
    def _preprocess_spectrum(self, x, y):
        """预处理单个光谱（与绘图逻辑一致）"""
        from src.core.preprocessor import DataPreProcessor
        
        y_proc = y.astype(float)
        
        # QC检查
        if self.qc_check.isChecked() and np.max(y_proc) < self.qc_threshold_spin.value():
            return y_proc
        
        # BE校正
        if self.be_check.isChecked():
            y_proc = DataPreProcessor.apply_bose_einstein_correction(x, y_proc, self.be_temp_spin.value())
        
        # 平滑
        if self.smoothing_check.isChecked():
            y_proc = DataPreProcessor.apply_smoothing(y_proc, self.smoothing_window_spin.value(), self.smoothing_poly_spin.value())
        
        # 基线校正
        if self.baseline_als_check.isChecked():
            b = DataPreProcessor.apply_baseline_als(y_proc, self.lam_spin.value(), self.p_spin.value())
            y_proc = y_proc - b
            y_proc[y_proc < 0] = 0
        
        # 归一化
        norm_mode = self.normalization_combo.currentText()
        if norm_mode == 'max':
            y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
        elif norm_mode == 'area':
            y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
        elif norm_mode == 'snv':
            y_proc = DataPreProcessor.apply_snv(y_proc)
        
        # 全局动态变换
        transform_mode = self.global_transform_combo.currentText()
        if transform_mode == '对数变换 (Log)':
            log_base = float(self.global_log_base_combo.currentText()) if self.global_log_base_combo.currentText() == '10' else np.e
            y_proc = DataPreProcessor.apply_log_transform(y_proc, base=log_base, offset=self.global_log_offset_spin.value())
        elif transform_mode == '平方根变换 (Sqrt)':
            y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=self.global_sqrt_offset_spin.value())
        
        # 二次导数
        if self.derivative_check.isChecked():
            d1 = np.gradient(y_proc, x)
            y_proc = np.gradient(d1, x)
        
        # 整体Y轴偏移
        if hasattr(self, 'global_y_offset_spin'):
            y_proc = y_proc + self.global_y_offset_spin.value()
        
        return y_proc
    
    def _on_rruff_item_double_clicked(self, item):
        """双击RRUFF匹配项时添加到绘图"""
        name = item.data(Qt.ItemDataRole.UserRole)
        if name:
            # 获取当前分组
            current_group = getattr(self, 'current_rruff_group', 'default')
            if current_group not in self.selected_rruff_spectra:
                self.selected_rruff_spectra[current_group] = set()
            
            if name in self.selected_rruff_spectra[current_group]:
                self.selected_rruff_spectra[current_group].remove(name)
            else:
                self.selected_rruff_spectra[current_group].add(name)
            self._update_plots_with_rruff()
    
    def _on_rruff_item_clicked(self, item):
        """RRUFF项目点击事件（检测Ctrl键）"""
        # 检测是否按下了Ctrl键
        modifiers = QApplication.keyboardModifiers()
        is_ctrl_click = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        
        # 存储Ctrl键状态，供_on_rruff_selection_changed使用
        self._is_ctrl_click_rruff = is_ctrl_click
    
    def _on_rruff_selection_changed(self):
        """RRUFF选择改变时更新（普通点击覆盖，Ctrl+点击叠加）"""
        selected_items = self.rruff_match_list.selectedItems()
        # 获取当前分组
        current_group = getattr(self, 'current_rruff_group', 'default')
        if current_group not in self.selected_rruff_spectra:
            self.selected_rruff_spectra[current_group] = set()
        
        # 检测是否按下了Ctrl键（从_on_rruff_item_clicked获取）
        is_ctrl_click = getattr(self, '_is_ctrl_click_rruff', False)
        
        # 如果不是Ctrl+点击，清除旧选择（覆盖模式）
        if not is_ctrl_click:
            self.selected_rruff_spectra[current_group] = set()
        
        # 添加新选择的项目
        selected_names = set(self.selected_rruff_spectra[current_group])  # 复制现有选择（如果Ctrl+点击）
        for item in selected_items:
            name = item.data(Qt.ItemDataRole.UserRole)
            if name:
                selected_names.add(name)
        
        self.selected_rruff_spectra[current_group] = selected_names
        self.btn_clear_rruff.setEnabled(len(selected_names) > 0)
        self._update_plots_with_rruff()
        
        # 重置Ctrl键状态
        self._is_ctrl_click_rruff = False
    
    def _clear_selected_rruff(self):
        """清除已选的RRUFF光谱"""
        current_group = getattr(self, 'current_rruff_group', 'default')
        if current_group in self.selected_rruff_spectra:
            self.selected_rruff_spectra[current_group].clear()
        for i in range(self.rruff_match_list.count()):
            self.rruff_match_list.item(i).setSelected(False)
        self.btn_clear_rruff.setEnabled(False)
        self._update_plots_with_rruff()
    
    def _update_plots_with_rruff(self):
        """更新绘图以包含选中的RRUFF光谱"""
        if not self.plot_windows or not self.rruff_loader:
            return
        
        # 重新运行绘图逻辑，会在绘图时添加RRUFF光谱
        self.run_plot_logic()

# -----------------------------------------------------------------
# 🚀 【程序入口】
# -----------------------------------------------------------------
if __name__ == "__main__":
    # 确保在运行应用程序之前设置了字体
    setup_matplotlib_fonts()
    
    # 1. 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 2. 创建主窗口实例
    ex = SpectraConfigDialog()
    
    # 3. 显示主窗口
    ex.show()
    
    # 4. 运行应用程序的主事件循环
    sys.exit(app.exec())

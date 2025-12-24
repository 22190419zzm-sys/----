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
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

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
    QHeaderView
)

from src.config.constants import C_H, C_C, C_K, C_CM_TO_HZ
from src.config.plot_config import PlotStyleConfig
from src.utils.fonts import setup_matplotlib_fonts
from src.utils.helpers import natural_sort_key, group_files_by_name
from src.core.preprocessor import DataPreProcessor
from src.core.generators import SyntheticDataGenerator
from src.core.matcher import SpectralMatcher
from src.core.transformers import AutoencoderTransformer, NonNegativeTransformer, AdaptiveMineralFilter
from src.ui.widgets.custom_widgets import CollapsibleGroupBox, SmartDoubleSpinBox
from src.ui.canvas import MplCanvas
from src.ui.windows.nmf_window import NMFResultWindow
from src.ui.windows.plot_window import MplPlotWindow


class NMFFitValidationWindow(QDialog):
    """NMF拟合验证窗口 - 显示原始光谱与拟合结果的对比（增强版）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NMF拟合验证")
        # 设置窗口图标
        try:
            from src.utils.icon_manager import set_window_icon
            set_window_icon(self)
        except:
            pass
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        
        # 样式配置
        self.style_config = PlotStyleConfig(self)
        self.style_params = self.style_config.load_style_params("NMFFitValidationWindow")
        
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 创建水平布局：左侧图表，右侧控制面板
        content_layout = QHBoxLayout()
        
        # 左侧：图表区域
        left_panel = QVBoxLayout()
        # 使用默认尺寸，实际尺寸由样式参数控制
        self.canvas = MplCanvas(self, width=12, height=8, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        left_panel.addWidget(self.toolbar)
        left_panel.addWidget(self.canvas)
        # 设置canvas可以扩展
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 右侧：控制面板（可滚动）
        right_panel = QVBoxLayout()
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(350)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_scroll.setWidget(right_widget)
        
        # 放大区域管理
        zoom_group = CollapsibleGroupBox("🔍 放大区域管理", is_expanded=True)
        zoom_layout = QVBoxLayout()
        
        self.zoom_regions_list = QListWidget()
        self.zoom_regions_list.setMaximumHeight(150)
        zoom_layout.addWidget(QLabel("已添加的放大区域:"))
        zoom_layout.addWidget(self.zoom_regions_list)
        
        # 添加新放大区域
        add_zoom_layout = QHBoxLayout()
        self.new_zoom_xmin = QDoubleSpinBox()
        self.new_zoom_xmin.setRange(-999999999.0, 999999999.0)
        self.new_zoom_xmin.setDecimals(15)
        self.new_zoom_xmin.setValue(1250)
        
        self.new_zoom_xmax = QDoubleSpinBox()
        self.new_zoom_xmax.setRange(-999999999.0, 999999999.0)
        self.new_zoom_xmax.setDecimals(15)
        self.new_zoom_xmax.setValue(1450)
        
        self.btn_add_zoom = QPushButton("添加")
        self.btn_add_zoom.clicked.connect(self.add_zoom_region)
        self.btn_remove_zoom = QPushButton("删除选中")
        self.btn_remove_zoom.clicked.connect(self.remove_zoom_region)
        
        add_zoom_layout.addWidget(QLabel("范围:"))
        add_zoom_layout.addWidget(self.new_zoom_xmin)
        add_zoom_layout.addWidget(QLabel("-"))
        add_zoom_layout.addWidget(self.new_zoom_xmax)
        add_zoom_layout.addWidget(self.btn_add_zoom)
        add_zoom_layout.addWidget(self.btn_remove_zoom)
        zoom_layout.addLayout(add_zoom_layout)
        
        zoom_group.setContentLayout(zoom_layout)
        right_panel.addWidget(zoom_group)
        
        # 样式配置面板
        style_group = CollapsibleGroupBox("样式配置（发表级设置）", is_expanded=False)
        style_layout = QFormLayout()
        
        # Figure/DPI
        self.fig_width_spin = QDoubleSpinBox()
        self.fig_width_spin.setRange(-999999999.0, 999999999.0)
        self.fig_width_spin.setDecimals(15)
        self.fig_width_spin.setValue(self.style_params['fig_width'])
        self.fig_width_spin.setSingleStep(0.1)
        self.fig_height_spin = QDoubleSpinBox()
        self.fig_height_spin.setRange(-999999999.0, 999999999.0)
        self.fig_height_spin.setDecimals(15)
        self.fig_height_spin.setValue(self.style_params['fig_height'])
        self.fig_height_spin.setSingleStep(0.1)
        self.fig_dpi_spin = QSpinBox()
        self.fig_dpi_spin.setRange(-999999999, 999999999)
        self.fig_dpi_spin.setValue(self.style_params['fig_dpi'])
        style_layout.addRow("图尺寸 W/H:", self._create_h_layout([self.fig_width_spin, self.fig_height_spin]))
        style_layout.addRow("DPI:", self.fig_dpi_spin)
        
        # Font
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(['Times New Roman', 'Arial', 'SimHei'])
        self.font_family_combo.setCurrentText(self.style_params['font_family'])
        
        self.axis_title_font_spin = QSpinBox()
        self.axis_title_font_spin.setRange(-999999999, 999999999)
        self.axis_title_font_spin.setValue(self.style_params['axis_title_fontsize'])
        self.tick_label_font_spin = QSpinBox()
        self.tick_label_font_spin.setRange(-999999999, 999999999)
        self.tick_label_font_spin.setValue(self.style_params['tick_label_fontsize'])
        self.legend_font_spin = QSpinBox()
        self.legend_font_spin.setRange(-999999999, 999999999)
        self.legend_font_spin.setValue(self.style_params['legend_fontsize'])
        self.title_font_spin = QSpinBox()
        self.title_font_spin.setRange(-999999999, 999999999)
        self.title_font_spin.setValue(self.style_params['title_fontsize'])
        # X轴标签字体大小（专门用于分类结果图的X轴）
        self.xaxis_label_font_spin = QSpinBox()
        self.xaxis_label_font_spin.setRange(-999999999, 999999999)
        self.xaxis_label_font_spin.setValue(self.style_params.get('xaxis_label_fontsize', 10))
        
        style_layout.addRow("字体家族:", self.font_family_combo)
        style_layout.addRow("字体大小 (轴/刻度/图例/标题):", 
                           self._create_h_layout([self.axis_title_font_spin, self.tick_label_font_spin, 
                                                 self.legend_font_spin, self.title_font_spin]))
        style_layout.addRow("X轴标签字体大小:", self.xaxis_label_font_spin)
        
        # Lines
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(-999999999.0, 999999999.0)
        self.line_width_spin.setDecimals(15)
        self.line_width_spin.setValue(self.style_params['line_width'])
        self.line_style_combo = QComboBox()
        self.line_style_combo.addItems(['-', '--', ':', '-.'])
        self.line_style_combo.setCurrentText(self.style_params['line_style'])
        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(-999999999.0, 999999999.0)
        self.marker_size_spin.setDecimals(15)
        self.marker_size_spin.setValue(self.style_params['marker_size'])
        style_layout.addRow("线宽 / 线型:", self._create_h_layout([self.line_width_spin, self.line_style_combo]))
        style_layout.addRow("标记大小:", self.marker_size_spin)
        
        # Ticks
        self.tick_direction_combo = QComboBox()
        self.tick_direction_combo.addItems(['in', 'out'])
        self.tick_direction_combo.setCurrentText(self.style_params['tick_direction'])
        self.tick_len_major_spin = QSpinBox()
        self.tick_len_major_spin.setRange(-999999999, 999999999)
        self.tick_len_major_spin.setValue(self.style_params['tick_len_major'])
        self.tick_len_minor_spin = QSpinBox()
        self.tick_len_minor_spin.setRange(-999999999, 999999999)
        self.tick_len_minor_spin.setValue(self.style_params['tick_len_minor'])
        self.tick_width_spin = QDoubleSpinBox()
        self.tick_width_spin.setRange(-999999999.0, 999999999.0)
        self.tick_width_spin.setDecimals(15)
        self.tick_width_spin.setValue(self.style_params['tick_width'])
        style_layout.addRow("刻度方向 / 宽度:", self._create_h_layout([self.tick_direction_combo, self.tick_width_spin]))
        style_layout.addRow("刻度长度 (大/小):", self._create_h_layout([self.tick_len_major_spin, self.tick_len_minor_spin]))
        
        # 纵横比控制
        self.aspect_ratio_spin = QDoubleSpinBox()
        self.aspect_ratio_spin.setRange(-999999999.0, 999999999.0)
        self.aspect_ratio_spin.setDecimals(15)
        self.aspect_ratio_spin.setValue(self.style_params.get('aspect_ratio', 0.0))
        style_layout.addRow("纵横比 (0=自动):", self.aspect_ratio_spin)
        
        # Grid
        self.show_grid_check = QCheckBox("显示网格")
        self.show_grid_check.setChecked(self.style_params['show_grid'])
        self.grid_alpha_spin = QDoubleSpinBox()
        self.grid_alpha_spin.setRange(-999999999.0, 999999999.0)
        self.grid_alpha_spin.setDecimals(15)
        self.grid_alpha_spin.setValue(self.style_params['grid_alpha'])
        style_layout.addRow(self._create_h_layout([self.show_grid_check, QLabel("透明度:"), self.grid_alpha_spin]))
        
        # Spines
        self.spine_top_check = QCheckBox("Top")
        self.spine_top_check.setChecked(self.style_params['spine_top'])
        self.spine_bottom_check = QCheckBox("Bottom")
        self.spine_bottom_check.setChecked(self.style_params['spine_bottom'])
        self.spine_left_check = QCheckBox("Left")
        self.spine_left_check.setChecked(self.style_params['spine_left'])
        self.spine_right_check = QCheckBox("Right")
        self.spine_right_check.setChecked(self.style_params['spine_right'])
        self.spine_width_spin = QDoubleSpinBox()
        self.spine_width_spin.setRange(-999999999.0, 999999999.0)
        self.spine_width_spin.setDecimals(15)
        self.spine_width_spin.setValue(self.style_params['spine_width'])
        style_layout.addRow("边框 (T/B/L/R):", self._create_h_layout([self.spine_top_check, self.spine_bottom_check, 
                                                                     self.spine_left_check, self.spine_right_check]))
        style_layout.addRow("边框线宽:", self.spine_width_spin)
        
        # Legend
        self.show_legend_check = QCheckBox("显示图例")
        self.show_legend_check.setChecked(self.style_params['show_legend'])
        self.legend_frame_check = QCheckBox("图例边框")
        self.legend_frame_check.setChecked(self.style_params['legend_frame'])
        self.legend_loc_combo = QComboBox()
        self.legend_loc_combo.addItems(['best', 'upper right', 'upper left', 'lower left', 'lower right', 
                                       'center left', 'center right', 'lower center', 'upper center', 'center'])
        self.legend_loc_combo.setCurrentText(self.style_params['legend_loc'])
        style_layout.addRow(self._create_h_layout([self.show_legend_check, self.legend_frame_check]))
        style_layout.addRow("图例位置:", self.legend_loc_combo)
        
        # Colors
        self.color_raw_input = QLineEdit(self.style_params['color_raw'])
        self.color_fit_input = QLineEdit(self.style_params['color_fit'])
        self.color_residual_input = QLineEdit(self.style_params['color_residual'])
        style_layout.addRow("原始数据颜色:", self._create_h_layout([self.color_raw_input, self._create_color_picker_button(self.color_raw_input)]))
        style_layout.addRow("拟合线颜色:", self._create_h_layout([self.color_fit_input, self._create_color_picker_button(self.color_fit_input)]))
        style_layout.addRow("残差颜色:", self._create_h_layout([self.color_residual_input, self._create_color_picker_button(self.color_residual_input)]))
        
        # 连接颜色控件到自动更新
        self.color_raw_input.textChanged.connect(self._on_fit_validation_color_changed)
        self.color_fit_input.textChanged.connect(self._on_fit_validation_color_changed)
        self.color_residual_input.textChanged.connect(self._on_fit_validation_color_changed)
        
        # Text labels
        self.title_text_input = QLineEdit(self.style_params.get('title_text', ''))
        self.title_text_input.setPlaceholderText("留空则使用默认标题")
        
        # NMF拟合验证窗口标题控制：大小、间距、显示/隐藏
        self.validation_title_font_spin = QSpinBox()
        self.validation_title_font_spin.setRange(-999999999, 999999999)
        self.validation_title_font_spin.setValue(self.style_params.get('validation_title_fontsize', self.style_params.get('title_fontsize', 18)))
        
        self.validation_title_pad_spin = QDoubleSpinBox()
        self.validation_title_pad_spin.setRange(-999999999.0, 999999999.0)
        self.validation_title_pad_spin.setDecimals(15)
        self.validation_title_pad_spin.setValue(self.style_params.get('validation_title_pad', 10.0))
        
        self.validation_title_show_check = QCheckBox("显示图表标题")
        self.validation_title_show_check.setChecked(self.style_params.get('validation_title_show', True))
        
        self.xlabel_text_input = QLineEdit(self.style_params.get('xlabel_text', 'Wavenumber (cm⁻¹)'))
        self.ylabel_main_input = QLineEdit(self.style_params.get('ylabel_main_text', 'Intensity'))
        self.ylabel_residual_input = QLineEdit(self.style_params.get('ylabel_residual_text', 'Residuals'))
        self.legend_raw_label_input = QLineEdit(self.style_params.get('legend_raw_label', 'Raw Low-Conc. Spectrum'))
        self.legend_fit_label_input = QLineEdit(self.style_params.get('legend_fit_label', 'Fitted Organic Contribution'))
        self.legend_residual_label_input = QLineEdit(self.style_params.get('legend_residual_label', 'Residuals'))
        style_layout.addRow("图表标题:", self.title_text_input)
        style_layout.addRow("图例 - 原始光谱:", self.legend_raw_label_input)
        style_layout.addRow("图例 - 拟合结果:", self.legend_fit_label_input)
        style_layout.addRow("图例 - 残差:", self.legend_residual_label_input)
        style_layout.addRow("标题控制:", self._create_h_layout([self.validation_title_show_check, QLabel("大小:"), self.validation_title_font_spin, QLabel("间距:"), self.validation_title_pad_spin]))
        style_layout.addRow("X轴标签:", self.xlabel_text_input)
        
        # NMF拟合验证窗口X轴标题控制：大小、间距、显示/隐藏
        self.validation_xlabel_font_spin = QSpinBox()
        self.validation_xlabel_font_spin.setRange(-999999999, 999999999)
        self.validation_xlabel_font_spin.setValue(self.style_params.get('validation_xlabel_fontsize', self.style_params.get('axis_title_fontsize', 20)))
        
        self.validation_xlabel_pad_spin = QDoubleSpinBox()
        self.validation_xlabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.validation_xlabel_pad_spin.setDecimals(15)
        self.validation_xlabel_pad_spin.setValue(self.style_params.get('validation_xlabel_pad', 10.0))
        
        self.validation_xlabel_show_check = QCheckBox("显示X轴标题")
        self.validation_xlabel_show_check.setChecked(self.style_params.get('validation_xlabel_show', True))
        
        style_layout.addRow("X轴标题控制:", self._create_h_layout([self.validation_xlabel_show_check, QLabel("大小:"), self.validation_xlabel_font_spin, QLabel("间距:"), self.validation_xlabel_pad_spin]))
        
        style_layout.addRow("Y轴标签 (主图/残差):", self._create_h_layout([self.ylabel_main_input, self.ylabel_residual_input]))
        
        # NMF拟合验证窗口Y轴标题控制：大小、间距、显示/隐藏（主图和残差图共用）
        self.validation_ylabel_font_spin = QSpinBox()
        self.validation_ylabel_font_spin.setRange(-999999999, 999999999)
        self.validation_ylabel_font_spin.setValue(self.style_params.get('validation_ylabel_fontsize', self.style_params.get('axis_title_fontsize', 20)))
        
        self.validation_ylabel_pad_spin = QDoubleSpinBox()
        self.validation_ylabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.validation_ylabel_pad_spin.setDecimals(15)
        self.validation_ylabel_pad_spin.setValue(self.style_params.get('validation_ylabel_pad', 10.0))
        
        self.validation_ylabel_show_check = QCheckBox("显示Y轴标题")
        self.validation_ylabel_show_check.setChecked(self.style_params.get('validation_ylabel_show', True))
        
        style_layout.addRow("Y轴标题控制:", self._create_h_layout([self.validation_ylabel_show_check, QLabel("大小:"), self.validation_ylabel_font_spin, QLabel("间距:"), self.validation_ylabel_pad_spin]))
        style_layout.addRow("图例标签 (原始/拟合):", self._create_h_layout([self.legend_raw_label_input, self.legend_fit_label_input]))
        
        # A/B Labels
        self.show_label_a_check = QCheckBox("显示 (A) 标签")
        self.show_label_a_check.setChecked(self.style_params.get('show_label_a', True))
        self.show_label_b_check = QCheckBox("显示 (B) 标签")
        self.show_label_b_check.setChecked(self.style_params.get('show_label_b', True))
        self.label_a_text_input = QLineEdit(self.style_params.get('label_a_text', '(A)'))
        self.label_b_text_input = QLineEdit(self.style_params.get('label_b_text', '(B)'))
        style_layout.addRow(self._create_h_layout([self.show_label_a_check, self.show_label_b_check]))
        style_layout.addRow("标签文本 (A/B):", self._create_h_layout([self.label_a_text_input, self.label_b_text_input]))
        
        # 堆叠偏移和单独预处理
        preprocess_group = CollapsibleGroupBox("数据预处理与偏移", is_expanded=False)
        preprocess_layout = QFormLayout()
        
        # 全局堆叠偏移
        self.global_stack_offset_spin = QDoubleSpinBox()
        self.global_stack_offset_spin.setRange(-999999999.0, 999999999.0)
        self.global_stack_offset_spin.setDecimals(15)
        self.global_stack_offset_spin.setValue(0.0)
        preprocess_layout.addRow("全局堆叠偏移:", self.global_stack_offset_spin)
        
        # 全局缩放因子
        self.global_scale_factor_spin = QDoubleSpinBox()
        self.global_scale_factor_spin.setRange(-999999999.0, 999999999.0)
        self.global_scale_factor_spin.setDecimals(15)
        self.global_scale_factor_spin.setValue(1.0)
        preprocess_layout.addRow("全局缩放因子:", self.global_scale_factor_spin)
        
        # 原始数据独立Y轴调整
        self.raw_scale_spin = QDoubleSpinBox()
        self.raw_scale_spin.setRange(-999999999.0, 999999999.0)
        self.raw_scale_spin.setDecimals(15)
        self.raw_scale_spin.setValue(1.0)
        self.raw_offset_spin = QDoubleSpinBox()
        self.raw_offset_spin.setRange(-999999999.0, 999999999.0)
        self.raw_offset_spin.setDecimals(15)
        self.raw_offset_spin.setValue(0.0)
        preprocess_layout.addRow("原始数据 (缩放/偏移):", self._create_h_layout([self.raw_scale_spin, self.raw_offset_spin]))
        
        # 拟合数据独立Y轴调整
        self.fit_scale_spin = QDoubleSpinBox()
        self.fit_scale_spin.setRange(-999999999.0, 999999999.0)
        self.fit_scale_spin.setDecimals(15)
        self.fit_scale_spin.setValue(1.0)
        self.fit_offset_spin = QDoubleSpinBox()
        self.fit_offset_spin.setRange(-999999999.0, 999999999.0)
        self.fit_offset_spin.setDecimals(15)
        self.fit_offset_spin.setValue(0.0)
        preprocess_layout.addRow("拟合数据 (缩放/偏移):", self._create_h_layout([self.fit_scale_spin, self.fit_offset_spin]))
        
        # 二阶导数
        self.is_derivative_check = QCheckBox("应用二阶导数")
        self.is_derivative_check.setChecked(False)
        preprocess_layout.addRow(self.is_derivative_check)
        
        preprocess_group.setContentLayout(preprocess_layout)
        right_panel.addWidget(preprocess_group)
        
        # 同步主窗口参数按钮
        sync_btn_layout = QHBoxLayout()
        self.btn_sync_main = QPushButton("🔄 同步主窗口参数")
        self.btn_sync_main.clicked.connect(self.sync_main_window_params)
        sync_btn_layout.addWidget(self.btn_sync_main)
        sync_btn_layout.addStretch()
        style_layout.addRow("", sync_btn_layout)
        
        # 更新图表按钮
        update_btn_layout = QHBoxLayout()
        self.btn_update_plot = QPushButton("🔄 更新图表")
        self.btn_update_plot.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_update_plot.clicked.connect(self.update_plot)
        update_btn_layout.addWidget(self.btn_update_plot)
        update_btn_layout.addStretch()
        style_layout.addRow("", update_btn_layout)
        
        style_group.setContentLayout(style_layout)
        right_panel.addWidget(style_group)
        right_panel.addStretch()
        
        # 组装布局
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        content_layout.addWidget(left_widget, stretch=3)
        content_layout.addWidget(right_scroll, stretch=0)
        
        self.main_layout.addLayout(content_layout)
        
        # 存储数据
        self.x_data = None
        self.y_raw = None  # 原始数据（未处理）
        self.y_raw_processed = None  # 预处理后的原始数据
        self.y_fit = None
        self.y_fit_processed = None  # 应用偏移后的拟合数据
        self.y_total_reconstructed = None
        self.sample_name = ""
        self.zoom_regions = []  # 存储放大区域列表 [(xmin, xmax), ...]
        self.inset_axes_list = []  # 存储插入轴对象列表
        self.vertical_lines = []  # 垂直参考线列表
        self.peak_detection_enabled = False  # 峰值检测开关
        self.peak_height_threshold = 0.0  # 峰值检测高度阈值
        self.peak_distance_min = 10  # 峰值检测最小距离
        self.control_data_list = []  # 对照组数据列表
        
        # 保存axes引用（避免每次重新创建）
        self.ax_main = None
        self.ax_residual = None
        self.gs = None
        
        # 预处理和偏移参数
        self.global_stack_offset = 0.0
        self.global_scale_factor = 1.0
        self.raw_scale = 1.0
        self.raw_offset = 0.0
        self.fit_scale = 1.0
        self.fit_offset = 0.0
        self.is_derivative = False
        
        # 保存窗口位置和大小（像主程序一样）
        self.last_geometry = None
        self.is_first_plot = True  # 标记是否是第一次绘图
    
    def _create_h_layout(self, widgets):
        """创建水平布局的辅助方法"""
        h_layout = QHBoxLayout()
        for widget in widgets:
            h_layout.addWidget(widget)
        h_layout.addStretch(1)
        return h_layout
    
    def _create_color_picker_button(self, color_input):
        """创建颜色选择器按钮的辅助方法"""
        color_button = QPushButton("选择颜色")
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
    
    def _on_fit_validation_color_changed(self):
        """拟合验证窗口颜色变化时的回调函数（自动更新图表）"""
        # 只有在数据已存在时才自动更新
        if self.x_data is not None and self.y_raw is not None and self.y_fit is not None:
            # 使用QTimer延迟更新，避免频繁触发（防抖）
            if not hasattr(self, '_fit_validation_update_timer'):
                self._fit_validation_update_timer = QTimer()
                self._fit_validation_update_timer.setSingleShot(True)
                self._fit_validation_update_timer.timeout.connect(self.update_plot)
            
            # 重置定时器，300ms后执行更新
            self._fit_validation_update_timer.stop()
            self._fit_validation_update_timer.start(300)
    
    def _create_h_layout(self, widgets):
        """创建水平布局的辅助方法"""
        h_layout = QHBoxLayout()
        for widget in widgets:
            h_layout.addWidget(widget)
        h_layout.addStretch(1)
        return h_layout
    
    def add_zoom_region(self):
        """添加新的放大区域"""
        xmin = self.new_zoom_xmin.value()
        xmax = self.new_zoom_xmax.value()
        if xmax <= xmin:
            QMessageBox.warning(self, "错误", "最大值必须大于最小值。")
            return
        
        # 检查是否已存在
        for region in self.zoom_regions:
            if abs(region[0] - xmin) < 1 and abs(region[1] - xmax) < 1:
                QMessageBox.warning(self, "提示", "该放大区域已存在。")
                return
        
        self.zoom_regions.append((xmin, xmax))
        self.zoom_regions_list.addItem(f"{xmin:.0f} - {xmax:.0f} cm⁻¹")
        self.update_plot()
    
    def remove_zoom_region(self):
        """删除选中的放大区域"""
        current_item = self.zoom_regions_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "提示", "请先选择一个要删除的放大区域。")
            return
        
        row = self.zoom_regions_list.row(current_item)
        if 0 <= row < len(self.zoom_regions):
            self.zoom_regions.pop(row)
            self.zoom_regions_list.takeItem(row)
            self.update_plot()
    
    def sync_main_window_params(self):
        """从主窗口同步样式参数"""
        if self.parent() and hasattr(self.parent(), 'parent_dialog'):
            main_dialog = self.parent().parent_dialog
            if main_dialog:
                try:
                    # 同步图尺寸和DPI
                    if hasattr(main_dialog, 'fig_width_spin'):
                        self.fig_width_spin.setValue(main_dialog.fig_width_spin.value())
                        self.fig_height_spin.setValue(main_dialog.fig_height_spin.value())
                        self.fig_dpi_spin.setValue(main_dialog.fig_dpi_spin.value())
                    
                    # 同步字体设置
                    if hasattr(main_dialog, 'font_family_combo'):
                        font_family = main_dialog.font_family_combo.currentText()
                        index = self.font_family_combo.findText(font_family)
                        if index >= 0:
                            self.font_family_combo.setCurrentIndex(index)
                    
                    if hasattr(main_dialog, 'axis_title_font_spin'):
                        self.axis_title_font_spin.setValue(main_dialog.axis_title_font_spin.value())
                        self.tick_label_font_spin.setValue(main_dialog.tick_label_font_spin.value())
                        self.legend_font_spin.setValue(main_dialog.legend_font_spin.value())
                    
                    # 同步线条设置
                    if hasattr(main_dialog, 'line_width_spin'):
                        self.line_width_spin.setValue(main_dialog.line_width_spin.value())
                        index = self.line_style_combo.findText(main_dialog.line_style_combo.currentText())
                        if index >= 0:
                            self.line_style_combo.setCurrentIndex(index)
                    
                    # 同步刻度设置
                    if hasattr(main_dialog, 'tick_direction_combo'):
                        index = self.tick_direction_combo.findText(main_dialog.tick_direction_combo.currentText())
                        if index >= 0:
                            self.tick_direction_combo.setCurrentIndex(index)
                        self.tick_len_major_spin.setValue(main_dialog.tick_len_major_spin.value())
                        self.tick_len_minor_spin.setValue(main_dialog.tick_len_minor_spin.value())
                        self.tick_width_spin.setValue(main_dialog.tick_width_spin.value())
                    
                    # 同步网格设置
                    if hasattr(main_dialog, 'show_grid_check'):
                        self.show_grid_check.setChecked(main_dialog.show_grid_check.isChecked())
                        self.grid_alpha_spin.setValue(main_dialog.grid_alpha_spin.value())
                    
                    # 同步边框设置
                    if hasattr(main_dialog, 'spine_top_check'):
                        self.spine_top_check.setChecked(main_dialog.spine_top_check.isChecked())
                        self.spine_bottom_check.setChecked(main_dialog.spine_bottom_check.isChecked())
                        self.spine_left_check.setChecked(main_dialog.spine_left_check.isChecked())
                        self.spine_right_check.setChecked(main_dialog.spine_right_check.isChecked())
                        self.spine_width_spin.setValue(main_dialog.spine_width_spin.value())
                    
                    # 同步图例设置
                    if hasattr(main_dialog, 'show_legend_check'):
                        self.show_legend_check.setChecked(main_dialog.show_legend_check.isChecked())
                        self.legend_frame_check.setChecked(main_dialog.legend_frame_check.isChecked())
                        index = self.legend_loc_combo.findText(main_dialog.legend_loc_combo.currentText())
                        if index >= 0:
                            self.legend_loc_combo.setCurrentIndex(index)
                    
                    QMessageBox.information(self, "完成", "已成功同步主窗口的样式参数！")
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"同步参数时出错: {e}")
        else:
            QMessageBox.warning(self, "警告", "无法访问主窗口。")
    
    def get_style_params(self):
        """获取当前样式参数"""
        return {
            'fig_width': self.fig_width_spin.value(),
            'fig_height': self.fig_height_spin.value(),
            'fig_dpi': self.fig_dpi_spin.value(),
            'font_family': self.font_family_combo.currentText(),
            'axis_title_fontsize': self.axis_title_font_spin.value(),
            'tick_label_fontsize': self.tick_label_font_spin.value(),
            'legend_fontsize': self.legend_font_spin.value(),
            'title_fontsize': self.title_font_spin.value(),
            'validation_title_fontsize': self.validation_title_font_spin.value(),
            'validation_title_pad': self.validation_title_pad_spin.value(),
            'validation_title_show': self.validation_title_show_check.isChecked(),
            'validation_xlabel_fontsize': self.validation_xlabel_font_spin.value(),
            'validation_xlabel_pad': self.validation_xlabel_pad_spin.value(),
            'validation_xlabel_show': self.validation_xlabel_show_check.isChecked(),
            'validation_ylabel_fontsize': self.validation_ylabel_font_spin.value(),
            'validation_ylabel_pad': self.validation_ylabel_pad_spin.value(),
            'validation_ylabel_show': self.validation_ylabel_show_check.isChecked(),
            'line_width': self.line_width_spin.value(),
            'line_style': self.line_style_combo.currentText(),
            'marker_size': self.marker_size_spin.value(),
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
            'color_raw': self.color_raw_input.text().strip() or 'gray',
            'color_fit': self.color_fit_input.text().strip() or 'blue',
            'color_residual': self.color_residual_input.text().strip() or 'black',
            'title_text': self.title_text_input.text().strip(),
            'xlabel_text': self.xlabel_text_input.text().strip() or 'Wavenumber (cm⁻¹)',
            'ylabel_main_text': self.ylabel_main_input.text().strip() or 'Intensity',
            'ylabel_residual_text': self.ylabel_residual_input.text().strip() or 'Residuals',
            'legend_raw_label': self.legend_raw_label_input.text().strip() or 'Raw Low-Conc. Spectrum',
            'legend_fit_label': self.legend_fit_label_input.text().strip() or 'Fitted Organic Contribution',
            'legend_residual_label': self.legend_residual_label_input.text().strip() or 'Residuals',
            'show_label_a': self.show_label_a_check.isChecked(),
            'show_label_b': self.show_label_b_check.isChecked(),
            'label_a_text': self.label_a_text_input.text().strip() or '(A)',
            'label_b_text': self.label_b_text_input.text().strip() or '(B)',
            'aspect_ratio': self.aspect_ratio_spin.value(),  # 横纵比支持（默认0.0表示自动）
        }
    
    def save_settings(self):
        """保存设置到QSettings"""
        params = self.get_style_params()
        self.style_config.save_style_params("NMFFitValidationWindow", params)
    
    def update_plot(self):
        """更新绘图（参考主程序逻辑，保持窗口位置不变）"""
        if self.x_data is None or self.y_raw is None or self.y_fit is None:
            return
        
        # 保存当前窗口位置和大小（如果窗口可见）
        if self.isVisible():
            current_rect = self.geometry()
            self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        
        # 保存设置
        self.save_settings()
        
        # 获取样式参数
        style_params = self.get_style_params()
        
        # 获取预处理和偏移参数
        global_stack_offset = self.global_stack_offset_spin.value()
        global_scale_factor = self.global_scale_factor_spin.value()
        raw_scale = self.raw_scale_spin.value()
        raw_offset = self.raw_offset_spin.value()
        fit_scale = self.fit_scale_spin.value()
        fit_offset = self.fit_offset_spin.value()
        is_derivative = self.is_derivative_check.isChecked()
        
        # 应用预处理和偏移
        y_raw_processed = self.y_raw.copy()
        y_fit_processed = self.y_fit.copy()
        
        # 获取X轴数据（可能需要调整长度）
        x_data_plot = self.x_data.copy()
        
        # 应用全局缩放
        y_raw_processed = y_raw_processed * global_scale_factor
        y_fit_processed = y_fit_processed * global_scale_factor
        
        # 应用独立Y轴调整
        y_raw_processed = y_raw_processed * raw_scale + raw_offset
        y_fit_processed = y_fit_processed * fit_scale + fit_offset
        
        # 应用二阶导数（如果启用）
        if is_derivative:
            y_raw_processed = np.gradient(np.gradient(y_raw_processed, x_data_plot), x_data_plot)
            y_fit_processed = np.gradient(np.gradient(y_fit_processed, x_data_plot), x_data_plot)
            # 二阶导数后，数据长度保持不变（gradient不会改变长度）
            # 但为了确保一致性，我们检查长度是否匹配
            if len(y_raw_processed) != len(x_data_plot):
                # 如果长度不匹配，截断或插值
                min_len = min(len(y_raw_processed), len(x_data_plot))
                y_raw_processed = y_raw_processed[:min_len]
                y_fit_processed = y_fit_processed[:min_len]
                x_data_plot = x_data_plot[:min_len]
        
        # 确保所有数据长度一致
        min_len = min(len(x_data_plot), len(y_raw_processed), len(y_fit_processed))
        x_data_plot = x_data_plot[:min_len]
        y_raw_processed = y_raw_processed[:min_len]
        y_fit_processed = y_fit_processed[:min_len]
        
        # 应用堆叠偏移（原始数据在0，拟合数据在offset位置）
        y_raw_final = y_raw_processed + 0 * global_stack_offset
        y_fit_final = y_fit_processed + 1 * global_stack_offset
        
        fig = self.canvas.figure
        
        # 第一次绘图：创建GridSpec和axes
        if self.is_first_plot or self.ax_main is None or self.ax_residual is None:
            # 清除figure（只在第一次）
            fig.clear()
            
            # 设置图形尺寸（只在第一次设置，之后让Qt自动适应）
            fig.set_size_inches(style_params['fig_width'], style_params['fig_height'])
            fig.set_dpi(style_params['fig_dpi'])
            
            # 使用 GridSpec 创建两个子图，使用sharex确保X轴对齐
            from matplotlib.gridspec import GridSpec
            self.gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3, figure=fig)
            
            # 顶部面板 - 主图
            self.ax_main = fig.add_subplot(self.gs[0])
            
            # 底部面板 - 残差图（与主图共享X轴，确保对齐）
            self.ax_residual = fig.add_subplot(self.gs[1], sharex=self.ax_main)
            
            self.is_first_plot = False
        else:
            # 后续绘图：只清除axes内容（像主程序一样），保持axes对象和布局
            self.ax_main.cla()
            self.ax_residual.cla()
            # 清除之前的插入图
            self.inset_axes_list = []
        
        # 使用保存的axes引用
        ax_main = self.ax_main
        ax_residual = self.ax_residual
        
        # 应用样式
        self.style_config.apply_style_to_axes(ax_main, style_params)
        self.style_config.apply_style_to_axes(ax_residual, style_params)
        
        # 获取图例重命名映射（从主窗口）
        rename_map = {}
        parent = self.parent()
        if parent and hasattr(parent, 'legend_rename_widgets'):
            try:
                # 尝试使用主窗口的安全方法
                if hasattr(parent, '_safe_get_legend_rename_map'):
                    rename_map = parent._safe_get_legend_rename_map()
                else:
                    # 否则使用安全的手动方法
                    for key, widget in list(parent.legend_rename_widgets.items()):
                        try:
                            if hasattr(widget, 'text'):
                                renamed = widget.text().strip()
                                if renamed:
                                    rename_map[key] = renamed
                        except (RuntimeError, AttributeError):
                            continue
            except (RuntimeError, AttributeError):
                pass
        
        # 使用重命名后的图例名称（如果有）
        raw_label = rename_map.get('原始光谱', style_params['legend_raw_label'])
        fit_label = rename_map.get('拟合结果', style_params['legend_fit_label'])
        residual_label = rename_map.get('残差', style_params.get('legend_residual_label', '残差'))
        
        # 绘制对照组数据（如果存在）
        if hasattr(self, 'control_data_list') and self.control_data_list:
            for ctrl_data in self.control_data_list:
                ctrl_x = ctrl_data['x']
                ctrl_y = ctrl_data['y']
                ctrl_label_base = ctrl_data.get('label', 'Control')
                # 使用图例重命名映射（如果存在）
                ctrl_label = rename_map.get(ctrl_label_base, ctrl_label_base)
                # 确保长度一致
                min_len_ctrl = min(len(ctrl_x), len(ctrl_y), len(x_data_plot))
                if min_len_ctrl > 0:
                    # 应用与原始数据相同的预处理（如果需要）
                    ctrl_y_plot = ctrl_y[:min_len_ctrl].copy()
                    # 应用全局缩放
                    ctrl_y_plot = ctrl_y_plot * global_scale_factor
                    # 应用二阶导数（如果启用）
                    if is_derivative:
                        ctrl_y_plot = np.gradient(np.gradient(ctrl_y_plot, ctrl_x[:min_len_ctrl]), ctrl_x[:min_len_ctrl])
                    ax_main.plot(ctrl_x[:min_len_ctrl], ctrl_y_plot, 
                               '--', color='gray', alpha=0.7, linewidth=style_params['line_width'] * 0.8,
                               label=ctrl_label)
        
        # 绘制原始光谱（使用处理后的数据，确保X轴长度一致）
        ax_main.plot(x_data_plot, y_raw_final, 'o', 
                    color=style_params['color_raw'], 
                    markersize=style_params['marker_size'], 
                    alpha=0.6, 
                    label=raw_label, 
                    linestyle=':', 
                    linewidth=style_params['line_width'] * 0.5)
        
        # 绘制拟合贡献（使用处理后的数据，确保X轴长度一致）
        ax_main.plot(x_data_plot, y_fit_final, style_params['line_style'], 
                    color=style_params['color_fit'], 
                    linewidth=style_params['line_width'], 
                    label=fit_label)
        
        # 绘制垂直参考线（如果存在）
        if hasattr(self, 'vertical_lines') and self.vertical_lines:
            vertical_line_color = style_params.get('vertical_line_color', '#034DFB')
            vertical_line_style = style_params.get('vertical_line_style', '--')
            vertical_line_width = style_params.get('vertical_line_width', 0.8)
            vertical_line_alpha = style_params.get('vertical_line_alpha', 0.8)
            for line_x in self.vertical_lines:
                ax_main.axvline(x=line_x, color=vertical_line_color, linestyle=vertical_line_style,
                              linewidth=vertical_line_width, alpha=vertical_line_alpha)
                # 残差图也绘制垂直参考线
                ax_residual.axvline(x=line_x, color=vertical_line_color, linestyle=vertical_line_style,
                                  linewidth=vertical_line_width, alpha=vertical_line_alpha)
        
        # 峰值检测（如果启用）
        if hasattr(self, 'peak_detection_enabled') and self.peak_detection_enabled:
            try:
                from scipy.signal import find_peaks
                # 对原始数据进行峰值检测
                y_for_peaks = -y_raw_final  # 对于吸收谱，峰值是向下的
                peaks, properties = find_peaks(y_for_peaks, 
                                             height=-self.peak_height_threshold if self.peak_height_threshold > 0 else None,
                                             distance=self.peak_distance_min)
                if len(peaks) > 0:
                    ax_main.plot(x_data_plot[peaks], y_raw_final[peaks], 'x', 
                               color=style_params['color_raw'], markersize=8, 
                               markeredgewidth=2, label='Peaks')
            except Exception as e:
                print(f"峰值检测出错: {e}")
        
        # 设置标签和标题
        # 使用GUI中的X轴标题控制参数
        validation_xlabel_fontsize = style_params.get('validation_xlabel_fontsize', style_params.get('axis_title_fontsize', 20))
        validation_xlabel_pad = style_params.get('validation_xlabel_pad', 10.0)
        validation_xlabel_show = style_params.get('validation_xlabel_show', True)
        
        if validation_xlabel_show:
            ax_main.set_xlabel(style_params['xlabel_text'], fontsize=validation_xlabel_fontsize, labelpad=validation_xlabel_pad)
        
        # 使用GUI中的Y轴标题控制参数
        validation_ylabel_fontsize = style_params.get('validation_ylabel_fontsize', style_params.get('axis_title_fontsize', 20))
        validation_ylabel_pad = style_params.get('validation_ylabel_pad', 10.0)
        validation_ylabel_show = style_params.get('validation_ylabel_show', True)
        
        if validation_ylabel_show:
            ax_main.set_ylabel(style_params['ylabel_main_text'], fontsize=validation_ylabel_fontsize, labelpad=validation_ylabel_pad)
        
        # 标题：如果用户指定了标题，使用用户标题；否则使用默认标题
        # 使用GUI中的标题控制参数
        validation_title_fontsize = style_params.get('validation_title_fontsize', style_params.get('title_fontsize', 18))
        validation_title_pad = style_params.get('validation_title_pad', 10.0)
        validation_title_show = style_params.get('validation_title_show', True)
        
        if validation_title_show:
            if style_params['title_text']:
                title_text = style_params['title_text']
            else:
                title_text = f'NMF Fit Validation - {self.sample_name}'
            ax_main.set_title(title_text, fontsize=validation_title_fontsize, pad=validation_title_pad)
        
        # 图例
        if style_params['show_legend']:
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            if style_params['font_family'] != 'SimHei':
                legend_font.set_family(style_params['font_family'])
            legend_font.set_size(style_params['legend_fontsize'])
            legend = ax_main.legend(loc=style_params['legend_loc'], 
                                   frameon=style_params['legend_frame'],
                                   prop=legend_font)
        
        # 添加 (A) 标签
        if style_params['show_label_a']:
            ax_main.text(0.02, 0.98, style_params['label_a_text'], transform=ax_main.transAxes, 
                        fontsize=style_params['title_fontsize'], 
                        fontweight='bold', 
                        verticalalignment='top',
                        fontfamily=style_params['font_family'])
        
        # 创建多个插入放大图
        self.inset_axes_list = []
        try:
            from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
            
            # 如果没有指定放大区域，使用默认的一个
            if not self.zoom_regions:
                self.zoom_regions = [(1250, 1450)]
            
            # 为每个放大区域创建插入图
            positions = ['upper right', 'upper left', 'lower right', 'lower left']
            for idx, (zoom_xmin, zoom_xmax) in enumerate(self.zoom_regions):
                if idx >= len(positions):
                    break  # 最多支持4个放大区域
                
                # 找到对应的索引（使用调整后的x_data_plot）
                zoom_idx_min = np.argmin(np.abs(x_data_plot - zoom_xmin))
                zoom_idx_max = np.argmin(np.abs(x_data_plot - zoom_xmax))
                
                if zoom_idx_max > zoom_idx_min:
                    # 创建插入图（使用更合适的尺寸和位置）
                    # 根据索引计算位置，避免重叠
                    if idx == 0:  # upper right
                        bbox_x, bbox_y = 0.98, 0.98
                    elif idx == 1:  # upper left
                        bbox_x, bbox_y = 0.02, 0.98
                    elif idx == 2:  # lower right
                        bbox_x, bbox_y = 0.98, 0.02
                    else:  # lower left
                        bbox_x, bbox_y = 0.02, 0.02
                    
                    axins = zoomed_inset_axes(ax_main, zoom=2.5, loc=positions[idx], 
                                           bbox_to_anchor=(bbox_x, bbox_y),
                                           bbox_transform=ax_main.transAxes,
                                           axes_class=None)
                    
                    axins.plot(x_data_plot[zoom_idx_min:zoom_idx_max+1], 
                             y_raw_final[zoom_idx_min:zoom_idx_max+1], 
                             'o', color=style_params['color_raw'], 
                             markersize=style_params['marker_size'], 
                             alpha=0.6, linestyle=':', 
                             linewidth=style_params['line_width'] * 0.5)
                    axins.plot(x_data_plot[zoom_idx_min:zoom_idx_max+1], 
                             y_fit_final[zoom_idx_min:zoom_idx_max+1], 
                             style_params['line_style'], 
                             color=style_params['color_fit'], 
                             linewidth=style_params['line_width'])
                    
                    axins.set_xlim(zoom_xmin, zoom_xmax)
                    y_min = min(np.min(y_raw_final[zoom_idx_min:zoom_idx_max+1]), 
                               np.min(y_fit_final[zoom_idx_min:zoom_idx_max+1])) * 0.9
                    y_max = max(np.max(y_raw_final[zoom_idx_min:zoom_idx_max+1]), 
                               np.max(y_fit_final[zoom_idx_min:zoom_idx_max+1])) * 1.1
                    axins.set_ylim(y_min, y_max)
                    axins.set_xticklabels([])
                    axins.set_yticklabels([])
                    axins.tick_params(axis='both', which='major', labelsize=8)
                    
                    # 标记插入区域（连接主图和插入图的框）
                    # loc1和loc2指定连接线的位置：1=右下, 2=左下, 3=左上, 4=右上
                    # 根据插入图位置选择合适的连接点
                    if idx == 0:  # upper right
                        mark_inset(ax_main, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle='--', linewidth=1)
                    elif idx == 1:  # upper left
                        mark_inset(ax_main, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--', linewidth=1)
                    elif idx == 2:  # lower right
                        mark_inset(ax_main, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle='--', linewidth=1)
                    else:  # lower left
                        mark_inset(ax_main, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--', linewidth=1)
                    self.inset_axes_list.append(axins)
        except Exception as e:
            print(f"创建插入图时出错: {e}")
        
        # 获取图例重命名映射（从主窗口）
        rename_map = {}
        parent = self.parent()
        if parent and hasattr(parent, 'legend_rename_widgets'):
            try:
                # 尝试使用主窗口的安全方法
                if hasattr(parent, '_safe_get_legend_rename_map'):
                    rename_map = parent._safe_get_legend_rename_map()
                else:
                    # 否则使用安全的手动方法
                    for key, widget in list(parent.legend_rename_widgets.items()):
                        try:
                            if hasattr(widget, 'text'):
                                renamed = widget.text().strip()
                                if renamed:
                                    rename_map[key] = renamed
                        except (RuntimeError, AttributeError):
                            continue
            except (RuntimeError, AttributeError):
                pass
        
        # 使用重命名后的图例名称（如果有）
        raw_label = rename_map.get('原始光谱', style_params['legend_raw_label'])
        fit_label = rename_map.get('拟合结果', style_params['legend_fit_label'])
        residual_label = rename_map.get('残差', style_params.get('legend_residual_label', '残差'))
        
        # 绘制残差图（使用处理后的数据计算残差）
        if self.y_total_reconstructed is not None:
            # 对总重构也应用相同的预处理
            y_total_processed = self.y_total_reconstructed.copy()
            y_total_processed = y_total_processed * global_scale_factor
            y_total_processed = y_total_processed * fit_scale + fit_offset
            if is_derivative:
                y_total_processed = np.gradient(np.gradient(y_total_processed, x_data_plot), x_data_plot)
            # 确保长度一致
            min_len_total = min(len(x_data_plot), len(y_total_processed), len(y_raw_final))
            y_total_processed = y_total_processed[:min_len_total]
            y_raw_final_residual = y_raw_final[:min_len_total]
            residuals = y_raw_final_residual - y_total_processed
            x_residual = x_data_plot[:min_len_total]
        else:
            residuals = y_raw_final - y_fit_final
            x_residual = x_data_plot
        
        ax_residual.scatter(x_residual, residuals, 
                          c=style_params['color_residual'], 
                          s=style_params['marker_size'] * 2, 
                          alpha=0.6,
                          label=residual_label)
        ax_residual.axhline(y=0, color=style_params['color_residual'], 
                          linestyle='-', linewidth=style_params['line_width'])
        # 残差图也使用相同的X和Y轴标题控制
        if validation_xlabel_show:
            ax_residual.set_xlabel(style_params['xlabel_text'], fontsize=validation_xlabel_fontsize, labelpad=validation_xlabel_pad)
        if validation_ylabel_show:
            ax_residual.set_ylabel(style_params['ylabel_residual_text'], fontsize=validation_ylabel_fontsize, labelpad=validation_ylabel_pad)
        
        # 添加 (B) 标签
        if style_params['show_label_b']:
            ax_residual.text(0.02, 0.98, style_params['label_b_text'], transform=ax_residual.transAxes, 
                            fontsize=style_params['title_fontsize'], 
                            fontweight='bold', 
                            verticalalignment='top',
                            fontfamily=style_params['font_family'])
        
        # 由于使用了sharex，X轴已经自动对齐
        # sharex会自动：
        # 1. 同步两个子图的X轴范围（确保对齐）
        # 2. 隐藏主图的X轴刻度标签（避免重复，只显示残差图的X轴标签）
        
        # Aspect Ratio 设置（横纵比调节）
        # 注意：对于上下排列的子图，由于Y轴范围不同，不应该对两个子图都设置相同的aspect_ratio
        # 只对主图设置aspect_ratio，残差图使用auto，这样可以保持X轴对齐
        aspect_ratio = style_params.get('aspect_ratio', 0.0)
        if aspect_ratio > 0:
            # 只对主图设置aspect_ratio
            ax_main.set_box_aspect(aspect_ratio)
            # 残差图使用auto，保持X轴对齐
            ax_residual.set_aspect('auto')
        else:
            # 如果aspect_ratio为0或负数，两个图都使用auto
            ax_main.set_aspect('auto')
            ax_residual.set_aspect('auto')
        
        # 使用subplots_adjust调整布局（避免tight_layout与inset axes的兼容性问题）
        try:
            # 由于有inset axes，tight_layout会报错，改用subplots_adjust手动调整
            fig.subplots_adjust(
                left=0.12,      # 左侧边距（Y轴标签）
                right=0.95,     # 右侧边距
                top=0.92,       # 顶部边距（标题和插入图）
                bottom=0.12,    # 底部边距（X轴标签）
                hspace=0.3      # 主图和残差图之间的间距
            )
        except Exception as e:
            print(f"布局调整警告: {e}")
            # 如果失败，尝试使用tight_layout（可能会警告但不影响功能）
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
                    fig.tight_layout()
            except:
                pass
        
        # 更新canvas显示
        self.canvas.draw()
        
        # 恢复窗口位置（如果之前保存过）
        if self.last_geometry:
            self.setGeometry(*self.last_geometry)
    
    def set_data(self, x_data, y_raw, y_fit, y_total_reconstructed=None, sample_name="", 
                 vertical_lines=None, peak_detection_enabled=False, peak_height_threshold=0.0, 
                 peak_distance_min=10, control_data_list=None):
        """设置要绘制的数据"""
        self.x_data = x_data
        self.y_raw = y_raw
        self.y_fit = y_fit
        self.y_total_reconstructed = y_total_reconstructed
        self.sample_name = sample_name
        self.vertical_lines = vertical_lines if vertical_lines is not None else []
        self.peak_detection_enabled = peak_detection_enabled
        self.peak_height_threshold = peak_height_threshold
        self.peak_distance_min = peak_distance_min
        self.control_data_list = control_data_list if control_data_list is not None else []
        # 初始化默认放大区域（如果列表为空）
        if not self.zoom_regions and self.zoom_regions_list.count() == 0:
            self.zoom_regions = [(1250, 1450)]
            self.zoom_regions_list.addItem("1250 - 1450 cm⁻¹")
        # 重置first_plot标志，确保重新创建axes
        self.is_first_plot = True
        self.update_plot()
    
    def closeEvent(self, event):
        """窗口关闭时保存设置和位置"""
        if self.isVisible():
            current_rect = self.geometry()
            self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        self.save_settings()
        event.accept()
    
    def showEvent(self, event):
        """窗口显示时恢复位置"""
        super().showEvent(event)
        if self.last_geometry:
            self.setGeometry(*self.last_geometry)



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
from sklearn.pipeline import Pipeline

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


class QuantitativeResultWindow(QDialog):
    """独立的定量校准结果窗口（参考4.py，所有参数在图外面板）"""
    def __init__(self, parent_dialog=None):
        super().__init__(parent_dialog)
        self.parent_dialog = parent_dialog  # 保存主窗口引用
        self.setWindowTitle("定量校准结果")
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        
        self.main_layout = QVBoxLayout(self)
        
        # 不指定尺寸，让matplotlib自动适应窗口（与MplPlotWindow一致）
        # 这样可以确保每次绘图都保持一致的布局
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)
        
        # 存储窗口位置和大小
        self.last_geometry = None
        
        # 连接窗口大小改变事件，让布局自动适应（与其他绘图窗口一致）
        self.resizeEvent = self._update_geometry_on_resize
        self.moveEvent = self._update_geometry_on_move
    
    def _update_geometry_on_move(self, event):
        """窗口移动时保存位置"""
        current_rect = self.geometry()
        self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        super().moveEvent(event)
    
    def _update_geometry_on_resize(self, event):
        """窗口大小改变时自动调整布局（与MplPlotWindow保持一致）"""
        current_rect = self.geometry()
        self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        
        # 让matplotlib自动适应窗口大小（与MplPlotWindow一致）
        try:
            # 使用tight_layout自动调整布局以适应窗口大小
            # 使用warnings抑制警告（当有特殊Axes时，tight_layout会产生警告但不影响功能）
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
                self.canvas.figure.tight_layout()
            self.canvas.draw()
        except:
            # 如果tight_layout失败，继续执行（不影响功能）
            pass
        
        super().resizeEvent(event)
    
    def update_plot(self, plot_params):
        """更新绘图（使用与其他图一致的样式参数系统）"""
        # 确保所有必要的属性都已初始化（防御性编程）
        try:
            # 检查并初始化 main_layout
            if not hasattr(self, 'main_layout') or self.main_layout is None:
                self.main_layout = QVBoxLayout(self)
            
            # 检查并初始化 canvas
            if not hasattr(self, 'canvas') or self.canvas is None:
                # 不指定尺寸，让matplotlib自动适应窗口（与MplPlotWindow一致）
                self.canvas = MplCanvas(self)
                # 如果canvas是新创建的，需要创建toolbar并添加到布局
                if not hasattr(self, 'toolbar') or self.toolbar is None:
                    self.toolbar = NavigationToolbar(self.canvas, self)
                    # 清除现有布局内容（如果有）
                    while self.main_layout.count():
                        item = self.main_layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                    # 添加新的widgets
                    self.main_layout.addWidget(self.toolbar)
                    self.main_layout.addWidget(self.canvas)
            
            # 检查并初始化 last_geometry
            if not hasattr(self, 'last_geometry'):
                self.last_geometry = None
        except Exception as e:
            # 如果初始化失败，记录错误并返回
            print(f"初始化 QuantitativeResultWindow 失败: {e}")
            traceback.print_exc()
            return
        
        # 保存当前窗口位置
        if self.isVisible():
            current_rect = self.geometry()
            self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        
        # 获取配置的参数
        fig_width = plot_params.get('fig_width', 10.0)
        fig_height = plot_params.get('fig_height', 6.0)
        fig_dpi = plot_params.get('fig_dpi', 300)
        
        title_text = plot_params.get('title', "定量校准结果")
        xlabel_text = plot_params.get('xlabel', "样品名称")
        ylabel_text = plot_params.get('ylabel', "权重值")
        
        font_family = plot_params.get('font_family', 'Times New Roman')
        axis_title_fontsize = plot_params.get('axis_title_fontsize', 20)
        tick_label_fontsize = plot_params.get('tick_label_fontsize', 16)
        legend_fontsize = plot_params.get('legend_fontsize', 10)
        
        # 图例高级控制参数
        legend_ncol = plot_params.get('legend_ncol', 1)
        legend_columnspacing = plot_params.get('legend_columnspacing', 2.0)
        legend_labelspacing = plot_params.get('legend_labelspacing', 0.5)
        legend_handlelength = plot_params.get('legend_handlelength', 2.0)
        show_legend = plot_params.get('show_legend', True)
        legend_frame = plot_params.get('legend_frame', True)
        legend_loc = plot_params.get('legend_loc', 'best')
        
        # 刻度样式参数
        tick_direction = plot_params.get('tick_direction', 'in')
        tick_len_major = plot_params.get('tick_len_major', 8)
        tick_len_minor = plot_params.get('tick_len_minor', 4)
        tick_width = plot_params.get('tick_width', 1.0)
        
        # 边框样式参数
        border_sides = plot_params.get('border_sides', ['top', 'right', 'left', 'bottom'])
        border_linewidth = plot_params.get('border_linewidth', 2.0)
        
        # 网格参数
        show_grid = plot_params.get('show_grid', True)
        grid_alpha = plot_params.get('grid_alpha', 0.2)
        
        bar_width = plot_params.get('bar_width', 0.35)
        bar_alpha = plot_params.get('bar_alpha', 0.7)
        bar_edge_color = plot_params.get('bar_edge_color', 'black')
        bar_edge_width = plot_params.get('bar_edge_width', 1.0)
        bar_hatch = plot_params.get('bar_hatch', '')  # 填充纹理
        
        color_low = plot_params.get('color_low', 'gray')
        color_calibrated = plot_params.get('color_calibrated', 'red')
        color_bias = plot_params.get('color_bias', 'blue')
        
        # 辅助线样式
        bias_line_style = plot_params.get('bias_line_style', '--')
        bias_line_width = plot_params.get('bias_line_width', 2.0)
        
        xlabel_rotation = plot_params.get('xlabel_rotation', 45)
        
        w_low = plot_params.get('w_low')
        w_calibrated = plot_params.get('w_calibrated')
        w_bias = plot_params.get('w_bias')
        sample_names = plot_params.get('sample_names', [])
        
        # 图例重命名映射
        rename_map = plot_params.get('legend_names', {})
        
        if w_low is None or w_calibrated is None:
            return
        
        # 更新canvas的尺寸和DPI（再次检查，确保安全）
        if not hasattr(self, 'canvas') or self.canvas is None:
            print("错误: canvas 未初始化，无法绘图")
            return  # 如果canvas仍然不存在，无法绘图
        
        try:
            fig = self.canvas.figure
        except AttributeError:
            print("错误: canvas.figure 不存在")
            return
        
        # 确保axes存在
        try:
            if not hasattr(self.canvas, 'axes') or self.canvas.axes is None:
                ax = fig.add_subplot(111)
                self.canvas.axes = ax
            else:
                ax = self.canvas.axes
        except Exception as e:
            print(f"错误: 无法创建或获取 axes: {e}")
            traceback.print_exc()
            return
        
        # 清除旧图（与MplPlotWindow保持一致，只清除内容，不改变布局）
        try:
            ax.cla()
        except Exception as e:
            print(f"警告: 清除旧图失败: {e}")
            # 继续执行，尝试创建新的axes
            ax = fig.add_subplot(111)
            self.canvas.axes = ax
        
        # 不设置figure尺寸和DPI，让matplotlib自动适应窗口大小（与MplPlotWindow一致）
        # 这样可以确保每次绘图都保持一致的布局，不会占满整个窗口
        # 注意：fig_width和fig_height参数保留用于其他用途，但不强制设置figure尺寸
        
        n_samples = len(sample_names)
        x_pos = np.arange(n_samples)
        
        # 使用重命名后的图例名称（如果有）
        label_low = rename_map.get('原始权重 ($w_{low}$)', rename_map.get('原始权重', '原始权重 ($w_{low}$)'))
        label_calibrated = rename_map.get('校准后权重 ($w_{calibrated}$)', rename_map.get('校准后权重', '校准后权重 ($w_{calibrated}$)'))
        label_bias = rename_map.get('空白偏差', f'空白偏差 ($w_{{bias}}$ = {w_bias:.6f})') if w_bias is not None else '空白偏差'
        
        # 绘制柱状图（带边框和填充纹理）
        bars1 = ax.bar(x_pos - bar_width/2, w_low, bar_width, 
                      label=label_low, 
                      color=color_low, 
                      alpha=bar_alpha,
                      edgecolor=bar_edge_color,
                      linewidth=bar_edge_width,
                      hatch=bar_hatch if bar_hatch else None)
        bars2 = ax.bar(x_pos + bar_width/2, w_calibrated, bar_width,
                      label=label_calibrated, 
                      color=color_calibrated, 
                      alpha=bar_alpha,
                      edgecolor=bar_edge_color,
                      linewidth=bar_edge_width,
                      hatch=bar_hatch if bar_hatch else None)
        
        # 绘制空白偏差水平线（使用自定义样式）
        if w_bias is not None:
            ax.axhline(y=w_bias, color=color_bias, linestyle=bias_line_style, linewidth=bias_line_width,
                     label=label_bias, alpha=0.8)
        
        # 设置字体
        current_font = 'Times New Roman' if font_family == 'Times New Roman' else font_family
        
        # 检测文本是否包含中文
        def contains_chinese(text):
            if not isinstance(text, str):
                return False
            return any('\u4e00' <= char <= '\u9fff' for char in text)
        
        has_chinese = (contains_chinese(title_text) or contains_chinese(xlabel_text) or 
                      contains_chinese(ylabel_text) or any(contains_chinese(name) for name in sample_names))
        
        # 如果包含中文，使用sans-serif字体族（支持中文）
        if has_chinese or font_family == 'SimHei':
            actual_font_family = 'sans-serif'
        else:
            actual_font_family = current_font
        
        # 设置标签和标题 - 使用GUI中的控制参数
        xlabel_fontsize = plot_params.get('xlabel_fontsize', axis_title_fontsize)
        xlabel_pad = plot_params.get('xlabel_pad', 10.0)
        xlabel_show = plot_params.get('xlabel_show', True)
        
        if xlabel_show:
            ax.set_xlabel(xlabel_text, fontsize=xlabel_fontsize, labelpad=xlabel_pad, fontfamily=actual_font_family)
        
        ylabel_fontsize = plot_params.get('ylabel_fontsize', axis_title_fontsize)
        ylabel_pad = plot_params.get('ylabel_pad', 10.0)
        ylabel_show = plot_params.get('ylabel_show', True)
        
        if ylabel_show:
            ax.set_ylabel(ylabel_text, fontsize=ylabel_fontsize, labelpad=ylabel_pad, fontfamily=actual_font_family)
        
        title_fontsize = plot_params.get('title_fontsize', axis_title_fontsize + 2)
        title_pad = plot_params.get('title_pad', 10.0)
        title_show = plot_params.get('title_show', True)
        
        if title_show:
            ax.set_title(title_text, fontsize=title_fontsize, fontweight='bold', fontfamily=actual_font_family, pad=title_pad)
        
        # 设置刻度标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sample_names, rotation=xlabel_rotation, ha='right', 
                          fontsize=tick_label_fontsize, fontfamily=actual_font_family)
        
        # Ticks 样式（使用与其他图一致的参数）
        ax.tick_params(labelsize=tick_label_fontsize, direction=tick_direction, width=tick_width, labelfontfamily=actual_font_family)
        ax.tick_params(which='major', length=tick_len_major)
        ax.tick_params(which='minor', length=tick_len_minor)
        ax.tick_params(axis='y', labelsize=tick_label_fontsize, which='both')
        
        # 边框设置 (Spines) - 使用与其他图一致的参数
        for side in ['top', 'right', 'left', 'bottom']:
            if side in border_sides:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(border_linewidth)
            else:
                ax.spines[side].set_visible(False)
        
        # Aspect Ratio 设置（横纵比调节）
        aspect_ratio = plot_params.get('aspect_ratio', 0.0)
        if aspect_ratio > 0:
            ax.set_box_aspect(aspect_ratio)
        else:
            ax.set_aspect('auto')
        
        # 网格设置
        if show_grid:
            ax.grid(True, alpha=grid_alpha, axis='y')
        else:
            ax.grid(False)
        
        # 图例设置 - 使用完整的图例控制参数
        if show_legend:
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            # 使用与轴标签相同的字体设置（支持中文）
            legend_font.set_family(actual_font_family)
            legend_font.set_size(legend_fontsize)
            
            ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                     ncol=legend_ncol, columnspacing=legend_columnspacing, 
                     labelspacing=legend_labelspacing, handlelength=legend_handlelength)
        
        # 使用tight_layout自动调整布局（与MplPlotWindow保持一致，确保每次绘图布局一致）
        # 使用warnings抑制警告（当有特殊Axes时，tight_layout会产生警告但不影响功能）
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
                fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            # 如果tight_layout失败（例如有特殊元素），使用subplots_adjust作为备选
            try:
                # 根据X轴标签旋转角度调整底部边距
                if xlabel_rotation > 0:
                    bottom_padding = 0.15 + (xlabel_rotation / 90.0) * 0.15
                else:
                    bottom_padding = 0.12
                
                # 设置合理的边距，确保所有元素可见
                fig.subplots_adjust(
                    left=0.12,      # 左侧边距（Y轴标签）
                    right=0.95,     # 右侧边距
                    top=0.92,       # 顶部边距（标题）
                    bottom=bottom_padding  # 底部边距（X轴标签）
                )
                self.canvas.draw()
            except Exception as e2:
                print(f"警告: 绘制图形时出错: {e2}")
                traceback.print_exc()
        
        # 恢复窗口位置（在绘制完成后）
        try:
            if hasattr(self, 'last_geometry') and self.last_geometry:
                self.setGeometry(*self.last_geometry)
            elif not self.isVisible():
                self.show()
        except Exception as e:
            print(f"警告: 恢复窗口位置失败: {e}")
            if not self.isVisible():
                self.show()


class QuantitativeAnalysisDialog(QDialog):
    """定量校准分析对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("定量校准分析")
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(1400, 900)  # 增加宽度以便更好地显示参数
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        self.parent_dialog = parent
        
        self.main_layout = QVBoxLayout(self)
        
        # 前提检查提示
        check_label = QLabel("前提检查：请确保已运行标准NMF分析并指定了目标组分索引。")
        check_label.setStyleSheet("color: #FF6B00; font-weight: bold; padding: 5px;")
        self.main_layout.addWidget(check_label)
        
        # 创建TabWidget来组织不同功能模块
        self.tab_widget = QTabWidget()
        
        # Tab 1: 文件分组与回归模式
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        
        # 文件分组区域
        files_group = QGroupBox("文件分组")
        files_layout = QHBoxLayout(files_group)
        
        # 左侧：空白样品列表
        blanks_layout = QVBoxLayout()
        blanks_layout.addWidget(QLabel("空白样品 (Blanks):"))
        self.blanks_list = QListWidget()
        self.blanks_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        blanks_layout.addWidget(self.blanks_list)
        
        blanks_buttons = QHBoxLayout()
        self.btn_add_to_blanks = QPushButton("添加文件到空白")
        self.btn_add_to_blanks.clicked.connect(lambda: self._add_files_to_list(self.blanks_list))
        self.btn_remove_from_blanks = QPushButton("移除选中")
        self.btn_remove_from_blanks.clicked.connect(lambda: self._remove_selected_from_list(self.blanks_list))
        blanks_buttons.addWidget(self.btn_add_to_blanks)
        blanks_buttons.addWidget(self.btn_remove_from_blanks)
        blanks_layout.addLayout(blanks_buttons)
        
        # 右侧：待测样品列表
        samples_layout = QVBoxLayout()
        samples_layout.addWidget(QLabel("待测样品 (Samples):"))
        self.samples_list = QListWidget()
        self.samples_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        samples_layout.addWidget(self.samples_list)
        
        samples_buttons = QHBoxLayout()
        self.btn_add_to_samples = QPushButton("添加文件到待测")
        self.btn_add_to_samples.clicked.connect(lambda: self._add_files_to_list(self.samples_list))
        self.btn_remove_from_samples = QPushButton("移除选中")
        self.btn_remove_from_samples.clicked.connect(lambda: self._remove_selected_from_list(self.samples_list))
        samples_buttons.addWidget(self.btn_add_to_samples)
        samples_buttons.addWidget(self.btn_remove_from_samples)
        samples_layout.addLayout(samples_buttons)
        
        files_layout.addLayout(blanks_layout)
        files_layout.addLayout(samples_layout)
        
        tab1_layout.addWidget(files_group)
        
        # 回归模式选择区域
        regression_mode_group = QGroupBox("低浓度组分回归模式")
        regression_mode_layout = QVBoxLayout(regression_mode_group)
        
        mode_info_label = QLabel("选择待测样品的回归方式：")
        mode_info_label.setWordWrap(True)
        regression_mode_layout.addWidget(mode_info_label)
        
        self.regression_mode_button_group = QButtonGroup()
        self.mode_individual = QRadioButton("A. 单独回归（每条低浓度组分单独计算权重）")
        self.mode_average = QRadioButton("B. 平均回归（多条低浓度组分先平均，再计算权重）")
        self.mode_individual.setChecked(True)  # 默认选择单独回归
        
        self.regression_mode_button_group.addButton(self.mode_individual, 0)
        self.regression_mode_button_group.addButton(self.mode_average, 1)
        
        regression_mode_layout.addWidget(self.mode_individual)
        regression_mode_layout.addWidget(self.mode_average)
        
        mode_tip_label = QLabel("提示：\n"
                               "• 单独回归：适合需要分析每个样品个体差异的情况\n"
                               "• 平均回归：适合需要提高信噪比、减少随机误差的情况")
        mode_tip_label.setWordWrap(True)
        mode_tip_label.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        regression_mode_layout.addWidget(mode_tip_label)
        
        tab1_layout.addWidget(regression_mode_group)
        
        # 预处理提示
        prep_info_group = QGroupBox("预处理设置")
        prep_info_layout = QVBoxLayout(prep_info_group)
        prep_info_label = QLabel("✓ 空白样品和待测样品将自动使用主程序中设置的所有预处理参数：\n"
                                "• QC质量检查、Bose-Einstein校正、平滑、基线校正（AsLS）、归一化等\n"
                                "• 确保在主程序Tab 1中正确配置预处理参数后再运行校准计算")
        prep_info_label.setWordWrap(True)
        prep_info_label.setStyleSheet("color: #2196F3; font-size: 9pt; padding: 5px; background-color: #E3F2FD; border-radius: 3px;")
        prep_info_layout.addWidget(prep_info_label)
        
        # 归一化选项（对权重结果进行归一化）
        normalization_group = QGroupBox("权重归一化（可选）")
        normalization_layout = QHBoxLayout(normalization_group)
        self.result_normalization_check = QCheckBox("对权重结果进行归一化")
        self.result_normalization_check.setToolTip("如果启用，将对w_low和w_calibrated进行归一化处理")
        self.result_normalization_combo = QComboBox()
        self.result_normalization_combo.addItems(['None', 'max', 'area'])
        self.result_normalization_combo.setCurrentText('None')
        self.result_normalization_combo.setEnabled(False)
        self.result_normalization_check.toggled.connect(lambda checked: self.result_normalization_combo.setEnabled(checked))
        normalization_layout.addWidget(self.result_normalization_check)
        normalization_layout.addWidget(QLabel("归一化方式:"))
        normalization_layout.addWidget(self.result_normalization_combo)
        normalization_layout.addStretch()
        prep_info_layout.addWidget(normalization_group)
        
        tab1_layout.addWidget(prep_info_group)
        
        # 计算按钮
        calc_layout = QHBoxLayout()
        self.btn_run_calculation = QPushButton("运行校准计算")
        self.btn_run_calculation.setStyleSheet("font-size: 12pt; padding: 8px; background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_run_calculation.clicked.connect(self.run_calculation)
        calc_layout.addStretch(1)
        calc_layout.addWidget(self.btn_run_calculation)
        calc_layout.addStretch(1)
        tab1_layout.addLayout(calc_layout)
        
        tab1_layout.addStretch()
        self.tab_widget.addTab(tab1, "📊 文件分组与回归")
        
        # Tab 2: 可视化参数配置
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_scroll = QScrollArea()
        tab2_scroll.setWidgetResizable(True)
        tab2_widget = QWidget()
        tab2_widget.setLayout(tab2_layout)
        tab2_scroll.setWidget(tab2_widget)
        
        # 可视化参数配置区域（可折叠）- 使用两列布局以便更好地显示参数
        style_group = CollapsibleGroupBox("🎨 可视化参数配置（发表级设置）", is_expanded=True)
        
        # 创建容器widget和布局（先创建布局，再设置为widget的布局）
        style_container = QWidget()
        style_container_layout = QHBoxLayout(style_container)
        style_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # 左列
        style_layout_left = QFormLayout()
        style_layout_left.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        # 右列
        style_layout_right = QFormLayout()
        style_layout_right.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        # 创建左右两列的容器，设置最小宽度以便更好地显示参数
        left_column = QWidget()
        left_column.setLayout(style_layout_left)
        left_column.setMinimumWidth(600)  # 设置最小宽度
        
        right_column = QWidget()
        right_column.setLayout(style_layout_right)
        right_column.setMinimumWidth(600)  # 设置最小宽度
        
        style_container_layout.addWidget(left_column)
        style_container_layout.addWidget(right_column)
        style_container_layout.setSpacing(20)  # 设置两列之间的间距
        
        # 标题和标签
        self.result_title_input = QLineEdit("定量校准结果")
        self.result_xlabel_input = QLineEdit("样品名称")
        self.result_ylabel_input = QLineEdit("权重值")
        
        # 定量校准结果标题控制：大小、间距、显示/隐藏
        self.result_title_font_spin = QSpinBox()
        self.result_title_font_spin.setRange(-999999999, 999999999)
        self.result_title_font_spin.setValue(22)  # 默认值（axis_title_fontsize + 2）
        
        self.result_title_pad_spin = QDoubleSpinBox()
        self.result_title_pad_spin.setRange(-999999999.0, 999999999.0)
        self.result_title_pad_spin.setDecimals(15)
        self.result_title_pad_spin.setValue(10.0)  # 默认值
        
        self.result_title_show_check = QCheckBox("显示图表标题")
        self.result_title_show_check.setChecked(True)  # 默认显示
        
        # 左列：标题和标签设置
        style_layout_left.addRow("图表标题:", self.result_title_input)
        style_layout_left.addRow("标题控制:", self._create_h_layout([self.result_title_show_check, QLabel("大小:"), self.result_title_font_spin, QLabel("间距:"), self.result_title_pad_spin]))
        style_layout_left.addRow("X轴标签:", self.result_xlabel_input)
        
        # 定量校准结果X轴标题控制：大小、间距、显示/隐藏
        self.result_xlabel_font_spin = QSpinBox()
        self.result_xlabel_font_spin.setRange(-999999999, 999999999)
        self.result_xlabel_font_spin.setValue(20)  # 默认值
        
        self.result_xlabel_pad_spin = QDoubleSpinBox()
        self.result_xlabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.result_xlabel_pad_spin.setDecimals(15)
        self.result_xlabel_pad_spin.setValue(10.0)  # 默认值
        
        self.result_xlabel_show_check = QCheckBox("显示X轴标题")
        self.result_xlabel_show_check.setChecked(True)  # 默认显示
        
        style_layout_left.addRow("X轴标题控制:", self._create_h_layout([self.result_xlabel_show_check, QLabel("大小:"), self.result_xlabel_font_spin, QLabel("间距:"), self.result_xlabel_pad_spin]))
        
        style_layout_left.addRow("Y轴标签:", self.result_ylabel_input)
        
        # 定量校准结果Y轴标题控制：大小、间距、显示/隐藏
        self.result_ylabel_font_spin = QSpinBox()
        self.result_ylabel_font_spin.setRange(-999999999, 999999999)
        self.result_ylabel_font_spin.setValue(20)  # 默认值
        
        self.result_ylabel_pad_spin = QDoubleSpinBox()
        self.result_ylabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.result_ylabel_pad_spin.setDecimals(15)
        self.result_ylabel_pad_spin.setValue(10.0)  # 默认值
        
        self.result_ylabel_show_check = QCheckBox("显示Y轴标题")
        self.result_ylabel_show_check.setChecked(True)  # 默认显示
        
        style_layout_left.addRow("Y轴标题控制:", self._create_h_layout([self.result_ylabel_show_check, QLabel("大小:"), self.result_ylabel_font_spin, QLabel("间距:"), self.result_ylabel_pad_spin]))
        
        # 图尺寸和DPI
        self.result_fig_width_spin = QDoubleSpinBox()
        self.result_fig_width_spin.setRange(-999999999.0, 999999999.0)
        self.result_fig_width_spin.setDecimals(15)
        self.result_fig_width_spin.setValue(10.0)
        
        self.result_fig_height_spin = QDoubleSpinBox()
        self.result_fig_height_spin.setRange(-999999999.0, 999999999.0)
        self.result_fig_height_spin.setDecimals(15)
        self.result_fig_height_spin.setValue(6.0)
        
        self.result_fig_dpi_spin = QSpinBox()
        self.result_fig_dpi_spin.setRange(-999999999, 999999999)
        self.result_fig_dpi_spin.setValue(300)
        
        style_layout_left.addRow("图尺寸 (宽/高):", self._create_h_layout([self.result_fig_width_spin, self.result_fig_height_spin]))
        style_layout_left.addRow("DPI:", self.result_fig_dpi_spin)
        
        # 字体设置
        self.result_font_family_combo = QComboBox()
        self.result_font_family_combo.addItems(['Times New Roman', 'Arial', 'SimHei'])
        self.result_font_family_combo.setCurrentText('Times New Roman')
        
        self.result_axis_title_font_spin = QSpinBox()
        self.result_axis_title_font_spin.setRange(-999999999, 999999999)
        self.result_axis_title_font_spin.setValue(20)
        
        self.result_tick_label_font_spin = QSpinBox()
        self.result_tick_label_font_spin.setRange(-999999999, 999999999)
        self.result_tick_label_font_spin.setValue(16)
        
        self.result_legend_font_spin = QSpinBox()
        self.result_legend_font_spin.setRange(-999999999, 999999999)
        self.result_legend_font_spin.setValue(10)
        
        style_layout_left.addRow("字体家族:", self.result_font_family_combo)
        style_layout_left.addRow("字体大小 (轴/刻度/图例):", self._create_h_layout([self.result_axis_title_font_spin, self.result_tick_label_font_spin, self.result_legend_font_spin]))
        
        # 右列：柱状图样式和其他设置
        # 柱状图样式
        self.result_bar_width_spin = QDoubleSpinBox()
        self.result_bar_width_spin.setRange(-999999999.0, 999999999.0)
        self.result_bar_width_spin.setDecimals(15)
        self.result_bar_width_spin.setValue(0.35)
        
        self.result_bar_alpha_spin = QDoubleSpinBox()
        self.result_bar_alpha_spin.setRange(-999999999.0, 999999999.0)
        self.result_bar_alpha_spin.setDecimals(15)
        self.result_bar_alpha_spin.setValue(0.7)
        
        # 柱边框设置
        self.result_bar_edge_color_input = QLineEdit("black")
        self.result_bar_edge_width_spin = QDoubleSpinBox()
        self.result_bar_edge_width_spin.setRange(-999999999.0, 999999999.0)
        self.result_bar_edge_width_spin.setDecimals(15)
        self.result_bar_edge_width_spin.setValue(1.0)
        
        # 填充纹理（Hatching）
        self.result_bar_hatch_combo = QComboBox()
        self.result_bar_hatch_combo.addItems(['无', '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'])
        self.result_bar_hatch_combo.setCurrentText('无')
        
        style_layout_right.addRow("柱宽 / 透明度:", self._create_h_layout([self.result_bar_width_spin, self.result_bar_alpha_spin]))
        style_layout_right.addRow("柱边框颜色 / 线宽:", self._create_h_layout([self.result_bar_edge_color_input, self._create_color_picker_button(self.result_bar_edge_color_input), self.result_bar_edge_width_spin]))
        style_layout_right.addRow("填充纹理 (Hatching):", self.result_bar_hatch_combo)
        
        # 颜色设置
        self.result_color_low_input = QLineEdit("gray")
        self.result_color_calibrated_input = QLineEdit("red")
        self.result_color_bias_input = QLineEdit("blue")
        
        style_layout_right.addRow("原始权重颜色:", self._create_h_layout([self.result_color_low_input, self._create_color_picker_button(self.result_color_low_input)]))
        style_layout_right.addRow("校准权重颜色:", self._create_h_layout([self.result_color_calibrated_input, self._create_color_picker_button(self.result_color_calibrated_input)]))
        style_layout_right.addRow("空白偏差颜色:", self._create_h_layout([self.result_color_bias_input, self._create_color_picker_button(self.result_color_bias_input)]))
        
        # 辅助线样式（空白偏差水平线）
        self.result_bias_line_style_combo = QComboBox()
        self.result_bias_line_style_combo.addItems(['-', '--', '-.', ':'])
        self.result_bias_line_style_combo.setCurrentText('--')
        
        self.result_bias_line_width_spin = QDoubleSpinBox()
        self.result_bias_line_width_spin.setRange(-999999999.0, 999999999.0)
        self.result_bias_line_width_spin.setDecimals(15)
        self.result_bias_line_width_spin.setValue(2.0)
        
        style_layout_right.addRow("辅助线样式 / 线宽:", self._create_h_layout([self.result_bias_line_style_combo, self.result_bias_line_width_spin]))
        
        # 网格和边框
        self.result_show_grid_check = QCheckBox("显示网格")
        self.result_show_grid_check.setChecked(True)
        
        self.result_grid_alpha_spin = QDoubleSpinBox()
        self.result_grid_alpha_spin.setRange(-999999999.0, 999999999.0)
        self.result_grid_alpha_spin.setDecimals(15)
        self.result_grid_alpha_spin.setValue(0.3)
        
        style_layout_right.addRow("网格设置:", self._create_h_layout([self.result_show_grid_check, QLabel("透明度:"), self.result_grid_alpha_spin]))
        
        # 图例控制选项
        self.result_show_legend_check = QCheckBox("显示图例", checked=True)
        self.result_legend_frame_check = QCheckBox("图例边框", checked=True)
        self.result_legend_loc_combo = QComboBox()
        self.result_legend_loc_combo.addItems(['best', 'upper right', 'upper left', 'lower left', 'lower right', 'center left', 'center right', 'lower center', 'upper center', 'center'])
        self.result_legend_loc_combo.setCurrentText('best')
        
        style_layout_right.addRow(self._create_h_layout([self.result_show_legend_check, self.result_legend_frame_check]))
        style_layout_right.addRow("图例位置:", self.result_legend_loc_combo)
        
        # X轴标签旋转角度
        self.result_xlabel_rotation_spin = QSpinBox()
        self.result_xlabel_rotation_spin.setRange(-999999999, 999999999)
        self.result_xlabel_rotation_spin.setValue(45)
        
        style_layout_right.addRow("X轴标签旋转角度:", self.result_xlabel_rotation_spin)
        
        # 刻度样式控制
        self.result_tick_direction_combo = QComboBox()
        self.result_tick_direction_combo.addItems(['in', 'out'])
        self.result_tick_direction_combo.setCurrentText('in')
        
        self.result_tick_len_major_spin = QSpinBox()
        self.result_tick_len_major_spin.setRange(-999999999, 999999999)
        self.result_tick_len_major_spin.setValue(8)
        
        self.result_tick_len_minor_spin = QSpinBox()
        self.result_tick_len_minor_spin.setRange(-999999999, 999999999)
        self.result_tick_len_minor_spin.setValue(4)
        
        self.result_tick_width_spin = QDoubleSpinBox()
        self.result_tick_width_spin.setRange(-999999999.0, 999999999.0)
        self.result_tick_width_spin.setDecimals(15)
        self.result_tick_width_spin.setValue(1.0)
        
        style_layout_right.addRow("刻度方向 / 宽度:", self._create_h_layout([self.result_tick_direction_combo, self.result_tick_width_spin]))
        style_layout_right.addRow("刻度长度 (大/小):", self._create_h_layout([self.result_tick_len_major_spin, self.result_tick_len_minor_spin]))
        
        # 边框控制
        self.result_spine_top_check = QCheckBox("上边框", checked=True)
        self.result_spine_bottom_check = QCheckBox("下边框", checked=True)
        self.result_spine_left_check = QCheckBox("左边框", checked=True)
        self.result_spine_right_check = QCheckBox("右边框", checked=True)
        
        self.result_spine_width_spin = QDoubleSpinBox()
        self.result_spine_width_spin.setRange(-999999999.0, 999999999.0)
        self.result_spine_width_spin.setDecimals(15)
        self.result_spine_width_spin.setValue(2.0)
        
        style_layout_right.addRow("边框显示:", self._create_h_layout([self.result_spine_top_check, self.result_spine_bottom_check, self.result_spine_left_check, self.result_spine_right_check]))
        style_layout_right.addRow("边框宽度:", self.result_spine_width_spin)
        
        # 图例高级控制
        self.result_legend_ncol_spin = QSpinBox()
        self.result_legend_ncol_spin.setRange(-999999999, 999999999)
        self.result_legend_ncol_spin.setValue(1)
        
        self.result_legend_columnspacing_spin = QDoubleSpinBox()
        self.result_legend_columnspacing_spin.setRange(-999999999.0, 999999999.0)
        self.result_legend_columnspacing_spin.setDecimals(15)
        self.result_legend_columnspacing_spin.setValue(2.0)
        
        self.result_legend_labelspacing_spin = QDoubleSpinBox()
        self.result_legend_labelspacing_spin.setRange(-999999999.0, 999999999.0)
        self.result_legend_labelspacing_spin.setDecimals(15)
        self.result_legend_labelspacing_spin.setValue(0.5)
        
        self.result_legend_handlelength_spin = QDoubleSpinBox()
        self.result_legend_handlelength_spin.setRange(-999999999.0, 999999999.0)
        self.result_legend_handlelength_spin.setDecimals(15)
        self.result_legend_handlelength_spin.setValue(2.0)
        
        style_layout_right.addRow("图例列数:", self.result_legend_ncol_spin)
        style_layout_right.addRow("图例列间距:", self.result_legend_columnspacing_spin)
        style_layout_right.addRow("图例标签间距:", self.result_legend_labelspacing_spin)
        style_layout_right.addRow("图例手柄长度:", self.result_legend_handlelength_spin)
        
        # 纵横比控制
        self.result_aspect_ratio_spin = QDoubleSpinBox()
        self.result_aspect_ratio_spin.setRange(-999999999.0, 999999999.0)
        self.result_aspect_ratio_spin.setDecimals(15)
        self.result_aspect_ratio_spin.setValue(0.0)  # 默认0.0表示自动
        
        style_layout_right.addRow("纵横比 (0=自动):", self.result_aspect_ratio_spin)
        
        # 添加同步主程序默认设置的按钮（放在右列底部）
        sync_button_layout = QHBoxLayout()
        self.btn_sync_defaults = QPushButton("🔄 同步主程序默认设置")
        self.btn_sync_defaults.setToolTip("将主程序中Tab 3的默认参数同步到此对话框")
        self.btn_sync_defaults.clicked.connect(self._sync_default_params)
        sync_button_layout.addWidget(self.btn_sync_defaults)
        sync_button_layout.addStretch(1)
        style_layout_right.addRow("", sync_button_layout)
        
        # 将容器widget添加到CollapsibleGroupBox的内容布局中
        # 注意：不能使用setContentLayout，因为style_container_layout已经有父级了
        # 应该直接将widget添加到content_layout
        style_group.content_layout.addWidget(style_container)
        tab2_layout.addWidget(style_group)
        tab2_layout.addStretch()
        
        self.tab_widget.addTab(tab2_scroll, "🎨 可视化参数")
        
        # Tab 3: 分类验证
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        
        # 分类验证区域
        classification_group = QGroupBox("分类验证 (Classification Validation)")
        classification_layout = QVBoxLayout(classification_group)
        
        classification_info_label = QLabel("✓ 对低浓度拉曼光谱进行二分类（'Organic Present' vs. 'Mineral Only'）\n"
                                          "✓ 使用SVC和PLS-DA算法，输入特征为完整预处理光谱（BE校正、AsLS基线校正、面积归一化、使用主菜单的截断范围）\n"
                                          "✓ 训练集：'Mineral Only' (Label 0) 和 'Organic High Concentration' (Label 1)\n"
                                          "✓ 测试集：选中的低浓度样本")
        classification_info_label.setWordWrap(True)
        classification_info_label.setStyleSheet("color: #2196F3; font-size: 9pt; padding: 5px; background-color: #E3F2FD; border-radius: 3px;")
        classification_layout.addWidget(classification_info_label)
        
        # 训练集选择
        training_set_layout = QHBoxLayout()
        training_set_layout.addWidget(QLabel("训练集 - Mineral Only (Label 0):"))
        self.training_mineral_list = QListWidget()
        self.training_mineral_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.training_mineral_list.setMaximumHeight(100)
        training_mineral_buttons = QVBoxLayout()
        self.btn_add_training_mineral = QPushButton("添加文件")
        self.btn_add_training_mineral.clicked.connect(lambda: self._add_files_to_list(self.training_mineral_list))
        self.btn_remove_training_mineral = QPushButton("移除选中")
        self.btn_remove_training_mineral.clicked.connect(lambda: self._remove_selected_from_list(self.training_mineral_list))
        training_mineral_buttons.addWidget(self.btn_add_training_mineral)
        training_mineral_buttons.addWidget(self.btn_remove_training_mineral)
        training_set_layout.addWidget(self.training_mineral_list)
        training_set_layout.addLayout(training_mineral_buttons)
        
        training_set_layout2 = QHBoxLayout()
        training_set_layout2.addWidget(QLabel("训练集 - Organic High Concentration (Label 1):"))
        self.training_organic_list = QListWidget()
        self.training_organic_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.training_organic_list.setMaximumHeight(100)
        training_organic_buttons = QVBoxLayout()
        self.btn_add_training_organic = QPushButton("添加文件")
        self.btn_add_training_organic.clicked.connect(lambda: self._add_files_to_list(self.training_organic_list))
        self.btn_remove_training_organic = QPushButton("移除选中")
        self.btn_remove_training_organic.clicked.connect(lambda: self._remove_selected_from_list(self.training_organic_list))
        training_organic_buttons.addWidget(self.btn_add_training_organic)
        training_organic_buttons.addWidget(self.btn_remove_training_organic)
        training_set_layout2.addWidget(self.training_organic_list)
        training_set_layout2.addLayout(training_organic_buttons)
        
        classification_layout.addLayout(training_set_layout)
        classification_layout.addLayout(training_set_layout2)
        
        # 算法选择
        algorithm_layout = QHBoxLayout()
        algorithm_layout.addWidget(QLabel("选择算法:"))
        self.classification_algorithm_combo = QComboBox()
        self.classification_algorithm_combo.addItems([
            'All',
            'SVC', 
            'PLS-DA', 
            'Logistic Regression (LR)',
            'k-Nearest Neighbors (k-NN)',
            'Random Forest (RF)',
            'PCA + LDA',
            'AdaBoost'
        ])
        self.classification_algorithm_combo.setCurrentText('All')
        algorithm_layout.addWidget(self.classification_algorithm_combo)
        algorithm_layout.addStretch(1)
        classification_layout.addLayout(algorithm_layout)
        
        # 预处理参数配置面板（新增，允许在分类验证时独立设置预处理参数）
        preprocess_params_group = CollapsibleGroupBox("🔬 预处理参数配置（分类验证专用）", is_expanded=False)
        preprocess_params_layout = QFormLayout()
        
        # 启用独立预处理参数（默认使用主菜单参数）
        self.classification_preprocess_enabled = False
        self.classification_preprocess_check = QCheckBox("使用独立预处理参数（不勾选则使用主菜单参数）")
        self.classification_preprocess_check.setChecked(False)
        self.classification_preprocess_check.toggled.connect(lambda checked: setattr(self, 'classification_preprocess_enabled', checked))
        preprocess_params_layout.addRow(self.classification_preprocess_check)
        
        # QC检查
        self.classification_qc_check = QCheckBox("启用 QC 质量检查")
        self.classification_qc_check.setChecked(False)
        self.classification_qc_threshold_spin = QDoubleSpinBox()
        self.classification_qc_threshold_spin.setRange(-999999999.0, 999999999.0)
        self.classification_qc_threshold_spin.setDecimals(15)
        self.classification_qc_threshold_spin.setValue(5.0)
        preprocess_params_layout.addRow(self.classification_qc_check)
        preprocess_params_layout.addRow("QC 阈值:", self.classification_qc_threshold_spin)
        
        # BE校正
        self.classification_be_check = QCheckBox("启用 Bose-Einstein 校正")
        self.classification_be_check.setChecked(False)
        self.classification_be_temp_spin = QDoubleSpinBox()
        self.classification_be_temp_spin.setRange(-999999999.0, 999999999.0)
        self.classification_be_temp_spin.setDecimals(15)
        self.classification_be_temp_spin.setValue(300.0)
        preprocess_params_layout.addRow(self.classification_be_check)
        preprocess_params_layout.addRow("BE 温度 (K):", self.classification_be_temp_spin)
        
        # 平滑
        self.classification_smoothing_check = QCheckBox("启用平滑")
        self.classification_smoothing_check.setChecked(False)
        self.classification_smoothing_window_spin = QSpinBox()
        self.classification_smoothing_window_spin.setRange(-999999999, 999999999)
        self.classification_smoothing_window_spin.setValue(15)
        self.classification_smoothing_poly_spin = QSpinBox()
        self.classification_smoothing_poly_spin.setRange(-999999999, 999999999)
        self.classification_smoothing_poly_spin.setValue(3)
        preprocess_params_layout.addRow(self.classification_smoothing_check)
        preprocess_params_layout.addRow("平滑窗口:", self.classification_smoothing_window_spin)
        preprocess_params_layout.addRow("平滑多项式阶数:", self.classification_smoothing_poly_spin)
        
        # AsLS基线校正
        self.classification_baseline_als_check = QCheckBox("启用 AsLS 基线校正（推荐）")
        self.classification_baseline_als_check.setChecked(True)  # 分类验证默认启用
        self.classification_lam_spin = QDoubleSpinBox()
        self.classification_lam_spin.setRange(-999999999.0, 999999999.0)
        self.classification_lam_spin.setDecimals(15)
        self.classification_lam_spin.setValue(10000)
        self.classification_p_spin = QDoubleSpinBox()
        self.classification_p_spin.setRange(-999999999.0, 999999999.0)
        self.classification_p_spin.setDecimals(15)
        self.classification_p_spin.setValue(0.005)
        preprocess_params_layout.addRow(self.classification_baseline_als_check)
        preprocess_params_layout.addRow("AsLS Lambda:", self.classification_lam_spin)
        preprocess_params_layout.addRow("AsLS P:", self.classification_p_spin)
        
        # 归一化（分类验证固定使用面积归一化，但显示提示）
        normalization_info_label = QLabel("注意：分类验证固定使用面积归一化（Area Normalization）")
        normalization_info_label.setWordWrap(True)
        normalization_info_label.setStyleSheet("color: #666; font-size: 9pt;")
        preprocess_params_layout.addRow(normalization_info_label)
        
        # StandardScaler标准化（用于分类算法）
        self.classification_standardscaler_check = QCheckBox("启用 StandardScaler 标准化（推荐，默认开启）")
        self.classification_standardscaler_check.setChecked(True)  # 默认开启
        self.classification_standardscaler_check.setToolTip("为所有非PLS算法启用StandardScaler标准化，确保在LOO-CV期间μ和σ只在训练折上计算，防止数据泄露。PLS-DA不使用此选项，因为它自带中心化和缩放。")
        preprocess_params_layout.addRow(self.classification_standardscaler_check)
        
        # Adaptive OBS (自适应正交背景抑制)
        self.classification_adaptive_obs_check = QCheckBox("启用自适应 OBS (Adaptive OBS)")
        self.classification_adaptive_obs_check.setChecked(False)
        self.classification_adaptive_obs_check.setToolTip("启用自适应正交背景抑制算法，用于从矿物基质中提取微量有机物信号")
        preprocess_params_layout.addRow(self.classification_adaptive_obs_check)
        
        self.classification_obs_n_components_spin = QSpinBox()
        self.classification_obs_n_components_spin.setRange(-999999999, 999999999)
        self.classification_obs_n_components_spin.setValue(5)
        self.classification_obs_n_components_spin.setToolTip("背景主成分数（建议4-6）")
        self.classification_obs_n_components_spin.setEnabled(False)
        self.classification_adaptive_obs_check.toggled.connect(lambda checked: self.classification_obs_n_components_spin.setEnabled(checked))
        preprocess_params_layout.addRow("背景主成分数 (n_components):", self.classification_obs_n_components_spin)
        
        self.classification_obs_organic_ranges_input = QLineEdit()
        self.classification_obs_organic_ranges_input.setText("2800-3050, 1600-1750")
        self.classification_obs_organic_ranges_input.setToolTip("有机物敏感区（避让区），格式：\"start-end, start-end\"，用于在学习背景时忽略这些区域")
        self.classification_obs_organic_ranges_input.setEnabled(False)
        self.classification_adaptive_obs_check.toggled.connect(lambda checked: self.classification_obs_organic_ranges_input.setEnabled(checked))
        preprocess_params_layout.addRow("有机物敏感区 (避让区):", self.classification_obs_organic_ranges_input)
        
        # 全局动态范围压缩
        self.classification_global_transform_combo = QComboBox()
        self.classification_global_transform_combo.addItems(['无', '对数变换 (Log)', '平方根变换 (Sqrt)'])
        self.classification_global_transform_combo.setCurrentText('无')
        self.classification_global_log_base_combo = QComboBox()
        self.classification_global_log_base_combo.addItems(['10', 'e'])
        self.classification_global_log_base_combo.setCurrentText('10')
        self.classification_global_log_offset_spin = QDoubleSpinBox()
        self.classification_global_log_offset_spin.setRange(-999999999.0, 999999999.0)
        self.classification_global_log_offset_spin.setDecimals(15)
        self.classification_global_log_offset_spin.setValue(1.0)
        self.classification_global_sqrt_offset_spin = QDoubleSpinBox()
        self.classification_global_sqrt_offset_spin.setRange(-999999999.0, 999999999.0)
        self.classification_global_sqrt_offset_spin.setDecimals(15)
        self.classification_global_sqrt_offset_spin.setValue(0.0)
        preprocess_params_layout.addRow("全局动态变换:", self.classification_global_transform_combo)
        preprocess_params_layout.addRow("对数底数:", self.classification_global_log_base_combo)
        preprocess_params_layout.addRow("对数偏移:", self.classification_global_log_offset_spin)
        preprocess_params_layout.addRow("平方根偏移:", self.classification_global_sqrt_offset_spin)
        
        # 二次导数
        self.classification_derivative_check = QCheckBox("应用二次导数")
        self.classification_derivative_check.setChecked(False)
        preprocess_params_layout.addRow(self.classification_derivative_check)
        
        # 整体Y轴偏移（预处理最后一步，在二次导数之后）
        self.classification_global_y_offset_spin = QDoubleSpinBox()
        self.classification_global_y_offset_spin.setRange(-999999999.0, 999999999.0)
        self.classification_global_y_offset_spin.setDecimals(15)
        self.classification_global_y_offset_spin.setValue(0.0)
        self.classification_global_y_offset_spin.setToolTip("整体Y轴偏移（预处理最后一步，在二次导数之后应用）")
        preprocess_params_layout.addRow("整体Y轴偏移（预处理）:", self.classification_global_y_offset_spin)
        
        # 同步主菜单参数按钮
        sync_preprocess_btn = QPushButton("🔄 从主菜单同步预处理参数")
        sync_preprocess_btn.clicked.connect(self._sync_preprocess_params_from_main)
        preprocess_params_layout.addRow(sync_preprocess_btn)
        
        preprocess_params_group.setContentLayout(preprocess_params_layout)
        classification_layout.addWidget(preprocess_params_group)
        
        # 算法参数配置面板（移到分类验证Tab中）
        algo_params_group = CollapsibleGroupBox("⚙️ 算法参数配置", is_expanded=False)
        algo_params_layout = QFormLayout()
        
        # SVC参数
        self.svc_kernel_combo = QComboBox()
        self.svc_kernel_combo.addItems(['rbf', 'linear', 'poly', 'sigmoid'])
        self.svc_kernel_combo.setCurrentText('rbf')
        self.svc_c_spin = QDoubleSpinBox()
        self.svc_c_spin.setRange(-999999999.0, 999999999.0)
        self.svc_c_spin.setDecimals(15)
        self.svc_c_spin.setValue(1.0)
        self.svc_gamma_combo = QComboBox()
        self.svc_gamma_combo.addItems(['scale', 'auto', '0.001', '0.01', '0.1', '1.0'])
        self.svc_gamma_combo.setCurrentText('scale')
        algo_params_layout.addRow("SVC 核函数:", self.svc_kernel_combo)
        algo_params_layout.addRow("SVC C参数:", self.svc_c_spin)
        algo_params_layout.addRow("SVC Gamma:", self.svc_gamma_combo)
        
        # PLS-DA参数
        self.plsda_ncomp_spin = QSpinBox()
        self.plsda_ncomp_spin.setRange(-999999999, 999999999)
        self.plsda_ncomp_spin.setValue(0)
        self.plsda_ncomp_spin.setToolTip("PLS-DA成分数（如果设为0则自动优化）")
        algo_params_layout.addRow("PLS-DA 成分数 (0=自动):", self.plsda_ncomp_spin)
        
        # Logistic Regression参数
        self.lr_c_spin = QDoubleSpinBox()
        self.lr_c_spin.setRange(-999999999.0, 999999999.0)
        self.lr_c_spin.setDecimals(15)
        self.lr_c_spin.setValue(1.0)
        self.lr_solver_combo = QComboBox()
        self.lr_solver_combo.addItems(['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
        self.lr_solver_combo.setCurrentText('lbfgs')
        algo_params_layout.addRow("LR C参数:", self.lr_c_spin)
        algo_params_layout.addRow("LR Solver:", self.lr_solver_combo)
        
        # k-NN参数
        self.knn_n_neighbors_spin = QSpinBox()
        self.knn_n_neighbors_spin.setRange(-999999999, 999999999)
        self.knn_n_neighbors_spin.setValue(5)
        self.knn_weights_combo = QComboBox()
        self.knn_weights_combo.addItems(['uniform', 'distance'])
        self.knn_weights_combo.setCurrentText('uniform')
        algo_params_layout.addRow("k-NN 邻居数:", self.knn_n_neighbors_spin)
        algo_params_layout.addRow("k-NN 权重:", self.knn_weights_combo)
        
        # Random Forest参数
        self.rf_n_estimators_spin = QSpinBox()
        self.rf_n_estimators_spin.setRange(-999999999, 999999999)
        self.rf_n_estimators_spin.setValue(100)
        self.rf_max_depth_spin = QSpinBox()
        self.rf_max_depth_spin.setRange(-999999999, 999999999)
        self.rf_max_depth_spin.setValue(0)
        self.rf_max_depth_spin.setSpecialValueText("无限制")
        self.rf_max_depth_spin.setToolTip("设为0表示无限制深度")
        algo_params_layout.addRow("RF 树数量:", self.rf_n_estimators_spin)
        algo_params_layout.addRow("RF 最大深度 (0=无限制):", self.rf_max_depth_spin)
        
        # PCA+LDA参数
        self.pcalda_ncomp_spin = QSpinBox()
        self.pcalda_ncomp_spin.setRange(-999999999, 999999999)
        self.pcalda_ncomp_spin.setValue(0)
        self.pcalda_ncomp_spin.setToolTip("PCA成分数（如果设为0则自动优化）")
        algo_params_layout.addRow("PCA+LDA PCA成分数 (0=自动):", self.pcalda_ncomp_spin)
        
        # AdaBoost参数
        self.adaboost_n_estimators_spin = QSpinBox()
        self.adaboost_n_estimators_spin.setRange(-999999999, 999999999)
        self.adaboost_n_estimators_spin.setValue(50)
        self.adaboost_learning_rate_spin = QDoubleSpinBox()
        self.adaboost_learning_rate_spin.setRange(-999999999.0, 999999999.0)
        self.adaboost_learning_rate_spin.setDecimals(15)
        self.adaboost_learning_rate_spin.setValue(1.0)
        algo_params_layout.addRow("AdaBoost 估计器数:", self.adaboost_n_estimators_spin)
        algo_params_layout.addRow("AdaBoost 学习率:", self.adaboost_learning_rate_spin)
        
        algo_params_group.setContentLayout(algo_params_layout)
        classification_layout.addWidget(algo_params_group)
        
        # 运行分类验证按钮
        classification_button_layout = QHBoxLayout()
        self.btn_run_classification = QPushButton("运行分类验证")
        self.btn_run_classification.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_run_classification.setToolTip("对选中的低浓度样本进行分类验证")
        self.btn_run_classification.clicked.connect(self.run_classification_validation)
        classification_button_layout.addWidget(self.btn_run_classification)
        classification_button_layout.addStretch(1)
        classification_layout.addLayout(classification_button_layout)
        
        tab3_layout.addWidget(classification_group)
        tab3_layout.addStretch()
        
        self.tab_widget.addTab(tab3, "🔬 分类验证")
        
        # Tab 4: 结果控制与验证
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        
        # 结果窗口控制区域
        result_control_group = QGroupBox("校准结果窗口控制")
        result_control_layout = QVBoxLayout(result_control_group)
        
        result_info_label = QLabel("✓ 校准结果将在独立窗口中显示\n"
                                  "✓ 调整样式参数（颜色、柱宽等）后点击'更新图表'按钮即可实时预览\n"
                                  "✓ 调整计算参数（文件分组、回归模式等）需要重新运行'运行校准计算'\n"
                                  "✓ 窗口位置会自动保持，方便对比不同参数的效果")
        result_info_label.setWordWrap(True)
        result_info_label.setStyleSheet("color: #2196F3; font-size: 9pt; padding: 5px; background-color: #E3F2FD; border-radius: 3px;")
        result_control_layout.addWidget(result_info_label)
        
        # 更新图表按钮（仅重新绘图，不重新计算）
        update_plot_layout = QHBoxLayout()
        self.btn_update_plot = QPushButton("🔄 更新图表")
        self.btn_update_plot.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_update_plot.setToolTip("使用当前样式参数重新绘制图表（不重新计算，仅更新显示效果）")
        self.btn_update_plot.clicked.connect(self.update_plot_only)
        self.btn_update_plot.setEnabled(False)  # 初始禁用，计算完成后启用
        update_plot_layout.addWidget(self.btn_update_plot)
        update_plot_layout.addStretch(1)
        result_control_layout.addLayout(update_plot_layout)
        
        # NMF拟合验证区域
        validation_group = QGroupBox("NMF拟合验证")
        validation_layout = QVBoxLayout(validation_group)
        
        validation_info_label = QLabel("✓ 选择待测样品并点击'验证拟合'按钮查看原始光谱与拟合结果的对比\n"
                                      "✓ 可以查看拟合质量、残差分布和局部放大细节")
        validation_info_label.setWordWrap(True)
        validation_info_label.setStyleSheet("color: #2196F3; font-size: 9pt; padding: 5px; background-color: #E3F2FD; border-radius: 3px;")
        validation_layout.addWidget(validation_info_label)
        
        # 样本选择下拉框
        sample_select_layout = QHBoxLayout()
        sample_select_layout.addWidget(QLabel("选择样本:"))
        self.sample_select_combo = QComboBox()
        self.sample_select_combo.setToolTip("选择要验证拟合的待测样品")
        sample_select_layout.addWidget(self.sample_select_combo)
        sample_select_layout.addStretch(1)
        validation_layout.addLayout(sample_select_layout)
        
        # 验证拟合按钮
        validation_button_layout = QHBoxLayout()
        self.btn_check_fitting = QPushButton("验证拟合 (Check Fitting)")
        self.btn_check_fitting.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_check_fitting.setToolTip("打开NMF拟合验证窗口，显示选中样本的原始光谱与拟合结果对比")
        self.btn_check_fitting.clicked.connect(self.check_fitting)
        self.btn_check_fitting.setEnabled(False)  # 初始禁用，计算完成后启用
        validation_button_layout.addWidget(self.btn_check_fitting)
        validation_button_layout.addStretch(1)
        validation_layout.addLayout(validation_button_layout)
        
        tab4_layout.addWidget(result_control_group)
        tab4_layout.addWidget(validation_group)
        tab4_layout.addStretch()
        
        self.tab_widget.addTab(tab4, "📈 结果控制")
        
        # 将TabWidget添加到主布局
        self.main_layout.addWidget(self.tab_widget)
        
        # 存储计算结果
        self.w_bias = None
        self.w_low = None
        self.w_calibrated = None
        self.sample_names = []
        
        # 存储NMF回归的完整数据（用于拟合验证）
        self.W_sample = None  # 权重矩阵 (n_samples, n_components)
        self.fixed_H = None  # 固定的H矩阵
        self.common_x = None  # 波数轴
        self.sample_files = []  # 样本文件路径列表
        
        # 独立的结果窗口
        self.result_window = None
        
        # NMF拟合验证窗口
        self.fit_validation_window = None
        
        # 分类验证窗口
        self.classification_window = None
        
        # 在所有控件创建完成后，从主程序同步默认参数
        self._sync_default_params()
        
        # 同步预处理参数
        self._sync_preprocess_params_from_main()
        
        # 连接样式参数控件的自动更新信号（当结果窗口存在且有数据时自动更新）
        self._connect_style_update_signals()
    
    def _sync_preprocess_params_from_main(self):
        """从主菜单同步预处理参数到分类验证Tab"""
        if self.parent_dialog:
            try:
                # 同步QC参数
                if hasattr(self.parent_dialog, 'qc_check'):
                    self.classification_qc_check.setChecked(self.parent_dialog.qc_check.isChecked())
                    self.classification_qc_threshold_spin.setValue(self.parent_dialog.qc_threshold_spin.value())
                
                # 同步BE参数
                if hasattr(self.parent_dialog, 'be_check'):
                    self.classification_be_check.setChecked(self.parent_dialog.be_check.isChecked())
                    self.classification_be_temp_spin.setValue(self.parent_dialog.be_temp_spin.value())
                
                # 同步平滑参数
                if hasattr(self.parent_dialog, 'smoothing_check'):
                    self.classification_smoothing_check.setChecked(self.parent_dialog.smoothing_check.isChecked())
                    self.classification_smoothing_window_spin.setValue(self.parent_dialog.smoothing_window_spin.value())
                    self.classification_smoothing_poly_spin.setValue(self.parent_dialog.smoothing_poly_spin.value())
                
                # 同步AsLS参数
                if hasattr(self.parent_dialog, 'baseline_als_check'):
                    self.classification_baseline_als_check.setChecked(self.parent_dialog.baseline_als_check.isChecked())
                    self.classification_lam_spin.setValue(self.parent_dialog.lam_spin.value())
                    self.classification_p_spin.setValue(self.parent_dialog.p_spin.value())
                
                # 同步全局动态变换参数
                if hasattr(self.parent_dialog, 'global_transform_combo'):
                    index = self.classification_global_transform_combo.findText(self.parent_dialog.global_transform_combo.currentText())
                    if index >= 0:
                        self.classification_global_transform_combo.setCurrentIndex(index)
                    self.classification_global_log_base_combo.setCurrentText(self.parent_dialog.global_log_base_combo.currentText())
                    self.classification_global_log_offset_spin.setValue(self.parent_dialog.global_log_offset_spin.value())
                    self.classification_global_sqrt_offset_spin.setValue(self.parent_dialog.global_sqrt_offset_spin.value())
                
                # 同步二次导数参数
                if hasattr(self.parent_dialog, 'derivative_check'):
                    self.classification_derivative_check.setChecked(self.parent_dialog.derivative_check.isChecked())
                
                # 同步整体Y轴偏移参数
                if hasattr(self.parent_dialog, 'global_y_offset_spin'):
                    self.classification_global_y_offset_spin.setValue(self.parent_dialog.global_y_offset_spin.value())
                
                QMessageBox.information(self, "完成", "已成功同步主菜单的预处理参数！")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"同步预处理参数时出错: {e}")
                traceback.print_exc()
    
    def _sync_default_params(self):
        """从主程序同步默认参数"""
        if self.parent_dialog:
            try:
                # 同步图尺寸和DPI
                if hasattr(self.parent_dialog, 'fig_width_spin'):
                    self.result_fig_width_spin.setValue(self.parent_dialog.fig_width_spin.value())
                if hasattr(self.parent_dialog, 'fig_height_spin'):
                    self.result_fig_height_spin.setValue(self.parent_dialog.fig_height_spin.value())
                if hasattr(self.parent_dialog, 'fig_dpi_spin'):
                    self.result_fig_dpi_spin.setValue(self.parent_dialog.fig_dpi_spin.value())
                
                # 同步字体设置
                if hasattr(self.parent_dialog, 'font_family_combo'):
                    font_family = self.parent_dialog.font_family_combo.currentText()
                    index = self.result_font_family_combo.findText(font_family)
                    if index >= 0:
                        self.result_font_family_combo.setCurrentIndex(index)
                
                if hasattr(self.parent_dialog, 'axis_title_font_spin'):
                    self.result_axis_title_font_spin.setValue(self.parent_dialog.axis_title_font_spin.value())
                if hasattr(self.parent_dialog, 'tick_label_font_spin'):
                    self.result_tick_label_font_spin.setValue(self.parent_dialog.tick_label_font_spin.value())
                if hasattr(self.parent_dialog, 'legend_font_spin'):
                    self.result_legend_font_spin.setValue(self.parent_dialog.legend_font_spin.value())
                
                QMessageBox.information(self, "完成", "已成功同步主程序的默认设置！")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"同步默认参数时出错: {e}")
        else:
            QMessageBox.warning(self, "警告", "无法访问主程序，请确保对话框已正确初始化。")
    
    def _connect_style_update_signals(self):
        """连接样式参数控件的自动更新信号"""
        # 柱状图样式参数
        self.result_bar_width_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_bar_alpha_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_bar_edge_width_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_bar_edge_color_input.textChanged.connect(self._on_style_param_changed)
        self.result_bar_hatch_combo.currentTextChanged.connect(self._on_style_param_changed)
        
        # 颜色设置
        self.result_color_low_input.textChanged.connect(self._on_style_param_changed)
        self.result_color_calibrated_input.textChanged.connect(self._on_style_param_changed)
        self.result_color_bias_input.textChanged.connect(self._on_style_param_changed)
        
        # 连接新增的样式控件
        self.result_tick_direction_combo.currentTextChanged.connect(self._on_style_param_changed)
        self.result_tick_len_major_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_tick_len_minor_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_tick_width_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_spine_top_check.stateChanged.connect(self._on_style_param_changed)
        self.result_spine_bottom_check.stateChanged.connect(self._on_style_param_changed)
        self.result_spine_left_check.stateChanged.connect(self._on_style_param_changed)
        self.result_spine_right_check.stateChanged.connect(self._on_style_param_changed)
        self.result_spine_width_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_legend_ncol_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_legend_columnspacing_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_legend_labelspacing_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_legend_handlelength_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_aspect_ratio_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 辅助线样式
        self.result_bias_line_style_combo.currentTextChanged.connect(self._on_style_param_changed)
        self.result_bias_line_width_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 网格设置
        self.result_show_grid_check.stateChanged.connect(self._on_style_param_changed)
        self.result_grid_alpha_spin.valueChanged.connect(self._on_style_param_changed)
        
        # X轴标签旋转
        self.result_xlabel_rotation_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 图例设置
        if hasattr(self, 'result_show_legend_check'):
            self.result_show_legend_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'result_legend_frame_check'):
            self.result_legend_frame_check.stateChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'result_legend_loc_combo'):
            self.result_legend_loc_combo.currentTextChanged.connect(self._on_style_param_changed)
        
        # 字体和标题设置
        self.result_font_family_combo.currentTextChanged.connect(self._on_style_param_changed)
        self.result_axis_title_font_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_tick_label_font_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_legend_font_spin.valueChanged.connect(self._on_style_param_changed)
        
        # 标题和标签设置
        self.result_title_input.textChanged.connect(self._on_style_param_changed)
        self.result_title_font_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_title_pad_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_title_show_check.stateChanged.connect(self._on_style_param_changed)
        
        self.result_xlabel_input.textChanged.connect(self._on_style_param_changed)
        self.result_xlabel_font_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_xlabel_pad_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_xlabel_show_check.stateChanged.connect(self._on_style_param_changed)
        
        self.result_ylabel_input.textChanged.connect(self._on_style_param_changed)
        self.result_ylabel_font_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_ylabel_pad_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_ylabel_show_check.stateChanged.connect(self._on_style_param_changed)
        
        # 图尺寸和DPI
        self.result_fig_width_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_fig_height_spin.valueChanged.connect(self._on_style_param_changed)
        self.result_fig_dpi_spin.valueChanged.connect(self._on_style_param_changed)
    
    def _on_style_param_changed(self):
        """样式参数变化时的回调函数（自动更新图表）"""
        # 只有在计算结果已存在时才自动更新
        if self.w_low is not None and self.w_calibrated is not None:
            # 使用QTimer延迟更新，避免频繁触发（防抖）
            if not hasattr(self, '_update_timer'):
                self._update_timer = QTimer()
                self._update_timer.setSingleShot(True)
                self._update_timer.timeout.connect(self.update_plot_only)
            
            # 重置定时器，300ms后执行更新
            self._update_timer.stop()
            self._update_timer.start(300)
    
    def _create_h_layout(self, widgets):
        """创建水平布局的辅助方法"""
        h_layout = QHBoxLayout()
        for widget in widgets:
            h_layout.addWidget(widget)
        h_layout.addStretch(1)
        return h_layout
    
    def _create_color_picker_button(self, color_input):
        """创建颜色选择器按钮的辅助方法"""
        color_button = QPushButton("🎨")
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
    
    def _get_checked_border_sides(self):
        """获取选中的边框边"""
        sides = []
        if hasattr(self, 'result_spine_top_check') and self.result_spine_top_check.isChecked():
            sides.append('top')
        if hasattr(self, 'result_spine_bottom_check') and self.result_spine_bottom_check.isChecked():
            sides.append('bottom')
        if hasattr(self, 'result_spine_left_check') and self.result_spine_left_check.isChecked():
            sides.append('left')
        if hasattr(self, 'result_spine_right_check') and self.result_spine_right_check.isChecked():
            sides.append('right')
        # 如果没有边框控件，返回默认值（所有边框）
        if not sides:
            sides = ['top', 'right', 'left', 'bottom']
        return sides
    
    def _add_files_to_list(self, target_list):
        """添加文件到指定列表"""
        if not self.parent_dialog:
            QMessageBox.warning(self, "错误", "无法访问主窗口。")
            return
        
        folder = self.parent_dialog.folder_input.text()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "错误", "请先选择数据文件夹。")
            return
        
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择文件", folder, "数据文件 (*.txt *.csv);;所有文件 (*.*)")
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            # 检查是否已存在
            existing_items = [target_list.item(i).data(Qt.ItemDataRole.UserRole) 
                            for i in range(target_list.count())]
            if file_path not in existing_items:
                item = QListWidgetItem(file_name)
                item.setData(Qt.ItemDataRole.UserRole, file_path)  # 存储完整路径
                target_list.addItem(item)
    
    def _remove_selected_from_list(self, target_list):
        """从列表中移除选中的项"""
        selected_items = target_list.selectedItems()
        for item in selected_items:
            target_list.takeItem(target_list.row(item))
    
    def run_calculation(self):
        """执行校准计算"""
        try:
            # 前提检查
            if not self.parent_dialog:
                QMessageBox.critical(self, "错误", "无法访问主窗口。")
                return
            
            if self.parent_dialog.last_fixed_H is None:
                QMessageBox.warning(self, "错误", "请先运行标准NMF分析以获取固定的H矩阵。")
                return
            
            target_idx = self.parent_dialog.get_nmf_target_component_index()
            if target_idx is None:
                QMessageBox.warning(self, "错误", "请在NMF结果窗口中指定目标组分索引。")
                return
            
            # 获取文件列表
            blank_files = []
            for i in range(self.blanks_list.count()):
                item = self.blanks_list.item(i)
                blank_files.append(item.data(Qt.ItemDataRole.UserRole))
            
            sample_files = []
            for i in range(self.samples_list.count()):
                item = self.samples_list.item(i)
                sample_files.append(item.data(Qt.ItemDataRole.UserRole))
            
            if not blank_files:
                QMessageBox.warning(self, "错误", "请至少添加一个空白样品文件。")
                return
            
            if not sample_files:
                QMessageBox.warning(self, "错误", "请至少添加一个待测样品文件。")
                return
            
            # 获取固定H矩阵
            fixed_H = self.parent_dialog.last_fixed_H
            
            # 计算空白样品的权重矩阵
            W_blank, _, _, blank_labels = self.parent_dialog.run_nmf_regression(blank_files, fixed_H)
            if W_blank is None:
                QMessageBox.critical(self, "错误", "空白样品权重计算失败。")
                return
            
            # 计算待测样品的权重矩阵
            W_sample, fixed_H_result, common_x_result, sample_labels = self.parent_dialog.run_nmf_regression(sample_files, fixed_H)
            if W_sample is None:
                QMessageBox.critical(self, "错误", "待测样品权重计算失败。")
                return
            
            # 存储完整数据用于拟合验证
            self.W_sample = W_sample
            # fixed_H_result是预滤波空间中的H（用于回归），但我们需要保存原始空间的H用于绘图
            # 优先使用parent_dialog保存的原始空间H
            if hasattr(self.parent_dialog, 'last_fixed_H_original') and self.parent_dialog.last_fixed_H_original is not None:
                self.fixed_H = self.parent_dialog.last_fixed_H_original.copy()
            else:
                self.fixed_H = fixed_H_result.copy()
            
            # 优先使用parent_dialog保存的common_x（训练时的波数轴），确保与H矩阵对齐
            if hasattr(self.parent_dialog, 'last_common_x') and self.parent_dialog.last_common_x is not None:
                self.common_x = self.parent_dialog.last_common_x.copy()
            else:
                self.common_x = common_x_result.copy()
            self.sample_files = sample_files
            
            # 更新样本选择下拉框
            self.sample_select_combo.clear()
            for label in sample_labels:
                self.sample_select_combo.addItem(label)
            
            # 空白校准：计算目标组分索引列的平均值
            if target_idx >= W_blank.shape[1]:
                QMessageBox.critical(self, "错误", f"目标组分索引 {target_idx} 超出范围（组分数量：{W_blank.shape[1]}）。")
                return
            
            w_bias = np.mean(W_blank[:, target_idx])
            
            # 计算 LOD 和 LOQ
            w_blank_target = W_blank[:, target_idx]
            std_blank = np.std(w_blank_target)
            S_sensitivity = 1.0  # 灵敏度（斜率），这里简化为1.0，实际应该从校准曲线获取
            
            LOD = 3.3 * std_blank / S_sensitivity if S_sensitivity > 0 else 0.0
            LOQ = 10.0 * std_blank / S_sensitivity if S_sensitivity > 0 else 0.0
            
            # 根据选择的回归模式处理待测样品
            if self.mode_average.isChecked():
                # 模式B：平均回归 - 先计算多条低浓度组分的平均权重
                w_low_mean = np.mean(W_sample[:, target_idx])
                w_low = np.array([w_low_mean])  # 转换为数组以保持一致性
                sample_labels = ["平均结果"]  # 更新标签
                QMessageBox.information(self, "提示", f"使用平均回归模式：\n"
                                                      f"待测样品数量：{len(W_sample)}\n"
                                                      f"平均权重值：{w_low_mean:.6f}")
            else:
                # 模式A：单独回归 - 每条低浓度组分单独计算权重
                w_low = W_sample[:, target_idx].copy()
                sample_labels = sample_labels  # 保持原有标签
            
            # 计算校准权重
            w_calibrated = w_low - w_bias
            w_calibrated[w_calibrated < 0] = 0  # 负值置零
            
            # 归一化处理（如果启用）
            if self.result_normalization_check.isChecked():
                norm_mode = self.result_normalization_combo.currentText()
                if norm_mode == 'max':
                    max_val = max(np.max(w_low), np.max(w_calibrated))
                    if max_val > 0:
                        w_low = w_low / max_val
                        w_calibrated = w_calibrated / max_val
                elif norm_mode == 'area':
                    area_low = np.sum(np.abs(w_low))
                    area_cal = np.sum(np.abs(w_calibrated))
                    if area_low > 0:
                        w_low = w_low / area_low
                    if area_cal > 0:
                        w_calibrated = w_calibrated / area_cal
            
            # 存储结果
            self.w_bias = w_bias
            self.w_low = w_low
            self.w_calibrated = w_calibrated
            self.sample_names = sample_labels  # 根据模式可能已更新
            
            # 绘制结果（在独立窗口中）
            self.plot_results()
            
            # 启用更新图表按钮
            if hasattr(self, 'btn_update_plot'):
                self.btn_update_plot.setEnabled(True)
            
            # 更新窗口标题以包含 LOD/LOQ
            if hasattr(self, 'result_window') and self.result_window:
                self.result_window.setWindowTitle(
                    f"定量校准结果 (LOD={LOD:.4f}, LOQ={LOQ:.4f})"
                )
            
            QMessageBox.information(self, "完成", 
                                  f"校准计算完成！\n"
                                  f"空白偏差 (w_bias) = {w_bias:.6f}\n"
                                  f"检出限 (LOD) = {LOD:.4f}\n"
                                  f"定量限 (LOQ) = {LOQ:.4f}\n\n"
                                  f"结果图已在独立窗口中显示，您可以调整样式参数后点击'更新图表'按钮实时预览效果。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败：{str(e)}")
            traceback.print_exc()
    
    def plot_results(self):
        """绘制校准结果柱状图（使用独立窗口，参考4.py）"""
        if self.w_low is None or self.w_calibrated is None:
            return
        
        # 创建或更新独立窗口 - 保留窗口位置
        if self.result_window is None or not self.result_window.isVisible():
            # 如果窗口不存在或已关闭，创建新窗口
            self.result_window = QuantitativeResultWindow(self)
        else:
            # 如果窗口已存在，保留其位置
            if hasattr(self.result_window, 'last_geometry') and self.result_window.last_geometry:
                pass  # 位置会在update_plot中恢复
        
        # 准备绘图参数（所有参数从对话框传递，包括样式参数）
        # 从主窗口获取样式参数（如果可用）
        parent_dialog = self.parent_dialog if hasattr(self, 'parent_dialog') else None
        
        plot_params = {
            'fig_width': self.result_fig_width_spin.value(),
            'fig_height': self.result_fig_height_spin.value(),
            'fig_dpi': self.result_fig_dpi_spin.value(),
            'title': self.result_title_input.text().strip() or "定量校准结果",
            'title_fontsize': self.result_title_font_spin.value(),
            'title_pad': self.result_title_pad_spin.value(),
            'title_show': self.result_title_show_check.isChecked(),
            'xlabel': self.result_xlabel_input.text().strip() or "样品名称",
            'xlabel_fontsize': self.result_xlabel_font_spin.value(),
            'xlabel_pad': self.result_xlabel_pad_spin.value(),
            'xlabel_show': self.result_xlabel_show_check.isChecked(),
            'ylabel': self.result_ylabel_input.text().strip() or "权重值",
            'ylabel_fontsize': self.result_ylabel_font_spin.value(),
            'ylabel_pad': self.result_ylabel_pad_spin.value(),
            'ylabel_show': self.result_ylabel_show_check.isChecked(),
            'font_family': self.result_font_family_combo.currentText(),
            'axis_title_fontsize': self.result_axis_title_font_spin.value(),
            'tick_label_fontsize': self.result_tick_label_font_spin.value(),
            'legend_fontsize': self.result_legend_font_spin.value(),
            # 图例高级控制参数（使用自己的控件）
            'legend_ncol': self.result_legend_ncol_spin.value(),
            'legend_columnspacing': self.result_legend_columnspacing_spin.value(),
            'legend_labelspacing': self.result_legend_labelspacing_spin.value(),
            'legend_handlelength': self.result_legend_handlelength_spin.value(),
            'show_legend': self.result_show_legend_check.isChecked() if hasattr(self, 'result_show_legend_check') else True,
            'legend_frame': self.result_legend_frame_check.isChecked() if hasattr(self, 'result_legend_frame_check') else True,
            'legend_loc': self.result_legend_loc_combo.currentText() if hasattr(self, 'result_legend_loc_combo') else 'best',
            # 刻度样式参数（使用自己的控件）
            'tick_direction': self.result_tick_direction_combo.currentText(),
            'tick_len_major': self.result_tick_len_major_spin.value(),
            'tick_len_minor': self.result_tick_len_minor_spin.value(),
            'tick_width': self.result_tick_width_spin.value(),
            # 边框样式参数（使用自己的控件）
            'border_sides': self._get_checked_border_sides(),
            'border_linewidth': self.result_spine_width_spin.value(),
            'bar_width': self.result_bar_width_spin.value(),
            'bar_alpha': self.result_bar_alpha_spin.value(),
            'bar_edge_color': self.result_bar_edge_color_input.text().strip() or 'black',
            'bar_edge_width': self.result_bar_edge_width_spin.value(),
            'bar_hatch': '' if self.result_bar_hatch_combo.currentText() == '无' else self.result_bar_hatch_combo.currentText(),
            'color_low': self.result_color_low_input.text().strip() or "gray",
            'color_calibrated': self.result_color_calibrated_input.text().strip() or "red",
            'color_bias': self.result_color_bias_input.text().strip() or "blue",
            'bias_line_style': self.result_bias_line_style_combo.currentText(),
            'bias_line_width': self.result_bias_line_width_spin.value(),
            # 纵横比控制（使用自己的控件）
            'aspect_ratio': self.result_aspect_ratio_spin.value(),
            'show_grid': self.result_show_grid_check.isChecked(),
            'grid_alpha': self.result_grid_alpha_spin.value(),
            'xlabel_rotation': self.result_xlabel_rotation_spin.value(),
            # 图例重命名映射（移除对主窗口的依赖，使用空字典）
            'legend_names': {},
            'w_low': self.w_low,
            'w_calibrated': self.w_calibrated,
            'w_bias': self.w_bias,
            'sample_names': self.sample_names
        }
        
        # 处理图例重命名映射（转换为实际使用的格式）
        if plot_params['legend_names']:
            rename_map = {}
            for key, widget in plot_params['legend_names'].items():
                if hasattr(widget, 'text'):
                    renamed = widget.text().strip()
                    if renamed:
                        rename_map[key] = renamed
            plot_params['legend_names'] = rename_map
        
        self.result_window.update_plot(plot_params)
        self.btn_check_fitting.setEnabled(True)  # 启用验证拟合按钮
    
    def _validate_color(self, color_str):
        """验证颜色字符串是否有效"""
        if not color_str or not color_str.strip():
            return False
        color_str = color_str.strip()
        # 检查是否是matplotlib支持的颜色名称
        try:
            import matplotlib.colors as mcolors
            # 尝试转换为RGB
            mcolors.to_rgba(color_str)
            return True
        except (ValueError, AttributeError):
            return False
    
    def update_plot_only(self):
        """仅更新图表显示（不重新计算），用于样式参数调整后的实时预览"""
        if self.w_low is None or self.w_calibrated is None:
            QMessageBox.warning(self, "提示", "请先运行校准计算以生成结果数据。")
            return
        
        # 验证颜色输入
        colors_to_check = [
            ('柱边框颜色', self.result_bar_edge_color_input.text().strip() or 'black'),
            ('原始权重颜色', self.result_color_low_input.text().strip() or 'gray'),
            ('校准权重颜色', self.result_color_calibrated_input.text().strip() or 'red'),
            ('空白偏差颜色', self.result_color_bias_input.text().strip() or 'blue')
        ]
        
        invalid_colors = []
        for name, color in colors_to_check:
            if not self._validate_color(color):
                invalid_colors.append(f"{name}: '{color}'")
        
        if invalid_colors:
            QMessageBox.warning(self, "颜色输入错误", 
                              f"以下颜色输入无效，请使用有效的颜色名称（如 'red', 'blue', '#FF0000' 等）：\n\n" + 
                              "\n".join(invalid_colors))
            return
        
        # 直接调用 plot_results 更新图表
        self.plot_results()
    
    def check_fitting(self):
        """验证NMF拟合 - 打开拟合验证窗口"""
        try:
            # 检查是否已运行计算
            if self.W_sample is None or self.fixed_H is None or self.common_x is None:
                QMessageBox.warning(self, "警告", "请先运行校准计算。")
                return
            
            # 检查是否有样本文件
            if not self.sample_files:
                QMessageBox.warning(self, "警告", "没有可用的样本文件。")
                return
            
            # 获取选中的样本索引
            selected_idx = self.sample_select_combo.currentIndex()
            if selected_idx < 0 or selected_idx >= len(self.sample_files):
                QMessageBox.warning(self, "警告", "请选择一个有效的样本。")
                return
            
            # 获取选中的样本文件路径
            selected_file = self.sample_files[selected_idx]
            sample_name = self.sample_select_combo.currentText()
            
            # 获取该样本的权重
            if selected_idx >= self.W_sample.shape[0]:
                QMessageBox.warning(self, "错误", f"样本索引 {selected_idx} 超出范围。")
                return
            
            w_selected = self.W_sample[selected_idx, :]  # (n_components,)
            target_idx = self.parent_dialog.get_nmf_target_component_index()
            
            if target_idx is None or target_idx >= self.fixed_H.shape[0]:
                QMessageBox.warning(self, "错误", "无效的目标组分索引。")
                return
            
            # 读取原始数据并进行相同的预处理
            skip = self.parent_dialog.skip_rows_spin.value()
            x_min_phys = self.parent_dialog._parse_optional_float(self.parent_dialog.x_min_phys_input.text())
            x_max_phys = self.parent_dialog._parse_optional_float(self.parent_dialog.x_max_phys_input.text())
            
            x_raw, y_raw = self.parent_dialog.read_data(selected_file, skip, x_min_phys, x_max_phys)
            
            # 应用相同的预处理
            y_proc = y_raw.astype(float)
            
            # 1. QC 检查（如果启用）
            if self.parent_dialog.qc_check.isChecked() and np.max(y_proc) < self.parent_dialog.qc_threshold_spin.value():
                QMessageBox.warning(self, "警告", "该样本未通过QC质量检查。")
                return
            
            # 2. BE 校正（如果启用）
            if self.parent_dialog.be_check.isChecked():
                y_proc = DataPreProcessor.apply_bose_einstein_correction(x_raw, y_proc, self.parent_dialog.be_temp_spin.value())
            
            # 3. 平滑（如果启用）
            if self.parent_dialog.smoothing_check.isChecked():
                y_proc = DataPreProcessor.apply_smoothing(y_proc, self.parent_dialog.smoothing_window_spin.value(), 
                                                          self.parent_dialog.smoothing_poly_spin.value())
            
            # 4. 基线校正（如果启用）
            if self.parent_dialog.baseline_als_check.isChecked():
                b = DataPreProcessor.apply_baseline_als(y_proc, self.parent_dialog.lam_spin.value(), 
                                                        self.parent_dialog.p_spin.value())
                y_proc = y_proc - b
                y_proc[y_proc < 0] = 0
            
            # 5. 归一化（如果启用）
            normalization_mode = self.parent_dialog.normalization_combo.currentText()
            if normalization_mode == 'max':
                y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
            elif normalization_mode == 'area':
                y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
            elif normalization_mode == 'snv':
                y_proc = DataPreProcessor.apply_snv(y_proc)
            
            # 6. 全局动态范围压缩（如果启用）
            global_transform_mode = self.parent_dialog.global_transform_combo.currentText()
            if global_transform_mode == '对数变换 (Log)':
                base = float(self.parent_dialog.global_log_base_combo.currentText()) if self.parent_dialog.global_log_base_combo.currentText() == '10' else np.e
                y_proc = DataPreProcessor.apply_log_transform(y_proc, base=base, 
                                                             offset=self.parent_dialog.global_log_offset_spin.value())
            elif global_transform_mode == '平方根变换 (Sqrt)':
                y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, 
                                                              offset=self.parent_dialog.global_sqrt_offset_spin.value())
            
            # 确保非负
            y_proc[y_proc < 0] = 0
            
            # 确保数据长度匹配
            if len(y_proc) != len(self.common_x):
                # 如果长度不匹配，尝试插值
                from scipy.interpolate import interp1d
                f_interp = interp1d(x_raw, y_proc, kind='linear', fill_value=0, bounds_error=False)
                y_proc = f_interp(self.common_x)
            
            # 使用原始空间的H矩阵（用于绘图和验证）
            # 优先使用parent_dialog保存的原始空间H和对应的波数轴
            if hasattr(self.parent_dialog, 'last_fixed_H_original') and self.parent_dialog.last_fixed_H_original is not None:
                H_original = self.parent_dialog.last_fixed_H_original
                # 如果保存了对应的波数轴，使用它；否则使用当前的common_x
                if hasattr(self.parent_dialog, 'last_common_x') and self.parent_dialog.last_common_x is not None:
                    # 确保H_original的维度与保存的波数轴匹配
                    if H_original.shape[1] == len(self.parent_dialog.last_common_x):
                        # 如果维度匹配，使用保存的波数轴（更准确）
                        self.common_x = self.parent_dialog.last_common_x.copy()
                    elif H_original.shape[1] != len(self.common_x):
                        # 维度不匹配，尝试插值对齐
                        from scipy.interpolate import interp1d
                        x_train = self.parent_dialog.last_common_x
                        H_aligned = np.zeros((H_original.shape[0], len(self.common_x)))
                        for i in range(H_original.shape[0]):
                            f_interp = interp1d(x_train, H_original[i, :], kind='linear', 
                                              fill_value=0, bounds_error=False)
                            H_aligned[i, :] = f_interp(self.common_x)
                        H_original = H_aligned
                        print(f"信息：H矩阵已从保存的波数轴插值对齐到当前波数轴")
            else:
                # 如果没有保存原始空间的H，使用fixed_H（可能不匹配，会报错）
                H_original = self.fixed_H.copy()
            
            # 最终确保H_original的维度正确
            if H_original.shape[1] != len(self.common_x):
                QMessageBox.warning(self, "错误", f"H矩阵维度 ({H_original.shape[1]}) 与波数轴长度 ({len(self.common_x)}) 不匹配。\n"
                                                  f"这可能是因为使用了预滤波但未保存原始空间的H矩阵，或数据维度不一致。")
                return
            
            # 计算拟合贡献：Y_fit = w * H_component（使用原始空间的H）
            H_component = H_original[target_idx, :]  # 目标组分的光谱
            w_component = w_selected[target_idx]  # 该样本的目标组分权重
            y_fit = w_component * H_component
            
            # 计算总重构：Y_total = sum(w_i * H_i) for all components（使用原始空间的H）
            y_total_reconstructed = np.zeros_like(self.common_x)
            for i in range(H_original.shape[0]):
                y_total_reconstructed += w_selected[i] * H_original[i, :]
            
            # 创建或更新验证窗口
            if self.fit_validation_window is None or not self.fit_validation_window.isVisible():
                self.fit_validation_window = NMFFitValidationWindow(self)
            
            # 获取主菜单的垂直参考线和峰值检测参数
            vertical_lines = []
            if hasattr(self.parent_dialog, 'vertical_lines_input'):
                vlines_text = self.parent_dialog.vertical_lines_input.toPlainText().strip()
                if vlines_text:
                    try:
                        # 解析垂直参考线（支持逗号、空格、换行分隔）
                        import re
                        vlines_str = re.split(r'[,;\s\n]+', vlines_text)
                        vertical_lines = [float(x.strip()) for x in vlines_str if x.strip()]
                    except:
                        pass
            
            peak_detection_enabled = False
            peak_height_threshold = 0.0
            peak_distance_min = 10
            if hasattr(self.parent_dialog, 'peak_check') and self.parent_dialog.peak_check.isChecked():
                peak_detection_enabled = True
                peak_height_threshold = self.parent_dialog.peak_height_spin.value() if hasattr(self.parent_dialog, 'peak_height_spin') else 0.0
                peak_distance_min = self.parent_dialog.peak_distance_spin.value() if hasattr(self.parent_dialog, 'peak_distance_spin') else 10
            
            # 获取对照组数据（如果存在）
            control_data_list = []
            if hasattr(self.parent_dialog, 'control_files_input'):
                control_text = self.parent_dialog.control_files_input.toPlainText().strip()
                if control_text:
                    control_names = [x.strip() for x in re.split(r'[,;\n]+', control_text) if x.strip()]
                    # 获取文件夹路径
                    folder = self.parent_dialog.folder_input.text()
                    if folder:
                        for c_name_base in control_names:
                            # 自动识别后缀
                            found_file = None
                            for ext in ['.txt', '.csv', '.TXT', '.CSV']:
                                c_name = c_name_base + ext if not c_name_base.endswith(ext) else c_name_base
                                full_p = os.path.join(folder, c_name)
                                if os.path.exists(full_p):
                                    found_file = full_p
                                    break
                            if found_file:
                                # 读取并预处理对照组数据
                                try:
                                    x_ctrl, y_ctrl = self.parent_dialog.read_data(found_file, skip, x_min_phys, x_max_phys)
                                    if x_ctrl is not None and y_ctrl is not None:
                                        # 应用相同的预处理
                                        y_ctrl_proc = y_ctrl.astype(float)
                                        if self.parent_dialog.qc_check.isChecked() and np.max(y_ctrl_proc) < self.parent_dialog.qc_threshold_spin.value():
                                            continue
                                        if self.parent_dialog.be_check.isChecked():
                                            y_ctrl_proc = DataPreProcessor.apply_bose_einstein_correction(x_ctrl, y_ctrl_proc, self.parent_dialog.be_temp_spin.value())
                                        if self.parent_dialog.smoothing_check.isChecked():
                                            y_ctrl_proc = DataPreProcessor.apply_smoothing(y_ctrl_proc, self.parent_dialog.smoothing_window_spin.value(), 
                                                                                          self.parent_dialog.smoothing_poly_spin.value())
                                        if self.parent_dialog.baseline_als_check.isChecked():
                                            b = DataPreProcessor.apply_baseline_als(y_ctrl_proc, self.parent_dialog.lam_spin.value(), 
                                                                                  self.parent_dialog.p_spin.value())
                                            y_ctrl_proc = y_ctrl_proc - b
                                            y_ctrl_proc[y_ctrl_proc < 0] = 0
                                        normalization_mode = self.parent_dialog.normalization_combo.currentText()
                                        if normalization_mode == 'max':
                                            y_ctrl_proc = DataPreProcessor.apply_normalization(y_ctrl_proc, 'max')
                                        elif normalization_mode == 'area':
                                            y_ctrl_proc = DataPreProcessor.apply_normalization(y_ctrl_proc, 'area')
                                        elif normalization_mode == 'snv':
                                            y_ctrl_proc = DataPreProcessor.apply_snv(y_ctrl_proc)
                                        global_transform_mode = self.parent_dialog.global_transform_combo.currentText()
                                        if global_transform_mode == '对数变换 (Log)':
                                            base = float(self.parent_dialog.global_log_base_combo.currentText()) if self.parent_dialog.global_log_base_combo.currentText() == '10' else np.e
                                            y_ctrl_proc = DataPreProcessor.apply_log_transform(y_ctrl_proc, base=base, 
                                                                                              offset=self.parent_dialog.global_log_offset_spin.value())
                                        elif global_transform_mode == '平方根变换 (Sqrt)':
                                            y_ctrl_proc = DataPreProcessor.apply_sqrt_transform(y_ctrl_proc, 
                                                                                               offset=self.parent_dialog.global_sqrt_offset_spin.value())
                                        y_ctrl_proc[y_ctrl_proc < 0] = 0
                                        # 确保长度匹配
                                        if len(y_ctrl_proc) != len(self.common_x):
                                            from scipy.interpolate import interp1d
                                            f_interp = interp1d(x_ctrl, y_ctrl_proc, kind='linear', fill_value=0, bounds_error=False)
                                            y_ctrl_proc = f_interp(self.common_x)
                                        # 调整强度使其与原始数据强度相近
                                        if len(y_proc) > 0 and np.max(y_proc) > 0:
                                            scale_factor = np.max(y_proc) / np.max(y_ctrl_proc) if np.max(y_ctrl_proc) > 0 else 1.0
                                            y_ctrl_proc = y_ctrl_proc * scale_factor
                                        control_data_list.append({
                                            'x': self.common_x,
                                            'y': y_ctrl_proc,
                                            'label': os.path.splitext(os.path.basename(found_file))[0]
                                        })
                                except Exception as e:
                                    print(f"读取对照组文件 {found_file} 时出错: {e}")
                                    continue
            
            # 设置数据并更新绘图
            self.fit_validation_window.set_data(
                x_data=self.common_x,
                y_raw=y_proc,
                y_fit=y_fit,
                y_total_reconstructed=y_total_reconstructed,
                sample_name=sample_name,
                vertical_lines=vertical_lines,
                peak_detection_enabled=peak_detection_enabled,
                peak_height_threshold=peak_height_threshold,
                peak_distance_min=peak_distance_min,
                control_data_list=control_data_list
            )
            
            self.fit_validation_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"验证拟合时出错：{str(e)}")
            traceback.print_exc()
    
    def _preprocess_spectrum_for_classification(self, file_path):
        """
        对光谱进行预处理，用于分类验证
        预处理流程（统一顺序）：BE校正 -> AsLS基线校正 -> 归一化 -> 全局动态变换 -> 二次导数
        返回: (x_truncated, y_preprocessed) - 截断并预处理后的光谱
        """
        try:
            # 获取预处理参数（优先使用分类验证Tab中的参数，如果没有则使用主菜单参数）
            parent = self.parent_dialog if hasattr(self, 'parent_dialog') else None
            if not parent:
                return None, None
            
            # 检查是否有分类验证Tab的预处理参数
            use_classification_params = hasattr(self, 'classification_preprocess_enabled') and self.classification_preprocess_enabled
            
            skip = parent.skip_rows_spin.value()
            x_min_phys = parent._parse_optional_float(parent.x_min_phys_input.text())
            x_max_phys = parent._parse_optional_float(parent.x_max_phys_input.text())
            
            x_raw, y_raw = parent.read_data(file_path, skip, x_min_phys, x_max_phys)
            
            # 应用预处理（统一顺序）
            y_proc = y_raw.astype(float)
            
            # 1. QC 检查（如果启用）
            qc_check = self.classification_qc_check.isChecked() if use_classification_params and hasattr(self, 'classification_qc_check') else parent.qc_check.isChecked()
            qc_threshold = self.classification_qc_threshold_spin.value() if use_classification_params and hasattr(self, 'classification_qc_threshold_spin') else parent.qc_threshold_spin.value()
            if qc_check and np.max(y_proc) < qc_threshold:
                return None, None
            
            # 2. BE 校正（如果启用）
            be_check = self.classification_be_check.isChecked() if use_classification_params and hasattr(self, 'classification_be_check') else parent.be_check.isChecked()
            be_temp = self.classification_be_temp_spin.value() if use_classification_params and hasattr(self, 'classification_be_temp_spin') else parent.be_temp_spin.value()
            if be_check:
                y_proc = DataPreProcessor.apply_bose_einstein_correction(x_raw, y_proc, be_temp)
            
            # 3. 平滑（如果启用）
            smoothing_check = self.classification_smoothing_check.isChecked() if use_classification_params and hasattr(self, 'classification_smoothing_check') else parent.smoothing_check.isChecked()
            smoothing_window = self.classification_smoothing_window_spin.value() if use_classification_params and hasattr(self, 'classification_smoothing_window_spin') else parent.smoothing_window_spin.value()
            smoothing_poly = self.classification_smoothing_poly_spin.value() if use_classification_params and hasattr(self, 'classification_smoothing_poly_spin') else parent.smoothing_poly_spin.value()
            if smoothing_check:
                y_proc = DataPreProcessor.apply_smoothing(y_proc, smoothing_window, smoothing_poly)
            
            # 4. AsLS 基线校正（必须启用，这是分类验证的要求）
            baseline_als_check = self.classification_baseline_als_check.isChecked() if use_classification_params and hasattr(self, 'classification_baseline_als_check') else parent.baseline_als_check.isChecked()
            lam = self.classification_lam_spin.value() if use_classification_params and hasattr(self, 'classification_lam_spin') else parent.lam_spin.value()
            p = self.classification_p_spin.value() if use_classification_params and hasattr(self, 'classification_p_spin') else parent.p_spin.value()
            if baseline_als_check:
                b = DataPreProcessor.apply_baseline_als(y_proc, lam, p)
                y_proc = y_proc - b
                y_proc[y_proc < 0] = 0
            else:
                # 如果未启用，强制应用AsLS基线校正
                b = DataPreProcessor.apply_baseline_als(y_proc, lam if lam > 0 else 10000, p if p > 0 else 0.005)
                y_proc = y_proc - b
                y_proc[y_proc < 0] = 0
            
            # 5. 归一化（必须，这是分类验证的要求，使用面积归一化）
            y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
            
            # 6. 全局动态范围压缩（如果启用）- 在归一化之后
            global_transform_mode = self.classification_global_transform_combo.currentText() if use_classification_params and hasattr(self, 'classification_global_transform_combo') else parent.global_transform_combo.currentText()
            if global_transform_mode == '对数变换 (Log)':
                log_base_text = self.classification_global_log_base_combo.currentText() if use_classification_params and hasattr(self, 'classification_global_log_base_combo') else parent.global_log_base_combo.currentText()
                base = float(log_base_text) if log_base_text == '10' else np.e
                log_offset = self.classification_global_log_offset_spin.value() if use_classification_params and hasattr(self, 'classification_global_log_offset_spin') else parent.global_log_offset_spin.value()
                y_proc = DataPreProcessor.apply_log_transform(y_proc, base=base, offset=log_offset)
            elif global_transform_mode == '平方根变换 (Sqrt)':
                sqrt_offset = self.classification_global_sqrt_offset_spin.value() if use_classification_params and hasattr(self, 'classification_global_sqrt_offset_spin') else parent.global_sqrt_offset_spin.value()
                y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=sqrt_offset)
            
            # 7. 二次导数（如果启用）- 在全局动态变换之后
            derivative_check = self.classification_derivative_check.isChecked() if use_classification_params and hasattr(self, 'classification_derivative_check') else parent.derivative_check.isChecked()
            if derivative_check:
                d1 = np.gradient(y_proc, x_raw)
                y_proc = np.gradient(d1, x_raw)
            
            # 8. 整体Y轴偏移（预处理最后一步，在二次导数之后）
            global_y_offset = self.classification_global_y_offset_spin.value() if use_classification_params and hasattr(self, 'classification_global_y_offset_spin') else (parent.global_y_offset_spin.value() if hasattr(parent, 'global_y_offset_spin') else 0.0)
            y_proc = y_proc + global_y_offset
            
            # 9. 确保非负（最终检查）
            y_proc[y_proc < 0] = 0
            
            return x_raw, y_proc
            
        except Exception as e:
            print(f"预处理文件 {file_path} 时出错: {e}")
            traceback.print_exc()
            return None, None
    
    def _calculate_vip_scores(self, pls_model, X, y):
        """
        计算PLS-DA的VIP (Variable Importance in Projection) 分数
        """
        try:
            # 获取PLS模型的权重和载荷
            w = pls_model.x_weights_  # (n_features, n_components)
            p = pls_model.x_loadings_  # (n_features, n_components)
            q = pls_model.y_loadings_  # (n_outputs, n_components)
            
            # 计算每个组分的方差解释率
            T = pls_model.transform(X)  # (n_samples, n_components)
            explained_variance = np.var(T, axis=0)  # (n_components,)
            
            # 计算VIP分数
            n_features = X.shape[1]
            n_components = w.shape[1]
            
            vip_scores = np.zeros(n_features)
            
            for i in range(n_features):
                numerator = 0
                denominator = 0
                
                for j in range(n_components):
                    # VIP公式: VIP_i = sqrt(p * sum((w_ij^2 * SSY_j) / SSY_total))
                    w_ij = w[i, j]
                    p_ij = p[i, j]
                    q_j = q[0, j] if q.shape[0] == 1 else q[j, 0]
                    
                    # SSY_j = explained variance of component j
                    ssy_j = explained_variance[j] * (q_j ** 2)
                    
                    numerator += (w_ij ** 2) * ssy_j
                    denominator += ssy_j
                
                if denominator > 0:
                    vip_scores[i] = np.sqrt(n_features * numerator / denominator)
                else:
                    vip_scores[i] = 0
            
            return vip_scores
            
        except Exception as e:
            print(f"计算VIP分数时出错: {e}")
            traceback.print_exc()
            return None
    
    def _run_algorithm_validation(self, algo_name, model_instance, X_train, y_train, X_test):
        """
        运行指定算法的LOO-CV并计算所有性能指标。
        返回: (预测结果字典, 性能指标字典)
        """
        n_samples = X_train.shape[0]
        loo = LeaveOneOut()
        y_true_cv_all = []
        y_pred_cv_all = []
        y_proba_pos_cv_all = []
        
        # 1. LOO-CV 训练与预测
        for train_idx, val_idx in loo.split(X_train):
            X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
            
            try:
                # 处理需要reshape的模型
                if algo_name == 'PLS-DA':
                    # PLS-DA需要复制模型实例
                    n_comp = model_instance.n_components if hasattr(model_instance, 'n_components') else 2
                    model_cv = PLSCanonical(n_components=n_comp)
                    model_cv.fit(X_train_cv, y_train_cv.reshape(-1, 1))
                    y_pred_cv = model_cv.predict(X_val_cv)
                    y_proba = y_pred_cv.flatten()
                    y_proba = np.clip(y_proba, 0, 1)
                    y_pred = (y_proba > 0.5).astype(int)
                elif algo_name == 'PCA + LDA':
                    # PCA+LDA Pipeline需要复制
                    model_cv = clone(model_instance)
                    model_cv.fit(X_train_cv, y_train_cv)
                    y_pred = model_cv.predict(X_val_cv)
                    # PCA+LDA通常不支持predict_proba，使用decision_function（通过Pipeline调用）
                    try:
                        if hasattr(model_cv, 'decision_function'):
                            # 通过Pipeline调用decision_function，Pipeline会自动处理PCA转换
                            y_proba_cont = model_cv.decision_function(X_val_cv)
                            y_proba = 1 / (1 + np.exp(-y_proba_cont))  # sigmoid转换
                        else:
                            y_proba = y_pred.astype(float)
                    except:
                        # 如果decision_function失败，使用预测值作为概率
                        y_proba = y_pred.astype(float)
                else:
                    # 其他模型：尝试复制，如果失败则使用原模型类型创建新实例
                    try:
                        model_cv = clone(model_instance)
                    except:
                        # 如果clone失败，尝试使用get_params创建新实例
                        if hasattr(model_instance, 'get_params'):
                            model_cv = type(model_instance)(**model_instance.get_params())
                        else:
                            model_cv = type(model_instance)()
                    
                    model_cv.fit(X_train_cv, y_train_cv)
                    
                    # 预测概率（如果模型支持）
                    if hasattr(model_cv, 'predict_proba'):
                        y_proba = model_cv.predict_proba(X_val_cv)[:, 1]
                        y_pred = model_cv.predict(X_val_cv)
                    else:
                        y_pred = model_cv.predict(X_val_cv)
                        # 对于不支持predict_proba的模型，使用decision_function或默认值
                        if hasattr(model_cv, 'decision_function'):
                            y_proba_cont = model_cv.decision_function(X_val_cv)
                            y_proba = 1 / (1 + np.exp(-y_proba_cont))  # sigmoid转换
                        else:
                            y_proba = y_pred.astype(float)
                
                y_true_cv_all.extend(y_val_cv)
                y_pred_cv_all.extend(y_pred.flatten().astype(int))
                y_proba_pos_cv_all.extend(y_proba.flatten())
                
            except Exception as e:
                print(f"LOO-CV for {algo_name} failed: {e}")
                traceback.print_exc()
                continue
        
        if not y_true_cv_all:
            return None, None
        
        # 2. 性能指标计算
        cv_accuracy = accuracy_score(y_true_cv_all, y_pred_cv_all)
        
        metrics = {
            'accuracy': cv_accuracy,
            'precision': precision_score(y_true_cv_all, y_pred_cv_all, zero_division=0),
            'recall': recall_score(y_true_cv_all, y_pred_cv_all, zero_division=0),
            'f1_score': f1_score(y_true_cv_all, y_pred_cv_all, zero_division=0),
        }
        
        try:
            metrics['auc'] = roc_auc_score(y_true_cv_all, y_proba_pos_cv_all)
        except:
            metrics['auc'] = 0.5  # 默认值
        
        # 3. 最终模型训练与测试集预测
        if algo_name == 'PLS-DA':
            model_instance.fit(X_train, y_train.reshape(-1, 1))
            y_test_pred_cont = model_instance.predict(X_test)
            y_test_pred = (y_test_pred_cont.flatten() > 0.5).astype(int)
            y_test_proba_cont = y_test_pred_cont.flatten()
            y_test_proba_pos = np.clip(y_test_proba_cont, 0, 1)
            y_test_proba = np.column_stack([1 - y_test_proba_pos, y_test_proba_pos])
        elif algo_name == 'PCA + LDA':
            model_instance.fit(X_train, y_train)
            y_test_pred = model_instance.predict(X_test).flatten().astype(int)
            # PCA+LDA使用decision_function（通过Pipeline调用）
            try:
                if hasattr(model_instance, 'decision_function'):
                    # 通过Pipeline调用decision_function，Pipeline会自动处理PCA转换
                    y_test_proba_cont = model_instance.decision_function(X_test)
                    y_test_proba_pos = 1 / (1 + np.exp(-y_test_proba_cont))  # sigmoid转换
                else:
                    y_test_proba_pos = y_test_pred.astype(float)
            except:
                # 如果decision_function失败，使用预测值作为概率
                y_test_proba_pos = y_test_pred.astype(float)
            y_test_proba = np.column_stack([1 - y_test_proba_pos, y_test_proba_pos])
        else:
            model_instance.fit(X_train, y_train)
            y_test_pred = model_instance.predict(X_test).flatten().astype(int)
            
            if hasattr(model_instance, 'predict_proba'):
                y_test_proba = model_instance.predict_proba(X_test)
            else:
                # 对于不支持predict_proba的模型，使用decision_function或默认值
                if hasattr(model_instance, 'decision_function'):
                    y_test_proba_cont = model_instance.decision_function(X_test)
                    y_test_proba_pos = 1 / (1 + np.exp(-y_test_proba_cont))  # sigmoid转换
                else:
                    y_test_proba_pos = y_test_pred.astype(float)
                y_test_proba = np.column_stack([1 - y_test_proba_pos, y_test_proba_pos])
        
        return {
            'cv_accuracy': cv_accuracy,
            'predictions': y_test_pred,
            'probabilities': y_test_proba,
            'model': model_instance,
        }, metrics
    
    def run_classification_validation(self):
        """运行分类验证"""
        try:
            # 检查训练集
            training_mineral_files = []
            for i in range(self.training_mineral_list.count()):
                item = self.training_mineral_list.item(i)
                training_mineral_files.append(item.data(Qt.ItemDataRole.UserRole))
            
            training_organic_files = []
            for i in range(self.training_organic_list.count()):
                item = self.training_organic_list.item(i)
                training_organic_files.append(item.data(Qt.ItemDataRole.UserRole))
            
            if not training_mineral_files:
                QMessageBox.warning(self, "警告", "请至少添加一个'Mineral Only'训练样本。")
                return
            
            if not training_organic_files:
                QMessageBox.warning(self, "警告", "请至少添加一个'Organic High Concentration'训练样本。")
                return
            
            # 检查测试集（低浓度样本）
            test_files = []
            for i in range(self.samples_list.count()):
                item = self.samples_list.item(i)
                test_files.append(item.data(Qt.ItemDataRole.UserRole))
            
            if not test_files:
                QMessageBox.warning(self, "警告", "请至少添加一个待测样品作为测试集。")
                return
            
            # 预处理训练集
            X_train_list = []
            y_train_list = []
            common_x_train = None
            
            # Mineral Only (Label 0)
            for file_path in training_mineral_files:
                x, y = self._preprocess_spectrum_for_classification(file_path)
                if x is not None and y is not None:
                    if common_x_train is None:
                        common_x_train = x
                    # 如果x长度不一致，进行插值
                    if len(x) != len(common_x_train):
                        from scipy.interpolate import interp1d
                        f_interp = interp1d(x, y, kind='linear', fill_value=0, bounds_error=False)
                        y = f_interp(common_x_train)
                    X_train_list.append(y)
                    y_train_list.append(0)
            
            # Organic High Concentration (Label 1)
            for file_path in training_organic_files:
                x, y = self._preprocess_spectrum_for_classification(file_path)
                if x is not None and y is not None:
                    if common_x_train is None:
                        common_x_train = x
                    # 如果x长度不一致，进行插值
                    if len(x) != len(common_x_train):
                        from scipy.interpolate import interp1d
                        f_interp = interp1d(x, y, kind='linear', fill_value=0, bounds_error=False)
                        y = f_interp(common_x_train)
                    X_train_list.append(y)
                    y_train_list.append(1)
            
            if not X_train_list:
                QMessageBox.warning(self, "警告", "训练集预处理失败，没有有效数据。")
                return
            
            X_train = np.array(X_train_list)
            y_train = np.array(y_train_list)
            
            # 预处理测试集
            X_test_list = []
            test_labels = []
            common_x_test = None
            
            for file_path in test_files:
                x, y = self._preprocess_spectrum_for_classification(file_path)
                if x is not None and y is not None:
                    if common_x_test is None:
                        common_x_test = x
                    # 如果x长度不一致，进行插值
                    if len(x) != len(common_x_test):
                        from scipy.interpolate import interp1d
                        f_interp = interp1d(x, y, kind='linear', fill_value=0, bounds_error=False)
                        y = f_interp(common_x_test)
                    X_test_list.append(y)
                    test_labels.append(os.path.basename(file_path))
            
            if not X_test_list:
                QMessageBox.warning(self, "警告", "测试集预处理失败，没有有效数据。")
                return
            
            X_test = np.array(X_test_list)
            
            # 确保训练集和测试集使用相同的波数轴（插值到共同范围）
            if common_x_train is not None and common_x_test is not None:
                # 找到共同的范围
                x_min = max(np.min(common_x_train), np.min(common_x_test))
                x_max = min(np.max(common_x_train), np.max(common_x_test))
                # 创建统一的波数轴（500-3200 cm^-1）
                common_x = np.linspace(500, 3200, min(len(common_x_train), len(common_x_test)))
                
                # 对训练集和测试集进行插值
                from scipy.interpolate import interp1d
                X_train_interp = []
                for i in range(X_train.shape[0]):
                    f_interp = interp1d(common_x_train, X_train[i], kind='linear', fill_value=0, bounds_error=False)
                    X_train_interp.append(f_interp(common_x))
                X_train = np.array(X_train_interp)
                
                X_test_interp = []
                for i in range(X_test.shape[0]):
                    f_interp = interp1d(common_x_test, X_test[i], kind='linear', fill_value=0, bounds_error=False)
                    X_test_interp.append(f_interp(common_x))
                X_test = np.array(X_test_interp)
                
                common_x_final = common_x
            else:
                common_x_final = common_x_train if common_x_train is not None else common_x_test
            
            # 选择算法
            algorithm_selection = self.classification_algorithm_combo.currentText()
            
            # 获取算法参数（从分类验证Tab中的控件读取）
            def get_algo_params():
                """从分类验证Tab的控件读取算法参数"""
                # 解析有机物敏感区字符串
                organic_ranges_str = self.classification_obs_organic_ranges_input.text() if hasattr(self, 'classification_obs_organic_ranges_input') else "2800-3050, 1600-1750"
                organic_ranges = []
                try:
                    for range_str in organic_ranges_str.split(','):
                        range_str = range_str.strip()
                        if '-' in range_str:
                            start, end = range_str.split('-')
                            organic_ranges.append((float(start.strip()), float(end.strip())))
                except:
                    organic_ranges = [(2800, 3050), (1600, 1750)]  # 默认值
                
                return {
                    'svc_kernel': self.svc_kernel_combo.currentText() if hasattr(self, 'svc_kernel_combo') else 'rbf',
                    'svc_c': self.svc_c_spin.value() if hasattr(self, 'svc_c_spin') else 1.0,
                    'svc_gamma': self.svc_gamma_combo.currentText() if hasattr(self, 'svc_gamma_combo') else 'scale',
                    'plsda_ncomp': self.plsda_ncomp_spin.value() if hasattr(self, 'plsda_ncomp_spin') else 0,
                    'lr_c': self.lr_c_spin.value() if hasattr(self, 'lr_c_spin') else 1.0,
                    'lr_solver': self.lr_solver_combo.currentText() if hasattr(self, 'lr_solver_combo') else 'lbfgs',
                    'knn_n_neighbors': self.knn_n_neighbors_spin.value() if hasattr(self, 'knn_n_neighbors_spin') else 5,
                    'knn_weights': self.knn_weights_combo.currentText() if hasattr(self, 'knn_weights_combo') else 'uniform',
                    'rf_n_estimators': self.rf_n_estimators_spin.value() if hasattr(self, 'rf_n_estimators_spin') else 100,
                    'rf_max_depth': self.rf_max_depth_spin.value() if hasattr(self, 'rf_max_depth_spin') else 0,
                    'pcalda_ncomp': self.pcalda_ncomp_spin.value() if hasattr(self, 'pcalda_ncomp_spin') else 0,
                    'adaboost_n_estimators': self.adaboost_n_estimators_spin.value() if hasattr(self, 'adaboost_n_estimators_spin') else 50,
                    'adaboost_learning_rate': self.adaboost_learning_rate_spin.value() if hasattr(self, 'adaboost_learning_rate_spin') else 1.0,
                    'use_standardscaler': self.classification_standardscaler_check.isChecked() if hasattr(self, 'classification_standardscaler_check') else True,  # 核心新增：StandardScaler选项
                    'use_adaptive_obs': self.classification_adaptive_obs_check.isChecked() if hasattr(self, 'classification_adaptive_obs_check') else False,  # 核心新增：Adaptive OBS选项
                    'obs_n_components': self.classification_obs_n_components_spin.value() if hasattr(self, 'classification_obs_n_components_spin') else 5,
                    'obs_organic_ranges': organic_ranges,
                }
            
            params = get_algo_params()
            
            # 处理gamma参数（字符串转数值）
            svc_gamma_val = params['svc_gamma']
            if svc_gamma_val not in ['scale', 'auto']:
                try:
                    svc_gamma_val = float(svc_gamma_val)
                except:
                    svc_gamma_val = 'scale'
            
            # 如果启用 Adaptive OBS，先对数据进行背景抑制
            use_adaptive_obs = params['use_adaptive_obs']
            obs_filter = None
            if use_adaptive_obs:
                # 创建 AdaptiveMineralFilter 实例
                obs_filter = AdaptiveMineralFilter(
                    n_components=params['obs_n_components'],
                    contamination=0.1,
                    organic_ranges=params['obs_organic_ranges']
                )
                # 在训练集上拟合（只使用 Mineral Only 样本）
                mineral_indices = y_train == 0
                if np.any(mineral_indices):
                    obs_filter.fit(X_train[mineral_indices], wavenumbers=common_x_final)
                    # 对训练集和测试集都应用背景抑制
                    X_train = obs_filter.transform(X_train)
                    X_test = obs_filter.transform(X_test)
            
            # 定义所有算法实例（使用参数控件中的值）
            # 核心修改：所有非PLS算法都封装在 Pipeline 中，以确保标准化在CV内部进行。
            use_scaler = params['use_standardscaler']
            
            # 辅助函数：构建 Pipeline，如果启用 Adaptive OBS 则在最前面添加
            def build_pipeline_with_obs(steps):
                """构建 Pipeline，如果启用 Adaptive OBS 则在最前面添加"""
                if use_adaptive_obs and obs_filter is not None:
                    # 注意：obs_filter 已经在外部拟合和转换过了，这里不需要再添加
                    # 因为 AdaptiveMineralFilter 不支持在 Pipeline 中传递 wavenumbers
                    # 所以我们在外部处理
                    pass
                return Pipeline(steps) if len(steps) > 1 else steps[0][1]
            
            all_algorithms = {
                'SVC': Pipeline([
                    ('scaler', StandardScaler()),  # 核心新增：在CV内部进行标准化
                    ('svc', SVC(kernel=params['svc_kernel'], C=params['svc_c'], 
                              gamma=svc_gamma_val, probability=True, random_state=42))
                ]) if use_scaler else SVC(kernel=params['svc_kernel'], C=params['svc_c'], 
                                         gamma=svc_gamma_val, probability=True, random_state=42),
                'PLS-DA': PLSCanonical(n_components=params['plsda_ncomp'] if params['plsda_ncomp'] > 0 else 2),  # PLS-DA不使用外部Scaler
                'Logistic Regression (LR)': Pipeline([
                    ('scaler', StandardScaler()),  # 核心新增
                    ('lr', LogisticRegression(C=params['lr_c'], solver=params['lr_solver'],
                                             max_iter=500, random_state=42))
                ]) if use_scaler else LogisticRegression(C=params['lr_c'], solver=params['lr_solver'],
                                                        max_iter=500, random_state=42),
                'k-Nearest Neighbors (k-NN)': Pipeline([
                    ('scaler', StandardScaler()),  # 核心新增
                    ('knn', KNeighborsClassifier(n_neighbors=params['knn_n_neighbors'],
                                                weights=params['knn_weights']))
                ]) if use_scaler else KNeighborsClassifier(n_neighbors=params['knn_n_neighbors'],
                                                           weights=params['knn_weights']),
                'Random Forest (RF)': Pipeline([
                    ('scaler', StandardScaler()),  # 核心新增
                    ('rf', RandomForestClassifier(n_estimators=params['rf_n_estimators'],
                                                 max_depth=params['rf_max_depth'] if params['rf_max_depth'] > 0 else None,
                                                 random_state=42))
                ]) if use_scaler else RandomForestClassifier(n_estimators=params['rf_n_estimators'],
                                                            max_depth=params['rf_max_depth'] if params['rf_max_depth'] > 0 else None,
                                                            random_state=42),
                # PCA + LDA 已经是 Pipeline，在最前面添加 Scaler
                'PCA + LDA': Pipeline([
                    ('scaler', StandardScaler()),  # 核心新增
                    ('pca', PCA(n_components=params['pcalda_ncomp'] if params['pcalda_ncomp'] > 0 else 2)), 
                    ('lda', LDA())
                ]) if use_scaler else Pipeline([('pca', PCA(n_components=params['pcalda_ncomp'] if params['pcalda_ncomp'] > 0 else 2)), 
                                               ('lda', LDA())]),  # n_components 在下面优化
                'AdaBoost': Pipeline([
                    ('scaler', StandardScaler()),  # 核心新增
                    ('ada', AdaBoostClassifier(n_estimators=params['adaboost_n_estimators'],
                                              learning_rate=params['adaboost_learning_rate'],
                                              random_state=42))
                ]) if use_scaler else AdaBoostClassifier(n_estimators=params['adaboost_n_estimators'],
                                                        learning_rate=params['adaboost_learning_rate'],
                                                        random_state=42),
            }
            
            # 根据用户选择确定要运行的算法
            algorithms_to_run = list(all_algorithms.keys()) if algorithm_selection == 'All' else [algorithm_selection]
            
            results = {}  # 存储预测结果和模型
            summary_metrics = {}  # 存储所有算法的综合指标（用于对比图）
            
            for algo_name in algorithms_to_run:
                model = all_algorithms[algo_name]
                
                # --- 优化 PLS-DA 组件数 ---
                if algo_name == 'PLS-DA':
                    # 如果用户指定了成分数（>0），使用用户指定的值；否则自动优化
                    if params['plsda_ncomp'] > 0:
                        best_n_components = params['plsda_ncomp']
                    else:
                        best_n_components = 2
                        best_cv_score = 0
                        
                        for n_comp in range(1, min(10, X_train.shape[0], X_train.shape[1] + 1)):
                            loo = LeaveOneOut()
                            cv_scores = []
                            for train_idx, val_idx in loo.split(X_train):
                                X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                                y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                                pls_model_cv = PLSCanonical(n_components=n_comp)
                                try:
                                    pls_model_cv.fit(X_train_cv, y_train_cv.reshape(-1, 1))
                                    y_pred_cv = pls_model_cv.predict(X_val_cv)
                                    y_pred_cv_binary = (y_pred_cv.flatten() > 0.5).astype(int)
                                    cv_scores.append(accuracy_score([y_val_cv], [y_pred_cv_binary]))
                                except:
                                    cv_scores.append(0)
                            
                            avg_score = np.mean(cv_scores) if cv_scores else 0
                            if avg_score > best_cv_score:
                                best_cv_score = avg_score
                                best_n_components = n_comp
                    
                    model = PLSCanonical(n_components=best_n_components)
                
                # --- 优化 PCA + LDA 组件数 ---
                elif algo_name == 'PCA + LDA':
                    # 如果用户指定了成分数（>0），使用用户指定的值；否则自动优化
                    if params['pcalda_ncomp'] > 0:
                        best_pca_comp = params['pcalda_ncomp']
                    else:
                        best_pca_comp = 2
                        best_cv_score = 0
                        
                        for n_comp in range(1, min(10, X_train.shape[0], X_train.shape[1] + 1)):
                            loo = LeaveOneOut()
                            cv_scores = []
                            for train_idx, val_idx in loo.split(X_train):
                                X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                                y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                                # 核心修正：PCA+LDA模型现在包含 StandardScaler（如果启用）
                                if use_scaler:
                                    pca_lda_model_cv = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_comp)), ('lda', LDA())])
                                else:
                                    pca_lda_model_cv = Pipeline([('pca', PCA(n_components=n_comp)), ('lda', LDA())])
                                try:
                                    pca_lda_model_cv.fit(X_train_cv, y_train_cv)
                                    y_pred_cv = pca_lda_model_cv.predict(X_val_cv)
                                    cv_scores.append(accuracy_score([y_val_cv], [y_pred_cv]))
                                except:
                                    cv_scores.append(0)
                            
                            avg_score = np.mean(cv_scores) if cv_scores else 0
                            if avg_score > best_cv_score:
                                best_cv_score = avg_score
                                best_pca_comp = n_comp
                    
                    # 最终模型也必须包含 StandardScaler（如果启用）
                    if use_scaler:
                        model = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=best_pca_comp)), ('lda', LDA())])
                    else:
                        model = Pipeline([('pca', PCA(n_components=best_pca_comp)), ('lda', LDA())])
                
                # --- 运行验证 ---
                algo_results, algo_metrics = self._run_algorithm_validation(algo_name, model, X_train, y_train, X_test)
                
                if algo_results:
                    results[algo_name] = algo_results
                    summary_metrics[algo_name] = algo_metrics
                    
                    if algo_name == 'PLS-DA':
                        # 计算 PLS-DA 的 VIP 分数并存储
                        vip_scores = self._calculate_vip_scores(algo_results['model'], X_train, y_train.reshape(-1, 1))
                        results[algo_name]['vip_scores'] = vip_scores
                        results[algo_name]['n_components'] = best_n_components  # 存储组件数
                    elif algo_name == 'PCA + LDA':
                        results[algo_name]['n_components'] = best_pca_comp  # 存储组件数
            
            if not results:
                QMessageBox.warning(self, "警告", "所有算法训练失败。")
                return
            
            # 创建或更新分类结果窗口
            if self.classification_window is None or not self.classification_window.isVisible():
                self.classification_window = ClassificationResultWindow(self)
            
            # 如果启用了 Adaptive OBS，保存原始测试数据和 obs_filter
            X_test_original = None
            if use_adaptive_obs and obs_filter is not None:
                # 重新读取原始测试数据（未经过 OBS 处理）
                X_test_original_list = []
                for file_path in test_files:
                    x, y = self._preprocess_spectrum_for_classification(file_path)
                    if x is not None and y is not None:
                        if common_x_test is None:
                            common_x_test = x
                        if len(x) != len(common_x_test):
                            from scipy.interpolate import interp1d
                            f_interp = interp1d(x, y, kind='linear', fill_value=0, bounds_error=False)
                            y = f_interp(common_x_test)
                        X_test_original_list.append(y)
                if X_test_original_list:
                    X_test_original = np.array(X_test_original_list)
                    # 插值到共同波数轴
                    if common_x_test is not None and common_x_final is not None:
                        from scipy.interpolate import interp1d
                        X_test_original_interp = []
                        for i in range(X_test_original.shape[0]):
                            f_interp = interp1d(common_x_test, X_test_original[i], kind='linear', fill_value=0, bounds_error=False)
                            X_test_original_interp.append(f_interp(common_x_final))
                        X_test_original = np.array(X_test_original_interp)
            
            # 设置数据并显示 - 核心修改：传递 summary_metrics 和 Adaptive OBS 相关信息
            self.classification_window.set_data(
                results=results,
                test_labels=test_labels,
                wavenumbers=common_x_final,
                algorithm=algorithm_selection,  # 传递用户选择的算法（All 或单个）
                summary_metrics=summary_metrics,  # 核心新增参数
                obs_filter=obs_filter if use_adaptive_obs else None,  # 传递 obs_filter
                X_test_original=X_test_original if use_adaptive_obs else None  # 传递原始测试数据
            )
            
            self.classification_window.show()
            
            QMessageBox.information(self, "完成", "分类验证完成！结果已在独立窗口中显示。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分类验证时出错：{str(e)}")
            traceback.print_exc()
    


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
from scipy.optimize import curve_fit, nnls
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
from src.core.transformers import AutoencoderTransformer, NonNegativeTransformer, AdaptiveMineralFilter, TORCH_AVAILABLE
from sklearn.pipeline import Pipeline
from src.ui.widgets.custom_widgets import (
    CollapsibleGroupBox,
    SmartDoubleSpinBox,
    UnlimitedNumericInput,
)
from src.ui.canvas import MplCanvas
from src.ui.windows.nmf_window import NMFResultWindow
from src.ui.windows.plot_window import MplPlotWindow

# 统一隐藏 QDoubleSpinBox 尾随零（仍可输入到小数点后15位）
QDoubleSpinBox.textFromValue = SmartDoubleSpinBox.textFromValue


class ClassificationResultWindow(QDialog):
    """分类验证结果窗口 - 显示分类准确率、预测结果和VIP分数"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分类验证结果")
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        
        self.parent_dialog = parent
        
        # 样式配置
        self.style_config = PlotStyleConfig(self)
        self.style_params = self.style_config.load_style_params("ClassificationResultWindow")
        
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
        right_scroll.setMaximumWidth(400)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_scroll.setWidget(right_widget)
        
        # 结果表格显示（学术论文标准格式）
        results_group = CollapsibleGroupBox("分类结果", is_expanded=True)
        results_layout = QVBoxLayout()
        
        # 创建表格显示分类结果
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(['算法', 'Accuracy', 'F1-Score', 'AUC'])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setMaximumHeight(200)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
        """)
        results_layout.addWidget(self.results_table)
        
        # 测试集预测结果表格（显示所有算法的预测结果）
        self.prediction_table = QTableWidget()
        # 列：样本 + 7种算法的预测类别和概率 = 1 + 7*2 = 15列
        self.prediction_table.setColumnCount(15)
        header_labels = ['样本']
        algo_short_names = ['SVC', 'PLS-DA', 'LR', 'k-NN', 'RF', 'PCA+LDA', 'AdaBoost']
        for algo_name in algo_short_names:
            header_labels.append(f'{algo_name}_类别')
            header_labels.append(f'{algo_name}_概率')
        self.prediction_table.setHorizontalHeaderLabels(header_labels)
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.prediction_table.setMaximumHeight(400)
        self.prediction_table.setAlternatingRowColors(True)
        self.prediction_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
        """)
        results_layout.addWidget(QLabel("测试集预测结果:"))
        results_layout.addWidget(self.prediction_table)
        
        results_group.setContentLayout(results_layout)
        right_panel.addWidget(results_group)
        
        # 样式配置面板（参考NMFFitValidationWindow）
        style_group = CollapsibleGroupBox("样式配置（发表级设置）", is_expanded=False)
        style_layout = QFormLayout()
        
        # Figure/DPI
        self.fig_width_spin = QDoubleSpinBox()
        self.fig_width_spin.setRange(-999999999.0, 999999999.0)
        self.fig_width_spin.setDecimals(15)
        self.fig_width_spin.setValue(self.style_params.get('fig_width', 12))
        self.fig_height_spin = QDoubleSpinBox()
        self.fig_height_spin.setRange(-999999999.0, 999999999.0)
        self.fig_height_spin.setDecimals(15)
        self.fig_height_spin.setValue(self.style_params.get('fig_height', 8))
        self.fig_dpi_spin = QSpinBox()
        self.fig_dpi_spin.setRange(-999999999, 999999999)
        self.fig_dpi_spin.setValue(self.style_params.get('fig_dpi', 300))
        style_layout.addRow("图尺寸 W/H:", self._create_h_layout([self.fig_width_spin, self.fig_height_spin]))
        style_layout.addRow("DPI:", self.fig_dpi_spin)
        
        # Font
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(['Times New Roman', 'Arial', 'SimHei'])
        self.font_family_combo.setCurrentText(self.style_params.get('font_family', 'Times New Roman'))
        
        self.axis_title_font_spin = QSpinBox()
        self.axis_title_font_spin.setRange(-999999999, 999999999)
        self.axis_title_font_spin.setValue(self.style_params.get('axis_title_fontsize', 20))
        self.tick_label_font_spin = QSpinBox()
        self.tick_label_font_spin.setRange(-999999999, 999999999)
        self.tick_label_font_spin.setValue(self.style_params.get('tick_label_fontsize', 16))
        self.legend_font_spin = QSpinBox()
        self.legend_font_spin.setRange(-999999999, 999999999)
        self.legend_font_spin.setValue(self.style_params.get('legend_fontsize', 10))
        self.title_font_spin = QSpinBox()
        self.title_font_spin.setRange(-999999999, 999999999)
        self.title_font_spin.setValue(self.style_params.get('title_fontsize', 18))
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
        self.line_width_spin.setValue(self.style_params.get('line_width', 1.2))
        style_layout.addRow("线宽:", self.line_width_spin)
        
        # Ticks
        self.tick_direction_combo = QComboBox()
        self.tick_direction_combo.addItems(['in', 'out'])
        self.tick_direction_combo.setCurrentText(self.style_params.get('tick_direction', 'in'))
        self.tick_len_major_spin = QSpinBox()
        self.tick_len_major_spin.setRange(-999999999, 999999999)
        self.tick_len_major_spin.setValue(self.style_params.get('tick_len_major', 8))
        self.tick_len_minor_spin = QSpinBox()
        self.tick_len_minor_spin.setRange(-999999999, 999999999)
        self.tick_len_minor_spin.setValue(self.style_params.get('tick_len_minor', 4))
        self.tick_width_spin = QDoubleSpinBox()
        self.tick_width_spin.setRange(-999999999.0, 999999999.0)
        self.tick_width_spin.setDecimals(15)
        self.tick_width_spin.setValue(self.style_params.get('tick_width', 1.0))
        style_layout.addRow("刻度方向 / 宽度:", self._create_h_layout([self.tick_direction_combo, self.tick_width_spin]))
        style_layout.addRow("刻度长度 (大/小):", self._create_h_layout([self.tick_len_major_spin, self.tick_len_minor_spin]))
        
        # Grid
        self.show_grid_check = QCheckBox("显示网格")
        self.show_grid_check.setChecked(self.style_params.get('show_grid', True))
        self.grid_alpha_spin = QDoubleSpinBox()
        self.grid_alpha_spin.setRange(-999999999.0, 999999999.0)
        self.grid_alpha_spin.setDecimals(15)
        self.grid_alpha_spin.setValue(self.style_params.get('grid_alpha', 0.2))
        style_layout.addRow(self._create_h_layout([self.show_grid_check, QLabel("透明度:"), self.grid_alpha_spin]))
        
        # Spines
        self.spine_top_check = QCheckBox("Top")
        self.spine_top_check.setChecked(self.style_params.get('spine_top', True))
        self.spine_bottom_check = QCheckBox("Bottom")
        self.spine_bottom_check.setChecked(self.style_params.get('spine_bottom', True))
        self.spine_left_check = QCheckBox("Left")
        self.spine_left_check.setChecked(self.style_params.get('spine_left', True))
        self.spine_right_check = QCheckBox("Right")
        self.spine_right_check.setChecked(self.style_params.get('spine_right', True))
        self.spine_width_spin = QDoubleSpinBox()
        self.spine_width_spin.setRange(-999999999.0, 999999999.0)
        self.spine_width_spin.setDecimals(15)
        self.spine_width_spin.setValue(self.style_params.get('spine_width', 2.0))
        style_layout.addRow("边框 (T/B/L/R):", self._create_h_layout([self.spine_top_check, self.spine_bottom_check, 
                                                                     self.spine_left_check, self.spine_right_check]))
        style_layout.addRow("边框线宽:", self.spine_width_spin)
        
        # Legend
        self.show_legend_check = QCheckBox("显示图例")
        self.show_legend_check.setChecked(self.style_params.get('show_legend', True))
        self.legend_frame_check = QCheckBox("图例边框")
        self.legend_frame_check.setChecked(self.style_params.get('legend_frame', True))
        self.legend_loc_combo = QComboBox()
        self.legend_loc_combo.addItems(['best', 'upper right', 'upper left', 'lower left', 'lower right', 
                                       'center left', 'center right', 'lower center', 'upper center', 'center'])
        self.legend_loc_combo.setCurrentText(self.style_params.get('legend_loc', 'best'))
        style_layout.addRow(self._create_h_layout([self.show_legend_check, self.legend_frame_check]))
        style_layout.addRow("图例位置:", self.legend_loc_combo)
        
        # 图例高级控制
        self.legend_ncol_spin = QSpinBox()
        self.legend_ncol_spin.setRange(-999999999, 999999999)
        self.legend_ncol_spin.setValue(self.style_params.get('legend_ncol', 1))
        self.legend_columnspacing_spin = QDoubleSpinBox()
        self.legend_columnspacing_spin.setRange(-999999999.0, 999999999.0)
        self.legend_columnspacing_spin.setDecimals(15)
        self.legend_columnspacing_spin.setValue(self.style_params.get('legend_columnspacing', 2.0))
        self.legend_labelspacing_spin = QDoubleSpinBox()
        self.legend_labelspacing_spin.setRange(-999999999.0, 999999999.0)
        self.legend_labelspacing_spin.setDecimals(15)
        self.legend_labelspacing_spin.setValue(self.style_params.get('legend_labelspacing', 0.5))
        self.legend_handlelength_spin = QDoubleSpinBox()
        self.legend_handlelength_spin.setRange(-999999999.0, 999999999.0)
        self.legend_handlelength_spin.setDecimals(15)
        self.legend_handlelength_spin.setValue(self.style_params.get('legend_handlelength', 2.0))
        style_layout.addRow("图例列数:", self.legend_ncol_spin)
        style_layout.addRow("图例列间距:", self.legend_columnspacing_spin)
        style_layout.addRow("图例标签间距:", self.legend_labelspacing_spin)
        style_layout.addRow("图例手柄长度:", self.legend_handlelength_spin)
        
        # 连接样式参数变化信号
        for widget in [self.fig_width_spin, self.fig_height_spin, self.fig_dpi_spin,
                       self.font_family_combo, self.axis_title_font_spin, self.tick_label_font_spin,
                       self.legend_font_spin, self.title_font_spin, self.xaxis_label_font_spin,
                       self.line_width_spin, self.tick_direction_combo, self.tick_len_major_spin, 
                       self.tick_len_minor_spin, self.tick_width_spin, self.show_grid_check, 
                       self.grid_alpha_spin, self.spine_top_check, self.spine_bottom_check, 
                       self.spine_left_check, self.spine_right_check, self.spine_width_spin, 
                       self.show_legend_check, self.legend_frame_check, self.legend_loc_combo, 
                       self.legend_ncol_spin, self.legend_columnspacing_spin, self.legend_labelspacing_spin, 
                       self.legend_handlelength_spin]:
            if isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self.update_plot)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self.update_plot)
            else:
                widget.valueChanged.connect(self.update_plot)
        
        style_group.setContentLayout(style_layout)
        right_panel.addWidget(style_group)
        
        # 更新图表按钮
        update_button = QPushButton("🔄 更新图表")
        update_button.clicked.connect(self.update_plot)
        right_panel.addWidget(update_button)
        
        # 可解释性分析按钮（仅在启用 Adaptive OBS 时显示）
        self.explainability_button = QPushButton("🧪 可解释性分析 (Explainability)")
        self.explainability_button.setStyleSheet("font-size: 11pt; padding: 8px; background-color: #9C27B0; color: white; font-weight: bold;")
        self.explainability_button.setToolTip("对选中的测试样本进行可解释性分析，显示原始光谱、拟合背景和提取的有机物残差")
        self.explainability_button.clicked.connect(self.show_explainability_analysis)
        self.explainability_button.setEnabled(False)  # 默认禁用，只有在启用 Adaptive OBS 时才启用
        right_panel.addWidget(self.explainability_button)
        
        # 光谱库匹配按钮和结果表格
        library_group = CollapsibleGroupBox("🔍 光谱库匹配", is_expanded=False)
        library_layout = QVBoxLayout()
        
        self.library_match_button = QPushButton("运行光谱库匹配")
        self.library_match_button.setStyleSheet("font-size: 11pt; padding: 8px; background-color: #FF9800; color: white; font-weight: bold;")
        self.library_match_button.setToolTip("提取残差谱并与标准库进行匹配，输出Top 3可能物质及置信度")
        self.library_match_button.clicked.connect(self.show_library_matching_analysis)
        self.library_match_button.setEnabled(False)  # 默认禁用
        library_layout.addWidget(self.library_match_button)
        
        # 匹配结果表格
        self.match_results_table = QTableWidget()
        self.match_results_table.setColumnCount(3)
        self.match_results_table.setHorizontalHeaderLabels(['排名', '物质名称', '相似度'])
        self.match_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.match_results_table.setMaximumHeight(200)
        self.match_results_table.setAlternatingRowColors(True)
        self.match_results_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
        """)
        library_layout.addWidget(self.match_results_table)
        
        library_group.setContentLayout(library_layout)
        right_panel.addWidget(library_group)
        
        right_panel.addStretch()
        
        content_layout.addLayout(left_panel, 3)
        content_layout.addWidget(right_scroll, 1)
        
        self.main_layout.addLayout(content_layout)
        
        # 存储数据
        self.results = None
        self.test_labels = None
        self.wavenumbers = None
        self.algorithm = None
        self.obs_filter = None  # Adaptive OBS 滤波器
        self.X_test_original = None  # 原始测试数据（未经过 OBS 处理）
        
        # 保存窗口位置和大小
        self.last_geometry = None
        self.resizeEvent = self._update_geometry_on_resize
        self.moveEvent = self._update_geometry_on_move
    
    def _create_h_layout(self, widgets):
        """创建水平布局的辅助方法"""
        h_layout = QHBoxLayout()
        for widget in widgets:
            h_layout.addWidget(widget)
        h_layout.addStretch(1)
        return h_layout
    
    def _update_geometry_on_move(self, event):
        """窗口移动时保存位置"""
        current_rect = self.geometry()
        self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        super().moveEvent(event)
    
    def _update_geometry_on_resize(self, event):
        """窗口大小改变时自动调整布局"""
        current_rect = self.geometry()
        self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
                self.canvas.figure.tight_layout()
            self.canvas.draw()
        except:
            pass
        super().resizeEvent(event)
    
    def set_data(self, results, test_labels, wavenumbers, algorithm, summary_metrics=None, obs_filter=None, X_test_original=None):
        """设置分类结果数据"""
        self.results = results
        self.test_labels = test_labels
        self.wavenumbers = wavenumbers
        self.algorithm = algorithm
        self.summary_metrics = summary_metrics if summary_metrics is not None else {}  # 核心新增：存储综合性能指标
        self.obs_filter = obs_filter  # Adaptive OBS 滤波器
        self.X_test_original = X_test_original  # 原始测试数据（未经过 OBS 处理）
        
        # 如果提供了 obs_filter，启用可解释性分析按钮
        if self.obs_filter is not None and self.X_test_original is not None:
            self.explainability_button.setEnabled(True)
        else:
            self.explainability_button.setEnabled(False)
        
        # 检查是否加载了标准库匹配器
        if self.parent_dialog and hasattr(self.parent_dialog, 'library_matcher') and self.parent_dialog.library_matcher is not None:
            self.library_match_button.setEnabled(True)
        else:
            self.library_match_button.setEnabled(False)
        
        # 更新结果表格（学术论文标准格式）
        if self.summary_metrics:
            self.results_table.setRowCount(len(self.summary_metrics))
            for row, (algo_name, metrics) in enumerate(self.summary_metrics.items()):
                # 缩短算法名字
                short_name = algo_name.replace('Logistic Regression (LR)', 'LR').replace('k-Nearest Neighbors (k-NN)', 'k-NN').replace('Random Forest (RF)', 'RF').replace('PCA + LDA', 'PCA+LDA')
                self.results_table.setItem(row, 0, QTableWidgetItem(short_name))
                self.results_table.setItem(row, 1, QTableWidgetItem(f"{metrics['accuracy']:.4f}"))
                self.results_table.setItem(row, 2, QTableWidgetItem(f"{metrics['f1_score']:.4f}"))
                self.results_table.setItem(row, 3, QTableWidgetItem(f"{metrics['auc']:.4f}"))
        
        # 更新测试集预测结果表格（显示所有算法的预测结果）
        if results:
            # 定义算法顺序（与绘图顺序一致）
            algo_order = ['SVC', 'PLS-DA', 'Logistic Regression (LR)', 
                         'k-Nearest Neighbors (k-NN)', 'Random Forest (RF)', 
                         'PCA + LDA', 'AdaBoost']
            
            self.prediction_table.setRowCount(len(test_labels))
            
            # 简化标签
            simplified_labels = self._simplify_sample_names(test_labels)
            
            for i, label in enumerate(test_labels):
                simplified_label = simplified_labels[i]
                self.prediction_table.setItem(i, 0, QTableWidgetItem(simplified_label))
                
                # 为每种算法填充预测类别和概率
                col_idx = 1
                for algo_name in algo_order:
                    if algo_name in results:
                        algo_results = results[algo_name]
                        pred = algo_results['predictions'][i]
                        proba = algo_results['probabilities'][i]
                        
                        # 处理概率数据
                        try:
                            if proba.ndim == 0:
                                proba_organic = float(proba)
                            elif proba.ndim == 1:
                                proba_organic = proba[1] if len(proba) > 1 else proba[0]
                            else:
                                proba_organic = proba[1] if proba.shape[0] > 1 else proba[0]
                        except:
                            proba_organic = float(pred)
                        
                        pred_class = "Organic Present" if pred == 1 else "Mineral Only"
                        
                        self.prediction_table.setItem(i, col_idx, QTableWidgetItem(pred_class))
                        self.prediction_table.setItem(i, col_idx + 1, QTableWidgetItem(f"{proba_organic:.4f}"))
                    else:
                        # 如果算法不存在，填充空值
                        self.prediction_table.setItem(i, col_idx, QTableWidgetItem("-"))
                        self.prediction_table.setItem(i, col_idx + 1, QTableWidgetItem("-"))
                    
                    col_idx += 2
        
        # 更新绘图
        self.update_plot()
    
    def _simplify_sample_names(self, labels):
        """简化样品名字：取前面的数字，然后同一种数字依次加上-1, -2, -3等"""
        import re
        simplified = []
        name_counts = {}  # 记录每个基础名字出现的次数
        
        for label in labels:
            # 提取文件名开头的数字（如果有）
            match = re.match(r'(\d+)', label)
            if match:
                base_num = match.group(1)
                if base_num not in name_counts:
                    name_counts[base_num] = 0
                name_counts[base_num] += 1
                count = name_counts[base_num]
                if count == 1:
                    simplified.append(base_num)
                else:
                    simplified.append(f"{base_num}-{count-1}")
            else:
                # 如果没有数字，尝试提取文件名（去掉扩展名）
                base_name = label.split('.')[0] if '.' in label else label
                if base_name not in name_counts:
                    name_counts[base_name] = 0
                name_counts[base_name] += 1
                count = name_counts[base_name]
                if count == 1:
                    simplified.append(base_name)
                else:
                    simplified.append(f"{base_name}-{count-1}")
        
        return simplified
    
    def show_library_matching_analysis(self):
        """显示光谱库匹配分析"""
        if self.parent_dialog is None or not hasattr(self.parent_dialog, 'library_matcher') or self.parent_dialog.library_matcher is None:
            QMessageBox.warning(self, "警告", "请先在主窗口中加载标准库。")
            return
        
        if self.obs_filter is None or self.X_test_original is None or self.wavenumbers is None:
            QMessageBox.warning(self, "警告", "光谱库匹配需要启用 Adaptive OBS 并存在测试数据。")
            return
        
        # 获取选中的测试样本索引
        selected_items = self.prediction_table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先在预测结果表格中选择一个测试样本。")
            return
        
        # 获取选中的行索引
        selected_row = selected_items[0].row()
        if selected_row >= len(self.test_labels) or selected_row >= self.X_test_original.shape[0]:
            QMessageBox.warning(self, "警告", "选中的样本索引无效。")
            return
        
        try:
            # 获取原始光谱
            x_spectrum = self.X_test_original[selected_row]
            
            # 使用 obs_filter 提取残差谱
            raw, background, residual = self.obs_filter.get_explanation(x_spectrum)
            
            # 确保残差谱非负
            residual = np.maximum(residual, 0)
            
            # 调用库匹配器
            library_matcher = self.parent_dialog.library_matcher
            matches = library_matcher.match(self.wavenumbers, residual, top_k=3)
            
            # 更新结果表格
            self.match_results_table.setRowCount(len(matches))
            for i, (name, similarity) in enumerate(matches):
                self.match_results_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
                self.match_results_table.setItem(i, 1, QTableWidgetItem(name))
                self.match_results_table.setItem(i, 2, QTableWidgetItem(f"{similarity:.4f}"))
            
            # 显示结果消息
            if matches:
                top_match = matches[0]
                QMessageBox.information(self, "匹配完成", 
                                      f"Top 3 匹配结果：\n\n"
                                      f"1. {top_match[0]} (相似度: {top_match[1]:.4f})\n"
                                      f"{f'2. {matches[1][0]} (相似度: {matches[1][1]:.4f})' if len(matches) > 1 else ''}\n"
                                      f"{f'3. {matches[2][0]} (相似度: {matches[2][1]:.4f})' if len(matches) > 2 else ''}")
            else:
                QMessageBox.warning(self, "警告", "未找到匹配结果。")
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"光谱库匹配失败：{str(e)}")
            traceback.print_exc()
    
    def show_explainability_analysis(self):
        """显示可解释性分析窗口"""
        if self.obs_filter is None or self.X_test_original is None or self.wavenumbers is None:
            QMessageBox.warning(self, "警告", "可解释性分析需要启用 Adaptive OBS 并存在测试数据。")
            return
        
        # 获取选中的测试样本索引
        selected_items = self.prediction_table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先在预测结果表格中选择一个测试样本。")
            return
        
        # 获取选中的行索引
        selected_row = selected_items[0].row()
        if selected_row >= len(self.test_labels) or selected_row >= self.X_test_original.shape[0]:
            QMessageBox.warning(self, "警告", "选中的样本索引无效。")
            return
        
        # 获取原始光谱
        x_spectrum = self.X_test_original[selected_row]
        
        # 使用 obs_filter 获取解释数据
        raw, background, residual = self.obs_filter.get_explanation(x_spectrum)
        
        # 创建绘图窗口
        plot_window = MplPlotWindow("可解释性分析", initial_geometry=(100, 100, 1200, 800), parent=self)
        fig = plot_window.canvas.figure
        fig.clear()
        
        # 获取样式参数
        style_params = self.get_style_params()
        
        # 设置字体
        font_family = style_params.get('font_family', 'Times New Roman')
        axis_title_fontsize = style_params.get('axis_title_fontsize', 20)
        tick_label_fontsize = style_params.get('tick_label_fontsize', 16)
        legend_fontsize = style_params.get('legend_fontsize', 10)
        line_width = style_params.get('line_width', 1.2)
        
        # 上图：原始光谱 + 拟合背景
        ax1 = fig.add_subplot(211)
        ax1.plot(self.wavenumbers, raw, 'k-', linewidth=line_width, label='Original Spectrum')
        ax1.plot(self.wavenumbers, background, 'r--', linewidth=line_width, alpha=0.8, label='Fitted Background')
        ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=axis_title_fontsize, fontfamily=font_family)
        ax1.set_ylabel('Intensity', fontsize=axis_title_fontsize, fontfamily=font_family)
        ax1.set_title('Original vs. Background', fontsize=style_params.get('title_fontsize', 18), fontfamily=font_family, fontweight='bold')
        ax1.tick_params(labelsize=tick_label_fontsize)
        ax1.legend(fontsize=legend_fontsize, frameon=True)
        ax1.grid(True, alpha=0.2)
        
        # 标记有机物敏感区
        for start, end in self.obs_filter.organic_ranges:
            ax1.axvspan(start, end, alpha=0.1, color='gray', label='Organic Sensitive Region' if start == self.obs_filter.organic_ranges[0][0] else '')
        
        # 下图：提取的有机物残差
        ax2 = fig.add_subplot(212)
        ax2.plot(self.wavenumbers, residual, 'g-', linewidth=line_width, label='Recovered Organic Signal')
        ax2.fill_between(self.wavenumbers, residual, 0, alpha=0.5, color='green')
        ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=axis_title_fontsize, fontfamily=font_family)
        ax2.set_ylabel('Residual Intensity', fontsize=axis_title_fontsize, fontfamily=font_family)
        ax2.set_title('Recovered Organic Signal', fontsize=style_params.get('title_fontsize', 18), fontfamily=font_family, fontweight='bold')
        ax2.tick_params(labelsize=tick_label_fontsize)
        ax2.legend(fontsize=legend_fontsize, frameon=True)
        ax2.grid(True, alpha=0.2)
        
        # 标记有机物敏感区
        for start, end in self.obs_filter.organic_ranges:
            ax2.axvspan(start, end, alpha=0.2, color='gray', label='Organic Sensitive Region' if start == self.obs_filter.organic_ranges[0][0] else '')
        
        # 设置刻度方向
        tick_direction = style_params.get('tick_direction', 'in')
        for ax in [ax1, ax2]:
            ax.tick_params(direction=tick_direction, width=style_params.get('tick_width', 1.0))
            # 设置边框
            for spine in ax.spines.values():
                spine.set_linewidth(style_params.get('spine_width', 2.0))
        
        # 调整布局
        fig.tight_layout()
        plot_window.canvas.draw()
        
        # 显示窗口
        plot_window.show()
    
    def update_plot(self):
        """更新绘图 - 使用3x3网格布局（9张图）"""
        if self.results is None or self.wavenumbers is None:
            return
        
        # 保存当前窗口位置
        if self.isVisible():
            current_rect = self.geometry()
            self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        
        fig = self.canvas.figure
        fig.clear()
        
        # 获取样式参数
        style_params = self.get_style_params()
        
        if len(self.results) == 0:
            return
        
        # 简化样品名字
        simplified_labels = self._simplify_sample_names(self.test_labels)
        
        # 判断是否选择"All"算法
        is_all_algorithms = self.algorithm == 'All' or len(self.results) > 1
        
        if is_all_algorithms:
            # 使用3x3布局：7种算法预测图 + VIP图 + 性能对比图 = 9张图
            from matplotlib.gridspec import GridSpec
            # 调整边界，确保所有内容都能显示，特别是顶部标题
            # 使用合理的初始间距，但允许工具栏调整（不要设置top=1.0，会阻止工具栏）
            gs = GridSpec(3, 3, figure=fig, hspace=0.75, wspace=0.5)
            
            # 定义算法顺序（固定顺序，确保布局一致）
            algo_order = ['SVC', 'PLS-DA', 'Logistic Regression (LR)', 
                         'k-Nearest Neighbors (k-NN)', 'Random Forest (RF)', 
                         'PCA + LDA', 'AdaBoost']
            
            # 获取X轴标签字体大小（可调节参数）
            xaxis_fontsize = style_params.get('xaxis_label_fontsize', 
                                             max(8, style_params['tick_label_fontsize'] - 4))
            
            plot_idx = 0
            
            # 第1-7个位置：绘制7种算法的预测结果图
            # 注意：PLS-DA在第1行第2列（i=1），AdaBoost在第3行第1列（i=6）
            for i, algo_name in enumerate(algo_order):
                if algo_name in self.results:
                    row = i // 3
                    col = i % 3
                    # 确保PLS-DA图正常显示
                    ax = fig.add_subplot(gs[row, col])
                    
                    algo_results = self.results[algo_name]
                    predictions = algo_results['predictions']
                    probabilities = algo_results['probabilities']
                    
                    # 检查probabilities的形状，确保正确处理PLS-DA的结果
                    try:
                        # 确保probabilities是numpy数组
                        if not isinstance(probabilities, np.ndarray):
                            probabilities = np.array(probabilities)
                        
                        # 处理不同维度的概率数据
                        if probabilities.ndim == 0:
                            # 标量 - 不应该出现，但处理一下
                            proba_organic = np.full(len(predictions), float(probabilities))
                        elif probabilities.ndim == 1:
                            # 一维数组 - 直接使用
                            proba_organic = probabilities.copy()
                        elif probabilities.ndim == 2:
                            # 二维数组 - 取第二列（Organic Present的概率）
                            if probabilities.shape[1] >= 2:
                                proba_organic = probabilities[:, 1].copy()
                            else:
                                proba_organic = probabilities[:, 0].copy()
                        else:
                            # 更高维度 - 展平后处理
                            proba_flat = probabilities.flatten()
                            if len(proba_flat) == len(predictions):
                                proba_organic = proba_flat
                            else:
                                # 如果展平后长度不匹配，尝试reshape
                                proba_organic = proba_flat[:len(predictions)]
                        
                        # 确保proba_organic是一维数组
                        if proba_organic.ndim == 0:
                            proba_organic = np.array([float(proba_organic)] * len(predictions))
                        elif proba_organic.ndim > 1:
                            proba_organic = proba_organic.flatten()
                        
                        # 确保长度匹配
                        if len(proba_organic) != len(predictions):
                            print(f"警告：{algo_name}的概率长度({len(proba_organic)})与预测长度({len(predictions)})不匹配")
                            if len(proba_organic) == 1:
                                proba_organic = np.repeat(proba_organic, len(predictions))
                            elif len(proba_organic) > len(predictions):
                                proba_organic = proba_organic[:len(predictions)]
                            else:
                                # 如果概率长度小于预测长度，用最后一个值填充
                                last_val = proba_organic[-1] if len(proba_organic) > 0 else 0.5
                                proba_organic = np.append(proba_organic, np.repeat(last_val, len(predictions) - len(proba_organic)))
                        
                        # 确保值在合理范围内
                        proba_organic = np.clip(proba_organic, 0.0, 1.0)
                                
                    except Exception as e:
                        print(f"处理{algo_name}的概率时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"probabilities类型: {type(probabilities)}")
                        print(f"probabilities形状: {probabilities.shape if hasattr(probabilities, 'shape') else 'N/A'}")
                        print(f"predictions形状: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
                        # 如果出错，使用predictions作为概率（0或1）
                        proba_organic = predictions.astype(float)
                    
                    x_pos = np.arange(len(simplified_labels))
                    colors = ['gray' if p == 0 else 'green' for p in predictions]
                    
                    bars = ax.bar(x_pos, proba_organic, color=colors, alpha=0.7, 
                                 edgecolor='black', linewidth=1)
                    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, 
                              label='Threshold (0.5)')
                    
                    ax.set_xlabel('', fontsize=style_params['axis_title_fontsize'] - 2)  # 删除"Sample"标签
                    ax.set_ylabel('Probability', fontsize=style_params['axis_title_fontsize'] - 2)
                    # 缩短标题，避免重叠
                    short_name = algo_name.replace('Logistic Regression (LR)', 'LR').replace('k-Nearest Neighbors (k-NN)', 'k-NN').replace('Random Forest (RF)', 'RF').replace('PCA + LDA', 'PCA+LDA')
                    # 根据行位置调整标题pad，第一行需要更多空间
                    title_pad = 10 if row == 0 else 8
                    ax.set_title(f'{short_name}\nAcc: {algo_results["cv_accuracy"]:.3f}', 
                               fontsize=style_params['title_fontsize'] - 2, fontweight='bold', pad=title_pad)
                    ax.set_xticks(x_pos)
                    # 使用可调节的X轴字体大小
                    ax.set_xticklabels(simplified_labels, rotation=45, ha='right', 
                                      fontsize=xaxis_fontsize)
                    ax.set_ylim([0, 1])
                    
                    # 应用样式
                    self.style_config.apply_style_to_axes(ax, style_params)
                    
                    if style_params.get('show_legend', True):
                        from matplotlib.font_manager import FontProperties
                        legend_font = FontProperties()
                        if style_params['font_family'] == 'SimHei':
                            legend_font.set_family('sans-serif')
                        else:
                            legend_font.set_family(style_params['font_family'])
                        legend_font.set_size(style_params['legend_fontsize'] - 2)
                        
                        ax.legend(loc=style_params.get('legend_loc', 'best'), 
                                 fontsize=style_params['legend_fontsize'] - 2,
                                 frameon=style_params.get('legend_frame', True),
                                 prop=legend_font,
                                 ncol=style_params.get('legend_ncol', 1),
                                 columnspacing=style_params.get('legend_columnspacing', 2.0),
                                 labelspacing=style_params.get('legend_labelspacing', 0.5),
                                 handlelength=style_params.get('legend_handlelength', 2.0))
                    plot_idx += 1
            
            # 第8个位置（第3行第2列）：VIP分数图（如果有PLS-DA结果）
            if 'PLS-DA' in self.results and self.results['PLS-DA'].get('vip_scores') is not None:
                ax_vip = fig.add_subplot(gs[2, 1])  # 第3行第2列
                vip_scores = self.results['PLS-DA']['vip_scores']
                
                ax_vip.plot(self.wavenumbers, vip_scores, linewidth=style_params['line_width'], color='blue')
                ax_vip.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='VIP = 1.0')
                ax_vip.set_xlabel('Wavenumber (cm⁻¹)', fontsize=style_params['axis_title_fontsize'] - 2)
                ax_vip.set_ylabel('VIP Score', fontsize=style_params['axis_title_fontsize'] - 2)
                ax_vip.set_title('PLS-DA VIP Scores', fontsize=style_params['title_fontsize'] - 2, 
                               fontweight='bold', pad=8)
                ax_vip.invert_xaxis()
                
                # 应用样式
                self.style_config.apply_style_to_axes(ax_vip, style_params)
                
                if style_params.get('show_legend', True):
                    from matplotlib.font_manager import FontProperties
                    legend_font = FontProperties()
                    if style_params['font_family'] == 'SimHei':
                        legend_font.set_family('sans-serif')
                    else:
                        legend_font.set_family(style_params['font_family'])
                    legend_font.set_size(style_params['legend_fontsize'] - 2)
                    
                    ax_vip.legend(loc=style_params.get('legend_loc', 'best'), 
                                 fontsize=style_params['legend_fontsize'] - 2,
                                 frameon=style_params.get('legend_frame', True),
                                 prop=legend_font,
                                 ncol=style_params.get('legend_ncol', 1),
                                 columnspacing=style_params.get('legend_columnspacing', 2.0),
                                 labelspacing=style_params.get('legend_labelspacing', 0.5),
                                 handlelength=style_params.get('legend_handlelength', 2.0))
            
            # 第9个位置（第3行第3列）：性能对比图
            if self.summary_metrics:
                ax_comparison = fig.add_subplot(gs[2, 2])  # 第3行第3列
                
                # 准备数据
                algo_names = list(self.summary_metrics.keys())
                metrics_to_plot = ['accuracy', 'f1_score', 'auc']
                metric_labels = ['Accuracy', 'F1-Score', 'AUC']
                
                x = np.arange(len(algo_names))
                width = 0.25  # 柱宽
                multiplier = 0
                
                colors_metrics = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
                
                for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
                    values = [self.summary_metrics[algo][metric] for algo in algo_names]
                    offset = width * multiplier
                    bars = ax_comparison.bar(x + offset, values, width, label=label, 
                                           color=colors_metrics[i], alpha=0.7, 
                                           edgecolor='black', linewidth=0.5)
                    multiplier += 1
                
                ax_comparison.set_xlabel('', fontsize=style_params['axis_title_fontsize'] - 2)  # 删除"Algorithm"标签
                ax_comparison.set_ylabel('Score', fontsize=style_params['axis_title_fontsize'] - 2)
                ax_comparison.set_title('Performance Comparison\n(LOO-CV)', 
                                       fontsize=style_params['title_fontsize'] - 2, fontweight='bold', pad=8)
                ax_comparison.set_xticks(x + width)
                # 缩短算法名字
                short_algo_names = [name.replace('Logistic Regression (LR)', 'LR')
                                  .replace('k-Nearest Neighbors (k-NN)', 'k-NN')
                                  .replace('Random Forest (RF)', 'RF')
                                  .replace('PCA + LDA', 'PCA+LDA') 
                                  for name in algo_names]
                # 使用可调节的X轴字体大小
                ax_comparison.set_xticklabels(short_algo_names, rotation=45, ha='right',
                                             fontsize=xaxis_fontsize)
                ax_comparison.set_ylim([0, 1.1])
                ax_comparison.legend(loc='upper left', fontsize=style_params['legend_fontsize'] - 2)
                ax_comparison.grid(True, alpha=0.3, axis='y')
                
                # 应用样式
                self.style_config.apply_style_to_axes(ax_comparison, style_params)
        else:
            # 单个算法：使用2x2布局（保持原有逻辑）
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # VIP图（如果有）
            if 'PLS-DA' in self.results and self.results['PLS-DA'].get('vip_scores') is not None:
                ax_vip = fig.add_subplot(gs[0, 0])
                vip_scores = self.results['PLS-DA']['vip_scores']
                
                ax_vip.plot(self.wavenumbers, vip_scores, linewidth=style_params['line_width'], color='blue')
                ax_vip.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='VIP = 1.0')
                ax_vip.set_xlabel('Wavenumber (cm⁻¹)', fontsize=style_params['axis_title_fontsize'])
                ax_vip.set_ylabel('VIP Score', fontsize=style_params['axis_title_fontsize'])
                ax_vip.set_title('PLS-DA VIP Scores', fontsize=style_params['title_fontsize'], fontweight='bold')
                ax_vip.invert_xaxis()
                
                self.style_config.apply_style_to_axes(ax_vip, style_params)
                
                if style_params.get('show_legend', True):
                    from matplotlib.font_manager import FontProperties
                    legend_font = FontProperties()
                    if style_params['font_family'] == 'SimHei':
                        legend_font.set_family('sans-serif')
                    else:
                        legend_font.set_family(style_params['font_family'])
                    legend_font.set_size(style_params['legend_fontsize'])
                    
                    ax_vip.legend(loc=style_params.get('legend_loc', 'best'), 
                                 fontsize=style_params['legend_fontsize'],
                                 frameon=style_params.get('legend_frame', True),
                                 prop=legend_font,
                                 ncol=style_params.get('legend_ncol', 1),
                                 columnspacing=style_params.get('legend_columnspacing', 2.0),
                                 labelspacing=style_params.get('legend_labelspacing', 0.5),
                                 handlelength=style_params.get('legend_handlelength', 2.0))
            
            # 预测结果图
            algo_list = list(self.results.items())
            for i, (algo_name, algo_results) in enumerate(algo_list[:2]):
                if i == 0:
                    ax = fig.add_subplot(gs[0, 1])
                elif i == 1:
                    ax = fig.add_subplot(gs[1, 0])
                else:
                    break
                
                predictions = algo_results['predictions']
                probabilities = algo_results['probabilities']
                proba_organic = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
                
                x_pos = np.arange(len(simplified_labels))
                colors = ['gray' if p == 0 else 'green' for p in predictions]
                
                bars = ax.bar(x_pos, proba_organic, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Decision Threshold (0.5)')
                
                ax.set_xlabel('', fontsize=style_params['axis_title_fontsize'])  # 删除"Test Sample"标签
                ax.set_ylabel('Probability (Organic Present)', fontsize=style_params['axis_title_fontsize'])
                ax.set_title(f'{algo_name} Predictions\n(LOO-CV Accuracy: {algo_results["cv_accuracy"]:.4f})', 
                            fontsize=style_params['title_fontsize'], fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(simplified_labels, rotation=45, ha='right')
                ax.set_ylim([0, 1])
                
                self.style_config.apply_style_to_axes(ax, style_params)
                
                if style_params.get('show_legend', True):
                    from matplotlib.font_manager import FontProperties
                    legend_font = FontProperties()
                    if style_params['font_family'] == 'SimHei':
                        legend_font.set_family('sans-serif')
                    else:
                        legend_font.set_family(style_params['font_family'])
                    legend_font.set_size(style_params['legend_fontsize'])
                    
                    ax.legend(loc=style_params.get('legend_loc', 'best'), 
                             fontsize=style_params['legend_fontsize'],
                             frameon=style_params.get('legend_frame', True),
                             prop=legend_font,
                             ncol=style_params.get('legend_ncol', 1),
                             columnspacing=style_params.get('legend_columnspacing', 2.0),
                             labelspacing=style_params.get('legend_labelspacing', 0.5),
                             handlelength=style_params.get('legend_handlelength', 2.0))
            
            # 性能对比图
            if self.summary_metrics:
                ax_comparison = fig.add_subplot(gs[1, 1])
                
                algo_names = list(self.summary_metrics.keys())
                metrics_to_plot = ['accuracy', 'f1_score', 'auc']
                metric_labels = ['Accuracy', 'F1-Score', 'AUC']
                
                x = np.arange(len(algo_names))
                width = 0.25
                multiplier = 0
                colors_metrics = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
                    values = [self.summary_metrics[algo][metric] for algo in algo_names]
                    offset = width * multiplier
                    bars = ax_comparison.bar(x + offset, values, width, label=label, 
                                           color=colors_metrics[i], alpha=0.7, 
                                           edgecolor='black', linewidth=0.5)
                    multiplier += 1
                
                ax_comparison.set_xlabel('Algorithm', fontsize=style_params['axis_title_fontsize'])
                ax_comparison.set_ylabel('Score', fontsize=style_params['axis_title_fontsize'])
                ax_comparison.set_title('Algorithm Performance Comparison\n(LOO-CV Metrics)', 
                                       fontsize=style_params['title_fontsize'], fontweight='bold')
                ax_comparison.set_xticks(x + width)
                ax_comparison.set_xticklabels(algo_names, rotation=45, ha='right')
                ax_comparison.set_ylim([0, 1.1])
                ax_comparison.legend(loc='upper left', fontsize=style_params['legend_fontsize'] - 2)
                ax_comparison.grid(True, alpha=0.3, axis='y')
                
                self.style_config.apply_style_to_axes(ax_comparison, style_params)
        
        # 调整布局 - GridSpec与tight_layout不完全兼容，使用subplots_adjust但允许工具栏覆盖
        # 注意：虽然subplots_adjust会设置初始布局，但工具栏的"Configure subplots"仍然可以调整
        # 工具栏会读取当前的subplot参数并允许用户修改
        try:
            if is_all_algorithms:
                # 对于3x3 GridSpec布局，使用subplots_adjust设置合理的初始布局
                # 这些参数会被工具栏读取，用户可以进一步调整
                fig.subplots_adjust(
                    left=0.06,      # 左侧边距
                    right=0.98,     # 右侧边距
                    top=0.99,       # 顶部边距（确保标题可见）
                    bottom=0.05,    # 底部边距
                    hspace=0.75,    # 垂直间距
                    wspace=0.5      # 水平间距
                )
            else:
                # 单个算法布局，使用tight_layout
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
                    fig.tight_layout()
        except Exception as e:
            # 如果失败，尝试使用tight_layout作为后备
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
                    fig.tight_layout()
            except:
                pass
        
        self.canvas.draw()
        
        # 恢复窗口位置
        if self.last_geometry:
            self.setGeometry(*self.last_geometry)
    
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
            'xaxis_label_fontsize': self.xaxis_label_font_spin.value(),  # 新增：X轴标签字体大小
            'line_width': self.line_width_spin.value(),
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
            'legend_ncol': self.legend_ncol_spin.value(),
            'legend_columnspacing': self.legend_columnspacing_spin.value(),
            'legend_labelspacing': self.legend_labelspacing_spin.value(),
            'legend_handlelength': self.legend_handlelength_spin.value(),
        }
    
    def closeEvent(self, event):
        """窗口关闭时保存设置和位置"""
        if self.isVisible():
            current_rect = self.geometry()
            self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        self.style_config.save_style_params("ClassificationResultWindow", self.get_style_params())
        event.accept()
    
    def showEvent(self, event):
        """窗口显示时恢复位置"""
        super().showEvent(event)
        if self.last_geometry:
            self.setGeometry(*self.last_geometry)


class SpectraConfigDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("光谱数据处理工作站（GTzhou组 - Pro版）")
        
        self.resize(1200, 900)
        # 设置最小尺寸，允许用户调整窗口大小
        self.setMinimumSize(800, 600)
        self.settings = QSettings("GTLab", "SpectraPro_v4") # 更新版本号
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(5) 
        self.main_layout.setContentsMargins(10, 10, 10, 10) 
        
        self.individual_control_widgets = {} 
        self.nmf_component_control_widgets = {}  # NMF组分的独立Y轴控制
        self.nmf_component_rename_widgets = {}  # NMF组分的图例重命名
        self.legend_rename_widgets = {}
        self.group_waterfall_control_widgets = {}  # 组瀑布图的独立堆叠位移控制
        self.last_fixed_H = None  # 存储上一次标准NMF运行得到的H矩阵，用于组分回归模式（预滤波空间）
        self.last_fixed_H_original = None  # 存储原始空间的H矩阵，用于绘图和验证
        self.last_pca_model = None  # 存储训练好的 PCA 模型实例
        self.last_common_x = None  # 存储NMF分析时的波数轴，用于定量分析
        self.nmf_target_component_index = 0  # 存储NMF目标组分索引，默认选择Component 1
        
        # 数据增强与光谱匹配相关
        self.library_matcher = None  # 存储 SpectralMatcher 实例
        self.library_folder_path = ""  # 存储标准库路径
        self.data_generator = None  # 存储 SyntheticDataGenerator 实例
        self.dae_window = None  # Deep Autoencoder 可视化窗口
        
        self.setup_ui()
        self.load_settings()
        
        # 连接所有样式参数的自动更新信号
        self._connect_all_style_update_signals()

        self.plot_windows = {} 
        self.nmf_window = None 
        
        # 存储当前激活的绘图窗口引用，用于叠加分析
        self.active_plot_window = None 
    
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
        # 前提检查
        if self.last_fixed_H is None:
            QMessageBox.warning(self, "错误", "请先运行标准NMF分析以获取固定的H矩阵。")
            return
        
        target_idx = self.get_nmf_target_component_index()
        if target_idx is None:
            QMessageBox.warning(self, "错误", "请在NMF结果窗口中指定目标组分索引。")
            return
        
        dialog = QuantitativeAnalysisDialog(self)
        dialog.exec()

    def _parse_optional_float(self, text):
        text = text.strip()
        if not text: return None
        try: return float(text)
        except ValueError: raise ValueError(f"输入 '{text}' 必须是数字。")

    def _create_h_layout(self, widgets):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(5)
        for wid in widgets: l.addWidget(wid)
        return w
    
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
        
        # 垂直线样式参数
        if hasattr(self, 'vertical_line_color_input'):
            self.vertical_line_color_input.textChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'vertical_line_width_spin'):
            self.vertical_line_width_spin.valueChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'vertical_line_style_combo'):
            self.vertical_line_style_combo.currentTextChanged.connect(self._on_style_param_changed)
        if hasattr(self, 'vertical_line_alpha_spin'):
            self.vertical_line_alpha_spin.valueChanged.connect(self._on_style_param_changed)
    
    def _on_style_param_changed(self):
        """样式参数变化时的回调函数（防抖）"""
        # 重置定时器，300ms后执行更新
        if hasattr(self, '_style_update_timer'):
            self._style_update_timer.stop()
            self._style_update_timer.start(300)
    
    def _on_file_color_changed(self):
        """文件颜色改变时的回调函数（自动更新图表）"""
        # 颜色改变时立即更新所有打开的绘图窗口
        self._on_style_param_changed()
    
    def _auto_update_all_plots(self):
        """自动更新所有打开的绘图窗口（仅更新样式，不重新读取数据）"""
        # 更新所有主绘图窗口
        for group_name, plot_window in self.plot_windows.items():
            if plot_window and plot_window.isVisible():
                try:
                    # 重新运行绘图逻辑（会使用当前参数）
                    self.run_plot_logic()
                    break  # 只更新一次，因为run_plot_logic会更新所有窗口
                except Exception as e:
                    print(f"自动更新绘图窗口 {group_name} 失败: {e}")
        
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
        if hasattr(self, 'nmf_window') and self.nmf_window and self.nmf_window.isVisible():
            try:
                # 如果有rerun_nmf_plot方法，使用它（不重新计算）
                if hasattr(self, 'rerun_nmf_plot'):
                    self.rerun_nmf_plot()
            except Exception as e:
                print(f"自动更新NMF窗口失败: {e}")

    # --- 核心：数据读取 (新增物理截断) ---
    def read_data(self, file_path, skip_rows, x_min_phys=None, x_max_phys=None):
        try:
            # 鲁棒读取
            try:
                df = pd.read_csv(file_path, header=None, skiprows=skip_rows, sep=None, engine='python')
            except:
                df = pd.read_csv(file_path, header=None, skiprows=skip_rows)
            
            if df.shape[1] < 2: raise ValueError("数据列不足2列")
            x = df.iloc[:, 0].values.astype(float)
            y = df.iloc[:, 1].values.astype(float)
            
            # 强制 X 降序 (Wavenumber 高->低)
            if len(x) > 1 and x[0] < x[-1]:
                x = x[::-1]
                y = y[::-1]
            
            # ⚠️ 物理截断 (Physical Truncation)
            mask = np.ones_like(x, dtype=bool)
            if x_min_phys is not None: mask &= (x >= x_min_phys)
            if x_max_phys is not None: mask &= (x <= x_max_phys)
            
            if not np.any(mask):
                raise ValueError(f"文件 {os.path.basename(file_path)} 在 X-Range [{x_min_phys}-{x_max_phys}] 内无数据。")

            x = x[mask]
            y = y[mask]
            
            return x, y
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            raise

    def parse_region_weights(self, weights_str, wavenumbers):
        """
        解析区域权重字符串并生成权重向量
        
        Args:
            weights_str: 权重字符串，格式如 "800-1000:0.1, 1000-1200:1.0"
            wavenumbers: 波数数组
        
        Returns:
            weight_vector: 权重向量，长度与 wavenumbers 相同
        """
        if not weights_str or not weights_str.strip():
            # 如果没有输入，返回全1向量
            return np.ones(len(wavenumbers))
        
        # 初始化权重向量为1.0
        weight_vector = np.ones(len(wavenumbers))
        
        try:
            # 解析字符串：800-1000:0.1, 1000-1200:1.0
            parts = weights_str.split(',')
            for part in parts:
                part = part.strip()
                if ':' in part:
                    range_str, weight_str = part.split(':', 1)
                    range_str = range_str.strip()
                    weight = float(weight_str.strip())
                    
                    # 解析范围：800-1000
                    if '-' in range_str:
                        min_w, max_w = map(float, range_str.split('-'))
                        # 找到该范围内的索引
                        mask = (wavenumbers >= min_w) & (wavenumbers <= max_w)
                        weight_vector[mask] = weight
        
        except Exception as e:
            print(f"警告：区域权重解析失败: {e}，使用默认权重（全1）")
            return np.ones(len(wavenumbers))
        
        return weight_vector

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
        # 使用现有的分组逻辑
        grouped_files = group_files_by_name(file_list, n_chars)
        
        averaged_data = {}
        common_x = None
        
        for group_key, files_in_group in grouped_files.items():
            group_spectra = []
            group_x_list = []
            
            # 读取组内所有文件
            for file_path in files_in_group:
                try:
                    x, y = self.read_data(file_path, skip_rows, x_min_phys, x_max_phys)
                    group_x_list.append(x)
                    group_spectra.append(y)
                except Exception as e:
                    print(f"警告：跳过文件 {os.path.basename(file_path)}: {e}")
                    continue
            
            if not group_spectra:
                continue
            
            # 检查所有光谱的X轴是否一致
            if common_x is None:
                common_x = group_x_list[0]
            else:
                # 如果X轴不一致，使用插值对齐到common_x
                aligned_spectra = []
                for i, (x_local, y_local) in enumerate(zip(group_x_list, group_spectra)):
                    if len(x_local) == len(common_x) and np.allclose(x_local, common_x):
                        aligned_spectra.append(y_local)
                    else:
                        # 需要插值对齐
                        from scipy.interpolate import interp1d
                        f_interp = interp1d(x_local, y_local, kind='linear', 
                                          fill_value=0, bounds_error=False)
                        y_aligned = f_interp(common_x)
                        aligned_spectra.append(y_aligned)
                group_spectra = aligned_spectra
            
            # 计算平均光谱
            group_matrix = np.array(group_spectra)
            y_averaged = np.mean(group_matrix, axis=0)
            
            averaged_data[group_key] = {
                'x': common_x,
                'y': y_averaged,
                'label': group_key,
                'files': files_in_group
            }
        
        return averaged_data, common_x

    # --- GUI 布局 ---
    def setup_ui(self):
        # --- 顶部全局控制 (文件 & 物理截断) ---
        top_bar = QFrame()
        top_bar.setFrameShape(QFrame.Shape.Panel)
        top_bar.setFrameShadow(QFrame.Shadow.Raised)
        top_bar_layout = QHBoxLayout(top_bar)
        
        # A. 文件夹选择
        folder_group = QGroupBox("数据文件夹")
        h_file = QHBoxLayout(folder_group)
        self.folder_input = QLineEdit()
        self.btn_browse = QPushButton("...")
        self.btn_browse.setFixedWidth(40)
        self.btn_browse.clicked.connect(self.browse_folder)
        h_file.addWidget(self.folder_input)
        h_file.addWidget(self.btn_browse)
        
        # B. 物理 X 范围
        x_range_group = QGroupBox("X 轴物理截断 (cm⁻¹)")
        x_range_layout = QHBoxLayout(x_range_group)
        x_range_layout.addWidget(QLabel("Min:"))
        self.x_min_phys_input = QLineEdit()
        self.x_min_phys_input.setPlaceholderText("例如: 600")
        x_range_layout.addWidget(self.x_min_phys_input)
        x_range_layout.addWidget(QLabel("Max:"))
        self.x_max_phys_input = QLineEdit()
        self.x_max_phys_input.setPlaceholderText("例如: 4000")
        x_range_layout.addWidget(self.x_max_phys_input)
        
        top_bar_layout.addWidget(folder_group)
        top_bar_layout.addWidget(x_range_group)
        self.main_layout.addWidget(top_bar)
        
        # --- 标签页布局 ---
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        self.setup_plotting_tab()
        self.setup_file_controls_tab()  # 新增：文件扫描与独立Y轴
        self.setup_peak_detection_tab()  # 新增：波峰检测
        self.setup_nmf_tab()
        self.setup_physics_tab()
        
        # --- 底部按钮区 (运行/导出/比较) ---
        btn_layout = QVBoxLayout()
        
        # 主要运行按钮行
        h_main_buttons = QHBoxLayout()
        self.run_button = QPushButton("运行绘图 (Plot Group Spectra)")
        self.run_button.setStyleSheet("font-size: 14pt; padding: 10px; background-color: #4CAF50; color: white; font-weight: bold;")
        self.run_button.clicked.connect(self.run_plot_logic)
        
        self.btn_run_nmf = QPushButton("运行 NMF 解混分析")
        self.btn_run_nmf.setStyleSheet("font-size: 14pt; padding: 10px; background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_run_nmf.clicked.connect(self.run_nmf_button_handler)
        
        self.btn_rerun_nmf_plot = QPushButton("🔄 重新绘制 NMF 图")
        self.btn_rerun_nmf_plot.setStyleSheet("font-size: 12pt; padding: 8px; background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_rerun_nmf_plot.clicked.connect(self.rerun_nmf_plot)
        self.btn_rerun_nmf_plot.setToolTip("使用当前设置重新绘制NMF图，不重新运行NMF分析")
        
        self.btn_quantitative = QPushButton("定量校准分析")
        self.btn_quantitative.setStyleSheet("font-size: 12pt; padding: 8px; background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_quantitative.clicked.connect(self.open_quantitative_dialog)
        
        h_main_buttons.addWidget(self.run_button)
        h_main_buttons.addWidget(self.btn_run_nmf)
        h_main_buttons.addWidget(self.btn_rerun_nmf_plot)
        h_main_buttons.addWidget(self.btn_quantitative)
        
        # 工具按钮行
        h_tools = QHBoxLayout()
        self.btn_export = QPushButton("导出预处理后数据")
        self.btn_export.clicked.connect(self.export_processed_data)
        self.btn_compare = QPushButton("绘制组间平均对比 (瀑布图)")
        self.btn_compare.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_compare.clicked.connect(self.run_group_average_waterfall)
        
        self.btn_2dcos = QPushButton("运行 2D-COS (组梯度分析)")
        self.btn_2dcos.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_2dcos.clicked.connect(self.run_2d_cos_analysis)
        self.btn_2dcos.setToolTip("2D-COS分析：基于浓度梯度数据解析重叠峰（如1100 vs 1107 cm⁻¹）")
        
        h_tools.addWidget(self.btn_compare)
        h_tools.addWidget(self.btn_2dcos)
        h_tools.addWidget(self.btn_export)
        
        btn_layout.addLayout(h_main_buttons)
        btn_layout.addLayout(h_tools)
        self.main_layout.addLayout(btn_layout)


    # --- Tab 1: 绘图设置 ---
    def setup_plotting_tab(self):
        tab1 = QWidget()
        grid_layout = QGridLayout(tab1)
        grid_layout.setSpacing(10)

        # --- 1. 左侧：数据/预处理/分组 ---
        left_vbox = QVBoxLayout()
        
        # 1.1 文件及分组配置
        file_group = CollapsibleGroupBox("1. 文件及分组配置", is_expanded=True)
        file_layout = QFormLayout()
        
        # FIX: 修正 QSpinBox 实例化错误
        self.n_chars_spin = QSpinBox()
        self.n_chars_spin.setRange(-999999999, 999999999)
        self.n_chars_spin.setValue(3)
        
        self.control_files_input = QTextEdit()
        self.control_files_input.setFixedHeight(40)
        self.control_files_input.setPlaceholderText("例如: His (自动识别.txt/.csv等后缀，多个文件用逗号或换行分隔)")
        self.groups_input = QLineEdit(placeholderText="例如: ant, mpt (留空则全选)")
        # 新增：分组平均复选框
        self.nmf_average_check = QCheckBox("启用分组平均 (NMF分析时对重复样本求平均)")
        self.nmf_average_check.setChecked(True)  # 默认启用
        self.nmf_average_check.setToolTip("启用后，NMF分析会将相同前缀的文件（如sample-1, sample-2）分组并计算平均光谱，提高信噪比")
        file_layout.addRow("分组前缀长度 (0=全名):", self.n_chars_spin)
        file_layout.addRow("指定组别 (可选):", self.groups_input)
        file_layout.addRow("对照文件 (优先绘制):", self.control_files_input)
        file_layout.addRow(self.nmf_average_check)
        file_group.setContentLayout(file_layout)
        left_vbox.addWidget(file_group)
        
        # 1.2 数据预处理
        preprocess_group = CollapsibleGroupBox("2. 数据预处理 (AsLS / QC / BE / SNV)", is_expanded=True)
        prep_layout = QFormLayout()
        
        self.skip_rows_spin = QSpinBox()
        self.skip_rows_spin.setRange(-999999999, 999999999)
        self.skip_rows_spin.setValue(2)
        prep_layout.addRow("跳过行数:", self.skip_rows_spin)
        
        self.qc_check = QCheckBox("启用 QC (剔除弱信号)")
        
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

        # 多点多项式基线校正（备选方案）
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
        
        # 新增：SVD 去噪选项
        self.svd_denoise_check = QCheckBox("启用 SVD 去噪 (物理去噪)")
        self.svd_denoise_check.setChecked(False)  # 默认不启用
        self.svd_components_spin = QSpinBox()
        self.svd_components_spin.setRange(-999999999, 999999999)
        self.svd_components_spin.setValue(5)
        self.svd_components_spin.setToolTip("保留的主成分数量，用于去除随机噪声")
        prep_layout.addRow(self.svd_denoise_check)
        prep_layout.addRow("SVD 主成分数:", self.svd_components_spin)
        
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
        
        preprocess_group.setContentLayout(prep_layout)
        left_vbox.addWidget(preprocess_group)
        
        grid_layout.addLayout(left_vbox, 0, 0, 1, 1) # 左侧布局

        # --- 2. 右侧：绘图样式 (出版质量控制) ---
        right_vbox = QVBoxLayout()
        
        # 2.1 绘图模式与标签
        plot_style_group = CollapsibleGroupBox("📈 4. 绘图模式与全局设置", is_expanded=True)
        style_layout = QFormLayout()
        
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(['Normal Overlay', 'Mean + Shadow', 'Waterfall (Stacked)'])
        style_layout.addRow("绘图模式:", self.plot_mode_combo)
        
        self.derivative_check = QCheckBox("二阶导数")
        self.x_axis_invert_check = QCheckBox("X轴翻转")
        self.show_y_val_check = QCheckBox("显示Y轴数值", checked=True)
        style_layout.addRow(self._create_h_layout([self.derivative_check, self.x_axis_invert_check, self.show_y_val_check]))
        
        # 整体Y轴偏移（预处理最后一步，在二次导数之后）
        self.global_y_offset_spin = QDoubleSpinBox()
        self.global_y_offset_spin.setRange(-999999999.0, 999999999.0)
        self.global_y_offset_spin.setDecimals(15)
        self.global_y_offset_spin.setValue(0.0)
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
        
        style_layout.addRow("堆叠偏移 / Y缩放:", self._create_h_layout([self.global_stack_offset_spin, self.global_y_scale_factor_spin]))
        
        self.xlabel_input = QLineEdit(r"Wavenumber ($\mathrm{cm^{-1}}$)")
        # FIX: 修正 Y 轴标题默认值，与原始代码保持一致
        self.ylabel_input = QLineEdit("Transmittance")
        self.main_title_input = QLineEdit(placeholderText="主图标题 (留空则显示组名)")
        
        # 主图标题控制：大小、间距、显示/隐藏
        self.main_title_font_spin = QSpinBox()
        self.main_title_font_spin.setRange(-999999999, 999999999)
        self.main_title_font_spin.setValue(20)  # 默认使用axis_title_fontsize的值
        
        self.main_title_pad_spin = QDoubleSpinBox()
        self.main_title_pad_spin.setRange(-999999999.0, 999999999.0)
        self.main_title_pad_spin.setDecimals(15)
        self.main_title_pad_spin.setValue(10.0)  # 默认值
        
        self.main_title_show_check = QCheckBox("显示主图标题")
        self.main_title_show_check.setChecked(True)  # 默认显示

        style_layout.addRow("X 标题:", self.xlabel_input)
        
        # X轴标题控制：大小、间距、显示/隐藏
        self.xlabel_font_spin = QSpinBox()
        self.xlabel_font_spin.setRange(-999999999, 999999999)
        self.xlabel_font_spin.setValue(20)  # 默认值（使用axis_title_fontsize）
        
        self.xlabel_pad_spin = QDoubleSpinBox()
        self.xlabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.xlabel_pad_spin.setDecimals(15)
        self.xlabel_pad_spin.setValue(10.0)  # 默认值
        
        self.xlabel_show_check = QCheckBox("显示X轴标题")
        self.xlabel_show_check.setChecked(True)  # 默认显示
        
        style_layout.addRow("X轴标题控制:", self._create_h_layout([self.xlabel_show_check, QLabel("大小:"), self.xlabel_font_spin, QLabel("间距:"), self.xlabel_pad_spin]))
        
        style_layout.addRow("Y 标题:", self.ylabel_input)
        
        # Y轴标题控制：大小、间距、显示/隐藏
        self.ylabel_font_spin = QSpinBox()
        self.ylabel_font_spin.setRange(-999999999, 999999999)
        self.ylabel_font_spin.setValue(20)  # 默认值（使用axis_title_fontsize）
        
        self.ylabel_pad_spin = QDoubleSpinBox()
        self.ylabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.ylabel_pad_spin.setDecimals(15)
        self.ylabel_pad_spin.setValue(10.0)  # 默认值
        
        self.ylabel_show_check = QCheckBox("显示Y轴标题")
        self.ylabel_show_check.setChecked(True)  # 默认显示
        
        style_layout.addRow("Y轴标题控制:", self._create_h_layout([self.ylabel_show_check, QLabel("大小:"), self.ylabel_font_spin, QLabel("间距:"), self.ylabel_pad_spin]))
        
        style_layout.addRow("主图标题:", self.main_title_input)
        style_layout.addRow("主图标题控制:", self._create_h_layout([self.main_title_show_check, QLabel("大小:"), self.main_title_font_spin, QLabel("间距:"), self.main_title_pad_spin]))
        
        # 浓度梯度图标题控制：大小、间距、显示/隐藏
        self.gradient_title_input = QLineEdit("Concentration Gradient (Group Averages)")
        self.gradient_title_font_spin = QSpinBox()
        self.gradient_title_font_spin.setRange(-999999999, 999999999)
        self.gradient_title_font_spin.setValue(22)  # 默认值（axis_title_fontsize + 2）
        
        self.gradient_title_pad_spin = QDoubleSpinBox()
        self.gradient_title_pad_spin.setRange(-999999999.0, 999999999.0)
        self.gradient_title_pad_spin.setDecimals(15)
        self.gradient_title_pad_spin.setValue(10.0)  # 默认值
        
        self.gradient_title_show_check = QCheckBox("显示浓度梯度图标题")
        self.gradient_title_show_check.setChecked(True)  # 默认显示
        
        style_layout.addRow("浓度梯度图标题:", self.gradient_title_input)
        style_layout.addRow("浓度梯度图标题控制:", self._create_h_layout([self.gradient_title_show_check, QLabel("大小:"), self.gradient_title_font_spin, QLabel("间距:"), self.gradient_title_pad_spin]))
        
        # 浓度梯度图X轴标题控制：大小、间距、显示/隐藏
        self.gradient_xlabel_font_spin = QSpinBox()
        self.gradient_xlabel_font_spin.setRange(-999999999, 999999999)
        self.gradient_xlabel_font_spin.setValue(20)  # 默认值
        
        self.gradient_xlabel_pad_spin = QDoubleSpinBox()
        self.gradient_xlabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.gradient_xlabel_pad_spin.setDecimals(15)
        self.gradient_xlabel_pad_spin.setValue(10.0)  # 默认值
        
        self.gradient_xlabel_show_check = QCheckBox("显示浓度梯度图X轴标题")
        self.gradient_xlabel_show_check.setChecked(True)  # 默认显示
        
        style_layout.addRow("浓度梯度图X轴标题控制:", self._create_h_layout([self.gradient_xlabel_show_check, QLabel("大小:"), self.gradient_xlabel_font_spin, QLabel("间距:"), self.gradient_xlabel_pad_spin]))
        
        # 浓度梯度图Y轴标题控制：大小、间距、显示/隐藏
        self.gradient_ylabel_font_spin = QSpinBox()
        self.gradient_ylabel_font_spin.setRange(-999999999, 999999999)
        self.gradient_ylabel_font_spin.setValue(20)  # 默认值
        
        self.gradient_ylabel_pad_spin = QDoubleSpinBox()
        self.gradient_ylabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.gradient_ylabel_pad_spin.setDecimals(15)
        self.gradient_ylabel_pad_spin.setValue(10.0)  # 默认值
        
        self.gradient_ylabel_show_check = QCheckBox("显示浓度梯度图Y轴标题")
        self.gradient_ylabel_show_check.setChecked(True)  # 默认显示
        
        style_layout.addRow("浓度梯度图Y轴标题控制:", self._create_h_layout([self.gradient_ylabel_show_check, QLabel("大小:"), self.gradient_ylabel_font_spin, QLabel("间距:"), self.gradient_ylabel_pad_spin]))
        
        # 瀑布图阴影控制 - 阴影颜色默认和线条颜色一样，只可调透明度
        self.waterfall_shadow_check = QCheckBox("显示阴影（标准差）")
        self.waterfall_shadow_check.setChecked(False)  # 默认不显示阴影
        
        self.waterfall_shadow_alpha_spin = QDoubleSpinBox()
        self.waterfall_shadow_alpha_spin.setRange(-999999999.0, 999999999.0)
        self.waterfall_shadow_alpha_spin.setDecimals(15)
        self.waterfall_shadow_alpha_spin.setValue(0.25)  # 默认值
        
        style_layout.addRow("瀑布图阴影控制:", self._create_h_layout([self.waterfall_shadow_check, QLabel("透明度:"), self.waterfall_shadow_alpha_spin]))

        plot_style_group.setContentLayout(style_layout)
        right_vbox.addWidget(plot_style_group)
        
        # 2.2 出版质量样式控制 (23个参数)
        pub_style_group = CollapsibleGroupBox("💎 5. 出版质量样式控制", is_expanded=False)
        pub_layout = QFormLayout()

        # Figure/DPI
        self.fig_width_spin = QDoubleSpinBox()
        self.fig_width_spin.setRange(-999999999.0, 999999999.0)
        self.fig_width_spin.setDecimals(15)
        self.fig_width_spin.setValue(10.0)
        
        self.fig_height_spin = QDoubleSpinBox()
        self.fig_height_spin.setRange(-999999999.0, 999999999.0)
        self.fig_height_spin.setDecimals(15)
        self.fig_height_spin.setValue(6.0)
        
        # FIX: 修正 QSpinBox 实例化错误
        self.fig_dpi_spin = QSpinBox()
        self.fig_dpi_spin.setRange(-999999999, 999999999)
        self.fig_dpi_spin.setValue(300)
        
        # Aspect Ratio (高度/宽度的比例，默认5:3即宽:高=5:3，所以高/宽=3/5=0.6)
        self.aspect_ratio_spin = QDoubleSpinBox()
        self.aspect_ratio_spin.setRange(-999999999.0, 999999999.0)
        self.aspect_ratio_spin.setDecimals(15)
        self.aspect_ratio_spin.setValue(0.6)  # 默认5:3（宽:高），即高/宽=3/5=0.6
        
        pub_layout.addRow("图尺寸 W/H:", self._create_h_layout([self.fig_width_spin, self.fig_height_spin]))
        pub_layout.addRow("DPI / 纵横比:", self._create_h_layout([self.fig_dpi_spin, self.aspect_ratio_spin]))

        # Font
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(['Times New Roman', 'Arial', 'SimHei'])
        
        # FIX: 修正字体默认值，与原始代码的 20/16 保持一致
        self.axis_title_font_spin = QSpinBox()
        self.axis_title_font_spin.setRange(-999999999, 999999999)
        self.axis_title_font_spin.setValue(20)
        
        self.tick_label_font_spin = QSpinBox()
        self.tick_label_font_spin.setRange(-999999999, 999999999)
        self.tick_label_font_spin.setValue(16)
        
        self.legend_font_spin = QSpinBox()
        self.legend_font_spin.setRange(-999999999, 999999999)
        self.legend_font_spin.setValue(10)
        
        # 图例字体大小同步：当legend_font_spin改变时，同步到legend_fontsize_spin
        def sync_legend_fontsize():
            if hasattr(self, 'legend_fontsize_spin'):
                self.legend_fontsize_spin.setValue(self.legend_font_spin.value())
        self.legend_font_spin.valueChanged.connect(sync_legend_fontsize)
        
        pub_layout.addRow("字体家族:", self.font_family_combo)
        pub_layout.addRow("字体大小 (轴/刻度/图例):", self._create_h_layout([self.axis_title_font_spin, self.tick_label_font_spin, self.legend_font_spin]))
        
        # Lines
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(-999999999.0, 999999999.0)
        self.line_width_spin.setDecimals(15)
        # 默认值使用 1.2 更接近原始代码的个体线宽
        self.line_width_spin.setValue(1.2) 
        
        self.line_style_combo = QComboBox()
        self.line_style_combo.addItems(['-', '--', ':', '-.'])
        pub_layout.addRow("线宽 / 线型:", self._create_h_layout([self.line_width_spin, self.line_style_combo]))
        
        # Ticks
        self.tick_direction_combo = QComboBox()
        self.tick_direction_combo.addItems(['in', 'out'])
        
        self.tick_len_major_spin = QSpinBox()
        self.tick_len_major_spin.setRange(-999999999, 999999999)
        self.tick_len_major_spin.setValue(8)
        
        self.tick_len_minor_spin = QSpinBox()
        self.tick_len_minor_spin.setRange(-999999999, 999999999)
        self.tick_len_minor_spin.setValue(4)
        
        self.tick_width_spin = QDoubleSpinBox()
        self.tick_width_spin.setRange(-999999999.0, 999999999.0)
        self.tick_width_spin.setDecimals(15)
        self.tick_width_spin.setValue(1.0)
        
        pub_layout.addRow("刻度方向 / 宽度:", self._create_h_layout([self.tick_direction_combo, self.tick_width_spin]))
        pub_layout.addRow("刻度长度 (大/小):", self._create_h_layout([self.tick_len_major_spin, self.tick_len_minor_spin]))
        
        # Grid/Shadow
        self.show_grid_check = QCheckBox("显示网格")
        
        # FIX: 修正 QDoubleSpinBox 实例化错误
        self.grid_alpha_spin = QDoubleSpinBox()
        self.grid_alpha_spin.setRange(-999999999.0, 999999999.0)
        self.grid_alpha_spin.setDecimals(15)
        self.grid_alpha_spin.setValue(0.2)
        
        self.shadow_alpha_spin = QDoubleSpinBox()
        self.shadow_alpha_spin.setRange(-999999999.0, 999999999.0)
        self.shadow_alpha_spin.setDecimals(15)
        self.shadow_alpha_spin.setValue(0.25)
        
        pub_layout.addRow(self._create_h_layout([self.show_grid_check, QLabel("网格 Alpha:"), self.grid_alpha_spin]))
        pub_layout.addRow("阴影 Alpha:", self.shadow_alpha_spin)

        # Axes Spines
        # FIX: 修正边框默认值，与原始代码的全部显示保持一致
        self.spine_top_check = QCheckBox("Top", checked=True)
        self.spine_bottom_check = QCheckBox("Bottom", checked=True)
        self.spine_left_check = QCheckBox("Left", checked=True)
        self.spine_right_check = QCheckBox("Right", checked=True)
        
        self.spine_width_spin = QDoubleSpinBox()
        self.spine_width_spin.setRange(-999999999.0, 999999999.0)
        self.spine_width_spin.setDecimals(15)
        self.spine_width_spin.setValue(2.0)
        
        pub_layout.addRow("边框 (T/B/L/R):", self._create_h_layout([self.spine_top_check, self.spine_bottom_check, self.spine_left_check, self.spine_right_check]))
        pub_layout.addRow("边框线宽:", self.spine_width_spin)
        
        # Legend
        self.show_legend_check = QCheckBox("显示图例", checked=True)
        self.legend_frame_check = QCheckBox("图例边框", checked=True)
        self.legend_loc_combo = QComboBox()
        self.legend_loc_combo.addItems(['best', 'upper right', 'upper left', 'lower left', 'lower right', 'center left', 'center right', 'lower center', 'upper center', 'center'])
        
        # 图例大小和间距控制
        self.legend_fontsize_spin = QSpinBox()
        self.legend_fontsize_spin.setRange(-999999999, 999999999)
        self.legend_fontsize_spin.setValue(10)  # 默认值，与legend_font_spin一致
        
        self.legend_column_spin = QSpinBox()
        self.legend_column_spin.setRange(-999999999, 999999999)
        self.legend_column_spin.setValue(1)  # 默认1列
        
        self.legend_columnspacing_spin = QDoubleSpinBox()
        self.legend_columnspacing_spin.setRange(-999999999.0, 999999999.0)
        self.legend_columnspacing_spin.setDecimals(15)
        self.legend_columnspacing_spin.setValue(2.0)  # 默认列间距
        
        self.legend_labelspacing_spin = QDoubleSpinBox()
        self.legend_labelspacing_spin.setRange(-999999999.0, 999999999.0)
        self.legend_labelspacing_spin.setDecimals(15)
        self.legend_labelspacing_spin.setValue(0.5)  # 默认标签间距
        
        self.legend_handlelength_spin = QDoubleSpinBox()
        self.legend_handlelength_spin.setRange(-999999999.0, 999999999.0)
        self.legend_handlelength_spin.setDecimals(15)
        self.legend_handlelength_spin.setValue(2.0)  # 默认句柄长度
        
        pub_layout.addRow(self._create_h_layout([self.show_legend_check, self.legend_frame_check]))
        pub_layout.addRow("图例位置:", self.legend_loc_combo)
        pub_layout.addRow("图例字体大小:", self.legend_fontsize_spin)
        pub_layout.addRow("图例列数:", self.legend_column_spin)
        pub_layout.addRow("图例列间距 / 标签间距:", self._create_h_layout([self.legend_columnspacing_spin, self.legend_labelspacing_spin]))
        pub_layout.addRow("图例句柄长度:", self.legend_handlelength_spin)
        
        pub_style_group.setContentLayout(pub_layout)
        right_vbox.addWidget(pub_style_group)
        
        right_vbox.addStretch(1) # 撑开
        grid_layout.addLayout(right_vbox, 0, 1, 1, 1) # 右侧布局
        self.tab_widget.addTab(tab1, "📊 绘图与预处理")
    
    # --- Tab 2: 文件扫描与独立Y轴 ---
    def setup_file_controls_tab(self):
        tab2 = QWidget()
        layout = QVBoxLayout(tab2)
        layout.setSpacing(10)
        
        # 1. 文件扫描与独立Y轴控制
        file_controls_group = CollapsibleGroupBox("📥 文件扫描与独立Y轴控制", is_expanded=True)
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
        nmf_controls_group = CollapsibleGroupBox("🔬 NMF组分独立Y轴控制和图例重命名", is_expanded=True)
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
        waterfall_controls_group = CollapsibleGroupBox("📊 组瀑布图独立堆叠位移控制", is_expanded=True)
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
        aug_lib_group = CollapsibleGroupBox("🔬 合成数据与标准库配置", is_expanded=True)
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
        
        aug_lib_group.setContentLayout(aug_lib_layout)
        layout.addWidget(aug_lib_group)
        
        layout.addStretch(1)
        self.tab_widget.addTab(tab2, "📥 文件扫描与独立Y轴")
    
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
        """加载标准库匹配器"""
        folder = self.library_folder_input.text()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "错误", "请先选择有效的标准库文件夹")
            return
        
        try:
            self.library_matcher = SpectralMatcher(folder)
            n_spectra = len(self.library_matcher.library_spectra)
            self.library_status_label.setText(f"状态: 已加载 {n_spectra} 条标准光谱")
            self.library_status_label.setStyleSheet("color: green; font-size: 9pt;")
            QMessageBox.information(self, "成功", f"标准库加载成功！\n共加载 {n_spectra} 条标准光谱")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载标准库失败：{str(e)}")
            self.library_status_label.setText("状态: 加载失败")
            self.library_status_label.setStyleSheet("color: red; font-size: 9pt;")
    
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
            
            QMessageBox.information(self, "成功", 
                                  f"合成数据生成完成！\n"
                                  f"加载纯组分: {loaded_count} 个\n"
                                  f"生成样本: {n_samples} 条\n"
                                  f"已保存: {saved_count} 个文件\n"
                                  f"保存目录: {save_dir}")
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据增强失败：{str(e)}")
            traceback.print_exc()
    
    # --- Tab 3: 波峰检测 ---
    def setup_peak_detection_tab(self):
        tab3 = QWidget()
        layout = QVBoxLayout(tab3)
        layout.setSpacing(10)
        
        # 波峰检测配置
        advanced_group = CollapsibleGroupBox("⚙️ 波峰检测与垂直参考线", is_expanded=True)
        adv_layout = QFormLayout()
        
        # 波峰检测开关
        self.peak_check = QCheckBox("启用自动波峰检测")
        adv_layout.addRow(self.peak_check)
        
        # 波峰检测参数组
        peak_params_group = QGroupBox("波峰检测参数")
        peak_params_layout = QFormLayout(peak_params_group)
        
        # 基础参数（使用相对较小的默认值，代码会自动根据数据范围调整）
        self.peak_height_spin = QDoubleSpinBox()
        self.peak_height_spin.setRange(-999999999.0, 999999999.0)
        self.peak_height_spin.setDecimals(15)
        self.peak_height_spin.setValue(0.0)  # 0表示自动（使用数据最大值的10%）
        self.peak_height_spin.setSpecialValueText("自动")
        
        self.peak_distance_spin = QSpinBox()
        self.peak_distance_spin.setRange(-999999999, 999999999)
        self.peak_distance_spin.setValue(10)  # 减小默认值，更容易检测到峰值
        self.peak_distance_spin.setSpecialValueText("自动")
        
        # 新增参数：prominence（突出度）
        self.peak_prominence_spin = QDoubleSpinBox()
        self.peak_prominence_spin.setRange(-999999999.0, 999999999.0)
        self.peak_prominence_spin.setDecimals(15)
        self.peak_prominence_spin.setValue(0.0)  # 0表示不使用此参数
        self.peak_prominence_spin.setSpecialValueText("禁用")
        
        # 新增参数：width（宽度）
        self.peak_width_spin = QDoubleSpinBox()
        self.peak_width_spin.setRange(-999999999.0, 999999999.0)
        self.peak_width_spin.setDecimals(15)
        self.peak_width_spin.setValue(1.0)
        
        # 新增参数：wlen（窗口长度）
        self.peak_wlen_spin = QSpinBox()
        self.peak_wlen_spin.setRange(-999999999, 999999999)
        self.peak_wlen_spin.setValue(200)
        
        # 新增参数：rel_height（相对高度，用于width计算）
        self.peak_rel_height_spin = QDoubleSpinBox()
        self.peak_rel_height_spin.setRange(-999999999.0, 999999999.0)
        self.peak_rel_height_spin.setDecimals(15)
        self.peak_rel_height_spin.setValue(0.5)
        
        peak_params_layout.addRow("峰高阈值 (height):", self.peak_height_spin)
        peak_params_layout.addRow("最小间距 (distance):", self.peak_distance_spin)
        peak_params_layout.addRow("突出度 (prominence):", self.peak_prominence_spin)
        peak_params_layout.addRow("最小宽度 (width):", self.peak_width_spin)
        peak_params_layout.addRow("窗口长度 (wlen):", self.peak_wlen_spin)
        peak_params_layout.addRow("相对高度 (rel_height):", self.peak_rel_height_spin)
        
        # 添加说明标签
        info_label = QLabel("提示：height和distance是基础参数，prominence和width是高级参数。\n如果检测不到峰值，尝试减小这些参数值。")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 9pt;")
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
        self.tab_widget.addTab(tab3, "⚙️ 波峰检测")

    # --- Tab 2: NMF 分析 ---
    def setup_nmf_tab(self):
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
        style_group = CollapsibleGroupBox("🎨 NMF 结果绘图样式", is_expanded=True)
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
        control_group = CollapsibleGroupBox("🔬 NMF 对照组设置", is_expanded=True)
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
        # NMF运行按钮已移到主界面底部按钮区，这里不再需要
        layout.addStretch(1)
        
        # 添加 NMF 提示
        info_label = QLabel("提示：NMF 分析将使用GUI中设置的所有预处理选项（QC、BE校正、平滑、基线校正、归一化等）。\n最终会将负值置零以满足NMF的非负要求。请确保在 'X 轴物理截断' 中设置了范围（例如 > 600 cm⁻¹）。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.tab_widget.addTab(tab2, "🧪 NMF 分析")

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
        
        self.tab_widget.addTab(tab3, "🔬 物理验证")

    # --- 辅助逻辑 (文件扫描和重命名) ---
    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据文件夹")
        if d: 
            self.folder_input.setText(d)
            self.scan_and_load_legend_rename() # 扫描并加载重命名

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
                QMessageBox.information(self, "提示", "未找到文件")
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
            QMessageBox.information(self, "完成", f"已加载 {len(file_list_full)} 个文件的独立控制项。\n每个文件都可以单独设置Y轴控制、颜色和动态范围压缩预处理。\n颜色改变时会自动更新图表，确保线条、阴影和图例颜色一致。")
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
                QMessageBox.information(self, "提示", "未找到文件")
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
            
            # 获取全局默认偏移值
            default_offset = self.global_stack_offset_spin.value()
            
            # 对组名进行排序
            sorted_group_names = sorted(groups.keys())
            
            # 为每组创建控制项
            for i, group_name in enumerate(sorted_group_names):
                group_widget = QWidget()
                group_vbox = QVBoxLayout(group_widget)
                group_vbox.setContentsMargins(5, 5, 5, 5)
                group_vbox.setSpacing(5)
                
                # 组名标签
                name_label = QLabel(f"📊 {group_name} (共 {len(groups[group_name])} 个文件)")
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
            QMessageBox.information(self, "完成", f"已加载 {len(sorted_group_names)} 个组的独立堆叠位移和颜色控制。\n每组都可以单独设置堆叠位移值和颜色。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"扫描组时出错: {str(e)}")
            traceback.print_exc()
    
    def run_2d_cos_analysis(self):
        """
        运行2D-COS分析：基于浓度梯度数据解析重叠峰
        
        关键点：
        - 扰动（浓度）存在于组之间，不在组内
        - 对每个组计算平均光谱
        - 使用自然排序确保组顺序正确（如 0mg -> 25mg -> 50mg）
        """
        try:
            folder = self.folder_input.text()
            if not os.path.isdir(folder):
                QMessageBox.warning(self, "错误", "请先选择数据文件夹")
                return
            
            # 物理截断值
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())
            
            # 读取基础参数
            skip = self.skip_rows_spin.value()
            n_chars = self.n_chars_spin.value()
            
            # 获取文件并分组
            files = sorted(glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt')))
            groups = group_files_by_name(files, n_chars)
            
            # 筛选指定组（如果用户指定了）
            target_gs = [x.strip() for x in self.groups_input.text().split(',') if x.strip()]
            if target_gs:
                groups = {k: v for k, v in groups.items() if k in target_gs}
            
            if len(groups) < 2:
                QMessageBox.warning(self, "错误", "2D-COS分析至少需要2个组（浓度梯度）")
                return
            
            # 使用自然排序对组名进行排序（关键：确保浓度顺序正确）
            initial_sorted_names = sorted(groups.keys(), key=natural_sort_key)
            
            # 创建手动确认组顺序的对话框
            order_dialog = QDialog(self)
            order_dialog.setWindowTitle("确认 2D-COS 浓度梯度顺序（从低到高）")
            order_dialog.setMinimumSize(400, 300)
            order_layout = QVBoxLayout(order_dialog)
            
            # 说明标签
            info_label = QLabel("请拖拽调整组的顺序（从上到下表示浓度从低到高）：")
            order_layout.addWidget(info_label)
            
            # 可拖拽排序的列表
            list_widget = QListWidget()
            list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            list_widget.addItems(initial_sorted_names)
            order_layout.addWidget(list_widget)
            
            # 按钮布局
            button_layout = QHBoxLayout()
            btn_ok = QPushButton("确定")
            btn_cancel = QPushButton("取消")
            btn_ok.clicked.connect(order_dialog.accept)
            btn_cancel.clicked.connect(order_dialog.reject)
            button_layout.addWidget(btn_ok)
            button_layout.addWidget(btn_cancel)
            order_layout.addLayout(button_layout)
            
            # 显示对话框并获取用户选择
            if order_dialog.exec() != QDialog.DialogCode.Accepted:
                # 用户点击取消或关闭对话框，终止函数执行
                return
            
            # 从 QListWidget 中按顺序提取最终的组名列表
            final_sorted_groups = []
            for i in range(list_widget.count()):
                final_sorted_groups.append(list_widget.item(i).text())
            
            # 收集每个组的平均光谱
            group_averages = []
            common_x = None
            
            for g_name in final_sorted_groups:
                g_files = groups[g_name]
                y_list = []
                group_x = None
                
                # 组内处理：收集所有有效光谱并计算平均
                for f in g_files:
                    try:
                        x, y = self.read_data(f, skip, x_min_phys, x_max_phys)
                        if group_x is None:
                            group_x = x
                        if common_x is None:
                            common_x = x
                        
                        # 应用预处理（与run_group_average_waterfall一致）
                        # A. QC
                        if self.qc_check.isChecked() and np.max(y) < self.qc_threshold_spin.value():
                            continue
                        
                        # B. BE 校正
                        if self.be_check.isChecked():
                            y = DataPreProcessor.apply_bose_einstein_correction(x, y, self.be_temp_spin.value())
                        
                        # C. 平滑
                        if self.smoothing_check.isChecked():
                            y = DataPreProcessor.apply_smoothing(y, self.smoothing_window_spin.value(), self.smoothing_poly_spin.value())
                        
                        # D. 基线校正 (AsLS优先)
                        if self.baseline_als_check.isChecked():
                            b = DataPreProcessor.apply_baseline_als(y, self.lam_spin.value(), self.p_spin.value())
                            y = y - b
                            y[y < 0] = 0
                        elif self.baseline_poly_check.isChecked():
                            y = DataPreProcessor.apply_baseline_correction(x, y, self.baseline_points_spin.value(), self.baseline_poly_spin.value())
                        
                        # E. 归一化
                        normalization_mode = self.normalization_combo.currentText()
                        if normalization_mode == 'snv':
                            y = DataPreProcessor.apply_snv(y)
                        elif normalization_mode == 'max':
                            y = DataPreProcessor.apply_normalization(y, 'max')
                        elif normalization_mode == 'area':
                            y = DataPreProcessor.apply_normalization(y, 'area')
                        
                        # 如果X轴不一致，需要插值对齐
                        if len(x) != len(common_x) or not np.allclose(x, common_x):
                            from scipy.interpolate import interp1d
                            f_interp = interp1d(x, y, kind='linear', fill_value=0, bounds_error=False)
                            y = f_interp(common_x)
                        
                        y_list.append(y)
                    except Exception as e:
                        print(f"警告：处理文件 {os.path.basename(f)} 时出错: {e}")
                        continue
                
                if not y_list:
                    print(f"警告：组 {g_name} 无有效数据，跳过")
                    continue
                
                # 计算该组的平均光谱
                y_array = np.array(y_list)
                y_avg = np.mean(y_array, axis=0)
                group_averages.append(y_avg)
            
            if len(group_averages) < 2:
                QMessageBox.warning(self, "错误", "有效组数不足（至少需要2个组）")
                return
            
            if common_x is None:
                QMessageBox.warning(self, "错误", "无法确定公共波数轴")
                return
            
            # 构建扰动矩阵 X (n_groups, n_wavenumbers)
            X_matrix = np.array(group_averages)
            
            # 打开2D-COS窗口
            if not hasattr(self, 'cos_window') or self.cos_window is None:
                self.cos_window = TwoDCOSWindow(self)
            
            self.cos_window.set_data(X_matrix, common_x, final_sorted_groups)
            self.cos_window.show()
            self.cos_window.raise_()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"2D-COS分析失败：{str(e)}")
            traceback.print_exc()
    
    def export_group_averages(self):
        """导出组瀑布图中所有组的平均值谱线"""
        try:
            folder_path = self.folder_input.text()
            if not os.path.isdir(folder_path):
                QMessageBox.warning(self, "警告", "请先设置数据文件夹路径")
                return
            
            # 获取分组参数
            skip = self.skip_rows_spin.value()
            n_chars = self.n_chars_spin.value()
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())
            
            # 扫描文件并分组
            files = sorted(glob.glob(os.path.join(folder_path, '*.csv')) + glob.glob(os.path.join(folder_path, '*.txt')))
            if not files:
                QMessageBox.warning(self, "警告", "未找到文件")
                return
            
            groups = group_files_by_name(files, n_chars)
            
            # 筛选指定组
            target_gs = [x.strip() for x in self.groups_input.text().split(',') if x.strip()]
            if target_gs:
                groups = {k: v for k, v in groups.items() if k in target_gs}
            
            if not groups:
                QMessageBox.warning(self, "警告", "未找到有效的组")
                return
            
            # 选择保存目录
            save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", folder_path)
            if not save_dir:
                return
            
            # 对组名进行排序
            sorted_group_names = sorted(groups.keys())
            
            exported_count = 0
            
            # 处理每一组
            for group_name in sorted_group_names:
                g_files = groups[group_name]
                y_list = []
                common_x = None
                
                # 组内处理：收集所有有效光谱
                for f in g_files:
                    try:
                        x, y = self.read_data(f, skip, x_min_phys, x_max_phys)
                        if common_x is None:
                            common_x = x
                        
                        # 预处理流程（与run_group_average_waterfall一致）
                        # A. QC
                        if self.qc_check.isChecked() and np.max(y) < self.qc_threshold_spin.value():
                            continue
                        
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
                            y[y < 0] = 0
                        
                        # E. 归一化
                        if self.normalization_combo.currentText() == 'snv':
                            y = DataPreProcessor.apply_snv(y)
                        elif self.normalization_combo.currentText() == 'max':
                            y = DataPreProcessor.apply_normalization(y, 'max')
                        
                        y_list.append(y)
                    except:
                        pass
                
                if not y_list or common_x is None:
                    continue
                
                # 计算平均值
                y_array = np.array(y_list)
                y_avg = np.mean(y_array, axis=0)
                y_std = np.std(y_array, axis=0)
                
                # 应用缩放
                scale = self.global_y_scale_factor_spin.value()
                y_avg_scaled = y_avg * scale
                y_std_scaled = y_std * scale
                
                # 是否求导
                if self.derivative_check.isChecked():
                    d1 = np.gradient(y_avg_scaled, common_x)
                    y_avg_scaled = np.gradient(d1, common_x)
                    y_std_scaled = None
                
                # 保存平均值谱线
                output_file = os.path.join(save_dir, f"{group_name}_average.txt")
                with open(output_file, 'w') as f:
                    f.write("Wavenumber\tIntensity_Avg")
                    if y_std_scaled is not None:
                        f.write("\tIntensity_Std")
                    f.write("\n")
                    
                    for i in range(len(common_x)):
                        f.write(f"{common_x[i]:.2f}\t{y_avg_scaled[i]:.6f}")
                        if y_std_scaled is not None:
                            f.write(f"\t{y_std_scaled[i]:.6f}")
                        f.write("\n")
                
                exported_count += 1
            
            QMessageBox.information(self, "完成", f"已成功导出 {exported_count} 个组的平均值谱线到:\n{save_dir}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出平均值谱线时出错: {str(e)}")
            traceback.print_exc()
    
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
            name_label = QLabel(f"🔬 {comp_label}")
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

    # --- 核心：运行绘图逻辑 ---
    def run_plot_logic(self):
        try:
            folder = self.folder_input.text()
            if not os.path.isdir(folder): return
            
            # 物理截断值
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())
            
            # 收集参数
            params = {
                # 模式与全局
                'plot_mode': self.plot_mode_combo.currentText(),
                'show_y_values': self.show_y_val_check.isChecked(),
                'is_derivative': self.derivative_check.isChecked(),
                'x_axis_invert': self.x_axis_invert_check.isChecked(),
                'global_stack_offset': self.global_stack_offset_spin.value(),
                'global_scale_factor': self.global_y_scale_factor_spin.value(),
                'main_title_text': self.main_title_input.text(),
                'main_title_fontsize': self.main_title_font_spin.value(),
                'main_title_pad': self.main_title_pad_spin.value(),
                'main_title_show': self.main_title_show_check.isChecked(),
                'plot_style': self.plot_style_combo.currentText(), # 新增：绘制风格

                # 标签与边距 
                'xlabel_text': self.xlabel_input.text(),
                'ylabel_text': self.ylabel_input.text(),
                'xlabel_fontsize': self.xlabel_font_spin.value(),
                'xlabel_pad': self.xlabel_pad_spin.value(),
                'xlabel_show': self.xlabel_show_check.isChecked(),
                'ylabel_fontsize': self.ylabel_font_spin.value(),
                'ylabel_pad': self.ylabel_pad_spin.value(),
                'ylabel_show': self.ylabel_show_check.isChecked(), 
                
                
                # 预处理
                'qc_enabled': self.qc_check.isChecked(),
                'qc_threshold': self.qc_threshold_spin.value(),
                'is_baseline_als': self.baseline_als_check.isChecked(),
                'als_lam': self.lam_spin.value(),
                'als_p': self.p_spin.value(),
                'is_baseline': False, # 旧版基线默认关闭，以免冲突
                'baseline_points': 50,
                'baseline_poly': 3,
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
                
                # 出版质量样式 
                'fig_width': self.fig_width_spin.value(),
                'fig_height': self.fig_height_spin.value(),
                'fig_dpi': self.fig_dpi_spin.value(),
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
                'shadow_alpha': self.shadow_alpha_spin.value(),
                'show_legend': self.show_legend_check.isChecked(),
                'legend_frame': self.legend_frame_check.isChecked(),
                'legend_loc': self.legend_loc_combo.currentText(),
                'legend_ncol': self.legend_column_spin.value() if hasattr(self, 'legend_column_spin') else 1,
                'legend_columnspacing': self.legend_columnspacing_spin.value() if hasattr(self, 'legend_columnspacing_spin') else 2.0,
                'legend_labelspacing': self.legend_labelspacing_spin.value() if hasattr(self, 'legend_labelspacing_spin') else 0.5,
                'legend_handlelength': self.legend_handlelength_spin.value() if hasattr(self, 'legend_handlelength_spin') else 2.0,
                'border_sides': self.get_checked_border_sides(),
                'border_linewidth': self.spine_width_spin.value(),
                'aspect_ratio': self.aspect_ratio_spin.value(), # 新增：纵横比
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

            # 读取文件列表
            skip = self.skip_rows_spin.value()
            all_files = sorted(glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt')))
            
            # 提取对照文件（自动识别后缀）
            c_text = self.control_files_input.toPlainText()
            c_names = [x.strip() for x in c_text.replace('\n', ',').split(',') if x.strip()]
            
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
                        x, y = self.read_data(f, skip, x_min_phys, x_max_phys) # 使用物理截断
                        g_data.append((f, x, y))
                    except ValueError as ve:
                         QMessageBox.warning(self, "警告", f"文件 {os.path.basename(f)} 读取失败: {ve}")
                    except: pass
                
                params['grouped_files_data'] = g_data
                
                if g_name not in self.plot_windows:
                    # 创建新窗口
                    self.plot_windows[g_name] = MplPlotWindow(g_name, parent=self)
                
                win = self.plot_windows[g_name]
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

    def get_checked_border_sides(self):
        # 收集边框可见性
        sides = []
        if self.spine_top_check.isChecked(): sides.append('top')
        if self.spine_bottom_check.isChecked(): sides.append('bottom')
        if self.spine_left_check.isChecked(): sides.append('left')
        if self.spine_right_check.isChecked(): sides.append('right')
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

    def run_nmf_regression_mode(self):
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
                    if self.derivative_check.isChecked():
                        d1 = np.gradient(y_proc, x)
                        y_proc = np.gradient(d1, x)
                    
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
                try:
                    for comp_label, rename_widget in list(self.nmf_component_rename_widgets.items()):
                        try:
                            new_name = rename_widget.text().strip()
                            if new_name:
                                nmf_legend_names[comp_label] = new_name
                        except (RuntimeError, AttributeError):
                            continue
                except (RuntimeError, AttributeError):
                    pass
            # 然后从主窗口的legend_rename_widgets获取（优先级更高）
            if hasattr(self, 'legend_rename_widgets'):
                try:
                    for key, widget in list(self.legend_rename_widgets.items()):
                        try:
                            if hasattr(widget, 'text'):
                                renamed = widget.text().strip()
                                if renamed and key.startswith('NMF Component'):
                                    # 提取组件编号
                                    comp_num = key.replace('NMF Component ', '')
                                    comp_label = f"Component {comp_num}"
                                    nmf_legend_names[comp_label] = renamed
                        except (RuntimeError, AttributeError):
                            continue
                except (RuntimeError, AttributeError):
                    pass
            
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
            
            # 收集 NMF 业务参数（不包含主窗口的样式参数，让窗口使用自己的默认设置）
            nmf_style_params = {
                # NMF特定业务参数
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
                'x_axis_invert': self.x_axis_invert_check.isChecked(),
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
                'is_derivative': self.derivative_check.isChecked(),
                'global_stack_offset': self.global_stack_offset_spin.value(),
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

    def run_nmf_analysis(self):
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
                    if self.derivative_check.isChecked():
                        d1 = np.gradient(y_proc, x)
                        y_proc = np.gradient(d1, x)
                    
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
                    if self.derivative_check.isChecked():
                        d1 = np.gradient(y_proc, x)
                        y_proc = np.gradient(d1, x)
                    
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
                        if self.derivative_check.isChecked():
                            d1 = np.gradient(y_proc, x)
                            y_proc = np.gradient(d1, x)
                        
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
            
            # 应用 SVD 去噪（如果启用）
            if hasattr(self, 'svd_denoise_check') and self.svd_denoise_check.isChecked():
                k_components = self.svd_components_spin.value() if hasattr(self, 'svd_components_spin') else 5
                X = DataPreProcessor.svd_denoise(X, k_components)
                print(f"已应用 SVD 去噪，保留 {k_components} 个主成分")
            
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
                    QMessageBox.information(self, "提示",
                                          f"已自动调整预滤波组件数为 {filter_components} 以匹配 NMF 组件数。")
            
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
                'x_axis_invert': self.x_axis_invert_check.isChecked(),
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
                'is_derivative': self.derivative_check.isChecked(),
                'global_stack_offset': self.global_stack_offset_spin.value(),
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
                    if self.derivative_check.isChecked():
                        d1 = np.gradient(y_proc, x)
                        y_proc = np.gradient(d1, x)
                    
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
            # 使用QTimer延迟更新，避免频繁触发（防抖）
            if not hasattr(self, '_nmf_update_timer'):
                self._nmf_update_timer = QTimer()
                self._nmf_update_timer.setSingleShot(True)
                self._nmf_update_timer.timeout.connect(self.rerun_nmf_plot)
            
            # 重置定时器，300ms后执行更新
            self._nmf_update_timer.stop()
            self._nmf_update_timer.start(300)
    
    def rerun_nmf_plot(self):
        """重新绘制NMF图，不重新运行NMF分析，保留已设置的参数"""
        try:
            # 检查是否有NMF窗口和数据
            if not hasattr(self, 'nmf_window') or self.nmf_window is None or not hasattr(self.nmf_window, 'H'):
                QMessageBox.warning(self, "警告", "请先运行NMF分析。")
                return
            
            # 检查是否有控件
            if not hasattr(self, 'nmf_component_control_widgets') or not self.nmf_component_control_widgets:
                QMessageBox.warning(self, "警告", "请先运行NMF分析以创建控制项。")
                return
            
            # 收集独立Y轴参数和预处理选项（用于NMF组分绘图）
            individual_y_params = {}
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
            
            # 收集对照组数据（如果存在）
            control_data_for_plot = []
            control_files_text = self.control_files_input.toPlainText().strip()
            if control_files_text:
                folder = self.folder_input.text()
                control_names = [name.strip() for name in control_files_text.replace(',', '\n').split('\n') if name.strip()]
                for c_name in control_names:
                    for ext in ['.txt', '.csv']:
                        c_file = os.path.join(folder, c_name + ext)
                        if os.path.exists(c_file):
                            try:
                                skip = self.skip_rows_spin.value()
                                x_min_phys = float(self.x_min_phys_input.text()) if self.x_min_phys_input.text().strip() else None
                                x_max_phys = float(self.x_max_phys_input.text()) if self.x_max_phys_input.text().strip() else None
                                x, y = self.read_data(c_file, skip, x_min_phys, x_max_phys)
                                y_proc = y.astype(float)
                                
                                # 应用预处理（与NMF输入数据相同的预处理，使用主菜单的所有预处理参数）
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
                                
                                # 6. 全局动态范围压缩（如果启用）- 在归一化之后
                                global_transform_mode = self.global_transform_combo.currentText()
                                if global_transform_mode == '对数变换 (Log)':
                                    base = float(self.global_log_base_combo.currentText()) if self.global_log_base_combo.currentText() == '10' else np.e
                                    y_proc = DataPreProcessor.apply_log_transform(y_proc, base=base, offset=self.global_log_offset_spin.value())
                                elif global_transform_mode == '平方根变换 (Sqrt)':
                                    y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=self.global_sqrt_offset_spin.value())
                                
                                # 7. 二次导数（如果启用）- 在全局动态变换之后
                                if self.derivative_check.isChecked():
                                    d1 = np.gradient(y_proc, x)
                                    y_proc = np.gradient(d1, x)
                                
                                # 8. 整体Y轴偏移（预处理最后一步，在二次导数之后）
                                global_y_offset = self.global_y_offset_spin.value() if hasattr(self, 'global_y_offset_spin') else 0.0
                                y_proc = y_proc + global_y_offset
                                
                                control_data_for_plot.append({
                                    'x': x,
                                    'y': y_proc,
                                    'label': os.path.splitext(os.path.basename(c_file))[0]
                                })
                                break
                            except: pass
            
            # 收集 NMF 样式参数
            nmf_style_params = {
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
                'x_axis_invert': self.x_axis_invert_check.isChecked(),
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
                'is_derivative': self.derivative_check.isChecked(),
                'global_stack_offset': self.global_stack_offset_spin.value(),
                'global_scale_factor': self.global_y_scale_factor_spin.value(),
                'individual_y_params': individual_y_params,
                'nmf_legend_names': nmf_legend_names,
                'control_data_list': control_data_for_plot,
            }
            
            # 更新现有窗口
            self.nmf_window.set_data(self.nmf_window.W, self.nmf_window.H, self.nmf_window.common_x, nmf_style_params, self.nmf_window.sample_labels)
            self.nmf_window.raise_()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重新绘制失败: {str(e)}")
            traceback.print_exc()

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
            offset_step = self.global_stack_offset_spin.value()
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
                # 创建新窗口
                self.plot_windows["GroupComparison"] = MplPlotWindow("Group Comparison (Averages)", parent=self)
            
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
                if self.waterfall_shadow_check.isChecked() and final_y_upper is not None and final_y_lower is not None:
                    shadow_alpha = self.waterfall_shadow_alpha_spin.value()
                    # 阴影颜色与线条颜色完全一致
                    ax.fill_between(common_x, final_y_lower, final_y_upper, 
                                   color=color, alpha=shadow_alpha, label=std_label)
                
                # 绘制平均线 - 使用主菜单的样式参数（线宽、线型）
                line_width = self.line_width_spin.value()
                line_style = self.line_style_combo.currentText()
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
            font_family = self.font_family_combo.currentText()
            current_font = 'Times New Roman' if font_family == 'Times New Roman' else font_family
            
            # 坐标轴翻转
            if self.x_axis_invert_check.isChecked(): ax.invert_xaxis()
            if not self.show_y_val_check.isChecked(): ax.set_yticks([])
            
            # 使用GUI中的浓度梯度图X轴标题控制参数
            if self.gradient_xlabel_show_check.isChecked():
                ax.set_xlabel(self.xlabel_input.text(), fontsize=self.gradient_xlabel_font_spin.value(), 
                            labelpad=self.gradient_xlabel_pad_spin.value(), fontfamily=current_font)
            
            # 使用GUI中的浓度梯度图Y轴标题控制参数
            ylabel = "2nd Derivative" if self.derivative_check.isChecked() else self.ylabel_input.text()
            if self.gradient_ylabel_show_check.isChecked():
                ax.set_ylabel(ylabel, fontsize=self.gradient_ylabel_font_spin.value(), 
                            labelpad=self.gradient_ylabel_pad_spin.value(), fontfamily=current_font)
            
            # 使用GUI中的标题控制参数
            if self.gradient_title_show_check.isChecked():
                gradient_title_text = self.gradient_title_input.text().strip() or "Concentration Gradient (Group Averages)"
                ax.set_title(gradient_title_text, fontsize=self.gradient_title_font_spin.value(), 
                           pad=self.gradient_title_pad_spin.value(), fontfamily=current_font)
            
            # Ticks 样式（使用主菜单的样式参数）
            tick_direction = self.tick_direction_combo.currentText()
            tick_len_major = self.tick_len_major_spin.value()
            tick_len_minor = self.tick_len_minor_spin.value()
            tick_width = self.tick_width_spin.value()
            tick_label_fontsize = self.tick_label_font_spin.value()
            
            ax.tick_params(labelsize=tick_label_fontsize, direction=tick_direction, width=tick_width, labelfontfamily=current_font)
            ax.tick_params(which='major', length=tick_len_major)
            ax.tick_params(which='minor', length=tick_len_minor)
            
            # 边框设置 (Spines) - 使用主菜单的样式参数
            border_sides = self.get_checked_border_sides()
            border_linewidth = self.spine_width_spin.value()
            for side in ['top', 'right', 'left', 'bottom']:
                if side in border_sides:
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_linewidth(border_linewidth)
                else:
                    ax.spines[side].set_visible(False)
            
            # 网格设置 - 使用主菜单的样式参数
            if self.show_grid_check.isChecked():
                ax.grid(True, alpha=self.grid_alpha_spin.value())
            
            # 图例设置 - 使用主菜单的样式参数
            if self.show_legend_check.isChecked():
                # 使用专门的图例字体大小控件（如果存在），否则使用通用的
                if hasattr(self, 'legend_fontsize_spin'):
                    legend_fontsize = self.legend_fontsize_spin.value()
                else:
                    legend_fontsize = self.legend_font_spin.value()
                
                legend_frame = self.legend_frame_check.isChecked()
                legend_loc = self.legend_loc_combo.currentText()
                
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
            aspect_ratio = self.aspect_ratio_spin.value()
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
            QMessageBox.information(self, "完成", f"已导出 {count} 个文件。")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    # --- 核心：参数保存与加载 ---
    def load_settings(self):
        # 1. 通用和预处理参数
        self.folder_input.setText(self.settings.value("folder", ""))
        self.n_chars_spin.setValue(int(self.settings.value("n_chars", 3)))
        self.skip_rows_spin.setValue(int(self.settings.value("skip_rows", 2)))
        
        self.qc_check.setChecked(self.settings.value("qc", False, type=bool))
        self.qc_threshold_spin.setValue(float(self.settings.value("qc_threshold", 5.0)))
        
        self.be_check.setChecked(self.settings.value("be_check", False, type=bool))
        self.be_temp_spin.setValue(float(self.settings.value("be_temp", 300.0)))

        self.baseline_als_check.setChecked(self.settings.value("asls", False, type=bool))
        self.lam_spin.setValue(float(self.settings.value("lam", 10000)))
        self.p_spin.setValue(float(self.settings.value("p", 0.005)))
        self.baseline_poly_check.setChecked(self.settings.value("baseline_poly_check", False, type=bool))
        self.baseline_points_spin.setValue(int(self.settings.value("baseline_points", 50)))
        self.baseline_poly_spin.setValue(int(self.settings.value("baseline_poly", 3)))
        
        self.smoothing_check.setChecked(self.settings.value("smooth_check", False, type=bool))
        self.smoothing_window_spin.setValue(int(self.settings.value("smooth_window", 15)))
        self.smoothing_poly_spin.setValue(int(self.settings.value("smooth_poly", 3)))

        self.normalization_combo.setCurrentText(self.settings.value("norm", "None"))
        
        # 2. 绘图模式和全局设置
        self.plot_mode_combo.setCurrentText(self.settings.value("mode", "Normal Overlay"))
        self.plot_style_combo.setCurrentText(self.settings.value("plot_style", "line"))
        self.derivative_check.setChecked(self.settings.value("derivative", False, type=bool))
        self.x_axis_invert_check.setChecked(self.settings.value("x_invert", False, type=bool))
        self.show_y_val_check.setChecked(self.settings.value("show_y", True, type=bool))
        self.global_stack_offset_spin.setValue(float(self.settings.value("stack_offset", 0.5)))
        self.global_y_scale_factor_spin.setValue(float(self.settings.value("y_scale", 1.0)))
        self.main_title_input.setText(self.settings.value("main_title", ""))
        self.main_title_font_spin.setValue(int(self.settings.value("main_title_fontsize", 20)))
        self.main_title_pad_spin.setValue(float(self.settings.value("main_title_pad", 10.0)))
        self.main_title_show_check.setChecked(self.settings.value("main_title_show", True, type=bool))
        
        # 浓度梯度图标题控制
        self.gradient_title_input.setText(self.settings.value("gradient_title", "Concentration Gradient (Group Averages)"))
        self.gradient_title_font_spin.setValue(int(self.settings.value("gradient_title_fontsize", 22)))
        self.gradient_title_pad_spin.setValue(float(self.settings.value("gradient_title_pad", 10.0)))
        self.gradient_title_show_check.setChecked(self.settings.value("gradient_title_show", True, type=bool))

        # 3. X/Y 标签和物理截断
        self.xlabel_input.setText(self.settings.value("xlabel_text", "Wavenumber ($\\mathrm{cm^{-1}}$)"))
        self.xlabel_font_spin.setValue(int(self.settings.value("xlabel_fontsize", 20)))
        self.xlabel_pad_spin.setValue(float(self.settings.value("xlabel_pad", 10.0)))
        self.xlabel_show_check.setChecked(self.settings.value("xlabel_show", True, type=bool))
        
        self.ylabel_input.setText(self.settings.value("ylabel_text", "Transmittance"))
        self.ylabel_font_spin.setValue(int(self.settings.value("ylabel_fontsize", 20)))
        self.ylabel_pad_spin.setValue(float(self.settings.value("ylabel_pad", 10.0)))
        self.ylabel_show_check.setChecked(self.settings.value("ylabel_show", True, type=bool))
        
        # 浓度梯度图轴标题控制
        self.gradient_xlabel_font_spin.setValue(int(self.settings.value("gradient_xlabel_fontsize", 20)))
        self.gradient_xlabel_pad_spin.setValue(float(self.settings.value("gradient_xlabel_pad", 10.0)))
        self.gradient_xlabel_show_check.setChecked(self.settings.value("gradient_xlabel_show", True, type=bool))
        
        self.gradient_ylabel_font_spin.setValue(int(self.settings.value("gradient_ylabel_fontsize", 20)))
        self.gradient_ylabel_pad_spin.setValue(float(self.settings.value("gradient_ylabel_pad", 10.0)))
        self.gradient_ylabel_show_check.setChecked(self.settings.value("gradient_ylabel_show", True, type=bool))
        self.x_min_phys_input.setText(self.settings.value("x_min_phys", ""))
        self.x_max_phys_input.setText(self.settings.value("x_max_phys", ""))
        
        # 4. 文件选择相关
        self.control_files_input.setPlainText(self.settings.value("control_files", ""))
        self.groups_input.setText(self.settings.value("groups_input", ""))
        
        # 5. 出版质量样式（完整加载）
        self.fig_width_spin.setValue(float(self.settings.value("fig_width", 10.0)))
        self.fig_height_spin.setValue(float(self.settings.value("fig_height", 6.0)))
        self.fig_dpi_spin.setValue(int(self.settings.value("fig_dpi", 300)))
        self.aspect_ratio_spin.setValue(float(self.settings.value("aspect_ratio", 0.6)))  # 默认0.6
        self.axis_title_font_spin.setValue(int(self.settings.value("axis_title_font", 20)))
        self.tick_label_font_spin.setValue(int(self.settings.value("tick_label_font", 16)))
        self.legend_font_spin.setValue(int(self.settings.value("legend_font", 10)))
        self.line_width_spin.setValue(float(self.settings.value("line_width", 1.2)))
        self.line_style_combo.setCurrentText(self.settings.value("line_style", "-"))
        self.font_family_combo.setCurrentText(self.settings.value("font_family", "Times New Roman"))
        self.tick_direction_combo.setCurrentText(self.settings.value("tick_direction", "in"))
        self.tick_len_major_spin.setValue(int(self.settings.value("tick_len_major", 8)))
        self.tick_len_minor_spin.setValue(int(self.settings.value("tick_len_minor", 4)))
        self.tick_width_spin.setValue(float(self.settings.value("tick_width", 1.0)))
        self.show_grid_check.setChecked(self.settings.value("show_grid", False, type=bool))
        self.grid_alpha_spin.setValue(float(self.settings.value("grid_alpha", 0.2)))
        self.shadow_alpha_spin.setValue(float(self.settings.value("shadow_alpha", 0.25)))
        self.show_legend_check.setChecked(self.settings.value("show_legend", True, type=bool))
        self.legend_frame_check.setChecked(self.settings.value("legend_frame", True, type=bool))
        self.legend_loc_combo.setCurrentText(self.settings.value("legend_loc", "best"))
        
        # 图例大小和间距控制
        if hasattr(self, 'legend_fontsize_spin'):
            self.legend_fontsize_spin.setValue(int(self.settings.value("legend_fontsize", 10)))
        if hasattr(self, 'legend_column_spin'):
            self.legend_column_spin.setValue(int(self.settings.value("legend_column", 1)))
        if hasattr(self, 'legend_columnspacing_spin'):
            self.legend_columnspacing_spin.setValue(float(self.settings.value("legend_columnspacing", 2.0)))
        if hasattr(self, 'legend_labelspacing_spin'):
            self.legend_labelspacing_spin.setValue(float(self.settings.value("legend_labelspacing", 0.5)))
        if hasattr(self, 'legend_handlelength_spin'):
            self.legend_handlelength_spin.setValue(float(self.settings.value("legend_handlelength", 2.0)))
        self.spine_top_check.setChecked(self.settings.value("spine_top", True, type=bool))
        self.spine_bottom_check.setChecked(self.settings.value("spine_bottom", True, type=bool))
        self.spine_left_check.setChecked(self.settings.value("spine_left", True, type=bool))
        self.spine_right_check.setChecked(self.settings.value("spine_right", True, type=bool))
        self.spine_width_spin.setValue(float(self.settings.value("spine_width", 2.0)))
        
        # 6. 高级设置（波峰检测、垂直参考线）
        self.peak_check.setChecked(self.settings.value("peak_check", False, type=bool))
        self.peak_height_spin.setValue(float(self.settings.value("peak_height", 0.0)))  # 默认0表示自动
        self.peak_distance_spin.setValue(int(self.settings.value("peak_distance", 10)))  # 减小默认值
        self.peak_prominence_spin.setValue(float(self.settings.value("peak_prominence", 0.0)))  # 默认0表示禁用
        self.peak_width_spin.setValue(float(self.settings.value("peak_width", 1.0)))
        self.peak_wlen_spin.setValue(int(self.settings.value("peak_wlen", 200)))
        self.peak_rel_height_spin.setValue(float(self.settings.value("peak_rel_height", 0.5)))
        self.peak_show_label_check.setChecked(self.settings.value("peak_show_label", True, type=bool))
        self.peak_label_font_combo.setCurrentText(self.settings.value("peak_label_font", "Times New Roman"))
        self.peak_label_size_spin.setValue(int(self.settings.value("peak_label_size", 10)))
        self.peak_label_color_input.setText(self.settings.value("peak_label_color", "black"))
        self.peak_label_bold_check.setChecked(self.settings.value("peak_label_bold", False, type=bool))
        self.peak_label_rotation_spin.setValue(float(self.settings.value("peak_label_rotation", 0.0)))
        self.peak_marker_shape_combo.setCurrentText(self.settings.value("peak_marker_shape", "x"))
        self.peak_marker_size_spin.setValue(int(self.settings.value("peak_marker_size", 10)))
        self.peak_marker_color_input.setText(self.settings.value("peak_marker_color", ""))
        self.vertical_lines_input.setPlainText(self.settings.value("vertical_lines", ""))
        self.vertical_line_color_input.setText(self.settings.value("vertical_line_color", "gray"))
        self.vertical_line_width_spin.setValue(float(self.settings.value("vertical_line_width", 0.8)))
        self.vertical_line_style_combo.setCurrentText(self.settings.value("vertical_line_style", ":"))
        self.vertical_line_alpha_spin.setValue(float(self.settings.value("vertical_line_alpha", 0.7)))
        
        # 7. NMF和物理拟合参数
        self.nmf_comp_spin.setValue(int(self.settings.value("nmf_comp", 2)))
        self.nmf_max_iter.setValue(int(self.settings.value("nmf_max_iter", 200)))
        self.nmf_top_title_input.setText(self.settings.value("nmf_top_title", "Extracted Spectra (Components)"))
        self.nmf_bottom_title_input.setText(self.settings.value("nmf_bottom_title", "Concentration Weights (vs. Sample)"))
        self.nmf_top_title_font_spin.setValue(int(self.settings.value("nmf_top_title_fontsize", 16)))
        self.nmf_top_title_pad_spin.setValue(float(self.settings.value("nmf_top_title_pad", 10.0)))
        self.nmf_top_title_show_check.setChecked(self.settings.value("nmf_top_title_show", True, type=bool))
        self.nmf_bottom_title_font_spin.setValue(int(self.settings.value("nmf_bottom_title_fontsize", 16)))
        self.nmf_bottom_title_pad_spin.setValue(float(self.settings.value("nmf_bottom_title_pad", 10.0)))
        self.nmf_bottom_title_show_check.setChecked(self.settings.value("nmf_bottom_title_show", True, type=bool))
        self.nmf_xlabel_top_input.setText(self.settings.value("nmf_top_xlabel", "Wavenumber ($\\mathrm{cm^{-1}}$)"))
        self.nmf_top_xlabel_font_spin.setValue(int(self.settings.value("nmf_top_xlabel_fontsize", 16)))
        self.nmf_top_xlabel_pad_spin.setValue(float(self.settings.value("nmf_top_xlabel_pad", 10.0)))
        self.nmf_top_xlabel_show_check.setChecked(self.settings.value("nmf_top_xlabel_show", True, type=bool))
        
        self.nmf_ylabel_top_input.setText(self.settings.value("nmf_top_ylabel", "Intensity (Arb. Unit)"))
        self.nmf_top_ylabel_font_spin.setValue(int(self.settings.value("nmf_top_ylabel_fontsize", 16)))
        self.nmf_top_ylabel_pad_spin.setValue(float(self.settings.value("nmf_top_ylabel_pad", 10.0)))
        self.nmf_top_ylabel_show_check.setChecked(self.settings.value("nmf_top_ylabel_show", True, type=bool))
        
        self.nmf_xlabel_bottom_input.setText(self.settings.value("nmf_bottom_xlabel", "Sample Name"))
        self.nmf_bottom_xlabel_font_spin.setValue(int(self.settings.value("nmf_bottom_xlabel_fontsize", 16)))
        self.nmf_bottom_xlabel_pad_spin.setValue(float(self.settings.value("nmf_bottom_xlabel_pad", 10.0)))
        self.nmf_bottom_xlabel_show_check.setChecked(self.settings.value("nmf_bottom_xlabel_show", True, type=bool))
        
        self.nmf_ylabel_bottom_input.setText(self.settings.value("nmf_bottom_ylabel", "Weight (Arb. Unit)"))
        self.nmf_bottom_ylabel_font_spin.setValue(int(self.settings.value("nmf_bottom_ylabel_fontsize", 16)))
        self.nmf_bottom_ylabel_pad_spin.setValue(float(self.settings.value("nmf_bottom_ylabel_pad", 10.0)))
        self.nmf_bottom_ylabel_show_check.setChecked(self.settings.value("nmf_bottom_ylabel_show", True, type=bool))
        self.nmf_sort_method_combo.setCurrentText(self.settings.value("nmf_sort_method", "按文件名排序"))
        self.nmf_sort_reverse_check.setChecked(self.settings.value("nmf_sort_reverse", False, type=bool))
        self.nmf_include_control_check.setChecked(self.settings.value("nmf_include_control", False, type=bool))
        self.nmf_mode_standard.setChecked(self.settings.value("nmf_mode_standard", True, type=bool))
        self.nmf_mode_regression.setChecked(self.settings.value("nmf_mode_regression", False, type=bool))
        self.nmf_target_component_index = int(self.settings.value("nmf_target_component_index", 0))
        self.fit_cutoff_spin.setValue(float(self.settings.value("fit_cutoff", 400.0)))
        self.fit_model_combo.setCurrentText(self.settings.value("fit_model", "Lorentzian"))
        
        # 全局变换设置
        self.global_transform_combo.setCurrentText(self.settings.value("global_transform", "无"))
        self.global_log_base_combo.setCurrentText(self.settings.value("global_log_base", "10"))
        self.global_log_offset_spin.setValue(float(self.settings.value("global_log_offset", 1.0)))
        self.global_sqrt_offset_spin.setValue(float(self.settings.value("global_sqrt_offset", 0.0)))


    def closeEvent(self, event):
        # 退出时保存所有参数
        
        # 1. 通用和预处理参数
        self.settings.setValue("folder", self.folder_input.text())
        self.settings.setValue("n_chars", self.n_chars_spin.value())
        self.settings.setValue("skip_rows", self.skip_rows_spin.value())
        self.settings.setValue("qc", self.qc_check.isChecked())
        self.settings.setValue("qc_threshold", self.qc_threshold_spin.value())
        
        self.settings.setValue("be_check", self.be_check.isChecked())
        self.settings.setValue("be_temp", self.be_temp_spin.value())

        self.settings.setValue("asls", self.baseline_als_check.isChecked())
        self.settings.setValue("lam", self.lam_spin.value())
        self.settings.setValue("p", self.p_spin.value())
        self.settings.setValue("baseline_poly_check", self.baseline_poly_check.isChecked())
        self.settings.setValue("baseline_points", self.baseline_points_spin.value())
        self.settings.setValue("baseline_poly", self.baseline_poly_spin.value())
        
        self.settings.setValue("smooth_check", self.smoothing_check.isChecked())
        self.settings.setValue("smooth_window", self.smoothing_window_spin.value())
        self.settings.setValue("smooth_poly", self.smoothing_poly_spin.value())

        self.settings.setValue("norm", self.normalization_combo.currentText())
        
        # 2. 绘图模式和全局设置
        self.settings.setValue("mode", self.plot_mode_combo.currentText())
        self.settings.setValue("plot_style", self.plot_style_combo.currentText())
        self.settings.setValue("derivative", self.derivative_check.isChecked())
        self.settings.setValue("x_invert", self.x_axis_invert_check.isChecked())
        self.settings.setValue("show_y", self.show_y_val_check.isChecked())
        self.settings.setValue("stack_offset", self.global_stack_offset_spin.value())
        self.settings.setValue("y_scale", self.global_y_scale_factor_spin.value())
        self.settings.setValue("main_title", self.main_title_input.text())
        self.settings.setValue("main_title_fontsize", self.main_title_font_spin.value())
        self.settings.setValue("main_title_pad", self.main_title_pad_spin.value())
        self.settings.setValue("main_title_show", self.main_title_show_check.isChecked())
        
        # 浓度梯度图标题控制
        self.settings.setValue("gradient_title", self.gradient_title_input.text())
        self.settings.setValue("gradient_title_fontsize", self.gradient_title_font_spin.value())
        self.settings.setValue("gradient_title_pad", self.gradient_title_pad_spin.value())
        self.settings.setValue("gradient_title_show", self.gradient_title_show_check.isChecked())
        
        # 3. X/Y 标签和物理截断
        self.settings.setValue("xlabel_text", self.xlabel_input.text())
        self.settings.setValue("xlabel_fontsize", self.xlabel_font_spin.value())
        self.settings.setValue("xlabel_pad", self.xlabel_pad_spin.value())
        self.settings.setValue("xlabel_show", self.xlabel_show_check.isChecked())
        
        self.settings.setValue("ylabel_text", self.ylabel_input.text())
        self.settings.setValue("ylabel_fontsize", self.ylabel_font_spin.value())
        self.settings.setValue("ylabel_pad", self.ylabel_pad_spin.value())
        self.settings.setValue("ylabel_show", self.ylabel_show_check.isChecked())
        
        # 浓度梯度图轴标题控制
        self.settings.setValue("gradient_xlabel_fontsize", self.gradient_xlabel_font_spin.value())
        self.settings.setValue("gradient_xlabel_pad", self.gradient_xlabel_pad_spin.value())
        self.settings.setValue("gradient_xlabel_show", self.gradient_xlabel_show_check.isChecked())
        
        self.settings.setValue("gradient_ylabel_fontsize", self.gradient_ylabel_font_spin.value())
        self.settings.setValue("gradient_ylabel_pad", self.gradient_ylabel_pad_spin.value())
        self.settings.setValue("gradient_ylabel_show", self.gradient_ylabel_show_check.isChecked())
        self.settings.setValue("x_min_phys", self.x_min_phys_input.text())
        self.settings.setValue("x_max_phys", self.x_max_phys_input.text())
        
        # 4. 文件选择相关
        self.settings.setValue("control_files", self.control_files_input.toPlainText())
        self.settings.setValue("groups_input", self.groups_input.text())
        
        # 5. 出版质量样式参数（完整保存）
        self.settings.setValue("fig_width", self.fig_width_spin.value())
        self.settings.setValue("fig_height", self.fig_height_spin.value())
        self.settings.setValue("fig_dpi", self.fig_dpi_spin.value())
        self.settings.setValue("aspect_ratio", self.aspect_ratio_spin.value())
        self.settings.setValue("axis_title_font", self.axis_title_font_spin.value())
        self.settings.setValue("tick_label_font", self.tick_label_font_spin.value())
        self.settings.setValue("legend_font", self.legend_font_spin.value())
        self.settings.setValue("line_width", self.line_width_spin.value())
        self.settings.setValue("line_style", self.line_style_combo.currentText())
        self.settings.setValue("font_family", self.font_family_combo.currentText())
        self.settings.setValue("tick_direction", self.tick_direction_combo.currentText())
        self.settings.setValue("tick_len_major", self.tick_len_major_spin.value())
        self.settings.setValue("tick_len_minor", self.tick_len_minor_spin.value())
        self.settings.setValue("tick_width", self.tick_width_spin.value())
        self.settings.setValue("show_grid", self.show_grid_check.isChecked())
        self.settings.setValue("grid_alpha", self.grid_alpha_spin.value())
        self.settings.setValue("shadow_alpha", self.shadow_alpha_spin.value())
        self.settings.setValue("show_legend", self.show_legend_check.isChecked())
        self.settings.setValue("legend_frame", self.legend_frame_check.isChecked())
        self.settings.setValue("legend_loc", self.legend_loc_combo.currentText())
        
        # 图例大小和间距控制
        if hasattr(self, 'legend_fontsize_spin'):
            self.settings.setValue("legend_fontsize", self.legend_fontsize_spin.value())
        if hasattr(self, 'legend_column_spin'):
            self.settings.setValue("legend_column", self.legend_column_spin.value())
        if hasattr(self, 'legend_columnspacing_spin'):
            self.settings.setValue("legend_columnspacing", self.legend_columnspacing_spin.value())
        if hasattr(self, 'legend_labelspacing_spin'):
            self.settings.setValue("legend_labelspacing", self.legend_labelspacing_spin.value())
        if hasattr(self, 'legend_handlelength_spin'):
            self.settings.setValue("legend_handlelength", self.legend_handlelength_spin.value())
        self.settings.setValue("spine_top", self.spine_top_check.isChecked())
        self.settings.setValue("spine_bottom", self.spine_bottom_check.isChecked())
        self.settings.setValue("spine_left", self.spine_left_check.isChecked())
        self.settings.setValue("spine_right", self.spine_right_check.isChecked())
        self.settings.setValue("spine_width", self.spine_width_spin.value())
        
        # 6. 高级设置（波峰检测、垂直参考线）
        self.settings.setValue("peak_check", self.peak_check.isChecked())
        self.settings.setValue("peak_height", self.peak_height_spin.value())
        self.settings.setValue("peak_distance", self.peak_distance_spin.value())
        self.settings.setValue("peak_prominence", self.peak_prominence_spin.value())
        self.settings.setValue("peak_width", self.peak_width_spin.value())
        self.settings.setValue("peak_wlen", self.peak_wlen_spin.value())
        self.settings.setValue("peak_rel_height", self.peak_rel_height_spin.value())
        self.settings.setValue("peak_show_label", self.peak_show_label_check.isChecked())
        self.settings.setValue("peak_label_font", self.peak_label_font_combo.currentText())
        self.settings.setValue("peak_label_size", self.peak_label_size_spin.value())
        self.settings.setValue("peak_label_color", self.peak_label_color_input.text())
        self.settings.setValue("peak_label_bold", self.peak_label_bold_check.isChecked())
        self.settings.setValue("peak_label_rotation", self.peak_label_rotation_spin.value())
        self.settings.setValue("peak_marker_shape", self.peak_marker_shape_combo.currentText())
        self.settings.setValue("peak_marker_size", self.peak_marker_size_spin.value())
        self.settings.setValue("peak_marker_color", self.peak_marker_color_input.text())
        self.settings.setValue("vertical_lines", self.vertical_lines_input.toPlainText())
        self.settings.setValue("vertical_line_color", self.vertical_line_color_input.text())
        self.settings.setValue("vertical_line_width", self.vertical_line_width_spin.value())
        self.settings.setValue("vertical_line_style", self.vertical_line_style_combo.currentText())
        self.settings.setValue("vertical_line_alpha", self.vertical_line_alpha_spin.value())
        
        # 7. NMF和物理拟合参数
        self.settings.setValue("nmf_comp", self.nmf_comp_spin.value())
        self.settings.setValue("nmf_max_iter", self.nmf_max_iter.value())
        # 保存NMF目标组分索引（如果窗口存在，从窗口获取最新值）
        if hasattr(self, 'nmf_window') and self.nmf_window is not None:
            if hasattr(self.nmf_window, 'get_target_component_index'):
                self.nmf_target_component_index = self.nmf_window.get_target_component_index()
        self.settings.setValue("nmf_target_component_index", self.nmf_target_component_index)
        self.settings.setValue("nmf_top_title", self.nmf_top_title_input.text())
        self.settings.setValue("nmf_bottom_title", self.nmf_bottom_title_input.text())
        self.settings.setValue("nmf_top_title_fontsize", self.nmf_top_title_font_spin.value())
        self.settings.setValue("nmf_top_title_pad", self.nmf_top_title_pad_spin.value())
        self.settings.setValue("nmf_top_title_show", self.nmf_top_title_show_check.isChecked())
        self.settings.setValue("nmf_bottom_title_fontsize", self.nmf_bottom_title_font_spin.value())
        self.settings.setValue("nmf_bottom_title_pad", self.nmf_bottom_title_pad_spin.value())
        self.settings.setValue("nmf_bottom_title_show", self.nmf_bottom_title_show_check.isChecked())
        self.settings.setValue("nmf_top_xlabel", self.nmf_xlabel_top_input.text())
        self.settings.setValue("nmf_top_xlabel_fontsize", self.nmf_top_xlabel_font_spin.value())
        self.settings.setValue("nmf_top_xlabel_pad", self.nmf_top_xlabel_pad_spin.value())
        self.settings.setValue("nmf_top_xlabel_show", self.nmf_top_xlabel_show_check.isChecked())
        
        self.settings.setValue("nmf_top_ylabel", self.nmf_ylabel_top_input.text())
        self.settings.setValue("nmf_top_ylabel_fontsize", self.nmf_top_ylabel_font_spin.value())
        self.settings.setValue("nmf_top_ylabel_pad", self.nmf_top_ylabel_pad_spin.value())
        self.settings.setValue("nmf_top_ylabel_show", self.nmf_top_ylabel_show_check.isChecked())
        
        self.settings.setValue("nmf_bottom_xlabel", self.nmf_xlabel_bottom_input.text())
        self.settings.setValue("nmf_bottom_xlabel_fontsize", self.nmf_bottom_xlabel_font_spin.value())
        self.settings.setValue("nmf_bottom_xlabel_pad", self.nmf_bottom_xlabel_pad_spin.value())
        self.settings.setValue("nmf_bottom_xlabel_show", self.nmf_bottom_xlabel_show_check.isChecked())
        
        self.settings.setValue("nmf_bottom_ylabel", self.nmf_ylabel_bottom_input.text())
        self.settings.setValue("nmf_bottom_ylabel_fontsize", self.nmf_bottom_ylabel_font_spin.value())
        self.settings.setValue("nmf_bottom_ylabel_pad", self.nmf_bottom_ylabel_pad_spin.value())
        self.settings.setValue("nmf_bottom_ylabel_show", self.nmf_bottom_ylabel_show_check.isChecked())
        self.settings.setValue("nmf_sort_method", self.nmf_sort_method_combo.currentText())
        self.settings.setValue("nmf_sort_reverse", self.nmf_sort_reverse_check.isChecked())
        # 保存NMF目标组分索引（如果窗口存在，从窗口获取最新值）
        if hasattr(self, 'nmf_window') and self.nmf_window is not None:
            if hasattr(self.nmf_window, 'get_target_component_index'):
                self.nmf_target_component_index = self.nmf_window.get_target_component_index()
        self.settings.setValue("nmf_target_component_index", self.nmf_target_component_index)
        self.settings.setValue("fit_cutoff", self.fit_cutoff_spin.value())
        self.settings.setValue("fit_model", self.fit_model_combo.currentText())
        
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

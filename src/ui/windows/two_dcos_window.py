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


class TwoDCOSWindow(QDialog):
    """2D-COS (二维相关光谱) 分析窗口"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D-COS Analysis (Two-Dimensional Correlation Spectroscopy)")
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
        # 初始尺寸：纵轴:横轴 = 3:4，即宽度:高度 = 4:3，设置为 1200x900
        self.resize(1200, 900)
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        
        self.main_layout = QVBoxLayout(self)
        
        # 顶部：控制面板
        control_group = QGroupBox("2D-COS 控制参数")
        control_layout = QFormLayout(control_group)
        
        # ROI 波数范围
        roi_layout = QHBoxLayout()
        self.roi_min_input = QLineEdit()
        self.roi_min_input.setPlaceholderText("例如: 800")
        self.roi_max_input = QLineEdit()
        self.roi_max_input.setPlaceholderText("例如: 1800")
        roi_layout.addWidget(QLabel("ROI Min (cm⁻¹):"))
        roi_layout.addWidget(self.roi_min_input)
        roi_layout.addWidget(QLabel("ROI Max (cm⁻¹):"))
        roi_layout.addWidget(self.roi_max_input)
        roi_layout.addStretch()
        control_layout.addRow("ROI 波数范围:", roi_layout)
        
        # 等高线层数
        self.contour_levels_spin = QSpinBox()
        self.contour_levels_spin.setRange(-999999999, 999999999)
        self.contour_levels_spin.setValue(20)
        control_layout.addRow("等高线层数:", self.contour_levels_spin)
        
        # 参考峰选择下拉框
        self.ref_peak_combo = QComboBox()
        self.ref_peak_combo.setToolTip("选择参考峰（用于灵敏度分析）")
        self.ref_peak_combo.currentIndexChanged.connect(self._on_ref_peak_changed)
        control_layout.addRow("参考峰选择:", self.ref_peak_combo)
        
        # 噪声阈值调整控件
        self.noise_threshold_spin = SmartDoubleSpinBox()
        self.noise_threshold_spin.setRange(-999999999.0, 999999999.0)
        self.noise_threshold_spin.setDecimals(15)
        self.noise_threshold_spin.setValue(0.01)
        self.noise_threshold_spin.setToolTip("调整噪声阈值（相对于最大相关强度），用于过滤噪声。降低阈值可以看到更多'杂峰'，但可能包含噪声。")
        self.noise_threshold_spin.valueChanged.connect(self._on_threshold_changed)
        control_layout.addRow("噪声阈值 (推荐: 0.01):", self.noise_threshold_spin)
        
        # 纯甘氨酸参考光谱选择
        ref_spectrum_layout = QHBoxLayout()
        self.ref_spectrum_path_label = QLabel("未选择")
        self.ref_spectrum_path_label.setStyleSheet("color: gray; font-style: italic;")
        self.btn_load_ref_spectrum = QPushButton("加载参考光谱")
        self.btn_load_ref_spectrum.setToolTip("选择纯甘氨酸光谱文件（.txt或.csv），用于峰值匹配验证")
        self.btn_load_ref_spectrum.clicked.connect(self._load_reference_spectrum)
        self.btn_clear_ref_spectrum = QPushButton("清除")
        self.btn_clear_ref_spectrum.clicked.connect(self._clear_reference_spectrum)
        ref_spectrum_layout.addWidget(self.btn_load_ref_spectrum)
        ref_spectrum_layout.addWidget(self.btn_clear_ref_spectrum)
        ref_spectrum_layout.addWidget(self.ref_spectrum_path_label)
        ref_spectrum_layout.addStretch()
        control_layout.addRow("参考光谱 (纯甘氨酸):", ref_spectrum_layout)
        
        # 匹配容差设置
        self.match_tolerance_spin = SmartDoubleSpinBox()
        self.match_tolerance_spin.setRange(-999999999.0, 999999999.0)
        self.match_tolerance_spin.setDecimals(15)
        self.match_tolerance_spin.setValue(5.0)
        self.match_tolerance_spin.setToolTip("峰值匹配容差（cm⁻¹），默认±5.0 cm⁻¹")
        self.match_tolerance_spin.valueChanged.connect(self._on_match_tolerance_changed)
        control_layout.addRow("匹配容差 (cm⁻¹):", self.match_tolerance_spin)
        
        # 高斯平滑选项（用于去除条纹伪影）
        self.gaussian_smooth_check = QCheckBox("启用高斯平滑（去除条纹伪影）")
        self.gaussian_smooth_check.setChecked(False)
        self.gaussian_smooth_check.setToolTip("对2D异步谱矩阵进行轻微高斯平滑，使等高线更圆润平滑")
        self.gaussian_smooth_sigma_spin = SmartDoubleSpinBox()
        self.gaussian_smooth_sigma_spin.setRange(-999999999.0, 999999999.0)
        self.gaussian_smooth_sigma_spin.setDecimals(15)
        self.gaussian_smooth_sigma_spin.setValue(0.8)
        self.gaussian_smooth_sigma_spin.setToolTip("高斯平滑的sigma参数（推荐0.5-1.0）")
        gaussian_layout = QHBoxLayout()
        gaussian_layout.addWidget(self.gaussian_smooth_check)
        gaussian_layout.addWidget(QLabel("Sigma:"))
        gaussian_layout.addWidget(self.gaussian_smooth_sigma_spin)
        gaussian_layout.addStretch()
        gaussian_widget = QWidget()
        gaussian_widget.setLayout(gaussian_layout)
        control_layout.addRow("高斯平滑:", gaussian_widget)
        
        # 最敏感有机物波段文本框
        self.sensitive_bands_text = QTextEdit()
        self.sensitive_bands_text.setReadOnly(True)
        self.sensitive_bands_text.setMaximumHeight(80)
        self.sensitive_bands_text.setPlaceholderText("点击'更新绘图'后将在此显示最敏感有机物波段建议...")
        control_layout.addRow("Adaptive OBS 波段建议:", self.sensitive_bands_text)
        
        # 更新按钮
        self.update_button = QPushButton("更新绘图")
        self.update_button.setStyleSheet("font-size: 12pt; padding: 8px; background-color: #2196F3; color: white; font-weight: bold;")
        self.update_button.clicked.connect(self.update_plot)
        control_layout.addRow(self.update_button)
        
        self.main_layout.addWidget(control_group)
        
        # 中心：图表区域（两个子图并排）
        # 创建 Figure 和两个子图
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(14, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)
        
        # 存储数据
        self.X_matrix = None  # 扰动矩阵 (n_groups, n_wavenumbers)
        self.wavenumbers = None
        self.group_names = None
        
        # 存储峰值信息（用于参考峰选择）
        self.peak_indices = None  # 峰值在wavenumbers_roi中的索引
        self.peak_wavenumbers = None  # 峰值对应的波数
        self.peak_intensities = None  # 峰值强度
        self.current_wavenumbers_roi = None  # 当前ROI的波数数组
        self.current_Phi = None  # 当前同步相关矩阵
        self.current_Psi = None  # 当前异步相关矩阵
        self.current_y_representative = None  # 当前代表性光谱
        self.selected_ref_peak_wavenumber = None  # 用户选择的参考峰波数（用于恢复选择）
        
        # 参考光谱数据（纯甘氨酸）
        self.ref_spectrum_path = None  # 参考光谱文件路径
        self.ref_spectrum_x = None  # 参考光谱波数
        self.ref_spectrum_y = None  # 参考光谱强度
        self.ref_spectrum_peaks = None  # 参考光谱峰值（波数列表）
        self.matched_peaks = []  # 匹配的峰值列表
        self.unmatched_peaks = []  # 未匹配的峰值列表
        
        # 边缘分布图窗口
        self._marginal_plot_window = None
        
    def set_data(self, X_matrix, wavenumbers, group_names):
        """
        设置数据
        
        Args:
            X_matrix: 扰动矩阵 (n_groups, n_wavenumbers)，每行是一个组的平均光谱
            wavenumbers: 波数数组
            group_names: 组名列表
        """
        self.X_matrix = X_matrix
        self.wavenumbers = wavenumbers
        self.group_names = group_names
        
        # 自动设置 ROI 范围（如果未设置）
        if not self.roi_min_input.text() or not self.roi_max_input.text():
            self.roi_min_input.setText(str(np.min(wavenumbers)))
            self.roi_max_input.setText(str(np.max(wavenumbers)))
        
        # 自动更新绘图
        self.update_plot()
    
    def update_plot(self):
        """更新2D-COS图 - 显示同步和异步两个并排的图"""
        if self.X_matrix is None or self.wavenumbers is None:
            return
        
        try:
            # 获取 ROI 范围
            raw_min = float(self.roi_min_input.text()) if self.roi_min_input.text() else np.min(self.wavenumbers)
            raw_max = float(self.roi_max_input.text()) if self.roi_max_input.text() else np.max(self.wavenumbers)
            roi_min = min(raw_min, raw_max)
            roi_max = max(raw_min, raw_max)
            
            # 应用 ROI 截断
            mask = (self.wavenumbers >= roi_min) & (self.wavenumbers <= roi_max)
            wavenumbers_roi = self.wavenumbers[mask]
            X_roi = self.X_matrix[:, mask]
            
            if X_roi.shape[1] == 0:
                QMessageBox.warning(self, "错误", "ROI范围内无数据点")
                return
            
            # 计算动态光谱
            y_mean = np.mean(X_roi, axis=0)
            y_tilde = X_roi - y_mean[np.newaxis, :]
            n = y_tilde.shape[0]
            if n < 2:
                QMessageBox.warning(self, "错误", "组数不足（至少需要2组）")
                return
            
            Phi = (1.0 / (n - 1)) * y_tilde.T @ y_tilde  # 同步
            
            # Hilbert 变换计算异步
            from scipy.signal import hilbert
            from scipy.ndimage import gaussian_filter
            y_hilbert = hilbert(y_tilde, axis=0)
            N = np.imag(y_hilbert)
            Psi = (1.0 / (n - 1)) * y_tilde.T @ N
            
            # 高斯平滑 (去噪)
            if self.gaussian_smooth_check.isChecked():
                sigma = self.gaussian_smooth_sigma_spin.value()
                Psi = gaussian_filter(Psi, sigma=sigma)
            
            # 清除现有图
            self.figure.clear()
            # 删除 set_size_inches，不要在 update 中强制重设尺寸，会让布局跳动
            
            # 检查是否有参考光谱
            has_ref_spectrum = (self.ref_spectrum_path is not None and 
                               self.ref_spectrum_x is not None and 
                               self.ref_spectrum_peaks is not None)
            
            # 没有参考光谱时，显示两个并排的图（同步和异步）
            ax1 = self.figure.add_subplot(121)  # 同步图
            ax2 = self.figure.add_subplot(122, sharex=ax1, sharey=ax1)  # 异步图，共享轴以实现同步缩放
            
            # 确保两个图都是正方形 (使用 box_aspect 代替 aspect='equal')
            # set_box_aspect(1) 强制绘图区域为正方形，但允许数据自由缩放，解决放大被切掉的问题
            ax1.set_box_aspect(1)
            ax2.set_box_aspect(1)
            
            # 等高线层数
            n_levels = self.contour_levels_spin.value()
            
            # 绘制同步图
            vmax_sync = np.max(np.abs(Phi))
            vmin_sync = -vmax_sync
            levels_sync = np.linspace(vmin_sync, vmax_sync, n_levels)
            
            contour1 = ax1.contourf(wavenumbers_roi, wavenumbers_roi, Phi, 
                                   levels=levels_sync, cmap='RdBu_r', extend='both')
            ax1.contour(wavenumbers_roi, wavenumbers_roi, Phi, 
                       levels=levels_sync, colors='black', linewidths=0.5, alpha=0.3)
            
            # 绘制主对角线（x=y）
            ax1.plot([roi_min, roi_max], [roi_min, roi_max], 
                    'k--', linewidth=1.5, label='Diagonal (Auto-peaks)')
            
            ax1.set_xlabel("Wavenumber (cm⁻¹)", fontfamily='Times New Roman', fontsize=16)
            ax1.set_ylabel("Wavenumber (cm⁻¹)", fontfamily='Times New Roman', fontsize=16)
            ax1.set_title("Synchronous Map (Φ)", fontfamily='Times New Roman', fontsize=18, fontweight='bold', pad=20)
            
            # 添加颜色条
            cbar1 = self.figure.colorbar(contour1, ax=ax1)
            cbar1.set_label("Correlation Intensity", fontfamily='Times New Roman', fontsize=12)
            cbar1.ax.tick_params(labelsize=12)
            for label in cbar1.ax.get_yticklabels():
                label.set_fontfamily('Times New Roman')
            
            # 应用发表级别样式
            self._apply_publication_style(ax1)
            
            # 绘制异步图
            vmax_async = np.max(np.abs(Psi))
            vmin_async = -vmax_async
            levels_async = np.linspace(vmin_async, vmax_async, n_levels)
            
            contour2 = ax2.contourf(wavenumbers_roi, wavenumbers_roi, Psi, 
                                   levels=levels_async, cmap='seismic', extend='both')
            ax2.contour(wavenumbers_roi, wavenumbers_roi, Psi, 
                       levels=levels_async, colors='black', linewidths=0.5, alpha=0.3)
            
            # 绘制主对角线（x=y）
            ax2.plot([roi_min, roi_max], [roi_min, roi_max], 
                    'k--', linewidth=1.5, alpha=0.5)
            
            ax2.set_xlabel("Wavenumber (cm⁻¹)", fontfamily='Times New Roman', fontsize=16)
            ax2.set_ylabel("Wavenumber (cm⁻¹)", fontfamily='Times New Roman', fontsize=16)
            ax2.set_title("Asynchronous Map (Ψ)", 
                        fontfamily='Times New Roman', fontsize=18, fontweight='bold', pad=20)
            
            # 添加颜色条
            cbar2 = self.figure.colorbar(contour2, ax=ax2)
            cbar2.set_label("Correlation Intensity", fontfamily='Times New Roman', fontsize=12)
            cbar2.ax.tick_params(labelsize=12)
            for label in cbar2.ax.get_yticklabels():
                label.set_fontfamily('Times New Roman')
            
            # 应用发表级别样式
            self._apply_publication_style(ax2)
            
            # === 坐标轴反转处理 ===
            # 检查波数是否需要反转 (光谱习惯从大到小)
            if wavenumbers_roi[0] < wavenumbers_roi[-1]:
                # 如果数据是升序，且用户习惯降序(通常IR/Raman是降序)，可以在这里反转
                # 或者根据您的习惯，这里保持原样。如果需要反转：
                ax1.invert_xaxis()
                ax1.invert_yaxis()
                # ax2 会自动反转，因为 sharex/sharey
            
            # 保存当前数据供参考峰选择使用
            y_representative = np.max(X_roi, axis=0) if X_roi.shape[0] > 0 else None
            self.current_wavenumbers_roi = wavenumbers_roi
            self.current_Phi = Phi
            self.current_Psi = Psi
            self.current_y_representative = y_representative
            
            if y_representative is not None:
                self._detect_peaks_and_update_combo(wavenumbers_roi, y_representative)
                
                current_idx = self.ref_peak_combo.currentIndex()
                ref_item_data = self.ref_peak_combo.itemData(current_idx) if current_idx >= 0 else None
                
                self._identify_sensitive_organic_bands(
                    wavenumbers_roi, Phi, Psi, y_representative, ref_item_data
                )
            else:
                # 清空下拉框
                self.ref_peak_combo.clear()
                self.peak_indices = None
                self.peak_wavenumbers = None
                self.peak_intensities = None
                self.matched_peaks = []
                self.unmatched_peaks = []
                self._identify_sensitive_organic_bands(wavenumbers_roi, Phi, Psi, None)
            
            # === 布局调整 ===
            # 使用 tight_layout 配合 rect 参数，给标题留出空间，避免被切掉
            # 重要：在打开边缘分布图窗口之前完成布局调整，避免影响主窗口布局
            self.figure.tight_layout(rect=[0, 0.03, 1, 0.92])
            self.canvas.draw()
            
            # 如果有参考光谱，打开边缘分布图窗口（在布局调整之后）
            if has_ref_spectrum:
                self._open_marginal_plot_window(wavenumbers_roi, Phi, Psi, y_representative)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新2D-COS图时出错：{str(e)}")
            traceback.print_exc()
    
    def _detect_peaks_and_update_combo(self, wavenumbers, y_representative):
        """
        检测峰值并更新参考峰选择下拉框
        
        Args:
            wavenumbers: 波数数组
            y_representative: 代表性光谱数组
        """
        try:
            # 设置合理的 prominence 阈值（数据最大值的 5%）
            y_max = np.max(y_representative)
            prominence_threshold = y_max * 0.05
            
            # 使用 find_peaks 进行峰值检测
            peak_indices, properties = find_peaks(
                y_representative, 
                prominence=prominence_threshold
            )
            
            if len(peak_indices) == 0:
                self.ref_peak_combo.clear()
                self.peak_indices = None
                self.peak_wavenumbers = None
                self.peak_intensities = None
                return
            
            # 保存峰值信息
            self.peak_indices = peak_indices
            self.peak_wavenumbers = wavenumbers[peak_indices]
            self.peak_intensities = y_representative[peak_indices]
            
            # 按强度降序排序，选择前N个最强峰（例如前10个）
            sorted_indices = np.argsort(self.peak_intensities)[::-1]
            top_n = min(10, len(sorted_indices))  # 最多显示10个最强峰
            
            # 更新下拉框
            self.ref_peak_combo.blockSignals(True)  # 暂时阻止信号，避免触发更新
            self.ref_peak_combo.clear()
            
            # 记录要恢复的索引位置
            restore_index = 0  # 默认选择最强峰
            
            for i in range(top_n):
                idx_in_peak_array = sorted_indices[i]  # 在 peak_indices 数组中的位置
                wavenumber = self.peak_wavenumbers[idx_in_peak_array]
                intensity = self.peak_intensities[idx_in_peak_array]
                # 格式化显示：波数 (强度)
                display_text = f"{wavenumber:.1f} cm⁻¹ (强度: {intensity:.2f})"
                # 存储 idx_in_peak_array（在 peak_indices 中的索引位置）
                self.ref_peak_combo.addItem(display_text, idx_in_peak_array)
                
                # 如果这个峰的波数与之前选择的参考峰波数匹配（允许小误差），则恢复选择
                if (self.selected_ref_peak_wavenumber is not None and 
                    abs(wavenumber - self.selected_ref_peak_wavenumber) < 0.5):  # 0.5 cm⁻¹ 的容差
                    restore_index = i
            
            # 恢复之前的选择，或默认选择最强峰（索引0）
            self.ref_peak_combo.setCurrentIndex(restore_index)
            self.ref_peak_combo.blockSignals(False)
            
        except Exception as e:
            print(f"检测峰值时出错：{str(e)}")
            traceback.print_exc()
    
    def _load_reference_spectrum(self):
        """加载纯甘氨酸参考光谱文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择纯甘氨酸参考光谱文件", 
            "", 
            "光谱文件 (*.txt *.csv);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # 使用主程序的read_data方法读取文件（自动检测跳过行数和分隔符）
            if self.parent() and hasattr(self.parent(), 'read_data'):
                # 使用父窗口的read_data方法
                skip = -1  # 自动检测
                x, y = self.parent().read_data(file_path, skip)
            else:
                # 如果没有父窗口，使用DataController读取
                from src.ui.controllers.data_controller import DataController
                data_controller = DataController()
                x, y = data_controller.read_data(file_path, skip_rows=-1)
            
            if len(x) == 0 or len(y) == 0:
                QMessageBox.warning(self, "错误", "文件格式错误：无法读取数据")
                return
            
            # 检测峰值
            y_max = np.max(y)
            prominence_threshold = y_max * 0.05  # 5%阈值
            peak_indices, _ = find_peaks(y, prominence=prominence_threshold)
            peak_wavenumbers = x[peak_indices]
            
            # 保存数据
            self.ref_spectrum_path = file_path
            self.ref_spectrum_x = x
            self.ref_spectrum_y = y
            self.ref_spectrum_peaks = peak_wavenumbers
            
            # 更新UI
            self.ref_spectrum_path_label.setText(os.path.basename(file_path))
            self.ref_spectrum_path_label.setStyleSheet("color: green; font-weight: bold;")
            
            QMessageBox.information(
                self, 
                "成功", 
                f"已加载参考光谱\n检测到 {len(peak_wavenumbers)} 个峰值\n峰值位置: {', '.join([f'{p:.1f}' for p in peak_wavenumbers[:5]])}..."
            )
            
            # 如果已有2D-COS数据，重新绘图（会自动打开边缘分布图窗口）
            if self.current_Psi is not None:
                self.update_plot()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载参考光谱失败：{str(e)}")
            traceback.print_exc()
    
    def _open_marginal_plot_window(self, wavenumbers_roi, Phi, Psi, y_representative):
        """打开边缘分布图窗口（当有参考光谱时）"""
        if not hasattr(self, '_marginal_plot_window') or self._marginal_plot_window is None:
            self._marginal_plot_window = TwoDCOSMarginalPlotWindow(self)
        
        self._marginal_plot_window.set_data(
            wavenumbers_roi, Phi, Psi, y_representative,
            self.ref_spectrum_x, self.ref_spectrum_y,
            self.matched_peaks, self.unmatched_peaks
        )
        self._marginal_plot_window.show()
        self._marginal_plot_window.raise_()
        self._marginal_plot_window.activateWindow()
    
    def _clear_reference_spectrum(self):
        """清除参考光谱"""
        self.ref_spectrum_path = None
        self.ref_spectrum_x = None
        self.ref_spectrum_y = None
        self.ref_spectrum_peaks = None
        self.matched_peaks = []
        self.unmatched_peaks = []
        
        self.ref_spectrum_path_label.setText("未选择")
        self.ref_spectrum_path_label.setStyleSheet("color: gray; font-style: italic;")
        
        # 如果已有2D-COS数据，重新绘图
        if self.current_Psi is not None:
            self.update_plot()
    
    def _match_peaks(self, sensitive_peaks, reference_peaks, tolerance=5.0):
        """
        匹配敏感峰值与参考峰值
        
        Args:
            sensitive_peaks: 2D-COS检测到的敏感峰值列表（波数）
            reference_peaks: 参考光谱的峰值列表（波数）
            tolerance: 匹配容差（cm⁻¹）
        
        Returns:
            matched: 匹配的峰值列表
            unmatched: 未匹配的峰值列表
        """
        matched = []
        unmatched = []
        
        for sens_peak in sensitive_peaks:
            # 检查是否在任意一个参考峰的容差范围内
            is_match = any(abs(sens_peak - ref) <= tolerance for ref in reference_peaks)
            if is_match:
                matched.append(sens_peak)
            else:
                unmatched.append(sens_peak)
        
        return matched, unmatched
    
    def _on_match_tolerance_changed(self, value):
        """当匹配容差变化时的回调函数"""
        if (self.current_wavenumbers_roi is None or 
            self.current_Phi is None or self.current_Psi is None or 
            self.current_y_representative is None or self.peak_indices is None):
            return
        
        # 如果已加载参考光谱，重新绘图
        if self.ref_spectrum_peaks is not None:
            self.update_plot()
    
    def _on_threshold_changed(self, value):
        """
        当噪声阈值变化时的回调函数
        重新运行灵敏度分析
        """
        if (self.current_wavenumbers_roi is None or 
            self.current_Phi is None or self.current_Psi is None or 
            self.current_y_representative is None or self.peak_indices is None):
            return
        
        try:
            # 获取当前选定的参考峰索引
            current_idx = self.ref_peak_combo.currentIndex()
            if current_idx >= 0:
                item_data = self.ref_peak_combo.itemData(current_idx)
                if item_data is not None and 0 <= item_data < len(self.peak_indices):
                    self._identify_sensitive_organic_bands(
                        self.current_wavenumbers_roi, 
                        self.current_Phi, 
                        self.current_Psi, 
                        self.current_y_representative,
                        ref_peak_idx_in_peaks=item_data
                    )
        except Exception as e:
            print(f"更新阈值分析时出错：{str(e)}")
            traceback.print_exc()
    
    def _on_ref_peak_changed(self, index):
        """
        当参考峰选择变化时的回调函数
        重新运行灵敏度分析
        """
        if (index < 0 or self.current_wavenumbers_roi is None or 
            self.current_Phi is None or self.current_Psi is None or 
            self.current_y_representative is None or self.peak_indices is None):
            return
        
        try:
            # 获取选定的参考峰在 peak_indices 中的索引位置
            # itemData 存储的是在 peak_indices 数组中的索引位置
            item_data = self.ref_peak_combo.itemData(index)
            if item_data is not None and 0 <= item_data < len(self.peak_indices):
                # 保存当前选择的参考峰波数（用于下次更新时恢复）
                if self.peak_wavenumbers is not None and 0 <= item_data < len(self.peak_wavenumbers):
                    self.selected_ref_peak_wavenumber = self.peak_wavenumbers[item_data]
                
                # item_data 就是在 peak_indices 中的索引位置，直接使用
                ref_peak_idx_in_peaks = item_data
                self._identify_sensitive_organic_bands(
                    self.current_wavenumbers_roi, 
                    self.current_Phi, 
                    self.current_Psi, 
                    self.current_y_representative,
                    ref_peak_idx_in_peaks=ref_peak_idx_in_peaks
                )
        except Exception as e:
            print(f"更新参考峰分析时出错：{str(e)}")
            traceback.print_exc()
    
    def _identify_sensitive_organic_bands(self, wavenumbers, Phi, Psi, y_representative, ref_peak_idx_in_peaks=None):
        """
        识别和显示最敏感有机物波段
        
        基于 Noda 规则和峰值检测：
        1. 使用峰值检测识别所有峰
        2. 使用用户选择的参考峰（或默认最强峰）
        3. 应用 Noda 规则：如果 Φ(ω_i, ω_Ref) 和 Ψ(ω_i, ω_Ref) 同号，则 ω_i 早于 ω_Ref（更灵敏）
        4. 按 |Ψ| 排序输出
        
        Args:
            wavenumbers: 波数数组
            Phi: 同步相关矩阵 (n_wavenumbers, n_wavenumbers)
            Psi: 异步相关矩阵 (n_wavenumbers, n_wavenumbers)
            y_representative: 代表性光谱数组（用于峰值检测），如果为 None 则跳过
            ref_peak_idx_in_peaks: 参考峰在 peak_indices 中的索引（如果为 None，则使用最强峰）
        """
        try:
            # 如果没有代表性光谱，无法进行峰值检测
            if y_representative is None or len(y_representative) == 0:
                self.sensitive_bands_text.setText("无法进行峰值检测：缺少代表性光谱数据")
                return
            
            # A. 峰值识别 (Peak Picking)
            # 设置合理的 prominence 阈值（数据最大值的 5%）
            y_max = np.max(y_representative)
            prominence_threshold = y_max * 0.05
            
            # 使用 find_peaks 进行峰值检测
            peak_indices, properties = find_peaks(
                y_representative, 
                prominence=prominence_threshold
            )
            
            if len(peak_indices) == 0:
                self.sensitive_bands_text.setText("未找到峰值（请降低 prominence 阈值或检查数据）")
                return
            
            peak_wavenumbers = wavenumbers[peak_indices]
            peak_intensities = y_representative[peak_indices]
            
            # B. 确定参考峰 (Reference Peak)
            # 如果指定了参考峰索引，使用指定的；否则使用强度最大的峰
            if ref_peak_idx_in_peaks is not None and 0 <= ref_peak_idx_in_peaks < len(peak_indices):
                ref_idx_in_peaks = peak_indices[ref_peak_idx_in_peaks]
                ref_wavenumber = peak_wavenumbers[ref_peak_idx_in_peaks]
                ref_intensity = peak_intensities[ref_peak_idx_in_peaks]
            else:
                # 默认使用强度最大的峰
                max_intensity_idx = np.argmax(peak_intensities)
                ref_idx_in_peaks = peak_indices[max_intensity_idx]
                ref_wavenumber = peak_wavenumbers[max_intensity_idx]
                ref_intensity = peak_intensities[max_intensity_idx]
            
            # C. 应用 Noda 规则进行排序
            earlier_peaks = []  # 早于参考峰的峰（更灵敏）
            later_peaks = []   # 晚于参考峰的峰
            sensitive_peak_wavenumbers = []  # 所有敏感峰值的波数（用于匹配）
            
            for i, peak_idx in enumerate(peak_indices):
                if peak_idx == ref_idx_in_peaks:
                    continue  # 跳过参考峰本身
                
                # 读取同步强度和异步强度
                phi_val = Phi[peak_idx, ref_idx_in_peaks]
                psi_val = Psi[peak_idx, ref_idx_in_peaks]
                
                # 排除太小的值（噪声）- 从UI控件读取阈值
                threshold_factor = self.noise_threshold_spin.value()
                threshold = threshold_factor * max(np.max(np.abs(Phi)), np.max(np.abs(Psi)))
                
                if np.abs(phi_val) > threshold and np.abs(psi_val) > threshold:
                    wavenumber = peak_wavenumbers[i]
                    intensity_psi = psi_val
                    intensity_phi = phi_val
                    
                    # 收集所有敏感峰值（用于匹配）
                    sensitive_peak_wavenumbers.append(wavenumber)
                    
                    # Noda 规则修正：同号表示早于参考峰（更灵敏）
                    if phi_val * psi_val > 0:
                        # 同号：早于参考峰（更敏感）
                        earlier_peaks.append((wavenumber, intensity_psi, intensity_phi))
                    else:
                        # 异号：晚于参考峰
                        later_peaks.append((wavenumber, intensity_psi, intensity_phi))
            
            # D. 如果加载了参考光谱，进行峰值匹配
            if self.ref_spectrum_peaks is not None and len(sensitive_peak_wavenumbers) > 0:
                tolerance = self.match_tolerance_spin.value()
                self.matched_peaks, self.unmatched_peaks = self._match_peaks(
                    sensitive_peak_wavenumbers, 
                    self.ref_spectrum_peaks, 
                    tolerance
                )
            else:
                self.matched_peaks = []
                self.unmatched_peaks = []
            
            # D. 生成报告
            # 排序：按 |Ψ| 降序排列
            earlier_peaks.sort(key=lambda x: abs(x[1]), reverse=True)
            later_peaks.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # 格式化输出
            result_lines = []
            result_lines.append(f"参考峰 (Ref): {ref_wavenumber:.1f} cm⁻¹ (强度: {ref_intensity:.2f})")
            result_lines.append("")
            result_lines.append("-" * 50)
            result_lines.append("")
            
            # 如果进行了峰值匹配，显示匹配信息
            if self.ref_spectrum_peaks is not None:
                result_lines.append(f"【峰值匹配结果】(容差: ±{self.match_tolerance_spin.value():.1f} cm⁻¹)")
                result_lines.append(f"匹配峰值: {len(self.matched_peaks)} 个")
                if self.matched_peaks:
                    result_lines.append(f"  位置: {', '.join([f'{p:.1f}' for p in self.matched_peaks[:5]])}" + 
                                      (f"..." if len(self.matched_peaks) > 5 else ""))
                result_lines.append(f"未匹配峰值: {len(self.unmatched_peaks)} 个 (可能为噪声)")
                result_lines.append("")
            
            # 高灵敏度序列（早于参考峰）
            result_lines.append("【高灵敏度序列】(先于参考峰出现)")
            if earlier_peaks:
                for idx, (w, psi, phi) in enumerate(earlier_peaks, 1):
                    # 标记是否匹配
                    match_status = "✓" if w in self.matched_peaks else ("✗" if w in self.unmatched_peaks else "")
                    result_lines.append(f"{idx}. {w:.1f} cm⁻¹ (|Ψ|={abs(psi):.2f}) {match_status}" + 
                                      (f" - 最敏感" if idx == 1 else ""))
            else:
                result_lines.append("（无）")
            
            result_lines.append("")
            result_lines.append("【低灵敏度序列】(晚于参考峰出现)")
            if later_peaks:
                for idx, (w, psi, phi) in enumerate(later_peaks, 1):
                    # 标记是否匹配
                    match_status = "✓" if w in self.matched_peaks else ("✗" if w in self.unmatched_peaks else "")
                    result_lines.append(f"{idx}. {w:.1f} cm⁻¹ (|Ψ|={abs(psi):.2f}) {match_status}")
            else:
                result_lines.append("（无）")
            
            result_text = "\n".join(result_lines)
            self.sensitive_bands_text.setText(result_text)
            
        except Exception as e:
            self.sensitive_bands_text.setText(f"识别敏感波段时出错：{str(e)}")
            traceback.print_exc()
    
    def _apply_publication_style(self, ax):
        """应用发表级别绘图样式"""
        # 强制使用 Times New Roman
        font_family = 'Times New Roman'
        
        # 刻度设置：direction='in', top=True, right=True
        ax.tick_params(axis='both', which='major', 
                      direction='in',
                      length=8,
                      width=1.0,
                      labelsize=14,
                      top=True,
                      right=True)
        ax.tick_params(axis='both', which='minor',
                      direction='in',
                      length=4,
                      width=1.0,
                      top=True,
                      right=True)
        
        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(14)
        
        # 边框设置：linewidth=1.5
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_visible(True)
        
        # 标签字体
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
        ax.title.set_fontfamily(font_family)



class TwoDCOSMarginalPlotWindow(QDialog):
    """2D-COS 边缘分布图窗口 - 显示参考光谱和2D异步谱的组合图"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D-COS Marginal Plot (边缘分布图)")
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        # 初始窗口比例：纵轴:横轴 = 4:3，即高度:宽度 = 4:3，设置为 600x800（宽x高）
        self.resize(600, 800)
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(300, 400)
        
        self.main_layout = QVBoxLayout(self)
        
        # 创建 Figure 和画布
        # 画布尺寸也要调整为长方形，比例与窗口一致（高度:宽度 = 4:3）
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(4.5, 6), dpi=100)  # 高度:宽度 = 6:4.5 = 4:3
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)
        
        # 存储数据
        self.wavenumbers_roi = None
        self.Phi = None
        self.Psi = None
        self.y_representative = None
        self.ref_spectrum_x = None
        self.ref_spectrum_y = None
        self.matched_peaks = []
        self.unmatched_peaks = []
    
    def set_data(self, wavenumbers_roi, Phi, Psi, y_representative, 
                 ref_spectrum_x, ref_spectrum_y, matched_peaks, unmatched_peaks):
        """设置要绘制的数据"""
        self.wavenumbers_roi = wavenumbers_roi
        self.Phi = Phi
        self.Psi = Psi
        self.y_representative = y_representative
        self.ref_spectrum_x = ref_spectrum_x
        self.ref_spectrum_y = ref_spectrum_y
        self.matched_peaks = matched_peaks
        self.unmatched_peaks = unmatched_peaks
        self.update_plot()
    
    def update_plot(self):
        """更新边缘分布图 - 终极对齐修复版"""
        if self.wavenumbers_roi is None or self.Psi is None:
            return
        
        try:
            # 引入专门用于对齐边缘图的工具
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib.pyplot as plt
            
            # 清除现有图
            self.figure.clear()
            
            # 1. 确定物理范围（X轴：Min在左，MAX在右，与其他图一致）
            x_max = np.max(self.wavenumbers_roi)
            x_min = np.min(self.wavenumbers_roi)
            
            # 2. 先创建主图 (ax_main)
            # 111 代表全图，我们通过 divider 来切分它
            ax_main = self.figure.add_subplot(111)
            
            # 关键设置：强制正方形 (解决放大显示不全的问题)
            ax_main.set_box_aspect(1)
            
            # 3. 创建 Divider (分割器)
            # 这神奇的工具会锁定 ax_main 的位置，并基于它创建附属图
            divider = make_axes_locatable(ax_main)
            
            # --- A. 在顶部附着参考谱 (Top Plot) ---
            # size="30%" 表示高度占主图的30%，pad=0.15 是间距
            # sharex=ax_main 确保放大缩放完美同步
            ax_ref = divider.append_axes("top", size="30%", pad=0.15, sharex=ax_main)
            
            # --- B. 在右侧附着 Colorbar (Right Plot) ---
            # size="5%" 宽度，pad=0.15 间距
            ax_cbar = divider.append_axes("right", size="5%", pad=0.15)
            
            # ================= 绘图逻辑 =================
            
            # 1. 绘制顶部参考谱
            if self.ref_spectrum_x is not None and self.ref_spectrum_y is not None:
                ax_ref.plot(self.ref_spectrum_x, self.ref_spectrum_y, 'k-', linewidth=1.2)
                
                # 动态调整 Y 轴范围 (只基于当前 ROI)
                mask_roi = (self.ref_spectrum_x >= x_min) & (self.ref_spectrum_x <= x_max)
                if np.any(mask_roi):
                    y_subset = self.ref_spectrum_y[mask_roi]
                    if len(y_subset) > 0:
                        ymin, ymax = np.min(y_subset), np.max(y_subset)
                        yrange = ymax - ymin if ymax != ymin else 1.0
                        ax_ref.set_ylim(ymin - 0.05 * yrange, ymax + 0.2 * yrange)
                
                # 设置顶部标题 - 调整位置避免与光谱线重叠
                ax_ref.set_title("Reference: Pure Glycine", 
                                fontfamily='Times New Roman', fontweight='bold', fontsize=11,
                                loc='left', pad=8, y=1.02)  # y=1.02 将标题放在图的上方
            
            # 顶部图样式清理
            ax_ref.set_yticks([]) # 隐藏Y刻度
            plt.setp(ax_ref.get_xticklabels(), visible=False) # 隐藏X刻度数字(因为和下面共享)
            # 关键修复：保留左边框但设为透明，确保与主图左边框位置完全对齐
            ax_ref.spines['top'].set_visible(False)
            ax_ref.spines['right'].set_visible(False)
            # 不隐藏左边框，而是设为透明，这样位置计算会一致
            ax_ref.spines['left'].set_visible(True)
            ax_ref.spines['left'].set_color('none')  # 透明但保留位置
            ax_ref.spines['left'].set_linewidth(0)   # 线宽为0
            ax_ref.patch.set_alpha(0) # 透明背景
            
            # 2. 绘制主图 (2D 异步谱)
            # 强制设定坐标轴范围：Min在左，MAX在右（与其他图一致）
            ax_main.set_xlim(x_min, x_max)
            ax_main.set_ylim(x_min, x_max)
            
            # 绘制等高线
            vmax_async = np.max(np.abs(self.Psi))
            if vmax_async == 0: vmax_async = 1.0
            levels_async = np.linspace(-vmax_async, vmax_async, 20)
            
            contour = ax_main.contourf(self.wavenumbers_roi, self.wavenumbers_roi, self.Psi, 
                                   levels=levels_async, cmap='seismic', extend='both')
            
            # 绘制对角线
            ax_main.plot([x_min, x_max], [x_min, x_max], 'k--', linewidth=1.0, alpha=0.5)
            
            # 坐标轴标签
            font_label = {'family': 'Times New Roman', 'size': 14}
            ax_main.set_xlabel("Wavenumber ($cm^{-1}$)", fontdict=font_label)
            ax_main.set_ylabel("Wavenumber ($cm^{-1}$)", fontdict=font_label)
            
            ax_main.tick_params(axis='both', which='major', labelsize=12, direction='in', top=True, right=True)
            for label in ax_main.get_xticklabels() + ax_main.get_yticklabels():
                label.set_fontfamily('Times New Roman')
            
            # 3. 绘制 Colorbar
            cbar = self.figure.colorbar(contour, cax=ax_cbar)
            cbar.set_label("Correlation Intensity", fontfamily='Times New Roman', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            
            # 4. 总标题
            self.figure.suptitle("Asynchronous Map (Ψ) with Reference Matching", 
                                fontfamily='Times New Roman', fontsize=16, fontweight='bold', 
                                y=0.98)
            
            # 5. 绘制匹配标记 (同步绘制)
            if len(self.matched_peaks) > 0:
                match_color = '#006400'
                for peak in self.matched_peaks:
                    # 顶部 1D 谱 (垂直线)
                    ax_ref.axvline(x=peak, color=match_color, linestyle=':', linewidth=1.0, alpha=0.8)
                    
                    # 标注 (根据Y轴范围) - 调整位置避免与光谱线重叠
                    y_lim_ref = ax_ref.get_ylim()
                    y_range = y_lim_ref[1] - y_lim_ref[0]
                    # 将标签放在Y轴最大值上方一点，避免重叠
                    label_y_pos = y_lim_ref[1] + 0.05 * y_range
                    ax_ref.text(peak, label_y_pos, f"{peak:.0f}", 
                               color=match_color, ha='center', va='bottom', fontsize=8, rotation=90,
                               fontfamily='Times New Roman', fontweight='bold')
                    
                    # 主图 2D 谱 (十字线)
                    ax_main.axvline(x=peak, color=match_color, linestyle=':', linewidth=0.8, alpha=0.6)
                    ax_main.axhline(y=peak, color=match_color, linestyle=':', linewidth=0.8, alpha=0.6)

            # 布局调整：给左边留出足够空间显示 "Wavenumber" 和刻度
            # 因为使用了 axes_grid1，这里的 adjust 调整的是整个组合块的位置
            # 画布缩小后，使用更紧凑的边距，增加右边距避免颜色条标签被裁剪
            self.figure.subplots_adjust(left=0.15, right=0.92, top=0.90, bottom=0.12)
            
            # 关键修复：在所有绘制完成后，强制对齐顶部图和主图的左边框
            # 获取主图的左边框位置（包括Y轴标签和刻度的空间）
            main_pos = ax_main.get_position()
            ref_pos = ax_ref.get_position()
            # 强制设置顶部图的左边框位置与主图完全一致（x0和width相同）
            ax_ref.set_position([main_pos.x0, ref_pos.y0, main_pos.width, ref_pos.height])
            
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新边缘分布图时出错：{str(e)}")
            traceback.print_exc()



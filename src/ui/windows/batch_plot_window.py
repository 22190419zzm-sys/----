"""
批量绘图窗口：为每个txt/csv光谱文件配备对应的png镜下光学图
使用Qt画板，复用主窗口的绘图逻辑和参数设置
支持RRUFF库加载和峰值匹配识别
"""
import os
import glob
import traceback
import warnings
import json
import hashlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from PIL import Image

from PyQt6.QtCore import Qt, QSettings, QPoint
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox, QScrollArea, QWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QApplication, QMenu, QDialogButtonBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QFormLayout, QProgressDialog
)

from src.core.rruff_loader import RRUFFLibraryLoader, PeakMatcher
from src.core.rruff_database import RRUFFDatabase
from src.ui.canvas import MplCanvas
from src.ui.controllers.data_controller import DataController
from src.core.preprocessor import DataPreProcessor
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class BatchPlotWindow(QDialog):
    """批量绘图窗口：为每个txt/csv文件配备对应的png图片"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Plot - Spectrum with Microscopy Image")
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        
        self.parent_window = parent  # 主窗口引用
        self.settings = QSettings("GTLab", "SpectraPro_v4")
        
        # 数据存储
        # 兼容历史命名：这里保存“光谱数据文件”列表（支持 .txt / .csv）
        self.txt_files = []
        self.png_files = {}  # {txt_basename: png_path}
        self.spectra_data = {}  # {txt_basename: {'x': x, 'y': y, 'peaks': peaks}}
        self.rruff_loader = None  # RRUFF库加载器
        self.peak_matcher = PeakMatcher(tolerance=5.0)
        self.data_controller = DataController()
        self.rruff_database = RRUFFDatabase()  # RRUFF数据库管理器
        self.auto_db_mode = True  # 自动数据库模式（根据预处理参数自动选择）
        
        # 每个谱图的单独标准库排除列表
        self.spectrum_exclusions = {}  # {txt_basename: [excluded_names]}
        
        # 绘图窗口字典 {txt_basename: MplPlotWindow}
        self.plot_windows = {}
        
        # RRUFF匹配相关
        self.rruff_match_results = {}  # {txt_basename: [match_results]}
        self.rruff_combination_results = {}  # {txt_basename: [combination_results]}
        self.selected_rruff_spectra = {}  # {txt_basename: set([rruff_names])}
        self.selected_rruff_combinations = {}  # {txt_basename: [{'phases': [...], 'ratios': [...]}]}
        
        # 匹配结果缓存（本次运行期间）
        self._match_cache = {}  # {cache_key: {'single': [...], 'combo': [...]}}
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI界面"""
        main_layout = QVBoxLayout(self)
        
        # 顶部控制栏
        control_layout = QHBoxLayout()
        
        # 文件夹选择
        self.folder_label = QLabel("Folder: Not selected")
        self.btn_select_folder = QPushButton("Select Folder")
        self.btn_select_folder.clicked.connect(self.select_folder)
        
        # RRUFF库选择
        self.rruff_label = QLabel("RRUFF Library: Not loaded")
        self.btn_select_rruff = QPushButton("Load RRUFF Library")
        self.btn_select_rruff.clicked.connect(self.select_rruff_library)
        self.btn_select_db = QPushButton("选择数据库")
        self.btn_select_db.clicked.connect(self.select_database)
        self.auto_db_check = QCheckBox("自动模式")
        self.auto_db_check.setChecked(True)
        self.auto_db_check.setToolTip("根据预处理参数自动选择对应的数据库")
        self.auto_db_check.stateChanged.connect(self._on_auto_db_mode_changed)
        
        # 扫描按钮
        self.btn_scan = QPushButton("Scan Files")
        self.btn_scan.clicked.connect(self.scan_files)
        self.btn_scan.setEnabled(False)
        
        # 批量导出按钮
        self.btn_export_all = QPushButton("Export All as PNG")
        self.btn_export_all.clicked.connect(self.export_all_plots)
        self.btn_export_all.setEnabled(False)
        
        control_layout.addWidget(self.folder_label)
        control_layout.addWidget(self.btn_select_folder)
        control_layout.addWidget(self.rruff_label)
        control_layout.addWidget(self.btn_select_rruff)
        control_layout.addWidget(self.btn_select_db)
        control_layout.addWidget(self.auto_db_check)
        control_layout.addWidget(self.btn_scan)
        control_layout.addWidget(self.btn_export_all)
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)
        
        # 文件列表和绘图区域
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：文件列表（可以无限拉伸）
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距
        
        file_label = QLabel("Files:")
        file_label.setStyleSheet("font-size: 9pt; font-weight: bold;")
        left_layout.addWidget(file_label)
        
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # 支持Ctrl+点击多选
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_file_context_menu)
        left_layout.addWidget(self.file_list)
        
        # 全局RRUFF库排除列表
        left_layout.addWidget(QLabel("Global RRUFF Exclusions:"))
        self.global_exclusion_list = QListWidget()
        self.global_exclusion_list.setMaximumHeight(150)
        left_layout.addWidget(self.global_exclusion_list)
        
        # RRUFF匹配按钮
        self.btn_rruff_match = QPushButton("🔍 匹配RRUFF光谱")
        self.btn_rruff_match.setStyleSheet("font-size: 10pt; padding: 6px; background-color: #FF5722; color: white; font-weight: bold;")
        self.btn_rruff_match.clicked.connect(self.match_rruff_spectra)
        self.btn_rruff_match.setEnabled(False)
        left_layout.addWidget(self.btn_rruff_match)

        # 自动RRUFF匹配开关（单文件绘图时自动进行匹配）
        self.auto_rruff_match_check = QCheckBox("自动匹配RRUFF光谱", checked=True)
        self.auto_rruff_match_check.setToolTip("勾选后，每次点击单个文件绘图时自动执行RRUFF光谱匹配。")
        left_layout.addWidget(self.auto_rruff_match_check)
        
        # 多物相组合匹配按钮
        self.btn_rruff_combination_match = QPushButton("🔗 多物相组合匹配")
        self.btn_rruff_combination_match.setStyleSheet("font-size: 10pt; padding: 6px; background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_rruff_combination_match.clicked.connect(self.match_rruff_combination)
        self.btn_rruff_combination_match.setEnabled(False)
        self.btn_rruff_combination_match.setToolTip("将多个RRUFF光谱按比例组合来匹配查询光谱")
        left_layout.addWidget(self.btn_rruff_combination_match)
        
        # RRUFF匹配结果列表
        left_layout.addWidget(QLabel("RRUFF匹配结果 (双击添加，Ctrl+点击叠加):"))
        self.rruff_match_list = QListWidget()
        self.rruff_match_list.setMaximumHeight(200)
        self.rruff_match_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # 支持Ctrl+点击多选
        self.rruff_match_list.itemDoubleClicked.connect(self.on_rruff_item_double_clicked)
        self.rruff_match_list.itemClicked.connect(self.on_rruff_item_clicked)  # 使用itemClicked检测Ctrl键
        self.rruff_match_list.itemSelectionChanged.connect(self.on_rruff_selection_changed)
        left_layout.addWidget(self.rruff_match_list)

        # RRUFF 结果总览按钮
        self.btn_rruff_summary = QPushButton("RRUFF 匹配结果总览")
        self.btn_rruff_summary.setStyleSheet("font-size: 9pt; padding: 4px; background-color: #607D8B; color: white;")
        self.btn_rruff_summary.clicked.connect(self.open_rruff_summary_window)
        left_layout.addWidget(self.btn_rruff_summary)
        
        # RRUFF参考线设置
        rruff_ref_lines_group = QGroupBox("RRUFF参考线设置")
        rruff_ref_lines_layout = QFormLayout(rruff_ref_lines_group)
        
        self.rruff_ref_lines_enabled_check = QCheckBox("启用RRUFF匹配参考线", checked=True)
        self.rruff_ref_lines_enabled_check.stateChanged.connect(self._on_rruff_ref_lines_enabled_changed)
        rruff_ref_lines_layout.addRow(self.rruff_ref_lines_enabled_check)
        
        # 匹配容差参数
        self.rruff_match_tolerance_spin = QDoubleSpinBox()
        self.rruff_match_tolerance_spin.setRange(0.1, 100.0)
        self.rruff_match_tolerance_spin.setDecimals(1)
        self.rruff_match_tolerance_spin.setValue(5.0)
        self.rruff_match_tolerance_spin.setSingleStep(0.1)
        self.rruff_match_tolerance_spin.setToolTip("峰值匹配容差（cm⁻¹）：两个峰值位置的距离小于此值时认为匹配。值越大匹配的峰值越多。对于自身匹配，建议设置为较大值（如10-20）以确保100%匹配。默认5.0 cm⁻¹")
        self.rruff_match_tolerance_spin.valueChanged.connect(self._on_rruff_tolerance_changed)
        rruff_ref_lines_layout.addRow("匹配容差 (cm⁻¹):", self.rruff_match_tolerance_spin)
        
        # 参考线偏移设置（用于批量绘图窗口）
        self.rruff_ref_line_offset_spin = QDoubleSpinBox()
        self.rruff_ref_line_offset_spin.setRange(-999999999.0, 999999999.0)
        self.rruff_ref_line_offset_spin.setDecimals(15)
        self.rruff_ref_line_offset_spin.setValue(0.0)
        self.rruff_ref_line_offset_spin.setToolTip("参考线偏移：用于批量绘图窗口中分离不同RRUFF光谱的Y轴偏移量。与匹配度无关，仅用于视觉分离。")
        rruff_ref_lines_layout.addRow("参考线偏移:", self.rruff_ref_line_offset_spin)
        
        # 过滤同一物相的不同变种（默认开启）
        self.rruff_filter_variants_check = QCheckBox("过滤同一物相变种", checked=True)
        self.rruff_filter_variants_check.setToolTip(
            "例如 talc-1 / talc-2 / talc-3 视为同一矿物 talc 的不同变种，组合匹配时每个矿物只允许出现一次。"
        )
        rruff_ref_lines_layout.addRow(self.rruff_filter_variants_check)

        # 组合匹配显示模式
        self.rruff_combination_as_single_check = QCheckBox("组合匹配显示为整体光谱", checked=False)
        self.rruff_combination_as_single_check.setToolTip("勾选：组合匹配显示为一条组合光谱；取消：组合匹配的各个物相分别显示为独立谱线")
        rruff_ref_lines_layout.addRow(self.rruff_combination_as_single_check)
        
        left_layout.addWidget(rruff_ref_lines_group)
        
        splitter.addWidget(left_widget)
        
        # 右侧：绘图区域（使用Qt画板）
        self.plot_area = QScrollArea()
        self.plot_area.setWidgetResizable(True)
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_area.setWidget(self.plot_widget)
        splitter.addWidget(self.plot_area)
        
        # 设置splitter的比例（左侧可以无限拉伸）
        splitter.setStretchFactor(0, 1)  # 左侧可以拉伸
        splitter.setStretchFactor(1, 3)   # 右侧拉伸更多
        splitter.setSizes([200, 1000])  # 初始大小：左侧200，右侧1000
        
        main_layout.addWidget(splitter)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # RRUFF 结果总览窗口（懒加载）
        self.rruff_summary_window = None
    
    def get_parent_plot_params(self):
        """从主窗口获取绘图参数"""
        if not self.parent_window:
            return None
        
        try:
            # 物理截断值
            x_min_phys = self.parent_window._parse_optional_float(
                self.parent_window.x_min_phys_input.text()
            )
            x_max_phys = self.parent_window._parse_optional_float(
                self.parent_window.x_max_phys_input.text()
            )
            
            # 从面板获取配置（如果可用）
            config = None
            ps = None
            if hasattr(self.parent_window, 'publication_style_panel') and self.parent_window.publication_style_panel:
                config = self.parent_window.publication_style_panel.get_config()
                ps = config.publication_style
            elif not ps:
                # 从配置管理器获取
                from src.core.plot_config_manager import PlotConfigManager
                config_manager = PlotConfigManager()
                config = config_manager.get_config()
                ps = config.publication_style
            
            # 收集参数（复用主窗口的run_plot_logic逻辑）
            params = {
                # 模式与全局
                'plot_mode': self.parent_window.plot_mode_combo.currentText(),
                'show_y_values': self.parent_window.show_y_val_check.isChecked(),
                'is_derivative': self.parent_window.derivative_check.isChecked(),
                'x_axis_invert': self.parent_window.x_axis_invert_check.isChecked(),
                'global_stack_offset': self.parent_window.global_stack_offset_spin.value(),
                'global_scale_factor': self.parent_window.global_y_scale_factor_spin.value(),
                'plot_style': self.parent_window.plot_style_combo.currentText(),
                
                # 标题和轴标签（从面板获取）
                'main_title_text': ps.title_text if ps else "",
                'main_title_fontsize': ps.title_fontsize if ps else 18,
                'main_title_pad': ps.title_pad if ps else 10.0,
                'main_title_show': ps.title_show if ps else True,
                'xlabel_text': ps.xlabel_text if ps else r"Wavenumber ($\mathrm{cm^{-1}}$)",
                'ylabel_text': ps.ylabel_text if ps else "Intensity",
                'xlabel_fontsize': ps.xlabel_fontsize if ps else 20,
                'xlabel_pad': ps.xlabel_pad if ps else 10.0,
                'xlabel_show': ps.xlabel_show if ps else True,
                'ylabel_fontsize': ps.ylabel_fontsize if ps else 20,
                'ylabel_pad': ps.ylabel_pad if ps else 10.0,
                'ylabel_show': ps.ylabel_show if ps else True,
                
                # 预处理
                'skip_rows': self.parent_window.skip_rows_spin.value(),
                'qc_enabled': self.parent_window.qc_check.isChecked(),
                'qc_threshold': self.parent_window.qc_threshold_spin.value(),
                'is_baseline_als': self.parent_window.baseline_als_check.isChecked(),
                'als_lam': self.parent_window.lam_spin.value(),
                'als_p': self.parent_window.p_spin.value(),
                'is_baseline': False,
                'baseline_points': 50,
                'baseline_poly': 3,
                'is_smoothing': self.parent_window.smoothing_check.isChecked(),
                'smoothing_window': self.parent_window.smoothing_window_spin.value(),
                'smoothing_poly': self.parent_window.smoothing_poly_spin.value(),
                'normalization_mode': self.parent_window.normalization_combo.currentText(),
                
                # Bose-Einstein
                'is_be_correction': self.parent_window.be_check.isChecked(),
                'be_temp': self.parent_window.be_temp_spin.value(),
                
                # 全局动态变换
                'global_transform_mode': self.parent_window.global_transform_combo.currentText(),
                'global_log_base': self.parent_window.global_log_base_combo.currentText(),
                'global_log_offset': self.parent_window.global_log_offset_spin.value(),
                'global_sqrt_offset': self.parent_window.global_sqrt_offset_spin.value(),
                'global_y_offset': self.parent_window.global_y_offset_spin.value() if hasattr(self.parent_window, 'global_y_offset_spin') else 0.0,
                
                # 峰值检测
                'peak_detection_enabled': self.parent_window.peak_check.isChecked(),
                'peak_height_threshold': self.parent_window.peak_height_spin.value(),
                'peak_distance_min': self.parent_window.peak_distance_spin.value(),
                'peak_prominence': self.parent_window.peak_prominence_spin.value(),
                'peak_width': self.parent_window.peak_width_spin.value(),
                'peak_wlen': self.parent_window.peak_wlen_spin.value(),
                'peak_rel_height': self.parent_window.peak_rel_height_spin.value(),
                'peak_show_label': self.parent_window.peak_show_label_check.isChecked(),
                'peak_label_font': self.parent_window.peak_label_font_combo.currentText(),
                'peak_label_size': self.parent_window.peak_label_size_spin.value(),
                'peak_label_color': self.parent_window.peak_label_color_input.text().strip() or 'black',
                'peak_label_bold': self.parent_window.peak_label_bold_check.isChecked(),
                'peak_label_rotation': self.parent_window.peak_label_rotation_spin.value(),
                'peak_marker_shape': self.parent_window.peak_marker_shape_combo.currentText(),
                'peak_marker_size': self.parent_window.peak_marker_size_spin.value(),
                'peak_marker_color': self.parent_window.peak_marker_color_input.text().strip() or '',
                'vertical_lines': self.parent_window.parse_list_input(self.parent_window.vertical_lines_input.toPlainText()) if hasattr(self.parent_window, 'vertical_lines_input') else [],
                'vertical_line_color': self.parent_window.vertical_line_color_input.text().strip() or 'gray' if hasattr(self.parent_window, 'vertical_line_color_input') else 'gray',
                'vertical_line_width': self.parent_window.vertical_line_width_spin.value() if hasattr(self.parent_window, 'vertical_line_width_spin') else 0.8,
                'vertical_line_style': self.parent_window.vertical_line_style_combo.currentText() if hasattr(self.parent_window, 'vertical_line_style_combo') else ':',
                'vertical_line_alpha': self.parent_window.vertical_line_alpha_spin.value() if hasattr(self.parent_window, 'vertical_line_alpha_spin') else 0.7,
                'rruff_ref_lines_enabled': self.rruff_ref_lines_enabled_check.isChecked() if hasattr(self, 'rruff_ref_lines_enabled_check') else True,
                'rruff_ref_line_offset': self.rruff_ref_line_offset_spin.value() if hasattr(self, 'rruff_ref_line_offset_spin') else 0.0,
                
                # 样式参数（从面板获取）
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
                'border_sides': self.parent_window._get_border_sides_from_config(ps) if ps and hasattr(self.parent_window, '_get_border_sides_from_config') else (self.parent_window.get_checked_border_sides() if hasattr(self.parent_window, 'get_checked_border_sides') else ['top', 'right', 'left', 'bottom']),
                'border_linewidth': ps.spine_width if ps else 2.0,
                'aspect_ratio': ps.aspect_ratio if ps else 0.6,
                
                # 物理截断
                'x_min_phys': x_min_phys,
                'x_max_phys': x_max_phys,
            }
            
            return params
            
        except Exception as e:
            print(f"Error getting parent plot params: {e}")
            traceback.print_exc()
            return None
    
    def select_folder(self):
        """选择包含txt和png文件的文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path = folder
            self.folder_label.setText(f"Folder: {os.path.basename(folder)}")
            self.btn_scan.setEnabled(True)
    
    def select_rruff_library(self):
        """选择RRUFF库文件夹（使用预处理参数和峰值检测参数，支持缓存）"""
        folder = QFileDialog.getExistingDirectory(self, "Select RRUFF Library Folder")
        if folder:
            try:
                # 创建进度对话框（立即显示，避免卡死）
                # 注意：QProgressDialog的maximum默认是100，但可以设置为更大的值
                progress = QProgressDialog("正在加载RRUFF库...", "取消", 0, 10000, self)  # 设置足够大的maximum值
                progress.setWindowTitle("加载RRUFF库")
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)  # 立即显示
                progress.setValue(0)
                progress.show()
                QApplication.processEvents()  # 立即更新UI
                
                # 获取预处理参数
                preprocess_params = self._get_preprocess_params()
                
                # 检查数据库（自动模式或手动模式）
                use_db = False
                db_name = None
                
                if self.auto_db_mode:
                    # 自动模式：根据预处理参数查找匹配的数据库
                    db_name = self.rruff_database.find_database_by_params(preprocess_params)
                    if db_name:
                        use_db = True
                        print(f"自动找到匹配的数据库: {db_name}")
                else:
                    # 手动模式：使用上次选择的数据库（如果有）
                    db_name = self.settings.value("rruff_selected_db", None)
                    if db_name:
                        use_db = True
                
                if use_db and db_name:
                    try:
                        progress.setLabelText("正在从数据库加载...")
                        QApplication.processEvents()
                        
                        # 从数据库加载
                        db_data = self.rruff_database.load_database(db_name)
                        if db_data:
                            # 验证数据库是否匹配当前文件夹
                            if db_data.get('folder_path') == folder:
                                self.rruff_loader = RRUFFLibraryLoader()
                                self.rruff_loader.library_folder = folder
                                self.rruff_loader.preprocess_params = preprocess_params
                                self.rruff_loader.library_spectra = db_data['library_spectra']
                                self.rruff_loader.peak_detection_params = db_data.get('peak_detection_params', {})
                                
                                print(f"从数据库加载成功: {len(self.rruff_loader.library_spectra)} 个光谱")
                                use_db = True
                            else:
                                print(f"数据库文件夹路径不匹配，将重新加载")
                                use_db = False
                        else:
                            print(f"数据库不存在，将重新加载")
                            use_db = False
                    except Exception as e:
                        print(f"加载数据库失败: {e}，将重新加载")
                        use_db = False
                
                if not use_db:
                    # 创建RRUFF库加载器
                    self.rruff_loader = RRUFFLibraryLoader()
                    
                    # 定义进度回调
                    def progress_callback(current, total, filename):
                        if progress.wasCanceled():
                            return
                        progress.setMaximum(total)
                        progress.setValue(current)
                        progress.setLabelText(f"正在加载: {filename} ({current}/{total})")
                        QApplication.processEvents()  # 确保UI更新
                    
                    # 加载库（使用多线程）
                    try:
                        self.rruff_loader.load_library(
                            library_folder=folder,
                            preprocess_params=preprocess_params,
                            progress_callback=progress_callback
                        )
                        
                        # 保存到数据库
                        if not progress.wasCanceled():
                            progress.setLabelText("正在保存到数据库...")
                            QApplication.processEvents()
                            # 生成数据库名称（基于文件夹名和参数哈希）
                            folder_name = os.path.basename(folder) or "RRUFF"
                            params_hash = hashlib.md5(json.dumps(preprocess_params, sort_keys=True).encode()).hexdigest()
                            db_name = f"{folder_name}_{params_hash[:8]}"
                            # 获取峰值检测参数（使用默认值，稍后会更新）
                            plot_params = self.get_parent_plot_params()
                            peak_detection_params = {
                                'peak_height_threshold': plot_params.get('peak_height_threshold', 0.0),
                                'peak_distance_min': plot_params.get('peak_distance_min', 10),
                                'peak_prominence': plot_params.get('peak_prominence', None),
                                'peak_width': plot_params.get('peak_width', None),
                                'peak_wlen': plot_params.get('peak_wlen', None),
                                'peak_rel_height': plot_params.get('peak_rel_height', None),
                            }
                            self._save_database(db_name, folder, preprocess_params, peak_detection_params)
                    except Exception as e:
                        if not progress.wasCanceled():
                            QMessageBox.critical(self, "错误", f"加载RRUFF库失败: {str(e)}")
                        traceback.print_exc()
                        progress.close()
                        return
                
                # 获取峰值检测参数并更新
                plot_params = self.get_parent_plot_params()
                peak_detection_params = {
                    'peak_height_threshold': plot_params.get('peak_height_threshold', 0.0),
                    'peak_distance_min': plot_params.get('peak_distance_min', 10),
                    'peak_prominence': plot_params.get('peak_prominence', None),
                    'peak_width': plot_params.get('peak_width', None),
                    'peak_wlen': plot_params.get('peak_wlen', None),
                    'peak_rel_height': plot_params.get('peak_rel_height', None),
                }
                
                # 如果峰值检测参数改变，需要重新检测峰值
                if self.rruff_loader.peak_detection_params != peak_detection_params:
                    progress.setLabelText("正在检测峰值...")
                    progress.setMaximum(len(self.rruff_loader.library_spectra))
                    progress.setValue(0)
                    QApplication.processEvents()
                    
                    peak_count = 0
                    for name, spectrum in self.rruff_loader.library_spectra.items():
                        if 'y_raw' in spectrum:
                            spectrum['peaks'] = self.rruff_loader._detect_peaks(
                                spectrum['x'], spectrum['y'], 
                                peak_detection_params=peak_detection_params
                            )
                        peak_count += 1
                        progress.setValue(peak_count)
                        QApplication.processEvents()
                    
                    self.rruff_loader.peak_detection_params = peak_detection_params
                    
                    # 更新数据库（包含新的峰值检测参数）
                    if db_name:
                        self._save_database(db_name, folder, preprocess_params, peak_detection_params)
                
                progress.setValue(progress.maximum())
                progress.close()
                
                count = len(self.rruff_loader.library_spectra)
                print(f"最终加载的光谱数量: {count}")
                self.rruff_label.setText(f"RRUFF Library: {count} spectra")
                
                # 启用匹配按钮
                self.btn_rruff_match.setEnabled(count > 0)
                self.btn_rruff_combination_match.setEnabled(count > 0)
                
                # 更新全局排除列表
                self.update_global_exclusion_list()
                
                # 如果数量仍然只有255，提示用户清除缓存
                if count == 255:
                    # 检查文件夹中的实际文件数量
                    import glob
                    files = glob.glob(os.path.join(folder, '*.txt')) + \
                           glob.glob(os.path.join(folder, '*.csv'))
                    total_files = len(files)
                    
                    if total_files > 255:
                        reply = QMessageBox.question(
                            self,
                            "缓存可能已损坏",
                            f"检测到只加载了255个光谱，但文件夹中有 {total_files} 个文件。\n"
                            f"这可能是因为旧缓存文件的问题。\n\n"
                            f"是否清除缓存并重新加载？",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            # 清除缓存
                            cache_file = self._get_cache_file_path(folder)
                            if os.path.exists(cache_file):
                                try:
                                    os.remove(cache_file)
                                    print(f"已删除缓存文件: {cache_file}")
                                except Exception as e:
                                    print(f"删除缓存文件失败: {e}")
                            
                            # 重新加载（递归调用，但这次不会使用缓存）
                            QMessageBox.information(
                                self,
                                "提示",
                                "缓存已清除，请再次点击'Load RRUFF Library'按钮重新加载。"
                            )
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load RRUFF library: {e}")
                traceback.print_exc()
    
    def _get_cache_file_path(self, folder):
        """获取缓存文件路径"""
        import hashlib
        # 使用文件夹路径的哈希值作为缓存文件名
        folder_hash = hashlib.md5(folder.encode()).hexdigest()
        cache_dir = os.path.join(os.path.expanduser("~"), ".spectrapro_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"rruff_cache_{folder_hash}.pkl")
    
    def _save_database(self, db_name, folder_path, preprocess_params, peak_detection_params):
        """保存RRUFF库数据到数据库"""
        try:
            spectra_count = len(self.rruff_loader.library_spectra)
            print(f"准备保存数据库: {db_name}, {spectra_count} 个光谱")
            
            self.rruff_database.save_database(
                name=db_name,
                folder_path=folder_path,
                preprocess_params=preprocess_params,
                peak_detection_params=peak_detection_params,
                library_spectra=self.rruff_loader.library_spectra,
                description=f"自动保存: {os.path.basename(folder_path)}"
            )
            
            print(f"数据库保存成功: {db_name}")
        except Exception as e:
            print(f"保存数据库失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_cache(self, folder, preprocess_params):
        """保存RRUFF库数据到缓存（已废弃，保留用于兼容性）"""
        # 这个方法已废弃，现在使用_save_database
        pass
    
    def select_database(self):
        """手动选择数据库"""
        databases = self.rruff_database.list_databases()
        
        if not databases:
            QMessageBox.information(self, "提示", "没有可用的数据库。请先加载RRUFF库。")
            return
        
        # 创建选择对话框
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("选择RRUFF数据库")
        dialog.setMinimumSize(600, 400)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("请选择要加载的数据库:"))
        
        db_list = QListWidget()
        for db in databases:
            item_text = f"{db['name']} ({db['spectra_count']} 光谱, {db['created_time']})"
            if db['description']:
                item_text += f" - {db['description']}"
            db_list.addItem(item_text)
        layout.addWidget(db_list)
        
        button_layout = QHBoxLayout()
        btn_load = QPushButton("加载")
        btn_delete = QPushButton("删除")
        btn_cancel = QPushButton("取消")
        
        def load_selected():
            selected_items = db_list.selectedItems()
            if selected_items:
                idx = db_list.row(selected_items[0])
                db_name = databases[idx]['name']
                self.settings.setValue("rruff_selected_db", db_name)
                self.auto_db_mode = False
                self.auto_db_check.setChecked(False)
                # 加载数据库
                self._load_database(db_name)
                dialog.accept()
        
        def delete_selected():
            selected_items = db_list.selectedItems()
            if selected_items:
                idx = db_list.row(selected_items[0])
                db_name = databases[idx]['name']
                reply = QMessageBox.question(
                    self, "确认删除", 
                    f"确定要删除数据库 '{db_name}' 吗？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    if self.rruff_database.delete_database(db_name):
                        QMessageBox.information(self, "成功", "数据库已删除")
                        dialog.accept()
                        self.select_database()  # 重新打开对话框
                    else:
                        QMessageBox.warning(self, "错误", "删除数据库失败")
        
        btn_load.clicked.connect(load_selected)
        btn_delete.clicked.connect(delete_selected)
        btn_cancel.clicked.connect(dialog.reject)
        
        button_layout.addWidget(btn_load)
        button_layout.addWidget(btn_delete)
        button_layout.addStretch()
        button_layout.addWidget(btn_cancel)
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def _load_database(self, db_name):
        """加载指定的数据库"""
        try:
            db_data = self.rruff_database.load_database(db_name)
            if db_data:
                self.rruff_loader = RRUFFLibraryLoader()
                self.rruff_loader.library_folder = db_data.get('folder_path', '')
                self.rruff_loader.preprocess_params = db_data.get('preprocess_params', {})
                self.rruff_loader.library_spectra = db_data.get('library_spectra', {})
                self.rruff_loader.peak_detection_params = db_data.get('peak_detection_params', {})
                
                count = len(self.rruff_loader.library_spectra)
                self.rruff_label.setText(f"RRUFF Library: {count} spectra ({db_name})")
                
                # 安全地设置按钮状态（如果按钮存在）
                if hasattr(self, 'btn_rruff_match'):
                    self.btn_rruff_match.setEnabled(count > 0)
                if hasattr(self, 'btn_rruff_combination_match'):
                    self.btn_rruff_combination_match.setEnabled(count > 0)
                
                self.update_global_exclusion_list()
                print(f"数据库加载成功: {db_name}, {count} 个光谱")
            else:
                QMessageBox.warning(self, "错误", f"无法加载数据库: {db_name}\n数据库文件不存在或已损坏。")
        except Exception as e:
            error_msg = f"加载数据库失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", error_msg)
    
    def _get_match_cache_key(self, basename, x, y, peaks, excluded_names, match_type):
        """生成匹配缓存键（基于文件basename、数据哈希、峰值检测参数和匹配参数）"""
        import hashlib
        # 获取峰值检测参数（用于缓存键，确保参数改变时缓存失效）
        plot_params = self.get_parent_plot_params()
        peak_params_str = ""
        if plot_params:
            peak_params = {
                'peak_height_threshold': plot_params.get('peak_height_threshold', 0.0),
                'peak_distance_min': plot_params.get('peak_distance_min', 10),
                'peak_prominence': plot_params.get('peak_prominence', None),
                'peak_width': plot_params.get('peak_width', None),
                'peak_wlen': plot_params.get('peak_wlen', None),
                'peak_rel_height': plot_params.get('peak_rel_height', None),
            }
            peak_params_str = str(sorted(peak_params.items()))
        
        # 使用数据的关键特征生成哈希
        peaks_hash = hash(tuple(peaks[:10])) if len(peaks) > 0 else 0
        excluded_str = str(sorted(excluded_names)) if excluded_names else "[]"
        data_hash = hashlib.md5(
            f"{basename}_{len(x)}_{len(y)}_{len(peaks)}_{peaks_hash}_{excluded_str}_{peak_params_str}_{match_type}".encode()
        ).hexdigest()
        return f"{basename}_{data_hash[:8]}_{match_type}"
    
    def _on_auto_db_mode_changed(self, state):
        """自动数据库模式改变时的回调"""
        self.auto_db_mode = (state == Qt.CheckState.Checked.value)
        if self.auto_db_mode:
            self.settings.remove("rruff_selected_db")  # 清除手动选择的数据库
    
    def update_rruff_preprocessing(self):
        """更新RRUFF库的预处理参数和峰值检测参数（当主窗口参数改变时调用）"""
        if self.rruff_loader and self.rruff_loader.library_spectra:
            try:
                preprocess_params = self._get_preprocess_params()
                # 获取峰值检测参数
                plot_params = self.get_parent_plot_params()
                peak_detection_params = {
                    'peak_height_threshold': plot_params.get('peak_height_threshold', 0.0),
                    'peak_distance_min': plot_params.get('peak_distance_min', 10),
                    'peak_prominence': plot_params.get('peak_prominence', None),
                    'peak_width': plot_params.get('peak_width', None),
                    'peak_wlen': plot_params.get('peak_wlen', None),
                    'peak_rel_height': plot_params.get('peak_rel_height', None),
                }
                # 保存旧的峰值检测参数（用于判断是否改变）
                old_peak_params = self.rruff_loader.peak_detection_params.copy() if self.rruff_loader.peak_detection_params else {}
                
                # 检查参数是否真正改变（避免不必要的进度条）
                preprocess_changed = (self.rruff_loader.preprocess_params != preprocess_params)
                peak_detection_changed = (self.rruff_loader.peak_detection_params != peak_detection_params)
                
                if not preprocess_changed and not peak_detection_changed:
                    # 参数没有改变，直接返回
                    return
                
                # 计算需要处理的光谱数量
                total_spectra = len(self.rruff_loader.library_spectra)
                
                # 创建进度对话框（只在有大量光谱时才显示）
                progress = None
                if total_spectra > 50:  # 只有超过50个光谱时才显示进度条
                    progress = QProgressDialog("正在更新RRUFF库...", "取消", 0, total_spectra, self)
                    progress.setWindowTitle("更新RRUFF库")
                    progress.setWindowModality(Qt.WindowModality.WindowModal)
                    progress.setMinimumDuration(500)  # 500ms后才显示
                    progress.setValue(0)
                
                # 定义进度回调
                def progress_callback(current, total, message):
                    if progress:
                        if progress.wasCanceled():
                            return
                        progress.setMaximum(total)
                        progress.setValue(current)
                        progress.setLabelText(message)
                        QApplication.processEvents()
                
                # 更新预处理参数（只在参数真正改变时才重新处理）
                params_changed = self.rruff_loader.update_preprocessing(
                    preprocess_params, 
                    peak_detection_params,
                    progress_callback=progress_callback if total_spectra > 50 else None
                )
                
                # 关闭进度条
                if progress:
                    progress.setValue(progress.maximum())
                    progress.close()
                
                # 如果峰值检测参数改变，清除匹配缓存（因为峰值变了，匹配结果也应该变）
                if params_changed:
                    # 检查是否只是峰值检测参数改变
                    peak_params_changed = (old_peak_params != peak_detection_params)
                    
                    if peak_params_changed:
                        print("[缓存] 峰值检测参数改变，清除匹配缓存")
                        self._match_cache.clear()  # 清除所有匹配缓存
                        # 注意：不清除 rruff_match_results 和 rruff_combination_results，
                        # 因为这些是用户已经匹配的结果，只是峰值显示会更新
                
                # 如果当前有选中的文件，重新绘制（不触发自动匹配）
                if self.file_list.selectedItems():
                    self._update_plots_with_rruff()  # 使用这个方法不会触发自动匹配
            except Exception as e:
                print(f"更新RRUFF库预处理参数失败: {e}")
                import traceback
                traceback.print_exc()
    
    def _get_preprocess_params(self):
        """获取当前预处理参数"""
        if not self.parent_window:
            return {}
        
        try:
            return {
                'qc_enabled': self.parent_window.qc_check.isChecked(),
                'qc_threshold': self.parent_window.qc_threshold_spin.value(),
                'is_be_correction': self.parent_window.be_check.isChecked(),
                'be_temp': self.parent_window.be_temp_spin.value(),
                'is_smoothing': self.parent_window.smoothing_check.isChecked(),
                'smoothing_window': self.parent_window.smoothing_window_spin.value(),
                'smoothing_poly': self.parent_window.smoothing_poly_spin.value(),
                'is_baseline_als': self.parent_window.baseline_als_check.isChecked(),
                'als_lam': self.parent_window.lam_spin.value(),
                'als_p': self.parent_window.p_spin.value(),
                'normalization_mode': self.parent_window.normalization_combo.currentText(),
                'global_transform_mode': self.parent_window.global_transform_combo.currentText(),
                'global_log_base': self.parent_window.global_log_base_combo.currentText(),
                'global_log_offset': self.parent_window.global_log_offset_spin.value(),
                'global_sqrt_offset': self.parent_window.global_sqrt_offset_spin.value(),
                'global_y_offset': self.parent_window.global_y_offset_spin.value() if hasattr(self.parent_window, 'global_y_offset_spin') else 0.0,
                'is_derivative': self.parent_window.derivative_check.isChecked(),
            }
        except Exception as e:
            print(f"Error getting preprocess params: {e}")
            return {}
    
    def update_global_exclusion_list(self):
        """更新全局排除列表显示"""
        self.global_exclusion_list.clear()
        if self.rruff_loader:
            # 获取所有光谱名称
            all_names = self.rruff_loader.get_all_spectra_names()
            total_count = len(all_names)
            
            print(f"开始更新Global RRUFF Exclusions列表: 总共 {total_count} 个光谱")
            
            # 如果光谱数量很多，显示进度条
            progress = None
            if total_count > 200:  # 超过200个光谱时显示进度条
                progress = QProgressDialog("正在更新全局排除列表...", "取消", 0, total_count, self)
                progress.setWindowTitle("更新排除列表")
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(300)  # 300ms后才显示
                progress.setValue(0)
                progress.show()
                QApplication.processEvents()
            
            # 批量添加项目，避免UI卡顿，并确保所有项目都被添加
            # 使用setUpdatesEnabled来优化大量项目的添加
            self.global_exclusion_list.setUpdatesEnabled(False)
            try:
                # 分批添加项目（每批1000个，避免一次性添加太多导致UI卡顿）
                batch_size = 1000
                added_count = 0
                
                for batch_start in range(0, total_count, batch_size):
                    if progress and progress.wasCanceled():
                        break
                    
                    batch_end = min(batch_start + batch_size, total_count)
                    batch_names = all_names[batch_start:batch_end]
                    
                    # 创建这一批的项目
                    batch_items = []
                    for name in batch_names:
                        item = QListWidgetItem(name)
                        item.setCheckState(Qt.CheckState.Unchecked)
                        batch_items.append(item)
                    
                    # 批量添加
                    for item in batch_items:
                        self.global_exclusion_list.addItem(item)
                        added_count += 1
                    
                    # 更新进度条
                    if progress:
                        progress.setValue(batch_end)
                        progress.setLabelText(f"正在添加: {batch_end}/{total_count}")
                    
                    # 每批后更新UI，避免长时间无响应
                    if batch_end < total_count:
                        QApplication.processEvents()
                
                # 验证添加的数量
                actual_count = self.global_exclusion_list.count()
                if actual_count != total_count:
                    print(f"警告: 期望添加 {total_count} 个项目，但实际只添加了 {actual_count} 个")
                    print(f"调试信息: added_count={added_count}, actual_count={actual_count}, total_count={total_count}")
                    
                    # 如果数量不匹配，尝试继续添加剩余的项目
                    if actual_count < total_count:
                        print(f"尝试添加剩余 {total_count - actual_count} 个项目...")
                        for i in range(actual_count, total_count):
                            try:
                                item = QListWidgetItem(all_names[i])
                                item.setCheckState(Qt.CheckState.Unchecked)
                                self.global_exclusion_list.addItem(item)
                            except Exception as e:
                                print(f"添加第 {i} 个项目时出错: {e}")
                                break
                        
                        # 再次验证
                        final_actual_count = self.global_exclusion_list.count()
                        if final_actual_count != total_count:
                            print(f"错误: 仍然无法添加所有项目。期望 {total_count}，实际 {final_actual_count}")
                            # 尝试使用insertItem而不是addItem
                            if final_actual_count < total_count:
                                print("尝试使用insertItem方法...")
                                for i in range(final_actual_count, total_count):
                                    try:
                                        item = QListWidgetItem(all_names[i])
                                        item.setCheckState(Qt.CheckState.Unchecked)
                                        self.global_exclusion_list.insertItem(i, item)
                                    except Exception as e:
                                        print(f"insertItem第 {i} 个项目时出错: {e}")
                                        break
            finally:
                self.global_exclusion_list.setUpdatesEnabled(True)
            
            # 关闭进度条
            if progress:
                progress.setValue(progress.maximum())
                progress.close()
            
            # 打印调试信息
            final_count = self.global_exclusion_list.count()
            print(f"Global RRUFF Exclusions列表已更新: 总共 {final_count} 个光谱（期望 {total_count} 个）")
            
            # 如果仍然不匹配，显示警告对话框
            if final_count != total_count:
                QMessageBox.warning(
                    self, 
                    "警告", 
                    f"Global RRUFF Exclusions列表可能不完整！\n"
                    f"期望显示 {total_count} 个光谱，但实际只显示了 {final_count} 个。\n"
                    f"这可能是因为QListWidget的性能限制。\n"
                    f"请检查控制台输出的详细调试信息。"
                )
    
    def scan_files(self):
        """扫描文件夹中的光谱数据文件（.txt/.csv）和对应的png/jpg图像文件"""
        if not hasattr(self, 'folder_path'):
            QMessageBox.warning(self, "Warning", "Please select folder first")
            return
        
        try:
            # 扫描光谱数据文件（.txt / .csv）
            txt_pattern = os.path.join(self.folder_path, '*.txt')
            csv_pattern = os.path.join(self.folder_path, '*.csv')
            self.txt_files = sorted(glob.glob(txt_pattern) + glob.glob(csv_pattern))
            
            if not self.txt_files:
                QMessageBox.warning(self, "Warning", "No txt/csv files found")
                return
            
            # 查找对应的png文件（支持带后缀的文件名匹配）
            self.png_files = {}
            import re
            for txt_file in self.txt_files:
                txt_basename = os.path.splitext(os.path.basename(txt_file))[0]
                png_file = None
                
                # 方法1：直接匹配（完整文件名）
                png_file_path = os.path.join(self.folder_path, f"{txt_basename}.png")
                if os.path.exists(png_file_path):
                    png_file = png_file_path
                else:
                    # 方法2：提取基础名称（去掉括号及其内容，如"serpentinite-2（1%）" -> "serpentinite-2"）
                    # 匹配括号及其内容：中文括号、英文括号、方括号等
                    base_name = re.sub(r'[（(（\[].*?[）)）\]]', '', txt_basename).strip()
                    if base_name and base_name != txt_basename:
                        png_file_path = os.path.join(self.folder_path, f"{base_name}.png")
                        if os.path.exists(png_file_path):
                            png_file = png_file_path
                    
                    # 方法3：如果还没找到，尝试其他常见格式
                    if not png_file:
                        for ext in ['.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
                            alt_file = os.path.join(self.folder_path, f"{txt_basename}{ext}")
                            if os.path.exists(alt_file):
                                png_file = alt_file
                                break
                            # 也尝试基础名称的其他格式
                            if base_name and base_name != txt_basename:
                                alt_file = os.path.join(self.folder_path, f"{base_name}{ext}")
                                if os.path.exists(alt_file):
                                    png_file = alt_file
                                    break
                
                if png_file:
                    self.png_files[txt_basename] = png_file
            
            # 更新文件列表
            self.file_list.clear()
            for txt_file in self.txt_files:
                txt_basename = os.path.splitext(os.path.basename(txt_file))[0]
                has_png = txt_basename in self.png_files
                item_text = f"{txt_basename} {'✓' if has_png else '✗'}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, txt_basename)
                self.file_list.addItem(item)
            
            self.btn_export_all.setEnabled(True)
            # 不再弹出“扫描完成”的提示框
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to scan files: {e}")
            traceback.print_exc()
    
    def show_file_context_menu(self, position: QPoint):
        """显示文件列表的右键菜单"""
        item = self.file_list.itemAt(position)
        if item is None:
            return
        
        txt_basename = item.data(Qt.ItemDataRole.UserRole)
        if not txt_basename:
            return
        
        menu = QMenu(self)
        
        action_exclude = menu.addAction("Set Individual RRUFF Exclusions")
        action_exclude.triggered.connect(lambda: self.set_spectrum_exclusions(txt_basename))
        
        action_clear = menu.addAction("Clear Individual Exclusions")
        action_clear.triggered.connect(lambda: self.clear_spectrum_exclusions(txt_basename))
        
        menu.exec(self.file_list.mapToGlobal(position))
    
    def set_spectrum_exclusions(self, txt_basename):
        """为指定谱图设置单独排除的RRUFF库项"""
        if not self.rruff_loader:
            QMessageBox.warning(self, "Warning", "Please load RRUFF library first")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Set Exclusions for {txt_basename}")
        dialog.setMinimumSize(400, 500)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select RRUFF library items to exclude (checked = excluded):"))
        
        exclusion_list = QListWidget()
        exclusion_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        current_exclusions = self.spectrum_exclusions.get(txt_basename, [])
        
        for name in self.rruff_loader.get_all_spectra_names():
            item = QListWidgetItem(name)
            if name in current_exclusions:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            exclusion_list.addItem(item)
        
        layout.addWidget(exclusion_list)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            excluded_names = []
            for i in range(exclusion_list.count()):
                item = exclusion_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    excluded_names.append(item.text())
            
            if excluded_names:
                self.spectrum_exclusions[txt_basename] = excluded_names
            else:
                if txt_basename in self.spectrum_exclusions:
                    del self.spectrum_exclusions[txt_basename]
            
            # 刷新绘图
            self.on_file_selected()
            
            # 不再弹出“设置排除项完成”的提示框
    
    def clear_spectrum_exclusions(self, txt_basename):
        """清除指定谱图的单独排除项"""
        if txt_basename in self.spectrum_exclusions:
            del self.spectrum_exclusions[txt_basename]
            self.on_file_selected()
            # 不再弹出“清除排除项完成”的提示框
        else:
            # 不再弹出“没有排除项”的提示框
            pass
    
    def _on_rruff_tolerance_changed(self, value):
        """RRUFF匹配容差改变时更新匹配器"""
        self.peak_matcher.tolerance = value

    @staticmethod
    def _filter_combinations_by_variants(combinations):
        """
        过滤掉同一物相的不同变种组合。
        例如 talc-1 / talc-2 / talc-3 视为同一矿物 talc，只允许在同一组合中出现一次。
        """
        import re

        def base_name(name: str) -> str:
            # 去掉末尾的 -数字 或 _数字，得到基础物相名
            return re.sub(r"[-_]\d+$", "", name)

        filtered = []
        for combo in combinations:
            phases = combo.get("phases", [])
            bases = [base_name(p) for p in phases]
            if len(bases) == len(set(bases)):
                filtered.append(combo)
        return filtered
    
    def match_rruff_spectra(self):
        """匹配当前选中光谱的RRUFF光谱"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "请先选择一个光谱文件")
            return
        
        if not self.rruff_loader or not self.rruff_loader.library_spectra:
            QMessageBox.warning(self, "Warning", "请先加载RRUFF库")
            return
        
        try:
            # 获取第一个选中的文件
            txt_basename = selected_items[0].data(Qt.ItemDataRole.UserRole)
            
            # 读取数据
            txt_file = None
            for f in self.txt_files:
                if os.path.splitext(os.path.basename(f))[0] == txt_basename:
                    txt_file = f
                    break
            
            if not txt_file:
                QMessageBox.warning(self, "Warning", "未找到文件")
                return
            
            # 获取绘图参数
            plot_params = self.get_parent_plot_params()
            if not plot_params:
                QMessageBox.warning(self, "Warning", "无法获取绘图参数")
                return
            
            # 读取光谱数据
            x, y = self.data_controller.read_data(
                txt_file,
                plot_params['skip_rows'],
                plot_params['x_min_phys'],
                plot_params['x_max_phys']
            )
            
            # 应用预处理（传入文件路径以支持缓存）
            y_proc = self._preprocess_spectrum(x, y, plot_params, file_path=txt_file)
            
            # 检测峰值（使用主菜单的峰值检测参数，降低阈值以检测更多小峰）
            # 使用主菜单的峰值检测参数（允许极小的值以检测所有峰值）
            peak_height = plot_params.get('peak_height_threshold', 0.0)
            peak_distance = plot_params.get('peak_distance_min', 10)
            peak_prominence = plot_params.get('peak_prominence', None)
            
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
            
            peak_wavenumbers = x[peaks] if len(peaks) > 0 else np.array([])
            
            # 获取排除列表
            excluded_names = list(self.spectrum_exclusions.get(txt_basename, []))
            for i in range(self.global_exclusion_list.count()):
                item = self.global_exclusion_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    excluded_name = item.text()
                    if excluded_name not in excluded_names:
                        excluded_names.append(excluded_name)
            
            # 更新匹配容差
            tolerance = self.rruff_match_tolerance_spin.value()
            self.peak_matcher.tolerance = tolerance
            
            # 确保RRUFF库使用相同的峰值检测参数（在匹配前更新）
            peak_detection_params = {
                'peak_height_threshold': plot_params.get('peak_height_threshold', 0.0),
                'peak_distance_min': plot_params.get('peak_distance_min', 10),
                'peak_prominence': plot_params.get('peak_prominence', None),
                'peak_width': plot_params.get('peak_width', None),
                'peak_wlen': plot_params.get('peak_wlen', None),
                'peak_rel_height': plot_params.get('peak_rel_height', None),
            }
            # 如果峰值检测参数已改变，重新检测RRUFF库的峰值
            if self.rruff_loader.peak_detection_params != peak_detection_params:
                for name, spectrum in self.rruff_loader.library_spectra.items():
                    if 'y_raw' in spectrum:
                        spectrum['peaks'] = self.rruff_loader._detect_peaks(
                            spectrum['x'], spectrum['y'], 
                            peak_detection_params=peak_detection_params
                        )
                self.rruff_loader.peak_detection_params = peak_detection_params
            
            # 创建进度对话框（设置足够大的maximum值，避免255限制）
            total_spectra = len(self.rruff_loader.library_spectra)
            progress = QProgressDialog("正在匹配RRUFF光谱...", "取消", 0, max(total_spectra, 10000), self)
            progress.setWindowTitle("匹配RRUFF光谱")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(500)  # 500ms后才显示
            progress.setValue(0)
            
            # 定义进度回调
            def progress_callback(current, total, name):
                if progress.wasCanceled():
                    return
                progress.setMaximum(total)
                progress.setValue(current)
                progress.setLabelText(f"正在匹配: {name} ({current}/{total})")
                QApplication.processEvents()
            
            # 检查缓存
            cache_key = self._get_match_cache_key(txt_basename, x, y_proc, peak_wavenumbers, excluded_names, 'single')
            if cache_key in self._match_cache and 'single' in self._match_cache[cache_key]:
                print(f"[缓存] 使用缓存的单物相匹配结果: {txt_basename}")
                matches = self._match_cache[cache_key]['single']
            else:
                # 匹配RRUFF光谱
                try:
                    matches = self.peak_matcher.find_best_matches(
                        x, y_proc, peak_wavenumbers, self.rruff_loader, top_k=100,  # 增加top_k以获取更多结果
                        excluded_names=excluded_names if excluded_names else None,
                        progress_callback=progress_callback,
                        max_workers=32
                    )
                finally:
                    progress.setValue(progress.maximum())
                    progress.close()
                
                # 保存到缓存
                if cache_key not in self._match_cache:
                    self._match_cache[cache_key] = {}
                self._match_cache[cache_key]['single'] = matches
            
            self.rruff_match_results[txt_basename] = matches
            
            # 更新列表（匹配分数显示在最前面）
            self.rruff_match_list.clear()
            for i, match in enumerate(matches):
                name = match['name']
                score = match['match_score']
                # 简化文件名显示（只显示前30个字符，完整名称在工具提示中）
                display_name = name[:30] + "..." if len(name) > 30 else name
                item = QListWidgetItem(f"{i+1}. [{score:.3f}] {display_name}")
                item.setToolTip(f"完整名称: {name}\n匹配分数: {score:.3f}")
                item.setData(Qt.ItemDataRole.UserRole, name)
                # 检查是否已选中
                if txt_basename in self.selected_rruff_spectra and name in self.selected_rruff_spectra[txt_basename]:
                    item.setSelected(True)
                self.rruff_match_list.addItem(item)
            
            # 启用组合匹配按钮
            self.btn_rruff_combination_match.setEnabled(len(matches) > 0)
            
            # 不再弹出“找到多少匹配光谱”的提示框
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"匹配RRUFF光谱失败：{str(e)}")
            traceback.print_exc()
    
    def match_rruff_combination(self):
        """多物相组合匹配：将多个RRUFF光谱组合来匹配查询光谱"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "请先选择一个光谱文件")
            return
        
        if not self.rruff_loader or not self.rruff_loader.library_spectra:
            QMessageBox.warning(self, "Warning", "请先加载RRUFF库")
            return
        
        try:
            # 获取第一个选中的文件
            txt_basename = selected_items[0].data(Qt.ItemDataRole.UserRole)
            
            # 读取数据
            txt_file = None
            for f in self.txt_files:
                if os.path.splitext(os.path.basename(f))[0] == txt_basename:
                    txt_file = f
                    break
            
            if not txt_file:
                QMessageBox.warning(self, "Warning", "未找到文件")
                return
            
            # 获取绘图参数
            plot_params = self.get_parent_plot_params()
            if not plot_params:
                QMessageBox.warning(self, "Warning", "无法获取绘图参数")
                return
            
            # 读取光谱数据
            x, y = self.data_controller.read_data(
                txt_file,
                plot_params['skip_rows'],
                plot_params['x_min_phys'],
                plot_params['x_max_phys']
            )
            
            # 应用预处理（传入文件路径以支持缓存）
            y_proc = self._preprocess_spectrum(x, y, plot_params, file_path=txt_file)
            
            # 检测峰值（使用主菜单的峰值检测参数）
            peak_height = plot_params.get('peak_height_threshold', 0.0)
            peak_distance = plot_params.get('peak_distance_min', 10)
            peak_prominence = plot_params.get('peak_prominence', None)
            
            y_max = np.max(y_proc) if len(y_proc) > 0 else 0
            y_min = np.min(y_proc) if len(y_proc) > 0 else 0
            y_range = y_max - y_min
            
            peak_kwargs = {}
            if peak_height == 0:
                if y_max > 0:
                    peak_height = y_max * 0.001
                else:
                    peak_height = 0
            if peak_height > y_range * 2 and y_range > 0:
                peak_height = y_max * 0.001
            if peak_height != 0:
                peak_kwargs['height'] = peak_height
            
            if peak_distance == 0:
                peak_distance = max(1, int(len(y_proc) * 0.001))
            if peak_distance > len(y_proc) * 0.5:
                peak_distance = max(1, int(len(y_proc) * 0.001))
            peak_distance = max(1, peak_distance)
            
            if peak_height < 0 or (y_max > 0 and peak_height < y_max * 0.001):
                pass  # 不使用distance
            else:
                peak_kwargs['distance'] = peak_distance
            
            if peak_prominence is not None and peak_prominence != 0:
                if peak_prominence > y_range * 2 and y_range > 0:
                    peak_prominence = y_range * 0.001
                peak_kwargs['prominence'] = peak_prominence
            
            try:
                peaks, properties = find_peaks(y_proc, **peak_kwargs)
            except:
                peaks, properties = find_peaks(y_proc, 
                                            height=y_max * 0.001 if y_max > 0 else 0,
                                            distance=max(1, int(len(y_proc) * 0.001)))
            
            peak_wavenumbers = x[peaks] if len(peaks) > 0 else np.array([])
            
            # 确保RRUFF库使用相同的峰值检测参数
            peak_detection_params = {
                'peak_height_threshold': plot_params.get('peak_height_threshold', 0.0),
                'peak_distance_min': plot_params.get('peak_distance_min', 10),
                'peak_prominence': plot_params.get('peak_prominence', None),
                'peak_width': plot_params.get('peak_width', None),
                'peak_wlen': plot_params.get('peak_wlen', None),
                'peak_rel_height': plot_params.get('peak_rel_height', None),
            }
            if self.rruff_loader.peak_detection_params != peak_detection_params:
                for name, spectrum in self.rruff_loader.library_spectra.items():
                    if 'y_raw' in spectrum:
                        spectrum['peaks'] = self.rruff_loader._detect_peaks(
                            spectrum['x'], spectrum['y'], 
                            peak_detection_params=peak_detection_params
                        )
                self.rruff_loader.peak_detection_params = peak_detection_params
            
            # 执行组合匹配
            tolerance = self.rruff_match_tolerance_spin.value()
            self.peak_matcher.tolerance = tolerance
            
            excluded_names = list(self.spectrum_exclusions.get(txt_basename, []))
            for i in range(self.global_exclusion_list.count()):
                item = self.global_exclusion_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    excluded_name = item.text()
                    if excluded_name not in excluded_names:
                        excluded_names.append(excluded_name)
            
            # 检查是否可以使用GPU加速
            use_gpu = False
            try:
                import cupy as cp
                use_gpu = True
            except ImportError:
                try:
                    import torch
                    if torch.cuda.is_available():
                        use_gpu = True
                except ImportError:
                    pass
            
            # 调用统一的匹配函数
            combinations = self._match_rruff_combination_for_file(
                txt_basename, x, y_proc, peak_wavenumbers, excluded_names, 
                use_gpu, show_progress=True  # 手动点击时显示进度条
            )
            
            if not combinations:
                QMessageBox.warning(self, "Warning", "未找到匹配结果")
                return
            
            # 确保结果按匹配分数排序（降序：分数高的在前）
            # 优先按匹配分数排序（越高越好），然后按未匹配峰值数排序（越少越好）
            def get_sort_key(x):
                """获取排序键：优先匹配分数高的，然后未匹配峰值数少的"""
                match_score = x.get('match_score', 0.0)
                unmatched_count = x.get('num_unmatched_peaks')
                if unmatched_count is None:
                    unmatched_peaks = x.get('unmatched_peaks', [])
                    unmatched_count = len(unmatched_peaks) if isinstance(unmatched_peaks, (list, np.ndarray)) else 0
                return (-match_score, unmatched_count)  # 匹配分数降序，未匹配峰值数升序
            
            combinations_sorted = sorted(combinations, key=get_sort_key, reverse=False)
            
            self.rruff_combination_results[txt_basename] = combinations_sorted
            
            # 更新列表（显示组合匹配结果，匹配分数显示在最前面）
            self.rruff_match_list.clear()
            for i, combo in enumerate(combinations_sorted):
                match_score = combo.get('match_score', 0.0)
                phases = combo.get('phases', [])
                ratios = combo.get('ratios', [])
                # 简化物相名称显示
                phases_display = []
                for phase in phases:
                    display_phase = phase[:20] + "..." if len(phase) > 20 else phase
                    phases_display.append(display_phase)
                phases_str = " + ".join(phases_display)
                ratios_str = " / ".join([f"{r:.2f}" for r in ratios])
                # 完整信息在工具提示中
                full_phases_str = " + ".join(phases)
                item_text = f"{i+1}. [{match_score:.3f}] {phases_str} ({ratios_str})"
                item = QListWidgetItem(item_text)
                item.setToolTip(f"完整物相: {full_phases_str}\n比例: {ratios_str}\n匹配分数: {match_score:.3f}")
                item.setData(Qt.ItemDataRole.UserRole, combo)  # 存储整个组合数据
                if txt_basename in self.selected_rruff_combinations:
                    for sel_combo in self.selected_rruff_combinations[txt_basename]:
                        if sel_combo['phases'] == combo['phases']:
                            item.setSelected(True)
                            break
                self.rruff_match_list.addItem(item)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"组合匹配失败：{str(e)}")
            traceback.print_exc()
    
    def _match_rruff_combination_for_file(self, txt_basename, x, y_proc, peak_wavenumbers, excluded_names, use_gpu, show_progress=False):
        """
        统一的RRUFF多物相组合匹配函数（手动匹配和自动匹配都调用此函数）
        
        Args:
            txt_basename: 文件basename
            x: 波数数组
            y_proc: 预处理后的强度数组
            peak_wavenumbers: 峰值波数数组
            excluded_names: 排除的光谱名称列表
            use_gpu: 是否使用GPU
            show_progress: 是否显示进度条（手动点击时显示，自动匹配时不显示）
        
        Returns:
            combinations: 匹配结果列表
        """
        # 检查缓存（如果命中缓存，直接返回，不显示进度条）
        cache_key = self._get_match_cache_key(txt_basename, x, y_proc, peak_wavenumbers, excluded_names, 'combo')
        if cache_key in self._match_cache and 'combo' in self._match_cache[cache_key]:
            print(f"[缓存] 使用缓存的多物相匹配结果: {txt_basename}")
            return self._match_cache[cache_key]['combo']
        
        # 如果未命中缓存且需要显示进度条，创建进度对话框
        progress = None
        if show_progress:
            progress = QProgressDialog("正在匹配多物相组合...", "取消", 0, 10000, self)
            progress.setWindowTitle("多物相组合匹配")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(500)
            progress.setValue(0)
        
        try:
            # 定义进度回调
            def progress_callback(current, total, name):
                if progress and progress.wasCanceled():
                    return
                if progress:
                    progress.setMaximum(total)
                    progress.setValue(current)
                    progress.setLabelText(f"正在匹配: {name}")
                    QApplication.processEvents()
            
            # 执行组合匹配（自动调整物相数量，不设上限）
            num_peaks = len(peak_wavenumbers)
            num_candidates = len(self.rruff_loader.library_spectra) - len(excluded_names) if excluded_names else len(self.rruff_loader.library_spectra)
            auto_max_phases = min(max(num_peaks // 3, 3), num_candidates, 10)  # 最多10个物相，避免组合爆炸
            
            combinations = self.peak_matcher.find_best_combination_matches(
                x, y_proc, peak_wavenumbers, self.rruff_loader, 
                max_phases=auto_max_phases, top_k=None,  # top_k=None表示不限制结果数量
                excluded_names=excluded_names if excluded_names else None,
                use_gpu=use_gpu, progress_callback=progress_callback if show_progress else None
            )
            
            # 按需过滤同一物相的不同变种
            if getattr(self, "rruff_filter_variants_check", None) is not None and self.rruff_filter_variants_check.isChecked():
                combinations = self._filter_combinations_by_variants(combinations)
            
            # 保存到缓存
            if cache_key not in self._match_cache:
                self._match_cache[cache_key] = {}
            self._match_cache[cache_key]['combo'] = combinations
            
            return combinations
        finally:
            if progress:
                progress.setValue(progress.maximum())
                progress.close()
    
    def _auto_match_rruff_combination_for_file(self, txt_basename: str):
        """
        自动为指定文件执行一次RRUFF多物相组合匹配，但不弹出任何提示框。
        仅在自动匹配开关勾选时由 on_file_selected 调用。
        """
        # 需要RRUFF库
        if not self.rruff_loader or not self.rruff_loader.library_spectra:
            return
        
        # 从文件列表中找到对应的 txt 文件路径
        txt_file = None
        for f in self.txt_files:
            if os.path.splitext(os.path.basename(f))[0] == txt_basename:
                txt_file = f
                break
        if not txt_file:
            return
        
        plot_params = self.get_parent_plot_params()
        if not plot_params:
            return
        
        try:
            # 读取光谱数据并预处理
            x, y = self.data_controller.read_data(
                txt_file,
                plot_params['skip_rows'],
                plot_params['x_min_phys'],
                plot_params['x_max_phys']
            )
            y_proc = self._preprocess_spectrum(x, y, plot_params, file_path=txt_file)
            
            # 峰值检测（与match_rruff_combination保持一致）
            peak_height = plot_params.get('peak_height_threshold', 0.0)
            peak_distance = plot_params.get('peak_distance_min', 10)
            peak_prominence = plot_params.get('peak_prominence', None)
            
            y_max = np.max(y_proc) if len(y_proc) > 0 else 0
            y_min = np.min(y_proc) if len(y_proc) > 0 else 0
            y_range = y_max - y_min
            
            peak_kwargs = {}
            if peak_height == 0:
                if y_max > 0:
                    peak_height = y_max * 0.001
                else:
                    peak_height = 0
            if peak_height > y_range * 2 and y_range > 0:
                peak_height = y_max * 0.001
            if peak_height != 0:
                peak_kwargs['height'] = peak_height
            
            if peak_distance == 0:
                peak_distance = max(1, int(len(y_proc) * 0.001))
            if peak_distance > len(y_proc) * 0.5:
                peak_distance = max(1, int(len(y_proc) * 0.001))
            peak_distance = max(1, peak_distance)
            
            if peak_height < 0 or (y_max > 0 and peak_height < y_max * 0.001):
                pass  # 不使用distance
            else:
                peak_kwargs['distance'] = peak_distance
            
            if peak_prominence is not None and peak_prominence != 0:
                if peak_prominence > y_range * 2 and y_range > 0:
                    peak_prominence = y_range * 0.001
                peak_kwargs['prominence'] = peak_prominence
            
            try:
                peaks, _ = find_peaks(y_proc, **peak_kwargs)
            except:
                peaks, _ = find_peaks(y_proc, 
                                    height=y_max * 0.001 if y_max > 0 else 0,
                                    distance=max(1, int(len(y_proc) * 0.001)))
            
            peak_wavenumbers = x[peaks] if len(peaks) > 0 else np.array([])
            
            # 确保RRUFF库使用相同的峰值检测参数
            peak_detection_params = {
                'peak_height_threshold': plot_params.get('peak_height_threshold', 0.0),
                'peak_distance_min': plot_params.get('peak_distance_min', 10),
                'peak_prominence': plot_params.get('peak_prominence', None),
                'peak_width': plot_params.get('peak_width', None),
                'peak_wlen': plot_params.get('peak_wlen', None),
                'peak_rel_height': plot_params.get('peak_rel_height', None),
            }
            if self.rruff_loader.peak_detection_params != peak_detection_params:
                for name, spectrum in self.rruff_loader.library_spectra.items():
                    if 'y_raw' in spectrum:
                        spectrum['peaks'] = self.rruff_loader._detect_peaks(
                            spectrum['x'], spectrum['y'], 
                            peak_detection_params=peak_detection_params
                        )
                self.rruff_loader.peak_detection_params = peak_detection_params
            
            # 计算排除列表
            excluded_names = list(self.spectrum_exclusions.get(txt_basename, []))
            for i in range(self.global_exclusion_list.count()):
                item = self.global_exclusion_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    name = item.text()
                    if name not in excluded_names:
                        excluded_names.append(name)
            
            # 检查是否可以使用GPU加速
            use_gpu = False
            try:
                import cupy as cp
                use_gpu = True
            except ImportError:
                try:
                    import torch
                    if torch.cuda.is_available():
                        use_gpu = True
                except ImportError:
                    pass
            
            # 调用统一的匹配函数（不显示进度条）
            combinations = self._match_rruff_combination_for_file(
                txt_basename, x, y_proc, peak_wavenumbers, excluded_names, 
                use_gpu, show_progress=False  # 自动匹配时不显示进度条
            )
            
            if combinations:
                # 排序并保存结果（与手动匹配保持一致）
                def get_sort_key(x):
                    match_score = x.get('match_score', 0.0)
                    unmatched_count = x.get('num_unmatched_peaks')
                    if unmatched_count is None:
                        unmatched_peaks = x.get('unmatched_peaks', [])
                        unmatched_count = len(unmatched_peaks) if isinstance(unmatched_peaks, (list, np.ndarray)) else 0
                    return (-match_score, unmatched_count)
                
                combinations_sorted = sorted(combinations, key=get_sort_key, reverse=False)
                self.rruff_combination_results[txt_basename] = combinations_sorted
                
                # 如果当前左侧选中的就是这个文件，刷新匹配结果列表
                selected_items = self.file_list.selectedItems()
                if selected_items and selected_items[0].data(Qt.ItemDataRole.UserRole) == txt_basename:
                    self.rruff_match_list.clear()
                    for i, combo in enumerate(combinations_sorted):
                        match_score = combo.get('match_score', 0.0)
                        phases = combo.get('phases', [])
                        ratios = combo.get('ratios', [])
                        phases_display = []
                        for phase in phases:
                            display_phase = phase[:20] + "..." if len(phase) > 20 else phase
                            phases_display.append(display_phase)
                        phases_str = " + ".join(phases_display)
                        ratios_str = " / ".join([f"{r:.2f}" for r in ratios])
                        full_phases_str = " + ".join(phases)
                        item_text = f"{i+1}. [{match_score:.3f}] {phases_str} ({ratios_str})"
                        item = QListWidgetItem(item_text)
                        item.setToolTip(f"完整物相: {full_phases_str}\n比例: {ratios_str}\n匹配分数: {match_score:.3f}")
                        item.setData(Qt.ItemDataRole.UserRole, combo)
                        if txt_basename in self.selected_rruff_combinations:
                            for sel_combo in self.selected_rruff_combinations[txt_basename]:
                                if sel_combo['phases'] == combo['phases']:
                                    item.setSelected(True)
                                    break
                        self.rruff_match_list.addItem(item)
        
        except Exception as e:
            # 自动模式静默失败，仅打印日志
            print(f"[Auto RRUFF Match] 自动匹配 {txt_basename} 失败: {e}")
    
    def _preprocess_spectrum(self, x, y, plot_params, file_path=None):
        """
        预处理单个光谱（使用统一预处理函数，支持缓存）
        
        Args:
            x: X轴数据
            y: Y轴数据
            plot_params: 绘图参数字典
            file_path: 文件路径（用于缓存，可选）
        
        Returns:
            预处理后的Y数据
        """
        from src.core.preprocessor import DataPreProcessor
        
        # 准备预处理参数
        preprocess_params = {
            'qc_enabled': plot_params.get('qc_enabled', False),
            'qc_threshold': plot_params.get('qc_threshold', 5.0),
            'is_be_correction': plot_params.get('is_be_correction', False),
            'be_temp': plot_params.get('be_temp', 300.0),
            'is_smoothing': plot_params.get('is_smoothing', False),
            'smoothing_window': plot_params.get('smoothing_window', 15),
            'smoothing_poly': plot_params.get('smoothing_poly', 3),
            'is_baseline_als': plot_params.get('is_baseline_als', False),
            'als_lam': plot_params.get('als_lam', 10000),
            'als_p': plot_params.get('als_p', 0.005),
            'is_baseline_poly': False,
            'baseline_points': 50,
            'baseline_poly': 3,
            'normalization_mode': plot_params.get('normalization_mode', 'None'),
            'global_transform_mode': plot_params.get('global_transform_mode', '无'),
            'global_log_base': plot_params.get('global_log_base', '10'),
            'global_log_offset': plot_params.get('global_log_offset', 1.0),
            'global_sqrt_offset': plot_params.get('global_sqrt_offset', 0.0),
            'is_quadratic_fit': plot_params.get('is_quadratic_fit', False),
            'quadratic_degree': plot_params.get('quadratic_degree', 2),
            'is_derivative': plot_params.get('is_derivative', False),
            'global_y_offset': plot_params.get('global_y_offset', 0.0),
        }
        
        # 检查缓存（如果提供了文件路径和父窗口）
        if file_path and self.parent_window and hasattr(self.parent_window, 'plot_data_cache'):
            cached_data = self.parent_window.plot_data_cache.get_preprocess_data(file_path, preprocess_params)
            if cached_data is not None:
                x_cached, y_cached = cached_data
                # 检查X轴是否匹配
                if len(x_cached) == len(x):
                    return y_cached
        
        # 使用统一预处理函数
        y_processed = DataPreProcessor.preprocess_spectrum(x, y, preprocess_params)
        
        # 缓存结果（如果提供了文件路径）
        if file_path and self.parent_window and hasattr(self.parent_window, 'plot_data_cache'):
            self.parent_window.plot_data_cache.cache_preprocess_data(file_path, preprocess_params, (x.copy(), y_processed.copy()))
        
        return y_processed
    
    def _detect_and_plot_peaks(self, ax, x_data, y_detect, y_final, plot_params, color='blue'):
        """
        通用的波峰检测和绘制函数（从MplPlotWindow移动而来）
        x_data: X轴数据（波数）
        y_detect: 用于检测的Y数据（去除偏移）
        y_final: 用于绘制的Y数据（包含偏移）
        plot_params: 绘图参数字典
        color: 线条颜色（用于标记颜色默认值）
        """
        if not plot_params.get('peak_detection_enabled', False):
            return
        
        # 计算数据的统计信息，用于智能调整参数
        y_max = np.max(y_detect)
        y_min = np.min(y_detect)
        y_range = y_max - y_min
        y_mean = np.mean(y_detect)
        y_std = np.std(y_detect)
        
        # 构建find_peaks的参数
        peak_kwargs = {}
        
        # 基础参数（智能调整：如果用户设置的height为0，使用相对值；否则使用用户设置的值）
        peak_height = plot_params.get('peak_height_threshold', 0.0)
        # 如果height为0，使用相对值（数据最大值的0.01%，极低阈值以检测所有小峰）
        if peak_height == 0:
            if y_max > 0:
                peak_height = y_max * 0.0001  # 进一步降低到0.01%以检测所有小峰
            else:
                peak_height = abs(y_mean) + y_std * 0.05  # 如果最大值<=0，使用均值+0.05倍标准差
        # 如果用户设置了height，直接使用（不再强制降低，允许用户设置极小的值，甚至负数）
        # 只有当height明显不合理（大于数据范围）时才调整
        if peak_height > y_range * 2 and y_range > 0:
            peak_height = y_max * 0.0001  # 只有在明显不合理时才调整到0.01%
        # 允许任何值（包括极小的正负值），甚至可以为负（检测负峰）
        # 始终添加height参数（即使是负数），让find_peaks使用它
        peak_kwargs['height'] = peak_height
        
        peak_distance = plot_params.get('peak_distance_min', 10)
        # 如果distance为0，使用数据点数的0.1%（极低阈值以检测所有峰值）
        if peak_distance == 0:
            peak_distance = max(1, int(len(y_detect) * 0.001))  # 降低到0.1%，最小为1
        # 如果用户设置了distance，直接使用（允许设置为1以检测所有峰值）
        # 只有当distance明显不合理（大于数据长度的一半）时才调整
        if peak_distance > len(y_detect) * 0.5:
            peak_distance = max(1, int(len(y_detect) * 0.001))  # 只有在明显不合理时才调整
        # 确保distance至少为1（find_peaks的要求）
        peak_distance = max(1, peak_distance)
        # 如果用户设置了极小的height（包括负数），完全移除distance限制以检测所有峰值
        # 或者如果distance=1，也尝试不使用distance
        use_distance = True
        if 'height' in peak_kwargs:
            height_val = peak_kwargs['height']
            # 如果height是负数或极小值（小于数据最大值的0.1%），不使用distance
            if height_val < 0 or (y_max > 0 and height_val < y_max * 0.001):
                use_distance = False
            # 或者如果distance=1，也不使用distance
            elif peak_distance == 1:
                use_distance = False
        
        if use_distance:
            peak_kwargs['distance'] = peak_distance
        
        # 添加可选参数（如果设置了且不为0）
        peak_prominence = plot_params.get('peak_prominence', None)
        if peak_prominence is not None and peak_prominence != 0:
            # 如果prominence为0或未设置，不使用此参数（允许检测更多峰值）
            # 如果用户设置了prominence，直接使用（允许设置为极小的值）
            # 只有当prominence明显不合理（大于数据范围）时才调整
            if peak_prominence > y_range * 2 and y_range > 0:
                peak_prominence = y_range * 0.001  # 只有在明显不合理时才调整到0.1%
            peak_kwargs['prominence'] = peak_prominence
        
        # width、wlen、rel_height 只有在明确设置且大于0时才使用
        # 如果为0或未设置，不使用这些参数（避免过滤掉峰值）
        peak_width = plot_params.get('peak_width', None)
        if peak_width is not None and peak_width > 0:
            peak_kwargs['width'] = peak_width
        
        peak_wlen = plot_params.get('peak_wlen', None)
        if peak_wlen is not None and peak_wlen > 0:
            # 如果wlen太大，限制为数据长度的一半
            if peak_wlen > len(y_detect) * 0.5:
                peak_wlen = max(1, int(len(y_detect) * 0.3))
            peak_kwargs['wlen'] = peak_wlen
        
        peak_rel_height = plot_params.get('peak_rel_height', None)
        if peak_rel_height is not None and peak_rel_height > 0:
            peak_kwargs['rel_height'] = peak_rel_height
        
        try:
            # 确保至少有一个参数
            if len(peak_kwargs) == 0:
                # 如果没有参数，使用基于数据统计的智能默认值（极低阈值以检测所有小峰）
                # 只使用height，不使用distance，以检测所有峰值（包括相邻的）
                if y_max > 0:
                    peak_kwargs = {
                        'height': y_max * 0.0001  # 降低到0.01%
                    }
                else:
                    peak_kwargs = {
                        'height': abs(y_mean) + y_std * 0.05
                    }
            
            # 如果height是负数或极小值，尝试不使用任何限制参数，只使用height
            # 这样可以检测到所有峰值，包括噪音峰
            if 'height' in peak_kwargs:
                height_val = peak_kwargs['height']
                # 如果height是负数或极小值（小于数据最大值的0.1%），完全移除所有限制
                if height_val < 0 or (y_max > 0 and height_val < y_max * 0.001):
                    # 移除distance、prominence等限制参数，只保留height
                    filtered_kwargs = {'height': height_val}
                    # 如果width、wlen、rel_height被设置，也移除它们（避免过滤噪音峰）
                    for key in ['width', 'wlen', 'rel_height']:
                        if key in peak_kwargs:
                            # 只有在用户明确设置且>0时才保留
                            if peak_kwargs[key] > 0:
                                filtered_kwargs[key] = peak_kwargs[key]
                    peak_kwargs = filtered_kwargs
                # 如果height非常小（接近0），尝试完全不使用任何参数（检测所有局部最大值）
                elif height_val < y_max * 0.00001 and y_max > 0:
                    # 尝试不使用任何参数，让find_peaks检测所有局部最大值
                    try:
                        peaks_all, _ = find_peaks(y_detect)
                        if len(peaks_all) > 0:
                            peak_kwargs = {}  # 空参数，检测所有峰值
                    except:
                        pass
            
            # 如果peak_kwargs为空，使用极小的height值
            if len(peak_kwargs) == 0:
                if y_max > 0:
                    peak_kwargs = {'height': y_max * 0.00001}  # 极小的阈值
                else:
                    peak_kwargs = {'height': abs(y_mean) + y_std * 0.01}
            
            peaks, properties = find_peaks(y_detect, **peak_kwargs)
            
            if len(peaks) > 0:
                # 获取标记样式参数
                peak_marker_shape = plot_params.get('peak_marker_shape', 'x')
                peak_marker_size = plot_params.get('peak_marker_size', 10)
                peak_marker_color = plot_params.get('peak_marker_color', None)
                # 如果未指定颜色，使用线条颜色
                if peak_marker_color is None or peak_marker_color == '':
                    peak_marker_color = color
                
                # 绘制峰值标记
                ax.plot(x_data[peaks], y_final[peaks], peak_marker_shape, 
                       color=peak_marker_color, markersize=peak_marker_size)
                
                # 显示波数值
                if plot_params.get('peak_show_label', True):
                    peak_x_coords = x_data[peaks]
                    peak_y_coords = y_final[peaks]
                    
                    # 获取标签样式参数
                    label_font = plot_params.get('peak_label_font', 'Times New Roman')
                    label_size = plot_params.get('peak_label_size', 10)
                    label_color = plot_params.get('peak_label_color', 'black')
                    label_bold = plot_params.get('peak_label_bold', False)
                    label_rotation = plot_params.get('peak_label_rotation', 0.0)
                    
                    # 构建字体属性
                    font_props = {
                        'fontsize': label_size,
                        'color': label_color,
                        'fontfamily': label_font,
                        'ha': 'center',
                        'va': 'bottom'
                    }
                    if label_bold:
                        font_props['weight'] = 'bold'
                    if label_rotation != 0:
                        font_props['rotation'] = label_rotation
                    
                    # 为每个峰值添加波数标签（移除白色方框）
                    for px, py in zip(peak_x_coords, peak_y_coords):
                        # 格式化波数（保留1位小数）
                        wavenumber_str = f"{px:.1f}"
                        ax.text(px, py, wavenumber_str, **font_props)
        except Exception as e:
            # 如果峰值检测失败，打印错误信息以便调试
            print(f"波峰检测失败: {e}, 参数={peak_kwargs}, 数据范围=[{y_min:.2f}, {y_max:.2f}]")
            pass
    
    def _core_plot_spectrum(self, ax, plot_params):
        """
        核心绘图逻辑（从MplPlotWindow.update_plot移动并修改而来）
        接受ax参数，不再依赖self.canvas
        """
        # 延迟设置字体（首次绘图时）
        if not hasattr(self, '_fonts_setup'):
            from src.utils.fonts import setup_matplotlib_fonts
            setup_matplotlib_fonts()
            self._fonts_setup = True
        
        # 只清除axes内容，保持axes对象和布局
        ax.cla()
        
        # 获取当前组名（从plot_params获取）
        current_group_name = plot_params.get('current_group_name', '')
        
        # --- 提取基础参数 ---
        grouped_files_data = plot_params['grouped_files_data'] 
        control_data_list = plot_params.get('control_data_list', []) 
        individual_y_params = plot_params['individual_y_params'] 
        
        # --- 提取显示/模式参数 ---
        plot_mode = plot_params.get('plot_mode', 'Normal Overlay')
        show_y_values = plot_params.get('show_y_values', True)
        is_derivative = plot_params['is_derivative']
        x_axis_invert = plot_params['x_axis_invert'] 
        
        global_stack_offset = plot_params['global_stack_offset']
        global_scale_factor = plot_params['global_scale_factor']
        
        # --- 提取预处理参数 ---
        qc_enabled = plot_params.get('qc_enabled', False)
        qc_threshold = plot_params.get('qc_threshold', 5.0)
        is_baseline_als = plot_params.get('is_baseline_als', False)
        als_lam = plot_params.get('als_lam', 10000)
        als_p = plot_params.get('als_p', 0.005)
        is_smoothing = plot_params['is_smoothing']
        smoothing_window = plot_params['smoothing_window']
        smoothing_poly = plot_params['smoothing_poly']
        normalization_mode = plot_params['normalization_mode']
        
        # Bose-Einstein
        is_be_correction = plot_params.get('is_be_correction', False)
        be_temp = plot_params.get('be_temp', 300.0)
        
        # 全局动态变换和整体Y轴偏移
        global_transform_mode = plot_params.get('global_transform_mode', '无')
        global_log_base_text = plot_params.get('global_log_base', '10')
        global_log_base = float(global_log_base_text) if global_log_base_text == '10' else np.e
        global_log_offset = plot_params.get('global_log_offset', 1.0)
        global_sqrt_offset = plot_params.get('global_sqrt_offset', 0.0)
        global_y_offset = plot_params.get('global_y_offset', 0.0)
        
        # --- 提取出版样式参数 ---
        line_width = plot_params['line_width']
        line_style = plot_params['line_style']
        font_family = plot_params['font_family']
        axis_title_fontsize = plot_params['axis_title_fontsize']
        tick_label_fontsize = plot_params['tick_label_fontsize']
        legend_fontsize = plot_params.get('legend_fontsize', 10)
        
        show_legend = plot_params['show_legend']
        legend_frame = plot_params['legend_frame']
        legend_loc = plot_params['legend_loc']
        
        # 图例高级控制参数
        legend_ncol = plot_params.get('legend_ncol', 1)
        legend_columnspacing = plot_params.get('legend_columnspacing', 2.0)
        legend_labelspacing = plot_params.get('legend_labelspacing', 0.5)
        legend_handlelength = plot_params.get('legend_handlelength', 2.0)
        
        show_grid = plot_params['show_grid']
        grid_alpha = plot_params['grid_alpha']
        shadow_alpha = plot_params['shadow_alpha']
        main_title_text = plot_params.get('main_title_text', "").strip()
        
        # Aspect Ratio & Plot Style
        aspect_ratio = plot_params.get('aspect_ratio', 0.0)
        plot_style = plot_params.get('plot_style', 'line') # line, scatter
        
        # 设置字体 (仅影响当前 Figure)
        current_font = 'Times New Roman' if font_family == 'Times New Roman' else font_family
        
        # 使用 Viridis 调色板，或用户自定义
        custom_colors = plot_params.get('custom_colors', ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal', 'darkred'])
        
        # ==========================================
        # A. 预处理所有数据（对照组+组内数据），归一化前处理
        # ==========================================
        max_y_value = -np.inf 
        min_y_value = np.inf
        all_data_before_norm = []
        
        control_data_before_norm = []
        for i, control_data in enumerate(control_data_list):
            x_c = control_data['df']['Wavenumber'].values
            y_c = control_data['df']['Intensity'].values
            
            temp_y = y_c.astype(float)
            if is_be_correction: temp_y = DataPreProcessor.apply_bose_einstein_correction(x_c, temp_y, be_temp)
            if is_smoothing: temp_y = DataPreProcessor.apply_smoothing(temp_y, smoothing_window, smoothing_poly)
            if is_baseline_als: 
                b = DataPreProcessor.apply_baseline_als(temp_y, als_lam, als_p)
                temp_y = temp_y - b
                temp_y[temp_y < 0] = 0
            
            base_name = os.path.splitext(control_data['filename'])[0]
            control_data_before_norm.append({
                'x': x_c,
                'y': temp_y,
                'base_name': base_name,
                'label': control_data['label'],
                'type': 'control',
                'index': i
            })
            all_data_before_norm.append(temp_y)
        
        group_data_before_norm = []
        for file_path, x_data, y_data in grouped_files_data:
            y_proc = y_data.astype(float)
            
            if qc_enabled and np.max(y_proc) < qc_threshold:
                continue
            
            if is_be_correction:
                y_proc = DataPreProcessor.apply_bose_einstein_correction(x_data, y_proc, be_temp)
            if is_smoothing:
                y_proc = DataPreProcessor.apply_smoothing(y_proc, smoothing_window, smoothing_poly)
            if is_baseline_als:
                b = DataPreProcessor.apply_baseline_als(y_proc, als_lam, als_p)
                y_proc = y_proc - b
                y_proc[y_proc < 0] = 0
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            group_data_before_norm.append({
                'x': x_data,
                'y': y_proc,
                'base_name': base_name,
                'file_path': file_path,
                'type': 'group'
            })
            all_data_before_norm.append(y_proc)
        
        # 一起归一化（如果启用）
        if normalization_mode != 'none' and all_data_before_norm:
            all_y_array = np.array(all_data_before_norm)  # (n_samples, n_features)
            
            if normalization_mode == 'max':
                max_vals = np.max(all_y_array, axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1
                all_y_array = all_y_array / max_vals
            elif normalization_mode == 'area':
                areas = np.trapezoid(all_y_array, axis=1)
                areas = areas[:, np.newaxis]
                areas[areas == 0] = 1
                all_y_array = all_y_array / areas
            elif normalization_mode == 'snv':
                means = np.mean(all_y_array, axis=1, keepdims=True)
                stds = np.std(all_y_array, axis=1, keepdims=True)
                stds[stds == 0] = 1
                all_y_array = (all_y_array - means) / stds
            
            idx = 0
            for item in control_data_before_norm:
                item['y'] = all_y_array[idx]
                idx += 1
            for item in group_data_before_norm:
                item['y'] = all_y_array[idx]
                idx += 1
        
        # ==========================================
        # B. 处理对照组（归一化后）
        # ==========================================
        control_plot_data = []
        for item in control_data_before_norm:
            x_c = item['x']
            temp_y = item['y']
            base_name = item['base_name']
            i = item['index']
            
            ind_params = individual_y_params.get(base_name, {'scale': 1.0, 'offset': 0.0, 'transform': 'none', 'transform_params': {}})
            
            if global_transform_mode == '对数变换 (Log)':
                temp_y = DataPreProcessor.apply_log_transform(temp_y, base=global_log_base, offset=global_log_offset)
            elif global_transform_mode == '平方根变换 (Sqrt)':
                temp_y = DataPreProcessor.apply_sqrt_transform(temp_y, offset=global_sqrt_offset)
            
            transform_mode = ind_params.get('transform', 'none')
            transform_params = ind_params.get('transform_params', {})
            
            if transform_mode == 'log':
                temp_y = DataPreProcessor.apply_log_transform(temp_y, 
                    base=transform_params.get('base', 10),
                    offset=transform_params.get('offset', 1.0))
            elif transform_mode == 'sqrt':
                temp_y = DataPreProcessor.apply_sqrt_transform(temp_y,
                    offset=transform_params.get('offset', 0.0))
            
            temp_y = temp_y * global_scale_factor * ind_params['scale']
            
            if is_derivative:
                d1 = np.gradient(temp_y, x_c)
                temp_y = np.gradient(d1, x_c)
            
            temp_y = temp_y + global_y_offset
            
            final_y = temp_y + ind_params['offset'] + (i * global_stack_offset) 
            
            file_colors = plot_params.get('file_colors', {})
            if base_name in file_colors:
                color = file_colors[base_name]
            else:
                color = custom_colors[i % len(custom_colors)]
            
            label = item['label'] + " (Ref)"
            control_plot_data.append((x_c, final_y, label, color))
            
            if plot_style == 'line':
                ax.plot(x_c, final_y, label=label, color=color, linestyle='--', linewidth=line_width, alpha=0.7)
            else:  # scatter
                ax.plot(x_c, final_y, label=label, color=color, marker='.', linestyle='', markersize=line_width*3, alpha=0.7)
            
            max_y_value = max(max_y_value, np.max(final_y))
            min_y_value = min(min_y_value, np.min(final_y))

        # ==========================================
        # C. 处理分组数据（归一化后）
        # ==========================================
        processed_group_data = []
        for item in group_data_before_norm:
            x_data = item['x']
            y_clean = item['y']
            base_name = item['base_name']
            file_path = item['file_path']
            
            label = plot_params['legend_names'].get(base_name, base_name)
            ind_params = individual_y_params.get(base_name, {'scale': 1.0, 'offset': 0.0, 'transform': 'none', 'transform_params': {}})
            
            y_transformed = y_clean.copy()
            if global_transform_mode == '对数变换 (Log)':
                y_transformed = DataPreProcessor.apply_log_transform(y_transformed, base=global_log_base, offset=global_log_offset)
            elif global_transform_mode == '平方根变换 (Sqrt)':
                y_transformed = DataPreProcessor.apply_sqrt_transform(y_transformed, offset=global_sqrt_offset)
            
            transform_mode = ind_params.get('transform', 'none')
            transform_params = ind_params.get('transform_params', {})
            
            if transform_mode == 'log':
                y_transformed = DataPreProcessor.apply_log_transform(y_transformed,
                    base=transform_params.get('base', 10),
                    offset=transform_params.get('offset', 1.0))
            elif transform_mode == 'sqrt':
                y_transformed = DataPreProcessor.apply_sqrt_transform(y_transformed,
                    offset=transform_params.get('offset', 0.0))
            
            processed_group_data.append({
                'x': x_data,
                'y_raw_processed': y_transformed,
                'ind_scale': ind_params['scale'],
                'ind_offset': ind_params['offset'],
                'label': label,
                'file_path': file_path,
                'base_name': base_name
            })
            
        if not processed_group_data and not control_data_list:
            ax.text(0.5, 0.5, "No valid data (Check QC threshold / X-range)", transform=ax.transAxes, ha='center')
            return

        # ==========================================
        # D. 根据模式绘图
        # ==========================================
        current_plot_index = len(control_data_list)

        if plot_mode == 'Mean + Shadow' and processed_group_data:
            common_x = processed_group_data[0]['x']
            all_y = []
            for item in processed_group_data:
                y_scaled = item['y_raw_processed'] * item['ind_scale']
                all_y.append(y_scaled)
            
            all_y = np.array(all_y)
            mean_y = np.mean(all_y, axis=0)
            std_y = np.std(all_y, axis=0)
            
            mean_y *= global_scale_factor
            std_y *= global_scale_factor
            
            if is_derivative:
                d1 = np.gradient(mean_y, common_x)
                mean_y = np.gradient(d1, common_x)
                std_y = None
            
            mean_y = mean_y + global_y_offset
            
            color = custom_colors[current_plot_index % len(custom_colors)]
            
            rename_map = plot_params.get('legend_names', {})
            base_name = current_group_name
            if base_name in rename_map and rename_map[base_name]:
                base_display_name = rename_map[base_name]
            else:
                base_display_name = base_name
            
            mean_label_key = f"{base_name} Mean"
            std_label_key = f"{base_name} Std Dev"
            
            if mean_label_key in rename_map and rename_map[mean_label_key]:
                mean_label = rename_map[mean_label_key]
            else:
                mean_label = f"{base_display_name} Mean"
            
            if std_label_key in rename_map and rename_map[std_label_key]:
                std_label = rename_map[std_label_key]
            else:
                std_label = f"{base_display_name} Std Dev"
            
            group_color_params = plot_params.get('group_colors', {})
            if current_group_name in group_color_params:
                color = group_color_params[current_group_name]
            else:
                color = custom_colors[current_plot_index % len(custom_colors)]
            
            if is_derivative:
                ax.plot(common_x, mean_y, color=color, linewidth=line_width, label=mean_label)
            else:
                ax.plot(common_x, mean_y, color=color, linewidth=line_width, label=mean_label)
                if std_y is not None:
                    ax.fill_between(common_x, mean_y - std_y, mean_y + std_y, color=color, alpha=shadow_alpha, label=std_label)
            
            if plot_params.get('peak_detection_enabled', False) and not is_derivative:
                self._detect_and_plot_peaks(ax, common_x, mean_y, mean_y, plot_params, color=color)
            
            if is_derivative:
                max_y_value = max(max_y_value, np.max(mean_y))
                min_y_value = min(min_y_value, np.min(mean_y))
            else:
                max_y_value = max(max_y_value, np.max(mean_y + std_y))
                min_y_value = min(min_y_value, np.min(mean_y - std_y))

        else:
            for i, item in enumerate(processed_group_data):
                y_val = item['y_raw_processed'] * global_scale_factor * item['ind_scale']
                
                if is_derivative:
                    d1 = np.gradient(y_val, item['x'])
                    y_val = np.gradient(d1, item['x'])
                
                y_val = y_val + global_y_offset
                
                stack_idx = i + current_plot_index
                y_final = y_val + item['ind_offset'] + (stack_idx * global_stack_offset)
                
                base_name = item.get('base_name', os.path.splitext(os.path.basename(item.get('file_path', '')))[0] if 'file_path' in item else item.get('label', ''))
                
                file_colors = plot_params.get('file_colors', {})
                if base_name in file_colors:
                    color = file_colors[base_name]
                else:
                    color = custom_colors[stack_idx % len(custom_colors)]
                
                if plot_style == 'line':
                    ax.plot(item['x'], y_final, label=item['label'], color=color, linewidth=line_width, linestyle=line_style)
                else:  # scatter
                    ax.plot(item['x'], y_final, label=item['label'], color=color, marker='.', linestyle='', markersize=line_width*3)

                if plot_mode == 'Waterfall (Stacked)':
                    ax.text(item['x'][0], y_final[0], item['label'], fontsize=legend_fontsize-1, va='center', color=color)

                if plot_params.get('peak_detection_enabled', False) and not is_derivative:
                    y_detect = y_val
                    self._detect_and_plot_peaks(ax, item['x'], y_detect, y_final, plot_params, color)
                    
                max_y_value = max(max_y_value, np.max(y_final))
                min_y_value = min(min_y_value, np.min(y_final))

        # --- 坐标轴设置 ---
        if x_axis_invert:
            ax.invert_xaxis()
            
        if aspect_ratio > 0:
            ax.set_box_aspect(aspect_ratio) 
        else:
            ax.set_aspect('auto')

        # 批量绘图模式下不需要缩放状态检查，直接设置Y轴范围
        if max_y_value != -np.inf and min_y_value != np.inf:
            y_range = max_y_value - min_y_value
            new_ylim = (min_y_value - y_range * 0.05, max_y_value + y_range * 0.05)
            ax.set_ylim(new_ylim[0], new_ylim[1])

        vertical_lines = plot_params.get('vertical_lines', [])
        vertical_line_color = plot_params.get('vertical_line_color', 'gray')
        vertical_line_width = plot_params.get('vertical_line_width', 0.8)
        vertical_line_style = plot_params.get('vertical_line_style', ':')
        vertical_line_alpha = plot_params.get('vertical_line_alpha', 0.7)
        
        for line_x in vertical_lines:
            ax.axvline(x=line_x, color=vertical_line_color, linestyle=vertical_line_style, 
                      linewidth=vertical_line_width, alpha=vertical_line_alpha)

        # 绘制RRUFF光谱和参考线（这部分代码很长，保持原逻辑但使用ax而不是self.canvas.axes）
        rruff_spectra = plot_params.get('rruff_spectra', [])
        if rruff_spectra:
            from scipy.interpolate import interp1d
            
            # 获取当前数据的X轴范围（用于插值对齐）
            ref_x_data = None
            if processed_group_data:
                ref_x_data = processed_group_data[0]['x']
            elif control_plot_data:
                ref_x_data = control_plot_data[0][0]  # control_plot_data是(x, y, label, color)元组
            
            if ref_x_data is None:
                # 如果没有数据，使用当前axes的X轴范围
                xlim = ax.get_xlim()
                ref_x_data = np.linspace(xlim[0], xlim[1], 1000)
            
            current_x_min = ref_x_data.min()
            current_x_max = ref_x_data.max()
            
            # 获取堆叠偏移和样式参数
            rruff_color_index = len(processed_group_data) if processed_group_data else (len(control_data_list) if control_data_list else 0)
            rruff_colors = plot_params.get('custom_colors', ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal', 'darkred'])
            
            for rruff_idx, rruff_data in enumerate(rruff_spectra):
                rruff_x = rruff_data['x']
                rruff_y = rruff_data['y']
                rruff_name = rruff_data['name']
                matches = rruff_data.get('matches', [])
                
                # 插值对齐到当前X轴
                if len(rruff_x) > 1:
                    # 确定插值范围（取交集）
                    interp_x_min = max(current_x_min, rruff_x.min())
                    interp_x_max = min(current_x_max, rruff_x.max())
                    
                    if interp_x_min < interp_x_max:
                        # 创建插值函数
                        f_interp = interp1d(rruff_x, rruff_y, kind='linear', fill_value=0, bounds_error=False)
                        
                        # 使用参考X轴进行插值
                        mask = (ref_x_data >= interp_x_min) & (ref_x_data <= interp_x_max)
                        interp_x = ref_x_data[mask]
                        interp_y = f_interp(interp_x)
                        
                        if len(interp_x) == 0:
                            continue
                        
                        # 应用堆叠偏移
                        is_combination_phase = rruff_data.get('is_combination_phase', False)
                        combination_stack_offset = rruff_data.get('stack_offset', 0.0)
                        
                        rruff_ref_line_offset = plot_params.get('rruff_ref_line_offset', 0.0)
                        stack_idx = rruff_color_index + rruff_idx
                        
                        if is_combination_phase:
                            rruff_y_final = interp_y + combination_stack_offset
                        elif rruff_ref_line_offset != 0.0:
                            rruff_y_final = interp_y + (rruff_idx * rruff_ref_line_offset)
                        else:
                            rruff_y_final = interp_y + (stack_idx * global_stack_offset)
                        
                        # 选择颜色
                        rruff_color = rruff_colors[stack_idx % len(rruff_colors)]
                        
                        # 更新Y轴范围以包含RRUFF光谱
                        if len(rruff_y_final) > 0:
                            max_y_value = max(max_y_value, np.max(rruff_y_final))
                            min_y_value = min(min_y_value, np.min(rruff_y_final))
                        
                        # 绘制RRUFF光谱
                        if plot_style == 'line':
                            ax.plot(interp_x, rruff_y_final, label=f"RRUFF: {rruff_name}", 
                                   color=rruff_color, linewidth=line_width, linestyle='-', alpha=0.7)
                        else:  # scatter
                            ax.plot(interp_x, rruff_y_final, label=f"RRUFF: {rruff_name}", 
                                   color=rruff_color, marker='.', linestyle='', markersize=line_width*3, alpha=0.7)
                        
                        # 绘制参考线连接匹配的峰值
                        rruff_ref_lines_enabled = plot_params.get('rruff_ref_lines_enabled', True)
                        if matches and rruff_ref_lines_enabled:
                            ref_line_color = rruff_color
                            ref_line_style = vertical_line_style
                            ref_line_width = vertical_line_width
                            ref_line_alpha = vertical_line_alpha
                            
                            # 获取当前光谱的峰值位置并绘制参考线
                            data_items = processed_group_data if processed_group_data else []
                            if not data_items and control_plot_data:
                                for x_c, y_c, label_c, color_c in control_plot_data:
                                    for match in matches:
                                        query_peak, lib_peak, distance = match
                                        query_y_idx = np.argmin(np.abs(x_c - query_peak))
                                        query_y = y_c[query_y_idx]
                                        
                                        lib_y_idx = np.argmin(np.abs(interp_x - lib_peak))
                                        lib_y = rruff_y_final[lib_y_idx] if lib_y_idx < len(rruff_y_final) else rruff_y_final[-1]
                                        
                                        ax.plot([query_peak, lib_peak], [query_y, lib_y], 
                                               color=ref_line_color, linestyle=ref_line_style, 
                                               linewidth=ref_line_width, alpha=ref_line_alpha)
                                    break
                            else:
                                for item in data_items:
                                    for match in matches:
                                        query_peak, lib_peak, distance = match
                                        query_y_idx = np.argmin(np.abs(item['x'] - query_peak))
                                        y_val = item['y_raw_processed'][query_y_idx] * global_scale_factor * item['ind_scale']
                                        if is_derivative:
                                            y_val = item['y_raw_processed'][query_y_idx]
                                        y_val = y_val + global_y_offset
                                        stack_idx_item = current_plot_index + data_items.index(item)
                                        query_y = y_val + item['ind_offset'] + (stack_idx_item * global_stack_offset)
                                        
                                        lib_y_idx = np.argmin(np.abs(interp_x - lib_peak))
                                        lib_y = rruff_y_final[lib_y_idx] if lib_y_idx < len(rruff_y_final) else rruff_y_final[-1]
                                        
                                        ax.plot([query_peak, lib_peak], [query_y, lib_y], 
                                               color=ref_line_color, linestyle=ref_line_style, 
                                               linewidth=ref_line_width, alpha=ref_line_alpha)
                                    break
            
            # 在绘制完所有RRUFF光谱后，重新调整Y轴范围
            if rruff_spectra:
                if max_y_value != -np.inf and min_y_value != np.inf:
                    y_range = max_y_value - min_y_value
                    if y_range > 0:
                        new_ylim = (min_y_value - y_range * 0.05, max_y_value + y_range * 0.05)
                        ax.set_ylim(new_ylim[0], new_ylim[1])

        ylabel_final = "2nd Derivative" if is_derivative else plot_params['ylabel_text']
        if is_be_correction:
             ylabel_final = f"BE Corrected {ylabel_final} @ {be_temp}K"

        xlabel_fontsize = plot_params.get('xlabel_fontsize', axis_title_fontsize)
        xlabel_pad = plot_params.get('xlabel_pad', 10.0)
        xlabel_show = plot_params.get('xlabel_show', True)
        
        if xlabel_show:
            ax.set_xlabel(plot_params['xlabel_text'], fontsize=xlabel_fontsize, labelpad=xlabel_pad, fontfamily=current_font)
        
        ylabel_fontsize = plot_params.get('ylabel_fontsize', axis_title_fontsize)
        ylabel_pad = plot_params.get('ylabel_pad', 10.0)
        ylabel_show = plot_params.get('ylabel_show', True)
        
        if ylabel_show:
            ax.set_ylabel(ylabel_final, fontsize=ylabel_fontsize, labelpad=ylabel_pad, fontfamily=current_font)
        
        if not show_y_values:
            ax.set_yticks([])
        
        tick_direction = plot_params['tick_direction']
        tick_len_major = plot_params['tick_len_major']
        tick_len_minor = plot_params['tick_len_minor']
        tick_width = plot_params['tick_width']
        
        ax.tick_params(labelsize=tick_label_fontsize, direction=tick_direction, width=tick_width)
        ax.tick_params(which='major', length=tick_len_major)
        ax.tick_params(which='minor', length=tick_len_minor)
        
        for side in ['top', 'right', 'left', 'bottom']:
            if side in plot_params['border_sides']:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(plot_params['border_linewidth'])
            else:
                ax.spines[side].set_visible(False)
                
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
            
        if show_legend and plot_mode != 'Waterfall (Stacked)':
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            if font_family != 'SimHei':
                legend_font.set_family(font_family)
            else:
                legend_font.set_family('sans-serif')
            legend_font.set_size(legend_fontsize)
            
            ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                     ncol=legend_ncol, columnspacing=legend_columnspacing, 
                     labelspacing=legend_labelspacing, handlelength=legend_handlelength)
            
        main_title_stripped = main_title_text.strip()
        main_title_fontsize = plot_params.get('main_title_fontsize', axis_title_fontsize)
        main_title_pad = plot_params.get('main_title_pad', 10.0)
        main_title_show = plot_params.get('main_title_show', True)
        
        if main_title_stripped != "" and main_title_show:
            final_title = main_title_stripped
            ax.set_title(
                final_title, 
                fontsize=main_title_fontsize, 
                fontfamily=current_font,
                pad=main_title_pad
            )
        
        # 注意：不再调用self.canvas.draw()，由调用者负责
    
    def on_rruff_item_double_clicked(self, item):
        """双击RRUFF匹配项时添加到绘图"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        txt_basename = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not txt_basename:
            return
        
        item_data = item.data(Qt.ItemDataRole.UserRole)
        
        # 判断是组合匹配还是单物相匹配
        if isinstance(item_data, dict) and 'phases' in item_data:
            # 组合匹配
            if txt_basename not in self.selected_rruff_combinations:
                self.selected_rruff_combinations[txt_basename] = []
            
            # 检查是否已选中
            is_selected = False
            for sel_combo in self.selected_rruff_combinations[txt_basename]:
                if sel_combo['phases'] == item_data['phases']:
                    is_selected = True
                    break
            
            if is_selected:
                # 移除
                self.selected_rruff_combinations[txt_basename] = [
                    c for c in self.selected_rruff_combinations[txt_basename] 
                    if c['phases'] != item_data['phases']
                ]
            else:
                # 添加
                self.selected_rruff_combinations[txt_basename].append(item_data)
        else:
            # 单物相匹配
            name = item_data
            if name:
                if txt_basename not in self.selected_rruff_spectra:
                    self.selected_rruff_spectra[txt_basename] = set()
                
                if name in self.selected_rruff_spectra[txt_basename]:
                    self.selected_rruff_spectra[txt_basename].remove(name)
                else:
                    self.selected_rruff_spectra[txt_basename].add(name)
        
        self._update_plots_with_rruff()
    
    def on_rruff_item_clicked(self, item):
        """RRUFF项目点击事件（检测Ctrl键）"""
        # 检测是否按下了Ctrl键
        modifiers = QApplication.keyboardModifiers()
        is_ctrl_click = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        
        # 存储Ctrl键状态，供on_rruff_selection_changed使用
        self._is_ctrl_click = is_ctrl_click
    
    def on_rruff_selection_changed(self):
        """RRUFF选择改变时更新（区分单物相和组合匹配，普通点击覆盖，Ctrl+点击叠加）"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        txt_basename = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not txt_basename:
            return
        
        selected_list_items = self.rruff_match_list.selectedItems()
        
        # 初始化
        if txt_basename not in self.selected_rruff_spectra:
            self.selected_rruff_spectra[txt_basename] = set()
        if txt_basename not in self.selected_rruff_combinations:
            self.selected_rruff_combinations[txt_basename] = []
        
        # 检测是否按下了Ctrl键（从on_rruff_item_clicked获取）
        is_ctrl_click = getattr(self, '_is_ctrl_click', False)
        
        # 如果不是Ctrl+点击，清除旧选择（覆盖模式）
        if not is_ctrl_click:
            self.selected_rruff_spectra[txt_basename] = set()
            self.selected_rruff_combinations[txt_basename] = []
        
        # 分别处理单物相和组合匹配
        selected_spectra = set(self.selected_rruff_spectra[txt_basename])  # 复制现有选择（如果Ctrl+点击）
        selected_combinations = list(self.selected_rruff_combinations[txt_basename])  # 复制现有选择（如果Ctrl+点击）
        
        # 添加新选择的项目
        for item in selected_list_items:
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if item_data:
                # 判断是组合匹配还是单物相匹配
                if isinstance(item_data, dict) and 'phases' in item_data:
                    # 组合匹配：检查是否已存在
                    if not any(c['phases'] == item_data['phases'] for c in selected_combinations):
                        selected_combinations.append(item_data)
                else:
                    # 单物相匹配：字符串名称
                    selected_spectra.add(item_data)
        
        # 更新选择
        self.selected_rruff_spectra[txt_basename] = selected_spectra
        self.selected_rruff_combinations[txt_basename] = selected_combinations
        
        self._update_plots_with_rruff()
        
        # 重置Ctrl键状态
        self._is_ctrl_click = False
    
    def on_file_selected(self):
        """文件选择改变时的回调（支持多选）"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        # 获取所有选中的文件
        selected_basenames = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        
        if len(selected_basenames) == 1:
            # 单个文件：显示光谱图+镜下光学图
            basename = selected_basenames[0]
            self.plot_single_spectrum(basename)
            # 如果启用自动RRUFF匹配，则自动执行一次匹配（单物相+多物相）
            if hasattr(self, "auto_rruff_match_check") and self.auto_rruff_match_check.isChecked():
                self._auto_match_rruff_for_file(basename)
                self._auto_match_rruff_combination_for_file(basename)
        else:
            # 多个文件：显示多个子图（上下排列）
            self.plot_multiple_spectra(selected_basenames)
    
    def _on_rruff_ref_lines_enabled_changed(self, state):
        """RRUFF参考线启用状态改变时自动更新绘图"""
        self._update_plots_with_rruff()
    
    def _update_plots_with_rruff(self):
        """更新绘图以包含选中的RRUFF光谱（不触发自动匹配）"""
        # 获取当前选中的文件，直接调用绘图函数，不触发自动匹配
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        # 获取所有选中的文件
        selected_basenames = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        
        if len(selected_basenames) == 1:
            # 单个文件：直接调用绘图函数，不触发自动匹配
            basename = selected_basenames[0]
            self.plot_single_spectrum(basename)
        else:
            # 多个文件：显示多个子图
            self.plot_multiple_spectra(selected_basenames)

    def _ensure_rruff_matches_for_all_files(self):
        """
        确保当前文件夹中每个 txt 文件都已经有：
        - 单物相 RRUFF 匹配结果 (rruff_match_results)
        - 多物相组合匹配结果 (rruff_combination_results)
        如果某个文件尚未匹配，则静默执行一次匹配。
        """
        if not self.rruff_loader or not self.rruff_loader.library_spectra:
            return
        if not self.txt_files:
            return

        # 从主窗口获取绘图参数（用于组合匹配的预处理）
        plot_params = self.get_parent_plot_params()
        if not plot_params:
            return

        # 为每一个 txt 文件生成 basename，并按自然顺序遍历
        basenames = []
        for f in self.txt_files:
            base = os.path.splitext(os.path.basename(f))[0]
            basenames.append(base)
        basenames = sorted(set(basenames))

        for basename in basenames:
            # 单物相匹配（如果还没有结果）
            if basename not in self.rruff_match_results or not self.rruff_match_results[basename]:
                self._auto_match_rruff_for_file(basename)
            # 组合匹配（如果还没有结果）
            if basename not in self.rruff_combination_results or not self.rruff_combination_results[basename]:
                self._auto_match_rruff_combination_for_file(basename)

    def _auto_match_rruff_for_file(self, txt_basename: str):
        """
        自动为指定文件执行一次RRUFF单光谱匹配，但不弹出任何提示框。
        仅在自动匹配开关勾选时由 on_file_selected 调用。
        """
        # 需要RRUFF库
        if not self.rruff_loader or not self.rruff_loader.library_spectra:
            return

        # 从文件列表中找到对应的 txt 文件路径
        txt_file = None
        for f in self.txt_files:
            if os.path.splitext(os.path.basename(f))[0] == txt_basename:
                txt_file = f
                break
        if not txt_file:
            return

        plot_params = self.get_parent_plot_params()
        if not plot_params:
            return

        try:
            # 读取光谱数据并预处理
            x, y = self.data_controller.read_data(
                txt_file,
                plot_params['skip_rows'],
                plot_params['x_min_phys'],
                plot_params['x_max_phys']
            )
            y_proc = self._preprocess_spectrum(x, y, plot_params, file_path=txt_file)

            # 按主菜单峰值参数检测峰
            peak_height = plot_params.get('peak_height_threshold', 0.0)
            peak_distance = plot_params.get('peak_distance_min', 10)
            peak_prominence = plot_params.get('peak_prominence', None)

            y_max = np.max(y_proc) if len(y_proc) > 0 else 0
            y_min = np.min(y_proc) if len(y_proc) > 0 else 0
            y_range = y_max - y_min

            peak_kwargs = {}
            if peak_height == 0:
                if y_max > 0:
                    peak_height = y_max * 0.001
                else:
                    peak_height = 0
            if peak_height > y_range * 2 and y_range > 0:
                peak_height = y_max * 0.001
            if peak_height != 0:
                peak_kwargs['height'] = peak_height

            if peak_distance == 0:
                peak_distance = max(1, int(len(y_proc) * 0.001))
            if peak_distance > len(y_proc) * 0.5:
                peak_distance = max(1, int(len(y_proc) * 0.001))
            peak_distance = max(1, peak_distance)
            peak_kwargs['distance'] = peak_distance

            if peak_prominence is not None and peak_prominence != 0:
                if peak_prominence > y_range * 2 and y_range > 0:
                    peak_prominence = y_range * 0.001
                peak_kwargs['prominence'] = peak_prominence

            try:
                peaks, _ = find_peaks(y_proc, **peak_kwargs)
            except Exception:
                peaks, _ = find_peaks(
                    y_proc,
                    height=y_max * 0.001 if y_max > 0 else 0,
                    distance=max(1, int(len(y_proc) * 0.001)),
                )

            peak_wavenumbers = x[peaks] if len(peaks) > 0 else np.array([])

            # 计算排除列表
            excluded_names = list(self.spectrum_exclusions.get(txt_basename, []))
            for i in range(self.global_exclusion_list.count()):
                item = self.global_exclusion_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    name = item.text()
                    if name not in excluded_names:
                        excluded_names.append(name)

            # 更新容差并执行匹配
            tolerance = self.rruff_match_tolerance_spin.value() if hasattr(self, 'rruff_match_tolerance_spin') else 5.0
            self.peak_matcher.tolerance = tolerance
            matches = self.peak_matcher.find_best_matches(
                x, y_proc, peak_wavenumbers, self.rruff_loader,
                top_k=20,
                excluded_names=excluded_names if excluded_names else None,
            )
            self.rruff_match_results[txt_basename] = matches

            # 如果当前左侧选中的就是这个文件，刷新匹配结果列表
            selected_items = self.file_list.selectedItems()
            if selected_items and selected_items[0].data(Qt.ItemDataRole.UserRole) == txt_basename:
                self.rruff_match_list.clear()
                for match in matches:
                    name = match.get("name", "")
                    score = float(match.get("match_score", 0.0))
                    item = QListWidgetItem(f"{name} (score={score:.3f})")
                    item.setData(Qt.ItemDataRole.UserRole, name)
                    self.rruff_match_list.addItem(item)

        except Exception as e:
            # 自动模式静默失败，仅打印日志
            print(f"[Auto RRUFF Match] 自动匹配 {txt_basename} 失败: {e}")

    def _ensure_rruff_matches_for_all_files(self):
        """
        为当前文件夹中的所有 txt 文件自动完成：
        1）单物相匹配（rruff_match_results）
        2）多物相组合匹配（rruff_combination_results）
        该过程静默运行，仅在控制台打印简要日志。
        """
        if not self.rruff_loader or not self.rruff_loader.library_spectra:
            return
        plot_params = self.get_parent_plot_params()
        if not plot_params or not self.txt_files:
            return

        print("[RRUFF] 开始为所有文件批量匹配（单物相 + 多物相组合）...")
        
        # 创建进度对话框
        total_files = len(self.txt_files)
        progress = QProgressDialog("正在批量匹配RRUFF光谱...", "取消", 0, total_files, self)
        progress.setWindowTitle("RRUFF批量匹配")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        for idx, txt_file in enumerate(self.txt_files):
            if progress.wasCanceled():
                break
            
            progress.setValue(idx)
            progress.setLabelText(f"正在匹配: {os.path.basename(txt_file)} ({idx+1}/{total_files})")
            QApplication.processEvents()
            basename = os.path.splitext(os.path.basename(txt_file))[0]
            try:
                # 读取光谱数据
                x, y = self.data_controller.read_data(
                    txt_file,
                    plot_params['skip_rows'],
                    plot_params['x_min_phys'],
                    plot_params['x_max_phys']
                )
                # 预处理（传入文件路径以支持缓存）
                y_proc = self._preprocess_spectrum(x, y, plot_params, file_path=txt_file)

                # 峰值检测与peak参数，与 match_rruff_spectra / match_rruff_combination 保持一致
                peak_height = plot_params.get('peak_height_threshold', 0.0)
                peak_distance = plot_params.get('peak_distance_min', 10)
                peak_prominence = plot_params.get('peak_prominence', None)

                y_max = np.max(y_proc) if len(y_proc) > 0 else 0
                y_min = np.min(y_proc) if len(y_proc) > 0 else 0
                y_range = y_max - y_min

                peak_kwargs = {}
                if peak_height == 0:
                    if y_max > 0:
                        peak_height = y_max * 0.001
                    else:
                        peak_height = 0
                if peak_height > y_range * 2 and y_range > 0:
                    peak_height = y_max * 0.001
                if peak_height != 0:
                    peak_kwargs['height'] = peak_height

                if peak_distance == 0:
                    peak_distance = max(1, int(len(y_proc) * 0.001))
                if peak_distance > len(y_proc) * 0.5:
                    peak_distance = max(1, int(len(y_proc) * 0.001))
                peak_distance = max(1, peak_distance)
                peak_kwargs['distance'] = peak_distance

                if peak_prominence is not None and peak_prominence != 0:
                    if peak_prominence > y_range * 2 and y_range > 0:
                        peak_prominence = y_range * 0.001
                    peak_kwargs['prominence'] = peak_prominence

                try:
                    peaks, _ = find_peaks(y_proc, **peak_kwargs)
                except Exception:
                    peaks, _ = find_peaks(
                        y_proc,
                        height=y_max * 0.001 if y_max > 0 else 0,
                        distance=max(1, int(len(y_proc) * 0.001)),
                    )

                peak_wavenumbers = x[peaks] if len(peaks) > 0 else np.array([])

                # 更新 RRUFF 库的峰值检测参数
                peak_detection_params = {
                    'peak_height_threshold': plot_params.get('peak_height_threshold', 0.0),
                    'peak_distance_min': plot_params.get('peak_distance_min', 10),
                    'peak_prominence': plot_params.get('peak_prominence', None),
                    'peak_width': plot_params.get('peak_width', None),
                    'peak_wlen': plot_params.get('peak_wlen', None),
                    'peak_rel_height': plot_params.get('peak_rel_height', None),
                }
                if self.rruff_loader.peak_detection_params != peak_detection_params:
                    for name, spectrum in self.rruff_loader.library_spectra.items():
                        if 'y_raw' in spectrum:
                            spectrum['peaks'] = self.rruff_loader._detect_peaks(
                                spectrum['x'], spectrum['y'],
                                peak_detection_params=peak_detection_params
                            )
                    self.rruff_loader.peak_detection_params = peak_detection_params

                # 计算排除列表
                excluded_names = list(self.spectrum_exclusions.get(basename, []))
                for i in range(self.global_exclusion_list.count()):
                    item = self.global_exclusion_list.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        name = item.text()
                        if name not in excluded_names:
                            excluded_names.append(name)

                # 单物相匹配（使用并行处理加速）
                tolerance = self.rruff_match_tolerance_spin.value() if hasattr(self, 'rruff_match_tolerance_spin') else 5.0
                self.peak_matcher.tolerance = tolerance
                
                # 定义单物相匹配的进度回调
                def single_progress_callback(current, total, message):
                    if progress.wasCanceled():
                        return
                    progress.setLabelText(f"正在匹配: {basename} - 单物相 ({current}/{total})")
                    QApplication.processEvents()
                
                # 检查缓存
                cache_key = self._get_match_cache_key(basename, x, y_proc, peak_wavenumbers, excluded_names, 'single')
                if cache_key in self._match_cache and 'single' in self._match_cache[cache_key]:
                    print(f"[缓存] 使用缓存的单物相匹配结果: {basename}")
                    single_matches = self._match_cache[cache_key]['single']
                else:
                    single_matches = self.peak_matcher.find_best_matches(
                        x, y_proc, peak_wavenumbers, self.rruff_loader,
                        top_k=100,  # 增加top_k以获取更多结果
                        excluded_names=excluded_names if excluded_names else None,
                        progress_callback=single_progress_callback,
                        max_workers=32,  # 充分利用32线程CPU
                    )
                    # 保存到缓存
                    if cache_key not in self._match_cache:
                        self._match_cache[cache_key] = {}
                    self._match_cache[cache_key]['single'] = single_matches
                
                self.rruff_match_results[basename] = single_matches

                # 多物相组合匹配
                use_gpu = False
                try:
                    import cupy as cp  # noqa: F401
                    use_gpu = True
                except ImportError:
                    try:
                        import torch  # noqa: F401
                        if torch.cuda.is_available():
                            use_gpu = True
                    except ImportError:
                        pass

                # 定义进度回调函数
                def combo_progress_callback(current, total, message):
                    if progress.wasCanceled():
                        return
                    progress.setLabelText(f"正在匹配: {basename} - {message} ({current}/{total})")
                    QApplication.processEvents()
                
                # 检查缓存
                cache_key_combo = self._get_match_cache_key(basename, x, y_proc, peak_wavenumbers, excluded_names, 'combo')
                if cache_key_combo in self._match_cache and 'combo' in self._match_cache[cache_key_combo]:
                    print(f"[缓存] 使用缓存的多物相匹配结果: {basename}")
                    combinations = self._match_cache[cache_key_combo]['combo']
                else:
                    # 自动确定最大物相数量
                    num_peaks = len(peak_wavenumbers)
                    num_candidates = len(self.rruff_loader.library_spectra) - len(excluded_names) if excluded_names else len(self.rruff_loader.library_spectra)
                    auto_max_phases = min(max(num_peaks // 3, 3), num_candidates, 10)
                    
                    combinations = self.peak_matcher.find_best_combination_matches(
                        x, y_proc, peak_wavenumbers, self.rruff_loader,
                        max_phases=auto_max_phases, top_k=None,  # top_k=None表示不限制结果数量
                        excluded_names=excluded_names if excluded_names else None,
                        use_gpu=use_gpu,
                        progress_callback=combo_progress_callback,
                    )
                    # 按需过滤同一物相的不同变种
                    if getattr(self, "rruff_filter_variants_check", None) is not None and self.rruff_filter_variants_check.isChecked():
                        combinations = self._filter_combinations_by_variants(combinations)
                    # 保存到缓存
                    if cache_key_combo not in self._match_cache:
                        self._match_cache[cache_key_combo] = {}
                    self._match_cache[cache_key_combo]['combo'] = combinations
                
                self.rruff_combination_results[basename] = combinations

            except Exception as e:
                print(f"[RRUFF] 文件 {basename} 匹配失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        progress.setValue(total_files)
        progress.close()
        print("[RRUFF] 批量匹配完成。")

    # --- RRUFF 结果总览窗口 ---
    def open_rruff_summary_window(self):
        """打开RRUFF匹配结果总览窗口（表格形式，可导出/绘图）"""
        # 打开总览前，先为所有文件批量完成单物相和组合匹配
        self._ensure_rruff_matches_for_all_files()

        if self.rruff_summary_window is None:
            from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QComboBox, QSplitter
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            self.rruff_summary_window = QDialog(self)
            self.rruff_summary_window.setWindowTitle("RRUFF 匹配结果总览")
            self.rruff_summary_window.setMinimumSize(900, 600)
            # 添加最小化和最大化按钮
            self.rruff_summary_window.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint
            )

            layout = QVBoxLayout(self.rruff_summary_window)

            splitter = QSplitter(Qt.Orientation.Horizontal)

            # 左侧：表格
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)

            self.rruff_summary_table = QTableWidget()
            # 扩展表格：支持显示多个匹配结果（最多30条）
            # 列结构：文件名 + Top 30单物相（名称+分数） + Top 30多物相（组合+分数）
            max_results = 30
            self.max_display_results = max_results
            
            # 计算列数：1(文件名) + max_results*2(单物相名称+分数) + max_results*2(多物相组合+分数) + 1(备注)
            total_cols = 1 + max_results * 2 + max_results * 2 + 1
            self.rruff_summary_table.setColumnCount(total_cols)
            
            # 构建表头
            headers = ["文件名"]
            # 单物相列
            for i in range(max_results):
                headers.append(f"单物相{i+1}")
                headers.append(f"分数{i+1}")
            # 多物相列
            for i in range(max_results):
                headers.append(f"多物相{i+1}")
                headers.append(f"分数{i+1}")
            headers.append("备注")
            
            self.rruff_summary_table.setHorizontalHeaderLabels(headers)
            self.rruff_summary_table.horizontalHeader().setStretchLastSection(True)
            # 选中某行时更新右侧图像
            self.rruff_summary_table.currentCellChanged.connect(self.update_rruff_fig_preview)
            left_layout.addWidget(self.rruff_summary_table)

            left_widget.setLayout(left_layout)
            splitter.addWidget(left_widget)

            # 右侧：图像预览（Matplotlib canvas + toolbar）
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)

            self.rruff_fig_canvas = MplCanvas(self, width=5, height=4, dpi=100)
            self.rruff_fig_toolbar = NavigationToolbar(self.rruff_fig_canvas, right_widget)

            right_layout.addWidget(self.rruff_fig_toolbar)
            right_layout.addWidget(self.rruff_fig_canvas)

            right_widget.setLayout(right_layout)
            splitter.addWidget(right_widget)

            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 3)

            layout.addWidget(splitter)

            # 按钮行：图像类型选择 + 分页控制 + 导出表格 / 导出图像
            btn_layout = QHBoxLayout()
            btn_layout.addWidget(QLabel("图像类型:"))
            self.rruff_fig_style_combo = QComboBox()
            self.rruff_fig_style_combo.addItems(
                [
                    "柱状图：最佳单物相 score",
                    "箱线图：组合 match_score 分布",
                    "热图：组合 match_score (文件×组合序号)",
                    "2D条形图：当前样品矿物成分+光学图",
                    "2D总览：所有样品矿物成分+光学图(Top N)",
                    "3D柱状图：所有匹配结果（Top 30）",
                ]
            )
            # 当图像类型改变时，根据当前选中行刷新预览图像
            self.rruff_fig_style_combo.currentTextChanged.connect(
                lambda _text: self._refresh_rruff_fig_preview_by_style()
            )
            btn_layout.addWidget(self.rruff_fig_style_combo)

            # --- 分页控制（用于 2D总览 模式） ---
            from PyQt6.QtWidgets import QSpinBox
            self.rruff_overview_page_size = 2  # 每页显示样品数（每页2个样品：2行）
            self.rruff_overview_page = 0

            btn_layout.addWidget(QLabel("总览页:"))
            self.rruff_overview_page_spin = QSpinBox()
            self.rruff_overview_page_spin.setMinimum(0)
            self.rruff_overview_page_spin.setMaximum(0)  # 打开窗口后会根据样品数更新
            self.rruff_overview_page_spin.setValue(0)
            self.rruff_overview_page_spin.setPrefix("第")
            self.rruff_overview_page_spin.setSuffix("页")
            self.rruff_overview_page_spin.valueChanged.connect(self._on_rruff_overview_page_changed)
            btn_layout.addWidget(self.rruff_overview_page_spin)

            btn_layout.addStretch()

            self.btn_export_rruff_table = QPushButton("导出表格 (CSV)")
            self.btn_export_rruff_table.clicked.connect(self.export_rruff_summary_table)
            self.btn_export_rruff_fig = QPushButton("导出图像 (PDF)")
            self.btn_export_rruff_fig.clicked.connect(self.export_rruff_summary_figure)
            btn_layout.addWidget(self.btn_export_rruff_table)
            btn_layout.addWidget(self.btn_export_rruff_fig)
            layout.addLayout(btn_layout)

        # 每次打开前刷新数据
        self.populate_rruff_summary_table()
        self.rruff_summary_window.show()
        self.rruff_summary_window.raise_()
        self.rruff_summary_window.activateWindow()

    def populate_rruff_summary_table(self):
        """根据 rruff_match_results 和 rruff_combination_results 填充总览表格（显示最多30条结果）。"""
        from PyQt6.QtWidgets import QTableWidgetItem
        from PyQt6.QtCore import Qt
        table = self.rruff_summary_table
        # 收集所有出现过结果的文件名
        all_keys = set(self.rruff_match_results.keys()) | set(self.rruff_combination_results.keys())
        keys_sorted = sorted(all_keys)

        table.setRowCount(len(keys_sorted))
        max_results = getattr(self, 'max_display_results', 30)

        # --- 更新 2D总览 分页控件的最大页数 ---
        if hasattr(self, "rruff_overview_page_spin"):
            # 仅统计有多物相结果的样品数量
            combo_names = [k for k in keys_sorted if k in self.rruff_combination_results and self.rruff_combination_results[k]]
            total_combo = len(combo_names)
            page_size = getattr(self, "rruff_overview_page_size", 6)
            if total_combo > 0:
                max_page = max(0, (total_combo - 1) // page_size)
            else:
                max_page = 0
            self.rruff_overview_page_spin.blockSignals(True)
            self.rruff_overview_page_spin.setMaximum(max_page)
            # 保证当前页在合法范围内
            if self.rruff_overview_page > max_page:
                self.rruff_overview_page = max_page
            self.rruff_overview_page_spin.setValue(self.rruff_overview_page)
            self.rruff_overview_page_spin.blockSignals(False)

        for row, basename in enumerate(keys_sorted):
            col_idx = 0
            
            # 文件名（简化显示，完整名称在工具提示中）
            display_basename = basename[:30] + "..." if len(basename) > 30 else basename
            basename_item = QTableWidgetItem(display_basename)
            # 在 UserRole 中保存完整 basename，供图像预览等功能使用
            basename_item.setData(Qt.ItemDataRole.UserRole, basename)
            basename_item.setToolTip(f"完整文件名: {basename}")
            table.setItem(row, col_idx, basename_item)
            col_idx += 1

            # 单物相匹配结果（最多30条）
            single_matches = self.rruff_match_results.get(basename, [])[:max_results]
            for i, match in enumerate(single_matches):
                match_name = match.get("name", "")
                match_score = float(match.get("match_score", 0.0))
                # 简化名称显示
                display_name = match_name[:20] + "..." if len(match_name) > 20 else match_name
                name_item = QTableWidgetItem(display_name)
                name_item.setToolTip(f"完整名称: {match_name}\n匹配分数: {match_score:.3f}")
                table.setItem(row, col_idx, name_item)
                col_idx += 1
                score_item = QTableWidgetItem(f"{match_score:.3f}")
                table.setItem(row, col_idx, score_item)
                col_idx += 1
            
            # 填充剩余的单物相列
            for i in range(len(single_matches), max_results):
                table.setItem(row, col_idx, QTableWidgetItem(""))
                col_idx += 1
                table.setItem(row, col_idx, QTableWidgetItem(""))
                col_idx += 1

            # 多物相组合匹配结果（最多30条）
            combo_matches = self.rruff_combination_results.get(basename, [])[:max_results]
            for i, combo in enumerate(combo_matches):
                phases = combo.get("phases", [])
                ratios = combo.get("ratios", [])
                match_score = float(combo.get("match_score", 0.0))
                
                # 物相+比例字符串（简化显示）
                combo_parts = []
                full_combo_parts = []
                for p, r in zip(phases, ratios):
                    display_p = p[:12] + "..." if len(p) > 12 else p
                    combo_parts.append(f"{display_p}({r:.2f})")
                    full_combo_parts.append(f"{p} ({r:.2f})")
                combo_str = "+".join(combo_parts)
                full_combo_str = " + ".join(full_combo_parts)
                
                combo_item = QTableWidgetItem(combo_str)
                combo_item.setToolTip(f"完整组合: {full_combo_str}\n综合分数: {match_score:.3f}")
                table.setItem(row, col_idx, combo_item)
                col_idx += 1
                score_item = QTableWidgetItem(f"{match_score:.3f}")
                table.setItem(row, col_idx, score_item)
                col_idx += 1
            
            # 填充剩余的多物相列
            for i in range(len(combo_matches), max_results):
                table.setItem(row, col_idx, QTableWidgetItem(""))
                col_idx += 1
                table.setItem(row, col_idx, QTableWidgetItem(""))
                col_idx += 1

            # 备注列
            table.setItem(row, col_idx, QTableWidgetItem(""))

        # 调整列宽（文件名列固定宽度，其他列自动调整）
        table.resizeColumnsToContents()
        # 设置文件名列的最小宽度
        table.setColumnWidth(0, 150)
        table.resizeRowsToContents()

        # 默认选中第一行并刷新预览图像
        if keys_sorted:
            table.setCurrentCell(0, 0)
            self.update_rruff_fig_preview(0, 0, -1, -1)

    def _refresh_rruff_fig_preview_by_style(self):
        """当图像类型改变时，根据当前选中的行刷新预览图像。"""
        if not hasattr(self, "rruff_summary_table"):
            return
        row = self.rruff_summary_table.currentRow()
        if row < 0:
            return
        # 使用当前行、占位列索引调用主预览函数
        self.update_rruff_fig_preview(row, 0, -1, -1)

    def _on_rruff_overview_page_changed(self, value: int):
        """当 2D总览 页码改变时，刷新预览图像。"""
        self.rruff_overview_page = max(0, int(value))
        # 仅当当前图像类型为 2D总览 时刷新
        if hasattr(self, "rruff_fig_style_combo") and self.rruff_fig_style_combo.currentText().startswith("2D总览"):
            # 使用当前选中行触发刷新
            if hasattr(self, "rruff_summary_table"):
                row = self.rruff_summary_table.currentRow()
                if row < 0 and self.rruff_summary_table.rowCount() > 0:
                    row = 0
                if row >= 0:
                    self.update_rruff_fig_preview(row, 0, -1, -1)

    def update_rruff_fig_preview(self, current_row, current_column, previous_row, previous_column):
        """根据当前选中的行和图像类型，在内置canvas里绘制预览图像。"""
        import matplotlib.pyplot as plt
        import numpy as np
        from PyQt6.QtCore import Qt

        if not hasattr(self, "rruff_fig_canvas"):
            return
        table = self.rruff_summary_table
        if current_row < 0 or current_row >= table.rowCount():
            return

        # 当前选中的文件名（优先从 UserRole 读取完整 basename）
        item = table.item(current_row, 0)
        if not item:
            return
        basename = item.data(Qt.ItemDataRole.UserRole) or item.text()

        style_text = self.rruff_fig_style_combo.currentText() if hasattr(self, "rruff_fig_style_combo") else ""

        fig = self.rruff_fig_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        # 设置出版级样式
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.0,
        })

        if style_text.startswith("柱状图"):
            # 全部样品的最佳匹配结果柱状图（学术标准）
            # x 轴：样品（文件名），y 轴：score
            basenames = sorted(set(self.rruff_match_results.keys()) | set(self.rruff_combination_results.keys()))
            if basenames:
                single_scores = []
                combo_scores = []
                labels = []
                for name in basenames:
                    labels.append(name)
                    single_matches = self.rruff_match_results.get(name, [])
                    combo_matches = self.rruff_combination_results.get(name, [])
                    single_scores.append(float(single_matches[0].get("match_score", 0.0)) if single_matches else 0.0)
                    combo_scores.append(float(combo_matches[0].get("match_score", 0.0)) if combo_matches else 0.0)

                x = np.arange(len(labels))
                width = 0.35
                ax.bar(x - width / 2, single_scores, width, label="Single", color="steelblue", edgecolor="black", linewidth=0.8)
                ax.bar(x + width / 2, combo_scores, width, label="Combo", color="darkorange", edgecolor="black", linewidth=0.8)

                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Match Score", fontsize=10)
                ax.set_xlabel("Sample", fontsize=10)
                ax.set_title("Best Match Scores for All Samples", fontsize=11)
                ax.grid(axis="y", alpha=0.3)
                ax.legend(fontsize=8)

        elif style_text.startswith("箱线图"):
            # 当前文件所有组合 match_score 的分布
            combos = self.rruff_combination_results.get(basename, [])
            scores = [c.get("match_score", 0.0) for c in combos]
            if scores:
                bp = ax.boxplot(
                    [scores],
                    labels=[basename],
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.2},
                    boxprops={"linewidth": 1.0},
                    whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0},
                )
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp["boxes"])))
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c)
                ax.set_ylabel("Combination Match Score", fontsize=10)
                ax.set_title(f"{basename} - Combination Score Distribution", fontsize=11)
                ax.tick_params(axis="x", rotation=0, labelsize=9)
                ax.grid(axis="y", alpha=0.3)

        elif style_text.startswith("2D条形图"):
            # 2D条形图：当前样品的矿物成分比例 + 光学图像
            from matplotlib.font_manager import FontProperties
            from PIL import Image
            import matplotlib.gridspec as gridspec

            # 获取当前样品的最佳多物相组合
            combo_matches = self.rruff_combination_results.get(basename, [])
            if not combo_matches:
                ax.text(
                    0.5,
                    0.5,
                    "当前样品没有多物相匹配结果",
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontfamily="Times New Roman",
                )
                fig.tight_layout()
                self.rruff_fig_canvas.draw()
                return

            best_combo = combo_matches[0]
            phases = best_combo.get("phases", [])
            ratios = best_combo.get("ratios", [])
            unmatched_peaks = best_combo.get("unmatched_peaks", [])
            if not phases or not ratios:
                ax.text(
                    0.5,
                    0.5,
                    "当前样品未检测到有效矿物成分",
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontfamily="Times New Roman",
                )
                fig.tight_layout()
                self.rruff_fig_canvas.draw()
                return

            # 创建上下布局：上方条形图，下方光学图像
            fig.clear()
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)
            ax_bar = fig.add_subplot(gs[0, 0])
            ax_img = fig.add_subplot(gs[1, 0])

            # 排序：按比例从大到小
            indices = sorted(range(len(ratios)), key=lambda i: ratios[i], reverse=True)
            phases_sorted = [phases[i] for i in indices]
            ratios_sorted = [ratios[i] for i in indices]

            # 只显示前 N 个主要矿物，避免标签太挤
            max_minerals = 10
            phases_main = phases_sorted[:max_minerals]
            ratios_main = ratios_sorted[:max_minerals]

            # 颜色映射（与3D图保持一致风格）
            import matplotlib.cm as cm

            colors = cm.Set3(np.linspace(0, 1, len(phases_main)))

            x = np.arange(len(phases_main))
            bars = ax_bar.bar(
                x,
                ratios_main,
                color=colors,
                edgecolor="black",
                linewidth=0.8,
            )

            # x轴标签：矿物名简写，完整名放在tooltips中（这里只在图例里显示）
            mineral_labels_short = [
                p[:18] + "..." if len(p) > 18 else p for p in phases_main
            ]
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(
                mineral_labels_short,
                rotation=45,
                ha="right",
                fontsize=8,
            )
            ax_bar.set_ylabel("Mineral Ratio", fontsize=10)
            ax_bar.set_xlabel("Mineral Phase", fontsize=10)
            ax_bar.set_ylim(0, max(1.0, max(ratios_main) * 1.1))

            # 标注每个条形顶部的数值
            for xx, rr in zip(x, ratios_main):
                ax_bar.text(
                    xx,
                    rr + 0.02,
                    f"{rr:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

            ax_bar.grid(axis="y", alpha=0.3)
            ax_bar.set_title(
                f"{basename} - Mineral Composition (RRUFF Combination Match)",
                fontsize=11,
            )

            # 构建图例（矿物名与颜色对应）
            from matplotlib.patches import Rectangle

            legend_elements = []
            for p, c in zip(phases_main, colors):
                label = p[:25] + "..." if len(p) > 25 else p
                legend_elements.append(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=c,
                        edgecolor="black",
                        linewidth=0.5,
                        label=label,
                    )
                )

            legend_font = FontProperties(family="Times New Roman", size=8)
            ax_bar.legend(
                handles=legend_elements,
                loc="upper right",
                frameon=True,
                fancybox=True,
                shadow=False,
                prop=legend_font,
                ncol=1,
            )

            # 下方光学图像
            ax_img.axis("off")
            if basename in self.png_files:
                png_path = self.png_files[basename]
                try:
                    img = Image.open(png_path)
                    ax_img.imshow(img)
                    ax_img.set_title(
                        f"Optical Image: {basename}",
                        fontsize=10,
                        fontfamily="Times New Roman",
                    )
                except Exception as e:
                    ax_img.text(
                        0.5,
                        0.5,
                        f"光学图像加载失败：{e}",
                        fontsize=9,
                        ha="center",
                        va="center",
                        fontfamily="Times New Roman",
                    )
            else:
                ax_img.text(
                    0.5,
                    0.5,
                    "未找到对应的光学图像",
                    fontsize=10,
                    ha="center",
                    va="center",
                    fontfamily="Times New Roman",
                )

            # 在右上角或图像下方提示未匹配峰值信息
            try:
                if isinstance(unmatched_peaks, (list, np.ndarray)) and len(unmatched_peaks) > 0:
                    # 只显示前若干个未匹配峰值，避免文字太长
                    unmatched_peaks = np.array(unmatched_peaks, dtype=float)
                    unmatched_sorted = np.sort(unmatched_peaks)
                    max_show = 10
                    show_peaks = unmatched_sorted[:max_show]
                    more_flag = "" if len(unmatched_sorted) <= max_show else " ..."
                    peaks_str = ", ".join([f"{p:.1f}" for p in show_peaks]) + more_flag
                    text = f"Unmatched peaks (cm⁻¹): {peaks_str}"
                    ax_img.text(
                        0.01,
                        -0.15,
                        text,
                        transform=ax_img.transAxes,
                        fontsize=7,
                        ha="left",
                        va="top",
                        fontfamily="Times New Roman",
                    )
                else:
                    ax_img.text(
                        0.01,
                        -0.15,
                        "All detected peaks matched within tolerance.",
                        transform=ax_img.transAxes,
                        fontsize=7,
                        ha="left",
                        va="top",
                        fontfamily="Times New Roman",
                    )
            except Exception:
                # 文本绘制失败时忽略，不影响主图
                pass

            try:
                fig.tight_layout()
            except Exception:
                pass

            self.rruff_fig_canvas.draw()

        elif style_text.startswith("2D总览"):
            # 2D总览：多个样品的矿物成分+光学图概览（Top N 样品）
            # 布局：每个样品一行，左：矿物条形图，右：光学图像（左右同宽）
            from matplotlib.font_manager import FontProperties
            from PIL import Image

            # 所有有组合匹配结果的样品（排序）
            all_combo_names = sorted(
                k for k, v in self.rruff_combination_results.items() if v
            )
            if not all_combo_names:
                ax.text(
                    0.5,
                    0.5,
                    "没有多物相匹配结果，无法生成总览",
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontfamily="Times New Roman",
                )
                fig.tight_layout()
                self.rruff_fig_canvas.draw()
                return

            # 使用分页控制：根据当前页码和每页样品数确定要显示的样品
            page_size = getattr(self, "rruff_overview_page_size", 6)
            current_page = getattr(self, "rruff_overview_page", 0)
            start_idx = current_page * page_size
            end_idx = min(len(all_combo_names), start_idx + page_size)
            basenames = all_combo_names[start_idx:end_idx]

            if not basenames:
                ax.text(
                    0.5,
                    0.5,
                    "当前页没有可显示的样品",
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontfamily="Times New Roman",
                )
                fig.tight_layout()
                self.rruff_fig_canvas.draw()
                return

            # 布局：每个样品一行，2 列（左柱状图，右光学图），左右同宽
            n_rows = len(basenames)
            n_cols = 2

            fig.clear()
            # 加大整体尺寸，让每个光学图/柱状图都足够大（每行大约 4 英寸高度）
            fig.set_size_inches(12, max(6, 4 * n_rows))

            import matplotlib.gridspec as gridspec
            import matplotlib.cm as cm

            gs = gridspec.GridSpec(
                n_rows,
                n_cols,
                width_ratios=[1, 1],  # 左右一样宽
                hspace=0.6,
                wspace=0.25,
            )

            for i, name in enumerate(basenames):
                combo_matches = self.rruff_combination_results.get(name, [])
                best_combo = combo_matches[0]
                phases = best_combo.get("phases", [])
                ratios = best_combo.get("ratios", [])

                # 左：该样品的矿物条形图
                ax_bar = fig.add_subplot(gs[i, 0])

                if phases and ratios:
                    idx_sorted = sorted(
                        range(len(ratios)), key=lambda j: ratios[j], reverse=True
                    )
                    max_minerals = 8
                    phases_main = [phases[j] for j in idx_sorted[:max_minerals]]
                    ratios_main = [ratios[j] for j in idx_sorted[:max_minerals]]

                    colors = cm.Set3(np.linspace(0, 1, len(phases_main)))
                    x = np.arange(len(phases_main))
                    ax_bar.bar(
                        x,
                        ratios_main,
                        color=colors,
                        edgecolor="black",
                        linewidth=0.7,
                    )
                    labels_short = [
                        p[:15] + "..." if len(p) > 15 else p for p in phases_main
                    ]
                    ax_bar.set_xticks(x)
                    ax_bar.set_xticklabels(
                        labels_short, rotation=45, ha="right", fontsize=7
                    )
                    ax_bar.set_ylabel("Ratio", fontsize=8)
                    ax_bar.set_ylim(0, max(1.0, max(ratios_main) * 1.1))
                    ax_bar.set_title(
                        f"{name} - Minerals", fontsize=9, fontfamily="Times New Roman"
                    )
                    ax_bar.grid(axis="y", alpha=0.3)
                else:
                    ax_bar.text(
                        0.5,
                        0.5,
                        "无有效矿物成分",
                        fontsize=9,
                        ha="center",
                        va="center",
                        fontfamily="Times New Roman",
                    )
                    ax_bar.set_axis_off()

                # 右：光学图像（轴范围与左侧同高，同宽显示）
                ax_img = fig.add_subplot(gs[i, 1])
                ax_img.axis("off")
                if name in self.png_files:
                    png_path = self.png_files[name]
                    try:
                        img = Image.open(png_path)
                        ax_img.imshow(img)
                        ax_img.set_title(
                            f"Optical: {name}",
                            fontsize=9,
                            fontfamily="Times New Roman",
                        )
                    except Exception as e:
                        ax_img.text(
                            0.5,
                            0.5,
                            f"图像加载失败：{e}",
                            fontsize=8,
                            ha="center",
                            va="center",
                            fontfamily="Times New Roman",
                        )
                else:
                    ax_img.text(
                        0.5,
                        0.5,
                        "未找到光学图像",
                        fontsize=8,
                        ha="center",
                        va="center",
                        fontfamily="Times New Roman",
                    )

            try:
                fig.tight_layout()
            except Exception:
                pass
            self.rruff_fig_canvas.draw()

        elif style_text.startswith("3D柱状图"):
            # 3D柱状图：展示矿物成分比例对比 + 光学镜下图 - 符合学术期刊要求
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.font_manager import FontProperties
            
            # 清除2D axes，创建3D axes
            fig.clear()
            # 设置更大的图形尺寸以适应3D图和图像
            fig.set_size_inches(16, 12)
            ax = fig.add_subplot(111, projection='3d')
            
            # 收集所有文件的多物相组合结果
            basenames = sorted(set(self.rruff_combination_results.keys()))
            if not basenames:
                ax.text(0.5, 0.5, 0.5, "No combination match results available", 
                       fontsize=12, ha='center', fontfamily='Times New Roman')
                return
            
            # 收集所有出现的矿物名称（从最佳匹配组合中提取）
            all_minerals = set()
            sample_mineral_data = {}  # {basename: {'phases': [...], 'ratios': [...]}}
            
            for basename in basenames:
                combo_matches = self.rruff_combination_results.get(basename, [])
                if combo_matches:
                    # 使用最佳匹配组合（Top 1）
                    best_combo = combo_matches[0]
                    phases = best_combo.get('phases', [])
                    ratios = best_combo.get('ratios', [])
                    if phases and ratios:
                        all_minerals.update(phases)
                        sample_mineral_data[basename] = {
                            'phases': phases,
                            'ratios': ratios,
                            'match_score': best_combo.get('match_score', 0.0)
                        }
            
            if not all_minerals:
                ax.text(0.5, 0.5, 0.5, "No mineral composition data available", 
                       fontsize=12, ha='center', fontfamily='Times New Roman')
                return
            
            # 排序矿物名称（按字母顺序，便于查找）
            mineral_list = sorted(list(all_minerals))
            num_minerals = len(mineral_list)
            num_samples = len(basenames)
            
            # 准备3D柱状图数据：X=样品索引, Y=矿物索引, Z=比例
            x_list = []
            y_list = []
            z_list = []
            colors_list = []
            
            # 为每个矿物分配一个颜色（使用colormap）
            import matplotlib.cm as cm
            mineral_colors = cm.Set3(np.linspace(0, 1, num_minerals))
            mineral_color_map = {mineral: mineral_colors[i] for i, mineral in enumerate(mineral_list)}
            
            for file_idx, basename in enumerate(basenames):
                if basename in sample_mineral_data:
                    data = sample_mineral_data[basename]
                    phases = data['phases']
                    ratios = data['ratios']
                    
                    for phase, ratio in zip(phases, ratios):
                        if phase in mineral_list:
                            mineral_idx = mineral_list.index(phase)
                            x_list.append(file_idx)
                            y_list.append(mineral_idx)
                            z_list.append(float(ratio))
                            colors_list.append(mineral_color_map[phase])
            
            # 绘制3D柱状图（每个柱子代表一个样品中某个矿物的比例）
            dx = 0.6  # 柱状图宽度（样品方向）
            dy = 0.6  # 柱状图深度（矿物方向）
            
            if x_list:
                # 使用每个矿物的颜色绘制柱状图
                ax.bar3d(x_list, y_list, [0]*len(x_list), 
                        dx=dx, dy=dy, dz=z_list,
                        color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.3,
                        shade=True)
            
            # 设置标签和标题（学术期刊风格）
            ax.set_xlabel("Sample Index", fontsize=12, fontfamily='Times New Roman', labelpad=10)
            ax.set_ylabel("Mineral Phase", fontsize=12, fontfamily='Times New Roman', labelpad=10)
            ax.set_zlabel("Mineral Ratio", fontsize=12, fontfamily='Times New Roman', labelpad=10)
            ax.set_title("3D Mineral Composition Comparison with Optical Microscopy Images", 
                        fontsize=14, fontfamily='Times New Roman', pad=20, fontweight='bold')
            
            # 设置x轴刻度（样品索引和文件名）
            ax.set_xticks(range(num_samples))
            if num_samples <= 15:
                ax.set_xticklabels([name[:15] + "..." if len(name) > 15 else name for name in basenames], 
                                  rotation=45, ha='right', fontsize=8, fontfamily='Times New Roman')
            else:
                # 如果样品太多，只显示部分标签
                step = max(1, num_samples // 10)
                ax.set_xticks(range(0, num_samples, step))
                ax.set_xticklabels([basenames[i][:15] + "..." if len(basenames[i]) > 15 else basenames[i] 
                                    for i in range(0, num_samples, step)], 
                                  rotation=45, ha='right', fontsize=7, fontfamily='Times New Roman')
            
            # 设置y轴刻度（矿物名称）
            ax.set_yticks(range(num_minerals))
            # 简化矿物名称显示（只显示前20个字符）
            mineral_labels = [m[:20] + "..." if len(m) > 20 else m for m in mineral_list]
            ax.set_yticklabels(mineral_labels, fontsize=7, fontfamily='Times New Roman')
            ax.set_ylim(-0.5, num_minerals - 0.5)
            
            # 设置z轴范围（比例：0-1）
            ax.set_zlim(0, 1.0)
            ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_zticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], 
                              fontsize=9, fontfamily='Times New Roman')
            
            # 添加图例（显示所有矿物及其颜色）
            from matplotlib.patches import Rectangle
            legend_elements = []
            for mineral, color in mineral_color_map.items():
                # 简化矿物名称用于图例
                legend_label = mineral[:25] + "..." if len(mineral) > 25 else mineral
                legend_elements.append(
                    Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8, 
                             edgecolor='black', linewidth=0.5, label=legend_label)
                )
            
            font_prop = FontProperties(family='Times New Roman', size=8)
            # 图例放在右侧，分两列显示
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.0),
                      frameon=True, fancybox=True, shadow=True, prop=font_prop, ncol=1)
            
            # 在每个样品位置显示光学镜下图和矿物成分标签
            img_z_position = -0.15  # 图像在z轴的位置（在柱状图下方）
            img_max_pixels = 120  # 降低图像分辨率以提升性能
            img_size_x = 0.4  # 图像在x方向的大小
            img_size_y = num_minerals * 0.15  # 图像在y方向的大小（根据矿物数量调整）
            
            for file_idx, basename in enumerate(basenames):
                # 显示光学镜下图
                if basename in self.png_files:
                    png_path = self.png_files[basename]
                    try:
                        # 加载图像
                        img = Image.open(png_path)
                        # 调整图像大小（保持宽高比，降低分辨率以提升性能）
                        img.thumbnail((img_max_pixels, img_max_pixels), Image.Resampling.LANCZOS)
                        img_array = np.array(img)
                        
                        # 转换图像格式
                        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                            background = np.ones((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8) * 255
                            alpha = img_array[:, :, 3:4] / 255.0
                            img_array = (background * (1 - alpha) + img_array[:, :, :3] * alpha).astype(np.uint8)
                        elif len(img_array.shape) == 2:
                            img_array = np.stack([img_array] * 3, axis=-1)
                        elif len(img_array.shape) == 3 and img_array.shape[2] > 3:
                            img_array = img_array[:, :, :3]
                        
                        # 计算图像位置（在样品位置，y方向居中）
                        img_x_start = file_idx - img_size_x / 2
                        img_x_end = file_idx + img_size_x / 2
                        img_y_start = num_minerals / 2 - img_size_y / 2
                        img_y_end = num_minerals / 2 + img_size_y / 2
                        
                        # 创建图像网格（降低网格密度以提升性能）
                        img_height, img_width = img_array.shape[:2]
                        # 降低网格分辨率：每2个像素采样一次
                        stride = max(1, min(2, img_width // 30))  # 确保网格不超过30x30
                        x_img = np.linspace(img_x_start, img_x_end, max(10, img_width // stride))
                        y_img = np.linspace(img_y_start, img_y_end, max(10, img_height // stride))
                        X_img, Y_img = np.meshgrid(x_img, y_img)
                        Z_img = np.full_like(X_img, img_z_position)
                        
                        # 下采样图像数组以匹配网格
                        img_sampled = img_array[::stride, ::stride, :]
                        if img_sampled.shape[:2] != (len(y_img), len(x_img)):
                            # 如果尺寸不匹配，调整
                            from scipy.ndimage import zoom
                            zoom_factors = (len(y_img) / img_sampled.shape[0], 
                                          len(x_img) / img_sampled.shape[1], 1)
                            img_sampled = zoom(img_sampled, zoom_factors, order=1)
                        
                        # 归一化到0-1范围
                        img_normalized = img_sampled.astype(float) / 255.0
                        
                        # 使用plot_surface显示图像（降低rstride和cstride以提升性能）
                        ax.plot_surface(X_img, Y_img, Z_img, 
                                       rstride=1, cstride=1,
                                       facecolors=img_normalized,
                                       shade=False,
                                       alpha=0.9,
                                       edgecolor='none',
                                       linewidth=0)
                        
                        # 添加图像边框
                        border_x = [img_x_start, img_x_end, img_x_end, img_x_start, img_x_start]
                        border_y = [img_y_start, img_y_start, img_y_end, img_y_end, img_y_start]
                        border_z = [img_z_position] * 5
                        ax.plot(border_x, border_y, border_z, 'k-', linewidth=1.0, alpha=0.9)
                        
                    except Exception as e:
                        print(f"加载图像 {png_path} 失败: {e}")
                        # 如果图像加载失败，显示占位符
                        ax.text(file_idx, num_minerals / 2, img_z_position, 
                               "No Image", 
                               fontsize=6, ha='center', va='center',
                               fontfamily='Times New Roman',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', 
                                        alpha=0.7, edgecolor='black'))
                
                # 显示矿物成分标签（在图像上方）
                if basename in sample_mineral_data:
                    data = sample_mineral_data[basename]
                    phases = data['phases']
                    ratios = data['ratios']
                    match_score = data['match_score']
                    
                    # 构建成分文本（最多显示前5个主要矿物）
                    composition_texts = []
                    sorted_indices = sorted(range(len(ratios)), key=lambda i: ratios[i], reverse=True)
                    for idx in sorted_indices[:5]:  # 只显示前5个
                        phase_name = phases[idx][:15] + "..." if len(phases[idx]) > 15 else phases[idx]
                        ratio_val = ratios[idx]
                        if ratio_val > 0.01:  # 只显示比例>1%的矿物
                            composition_texts.append(f"{phase_name}: {ratio_val:.2f}")
                    
                    composition_str = "\n".join(composition_texts)
                    if composition_str:
                        # 在图像上方显示成分标签
                        label_y = num_minerals / 2 + img_size_y / 2 + 0.5
                        ax.text(file_idx, label_y, img_z_position + 0.05,
                               composition_str,
                               fontsize=6, ha='center', va='bottom',
                               fontfamily='Times New Roman',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        alpha=0.8, edgecolor='black', linewidth=0.5))
            
            # 调整z轴范围以包含图像
            ax.set_zlim(img_z_position - 0.1, 1.0)
            
            # 设置网格和样式
            ax.grid(True, alpha=0.3)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.xaxis.pane.set_alpha(0.1)
            ax.yaxis.pane.set_alpha(0.1)
            ax.zaxis.pane.set_alpha(0.1)
            
            # 设置视角（最佳视角，确保能看到图像和柱状图）
            ax.view_init(elev=20, azim=45)

        else:
            # 单文件的组合 score 热图（行=1，列=组合序号）
            combos = self.rruff_combination_results.get(basename, [])
            if combos:
                scores = [c.get("match_score", 0.0) for c in combos]
                mat = np.array([scores])
                im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
                ax.set_xticks(range(len(scores)))
                ax.set_xticklabels([f"{i+1}" for i in range(len(scores))], fontsize=8)
                ax.set_yticks([])
                ax.set_xlabel("Combination Rank", fontsize=10)
                ax.set_title(f"{basename} - Combination Match Score Heatmap", fontsize=11)
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Match Score", fontsize=9)

        # 安全地调用 tight_layout，避免 Singular matrix 错误
        try:
            # 检查图形尺寸是否有效
            fig_width, fig_height = fig.get_size_inches()
            if fig_width > 0 and fig_height > 0:
                fig.tight_layout()
            else:
                # 如果尺寸无效，设置默认尺寸
                fig.set_size_inches(5, 4)
                fig.tight_layout()
        except Exception as e:
            # 如果 tight_layout 失败，尝试设置默认尺寸后重试
            try:
                fig.set_size_inches(5, 4)
                fig.tight_layout()
            except:
                # 如果仍然失败，跳过 tight_layout，只绘制图形
                print(f"警告: tight_layout 失败，跳过布局调整: {e}")
        
        self.rruff_fig_canvas.draw()

    def export_rruff_summary_table(self):
        """导出RRUFF匹配总览表为CSV。"""
        from PyQt6.QtWidgets import QFileDialog
        import csv

        if not hasattr(self, "rruff_summary_table"):
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存RRUFF匹配表格", "rruff_summary.csv", "CSV Files (*.csv)")
        if not path:
            return

        table = self.rruff_summary_table
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 写标题
            headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
            writer.writerow(headers)
            # 写内容
            for row in range(table.rowCount()):
                row_data = []
                for col in range(table.columnCount()):
                    item = table.item(row, col)
                    row_data.append(item.text() if item else "")
                writer.writerow(row_data)

    def export_rruff_summary_figure(self):
        """
        导出当前总览图像为文件（与内置预览一致）。
        """
        from PyQt6.QtWidgets import QFileDialog
        import matplotlib.pyplot as plt

        if not hasattr(self, "rruff_fig_canvas"):
            return

        fig = self.rruff_fig_canvas.figure
        path, _ = QFileDialog.getSaveFileName(
            self, "保存RRUFF匹配示意图", "rruff_summary.pdf", "PDF Files (*.pdf);;PNG Files (*.png)"
        )
        if not path:
            return
        fig.savefig(path, dpi=300)
    
    def plot_single_spectrum(self, txt_basename):
        """绘制单个光谱图（使用Qt画板，复用主窗口绘图逻辑）"""
        # 清除之前的绘图和工具栏（确保完全清除，避免弹出独立窗口）
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                widget.setParent(None)
                widget.hide()  # 先隐藏
                widget.deleteLater()  # 延迟删除，确保完全清理
            elif item.layout():
                # 如果有嵌套布局，也清除
                while item.layout().count():
                    nested_item = item.layout().takeAt(0)
                    if nested_item.widget():
                        nested_widget = nested_item.widget()
                        nested_widget.setParent(None)
                        nested_widget.hide()  # 先隐藏
                        nested_widget.deleteLater()  # 延迟删除
        
        # 获取主窗口的绘图参数
        plot_params = self.get_parent_plot_params()
        if not plot_params:
            QMessageBox.warning(self, "Warning", "Cannot get plot parameters from main window")
            return
        
        # 读取数据
        txt_file = None
        for f in self.txt_files:
            if os.path.splitext(os.path.basename(f))[0] == txt_basename:
                txt_file = f
                break
        
        if not txt_file:
            return
        
        try:
            # 读取光谱数据
            x, y = self.data_controller.read_data(
                txt_file,
                plot_params['skip_rows'],
                plot_params['x_min_phys'],
                plot_params['x_max_phys']
            )
            
            # 创建Qt画板（使用GridSpec布局：左侧光谱，右侧镜下光学图）
            # 调整比例：光谱图和镜下光学图大小一致（1:1）
            fig_width = plot_params['fig_width'] * 2.0  # 增加总宽度以容纳两个等大的图
            fig_height = plot_params['fig_height']
            
            canvas = MplCanvas(self, width=fig_width, height=fig_height, dpi=100)
            fig = canvas.figure
            
            # 清除默认的axes（避免anonymous Axes）
            fig.clear()
            
            # 使用GridSpec：左侧光谱图，右侧镜下光学图，下方饼图
            # 如果有组合匹配，添加饼图区域
            has_combination = (self.rruff_loader and txt_basename in self.selected_rruff_combinations and 
                             len(self.selected_rruff_combinations[txt_basename]) > 0)
            
            if has_combination:
                # 3行布局：光谱图、光学图、饼图
                gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[1, 1], 
                            hspace=0.15, wspace=0.1)
                ax_spectrum = fig.add_subplot(gs[0, 0])
                ax_image = fig.add_subplot(gs[0, 1])
                ax_pie = fig.add_subplot(gs[1, :])  # 饼图跨越两列
            else:
                # 2列布局：光谱图、光学图
                gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], hspace=0.1, wspace=0.1)
                ax_spectrum = fig.add_subplot(gs[0])
                ax_image = fig.add_subplot(gs[1])
                ax_pie = None
            
            # 准备绘图数据（复用主窗口的预处理逻辑）
            grouped_files_data = [(txt_file, x, y)]
            control_data_list = []
            individual_y_params = {}
            legend_names = {txt_basename: txt_basename}
            
            # 更新plot_params以包含必要的数据
            plot_params['grouped_files_data'] = grouped_files_data
            plot_params['control_data_list'] = control_data_list
            plot_params['individual_y_params'] = individual_y_params
            plot_params['legend_names'] = legend_names
            plot_params['plot_mode'] = 'Normal Overlay'
            
            # 添加RRUFF光谱数据（如果已选中，包括单物相和组合匹配）
            plot_params['rruff_spectra'] = []
            plot_params['rruff_match_results'] = []
            
            # 添加单物相匹配的光谱
            if self.rruff_loader and txt_basename in self.selected_rruff_spectra:
                for rruff_name in self.selected_rruff_spectra[txt_basename]:
                    rruff_data = self.rruff_loader.get_spectrum(rruff_name)
                    if rruff_data:
                        # 找到对应的匹配结果
                        match_result = None
                        if txt_basename in self.rruff_match_results:
                            for match in self.rruff_match_results[txt_basename]:
                                if match['name'] == rruff_name:
                                    match_result = match
                                    break
                        plot_params['rruff_spectra'].append({
                            'name': rruff_name,
                            'x': rruff_data['x'],
                            'y': rruff_data['y'],
                            'matches': match_result['matches'] if match_result else []
                        })
                        if match_result:
                            plot_params['rruff_match_results'].append(match_result)
            
            # 检测峰值（用于组合匹配的参考线）
            from scipy.signal import find_peaks
            # 获取txt_file路径（用于缓存）
            txt_file = None
            if txt_basename:
                for f in self.txt_files:
                    if os.path.splitext(os.path.basename(f))[0] == txt_basename:
                        txt_file = f
                        break
            y_proc_for_peaks = self._preprocess_spectrum(x, y, plot_params, file_path=txt_file)
            peak_height = plot_params.get('peak_height_threshold', 0.0)
            peak_distance = plot_params.get('peak_distance_min', 10)
            peak_prominence = plot_params.get('peak_prominence', None)
            
            y_max = np.max(y_proc_for_peaks) if len(y_proc_for_peaks) > 0 else 0
            y_min = np.min(y_proc_for_peaks) if len(y_proc_for_peaks) > 0 else 0
            y_range = y_max - y_min
            
            peak_kwargs = {}
            if peak_height == 0:
                if y_max > 0:
                    peak_height = y_max * 0.001
                else:
                    peak_height = 0
            if peak_height > y_range * 2 and y_range > 0:
                peak_height = y_max * 0.001
            if peak_height != 0:
                peak_kwargs['height'] = peak_height
            
            if peak_distance == 0:
                peak_distance = max(1, int(len(y_proc_for_peaks) * 0.001))
            if peak_distance > len(y_proc_for_peaks) * 0.5:
                peak_distance = max(1, int(len(y_proc_for_peaks) * 0.001))
            peak_distance = max(1, peak_distance)
            
            if peak_height < 0 or (y_max > 0 and peak_height < y_max * 0.001):
                pass  # 不使用distance
            else:
                peak_kwargs['distance'] = peak_distance
            
            if peak_prominence is not None and peak_prominence != 0:
                if peak_prominence > y_range * 2 and y_range > 0:
                    peak_prominence = y_range * 0.001
                peak_kwargs['prominence'] = peak_prominence
            
            try:
                peaks_for_ref, properties = find_peaks(y_proc_for_peaks, **peak_kwargs)
            except:
                peaks_for_ref, properties = find_peaks(y_proc_for_peaks, 
                                                    height=y_max * 0.001 if y_max > 0 else 0,
                                                    distance=max(1, int(len(y_proc_for_peaks) * 0.001)))
            
            peak_wavenumbers_for_ref = x[peaks_for_ref] if len(peaks_for_ref) > 0 else np.array([])
            
            # 添加组合匹配的光谱（根据GUI控件决定显示模式）
            # combination_info: [{ 'phases': [...], 'ratios': [...], 'match_score': float, 'colors': [...]}]
            combination_info = []  # 存储组合信息用于饼图（颜色稍后根据实际谱线颜色填充）
            if self.rruff_loader and txt_basename in self.selected_rruff_combinations:
                from scipy.interpolate import interp1d
                global_stack_offset = plot_params.get('global_stack_offset', 0.0)
                
                # 检查是否显示为整体光谱
                show_as_single = self.rruff_combination_as_single_check.isChecked() if hasattr(self, 'rruff_combination_as_single_check') else False
                
                for combo_idx, combo in enumerate(self.selected_rruff_combinations[txt_basename]):
                    phases = combo['phases']
                    ratios = combo['ratios']
                    matches = combo.get('matches', [])
                    
                    # 存储组合信息用于饼图（colors 稍后填充）
                    combination_info.append({
                        'phases': phases,
                        'ratios': ratios,
                        'match_score': combo.get('match_score', 0.0),
                        'colors': None
                    })
                    
                    if show_as_single:
                        # 显示为整体组合光谱
                        try:
                            combined_y = None
                            combined_x = None
                            
                            for i, phase_name in enumerate(phases):
                                rruff_data = self.rruff_loader.get_spectrum(phase_name)
                                if rruff_data:
                                    if combined_x is None:
                                        combined_x = rruff_data['x']
                                        combined_y = np.zeros_like(rruff_data['y'])
                                    
                                    # 插值对齐
                                    f_interp = interp1d(rruff_data['x'], rruff_data['y'], 
                                                      kind='linear', fill_value=0, bounds_error=False)
                                    aligned_y = f_interp(combined_x)
                                    combined_y += aligned_y * ratios[i]
                            
                            if combined_y is not None:
                                phases_str = " + ".join(phases)
                                plot_params['rruff_spectra'].append({
                                    'name': f"组合: {phases_str}",
                                    'x': combined_x,
                                    'y': combined_y,
                                    'matches': matches,
                                    'is_combination': True,
                                    'phases': phases,
                                    'ratios': ratios
                                })
                        except Exception as e:
                            print(f"Warning: Failed to add combination spectrum: {e}")
                            continue
                    else:
                        # 将各个物相分别添加为独立谱线
                        # 计算已添加的单物相匹配的RRUFF光谱数量
                        num_single_phases = len(self.selected_rruff_spectra.get(txt_basename, set()))
                        
                        for i, phase_name in enumerate(phases):
                            try:
                                rruff_data = self.rruff_loader.get_spectrum(phase_name)
                                if rruff_data:
                                    # 插值对齐到查询光谱的波数轴
                                    f_interp = interp1d(rruff_data['x'], rruff_data['y'], 
                                                      kind='linear', fill_value=0, bounds_error=False)
                                    aligned_y = f_interp(x)
                                    
                                    # 应用比例
                                    scaled_y = aligned_y * ratios[i]
                                    
                                    # 计算堆叠偏移（每个物相单独一条线）
                                    # 考虑已添加的单物相匹配光谱数量，确保第一个物相也有偏移
                                    stack_offset = (num_single_phases + combo_idx * len(phases) + i + 1) * global_stack_offset
                                    
                                    # 为单个物相计算匹配的峰值（使用该物相的峰值与查询光谱的峰值匹配）
                                    phase_matches = []
                                    try:
                                        rruff_peaks = rruff_data.get('peaks', (np.array([]), np.array([])))[1]
                                        # peak_wavenumbers 在 match_rruff_combination 中已定义，需要在这里使用
                                        # 但由于这是在 plot_single_spectrum 中，peak_wavenumbers 可能不在作用域内
                                        # 使用之前检测的峰值
                                        if len(rruff_peaks) > 0 and len(peak_wavenumbers_for_ref) > 0:
                                            # 使用当前的匹配容差
                                            tolerance = self.rruff_match_tolerance_spin.value() if hasattr(self, 'rruff_match_tolerance_spin') else 5.0
                                            phase_matches, _ = self.peak_matcher.match_peaks(peak_wavenumbers_for_ref, rruff_peaks, tolerance=tolerance)
                                    except Exception as e:
                                        print(f"Warning: Failed to match peaks for phase {phase_name}: {e}")
                                    
                                    plot_params['rruff_spectra'].append({
                                        'name': f"{phase_name} ({ratios[i]:.2%})",
                                        'x': x,
                                        'y': scaled_y,
                                        'matches': phase_matches,  # 使用该物相的峰值匹配结果
                                        'is_combination_phase': True,
                                        'combination_idx': combo_idx,
                                        'phase_idx': i,
                                        'stack_offset': stack_offset,
                                        'original_phase_name': phase_name,
                                        'ratio': ratios[i]
                                    })
                            except Exception as e:
                                print(f"Warning: Failed to add phase {phase_name} from combination: {e}")
                                continue
            
            # 存储组合信息到plot_params用于饼图
            plot_params['combination_info'] = combination_info
            
            # 确保包含RRUFF参考线设置
            if 'rruff_ref_lines_enabled' not in plot_params:
                plot_params['rruff_ref_lines_enabled'] = self.rruff_ref_lines_enabled_check.isChecked() if hasattr(self, 'rruff_ref_lines_enabled_check') else True
            if 'rruff_ref_line_offset' not in plot_params:
                plot_params['rruff_ref_line_offset'] = self.rruff_ref_line_offset_spin.value() if hasattr(self, 'rruff_ref_line_offset_spin') else 0.0
            
            # 设置当前组名用于绘图
            plot_params['current_group_name'] = txt_basename
            
            # 使用核心绘图函数（不再创建临时窗口）
            self._core_plot_spectrum(ax_spectrum, plot_params)
            
            # 应用样式（确保样式正确应用）
            self.apply_spectrum_style(ax_spectrum, plot_params, txt_basename)

            # 根据实际绘制的RRUFF谱线颜色，回填组合信息中的颜色，用于饼图
            if combination_info:
                # 从axes中获取所有RRUFF谱线的颜色映射：label(去掉前缀) -> color
                phase_color_map = {}
                for line in ax_spectrum.get_lines():
                    label = line.get_label()
                    if isinstance(label, str) and label.startswith("RRUFF: "):
                        phase_label = label.replace("RRUFF: ", "").strip()
                        phase_color_map[phase_label] = line.get_color()

                # 为每个组合计算颜色列表（与phases/ratios顺序一致）
                for combo in combination_info:
                    phases = combo.get('phases', [])
                    ratios = combo.get('ratios', [])
                    colors = []
                    for p, r in zip(phases, ratios):
                        # 组合物相在绘图中的label格式为: "{phase_name} ({ratio:.2%})"
                        label_with_ratio = f"{p} ({r:.2%})"
                        c = phase_color_map.get(label_with_ratio)
                        if c is None:
                            # 回退：有些情况下label可能只包含物相名
                            c = phase_color_map.get(p, None)
                        colors.append(c)
                    combo['colors'] = colors
            
            # 绘制镜下光学图
            if txt_basename in self.png_files:
                self.plot_microscopy_image(ax_image, self.png_files[txt_basename], plot_params)
            else:
                ax_image.text(0.5, 0.5, "No microscopy\nimage found",
                             ha='center', va='center', transform=ax_image.transAxes,
                             fontsize=12, color='gray', fontfamily='Times New Roman')
                ax_image.axis('off')
            
            # 绘制饼图（如果有组合匹配）
            if ax_pie is not None and combination_info:
                self.plot_combination_pie_chart(ax_pie, combination_info, plot_params)
            
            # 应用整体样式（移除外框）
            fig.patch.set_visible(False)  # 移除figure的背景框
            # 使用subplots_adjust减小边距，减少留白
            if has_combination:
                fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.15, wspace=0.1, hspace=0.15)
            else:
                fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, wspace=0.1)
            
            # 添加工具栏
            toolbar = NavigationToolbar(canvas, self)
            
            # 添加到布局
            self.plot_layout.addWidget(canvas)
            self.plot_layout.addWidget(toolbar)
            canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot spectrum: {e}")
            traceback.print_exc()
    
    def plot_combination_pie_chart(self, ax, combination_info, plot_params):
        """绘制组合匹配的饼图（学术标准）"""
        if not combination_info:
            ax.axis('off')
            return
        
        # 使用第一个组合的信息（如果有多个，可以合并或选择最佳）
        combo = combination_info[0]  # 使用第一个组合
        phases = combo['phases']
        ratios = combo['ratios']
        phase_colors = combo.get('colors') or []
        
        # 过滤掉比例很小的组分（小于1%），同时保留对应颜色索引
        filtered = [(idx, p, r) for idx, (p, r) in enumerate(zip(phases, ratios)) if r >= 0.01]
        if not filtered:
            filtered = [(idx, p, r) for idx, (p, r) in enumerate(zip(phases, ratios))]
        
        if not filtered:
            ax.axis('off')
            return

        indices, phases_filtered, ratios_filtered = zip(*filtered)
        
        # 根据索引提取对应的颜色（如果有），否则后面使用默认色图
        colors_from_lines = []
        for i in indices:
            if i < len(phase_colors) and phase_colors[i] is not None:
                colors_from_lines.append(phase_colors[i])
            else:
                colors_from_lines.append(None)
        # 如果有来自谱线的颜色，则优先使用；否则使用学术标准色图
        if any(c is not None for c in colors_from_lines):
            # 用现有颜色填空：没有颜色的项使用默认色图
            import matplotlib.cm as cm
            default_colors = cm.Set3(np.linspace(0, 1, len(phases_filtered)))
            colors = []
            default_idx = 0
            for c in colors_from_lines:
                if c is not None:
                    colors.append(c)
                else:
                    colors.append(default_colors[default_idx])
                    default_idx += 1
        else:
            import matplotlib.cm as cm
            colors = cm.Set3(np.linspace(0, 1, len(phases_filtered)))
        
        # 绘制饼图（学术标准：简洁、清晰）
        wedges, texts, autotexts = ax.pie(
            ratios_filtered,
            labels=[f"{p}\n({r:.1%})" for p, r in zip(phases_filtered, ratios_filtered)],
            autopct='',  # 不使用自动百分比，使用labels显示
            startangle=90,
            colors=colors,
            textprops={'fontsize': plot_params.get('legend_fontsize', 10), 
                      'fontfamily': plot_params.get('font_family', 'Times New Roman')},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        
        # 设置标题
        match_score = combo.get('match_score', 0.0)
        ax.set_title(f'Phase Composition (Match Score: {match_score:.2%})', 
                    fontsize=plot_params.get('axis_title_fontsize', 12),
                    fontfamily=plot_params.get('font_family', 'Times New Roman'),
                    pad=10)
        
        # 移除坐标轴
        ax.axis('equal')
        ax.set_aspect('equal')
    
    def plot_multiple_spectra(self, txt_basenames):
        """绘制多个光谱图（上下排列的子图）"""
        # 清除之前的绘图和工具栏（确保完全清除，避免弹出独立窗口）
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                widget.setParent(None)
                widget.hide()  # 先隐藏
                widget.deleteLater()  # 延迟删除，确保完全清理
            elif item.layout():
                # 如果有嵌套布局，也清除
                while item.layout().count():
                    nested_item = item.layout().takeAt(0)
                    if nested_item.widget():
                        nested_widget = nested_item.widget()
                        nested_widget.setParent(None)
                        nested_widget.hide()  # 先隐藏
                        nested_widget.deleteLater()  # 延迟删除
        
        # 获取主窗口的绘图参数
        plot_params = self.get_parent_plot_params()
        if not plot_params:
            QMessageBox.warning(self, "Warning", "Cannot get plot parameters from main window")
            return
        
        try:
            n_files = len(txt_basenames)
            if n_files == 0:
                return
            
            # 创建Qt画板
            fig_width = plot_params['fig_width'] * 2.0
            # 高度根据文件数量调整
            fig_height = plot_params['fig_height'] * max(3, n_files * 0.8)
            
            canvas = MplCanvas(self, width=fig_width, height=fig_height, dpi=100)
            fig = canvas.figure
            
            # 清除默认的axes（避免anonymous Axes）
            fig.clear()
            
            # 创建上下排列的子图：每个文件一行，每行包含光谱图（左）和镜下光学图（右）
            # 减小间距：wspace减小到0.1，hspace减小到0.2（避免重叠）
            gs = GridSpec(n_files, 2, figure=fig, width_ratios=[1, 1], 
                         hspace=0.2, wspace=0.1, height_ratios=[1]*n_files)
            
            axes_spectrum = []
            axes_image = []
            
            # 为每个文件创建子图
            for i, txt_basename in enumerate(txt_basenames):
                # 找到对应的光谱数据文件（txt/csv）
                txt_file = None
                for f in self.txt_files:
                    if os.path.splitext(os.path.basename(f))[0] == txt_basename:
                        txt_file = f
                        break
                
                if not txt_file:
                    continue
                
                # 读取数据
                x, y = self.data_controller.read_data(
                    txt_file,
                    plot_params['skip_rows'],
                    plot_params['x_min_phys'],
                    plot_params['x_max_phys']
                )
                
                # 创建子图
                ax_spectrum = fig.add_subplot(gs[i, 0])
                ax_image = fig.add_subplot(gs[i, 1])
                axes_spectrum.append((ax_spectrum, x, y, txt_basename, txt_file))
                axes_image.append((ax_image, txt_basename))
            
            # 绘制所有光谱图
            for ax_spectrum, x, y, txt_basename, txt_file in axes_spectrum:
                # 预处理当前文件的光谱数据（用于峰值检测和组合匹配，传入文件路径以支持缓存）
                y_proc_current = self._preprocess_spectrum(x, y, plot_params, file_path=txt_file)
                
                # 检测当前文件的峰值（用于组合匹配的参考线）
                from scipy.signal import find_peaks
                peak_height = plot_params.get('peak_height_threshold', 0.0)
                peak_distance = plot_params.get('peak_distance_min', 10)
                peak_prominence = plot_params.get('peak_prominence', None)
                
                y_max = np.max(y_proc_current) if len(y_proc_current) > 0 else 0
                y_min = np.min(y_proc_current) if len(y_proc_current) > 0 else 0
                y_range = y_max - y_min
                
                peak_kwargs = {}
                if peak_height == 0:
                    if y_max > 0:
                        peak_height = y_max * 0.001
                    else:
                        peak_height = 0
                if peak_height > y_range * 2 and y_range > 0:
                    peak_height = y_max * 0.001
                if peak_height != 0:
                    peak_kwargs['height'] = peak_height
                
                if peak_distance == 0:
                    peak_distance = max(1, int(len(y_proc_current) * 0.001))
                if peak_distance > len(y_proc_current) * 0.5:
                    peak_distance = max(1, int(len(y_proc_current) * 0.001))
                peak_distance = max(1, peak_distance)
                
                if peak_height < 0 or (y_max > 0 and peak_height < y_max * 0.001):
                    pass  # 不使用distance
                else:
                    peak_kwargs['distance'] = peak_distance
                
                if peak_prominence is not None and peak_prominence != 0:
                    if peak_prominence > y_range * 2 and y_range > 0:
                        peak_prominence = y_range * 0.001
                    peak_kwargs['prominence'] = peak_prominence
                
                try:
                    peaks_current, properties = find_peaks(y_proc_current, **peak_kwargs)
                except:
                    peaks_current, properties = find_peaks(y_proc_current, 
                                                        height=y_max * 0.001 if y_max > 0 else 0,
                                                        distance=max(1, int(len(y_proc_current) * 0.001)))
                
                current_file_peak_wavenumbers = x[peaks_current] if len(peaks_current) > 0 else np.array([])
                
                # 准备绘图数据
                grouped_files_data = [(txt_file, x, y)]
                control_data_list = []
                individual_y_params = {}
                legend_names = {txt_basename: txt_basename}
                
                plot_params['grouped_files_data'] = grouped_files_data
                plot_params['control_data_list'] = control_data_list
                plot_params['individual_y_params'] = individual_y_params
                plot_params['legend_names'] = legend_names
                plot_params['plot_mode'] = 'Normal Overlay'
                
                # 添加RRUFF光谱数据（如果已选中，包括单物相和组合匹配）
                plot_params['rruff_spectra'] = []
                plot_params['rruff_match_results'] = []
                
                # 添加单物相匹配的光谱
                if self.rruff_loader and txt_basename in self.selected_rruff_spectra:
                        for rruff_name in self.selected_rruff_spectra[txt_basename]:
                            rruff_data = self.rruff_loader.get_spectrum(rruff_name)
                            if rruff_data:
                                # 找到对应的匹配结果
                                match_result = None
                                if txt_basename in self.rruff_match_results:
                                    for match in self.rruff_match_results[txt_basename]:
                                        if match['name'] == rruff_name:
                                            match_result = match
                                            break
                                plot_params['rruff_spectra'].append({
                                    'name': rruff_name,
                                    'x': rruff_data['x'],
                                    'y': rruff_data['y'],
                                    'matches': match_result['matches'] if match_result else []
                                })
                                if match_result:
                                    plot_params['rruff_match_results'].append(match_result)
                
                # 添加组合匹配的光谱（根据GUI控件决定显示模式）
                if self.rruff_loader and txt_basename in self.selected_rruff_combinations:
                    from scipy.interpolate import interp1d
                    global_stack_offset = plot_params.get('global_stack_offset', 0.0)
                    
                    # 检查是否显示为整体光谱
                    show_as_single = self.rruff_combination_as_single_check.isChecked() if hasattr(self, 'rruff_combination_as_single_check') else False
                    
                    for combo_idx, combo in enumerate(self.selected_rruff_combinations[txt_basename]):
                            phases = combo['phases']
                            ratios = combo['ratios']
                            matches = combo.get('matches', [])
                            
                            if show_as_single:
                                # 显示为整体组合光谱
                                try:
                                    combined_y = None
                                    combined_x = None
                                    
                                    for i, phase_name in enumerate(phases):
                                        rruff_data = self.rruff_loader.get_spectrum(phase_name)
                                        if rruff_data:
                                            if combined_x is None:
                                                combined_x = rruff_data['x']
                                                combined_y = np.zeros_like(rruff_data['y'])
                                            
                                            # 插值对齐
                                            f_interp = interp1d(rruff_data['x'], rruff_data['y'], 
                                                              kind='linear', fill_value=0, bounds_error=False)
                                            aligned_y = f_interp(combined_x)
                                            combined_y += aligned_y * ratios[i]
                                    
                                    if combined_y is not None:
                                        phases_str = " + ".join(phases)
                                        plot_params['rruff_spectra'].append({
                                            'name': f"组合: {phases_str}",
                                            'x': combined_x,
                                            'y': combined_y,
                                            'matches': matches,
                                            'is_combination': True,
                                            'phases': phases,
                                            'ratios': ratios
                                        })
                                except Exception as e:
                                    print(f"Warning: Failed to add combination spectrum: {e}")
                                    continue
                            else:
                                # 将各个物相分别添加为独立谱线
                                # 计算已添加的单物相匹配的RRUFF光谱数量
                                num_single_phases = len(self.selected_rruff_spectra.get(txt_basename, set()))
                                
                                for i, phase_name in enumerate(phases):
                                    try:
                                        rruff_data = self.rruff_loader.get_spectrum(phase_name)
                                        if rruff_data:
                                            # 插值对齐到查询光谱的波数轴
                                            f_interp = interp1d(rruff_data['x'], rruff_data['y'], 
                                                              kind='linear', fill_value=0, bounds_error=False)
                                            aligned_y = f_interp(x)
                                            
                                            # 应用比例
                                            scaled_y = aligned_y * ratios[i]
                                            
                                            # 计算堆叠偏移（每个物相单独一条线）
                                            # 考虑已添加的单物相匹配光谱数量，确保第一个物相也有偏移
                                            stack_offset = (num_single_phases + combo_idx * len(phases) + i + 1) * global_stack_offset
                                            
                                            # 为单个物相计算匹配的峰值（使用该物相的峰值与当前文件的峰值匹配）
                                            phase_matches = []
                                            try:
                                                rruff_peaks = rruff_data.get('peaks', (np.array([]), np.array([])))[1]
                                                if len(rruff_peaks) > 0 and len(current_file_peak_wavenumbers) > 0:
                                                    # 使用当前的匹配容差
                                                    tolerance = self.rruff_match_tolerance_spin.value() if hasattr(self, 'rruff_match_tolerance_spin') else 5.0
                                                    phase_matches, _ = self.peak_matcher.match_peaks(current_file_peak_wavenumbers, rruff_peaks, tolerance=tolerance)
                                            except Exception as e:
                                                print(f"Warning: Failed to match peaks for phase {phase_name}: {e}")
                                            
                                            plot_params['rruff_spectra'].append({
                                                'name': f"{phase_name} ({ratios[i]:.2%})",
                                                'x': x,
                                                'y': scaled_y,
                                                'matches': phase_matches,  # 使用该物相的峰值匹配结果
                                                'is_combination_phase': True,
                                                'combination_idx': combo_idx,
                                                'phase_idx': i,
                                                'stack_offset': stack_offset,
                                                'original_phase_name': phase_name,
                                                'ratio': ratios[i]
                                            })
                                    except Exception as e:
                                        print(f"Warning: Failed to add phase {phase_name} from combination: {e}")
                                        continue
                
                # 确保包含RRUFF参考线设置
                if 'rruff_ref_lines_enabled' not in plot_params:
                    plot_params['rruff_ref_lines_enabled'] = self.rruff_ref_lines_enabled_check.isChecked() if hasattr(self, 'rruff_ref_lines_enabled_check') else True
                if 'rruff_ref_line_offset' not in plot_params:
                    plot_params['rruff_ref_line_offset'] = self.rruff_ref_line_offset_spin.value() if hasattr(self, 'rruff_ref_line_offset_spin') else 0.0
                
                # 设置当前组名用于绘图
                plot_params['current_group_name'] = txt_basename
                
                # 使用核心绘图函数（不再创建临时窗口）
                self._core_plot_spectrum(ax_spectrum, plot_params)
                
                # 设置子图标题（文件名）- 使用较小的pad避免与x轴标签重叠
                ax_spectrum.set_title(txt_basename, fontsize=10, fontfamily=plot_params['font_family'], pad=3)
                
                # 应用样式（多子图模式下，只给最下面的子图显示x轴标签）
                show_xlabel = (i == n_files - 1)  # 只有最后一个子图显示x轴标签
                self.apply_spectrum_style(ax_spectrum, plot_params, txt_basename, show_xlabel=show_xlabel)
            
            # 绘制所有镜下光学图
            for ax_image, txt_basename in axes_image:
                if txt_basename in self.png_files:
                    self.plot_microscopy_image(ax_image, self.png_files[txt_basename], plot_params)
                else:
                    ax_image.text(0.5, 0.5, "No microscopy\nimage found",
                                ha='center', va='center', transform=ax_image.transAxes,
                                fontsize=10, color='gray', fontfamily='Times New Roman')
                    ax_image.axis('off')
            
            # 应用整体样式（移除外框）
            fig.patch.set_visible(False)
            # 使用subplots_adjust减小边距，减少留白，避免重叠
            # 根据子图数量调整底部边距（确保x轴标签可见）
            bottom_margin = 0.08 + (n_files * 0.02)  # 根据子图数量动态调整
            fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=bottom_margin, 
                              hspace=0.2, wspace=0.1)
            
            # 添加工具栏
            toolbar = NavigationToolbar(canvas, self)
            
            # 添加到布局
            self.plot_layout.addWidget(canvas)
            self.plot_layout.addWidget(toolbar)
            canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot multiple spectra: {e}")
            traceback.print_exc()
    
    def apply_spectrum_style(self, ax, plot_params, txt_basename, show_xlabel=True):
        """应用光谱图样式（复用主窗口样式逻辑）"""
        # 设置标签
        if plot_params['xlabel_show'] and show_xlabel:
            ax.set_xlabel(plot_params['xlabel_text'], 
                         fontsize=plot_params['xlabel_fontsize'],
                         labelpad=plot_params['xlabel_pad'],
                         fontfamily=plot_params['font_family'])
        elif not show_xlabel:
            # 多子图模式下，如果不是最后一个子图，不显示x轴标签
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
        
        if plot_params['ylabel_show']:
            ax.set_ylabel(plot_params['ylabel_text'],
                         fontsize=plot_params['ylabel_fontsize'],
                         labelpad=plot_params['ylabel_pad'],
                         fontfamily=plot_params['font_family'])
        
        # 应用刻度样式
        ax.tick_params(labelsize=plot_params['tick_label_fontsize'],
                      direction=plot_params['tick_direction'],
                      width=plot_params['tick_width'])
        ax.tick_params(which='major', length=plot_params['tick_len_major'])
        ax.tick_params(which='minor', length=plot_params['tick_len_minor'])
        
        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(plot_params['font_family'])
        
        # 边框设置
        border_sides = plot_params.get('border_sides', ['top', 'bottom', 'left', 'right'])
        for side in ['top', 'right', 'left', 'bottom']:
            if side in border_sides:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(plot_params['border_linewidth'])
            else:
                ax.spines[side].set_visible(False)
        
        # 网格
        if plot_params['show_grid']:
            ax.grid(True, alpha=plot_params['grid_alpha'])
        else:
            ax.grid(False)
        
        # 图例（与主菜单的高级控制保持一致）
        if plot_params['show_legend']:
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            legend_font.set_family(plot_params['font_family'])
            legend_font.set_size(plot_params['legend_fontsize'])

            legend_ncol = plot_params.get('legend_ncol', 1)
            legend_columnspacing = plot_params.get('legend_columnspacing', 2.0)
            legend_labelspacing = plot_params.get('legend_labelspacing', 0.5)
            legend_handlelength = plot_params.get('legend_handlelength', 2.0)

            ax.legend(
                loc=plot_params['legend_loc'],
                frameon=plot_params['legend_frame'],
                prop=legend_font,
                ncol=legend_ncol,
                columnspacing=legend_columnspacing,
                labelspacing=legend_labelspacing,
                handlelength=legend_handlelength,
            )
        
        # RRUFF匹配结果（如果启用）
        if self.rruff_loader:
            # 获取峰值
            data = self.spectra_data.get(txt_basename)
            if data and len(data.get('peaks', ([], []))[1]) > 0:
                peak_wavenumbers = data['peaks'][1]
                
                # 获取排除列表
                excluded_names = list(self.spectrum_exclusions.get(txt_basename, []))
                for i in range(self.global_exclusion_list.count()):
                    item = self.global_exclusion_list.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        excluded_name = item.text()
                        if excluded_name not in excluded_names:
                            excluded_names.append(excluded_name)
                
                # 查找匹配
                matches = self.peak_matcher.find_best_matches(
                    data['x'], data['y'], peak_wavenumbers, self.rruff_loader,
                    top_k=5, excluded_names=excluded_names if excluded_names else None
                )
                
                if matches:
                    match_text = "RRUFF Matches:\n"
                    for i, match in enumerate(matches[:3]):
                        match_text += f"{i+1}. {match['name']} ({match['match_score']:.2%})\n"
                    ax.text(0.02, 0.98, match_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontfamily='Times New Roman')
    
    def plot_microscopy_image(self, ax, image_path, plot_params):
        """绘制镜下光学图"""
        try:
            img = Image.open(image_path)
            ax.imshow(img)
            ax.set_title('Microscopy Image', fontsize=plot_params.get('axis_title_fontsize', 14),
                        fontfamily=plot_params['font_family'])
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load image:\n{str(e)}",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='red', fontfamily='Times New Roman')
            ax.axis('off')
    
    def export_all_plots(self):
        """批量导出所有图片为PNG"""
        if not self.txt_files:
            QMessageBox.warning(self, "Warning", "Please scan files first")
            return
        
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
        
        plot_params = self.get_parent_plot_params()
        if not plot_params:
            QMessageBox.warning(self, "Warning", "Cannot get plot parameters from main window")
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(self.txt_files))
            self.progress_bar.setValue(0)
            
            for i, txt_file in enumerate(self.txt_files):
                txt_basename = os.path.splitext(os.path.basename(txt_file))[0]
                
                # 读取数据
                x, y = self.data_controller.read_data(
                    txt_file,
                    plot_params['skip_rows'],
                    plot_params['x_min_phys'],
                    plot_params['x_max_phys']
                )
                
                # 创建图形（光谱图和镜下光学图大小一致）
                fig_width = plot_params['fig_width'] * 2.0  # 增加总宽度以容纳两个等大的图
                fig_height = plot_params['fig_height']
                fig_dpi = plot_params['fig_dpi']
                
                fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
                fig.clear()  # 清除默认axes，避免anonymous Axes
                fig.patch.set_visible(False)  # 移除figure的背景框
                gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], hspace=0.1, wspace=0.1)
                
                ax_spectrum = fig.add_subplot(gs[0])
                ax_image = fig.add_subplot(gs[1])
                
                # 复用主窗口的预处理逻辑
                from src.core.preprocessor import DataPreProcessor
                
                y_proc = y.astype(float)
                
                # 预处理（与plot_single_spectrum相同）
                if plot_params.get('qc_enabled', False) and np.max(y_proc) < plot_params.get('qc_threshold', 5.0):
                    continue
                
                if plot_params.get('is_be_correction', False):
                    y_proc = DataPreProcessor.apply_bose_einstein_correction(
                        x, y_proc, plot_params.get('be_temp', 300.0)
                    )
                
                if plot_params.get('is_smoothing', False):
                    y_proc = DataPreProcessor.apply_smoothing(
                        y_proc,
                        plot_params.get('smoothing_window', 15),
                        plot_params.get('smoothing_poly', 3)
                    )
                
                if plot_params.get('is_baseline_als', False):
                    b = DataPreProcessor.apply_baseline_als(
                        y_proc,
                        plot_params.get('als_lam', 10000),
                        plot_params.get('als_p', 0.005)
                    )
                    y_proc = y_proc - b
                    y_proc[y_proc < 0] = 0
                
                norm_mode = plot_params.get('normalization_mode', 'None')
                if norm_mode != 'None':
                    y_proc = DataPreProcessor.apply_normalization(y_proc, norm_mode.lower())
                
                # 绘制光谱
                ax_spectrum.plot(x, y_proc, color='blue', linewidth=plot_params['line_width'],
                               label=txt_basename, linestyle=plot_params['line_style'])
                
                # 峰值检测
                if plot_params.get('peak_detection_enabled', False):
                    peak_params = plot_params.copy()  # 使用plot_params的峰值检测参数
                    self._detect_and_plot_peaks(ax_spectrum, x, y_proc, y_proc, peak_params, color='blue')
                
                # 应用样式
                self.apply_spectrum_style(ax_spectrum, plot_params, txt_basename)
                
                # 绘制镜下光学图
                if txt_basename in self.png_files:
                    self.plot_microscopy_image(ax_image, self.png_files[txt_basename], plot_params)
                else:
                    ax_image.text(0.5, 0.5, "No microscopy\nimage found",
                                 ha='center', va='center', transform=ax_image.transAxes,
                                 fontsize=12, color='gray', fontfamily='Times New Roman')
                    ax_image.axis('off')
                
                # 调整布局（减小边距和间距）
                fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, wspace=0.1)
                
                # 保存
                output_path = os.path.join(save_dir, f"{txt_basename}_plot.png")
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    fig.savefig(output_path, dpi=fig_dpi, bbox_inches='tight', facecolor='white')
                
                plt.close(fig)
                
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()
            
            self.progress_bar.setVisible(False)
            QMessageBox.information(self, "Complete", f"Exported {len(self.txt_files)} images")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to export images: {e}")
            traceback.print_exc()

"""
NMF 面板 mixin
1) 保留 setup_nmf_tab 对外接口
2) 将原 _setup_nmf_tab_internal 的 UI 构建逻辑迁移到此处
"""
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QLabel, QButtonGroup, QRadioButton, QLineEdit,
    QPushButton, QTabWidget, QListWidget, QAbstractItemView, QMenu,
    QGridLayout, QFileDialog
)
from PyQt6.QtCore import Qt

from src.ui.widgets.custom_widgets import CollapsibleGroupBox


class NMFPanelMixin:
    def setup_nmf_tab(self):
        return self._setup_nmf_tab_internal()

    # 委托运行入口到主窗口中的 legacy 实现，方便后续迁移计算逻辑
    def run_nmf_analysis(self):
        if hasattr(self, "_run_nmf_analysis_legacy"):
            return self._run_nmf_analysis_legacy()
        raise AttributeError("NMF legacy analysis implementation not found")

    def run_nmf_regression_mode(self):
        if hasattr(self, "_run_nmf_regression_mode_legacy"):
            return self._run_nmf_regression_mode_legacy()
        raise AttributeError("NMF legacy regression implementation not found")

    # --- Tab 2: NMF 分析 ---
    def _setup_nmf_tab_internal(self):
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
        try:
            from src.core.transformers import TORCH_AVAILABLE
        except Exception:
            TORCH_AVAILABLE = False
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

        # 新增：SVD 去噪选项（NMF专用降噪）
        self.nmf_svd_denoise_check = QCheckBox("启用 SVD 去噪 (NMF专用降噪)")
        self.nmf_svd_denoise_check.setChecked(False)  # 默认不启用
        self.nmf_svd_denoise_check.setToolTip("对NMF输入数据应用SVD去噪，保留指定数量的主成分")

        self.nmf_svd_components_spin = QSpinBox()
        self.nmf_svd_components_spin.setRange(1, 100)
        self.nmf_svd_components_spin.setValue(5)
        self.nmf_svd_components_spin.setToolTip("保留的主成分数量，用于去除随机噪声")

        nmf_layout.addRow(self.nmf_svd_denoise_check)
        nmf_layout.addRow("SVD 主成分数:", self.nmf_svd_components_spin)

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
        self.nmf_weight_line_style.addItems(['-', '--', ':', '-.'])
        self.nmf_weight_line_style.setCurrentText('-')
        
        style_layout.addRow("权重线宽 / 线型:", self._create_h_layout([self.nmf_weight_line_width, self.nmf_weight_line_style]))
        
        # NMF 文件排序设置
        nmf_sort_group = QGroupBox("NMF 文件排序")
        nmf_sort_layout = QGridLayout(nmf_sort_group)
        
        self.nmf_sort_method_combo = QComboBox()
        self.nmf_sort_method_combo.addItems(['按文件名排序', '按数值排序'])
        self.nmf_sort_method_combo.setCurrentText('按文件名排序')
        
        self.nmf_sort_reverse_check = QCheckBox("反向排序")
        self.nmf_sort_reverse_check.setChecked(False)
        
        nmf_sort_layout.addWidget(QLabel("排序方法:"), 0, 0)
        nmf_sort_layout.addWidget(self.nmf_sort_method_combo, 0, 1)
        nmf_sort_layout.addWidget(self.nmf_sort_reverse_check, 0, 2)
        
        style_layout.addRow(nmf_sort_group)
        
        # 将样式布局添加到 CollapsibleGroupBox
        style_group.setContentLayout(style_layout)
        layout.addWidget(style_group)
        
        # --- C. NMF 文件预览和控制 ---
        file_group = CollapsibleGroupBox("📂 NMF 文件预览与控制", is_expanded=True)
        file_layout = QVBoxLayout()
        
        # 文件列表预览
        self.nmf_file_preview_list = QListWidget()
        self.nmf_file_preview_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        # 添加右键菜单：删除/恢复文件（使用主类实现的显示逻辑）
        self.nmf_file_preview_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        if hasattr(self, "_show_nmf_file_context_menu"):
            self.nmf_file_preview_list.customContextMenuRequested.connect(self._show_nmf_file_context_menu)
        
        file_layout.addWidget(self.nmf_file_preview_list)
        
        # 控制按钮：刷新预览、全选、清空
        h_file_ctrl = self._create_h_layout([
            QPushButton("🔄 刷新预览"),
            QPushButton("✅ 全选"),
            QPushButton("🧹 清空")
        ])
        self.nmf_refresh_btn, self.nmf_select_all_btn, self.nmf_clear_btn = h_file_ctrl.findChildren(QPushButton)
        # 绑定主类已有的预览刷新/全选/清空方法
        if hasattr(self, "_update_nmf_sort_preview"):
            self.nmf_refresh_btn.clicked.connect(self._update_nmf_sort_preview)
        if hasattr(self, "_nmf_select_all"):
            self.nmf_select_all_btn.clicked.connect(self._nmf_select_all)
        if hasattr(self, "_nmf_clear_selection"):
            self.nmf_clear_btn.clicked.connect(self._nmf_clear_selection)
        
        file_layout.addWidget(h_file_ctrl)
        
        file_group.setContentLayout(file_layout)
        layout.addWidget(file_group)

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
        
        # --- D. NMF 运行按钮已移除（在主窗口左侧按钮区）---
        
        # --- E. Tab 容器 ---
        self.nmf_tab_container = tab2
        return tab2

    def save_config(self):
        """保存NMF配置"""
        # 注意：NMF面板的配置通过主窗口的settings保存
        pass

    def load_config(self):
        """加载NMF配置"""
        # 注意：NMF面板的配置通过主窗口的settings加载
        pass


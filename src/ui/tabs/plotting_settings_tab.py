"""Plotting settings tab widget"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QLabel, QGroupBox
)
from src.ui.widgets.custom_widgets import CollapsibleGroupBox, UnlimitedNumericInput


class PlottingSettingsTab(QWidget):
    """绘图设置 Tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent  # 保存主窗口引用，用于访问辅助方法
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        grid_layout = QGridLayout(self)
        grid_layout.setSpacing(10)

        # --- 1. 左侧：X轴截断 + 预处理 ---
        left_vbox = QVBoxLayout()

        # 1.1 X 轴截断（物理 + 多段）
        x_trunc_group = CollapsibleGroupBox("1. X 轴截断", is_expanded=True)
        x_trunc_layout = QFormLayout()

        # 物理 Min / Max 截断
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
        
        # 1.2 数据预处理
        preprocess_group = CollapsibleGroupBox("2. 数据预处理 (AsLS / QC / BE / SNV)", is_expanded=True)
        prep_layout = QFormLayout()
        
        self.qc_check = QCheckBox("启用 QC (剔除弱信号)")
        self.qc_threshold_spin = UnlimitedNumericInput(default_value="5.0")
        prep_layout.addRow(self._create_h_layout([self.qc_check, QLabel("阈值:"), self.qc_threshold_spin]))
        
        # Bose-Einstein 修正
        self.be_check = QCheckBox("启用 Bose-Einstein 校正")
        self.be_temp_spin = UnlimitedNumericInput(default_value="300.0")
        prep_layout.addRow(self.be_check)
        prep_layout.addRow("BE 温度 T (K):", self.be_temp_spin)
        
        self.baseline_als_check = QCheckBox("启用 AsLS 基线校正 (推荐)")
        self.lam_spin = UnlimitedNumericInput(default_value="10000")
        self.p_spin = UnlimitedNumericInput(default_value="0.005")
        prep_layout.addRow(self.baseline_als_check)
        prep_layout.addRow("Lambda (平滑度):", self.lam_spin)
        prep_layout.addRow("P (非对称度):", self.p_spin)

        # 多点多项式基线校正
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
        
        grid_layout.addLayout(left_vbox, 0, 0, 1, 1)  # 左侧布局

        # --- 2. 右侧：绘图样式 ---
        right_vbox = QVBoxLayout()
        
        # 2.0 自动更新开关
        auto_update_group = CollapsibleGroupBox("⚙️ 自动更新设置", is_expanded=False)
        auto_update_layout = QFormLayout()
        
        self.auto_update_check = QCheckBox("启用自动更新（参数改变时自动重新绘制当前谱图）")
        self.auto_update_check.setChecked(True)
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
        
        # 整体Y轴偏移
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
        
        self.global_stack_offset_spin = QDoubleSpinBox()
        self.global_stack_offset_spin.setRange(-999999999.0, 999999999.0)
        self.global_stack_offset_spin.setDecimals(15)
        self.global_stack_offset_spin.setValue(0.5)
        
        self.global_y_scale_factor_spin = QDoubleSpinBox()
        self.global_y_scale_factor_spin.setRange(-999999999.0, 999999999.0)
        self.global_y_scale_factor_spin.setDecimals(15)
        self.global_y_scale_factor_spin.setValue(1.0)
        self.global_y_scale_factor_spin.setSingleStep(0.1)
        
        style_layout.addRow("Y缩放:", self.global_y_scale_factor_spin)

        plot_style_group.setContentLayout(style_layout)
        right_vbox.addWidget(plot_style_group)
        
        right_vbox.addStretch(1)
        grid_layout.addLayout(right_vbox, 0, 1, 1, 1)  # 右侧布局
    
    def _create_h_layout(self, widgets):
        """创建水平布局的辅助方法"""
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(5)
        for wid in widgets:
            l.addWidget(wid)
        return w
    
    def get_widgets_dict(self):
        """获取所有控件的字典，用于 ConfigBinder"""
        return {
            'x_min_phys_input': self.x_min_phys_input,
            'x_max_phys_input': self.x_max_phys_input,
            'x_segments_input': self.x_segments_input,
            'qc_check': self.qc_check,
            'qc_threshold_spin': self.qc_threshold_spin,
            'be_check': self.be_check,
            'be_temp_spin': self.be_temp_spin,
            'baseline_als_check': self.baseline_als_check,
            'lam_spin': self.lam_spin,
            'p_spin': self.p_spin,
            'baseline_poly_check': self.baseline_poly_check,
            'baseline_points_spin': self.baseline_points_spin,
            'baseline_poly_spin': self.baseline_poly_spin,
            'smoothing_check': self.smoothing_check,
            'smoothing_window_spin': self.smoothing_window_spin,
            'smoothing_poly_spin': self.smoothing_poly_spin,
            'normalization_combo': self.normalization_combo,
            'global_transform_combo': self.global_transform_combo,
            'global_log_base_combo': self.global_log_base_combo,
            'global_log_offset_spin': self.global_log_offset_spin,
            'global_sqrt_offset_spin': self.global_sqrt_offset_spin,
            'auto_update_check': self.auto_update_check,
            'plot_mode_combo': self.plot_mode_combo,
            'global_y_offset_spin': self.global_y_offset_spin,
            'plot_style_combo': self.plot_style_combo,
            'global_stack_offset_spin': self.global_stack_offset_spin,
            'global_y_scale_factor_spin': self.global_y_scale_factor_spin,
        }


"""
峰值匹配面板
支持多模式峰值匹配功能
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QLabel, QGroupBox, QLineEdit, QPushButton
)
from PyQt6.QtCore import pyqtSignal

from src.ui.widgets.custom_widgets import CollapsibleGroupBox
from src.core.plot_config_manager import PlotConfigManager


class PeakMatchingPanel(QWidget):
    """峰值匹配面板"""
    
    # 信号：配置改变时发出
    config_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_manager = PlotConfigManager()
        self.setup_ui()
        self.load_config()
        self.connect_signals()
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 峰值匹配组（紧凑版）
        matching_group = CollapsibleGroupBox("🔍 峰值匹配", is_expanded=True)
        matching_layout = QFormLayout()
        matching_layout.setSpacing(8)  # 减小间距
        matching_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)  # 允许换行
        
        # 启用峰值匹配
        self.enabled_check = QCheckBox("启用峰值匹配")
        self.enabled_check.setChecked(False)
        matching_layout.addRow(self.enabled_check)
        
        # 匹配模式（使用更紧凑的布局）
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        # 使用元组存储 (显示文本, 实际值, 说明)
        self.mode_items = [
            ("显示参考谱所有峰值", "all_peaks", "只显示参考光谱检测到的所有峰值，不进行匹配"),
            ("显示匹配到的峰值", "matched_only", "显示每个谱线与参考谱线匹配到的峰值"),
            ("显示所有谱线共有的峰值", "all_matched", "只显示所有谱线都匹配到的共同峰值"),
            ("在顶部显示参考峰值", "top_display", "在最上方谱线显示参考光谱的峰值")
        ]
        for display_text, _, _ in self.mode_items:
            self.mode_combo.addItem(display_text)
        self.mode_combo.setMaximumWidth(200)
        # 添加工具提示
        for i, (_, _, tooltip) in enumerate(self.mode_items):
            self.mode_combo.setItemData(i, tooltip, role=256)  # Qt.ToolTipRole
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        matching_layout.addRow("匹配模式:", mode_layout)
        
        # 模式说明标签
        self.mode_description_label = QLabel("")
        self.mode_description_label.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        self.mode_description_label.setWordWrap(True)
        matching_layout.addRow("", self.mode_description_label)
        
        # 匹配容差和参考索引（同一行）
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("容差:"))
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.1, 100.0)
        self.tolerance_spin.setDecimals(1)
        self.tolerance_spin.setValue(5.0)
        self.tolerance_spin.setMaximumWidth(80)
        self.tolerance_spin.setToolTip("峰值匹配容差（cm⁻¹）")
        params_layout.addWidget(self.tolerance_spin)
        
        params_layout.addWidget(QLabel("参考索引:"))
        self.reference_index_spin = QSpinBox()
        self.reference_index_spin.setRange(-999, 999)
        self.reference_index_spin.setValue(-1)
        self.reference_index_spin.setMaximumWidth(80)
        self.reference_index_spin.setToolTip("参考光谱索引（-1=最后一个）")
        params_layout.addWidget(self.reference_index_spin)
        params_layout.addStretch()
        
        matching_layout.addRow("参数:", params_layout)
        
        # ========== 标记样式控制 ==========
        marker_group = CollapsibleGroupBox("📍 标记样式", is_expanded=False)
        marker_layout = QFormLayout()
        marker_layout.setSpacing(8)
        
        # 标记形状
        self.marker_shape_combo = QComboBox()
        self.marker_shape_combo.addItems(['v', 'o', 's', '^', 'D', '*', '+', 'x'])
        self.marker_shape_combo.setCurrentText('v')
        marker_layout.addRow("标记形状:", self.marker_shape_combo)
        
        # 标记大小和距离（同一行）
        marker_size_layout = QHBoxLayout()
        marker_size_layout.addWidget(QLabel("大小:"))
        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(0.1, 100.0)
        self.marker_size_spin.setDecimals(1)
        self.marker_size_spin.setValue(8.0)
        self.marker_size_spin.setMaximumWidth(80)
        marker_size_layout.addWidget(self.marker_size_spin)
        
        marker_size_layout.addWidget(QLabel("距离:"))
        self.marker_distance_spin = QDoubleSpinBox()
        self.marker_distance_spin.setRange(-1000.0, 1000.0)
        self.marker_distance_spin.setDecimals(2)
        self.marker_distance_spin.setValue(0.0)
        self.marker_distance_spin.setMaximumWidth(80)
        self.marker_distance_spin.setToolTip("标记离谱线的Y轴偏移距离")
        marker_size_layout.addWidget(self.marker_distance_spin)
        marker_size_layout.addStretch()
        marker_layout.addRow("标记大小/距离:", marker_size_layout)
        
        # 标记旋转
        self.marker_rotation_spin = QDoubleSpinBox()
        self.marker_rotation_spin.setRange(-360.0, 360.0)
        self.marker_rotation_spin.setDecimals(1)
        self.marker_rotation_spin.setValue(0.0)
        self.marker_rotation_spin.setSuffix("°")
        self.marker_rotation_spin.setMaximumWidth(100)
        marker_layout.addRow("标记旋转角度:", self.marker_rotation_spin)
        
        marker_group.setContentLayout(marker_layout)
        layout.addWidget(marker_group)
        
        # ========== 谱线连接控制 ==========
        connection_group = CollapsibleGroupBox("🔗 谱线连接", is_expanded=False)
        connection_layout = QFormLayout()
        connection_layout.setSpacing(8)
        
        # 启用连接线
        self.show_connection_lines_check = QCheckBox("显示连接匹配峰值的谱线")
        self.show_connection_lines_check.setChecked(False)
        connection_layout.addRow(self.show_connection_lines_check)
        
        # 连接线颜色模式
        self.use_spectrum_color_check = QCheckBox("使用各自谱线颜色（取消勾选则使用统一颜色）")
        self.use_spectrum_color_check.setChecked(True)  # 默认使用谱线颜色
        connection_layout.addRow(self.use_spectrum_color_check)
        
        # 连接线颜色（仅在统一颜色模式下使用）
        color_layout = QHBoxLayout()
        self.connection_line_color_input = QLineEdit("red")
        self.connection_line_color_input.setMaximumWidth(100)
        self.connection_line_color_input.setEnabled(False)  # 默认禁用，因为使用谱线颜色
        btn_color = QPushButton("...")
        btn_color.setMaximumWidth(30)
        btn_color.setEnabled(False)  # 默认禁用
        btn_color.clicked.connect(lambda: self._pick_color(self.connection_line_color_input))
        color_layout.addWidget(QLabel("统一颜色:"))
        color_layout.addWidget(self.connection_line_color_input)
        color_layout.addWidget(btn_color)
        color_layout.addStretch()
        connection_layout.addRow(color_layout)
        
        # 连接颜色模式改变时，启用/禁用颜色输入
        self.use_spectrum_color_check.stateChanged.connect(lambda state: self.connection_line_color_input.setEnabled(state == 0))
        self.use_spectrum_color_check.stateChanged.connect(lambda state: btn_color.setEnabled(state == 0))
        
        # 连接线宽度和样式（同一行）
        line_style_layout = QHBoxLayout()
        line_style_layout.addWidget(QLabel("宽度:"))
        self.connection_line_width_spin = QDoubleSpinBox()
        self.connection_line_width_spin.setRange(0.1, 10.0)
        self.connection_line_width_spin.setDecimals(2)
        self.connection_line_width_spin.setValue(1.0)
        self.connection_line_width_spin.setMaximumWidth(80)
        line_style_layout.addWidget(self.connection_line_width_spin)
        
        line_style_layout.addWidget(QLabel("样式:"))
        self.connection_line_style_combo = QComboBox()
        self.connection_line_style_combo.addItems(['-', '--', ':', '-.'])
        self.connection_line_style_combo.setCurrentText('-')
        self.connection_line_style_combo.setMaximumWidth(80)
        line_style_layout.addWidget(self.connection_line_style_combo)
        line_style_layout.addStretch()
        connection_layout.addRow("连接线宽度/样式:", line_style_layout)
        
        # 连接线透明度
        self.connection_line_alpha_spin = QDoubleSpinBox()
        self.connection_line_alpha_spin.setRange(0.0, 1.0)
        self.connection_line_alpha_spin.setDecimals(2)
        self.connection_line_alpha_spin.setValue(0.8)
        self.connection_line_alpha_spin.setSingleStep(0.1)
        self.connection_line_alpha_spin.setMaximumWidth(100)
        connection_layout.addRow("连接线透明度:", self.connection_line_alpha_spin)
        
        connection_group.setContentLayout(connection_layout)
        layout.addWidget(connection_group)
        
        # ========== 峰值数字显示控制 ==========
        label_group = CollapsibleGroupBox("🔢 峰值数字显示", is_expanded=False)
        label_layout = QFormLayout()
        label_layout.setSpacing(8)
        
        # 启用数字显示
        self.show_peak_labels_check = QCheckBox("显示峰值数字")
        self.show_peak_labels_check.setChecked(False)
        label_layout.addRow(self.show_peak_labels_check)
        
        # 标签字体大小和颜色（同一行）
        label_font_layout = QHBoxLayout()
        label_font_layout.addWidget(QLabel("字体大小:"))
        self.label_fontsize_spin = QDoubleSpinBox()
        self.label_fontsize_spin.setRange(1.0, 100.0)
        self.label_fontsize_spin.setDecimals(1)
        self.label_fontsize_spin.setValue(10.0)
        self.label_fontsize_spin.setMaximumWidth(80)
        label_font_layout.addWidget(self.label_fontsize_spin)
        
        label_font_layout.addWidget(QLabel("颜色:"))
        self.label_color_input = QLineEdit("black")
        self.label_color_input.setMaximumWidth(100)
        btn_label_color = QPushButton("...")
        btn_label_color.setMaximumWidth(30)
        btn_label_color.clicked.connect(lambda: self._pick_color(self.label_color_input))
        label_font_layout.addWidget(self.label_color_input)
        label_font_layout.addWidget(btn_label_color)
        label_font_layout.addStretch()
        label_layout.addRow("标签字体/颜色:", label_font_layout)
        
        # 标签旋转和距离（同一行）
        label_pos_layout = QHBoxLayout()
        label_pos_layout.addWidget(QLabel("旋转:"))
        self.label_rotation_spin = QDoubleSpinBox()
        self.label_rotation_spin.setRange(-360.0, 360.0)
        self.label_rotation_spin.setDecimals(1)
        self.label_rotation_spin.setValue(0.0)
        self.label_rotation_spin.setSuffix("°")
        self.label_rotation_spin.setMaximumWidth(100)
        label_pos_layout.addWidget(self.label_rotation_spin)
        
        label_pos_layout.addWidget(QLabel("距离:"))
        self.label_distance_spin = QDoubleSpinBox()
        self.label_distance_spin.setRange(0.0, 100.0)
        self.label_distance_spin.setDecimals(1)
        self.label_distance_spin.setValue(5.0)
        self.label_distance_spin.setMaximumWidth(80)
        self.label_distance_spin.setToolTip("标签离谱线的距离（像素）")
        label_pos_layout.addWidget(self.label_distance_spin)
        label_pos_layout.addStretch()
        label_layout.addRow("标签旋转/距离:", label_pos_layout)
        
        label_group.setContentLayout(label_layout)
        layout.addWidget(label_group)
        
        matching_group.setContentLayout(matching_layout)
        layout.addWidget(matching_group)
    
    def _pick_color(self, line_edit):
        """选择颜色"""
        from PyQt6.QtWidgets import QColorDialog
        color = QColorDialog.getColor()
        if color.isValid():
            line_edit.setText(color.name())
            self._on_config_changed()
    
    def _on_mode_changed(self, index):
        """匹配模式改变时"""
        if 0 <= index < len(self.mode_items):
            _, _, description = self.mode_items[index]
            self.mode_description_label.setText(description)
        self._on_config_changed()
    
    def connect_signals(self):
        """连接信号"""
        self.enabled_check.stateChanged.connect(self._on_config_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.tolerance_spin.valueChanged.connect(self._on_config_changed)
        self.reference_index_spin.valueChanged.connect(self._on_config_changed)
        
        # 标记样式
        self.marker_shape_combo.currentTextChanged.connect(self._on_config_changed)
        self.marker_size_spin.valueChanged.connect(self._on_config_changed)
        self.marker_distance_spin.valueChanged.connect(self._on_config_changed)
        self.marker_rotation_spin.valueChanged.connect(self._on_config_changed)
        
        # 谱线连接
        self.show_connection_lines_check.stateChanged.connect(self._on_config_changed)
        self.use_spectrum_color_check.stateChanged.connect(self._on_config_changed)
        self.connection_line_color_input.textChanged.connect(self._on_config_changed)
        self.connection_line_width_spin.valueChanged.connect(self._on_config_changed)
        self.connection_line_style_combo.currentTextChanged.connect(self._on_config_changed)
        self.connection_line_alpha_spin.valueChanged.connect(self._on_config_changed)
        
        # 峰值数字显示
        self.show_peak_labels_check.stateChanged.connect(self._on_config_changed)
        self.label_fontsize_spin.valueChanged.connect(self._on_config_changed)
        self.label_color_input.textChanged.connect(self._on_config_changed)
        self.label_rotation_spin.valueChanged.connect(self._on_config_changed)
        self.label_distance_spin.valueChanged.connect(self._on_config_changed)
    
    def _on_config_changed(self):
        """配置改变时"""
        self.save_config()
        self.config_changed.emit()
    
    def load_config(self):
        """从配置管理器加载配置"""
        config = self.config_manager.get_config()
        pm = config.peak_matching
        
        self.enabled_check.setChecked(pm.enabled)
        # 根据模式找到对应的索引
        mode_index = -1
        for i, (_, value, _) in enumerate(self.mode_items):
            if value == pm.mode:
                mode_index = i
                break
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)
            self._on_mode_changed(mode_index)
        
        self.tolerance_spin.setValue(pm.tolerance)
        self.reference_index_spin.setValue(pm.reference_index)
        
        # 加载标记样式
        self.marker_shape_combo.setCurrentText(pm.marker_shape)
        self.marker_size_spin.setValue(pm.marker_size)
        self.marker_distance_spin.setValue(pm.marker_distance)
        self.marker_rotation_spin.setValue(pm.marker_rotation)
        
        # 加载谱线连接
        self.show_connection_lines_check.setChecked(pm.show_connection_lines)
        self.use_spectrum_color_check.setChecked(pm.use_spectrum_color_for_connection)
        self.connection_line_color_input.setText(pm.connection_line_color)
        self.connection_line_color_input.setEnabled(not pm.use_spectrum_color_for_connection)
        # 找到颜色按钮并设置启用状态
        for widget in self.findChildren(QPushButton):
            if widget.text() == "...":
                # 检查是否是连接线颜色按钮（通过布局位置判断）
                parent_layout = widget.parent().layout()
                if parent_layout and isinstance(parent_layout, QHBoxLayout):
                    if self.connection_line_color_input in [parent_layout.itemAt(i).widget() for i in range(parent_layout.count())]:
                        widget.setEnabled(not pm.use_spectrum_color_for_connection)
                        break
        self.connection_line_width_spin.setValue(pm.connection_line_width)
        self.connection_line_style_combo.setCurrentText(pm.connection_line_style)
        self.connection_line_alpha_spin.setValue(pm.connection_line_alpha)
        
        # 加载峰值数字显示
        self.show_peak_labels_check.setChecked(pm.show_peak_labels)
        self.label_fontsize_spin.setValue(pm.label_fontsize)
        self.label_color_input.setText(pm.label_color)
        self.label_rotation_spin.setValue(pm.label_rotation)
        self.label_distance_spin.setValue(pm.label_distance)
    
    def save_config(self):
        """保存配置到配置管理器"""
        config = self.config_manager.get_config()
        pm = config.peak_matching
        
        pm.enabled = self.enabled_check.isChecked()
        # 获取模式的实际值
        current_index = self.mode_combo.currentIndex()
        if 0 <= current_index < len(self.mode_items):
            _, pm.mode, _ = self.mode_items[current_index]
        else:
            pm.mode = "all_matched"
        
        pm.tolerance = self.tolerance_spin.value()
        pm.reference_index = self.reference_index_spin.value()
        
        # 保存标记样式
        pm.marker_shape = self.marker_shape_combo.currentText()
        pm.marker_size = self.marker_size_spin.value()
        pm.marker_distance = self.marker_distance_spin.value()
        pm.marker_rotation = self.marker_rotation_spin.value()
        
        # 保存谱线连接
        pm.show_connection_lines = self.show_connection_lines_check.isChecked()
        pm.use_spectrum_color_for_connection = self.use_spectrum_color_check.isChecked()
        pm.connection_line_color = self.connection_line_color_input.text().strip() or 'red'
        pm.connection_line_width = self.connection_line_width_spin.value()
        pm.connection_line_style = self.connection_line_style_combo.currentText()
        pm.connection_line_alpha = self.connection_line_alpha_spin.value()
        
        # 保存峰值数字显示
        pm.show_peak_labels = self.show_peak_labels_check.isChecked()
        pm.label_fontsize = self.label_fontsize_spin.value()
        pm.label_color = self.label_color_input.text().strip() or 'black'
        pm.label_rotation = self.label_rotation_spin.value()
        pm.label_distance = self.label_distance_spin.value()
        
        self.config_manager.update_config(config)
    
    def get_config(self):
        """获取当前配置"""
        self.save_config()
        return self.config_manager.get_config()


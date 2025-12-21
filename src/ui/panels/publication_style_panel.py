"""
出版质量样式控制面板
包含所有出版质量样式设置，包括X/Y轴标题、主标题控制
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QLineEdit, QPushButton, QLabel
)
from PyQt6.QtCore import pyqtSignal

from src.ui.widgets.custom_widgets import CollapsibleGroupBox
from src.core.plot_config_manager import PlotConfigManager, PlotConfig


class PublicationStylePanel(QWidget):
    """出版质量样式控制面板"""
    
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
        
        # 创建可折叠组
        pub_style_group = CollapsibleGroupBox("💎 出版质量样式控制", is_expanded=True)
        pub_layout = QFormLayout()
        
        # Figure/DPI
        # 注意：fig_width 和 fig_height 已删除（没有实际作用）
        self.fig_dpi_spin = QSpinBox()
        self.fig_dpi_spin.setRange(-999999999, 999999999)
        self.fig_dpi_spin.setValue(300)
        
        self.aspect_ratio_spin = QDoubleSpinBox()
        self.aspect_ratio_spin.setRange(-999999999.0, 999999999.0)
        self.aspect_ratio_spin.setDecimals(15)
        self.aspect_ratio_spin.setValue(0.6)
        
        # 风格预设
        self.style_preset_combo = QComboBox()
        self._load_custom_presets()
        
        btn_manage_presets = QPushButton("管理预设")
        btn_manage_presets.setToolTip("创建、编辑或删除自定义风格预设")
        btn_manage_presets.clicked.connect(self._manage_style_presets)
        
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(self.style_preset_combo)
        preset_layout.addWidget(btn_manage_presets)
        pub_layout.addRow("风格预设:", preset_layout)
        
        # 注意：图尺寸W/H已删除（没有实际作用）
        # 注意：字体大小（轴/刻度/图例）已删除（下面有实现方法）
        pub_layout.addRow("DPI / 纵横比:", self._create_h_layout([self.fig_dpi_spin, self.aspect_ratio_spin]))
        
        # Font
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(['Times New Roman', 'Arial', 'SimHei'])
        
        pub_layout.addRow("字体家族:", self.font_family_combo)
        
        # Lines
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(-999999999.0, 999999999.0)
        self.line_width_spin.setDecimals(15)
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
        
        # 注意：刻度显示控制（上下左右）已删除（没用且不会自动更新）
        
        pub_layout.addRow("刻度方向 / 宽度:", self._create_h_layout([self.tick_direction_combo, self.tick_width_spin]))
        pub_layout.addRow("刻度长度 (大/小):", self._create_h_layout([self.tick_len_major_spin, self.tick_len_minor_spin]))
        
        # Grid/Shadow
        self.show_grid_check = QCheckBox("显示网格")
        self.show_grid_check.setChecked(True)
        
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
        
        self.legend_fontsize_spin = QSpinBox()
        self.legend_fontsize_spin.setRange(-999999999, 999999999)
        self.legend_fontsize_spin.setValue(10)
        
        self.legend_column_spin = QSpinBox()
        self.legend_column_spin.setRange(-999999999, 999999999)
        self.legend_column_spin.setValue(1)
        
        self.legend_columnspacing_spin = QDoubleSpinBox()
        self.legend_columnspacing_spin.setRange(-999999999.0, 999999999.0)
        self.legend_columnspacing_spin.setDecimals(15)
        self.legend_columnspacing_spin.setValue(2.0)
        
        self.legend_labelspacing_spin = QDoubleSpinBox()
        self.legend_labelspacing_spin.setRange(-999999999.0, 999999999.0)
        self.legend_labelspacing_spin.setDecimals(15)
        self.legend_labelspacing_spin.setValue(0.5)
        
        self.legend_handlelength_spin = QDoubleSpinBox()
        self.legend_handlelength_spin.setRange(-999999999.0, 999999999.0)
        self.legend_handlelength_spin.setDecimals(15)
        self.legend_handlelength_spin.setValue(2.0)
        
        pub_layout.addRow(self._create_h_layout([self.show_legend_check, self.legend_frame_check]))
        pub_layout.addRow("图例位置:", self.legend_loc_combo)
        pub_layout.addRow("图例字体大小:", self.legend_fontsize_spin)
        pub_layout.addRow("图例列数:", self.legend_column_spin)
        pub_layout.addRow("图例列间距 / 标签间距:", self._create_h_layout([self.legend_columnspacing_spin, self.legend_labelspacing_spin]))
        pub_layout.addRow("图例句柄长度:", self.legend_handlelength_spin)
        
        # ========== 标题控制（新增，集成到出版质量样式控制中）==========
        # X轴标题
        self.xlabel_input = QLineEdit(r"Wavenumber ($\mathrm{cm^{-1}}$)")
        self.xlabel_show_check = QCheckBox("显示X轴标题")
        self.xlabel_show_check.setChecked(True)
        self.xlabel_font_spin = QSpinBox()
        self.xlabel_font_spin.setRange(-999999999, 999999999)
        self.xlabel_font_spin.setValue(20)
        self.xlabel_pad_spin = QDoubleSpinBox()
        self.xlabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.xlabel_pad_spin.setDecimals(15)
        self.xlabel_pad_spin.setValue(10.0)
        
        pub_layout.addRow("X轴标题:", self.xlabel_input)
        pub_layout.addRow("X轴标题控制:", self._create_h_layout([self.xlabel_show_check, QLabel("大小:"), self.xlabel_font_spin, QLabel("间距:"), self.xlabel_pad_spin]))
        
        # Y轴标题
        self.ylabel_input = QLineEdit("Intensity")
        self.ylabel_show_check = QCheckBox("显示Y轴标题")
        self.ylabel_show_check.setChecked(True)
        self.ylabel_font_spin = QSpinBox()
        self.ylabel_font_spin.setRange(-999999999, 999999999)
        self.ylabel_font_spin.setValue(20)
        self.ylabel_pad_spin = QDoubleSpinBox()
        self.ylabel_pad_spin.setRange(-999999999.0, 999999999.0)
        self.ylabel_pad_spin.setDecimals(15)
        self.ylabel_pad_spin.setValue(10.0)
        
        pub_layout.addRow("Y轴标题:", self.ylabel_input)
        pub_layout.addRow("Y轴标题控制:", self._create_h_layout([self.ylabel_show_check, QLabel("大小:"), self.ylabel_font_spin, QLabel("间距:"), self.ylabel_pad_spin]))
        
        # 主标题
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("主图标题 (留空则显示组名)")
        self.title_show_check = QCheckBox("显示主标题")
        self.title_show_check.setChecked(True)
        self.title_font_spin = QSpinBox()
        self.title_font_spin.setRange(-999999999, 999999999)
        self.title_font_spin.setValue(18)
        self.title_pad_spin = QDoubleSpinBox()
        self.title_pad_spin.setRange(-999999999.0, 999999999.0)
        self.title_pad_spin.setDecimals(15)
        self.title_pad_spin.setValue(10.0)
        
        pub_layout.addRow("主标题:", self.title_input)
        pub_layout.addRow("主标题控制:", self._create_h_layout([self.title_show_check, QLabel("大小:"), self.title_font_spin, QLabel("间距:"), self.title_pad_spin]))
        
        # ========== 坐标轴显示控制（新增）==========
        # X轴翻转
        self.x_axis_invert_check = QCheckBox("X轴翻转")
        self.x_axis_invert_check.setChecked(False)
        
        # 显示X轴数值
        self.show_x_values_check = QCheckBox("显示X轴数值")
        self.show_x_values_check.setChecked(True)
        
        # 显示Y轴数值
        self.show_y_values_check = QCheckBox("显示Y轴数值")
        self.show_y_values_check.setChecked(True)
        
        pub_layout.addRow("坐标轴控制:", self._create_h_layout([self.x_axis_invert_check, self.show_x_values_check, self.show_y_values_check]))
        # ============================================================
        
        pub_style_group.setContentLayout(pub_layout)
        layout.addWidget(pub_style_group)
    
    def _create_h_layout(self, widgets):
        """创建水平布局"""
        layout = QHBoxLayout()
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch()
        widget = QWidget()
        widget.setLayout(layout)
        return widget
    
    def connect_signals(self):
        """连接信号"""
        # 连接所有控件的信号到配置更新
        for widget in self.findChildren(QDoubleSpinBox):
            widget.valueChanged.connect(self._on_config_changed)
        for widget in self.findChildren(QSpinBox):
            widget.valueChanged.connect(self._on_config_changed)
        for widget in self.findChildren(QComboBox):
            widget.currentTextChanged.connect(self._on_config_changed)
        for widget in self.findChildren(QCheckBox):
            widget.stateChanged.connect(self._on_config_changed)
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self._on_config_changed)
        
        # 预设选择
        self.style_preset_combo.currentTextChanged.connect(self._on_preset_changed)
    
    def _on_config_changed(self):
        """配置改变时"""
        self.save_config()
        self.config_changed.emit()
    
    def _on_preset_changed(self, preset_name: str):
        """预设改变时"""
        self.apply_preset(preset_name)
        self.config_changed.emit()
    
    def load_config(self):
        """从配置管理器加载配置"""
        config = self.config_manager.get_config()
        ps = config.publication_style
        
        # 加载基本样式
        # 注意：fig_width 和 fig_height 已删除（没有实际作用）
        self.fig_dpi_spin.setValue(ps.fig_dpi)
        self.aspect_ratio_spin.setValue(ps.aspect_ratio)
        self.font_family_combo.setCurrentText(ps.font_family)
        # 注意：axis_title_fontsize 和 tick_label_fontsize 已删除（下面有实现方法）
        self.line_width_spin.setValue(ps.line_width)
        self.line_style_combo.setCurrentText(ps.line_style)
        self.tick_direction_combo.setCurrentText(ps.tick_direction)
        self.tick_len_major_spin.setValue(ps.tick_len_major)
        self.tick_len_minor_spin.setValue(ps.tick_len_minor)
        # 注意：tick_top/bottom/left/right 已删除（没用且不会自动更新）
        self.tick_width_spin.setValue(ps.tick_width)
        self.show_grid_check.setChecked(ps.show_grid)
        self.grid_alpha_spin.setValue(ps.grid_alpha)
        self.shadow_alpha_spin.setValue(ps.shadow_alpha)
        self.spine_top_check.setChecked(ps.spine_top)
        self.spine_bottom_check.setChecked(ps.spine_bottom)
        self.spine_left_check.setChecked(ps.spine_left)
        self.spine_right_check.setChecked(ps.spine_right)
        self.spine_width_spin.setValue(ps.spine_width)
        self.show_legend_check.setChecked(ps.show_legend)
        self.legend_frame_check.setChecked(ps.legend_frame)
        self.legend_loc_combo.setCurrentText(ps.legend_loc)
        self.legend_fontsize_spin.setValue(ps.legend_fontsize)
        self.legend_column_spin.setValue(ps.legend_ncol)
        self.legend_columnspacing_spin.setValue(ps.legend_columnspacing)
        self.legend_labelspacing_spin.setValue(ps.legend_labelspacing)
        self.legend_handlelength_spin.setValue(ps.legend_handlelength)
        
        # 加载标题控制
        self.xlabel_input.setText(ps.xlabel_text)
        self.xlabel_show_check.setChecked(ps.xlabel_show)
        self.xlabel_font_spin.setValue(ps.xlabel_fontsize)
        self.xlabel_pad_spin.setValue(ps.xlabel_pad)
        self.ylabel_input.setText(ps.ylabel_text)
        self.ylabel_show_check.setChecked(ps.ylabel_show)
        self.ylabel_font_spin.setValue(ps.ylabel_fontsize)
        self.ylabel_pad_spin.setValue(ps.ylabel_pad)
        self.title_input.setText(ps.title_text)
        self.title_show_check.setChecked(ps.title_show)
        self.title_font_spin.setValue(ps.title_fontsize)
        self.title_pad_spin.setValue(ps.title_pad)
        
        # 加载坐标轴显示控制
        self.x_axis_invert_check.setChecked(ps.x_axis_invert)
        self.show_x_values_check.setChecked(ps.show_x_values)
        self.show_y_values_check.setChecked(ps.show_y_values)
    
    def save_config(self):
        """保存配置到配置管理器"""
        config = self.config_manager.get_config()
        ps = config.publication_style
        
        # 保存基本样式
        # 注意：fig_width 和 fig_height 已删除（没有实际作用）
        ps.fig_dpi = self.fig_dpi_spin.value()
        ps.aspect_ratio = self.aspect_ratio_spin.value()
        ps.font_family = self.font_family_combo.currentText()
        # 注意：axis_title_fontsize 和 tick_label_fontsize 已删除（下面有实现方法）
        ps.line_width = self.line_width_spin.value()
        ps.line_style = self.line_style_combo.currentText()
        ps.tick_direction = self.tick_direction_combo.currentText()
        ps.tick_len_major = self.tick_len_major_spin.value()
        ps.tick_len_minor = self.tick_len_minor_spin.value()
        # 注意：tick_top/bottom/left/right 已删除（没用且不会自动更新）
        ps.tick_width = self.tick_width_spin.value()
        ps.show_grid = self.show_grid_check.isChecked()
        ps.grid_alpha = self.grid_alpha_spin.value()
        ps.shadow_alpha = self.shadow_alpha_spin.value()
        ps.spine_top = self.spine_top_check.isChecked()
        ps.spine_bottom = self.spine_bottom_check.isChecked()
        ps.spine_left = self.spine_left_check.isChecked()
        ps.spine_right = self.spine_right_check.isChecked()
        ps.spine_width = self.spine_width_spin.value()
        ps.show_legend = self.show_legend_check.isChecked()
        ps.legend_frame = self.legend_frame_check.isChecked()
        ps.legend_loc = self.legend_loc_combo.currentText()
        ps.legend_fontsize = self.legend_fontsize_spin.value()
        ps.legend_ncol = self.legend_column_spin.value()
        ps.legend_columnspacing = self.legend_columnspacing_spin.value()
        ps.legend_labelspacing = self.legend_labelspacing_spin.value()
        ps.legend_handlelength = self.legend_handlelength_spin.value()
        
        # 保存标题控制
        ps.xlabel_text = self.xlabel_input.text()
        ps.xlabel_show = self.xlabel_show_check.isChecked()
        ps.xlabel_fontsize = self.xlabel_font_spin.value()
        ps.xlabel_pad = self.xlabel_pad_spin.value()
        ps.ylabel_text = self.ylabel_input.text()
        ps.ylabel_show = self.ylabel_show_check.isChecked()
        ps.ylabel_fontsize = self.ylabel_font_spin.value()
        ps.ylabel_pad = self.ylabel_pad_spin.value()
        ps.title_text = self.title_input.text()
        ps.title_show = self.title_show_check.isChecked()
        ps.title_fontsize = self.title_font_spin.value()
        ps.title_pad = self.title_pad_spin.value()
        
        # 保存坐标轴显示控制
        ps.x_axis_invert = self.x_axis_invert_check.isChecked()
        ps.show_x_values = self.show_x_values_check.isChecked()
        ps.show_y_values = self.show_y_values_check.isChecked()
        
        self.config_manager.update_config(config)
    
    def get_config(self) -> PlotConfig:
        """获取当前配置"""
        self.save_config()
        return self.config_manager.get_config()
    
    def apply_preset(self, preset_name: str):
        """应用预设"""
        if preset_name == "默认":
            # 恢复默认值
            self.fig_width_spin.setValue(10.0)
            self.fig_height_spin.setValue(6.0)
            self.fig_dpi_spin.setValue(300)
            self.aspect_ratio_spin.setValue(0.6)
        elif preset_name == "Icarus 单栏":
            # 注意：fig_width 和 fig_height 已删除（没有实际作用）
            self.fig_dpi_spin.setValue(300)
            self.aspect_ratio_spin.setValue(2.6 / 3.4)
            self.font_family_combo.setCurrentText("Times New Roman")
            # 注意：axis_title_font_spin 和 tick_label_font_spin 已删除（下面有实现方法）
            self.legend_fontsize_spin.setValue(8)
            self.line_width_spin.setValue(1.0)
            self.tick_direction_combo.setCurrentText("in")
            self.tick_len_major_spin.setValue(6)
            self.tick_len_minor_spin.setValue(3)
            self.tick_width_spin.setValue(1.0)
            self.spine_width_spin.setValue(1.0)
        elif preset_name == "Icarus 双栏":
            # 注意：fig_width 和 fig_height 已删除（没有实际作用）
            self.fig_dpi_spin.setValue(300)
            self.aspect_ratio_spin.setValue(5.0 / 7.0)
            self.font_family_combo.setCurrentText("Times New Roman")
            # 注意：axis_title_font_spin 和 tick_label_font_spin 已删除（下面有实现方法）
            self.legend_fontsize_spin.setValue(10)
            self.line_width_spin.setValue(1.2)
            self.tick_direction_combo.setCurrentText("in")
            self.tick_len_major_spin.setValue(6)
            self.tick_len_minor_spin.setValue(3)
            self.tick_width_spin.setValue(1.0)
            self.spine_width_spin.setValue(1.0)
        
        self.save_config()
    
    def _load_custom_presets(self):
        """加载自定义预设"""
        from PyQt6.QtCore import QSettings
        settings = QSettings("GTLab", "SpectraPro_v4")
        custom_presets_json = settings.value("custom_style_presets", "{}")
        try:
            import json
            custom_presets = json.loads(custom_presets_json)
            self.style_preset_combo.clear()
            self.style_preset_combo.addItems(["默认", "Icarus 单栏", "Icarus 双栏"])
            self.style_preset_combo.addItems(sorted(custom_presets.keys()))
        except:
            self.style_preset_combo.clear()
            self.style_preset_combo.addItems(["默认", "Icarus 单栏", "Icarus 双栏"])
    
    def _manage_style_presets(self):
        """管理样式预设"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QMessageBox, QInputDialog
        from PyQt6.QtCore import QSettings
        import json
        
        dialog = QDialog(self)
        dialog.setWindowTitle("管理样式预设")
        dialog.resize(400, 300)
        layout = QVBoxLayout(dialog)
        
        preset_list = QListWidget()
        settings = QSettings("GTLab", "SpectraPro_v4")
        custom_presets_json = settings.value("custom_style_presets", "{}")
        try:
            custom_presets = json.loads(custom_presets_json)
        except:
            custom_presets = {}
        
        preset_list.addItems(sorted(custom_presets.keys()))
        layout.addWidget(preset_list)
        
        from PyQt6.QtWidgets import QPushButton, QHBoxLayout
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("保存当前")
        btn_save.clicked.connect(lambda: self._save_current_preset(dialog, preset_list, custom_presets, settings))
        btn_load = QPushButton("加载")
        btn_load.clicked.connect(lambda: self._load_preset(dialog, preset_list, custom_presets))
        btn_delete = QPushButton("删除")
        btn_delete.clicked.connect(lambda: self._delete_preset(dialog, preset_list, custom_presets, settings))
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dialog.accept)
        
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_delete)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        
        dialog.exec()
        self._load_custom_presets()
    
    def _save_current_preset(self, dialog, preset_list, custom_presets, settings):
        """保存当前预设"""
        from PyQt6.QtWidgets import QInputDialog, QMessageBox
        name, ok = QInputDialog.getText(dialog, "保存预设", "请输入预设名称:")
        if not ok or not name.strip():
            return
        
        name = name.strip()
        if name in custom_presets or name in ["默认", "Icarus 单栏", "Icarus 双栏"]:
            QMessageBox.warning(dialog, "错误", f"预设名称 '{name}' 已存在。")
            return
        
        # 保存当前配置
        config = self.get_config()
        custom_presets[name] = config.to_dict()
        settings.setValue("custom_style_presets", json.dumps(custom_presets))
        settings.sync()
        
        preset_list.addItem(name)
        QMessageBox.information(dialog, "成功", f"预设 '{name}' 已保存。")
    
    def _load_preset(self, dialog, preset_list, custom_presets):
        """加载预设"""
        from PyQt6.QtWidgets import QMessageBox
        selected = preset_list.currentItem()
        if not selected:
            QMessageBox.warning(dialog, "提示", "请先选择一个预设。")
            return
        
        name = selected.text()
        if name not in custom_presets:
            QMessageBox.warning(dialog, "错误", f"预设 '{name}' 不存在。")
            return
        
        # 加载预设配置
        config_dict = custom_presets[name]
        config = PlotConfig.from_dict(config_dict)
        self.config_manager.update_config(config)
        self.load_config()
        
        QMessageBox.information(dialog, "成功", f"预设 '{name}' 已加载。")
    
    def _delete_preset(self, dialog, preset_list, custom_presets, settings):
        """删除预设"""
        from PyQt6.QtWidgets import QMessageBox
        import json
        selected = preset_list.currentItem()
        if not selected:
            QMessageBox.warning(dialog, "提示", "请先选择一个预设。")
            return
        
        name = selected.text()
        if name not in custom_presets:
            QMessageBox.warning(dialog, "错误", f"预设 '{name}' 不存在。")
            return
        
        reply = QMessageBox.question(dialog, "确认", f"确定要删除预设 '{name}' 吗？")
        if reply == QMessageBox.StandardButton.Yes:
            del custom_presets[name]
            settings.setValue("custom_style_presets", json.dumps(custom_presets))
            settings.sync()
            preset_list.takeItem(preset_list.row(selected))
            QMessageBox.information(dialog, "成功", f"预设 '{name}' 已删除。")


"""Peak detection tab widget"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QLabel, QPushButton, QScrollArea, QTextEdit
)
from src.ui.widgets.custom_widgets import CollapsibleGroupBox


class PeakDetectionTab(QWidget):
    """峰值检测 Tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
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
        
        self.peak_height_spin = QDoubleSpinBox()
        self.peak_height_spin.setRange(-999999999.0, 999999999.0)
        self.peak_height_spin.setDecimals(15)
        self.peak_height_spin.setValue(0.0)
        self.peak_height_spin.setSpecialValueText("自动 (0.01% of max)")
        self.peak_height_spin.setToolTip("峰高阈值：0=自动(0.01%)，可设置为极小值(如-999999)或负数以检测所有峰值，包括负峰")
        self.peak_height_spin.setSingleStep(0.1)
        
        self.peak_distance_spin = QSpinBox()
        self.peak_distance_spin.setRange(0, 999999999)
        self.peak_distance_spin.setValue(0)
        self.peak_distance_spin.setSpecialValueText("自动 (0.1% of points)")
        self.peak_distance_spin.setToolTip("最小间距：0=自动(0.1%)，设置为1可检测所有相邻峰值，设置为更大值可过滤假峰")
        
        self.peak_prominence_spin = QDoubleSpinBox()
        self.peak_prominence_spin.setRange(0.0, 999999999.0)
        self.peak_prominence_spin.setDecimals(15)
        self.peak_prominence_spin.setValue(0.0)
        self.peak_prominence_spin.setSpecialValueText("禁用 (推荐)")
        
        self.peak_width_spin = QDoubleSpinBox()
        self.peak_width_spin.setRange(0.0, 999999999.0)
        self.peak_width_spin.setDecimals(15)
        self.peak_width_spin.setValue(0.0)
        self.peak_width_spin.setSpecialValueText("禁用 (推荐)")
        
        self.peak_wlen_spin = QSpinBox()
        self.peak_wlen_spin.setRange(0, 999999999)
        self.peak_wlen_spin.setValue(0)
        self.peak_wlen_spin.setSpecialValueText("禁用 (推荐)")
        
        self.peak_rel_height_spin = QDoubleSpinBox()
        self.peak_rel_height_spin.setRange(0.0, 999999999.0)
        self.peak_rel_height_spin.setDecimals(15)
        self.peak_rel_height_spin.setValue(0.0)
        self.peak_rel_height_spin.setSpecialValueText("禁用 (推荐)")
        
        peak_params_layout.addRow("峰高阈值 (height):", self.peak_height_spin)
        peak_params_layout.addRow("最小间距 (distance):", self.peak_distance_spin)
        peak_params_layout.addRow("突出度 (prominence):", self.peak_prominence_spin)
        peak_params_layout.addRow("最小宽度 (width):", self.peak_width_spin)
        peak_params_layout.addRow("窗口长度 (wlen):", self.peak_wlen_spin)
        peak_params_layout.addRow("相对高度 (rel_height):", self.peak_rel_height_spin)
        
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
        
        # 匹配线样式设置
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
    
    def _create_h_layout(self, widgets):
        """创建水平布局的辅助方法"""
        from PyQt6.QtWidgets import QWidget, QHBoxLayout
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(5)
        for wid in widgets:
            l.addWidget(wid)
        return w
    
    def _create_color_picker_button(self, color_input):
        """创建颜色选择器按钮"""
        from PyQt6.QtWidgets import QPushButton
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QColorDialog
        
        color_button = QPushButton("颜色")
        color_button.setFixedSize(30, 25)
        color_button.setToolTip("点击选择颜色")
        
        def update_button_color():
            color_str = color_input.text().strip()
            if color_str:
                try:
                    if color_str.startswith('#'):
                        qcolor = QColor(color_str)
                    else:
                        import matplotlib.colors as mcolors
                        rgba = mcolors.to_rgba(color_str)
                        qcolor = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                    color_button.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #999;")
                except:
                    color_button.setStyleSheet("background-color: #CCCCCC; border: 1px solid #999;")
            else:
                color_button.setStyleSheet("background-color: #CCCCCC; border: 1px solid #999;")
        
        update_button_color()
        color_input.textChanged.connect(update_button_color)
        
        def pick_color():
            color_str = color_input.text().strip()
            initial_color = QColor(128, 128, 128)
            if color_str:
                try:
                    if color_str.startswith('#'):
                        initial_color = QColor(color_str)
                    else:
                        import matplotlib.colors as mcolors
                        rgba = mcolors.to_rgba(color_str)
                        initial_color = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                except:
                    pass
            color = QColorDialog.getColor(initial_color, self, "选择颜色")
            if color.isValid():
                color_input.setText(color.name())
        
        color_button.clicked.connect(pick_color)
        return color_button
    
    def get_widgets_dict(self):
        """获取所有控件的字典，用于 ConfigBinder"""
        return {
            'peak_check': self.peak_check,
            'peak_height_spin': self.peak_height_spin,
            'peak_distance_spin': self.peak_distance_spin,
            'peak_prominence_spin': self.peak_prominence_spin,
            'peak_width_spin': self.peak_width_spin,
            'peak_wlen_spin': self.peak_wlen_spin,
            'peak_rel_height_spin': self.peak_rel_height_spin,
            'peak_marker_shape_combo': self.peak_marker_shape_combo,
            'peak_marker_size_spin': self.peak_marker_size_spin,
            'peak_marker_color_input': self.peak_marker_color_input,
            'peak_show_label_check': self.peak_show_label_check,
            'peak_label_font_combo': self.peak_label_font_combo,
            'peak_label_size_spin': self.peak_label_size_spin,
            'peak_label_color_input': self.peak_label_color_input,
            'peak_label_bold_check': self.peak_label_bold_check,
            'peak_label_rotation_spin': self.peak_label_rotation_spin,
            'vertical_lines_input': self.vertical_lines_input,
            'vertical_line_color_input': self.vertical_line_color_input,
            'vertical_line_width_spin': self.vertical_line_width_spin,
            'vertical_line_style_combo': self.vertical_line_style_combo,
            'vertical_line_alpha_spin': self.vertical_line_alpha_spin,
            'match_line_color_input': self.match_line_color_input,
            'match_line_width_spin': self.match_line_width_spin,
            'match_line_style_combo': self.match_line_style_combo,
            'match_line_alpha_spin': self.match_line_alpha_spin,
            'rruff_ref_lines_enabled_check': self.rruff_ref_lines_enabled_check,
        }


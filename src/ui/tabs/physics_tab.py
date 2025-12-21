"""Physics tab widget"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit, QCheckBox, QPushButton, QTextEdit
)
from src.ui.widgets.custom_widgets import CollapsibleGroupBox


class PhysicsTab(QWidget):
    """物理验证 Tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 散射尾部拟合
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
        
        fit_layout.addRow("拟合曲线数量:", self.fit_curve_count_spin)
        fit_layout.addRow("", self.btn_clear_fits)
        
        self.btn_run_fit = QPushButton("运行拟合并叠加到当前图")
        self.btn_run_fit.setStyleSheet("background-color: #555555; color: white; font-weight: bold;")
        fit_layout.addRow("", self.btn_run_fit)
        
        self.fit_output_text = QTextEdit()
        self.fit_output_text.setReadOnly(True)
        self.fit_output_text.setFixedHeight(150)
        fit_layout.addRow("拟合结果:", self.fit_output_text)
        
        fit_group.setContentLayout(fit_layout)
        layout.addWidget(fit_group)
        
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
            'fit_cutoff_spin': self.fit_cutoff_spin,
            'fit_model_combo': self.fit_model_combo,
            'fit_line_color_input': self.fit_line_color_input,
            'fit_line_style_combo': self.fit_line_style_combo,
            'fit_line_width_spin': self.fit_line_width_spin,
            'fit_marker_combo': self.fit_marker_combo,
            'fit_marker_size_spin': self.fit_marker_size_spin,
            'fit_legend_label_input': self.fit_legend_label_input,
            'fit_show_legend_check': self.fit_show_legend_check,
            'fit_curve_count_spin': self.fit_curve_count_spin,
        }


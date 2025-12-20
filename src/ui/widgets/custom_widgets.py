from decimal import Decimal, InvalidOperation
import sys

from PyQt6.QtCore import Qt, QLocale
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QGroupBox, QVBoxLayout, QWidget, QHBoxLayout,
    QToolButton, QLabel, QScrollArea, QFormLayout, QLineEdit, QSizePolicy
)
from PyQt6.QtGui import QValidator


class SmartDoubleSpinBox(QDoubleSpinBox):
    """自定义 QDoubleSpinBox，确保默认隐藏尾随零"""
    def textFromValue(self, value):
        # 使用 'g' 格式（通用格式），自动隐藏尾随零
        # 最多显示15位有效数字
        return f"{value:.15g}".rstrip('0').rstrip('.')


class UnlimitedNumericInput(QLineEdit):
    """
    无限制长度的数字输入框：
    - 允许任意长度的整数/小数（含负号、科学计数）
    - 通过 value() 取得 float/Decimal，兼容 SpinBox 常用接口
    """
    def __init__(self, parent=None, placeholder=None, default_value="0"):
        super().__init__(parent)
        # Qt 中 0 会阻止输入，这里使用一个极大值表示“无限”
        self.setMaxLength(1_000_000)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setReadOnly(False)
        self.setEnabled(True)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.setAcceptDrops(True)
        if placeholder:
            self.setPlaceholderText(placeholder)
        self.setText(str(default_value))

    def value(self, as_decimal: bool = False):
        text = self.text().strip()
        if not text:
            return Decimal(0) if as_decimal else 0.0
        # 尝试支持逗号作为小数点
        normalized = text.replace(",", ".")
        try:
            return Decimal(normalized) if as_decimal else float(normalized)
        except (InvalidOperation, ValueError):
            return Decimal(0) if as_decimal else 0.0

    # 兼容 QDoubleSpinBox 常用接口（作为空操作保持 API 一致）
    def setValue(self, v):
        self.setText(str(v))

    def setRange(self, *_):
        pass

    def setDecimals(self, *_):
        pass

    def setSingleStep(self, *_):
        pass

    def setKeyboardTracking(self, *_):
        pass


class UnlimitedDoubleSpinBox(QDoubleSpinBox):
    """
    基于 QDoubleSpinBox，放开长度/小数位限制，同时保留 SpinBox 的交互体验和信号。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-sys.float_info.max, sys.float_info.max)
        self.setDecimals(100)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        # 使用 C locale，确保小数点为 '.'
        self.setLocale(QLocale.c())
        # 确保行编辑可输入、可见
        le = self.lineEdit()
        if le:
            le.setReadOnly(False)
            le.setEnabled(True)
            le.setMaxLength(1_000_000)
            le.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            le.setStyleSheet("color: black; background: white; selection-background-color: #cce5ff;")

    # 始终接受输入（包含科学计数与逗号小数点）
    def validate(self, text, pos):
        # 返回 (state, text, pos) 以符合 QValidator 签名
        return (QValidator.State.Acceptable, text, pos)

    def valueFromText(self, text: str):
        normalized = text.strip().replace(",", ".")
        if not normalized:
            return 0.0
        try:
            return float(normalized)
        except Exception:
            return 0.0

    def textFromValue(self, value: float):
        """
        使用通用格式，自动隐藏尾随零，并处理科学计数法
        """
        if abs(value) < 1e-15 or abs(value) > 1e15:
            return f"{value:.15g}"
        return f"{value}"


class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title, parent=None, is_expanded=True):
        super().__init__(parent)
        
        self.main_vbox = QVBoxLayout(self)
        self.main_vbox.setContentsMargins(1, 1, 1, 1)
        self.main_vbox.setSpacing(0)

        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_layout.setContentsMargins(5, 5, 5, 5)

        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("""
            QToolButton { 
                border: none; 
                background: white; 
                color: #333333; 
                font-weight: bold;
                padding: 0;
            }
        """)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(is_expanded)
        self.toggle_button.setToolTip("点击折叠/展开")
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold; padding: 0;")
        
        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch(1)

        self.content_widget = QWidget()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setWidget(self.content_widget)

        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 5, 10, 10)
        self.content_layout.setSpacing(5)
        
        self.toggle_button.clicked.connect(self.toggle_content)
        
        self.main_vbox.addWidget(self.header_widget)
        self.main_vbox.addWidget(self.scroll_area)

        self.update_icon()
        self.scroll_area.setVisible(is_expanded)
        
    def toggle_content(self, checked):
        self.scroll_area.setVisible(checked)
        self.update_icon()

    def update_icon(self):
        if self.toggle_button.isChecked():
            self.toggle_button.setText("▼")
        else:
            self.toggle_button.setText("▶")
            
    def _clear_layout_recursively(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                elif item.layout() is not None:
                    self._clear_layout_recursively(item.layout())
                    
    def setContentLayout(self, layout):
        # 1. 清除现有内容
        self._clear_layout_recursively(self.content_layout)
        
        # 2. 将新的布局添加到 self.content_layout
        self.content_layout.addLayout(layout)
        
        # 3. 强制设置新布局的格式
        if isinstance(layout, QFormLayout):
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
        elif isinstance(layout, QVBoxLayout) or isinstance(layout, QHBoxLayout):
             layout.setContentsMargins(0, 0, 0, 0)
             # 修正：将错误的 'l' 变量改为 'layout'
             layout.setSpacing(5)  
             
        # 4. 添加拉伸项确保内容靠上
        self.content_layout.addStretch(1)


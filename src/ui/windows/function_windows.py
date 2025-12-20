"""
功能独立窗口 - 将各个功能标签页内容显示在独立窗口中
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QWidget, QScrollArea
)
from PyQt6.QtCore import Qt


class FunctionWindow(QDialog):
    """功能窗口基类"""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        self.resize(1000, 800)
        
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # 内容widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(10)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        
        scroll_area.setWidget(self.content_widget)
        main_layout.addWidget(scroll_area)
    
    def add_content(self, widget):
        """添加内容widget"""
        self.content_layout.addWidget(widget)


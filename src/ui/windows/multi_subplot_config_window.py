"""
多子图配置窗口
独立窗口，用于控制所有多子图窗口的样式、峰值检测、峰值匹配和谱线扫描
支持自动检测当前打开的多子图窗口
"""
from typing import Dict, List, Optional, Any
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QScrollArea, QWidget, QMessageBox,
    QSplitter
)

from src.ui.panels.multi_subplot_style_panel import MultiSubplotStylePanel
from src.ui.widgets.custom_widgets import CollapsibleGroupBox


class MultiSubplotConfigWindow(QDialog):
    """多子图配置窗口"""
    
    # 信号：配置改变时发出
    config_changed = pyqtSignal(int, int)  # (window_id, subplot_index)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("多子图配置")
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(800, 900)
        self.setMinimumSize(600, 500)
        
        self.parent_window = parent
        self.detected_windows: Dict[str, Any] = {}  # {window_id: window_object}
        self.current_window_id: Optional[str] = None
        self.multi_subplot_panel: Optional[MultiSubplotStylePanel] = None
        
        self.setup_ui()
        self.setup_auto_detection()
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("多子图配置控制")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 5px;")
        layout.addWidget(title_label)
        
        # 窗口选择区域
        window_select_group = QGroupBox("当前多子图窗口")
        window_select_layout = QVBoxLayout(window_select_group)
        
        # 窗口选择下拉框
        window_select_layout.addWidget(QLabel("选择窗口:"))
        self.window_combo = QComboBox()
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        window_select_layout.addWidget(self.window_combo)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新窗口列表")
        refresh_btn.clicked.connect(self.detect_multi_subplot_windows)
        window_select_layout.addWidget(refresh_btn)
        
        # 状态标签
        self.status_label = QLabel("状态: 未检测到多子图窗口")
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        window_select_layout.addWidget(self.status_label)
        
        layout.addWidget(window_select_group)
        
        # 多子图样式控制面板（使用滚动区域）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(600)
        
        self.multi_subplot_panel = MultiSubplotStylePanel(self)
        self.multi_subplot_panel.config_changed.connect(self._on_subplot_config_changed)
        
        scroll_area.setWidget(self.multi_subplot_panel)
        layout.addWidget(scroll_area)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        apply_btn = QPushButton("应用并更新绘图")
        apply_btn.setStyleSheet("font-size: 11pt; padding: 8px; background-color: #4CAF50; color: white; font-weight: bold;")
        apply_btn.clicked.connect(self.apply_and_update)
        button_layout.addWidget(apply_btn)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def setup_auto_detection(self):
        """设置自动检测"""
        # 定时检测多子图窗口
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.detect_multi_subplot_windows)
        self.detection_timer.start(2000)  # 每2秒检测一次
        
        # 立即检测一次
        self.detect_multi_subplot_windows()
    
    def detect_multi_subplot_windows(self):
        """检测当前打开的多子图窗口"""
        if not self.parent_window:
            return
        
        detected = {}
        
        # 检测NMF窗口
        if hasattr(self.parent_window, 'nmf_window') and self.parent_window.nmf_window:
            nmf_window = self.parent_window.nmf_window
            if nmf_window.isVisible():
                # 检查是否有多个子图
                fig = nmf_window.canvas.figure if hasattr(nmf_window, 'canvas') else None
                if fig and len(fig.axes) >= 2:
                    detected['nmf'] = {
                        'window': nmf_window,
                        'name': 'NMF分析窗口',
                        'type': 'nmf',
                        'subplots': len(fig.axes)
                    }
        
        # 检测残差拟合窗口
        if hasattr(self.parent_window, 'nmf_validation_window') and self.parent_window.nmf_validation_window:
            val_window = self.parent_window.nmf_validation_window
            if val_window.isVisible():
                fig = val_window.canvas.figure if hasattr(val_window, 'canvas') else None
                if fig and len(fig.axes) >= 2:
                    detected['nmf_validation'] = {
                        'window': val_window,
                        'name': 'NMF残差拟合窗口',
                        'type': 'nmf_validation',
                        'subplots': len(fig.axes)
                    }
        
        # 检测2D-COS窗口
        if hasattr(self.parent_window, 'two_dcos_window') and self.parent_window.two_dcos_window:
            cos_window = self.parent_window.two_dcos_window
            if cos_window.isVisible():
                fig = cos_window.canvas.figure if hasattr(cos_window, 'canvas') else None
                if fig and len(fig.axes) >= 2:
                    detected['2dcos'] = {
                        'window': cos_window,
                        'name': '2D-COS窗口',
                        'type': '2dcos',
                        'subplots': len(fig.axes)
                    }
        
        # 更新窗口列表
        self.detected_windows = detected
        
        # 更新下拉框
        self.window_combo.clear()
        if detected:
            for window_id, info in detected.items():
                self.window_combo.addItem(f"{info['name']} ({info['subplots']}个子图)", window_id)
            
            # 如果当前窗口不在列表中，选择第一个
            if self.current_window_id not in detected:
                self.current_window_id = list(detected.keys())[0]
                self.window_combo.setCurrentIndex(0)
            
            self.status_label.setText(f"状态: 检测到 {len(detected)} 个多子图窗口")
            self.status_label.setStyleSheet("color: #4CAF50; padding: 5px;")
        else:
            self.current_window_id = None
            self.status_label.setText("状态: 未检测到多子图窗口")
            self.status_label.setStyleSheet("color: #666; padding: 5px;")
            # 清空多子图面板
            self.multi_subplot_panel.subplot_controllers.clear()
            self.multi_subplot_panel.subplot_combo.clear()
    
    def _on_window_changed(self, text):
        """窗口改变时"""
        if not text:
            return
        
        window_id = self.window_combo.currentData()
        if not window_id or window_id not in self.detected_windows:
            return
        
        self.current_window_id = window_id
        info = self.detected_windows[window_id]
        window = info['window']
        
        # 注册子图
        self.multi_subplot_panel.subplot_controllers.clear()
        self.multi_subplot_panel.subplot_combo.clear()
        
        # 根据窗口类型注册子图
        if info['type'] == 'nmf':
            # 注册NMF的两个子图
            self.multi_subplot_panel.register_subplot(0, "NMF Components (Spectra)")
            self.multi_subplot_panel.register_subplot(1, "Concentration Weights")
            
            # 保存窗口引用到面板
            if not hasattr(self.multi_subplot_panel, 'current_window'):
                self.multi_subplot_panel.current_window = None
            if not hasattr(self.multi_subplot_panel, 'current_window_id'):
                self.multi_subplot_panel.current_window_id = None
            
            self.multi_subplot_panel.current_window = window
            self.multi_subplot_panel.current_window_id = window_id
            
            # 保存到NMF窗口，以便它知道多子图配置窗口存在
            if hasattr(window, '_multi_subplot_config_window'):
                window._multi_subplot_config_window = self
            else:
                setattr(window, '_multi_subplot_config_window', self)
                
        elif info['type'] == 'nmf_validation':
            self.multi_subplot_panel.register_subplot(0, "主图")
            self.multi_subplot_panel.register_subplot(1, "残差图")
            self.multi_subplot_panel.current_window = window
            self.multi_subplot_panel.current_window_id = window_id
        elif info['type'] == '2dcos':
            # 2D-COS可能有多个子图，需要动态检测
            fig = window.canvas.figure if hasattr(window, 'canvas') else None
            if fig:
                for i, ax in enumerate(fig.axes):
                    self.multi_subplot_panel.register_subplot(i, f"子图 {i+1}")
            self.multi_subplot_panel.current_window = window
            self.multi_subplot_panel.current_window_id = window_id
    
    def _on_subplot_config_changed(self, subplot_index: int):
        """子图配置改变时"""
        if self.current_window_id:
            self.config_changed.emit(hash(self.current_window_id), subplot_index)
            # 自动更新绘图
            self.apply_and_update()
    
    def apply_and_update(self):
        """应用配置并更新绘图"""
        if not self.current_window_id or self.current_window_id not in self.detected_windows:
            QMessageBox.information(self, "提示", "请先选择一个多子图窗口")
            return
        
        info = self.detected_windows[self.current_window_id]
        window = info['window']
        
        # 根据窗口类型更新绘图
        if info['type'] == 'nmf':
            # NMF窗口需要重新调用plot_results
            if hasattr(window, 'plot_results') and hasattr(window, 'style_params'):
                window.plot_results(window.style_params)
        elif info['type'] == 'nmf_validation':
            # 残差拟合窗口
            if hasattr(window, 'update_plot'):
                window.update_plot()
        elif info['type'] == '2dcos':
            # 2D-COS窗口
            if hasattr(window, 'update_plot'):
                window.update_plot()
        
        QMessageBox.information(self, "完成", "配置已应用并更新绘图")
    
    def closeEvent(self, event):
        """窗口关闭时停止定时器"""
        if hasattr(self, 'detection_timer'):
            self.detection_timer.stop()
        super().closeEvent(event)


"""
样式与匹配窗口
包含出版质量样式控制、峰值匹配、谱线扫描等通用设置
新增：左侧列用于选择当前窗口和子图
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QScrollArea,
    QLabel, QComboBox, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from typing import Optional, Dict, Any

from src.ui.panels.publication_style_panel import PublicationStylePanel
from src.ui.panels.peak_matching_panel import PeakMatchingPanel
from src.ui.panels.peak_detection_panel import PeakDetectionPanel
from src.ui.panels.spectrum_scan_panel import SpectrumScanPanel


class StyleMatchingWindow(QDialog):
    """样式与匹配配置窗口"""
    
    # 信号：当前窗口/子图改变时发出
    target_changed = pyqtSignal(str, int)  # (window_id, subplot_index)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("样式与匹配设置")
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(1600, 900)  # 增加宽度以容纳左侧列
        self.setMinimumSize(1200, 600)
        
        self.current_window_id: Optional[str] = None
        self.current_subplot_index: int = 0
        self.detected_windows: Dict[str, Any] = {}  # {window_id: window_object}
        
        self.setup_ui()
        self.setup_auto_detection()
    
    def setup_ui(self):
        """设置UI - 新增左侧列用于选择当前窗口和子图"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # 创建四列布局（使用QSplitter实现可调整）
        from PyQt6.QtWidgets import QSplitter
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 最左列：窗口和子图选择
        target_widget = QWidget()
        target_layout = QVBoxLayout(target_widget)
        target_layout.setContentsMargins(5, 5, 5, 5)
        target_layout.setSpacing(10)
        
        target_group = QGroupBox("目标窗口/子图")
        target_group_layout = QVBoxLayout(target_group)
        target_group_layout.setSpacing(8)
        
        # 窗口选择
        target_group_layout.addWidget(QLabel("选择窗口:"))
        self.window_combo = QComboBox()
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        target_group_layout.addWidget(self.window_combo)
        
        # 子图选择
        target_group_layout.addWidget(QLabel("选择子图:"))
        self.subplot_combo = QComboBox()
        self.subplot_combo.addItems(["所有子图", "子图 0", "子图 1"])
        self.subplot_combo.currentIndexChanged.connect(self._on_subplot_changed)
        target_group_layout.addWidget(self.subplot_combo)
        
        # 总布局选项（应用到所有子图）
        self.apply_to_all_subplots_check = QCheckBox("应用到所有子图")
        self.apply_to_all_subplots_check.setChecked(False)
        self.apply_to_all_subplots_check.setToolTip("勾选后，样式设置将应用到当前窗口的所有子图")
        target_group_layout.addWidget(self.apply_to_all_subplots_check)
        
        # 刷新按钮
        refresh_btn = QPushButton("🔄 刷新窗口列表")
        refresh_btn.clicked.connect(self.detect_plot_windows)
        target_group_layout.addWidget(refresh_btn)
        
        # 状态标签
        self.status_label = QLabel("状态: 未检测到窗口")
        self.status_label.setStyleSheet("color: #666; padding: 5px; font-size: 9pt;")
        target_group_layout.addWidget(self.status_label)
        
        target_group.setLayout(target_group_layout)
        target_layout.addWidget(target_group)
        target_layout.addStretch()
        
        splitter.addWidget(target_widget)
        
        # 第二列：出版质量样式面板
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.publication_style_panel = PublicationStylePanel(self)
        left_layout.addWidget(self.publication_style_panel)
        splitter.addWidget(left_widget)
        
        # 第三列：波峰检测与垂直参考线面板
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        self.peak_detection_panel = PeakDetectionPanel(self)
        middle_layout.addWidget(self.peak_detection_panel)
        splitter.addWidget(middle_widget)
        
        # 第四列：峰值匹配面板（紧凑）
        matching_widget = QWidget()
        matching_layout = QVBoxLayout(matching_widget)
        matching_layout.setContentsMargins(0, 0, 0, 0)
        self.peak_matching_panel = PeakMatchingPanel(self)
        matching_layout.addWidget(self.peak_matching_panel)
        splitter.addWidget(matching_widget)
        
        # 最右列：谱线扫描面板（给更多空间）
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.spectrum_scan_panel = SpectrumScanPanel(self)
        right_layout.addWidget(self.spectrum_scan_panel)
        splitter.addWidget(right_widget)
        
        # 设置列宽比例：目标:样式:波峰检测:匹配:扫描 = 1:2:2:1.5:2
        splitter.setSizes([200, 400, 350, 250, 500])
        
        main_layout.addWidget(splitter)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)
        button_layout.addWidget(self.btn_close)
        
        main_layout.addLayout(button_layout)
        
        # 连接信号
        self.publication_style_panel.config_changed.connect(self._on_config_changed)
        self.peak_detection_panel.config_changed.connect(self._on_config_changed)
        self.peak_matching_panel.config_changed.connect(self._on_config_changed)
        self.spectrum_scan_panel.config_changed.connect(self._on_config_changed)
    
    def _on_config_changed(self):
        """配置改变时，通知主窗口更新"""
        # 如果"应用到所有子图"被勾选，应用样式到所有子图
        if self.apply_to_all_subplots_check.isChecked() and self.current_window_id:
            self._apply_style_to_all_subplots()
        
        if self.parent():
            if hasattr(self.parent(), '_on_style_param_changed'):
                self.parent()._on_style_param_changed()
    
    def _apply_style_to_all_subplots(self):
        """应用样式到当前窗口的所有子图"""
        if not self.current_window_id or self.current_window_id not in self.detected_windows:
            return
        
        window_info = self.detected_windows[self.current_window_id]
        window = window_info['window']
        
        # 获取当前配置
        config = self.publication_style_panel.get_config()
        ps = config.publication_style
        
        # 根据窗口类型应用样式
        if window_info['type'] == 'NMFResultWindow':
            # NMF窗口有两个子图（ax1和ax2）
            if hasattr(window, 'ax1') and window.ax1:
                self._apply_publication_style_to_axes(window.ax1, ps)
            if hasattr(window, 'ax2') and window.ax2:
                self._apply_publication_style_to_axes(window.ax2, ps)
            # 重绘
            if hasattr(window, 'canvas'):
                window.canvas.draw()
        elif window_info['type'] == 'MplPlotWindow':
            # 普通绘图窗口只有一个axes
            if hasattr(window, 'canvas') and hasattr(window.canvas, 'axes'):
                self._apply_publication_style_to_axes(window.canvas.axes, ps)
                window.canvas.draw()
    
    def _apply_publication_style_to_axes(self, ax, ps):
        """应用出版质量样式到指定的axes"""
        import matplotlib.pyplot as plt
        
        # 设置字体
        font_family = ps.font_family
        current_font = 'Times New Roman' if font_family == 'Times New Roman' else font_family
        
        # 设置刻度显示控制
        ax.tick_params(axis='both', which='major',
                      direction=ps.tick_direction,
                      length=ps.tick_len_major,
                      width=ps.tick_width,
                      labelsize=ps.tick_label_fontsize,
                      top=ps.tick_top,
                      bottom=ps.tick_bottom,
                      left=ps.tick_left,
                      right=ps.tick_right,
                      labeltop=ps.show_top_xaxis,
                      labelbottom=ps.show_bottom_xaxis,
                      labelleft=ps.show_left_yaxis,
                      labelright=ps.show_right_yaxis)
        ax.tick_params(axis='both', which='minor',
                      direction=ps.tick_direction,
                      length=ps.tick_len_minor,
                      width=ps.tick_width,
                      top=ps.tick_top,
                      bottom=ps.tick_bottom,
                      left=ps.tick_left,
                      right=ps.tick_right)
        
        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(current_font)
        
        # 设置坐标轴标签字体
        if ax.xaxis.label:
            ax.xaxis.label.set_fontfamily(current_font)
            ax.xaxis.label.set_fontsize(ps.xlabel_fontsize)
        if ax.yaxis.label:
            ax.yaxis.label.set_fontfamily(current_font)
            ax.yaxis.label.set_fontsize(ps.ylabel_fontsize)
        if ax.title:
            ax.title.set_fontfamily(current_font)
            ax.title.set_fontsize(ps.title_fontsize)
        
        # 设置边框
        ax.spines['top'].set_visible(ps.spine_top)
        ax.spines['bottom'].set_visible(ps.spine_bottom)
        ax.spines['left'].set_visible(ps.spine_left)
        ax.spines['right'].set_visible(ps.spine_right)
        for spine in ax.spines.values():
            spine.set_linewidth(ps.spine_width)
        
        # 设置网格
        if ps.show_grid:
            ax.grid(True, alpha=ps.grid_alpha)
        else:
            ax.grid(False)
    
    def get_publication_style_panel(self):
        """获取出版质量样式面板"""
        return self.publication_style_panel
    
    def get_peak_detection_panel(self):
        """获取波峰检测面板"""
        return self.peak_detection_panel
    
    def get_peak_matching_panel(self):
        """获取峰值匹配面板"""
        return self.peak_matching_panel
    
    def get_spectrum_scan_panel(self):
        """获取谱线扫描面板"""
        return self.spectrum_scan_panel
    
    def setup_auto_detection(self):
        """设置自动检测定时器"""
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.detect_plot_windows)
        self.detection_timer.start(2000)  # 每2秒检测一次
        # 立即执行一次检测
        self.detect_plot_windows()
    
    def detect_plot_windows(self):
        """检测所有打开的绘图窗口"""
        if not self.parent():
            return
        
        parent = self.parent()
        detected = {}
        
        # 检测普通绘图窗口（MplPlotWindow）
        if hasattr(parent, 'plot_windows'):
            for group_name, window in parent.plot_windows.items():
                if window and window.isVisible():
                    window_id = f"MplPlotWindow_{group_name}"
                    detected[window_id] = {
                        'window': window,
                        'type': 'MplPlotWindow',
                        'name': f"光谱图 - {group_name}",
                        'has_subplots': False
                    }
        
        # 检测NMF窗口（NMFResultWindow）
        if hasattr(parent, 'nmf_window') and parent.nmf_window and parent.nmf_window.isVisible():
            window_id = "NMFResultWindow"
            detected[window_id] = {
                'window': parent.nmf_window,
                'type': 'NMFResultWindow',
                'name': "NMF 分析结果",
                'has_subplots': True,
                'subplot_count': 2
            }
        
        # 检测2D-COS窗口
        if hasattr(parent, 'cos_window') and parent.cos_window and parent.cos_window.isVisible():
            window_id = "TwoDCOSWindow"
            detected[window_id] = {
                'window': parent.cos_window,
                'type': 'TwoDCOSWindow',
                'name': "2D-COS 分析",
                'has_subplots': False
            }
        
        # 更新窗口列表
        self.detected_windows = detected
        
        # 更新下拉框
        current_text = self.window_combo.currentText()
        self.window_combo.clear()
        
        if detected:
            for window_id, info in detected.items():
                self.window_combo.addItem(info['name'], window_id)
            
            # 恢复之前的选择（如果还存在）
            index = self.window_combo.findText(current_text)
            if index >= 0:
                self.window_combo.setCurrentIndex(index)
            else:
                # 选择第一个窗口
                self.window_combo.setCurrentIndex(0)
                self._on_window_changed()
            
            self.status_label.setText(f"状态: 检测到 {len(detected)} 个窗口")
        else:
            self.status_label.setText("状态: 未检测到窗口")
            self.current_window_id = None
            self.current_subplot_index = 0
    
    def _on_window_changed(self):
        """窗口选择改变时"""
        window_id = self.window_combo.currentData()
        if not window_id:
            return
        
        self.current_window_id = window_id
        
        # 更新子图选择
        if window_id in self.detected_windows:
            info = self.detected_windows[window_id]
            if info.get('has_subplots', False):
                subplot_count = info.get('subplot_count', 2)
                self.subplot_combo.clear()
                for i in range(subplot_count):
                    self.subplot_combo.addItem(f"子图 {i}", i)
                self.subplot_combo.setEnabled(True)
            else:
                self.subplot_combo.clear()
                self.subplot_combo.addItem("无子图", 0)
                self.subplot_combo.setEnabled(False)
                self.current_subplot_index = 0
        
        self._on_subplot_changed()
    
    def _on_subplot_changed(self):
        """子图选择改变时"""
        subplot_index = self.subplot_combo.currentData()
        if subplot_index is not None:
            self.current_subplot_index = subplot_index
        
        # 发出信号
        if self.current_window_id:
            self.target_changed.emit(self.current_window_id, self.current_subplot_index)
    
    def get_current_target(self):
        """获取当前目标窗口和子图"""
        return self.current_window_id, self.current_subplot_index


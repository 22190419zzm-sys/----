"""
多子图样式控制面板
支持对多子图窗口（如NMF、残差拟合图等）的每个子图进行独立样式控制
包括样式设置、峰值检测、峰值匹配、谱线扫描等功能
"""
from typing import Dict, List, Optional, Any
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QLabel, QGroupBox, QScrollArea,
    QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit
)
from PyQt6.QtCore import pyqtSignal, Qt

from src.ui.widgets.custom_widgets import CollapsibleGroupBox
from src.core.plot_config_manager import PlotConfigManager, PlotConfig
from src.ui.panels.publication_style_panel import PublicationStylePanel
from src.ui.panels.peak_matching_panel import PeakMatchingPanel
from src.ui.panels.spectrum_scan_panel import SpectrumScanPanel


class SubplotStyleController:
    """单个子图的样式控制器"""
    
    def __init__(self, subplot_index: int, subplot_name: str, parent_panel):
        self.subplot_index = subplot_index
        self.subplot_name = subplot_name
        self.parent_panel = parent_panel
        
        # 样式配置（使用独立的配置管理器）
        self.style_config_manager = PlotConfigManager()
        self.style_config = self.style_config_manager.get_config()
        
        # 峰值匹配配置
        self.peak_matching_config = None
        
        # 谱线扫描配置
        self.spectrum_scan_config = None
        
        # 当前扫描的谱线数据
        self.scanned_spectra = []
    
    def get_style_config(self) -> PlotConfig:
        """获取样式配置"""
        return self.style_config
    
    def apply_style_to_axes(self, ax):
        """应用样式到axes"""
        from src.core.style_applier import apply_publication_style_to_axes
        apply_publication_style_to_axes(ax, self.style_config)
    
    def scan_current_subplot(self, plot_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """扫描当前子图的谱线"""
        from src.core.spectrum_scanner import SpectrumScanner
        
        scanner = SpectrumScanner()
        scanned = scanner.scan_last_plot(plot_data)
        self.scanned_spectra = scanned
        
        return scanned
    
    def get_scanned_spectra(self) -> List[Dict[str, Any]]:
        """获取扫描的谱线"""
        return self.scanned_spectra


class MultiSubplotStylePanel(QWidget):
    """多子图样式控制面板"""
    
    # 信号：配置改变时发出
    config_changed = pyqtSignal(int)  # 发出子图索引
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.subplot_controllers: Dict[int, SubplotStyleController] = {}
        self.current_subplot_index: Optional[int] = None
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 子图选择
        subplot_select_group = QGroupBox("子图选择")
        subplot_select_layout = QVBoxLayout(subplot_select_group)
        
        self.subplot_combo = QComboBox()
        self.subplot_combo.currentIndexChanged.connect(self._on_subplot_changed)
        subplot_select_layout.addWidget(QLabel("选择子图:"))
        subplot_select_layout.addWidget(self.subplot_combo)
        
        btn_scan_subplot = QPushButton("扫描当前子图")
        btn_scan_subplot.clicked.connect(self._scan_current_subplot)
        subplot_select_layout.addWidget(btn_scan_subplot)
        
        layout.addWidget(subplot_select_group)
        
        # 样式控制面板（复用PublicationStylePanel）
        self.style_panel = PublicationStylePanel(self)
        self.style_panel.config_changed.connect(self._on_style_changed)
        
        style_group = CollapsibleGroupBox("样式控制", is_expanded=True)
        style_group_content = QVBoxLayout()
        style_group_content.addWidget(self.style_panel)
        style_group.setContentLayout(style_group_content)
        layout.addWidget(style_group)
        
        # 峰值匹配面板（复用PeakMatchingPanel）
        self.peak_matching_panel = PeakMatchingPanel(self)
        self.peak_matching_panel.config_changed.connect(self._on_peak_matching_changed)
        
        peak_group = CollapsibleGroupBox("峰值匹配", is_expanded=True)
        peak_group_content = QVBoxLayout()
        peak_group_content.addWidget(self.peak_matching_panel)
        peak_group.setContentLayout(peak_group_content)
        layout.addWidget(peak_group)
        
        # 谱线扫描面板（复用SpectrumScanPanel）
        self.spectrum_scan_panel = SpectrumScanPanel(self)
        self.spectrum_scan_panel.config_changed.connect(self._on_spectrum_scan_changed)
        
        scan_group = CollapsibleGroupBox("谱线扫描", is_expanded=True)
        scan_group_content = QVBoxLayout()
        scan_group_content.addWidget(self.spectrum_scan_panel)
        scan_group.setContentLayout(scan_group_content)
        layout.addWidget(scan_group)
        
        layout.addStretch()
    
    def register_subplot(self, subplot_index: int, subplot_name: str):
        """注册子图"""
        if subplot_index not in self.subplot_controllers:
            controller = SubplotStyleController(subplot_index, subplot_name, self)
            self.subplot_controllers[subplot_index] = controller
            
            # 添加到下拉框
            self.subplot_combo.addItem(f"{subplot_index + 1}: {subplot_name}", subplot_index)
            
            # 如果是第一个，设置为当前
            if self.current_subplot_index is None:
                self.current_subplot_index = subplot_index
                self.subplot_combo.setCurrentIndex(0)
    
    def _on_subplot_changed(self, index: int):
        """子图改变时"""
        if index < 0:
            return
        
        subplot_index = self.subplot_combo.itemData(index)
        if subplot_index is None:
            return
        
        self.current_subplot_index = subplot_index
        
        # 加载当前子图的配置
        if subplot_index in self.subplot_controllers:
            controller = self.subplot_controllers[subplot_index]
            # 更新样式面板的配置管理器
            self.style_panel.config_manager = controller.style_config_manager
            self.style_panel.load_config()
            
            # 更新峰值匹配面板的配置管理器
            self.peak_matching_panel.config_manager = controller.style_config_manager
            self.peak_matching_panel.load_config()
            
            # 更新谱线扫描面板的配置管理器
            self.spectrum_scan_panel.config_manager = controller.style_config_manager
            self.spectrum_scan_panel.load_config()
    
    def _on_style_changed(self):
        """样式改变时"""
        if self.current_subplot_index is not None:
            # 保存当前子图的样式配置
            if self.current_subplot_index in self.subplot_controllers:
                controller = self.subplot_controllers[self.current_subplot_index]
                controller.style_config = self.style_panel.get_config()
                # 更新控制器的配置管理器
                controller.style_config_manager.save_config()
            
            # 发出信号
            self.config_changed.emit(self.current_subplot_index)
    
    def _on_peak_matching_changed(self):
        """峰值匹配改变时"""
        if self.current_subplot_index is not None:
            if self.current_subplot_index in self.subplot_controllers:
                controller = self.subplot_controllers[self.current_subplot_index]
                config = self.peak_matching_panel.get_config()
                controller.peak_matching_config = config.peak_matching
                # 更新控制器的配置管理器
                controller.style_config_manager.save_config()
            
            self.config_changed.emit(self.current_subplot_index)
    
    def _on_spectrum_scan_changed(self):
        """谱线扫描改变时"""
        if self.current_subplot_index is not None:
            if self.current_subplot_index in self.subplot_controllers:
                controller = self.subplot_controllers[self.current_subplot_index]
                config = self.spectrum_scan_panel.get_config()
                controller.spectrum_scan_config = config.spectrum_scan
                # 更新控制器的配置管理器
                controller.style_config_manager.save_config()
            
            self.config_changed.emit(self.current_subplot_index)
    
    def _scan_current_subplot(self):
        """扫描当前子图"""
        if self.current_subplot_index is None:
            return
        
        # 发出扫描信号，由窗口处理
        self.config_changed.emit(self.current_subplot_index)
    
    def get_current_controller(self) -> Optional[SubplotStyleController]:
        """获取当前子图控制器"""
        if self.current_subplot_index is not None:
            return self.subplot_controllers.get(self.current_subplot_index)
        return None
    
    def get_controller(self, subplot_index: int) -> Optional[SubplotStyleController]:
        """获取指定子图控制器"""
        return self.subplot_controllers.get(subplot_index)
    
    def apply_style_to_subplot(self, subplot_index: int, ax):
        """应用样式到指定子图"""
        controller = self.get_controller(subplot_index)
        if controller:
            controller.apply_style_to_axes(ax)


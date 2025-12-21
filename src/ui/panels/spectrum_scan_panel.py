"""
谱线扫描与堆叠偏移面板
支持扫描最后一次绘图的所有谱线，并可微调每根线的距离和指定匹配关系
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit
)
from PyQt6.QtCore import pyqtSignal

from src.ui.widgets.custom_widgets import CollapsibleGroupBox
from src.core.plot_config_manager import PlotConfigManager
from src.core.spectrum_scanner import SpectrumScanner


class SpectrumScanPanel(QWidget):
    """谱线扫描与堆叠偏移面板"""
    
    # 信号：配置改变时发出
    config_changed = pyqtSignal()
    scan_requested = pyqtSignal()  # 请求扫描最后一次绘图
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_manager = PlotConfigManager()
        self.spectrum_scanner = SpectrumScanner()
        self.setup_ui()
        self.load_config()
        self.connect_signals()
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 谱线扫描组（优化布局，给更多空间）
        scan_group = CollapsibleGroupBox("📊 谱线扫描与堆叠偏移", is_expanded=True)
        scan_layout = QVBoxLayout()
        scan_layout.setSpacing(10)
        
        # 扫描按钮（更紧凑）
        self.scan_button = QPushButton("扫描最后一次绘图的所有谱线")
        self.scan_button.setStyleSheet("font-size: 11pt; padding: 6px; background-color: #4CAF50; color: white; font-weight: bold;")
        self.scan_button.clicked.connect(self._on_scan_clicked)
        scan_layout.addWidget(self.scan_button)
        
        # 扫描状态（更紧凑）
        self.scan_status_label = QLabel("状态: 未扫描")
        self.scan_status_label.setStyleSheet("font-size: 9pt; color: #666;")
        scan_layout.addWidget(self.scan_status_label)
        
        # 扫描到的谱线列表（给更多空间）
        scan_layout.addWidget(QLabel("扫描到的谱线/图例:"))
        self.spectrum_list = QListWidget()
        self.spectrum_list.setMinimumHeight(150)
        self.spectrum_list.setMaximumHeight(250)
        self.spectrum_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.spectrum_list.itemSelectionChanged.connect(self._on_spectrum_selected)
        scan_layout.addWidget(self.spectrum_list)
        
        # 选中谱线的编辑控件
        edit_group = CollapsibleGroupBox("编辑选中谱线", is_expanded=False)
        edit_layout = QFormLayout()
        
        # 图例名称编辑
        self.legend_edit_input = QLineEdit()
        self.legend_edit_input.setPlaceholderText("图例名称")
        self.legend_edit_input.textChanged.connect(self._on_legend_changed)
        edit_layout.addRow("图例名称:", self.legend_edit_input)
        
        # 颜色选择
        from PyQt6.QtWidgets import QColorDialog
        self.color_edit_input = QLineEdit()
        self.color_edit_input.setPlaceholderText("例如: #FF0000 或 red")
        self.color_edit_input.textChanged.connect(self._on_color_changed)
        self.color_picker_btn = QPushButton("选择颜色")
        self.color_picker_btn.clicked.connect(self._pick_color)
        color_layout = QHBoxLayout()
        color_layout.addWidget(self.color_edit_input)
        color_layout.addWidget(self.color_picker_btn)
        edit_layout.addRow("颜色:", color_layout)
        
        edit_group.setContentLayout(edit_layout)
        scan_layout.addWidget(edit_group)
        
        # 堆叠偏移设置（紧凑布局）
        stack_layout = QFormLayout()
        stack_layout.setSpacing(8)
        
        # 堆叠偏移和启用复选框同一行
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("堆叠偏移:"))
        self.stack_offset_spin = QDoubleSpinBox()
        self.stack_offset_spin.setRange(-999999999.0, 999999999.0)
        self.stack_offset_spin.setDecimals(15)
        self.stack_offset_spin.setValue(0.5)
        self.stack_offset_spin.setMaximumWidth(120)
        self.stack_offset_spin.setToolTip("每个谱线按索引递增的偏移值")
        offset_layout.addWidget(self.stack_offset_spin)
        offset_layout.addStretch()
        
        self.scan_enabled_check = QCheckBox("启用谱线扫描")
        self.scan_enabled_check.setChecked(False)
        offset_layout.addWidget(self.scan_enabled_check)
        
        stack_layout.addRow(offset_layout)
        scan_layout.addLayout(stack_layout)
        
        # 谱线匹配映射
        mapping_group = CollapsibleGroupBox("谱线匹配映射", is_expanded=False)
        mapping_layout = QVBoxLayout()
        
        # 匹配映射表格
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(3)
        self.mapping_table.setHorizontalHeaderLabels(["源谱线", "目标谱线", "操作"])
        self.mapping_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.mapping_table.setMaximumHeight(200)
        mapping_layout.addWidget(self.mapping_table)
        
        # 添加/删除映射按钮
        mapping_btn_layout = QHBoxLayout()
        self.add_mapping_btn = QPushButton("添加映射")
        self.add_mapping_btn.clicked.connect(self._add_mapping)
        self.remove_mapping_btn = QPushButton("删除选中")
        self.remove_mapping_btn.clicked.connect(self._remove_mapping)
        self.clear_mapping_btn = QPushButton("清除所有")
        self.clear_mapping_btn.clicked.connect(self._clear_mappings)
        mapping_btn_layout.addWidget(self.add_mapping_btn)
        mapping_btn_layout.addWidget(self.remove_mapping_btn)
        mapping_btn_layout.addWidget(self.clear_mapping_btn)
        mapping_btn_layout.addStretch()
        mapping_layout.addLayout(mapping_btn_layout)
        
        mapping_group.setContentLayout(mapping_layout)
        scan_layout.addWidget(mapping_group)
        
        # 独立偏移设置（针对每个谱线）
        offset_group = CollapsibleGroupBox("独立偏移设置", is_expanded=False)
        offset_layout = QVBoxLayout()
        
        self.offset_table = QTableWidget()
        self.offset_table.setColumnCount(2)
        self.offset_table.setHorizontalHeaderLabels(["谱线标签", "偏移值"])
        self.offset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.offset_table.setMaximumHeight(200)
        offset_layout.addWidget(self.offset_table)
        
        offset_btn_layout = QHBoxLayout()
        self.apply_offsets_btn = QPushButton("应用偏移")
        self.apply_offsets_btn.clicked.connect(self._apply_offsets)
        offset_btn_layout.addWidget(self.apply_offsets_btn)
        offset_btn_layout.addStretch()
        offset_layout.addLayout(offset_btn_layout)
        
        offset_group.setContentLayout(offset_layout)
        scan_layout.addWidget(offset_group)
        
        scan_group.setContentLayout(scan_layout)
        layout.addWidget(scan_group)
    
    def connect_signals(self):
        """连接信号"""
        self.stack_offset_spin.valueChanged.connect(self._on_stack_offset_changed)
        self.scan_enabled_check.stateChanged.connect(self._on_config_changed)
    
    def _on_config_changed(self):
        """配置改变时"""
        self.save_config()
        self.config_changed.emit()
        # 通知主窗口更新绘图（如果颜色或图例改变）
        if self.parent():
            parent = self.parent()
            if hasattr(parent, '_on_style_param_changed'):
                parent._on_style_param_changed()
    
    def _on_stack_offset_changed(self):
        """堆叠偏移改变时"""
        # 应用堆叠偏移到扫描器
        if self.spectrum_scanner.scanned_spectra:
            self.spectrum_scanner.set_stack_offset(self.stack_offset_spin.value())
            # 更新偏移表格
            self._update_offset_table(self.spectrum_scanner.scanned_spectra)
        self._on_config_changed()
    
    def _on_scan_clicked(self):
        """扫描按钮点击"""
        self.scan_requested.emit()
    
    def scan_last_plot(self, plot_data):
        """
        扫描最后一次绘图的所有谱线
        
        Args:
            plot_data: 绘图数据列表，每个元素包含 {'x': x_data, 'y': y_data, 'label': label, 'color': color, ...}
        """
        if not plot_data:
            QMessageBox.warning(self, "警告", "没有可扫描的绘图数据")
            return
        
        # 使用扫描器扫描
        scanned = self.spectrum_scanner.scan_last_plot(plot_data)
        
        # 更新UI
        self.spectrum_list.clear()
        for i, spec in enumerate(scanned):
            item = QListWidgetItem(f"{i}: {spec.get('label', f'Spectrum {i}')}")
            item.setData(256, i)  # 存储索引
            self.spectrum_list.addItem(item)
        
        # 更新偏移表格
        self._update_offset_table(scanned)
        
        self.scan_status_label.setText(f"状态: 已扫描 {len(scanned)} 条谱线")
        self.scan_status_label.setStyleSheet("color: green;")
    
    def _update_offset_table(self, scanned_spectra):
        """更新偏移表格"""
        self.offset_table.setRowCount(len(scanned_spectra))
        for i, spec in enumerate(scanned_spectra):
            label = spec.get('label', f'Spectrum {i}')
            offset = spec.get('offset', 0.0)
            
            # 标签
            label_item = QTableWidgetItem(label)
            # PyQt6 中使用 ItemFlag
            from PyQt6.QtCore import Qt
            label_item.setFlags(label_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # 只读
            self.offset_table.setItem(i, 0, label_item)
            
            # 偏移值
            offset_item = QTableWidgetItem(str(offset))
            self.offset_table.setItem(i, 1, offset_item)
    
    def _add_mapping(self):
        """添加映射"""
        selected_items = self.spectrum_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "警告", "请至少选择2条谱线（第一条作为源，第二条作为目标）")
            return
        
        source_idx = selected_items[0].data(256)
        target_idx = selected_items[1].data(256)
        
        # 添加到表格
        row = self.mapping_table.rowCount()
        self.mapping_table.insertRow(row)
        
        source_item = QTableWidgetItem(f"Spectrum {source_idx}")
        source_item.setData(256, source_idx)
        # PyQt6 中使用 ItemFlag
        from PyQt6.QtCore import Qt
        source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # 只读
        self.mapping_table.setItem(row, 0, source_item)
        
        target_item = QTableWidgetItem(f"Spectrum {target_idx}")
        target_item.setData(256, target_idx)
        target_item.setFlags(target_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # 只读
        self.mapping_table.setItem(row, 1, target_item)
        
        remove_btn = QPushButton("删除")
        remove_btn.clicked.connect(lambda: self._remove_mapping_row(row))
        self.mapping_table.setCellWidget(row, 2, remove_btn)
        
        self._on_config_changed()
    
    def _remove_mapping(self):
        """删除选中的映射"""
        current_row = self.mapping_table.currentRow()
        if current_row >= 0:
            self.mapping_table.removeRow(current_row)
            self._on_config_changed()
    
    def _remove_mapping_row(self, row):
        """删除指定行的映射"""
        self.mapping_table.removeRow(row)
        self._on_config_changed()
    
    def _clear_mappings(self):
        """清除所有映射"""
        self.mapping_table.setRowCount(0)
        self._on_config_changed()
    
    def _apply_offsets(self):
        """应用偏移"""
        # 从表格读取偏移值
        offsets = {}
        for i in range(self.offset_table.rowCount()):
            label_item = self.offset_table.item(i, 0)
            offset_item = self.offset_table.item(i, 1)
            if label_item and offset_item:
                label = label_item.text()
                try:
                    offset = float(offset_item.text())
                    offsets[label] = offset
                except ValueError:
                    continue
        
        # 应用偏移到扫描器
        self.spectrum_scanner.apply_custom_offsets(offsets)
        # 更新偏移表格显示
        self._update_offset_table(self.spectrum_scanner.scanned_spectra)
        # 保存配置并触发更新
        self._on_config_changed()
    
    def load_config(self):
        """从配置管理器加载配置"""
        config = self.config_manager.get_config()
        ss = config.spectrum_scan
        
        self.stack_offset_spin.setValue(ss.stack_offset)
        self.scan_enabled_check.setChecked(ss.enabled)
        
        # 加载映射（如果有）
        if ss.custom_mappings:
            self.mapping_table.setRowCount(0)
            for src_idx, tgt_idx in ss.custom_mappings:
                row = self.mapping_table.rowCount()
                self.mapping_table.insertRow(row)
                self.mapping_table.setItem(row, 0, QTableWidgetItem(f"Spectrum {src_idx}"))
                self.mapping_table.setItem(row, 1, QTableWidgetItem(f"Spectrum {tgt_idx}"))
                remove_btn = QPushButton("删除")
                remove_btn.clicked.connect(lambda r=row: self._remove_mapping_row(r))
                self.mapping_table.setCellWidget(row, 2, remove_btn)
    
    def save_config(self):
        """保存配置到配置管理器"""
        config = self.config_manager.get_config()
        ss = config.spectrum_scan
        
        ss.stack_offset = self.stack_offset_spin.value()
        ss.enabled = self.scan_enabled_check.isChecked()
        
        # 保存映射
        mappings = []
        for i in range(self.mapping_table.rowCount()):
            source_item = self.mapping_table.item(i, 0)
            target_item = self.mapping_table.item(i, 1)
            if source_item and target_item:
                # 从文本中提取索引
                try:
                    src_text = source_item.text()
                    tgt_text = target_item.text()
                    src_idx = int(src_text.split()[-1])
                    tgt_idx = int(tgt_text.split()[-1])
                    mappings.append((src_idx, tgt_idx))
                except:
                    continue
        
        ss.custom_mappings = mappings
        
        # 保存独立偏移
        offsets = {}
        for i in range(self.offset_table.rowCount()):
            label_item = self.offset_table.item(i, 0)
            offset_item = self.offset_table.item(i, 1)
            if label_item and offset_item:
                label = label_item.text()
                try:
                    offset = float(offset_item.text())
                    offsets[label] = offset
                except ValueError:
                    continue
        
        ss.individual_offsets = offsets
        
        self.config_manager.update_config(config)
    
    def get_config(self):
        """获取当前配置"""
        self.save_config()
        return self.config_manager.get_config()
    
    def _on_spectrum_selected(self):
        """当选中谱线时，更新编辑控件"""
        selected_items = self.spectrum_list.selectedItems()
        if not selected_items:
            # 清空编辑控件
            self.legend_edit_input.setText("")
            self.color_edit_input.setText("")
            return
        
        # 获取选中的谱线索引
        item = selected_items[0]
        idx = item.data(256)  # 存储的索引
        
        # 从扫描器中获取谱线信息
        if self.spectrum_scanner.scanned_spectra and idx < len(self.spectrum_scanner.scanned_spectra):
            spec = self.spectrum_scanner.scanned_spectra[idx]
            label = spec.get('label', f'Spectrum {idx}')
            color = spec.get('color', '')
            
            # 更新编辑控件
            self.legend_edit_input.setText(label)
            if color:
                self.color_edit_input.setText(color)
            else:
                self.color_edit_input.setText("")
    
    def _on_legend_changed(self, text):
        """图例名称改变时"""
        selected_items = self.spectrum_list.selectedItems()
        if not selected_items:
            return
        
        # 获取选中的谱线索引
        item = selected_items[0]
        idx = item.data(256)
        
        # 更新扫描器中的标签
        if self.spectrum_scanner.scanned_spectra and idx < len(self.spectrum_scanner.scanned_spectra):
            self.spectrum_scanner.scanned_spectra[idx]['label'] = text
            # 更新列表项显示
            item.setText(f"{idx}: {text}")
            # 更新偏移表格
            self._update_offset_table(self.spectrum_scanner.scanned_spectra)
            # 触发配置更新
            self._on_config_changed()
    
    def _on_color_changed(self, text):
        """颜色改变时"""
        selected_items = self.spectrum_list.selectedItems()
        if not selected_items:
            return
        
        # 获取选中的谱线索引
        item = selected_items[0]
        idx = item.data(256)
        
        # 更新扫描器中的颜色
        if self.spectrum_scanner.scanned_spectra and idx < len(self.spectrum_scanner.scanned_spectra):
            self.spectrum_scanner.scanned_spectra[idx]['color'] = text if text else None
            # 触发配置更新
            self._on_config_changed()
    
    def _pick_color(self):
        """打开颜色选择对话框"""
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtGui import QColor
        
        # 获取当前颜色
        current_color = self.color_edit_input.text()
        initial_color = QColor(current_color) if current_color else QColor(255, 0, 0)
        
        # 打开颜色选择对话框
        color = QColorDialog.getColor(initial_color, self, "选择颜色")
        if color.isValid():
            # 转换为字符串格式（优先使用十六进制）
            color_str = color.name()
            self.color_edit_input.setText(color_str)
            # 触发颜色改变事件
            self._on_color_changed(color_str)


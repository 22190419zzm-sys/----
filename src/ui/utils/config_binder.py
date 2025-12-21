"""Config binder for automatic UI-Config synchronization"""

from typing import Dict, Any, Callable, Optional, Union
from PyQt6.QtWidgets import (
    QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox, 
    QComboBox, QTextEdit
)
from src.ui.widgets.custom_widgets import UnlimitedNumericInput


class ConfigBinder:
    """
    配置与UI控件的双向绑定类
    
    使用反射机制自动将UI控件的值变化同步到Config对象中，反之亦然。
    支持以下控件类型：
    - QCheckBox: stateChanged -> bool
    - QLineEdit: textChanged -> str
    - QTextEdit: textChanged -> str
    - QSpinBox/QDoubleSpinBox: valueChanged -> int/float
    - QComboBox: currentTextChanged -> str
    - UnlimitedNumericInput: textChanged -> str (转换为float)
    """
    
    def __init__(self, config_obj: Any, widgets: Dict[str, Any], 
                 on_change_callback: Optional[Callable] = None,
                 force_data_reload_widgets: Optional[list] = None):
        """
        初始化配置绑定器
        
        Args:
            config_obj: 配置对象（可以是任何有属性的对象）
            widgets: UI控件字典，键为配置属性名，值为控件对象
            on_change_callback: 当配置改变时的回调函数，接收 (attr_name, new_value, force_data_reload) 参数
            force_data_reload_widgets: 需要强制重新加载数据的控件名称列表
        """
        self.config_obj = config_obj
        self.widgets = widgets
        self.on_change_callback = on_change_callback
        self.force_data_reload_widgets = force_data_reload_widgets or []
        self._binding_enabled = True
        
        # 建立绑定
        self._bind_all()
    
    def _bind_all(self):
        """绑定所有控件"""
        for attr_name, widget in self.widgets.items():
            if widget is None:
                continue
            
            # 检测控件类型并连接相应的信号
            if isinstance(widget, QCheckBox):
                widget.stateChanged.connect(
                    lambda checked, name=attr_name: self._on_widget_changed(name, checked, widget)
                )
            elif isinstance(widget, (QLineEdit, QTextEdit)):
                widget.textChanged.connect(
                    lambda text, name=attr_name: self._on_widget_changed(name, text, widget)
                )
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(
                    lambda value, name=attr_name: self._on_widget_changed(name, value, widget)
                )
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(
                    lambda text, name=attr_name: self._on_widget_changed(name, text, widget)
                )
            elif isinstance(widget, UnlimitedNumericInput):
                widget.textChanged.connect(
                    lambda text, name=attr_name: self._on_widget_changed(name, text, widget)
                )
    
    def _on_widget_changed(self, attr_name: str, value: Any, widget: Any):
        """当控件值改变时的处理函数"""
        if not self._binding_enabled:
            return
        
        # 转换值类型
        converted_value = self._convert_value(widget, value)
        
        # 更新配置对象
        if hasattr(self.config_obj, attr_name):
            try:
                setattr(self.config_obj, attr_name, converted_value)
            except Exception as e:
                print(f"Warning: Failed to set {attr_name} on config object: {e}")
        
        # 调用回调函数
        if self.on_change_callback:
            force_reload = attr_name in self.force_data_reload_widgets
            self.on_change_callback(attr_name, converted_value, force_reload)
    
    def _convert_value(self, widget: Any, value: Any) -> Any:
        """根据控件类型转换值"""
        if isinstance(widget, QCheckBox):
            # QCheckBox.stateChanged 返回 Qt.CheckState，需要转换为 bool
            from PyQt6.QtCore import Qt
            return value == Qt.CheckState.Checked.value
        elif isinstance(widget, UnlimitedNumericInput):
            # UnlimitedNumericInput 返回文本，需要转换为 float
            try:
                text = value.strip().replace(",", ".")
                return float(text) if text else 0.0
            except (ValueError, AttributeError):
                return 0.0
        elif isinstance(widget, QLineEdit):
            return str(value) if value is not None else ""
        elif isinstance(widget, QTextEdit):
            return str(value) if value is not None else ""
        else:
            return value
    
    def sync_config_to_ui(self, attr_name: Optional[str] = None):
        """
        将配置对象的值同步到UI控件
        
        Args:
            attr_name: 要同步的属性名，如果为None则同步所有属性
        """
        self._binding_enabled = False  # 临时禁用绑定，避免循环更新
        
        try:
            if attr_name:
                # 同步单个属性
                if attr_name in self.widgets and hasattr(self.config_obj, attr_name):
                    widget = self.widgets[attr_name]
                    config_value = getattr(self.config_obj, attr_name)
                    self._set_widget_value(widget, config_value)
            else:
                # 同步所有属性
                for attr_name, widget in self.widgets.items():
                    if widget is not None and hasattr(self.config_obj, attr_name):
                        config_value = getattr(self.config_obj, attr_name)
                        self._set_widget_value(widget, config_value)
        finally:
            self._binding_enabled = True
    
    def _set_widget_value(self, widget: Any, value: Any):
        """设置控件值"""
        try:
            if isinstance(widget, QCheckBox):
                from PyQt6.QtCore import Qt
                widget.setCheckState(Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)
            elif isinstance(widget, (QLineEdit, QTextEdit)):
                widget.setText(str(value) if value is not None else "")
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QComboBox):
                index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
                else:
                    widget.setCurrentText(str(value))
            elif isinstance(widget, UnlimitedNumericInput):
                widget.setText(str(value) if value is not None else "")
        except Exception as e:
            print(f"Warning: Failed to set value on widget {type(widget).__name__}: {e}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取当前配置字典"""
        config_dict = {}
        for attr_name in self.widgets.keys():
            if hasattr(self.config_obj, attr_name):
                config_dict[attr_name] = getattr(self.config_obj, attr_name)
        return config_dict
    
    def update_config_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置对象，然后同步到UI"""
        for attr_name, value in config_dict.items():
            if hasattr(self.config_obj, attr_name):
                setattr(self.config_obj, attr_name, value)
        self.sync_config_to_ui()


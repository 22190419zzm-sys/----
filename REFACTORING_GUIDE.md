# 重构指南

本文档说明如何使用新创建的重构组件。

## 任务一：Tab 组件拆分

### 已创建的 Tab 类

1. **PlottingSettingsTab** (`src/ui/tabs/plotting_settings_tab.py`)
   - 包含 X 轴截断和预处理设置
   - 包含绘图模式和全局设置

2. **FileControlsTab** (`src/ui/tabs/file_controls_tab.py`)
   - 文件扫描与独立Y轴控制
   - NMF组分独立Y轴控制
   - 组瀑布图独立堆叠位移控制

3. **PeakDetectionTab** (`src/ui/tabs/peak_detection_tab.py`)
   - 波峰检测参数
   - 峰值标记样式
   - 垂直参考线设置

4. **PhysicsTab** (`src/ui/tabs/physics_tab.py`)
   - 散射尾部拟合设置

### 使用方法

在 `main_window.py` 中，将原来的 `setup_*_tab` 方法替换为：

```python
from src.ui.tabs import PlottingSettingsTab, FileControlsTab, PeakDetectionTab, PhysicsTab

# 在 __init__ 或 setup_ui 方法中：
self.plotting_tab = PlottingSettingsTab(parent=self)
self.file_controls_tab = FileControlsTab(parent=self)
self.peak_detection_tab = PeakDetectionTab(parent=self)
self.physics_tab = PhysicsTab(parent=self)

# 添加到 TabWidget
tab_widget.addTab(self.plotting_tab, "绘图设置")
tab_widget.addTab(self.file_controls_tab, "文件控制")
tab_widget.addTab(self.peak_detection_tab, "峰值检测")
tab_widget.addTab(self.physics_tab, "物理验证")
```

### 访问控件

Tab 类中的控件作为实例属性可以直接访问：

```python
# 访问绘图设置 Tab 中的控件
self.plotting_tab.x_min_phys_input.setText("600")
value = self.plotting_tab.qc_check.isChecked()
```

## 任务二：ConfigBinder 双向绑定

### 使用方法

`ConfigBinder` 类可以自动将 UI 控件的值变化同步到配置对象中。

```python
from src.ui.utils.config_binder import ConfigBinder

# 1. 创建一个配置对象（可以是任何有属性的对象）
class PlotConfig:
    def __init__(self):
        self.x_min_phys = ""
        self.qc_enabled = False
        self.qc_threshold = 5.0
        # ... 其他配置属性

config = PlotConfig()

# 2. 获取 Tab 中的所有控件
widgets_dict = self.plotting_tab.get_widgets_dict()

# 3. 创建绑定器
def on_config_changed(attr_name, new_value, force_data_reload):
    """配置改变时的回调"""
    print(f"{attr_name} changed to {new_value}")
    if force_data_reload:
        # 需要重新加载数据
        self._reload_data()

binder = ConfigBinder(
    config_obj=config,
    widgets=widgets_dict,
    on_change_callback=on_config_changed,
    force_data_reload_widgets=['qc_check', 'qc_threshold_spin', 'x_min_phys_input']
)

# 4. 同步配置到 UI（初始化时）
binder.sync_config_to_ui()

# 5. 从配置字典更新（可选）
config_dict = {'x_min_phys': '600', 'qc_enabled': True}
binder.update_config_from_dict(config_dict)
```

### 替换 _connect_all_style_update_signals

原来的 `_connect_all_style_update_signals` 方法可以替换为：

```python
def _connect_all_style_update_signals(self):
    """使用 ConfigBinder 连接所有样式参数控件"""
    from src.ui.utils.config_binder import ConfigBinder
    
    # 收集所有 Tab 的控件
    all_widgets = {}
    all_widgets.update(self.plotting_tab.get_widgets_dict())
    all_widgets.update(self.peak_detection_tab.get_widgets_dict())
    # ... 其他 Tab
    
    # 创建配置对象（如果还没有）
    if not hasattr(self, 'plot_config'):
        self.plot_config = type('PlotConfig', (), {})()
        # 初始化配置属性
        for attr_name in all_widgets.keys():
            setattr(self.plot_config, attr_name, None)
    
    # 创建绑定器
    self.config_binder = ConfigBinder(
        config_obj=self.plot_config,
        widgets=all_widgets,
        on_change_callback=self._on_style_param_changed,
        force_data_reload_widgets=[
            'qc_check', 'qc_threshold_spin', 'be_check', 'be_temp_spin',
            'smoothing_check', 'baseline_als_check', 'x_min_phys_input', 'x_max_phys_input',
            # ... 其他需要重新加载数据的控件
        ]
    )
```

## 任务三：FileService 服务层

### 使用方法

`FileService` 类提供了文件扫描、分组和跳过行数检测的功能。

```python
from src.services.file_service import FileService

# 创建服务实例
file_service = FileService()

# 1. 扫描文件夹
result = file_service.scan_folder("/path/to/folder")
print(f"找到 {len(result['files'])} 个文件")
print(f"跳过行数: {result['skip_rows']}")
print(f"检测信息: {result['skip_rows_info']}")

# 2. 分组文件
file_list = result['files']
groups = file_service.group_files(file_list, n_chars=5)
for group_name, files in groups.items():
    print(f"组 {group_name}: {len(files)} 个文件")

# 3. 检测跳过行数
skip_info = file_service.detect_skip_rows("/path/to/folder", sample_count=3)
print(f"最常见的跳过行数: {skip_info['skip_rows']}")
print(f"检测信息: {skip_info['info']}")

# 4. 扫描并准备图例重命名数据
rename_data = file_service.scan_and_load_legend_rename_data(
    folder_path="/path/to/folder",
    n_chars=5,
    target_groups=['group1', 'group2']  # 可选：指定组
)
print(f"分组: {rename_data['groups']}")
print(f"组中的文件: {rename_data['files_in_groups']}")
```

### 替换 main_window.py 中的方法

原来的 `scan_and_load_legend_rename` 方法可以替换为：

```python
def scan_and_load_legend_rename(self):
    """扫描文件并为图例重命名准备数据"""
    from src.services.file_service import FileService
    
    folder_path = self.folder_input.text()
    if not os.path.isdir(folder_path):
        return
    
    # 使用 FileService
    file_service = FileService()
    n_chars = self.n_chars_spin.value()
    target_groups = [x.strip() for x in self.groups_input.text().split(',') if x.strip()]
    
    rename_data = file_service.scan_and_load_legend_rename_data(
        folder_path=folder_path,
        n_chars=n_chars,
        target_groups=target_groups if target_groups else None
    )
    
    # 使用返回的数据创建 UI
    self.legend_rename_widgets.clear()
    self._clear_layout_recursively(self.rename_layout)
    
    # ... 使用 rename_data['groups'] 创建 UI 控件
```

原来的 `_detect_skip_rows_for_all_files` 方法可以替换为：

```python
def _detect_skip_rows_for_all_files(self):
    """检测所有文件的跳过行数"""
    from src.services.file_service import FileService
    
    folder = self.folder_input.text()
    if not folder or not os.path.isdir(folder):
        return
    
    # 使用 FileService
    file_service = FileService()
    skip_info = file_service.detect_skip_rows(folder, sample_count=3)
    
    # 更新 UI
    self.skip_rows_info_label.setText(skip_info['info'])
    
    # 如果当前是自动检测模式，更新跳过行数
    if self.skip_rows_spin.value() == -1:
        # 不改变值，但更新显示
        pass
```

## 迁移步骤

1. **逐步迁移**：不要一次性替换所有代码，先在一个 Tab 上测试
2. **保持兼容**：在迁移过程中，可以保留原来的方法作为备用
3. **测试**：确保所有功能正常工作后再移除旧代码

## 注意事项

1. Tab 类需要访问主窗口的辅助方法（如 `_create_h_layout`），所以需要传递 `parent` 参数
2. ConfigBinder 需要配置对象有对应的属性，确保属性名与控件名匹配
3. FileService 使用缓存来优化性能，如果需要清除缓存，可以创建新的实例


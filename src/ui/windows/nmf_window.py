import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import find_peaks
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from src.core.preprocessor import DataPreProcessor
from src.core.peak_detection_helper import detect_and_plot_peaks
from src.core.spectrum_scanner import SpectrumScanner
from src.core.peak_matcher import PeakMatcher
from src.ui.canvas import MplCanvas


class NMFResultWindow(QDialog):
    """[新增] NMF 分析结果独立窗口（参考4.py，所有参数在NMF分析中）"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        # 设置窗口图标
        try:
            from src.utils.icon_manager import set_window_icon
            set_window_icon(self)
        except:
            pass
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(1200, 900)
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        self.main_layout = QVBoxLayout(self)
        
        # 创建水平布局，左侧是图表，右侧是控制面板
        content_layout = QHBoxLayout()
        
        # 左侧：图表区域
        left_panel = QVBoxLayout()
        self.canvas = MplCanvas(self, width=12, height=9, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        h_layout = QHBoxLayout()
        self.export_button = QPushButton("导出 NMF 结果 (W & H)")
        self.export_button.clicked.connect(self.export_data)
        h_layout.addStretch(1)
        h_layout.addWidget(self.export_button)
        h_layout.addStretch(1)
        
        left_panel.addLayout(h_layout)
        left_panel.addWidget(self.toolbar)
        left_panel.addWidget(self.canvas)
        
        # 右侧：控制面板（使用滚动区域）
        right_panel_scroll = QScrollArea()
        right_panel_scroll.setWidgetResizable(True)
        right_panel_widget = QWidget()
        right_panel = QVBoxLayout(right_panel_widget)
        right_panel_widget.setMaximumWidth(350)
        right_panel_widget.setMinimumWidth(250)
        
        # 目标组分选择组
        target_group = QGroupBox("目标组分选择")
        target_layout = QVBoxLayout(target_group)
        
        self.target_component_button_group = QButtonGroup()
        self.target_component_radios = []  # 存储所有单选按钮
        
        target_layout.addWidget(QLabel("请选择目标信号组分："))
        
        # 初始时没有组分，set_data时会更新
        self.target_component_container = QWidget()
        self.target_component_layout = QVBoxLayout(self.target_component_container)
        self.target_component_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.target_component_container)
        
        target_layout.addStretch(1)
        right_panel.addWidget(target_group)
        right_panel.addStretch(1)
        right_panel_scroll.setWidget(right_panel_widget)
        
        # 将左右面板添加到内容布局
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        content_layout.addWidget(left_widget, stretch=3)
        content_layout.addWidget(right_panel_scroll, stretch=0)
        
        self.main_layout.addLayout(content_layout)
        
        self.W: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None
        self.common_x = None
        self.sample_labels = []
        self.style_params = {}
        self.n_components = 0
        self.target_component_index = 0  # 默认选择第一个组分
        
        # 子图axes引用（用于样式应用）
        self.ax1 = None
        self.ax2 = None
        
        # 谱线扫描器（每个子图独立）
        self.spectrum_scanners = {}  # {subplot_index: SpectrumScanner}
        self.peak_matchers = {}  # {subplot_index: PeakMatcher}
        
        # 当前子图的绘图数据（用于谱线扫描）
        self.subplot_plot_data = {}  # {subplot_index: List[Dict]}

    def set_data(self, W, H, common_x, style_params, sample_labels):
        self.W = W
        self.H = H
        self.common_x = common_x
        self.sample_labels = sample_labels
        self.style_params = style_params
        self.n_components = H.shape[0] if H is not None else 0
        
        # 初始化谱线扫描器和峰值匹配器
        self.spectrum_scanners[0] = SpectrumScanner()
        self.spectrum_scanners[1] = SpectrumScanner()
        self.peak_matchers[0] = PeakMatcher(tolerance=5.0)  # 默认tolerance
        self.peak_matchers[1] = PeakMatcher(tolerance=5.0)  # 默认tolerance
        
        # 更新目标组分选择UI
        self._update_target_component_radios()
        
        self.plot_results(style_params)
    
    def _update_target_component_radios(self):
        """更新目标组分选择单选按钮"""
        # 清除旧的单选按钮
        for radio in self.target_component_radios:
            self.target_component_button_group.removeButton(radio)
            radio.deleteLater()
        self.target_component_radios.clear()
        
        # 清除布局
        while self.target_component_layout.count():
            item = self.target_component_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 创建新的单选按钮
        if self.n_components > 0:
            # 获取NMF组分图例重命名（如果有）
            nmf_legend_names = self.style_params.get('nmf_legend_names', {})
            
            for i in range(self.n_components):
                comp_label = f"Component {i+1}"
                display_label = nmf_legend_names.get(comp_label, comp_label)
                
                radio = QRadioButton(display_label)
                radio.setChecked(i == self.target_component_index)  # 默认选择第一个
                self.target_component_button_group.addButton(radio, i)
                self.target_component_radios.append(radio)
                self.target_component_layout.addWidget(radio)
                
                # 连接信号，当选择改变时更新索引并通知父窗口
                radio.toggled.connect(lambda checked, idx=i: self._on_target_component_changed(idx) if checked else None)
    
    def _on_target_component_changed(self, index):
        """当目标组分选择改变时调用"""
        self.target_component_index = index
        # 通知父窗口（如果存在）
        parent = self.parent()
        if parent and hasattr(parent, 'update_nmf_target_component'):
            parent.update_nmf_target_component(index)
    
    def get_target_component_index(self):
        """返回当前选中的目标组分索引"""
        return self.target_component_index
        
    def export_data(self):
        if self.W is None or self.H is None:
            QMessageBox.warning(self, "警告", "没有数据可以导出。")
            return
            
        save_dir = QFileDialog.getExistingDirectory(self, "选择 NMF 结果保存目录")
        if not save_dir: return

        # 导出 H (Spectra)
        h_df = pd.DataFrame(self.H.T, index=self.common_x, columns=[f"Component_{i+1}" for i in range(self.H.shape[0])])
        h_df.index.name = "Wavenumber"
        h_df.to_csv(os.path.join(save_dir, "NMF_H_Components.csv"))
        
        # 导出 W (Weights)
        w_df = pd.DataFrame(self.W, columns=[f"Weight_Comp_{i+1}" for i in range(self.W.shape[1])])
        w_df.index = self.sample_labels
        w_df.index.name = "Sample Name"
        w_df.to_csv(os.path.join(save_dir, "NMF_W_Weights.csv"))
        
        QMessageBox.information(self, "完成", f"NMF 结果已导出到 {save_dir}。")
    
    def closeEvent(self, event):
        """窗口关闭时，保存目标组分选择到父窗口"""
        parent = self.parent()
        if parent and hasattr(parent, 'update_nmf_target_component'):
            parent.update_nmf_target_component(self.target_component_index)
        super().closeEvent(event)

    def plot_results(self, style_params):
        """绘制NMF结果（参考4.py，每次都用fig.clear()）"""
        # 使用现有的figure，只清除内容（与4.py保持一致）
        fig = self.canvas.figure
        fig.clear()  # 使用clear而不是clf，保持窗口状态
        
        # 确保 Matplotlib 有足够的空间
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # 保存axes引用
        self.ax1 = ax1
        self.ax2 = ax2
        
        n_components = self.H.shape[0]
        
        # Comp Colors
        c1_color = style_params['comp1_color']
        c2_color = style_params['comp2_color']
        colors = [c1_color, c2_color] + ['green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal', 'darkred']

        # 提取绘图参数
        is_derivative = style_params.get('is_derivative', False)
        global_stack_offset = style_params.get('global_stack_offset', 0.0)
        global_scale_factor = style_params.get('global_scale_factor', 1.0)
        individual_y_params = style_params.get('individual_y_params', {})
        control_data_list = style_params.get('control_data_list', [])
        
        # 获取NMF组分图例重命名
        nmf_legend_names = style_params.get('nmf_legend_names', {})
        
        # 获取全局配置（从主窗口的样式面板）
        global_config = None
        if self.parent():
            # 优先从样式与匹配窗口获取配置
            if hasattr(self.parent(), '_style_matching_window') and self.parent()._style_matching_window:
                style_window = self.parent()._style_matching_window
                if hasattr(style_window, 'publication_style_panel'):
                    from src.core.plot_config_manager import PlotConfigManager
                    config_manager = PlotConfigManager()
                    global_config = config_manager.get_config()
            # 否则从主窗口的样式面板获取
            elif hasattr(self.parent(), 'publication_style_panel') and self.parent().publication_style_panel:
                from src.core.plot_config_manager import PlotConfigManager
                config_manager = PlotConfigManager()
                global_config = config_manager.get_config()
        
        # 如果没有获取到配置，使用默认配置管理器
        if global_config is None:
            from src.core.plot_config_manager import PlotConfigManager
            config_manager = PlotConfigManager()
            global_config = config_manager.get_config()
        
        # 准备子图0的绘图数据（用于谱线扫描和峰值匹配）
        subplot0_plot_data = []
        
        # 绘制 H (Components/Spectra)
        for i in range(n_components):
            comp_color = colors[i % len(colors)]
            y_data = self.H[i].copy()
            
            # 应用动态范围压缩预处理（在对数/平方根变换之前）
            comp_label = f"Component {i+1}"
            ind_params = individual_y_params.get(comp_label, {'scale': 1.0, 'offset': 0.0, 'transform': 'none', 'transform_params': {}})
            
            transform_mode = ind_params.get('transform', 'none')
            transform_params = ind_params.get('transform_params', {})
            
            if transform_mode == 'log':
                y_data = DataPreProcessor.apply_log_transform(y_data,
                    base=transform_params.get('base', 10),
                    offset=transform_params.get('offset', 1.0))
            elif transform_mode == 'sqrt':
                y_data = DataPreProcessor.apply_sqrt_transform(y_data,
                    offset=transform_params.get('offset', 0.0))
            
            # 应用二阶导数（如果启用）
            if is_derivative:
                y_data = np.gradient(np.gradient(y_data))
            
            # 应用全局缩放
            y_data = y_data * global_scale_factor
            
            # 应用独立Y轴参数（如果存在）
            y_data = y_data * ind_params['scale'] + ind_params['offset']
            
            # 应用堆叠偏移
            y_final = y_data + (i * global_stack_offset)
            
            # 使用重命名后的图例名称（如果存在），否则使用默认名称
            display_label = nmf_legend_names.get(comp_label, comp_label)
            
            ax1.plot(self.common_x, y_final, 
                     label=display_label, 
                     color=comp_color, 
                     linewidth=style_params['comp_line_width'],
                     linestyle=style_params['comp_line_style'])
            
            # 保存绘图数据用于谱线扫描和峰值匹配
            subplot0_plot_data.append({
                'x': self.common_x,
                'y': y_final,
                'y_raw': self.H[i],
                'label': display_label,
                'color': comp_color,
                'linewidth': style_params['comp_line_width'],
                'linestyle': style_params['comp_line_style']
            })
            
            # 峰值检测（优先使用子图独立配置，否则使用全局配置）
            peak_config = None
            # 尝试获取子图0的独立峰值检测配置
            if self.parent():
                if hasattr(self.parent(), '_style_matching_window') and self.parent()._style_matching_window:
                    style_window = self.parent()._style_matching_window
                    if hasattr(style_window, 'get_subplot_controller'):
                        try:
                            subplot0_controller = style_window.get_subplot_controller("NMFResultWindow", 0)
                            if subplot0_controller and hasattr(subplot0_controller, 'get_peak_config'):
                                peak_config = subplot0_controller.get_peak_config()
                        except:
                            pass
            
            # 如果没有子图独立配置，使用全局配置
            if peak_config is None:
                peak_config = global_config.peak_detection if global_config else None
            
            if peak_config and peak_config.enabled:
                    # 准备峰值检测参数
                    peak_params = {
                        'peak_detection_enabled': True,
                        'peak_height_threshold': peak_config.height_threshold,
                        'peak_distance_min': peak_config.distance_min,
                        'peak_prominence': peak_config.prominence,
                        'peak_width': peak_config.width,
                        'peak_wlen': peak_config.wlen,
                        'peak_rel_height': peak_config.rel_height,
                        'peak_show_label': peak_config.show_label,
                        'peak_label_font': peak_config.label_font,
                        'peak_label_size': peak_config.label_size,
                        'peak_label_color': peak_config.label_color,
                        'peak_label_bold': peak_config.label_bold,
                        'peak_label_rotation': peak_config.label_rotation,
                        'peak_marker_shape': peak_config.marker_shape,
                        'peak_marker_size': peak_config.marker_size,
                        'peak_marker_color': None,  # 使用线条颜色
                    }
                    try:
                        # 使用统一的峰值检测函数
                        detect_and_plot_peaks(
                            ax1, 
                            self.common_x, 
                            self.H[i],  # 使用原始数据检测峰值
                            y_final,  # 使用最终数据绘制峰值
                            peak_params, 
                            color=comp_color
                        )
                    except Exception as e:
                        print(f"NMF峰值检测失败: {e}")
        
        # 保存子图0的绘图数据
        self.subplot_plot_data[0] = subplot0_plot_data
        
        # 谱线扫描（子图0）- 优先使用子图独立配置，否则使用全局配置
        scan_config = None
        # 尝试获取子图0的独立配置
        if self.parent():
            if hasattr(self.parent(), '_style_matching_window') and self.parent()._style_matching_window:
                style_window = self.parent()._style_matching_window
                if hasattr(style_window, 'get_subplot_controller'):
                    try:
                        subplot0_controller = style_window.get_subplot_controller("NMFResultWindow", 0)
                        if subplot0_controller and hasattr(subplot0_controller, 'get_scan_config'):
                            scan_config = subplot0_controller.get_scan_config()
                    except:
                        pass
        
        # 如果没有子图独立配置，使用全局配置
        if scan_config is None:
            scan_config = global_config.spectrum_scan if global_config else None
        
        if scan_config and scan_config.enabled and len(subplot0_plot_data) > 1:
            try:
                # 应用堆叠偏移（必须在扫描之前设置）
                if scan_config.stack_offset > 0:
                    self.spectrum_scanners[0].set_stack_offset(scan_config.stack_offset)
                
                # 应用自定义偏移（必须在扫描之前设置）
                if scan_config.individual_offsets:
                    self.spectrum_scanners[0].apply_custom_offsets(scan_config.individual_offsets)
                
                # 扫描谱线
                scanned = self.spectrum_scanners[0].scan_last_plot(subplot0_plot_data)
                
                # 应用映射（如果提供）
                if scan_config.custom_mappings:
                    aligned = self.spectrum_scanners[0].apply_mappings(
                        scan_config.custom_mappings, 
                        interpolation=True,
                        common_x=self.common_x
                    )
                    # 更新绘图数据
                    for idx, item in enumerate(subplot0_plot_data):
                        if idx < len(aligned):
                            aligned_item = aligned[idx]
                            item['y'] = aligned_item.get('y', item['y'])
                            item['x'] = aligned_item.get('x', item['x'])
                            # 重新绘制
                            if idx < len(ax1.lines):
                                ax1.lines[idx].set_xdata(item['x'])
                                ax1.lines[idx].set_ydata(item['y'])
                
                # 重新绘制所有谱线（应用堆叠偏移）
                for idx, item in enumerate(subplot0_plot_data):
                    if idx < len(ax1.lines):
                        ax1.lines[idx].set_ydata(item['y'])
                ax1.figure.canvas.draw()
            except Exception as e:
                import traceback
                print(f"NMF谱线扫描失败: {e}")
                traceback.print_exc()
        
        # 峰值匹配（子图0）- 使用全局配置
        match_config = global_config.peak_matching if global_config else None
        
        if match_config and match_config.enabled and len(subplot0_plot_data) > 1:
                try:
                    # 准备峰值匹配数据
                    match_data = []
                    for item in subplot0_plot_data:
                        match_data.append({
                            'x': item['x'],
                            'y': item['y_raw'],  # 使用原始数据匹配峰值
                            'label': item['label'],
                            'color': item['color']
                        })
                    
                    # 使用窗口的峰值匹配器（已初始化）
                    if 0 in self.peak_matchers:
                        peak_matcher = self.peak_matchers[0]
                        peak_matcher.tolerance = match_config.tolerance  # 更新tolerance
                        
                        # 执行峰值匹配
                        matched_result = peak_matcher.match_multiple_spectra(
                            match_data,
                            reference_index=match_config.reference_index,
                            mode=match_config.mode
                        )
                        
                        # 绘制匹配的峰值（使用配置中的标记样式）
                        if matched_result and 'matches' in matched_result:
                            matches = matched_result['matches']
                            marker_shape = match_config.marker_shape if hasattr(match_config, 'marker_shape') else 'o'
                            marker_size = match_config.marker_size if hasattr(match_config, 'marker_size') else 8
                            marker_color = match_config.marker_color if hasattr(match_config, 'marker_color') else None
                            
                            for spec_idx, match_info in matches.items():
                                if 'positions' in match_info:
                                    peak_positions = match_info['positions']
                                    # 获取对应的Y值
                                    spec_data = match_data[int(spec_idx)]
                                    for peak_x in peak_positions:
                                        # 找到最近的X值索引
                                        x_idx = np.argmin(np.abs(spec_data['x'] - peak_x))
                                        peak_y = spec_data['y'][x_idx] if x_idx < len(spec_data['y']) else 0
                                        # 使用配置的颜色，如果没有则使用线条颜色
                                        plot_color = marker_color if marker_color else spec_data['color']
                                        # 绘制峰值标记
                                        ax1.plot(peak_x, peak_y, marker=marker_shape, 
                                               markersize=marker_size, color=plot_color, alpha=0.7)
                except Exception as e:
                    print(f"NMF峰值匹配失败: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 绘制对照组（如果存在）
        if control_data_list:
            control_colors = ['black', 'darkblue', 'darkred', 'darkgreen', 'darkmagenta']
            for idx, ctrl_data in enumerate(control_data_list):
                ctrl_y = ctrl_data['y'].copy()
                
                # 应用二阶导数（如果启用）
                if is_derivative:
                    ctrl_y = np.gradient(np.gradient(ctrl_y))
                
                # 应用全局缩放
                ctrl_y = ctrl_y * global_scale_factor
                
                # 应用独立Y轴参数（如果存在）
                ctrl_label = ctrl_data['label']
                ind_params = individual_y_params.get(ctrl_label, {'scale': 1.0, 'offset': 0.0})
                ctrl_y = ctrl_y * ind_params['scale'] + ind_params['offset']
                
                # 应用堆叠偏移（对照组放在最后）
                ctrl_y_final = ctrl_y + (n_components * global_stack_offset)
                
                ctrl_color = control_colors[idx % len(control_colors)]
                ax1.plot(ctrl_data['x'], ctrl_y_final,
                        label=f"{ctrl_label} (Ref)",
                        color=ctrl_color,
                        linewidth=style_params['comp_line_width'],
                        linestyle='--',  # 对照组用虚线
                        alpha=0.7)
            
        # 绘制垂直参考线（如果存在）
        vertical_lines = style_params.get('vertical_lines', [])
        if vertical_lines:
            vertical_line_color = style_params.get('vertical_line_color', '#034DFB')
            vertical_line_style = style_params.get('vertical_line_style', '--')
            vertical_line_width = style_params.get('vertical_line_width', 0.8)
            vertical_line_alpha = style_params.get('vertical_line_alpha', 0.8)
            for line_x in vertical_lines:
                ax1.axvline(x=line_x, color=vertical_line_color, linestyle=vertical_line_style,
                          linewidth=vertical_line_width, alpha=vertical_line_alpha)
        
        if style_params['x_axis_invert']: ax1.invert_xaxis()
        # 图例将在后面根据子图样式设置
        # 使用自定义标题和轴标签
        top_title = style_params.get('nmf_top_title', 'Extracted Spectra (Components)')
        bottom_title = style_params.get('nmf_bottom_title', 'Concentration Weights (vs. Sample)')
        top_xlabel = style_params.get('nmf_top_xlabel', 'Wavenumber ($\\mathrm{cm^{-1}}$)')
        top_ylabel = style_params.get('nmf_top_ylabel', 'Intensity (Arb. Unit)')
        bottom_xlabel = style_params.get('nmf_bottom_xlabel', 'Sample Name')
        bottom_ylabel = style_params.get('nmf_bottom_ylabel', 'Weight (Arb. Unit)')
        
        # 使用GUI中的标题控制参数
        top_title_fontsize = style_params.get('nmf_top_title_fontsize', style_params['title_font_size'])
        top_title_pad = style_params.get('nmf_top_title_pad', 10.0)
        top_title_show = style_params.get('nmf_top_title_show', True)
        
        if top_title_show:
            ax1.set_title(top_title, fontsize=top_title_fontsize, pad=top_title_pad)
        
        # 使用GUI中的上图X轴标题控制参数
        top_xlabel_fontsize = style_params.get('nmf_top_xlabel_fontsize', style_params['label_font_size'])
        top_xlabel_pad = style_params.get('nmf_top_xlabel_pad', 10.0)
        top_xlabel_show = style_params.get('nmf_top_xlabel_show', True)
        
        if top_xlabel_show:
            ax1.set_xlabel(top_xlabel, fontsize=top_xlabel_fontsize, labelpad=top_xlabel_pad)
        
        # 使用GUI中的上图Y轴标题控制参数
        top_ylabel_fontsize = style_params.get('nmf_top_ylabel_fontsize', style_params['label_font_size'])
        top_ylabel_pad = style_params.get('nmf_top_ylabel_pad', 10.0)
        top_ylabel_show = style_params.get('nmf_top_ylabel_show', True)
        
        if top_ylabel_show:
            ax1.set_ylabel(top_ylabel, fontsize=top_ylabel_fontsize, labelpad=top_ylabel_pad)
        
        ax1.tick_params(labelsize=style_params['tick_font_size'])

        # subplot1_config 已移除，统一使用 global_config
        
        # 准备子图1的绘图数据（用于谱线扫描）
        subplot1_plot_data = []
        
        # 绘制 W (Weights/Concentrations)
        sample_indices = np.arange(len(self.sample_labels))
        
        for i in range(n_components):
            ax2.plot(sample_indices, self.W[:, i], 
                     marker=style_params['weight_marker_style'], 
                     markersize=style_params['weight_marker_size'],
                     linestyle=style_params['weight_line_style'],
                     linewidth=style_params['weight_line_width'],
                     label=f"Comp {i+1} Weight", 
                     color=colors[i % len(colors)])
            
            # 保存绘图数据用于谱线扫描
            subplot1_plot_data.append({
                'x': sample_indices,
                'y': self.W[:, i],
                'label': f"Comp {i+1} Weight",
                'color': colors[i % len(colors)],
                'linewidth': style_params['weight_line_width'],
                'linestyle': style_params['weight_line_style']
            })
        
        # 保存子图1的绘图数据
        self.subplot_plot_data[1] = subplot1_plot_data
        
        # 谱线扫描（子图1）- 优先使用子图独立配置，否则使用全局配置
        scan_config1 = None
        # 尝试获取子图1的独立配置
        if self.parent():
            if hasattr(self.parent(), '_style_matching_window') and self.parent()._style_matching_window:
                style_window = self.parent()._style_matching_window
                if hasattr(style_window, 'get_subplot_controller'):
                    try:
                        subplot1_controller = style_window.get_subplot_controller("NMFResultWindow", 1)
                        if subplot1_controller and hasattr(subplot1_controller, 'get_scan_config'):
                            scan_config1 = subplot1_controller.get_scan_config()
                    except:
                        pass
        
        # 如果没有子图独立配置，使用全局配置
        if scan_config1 is None:
            scan_config1 = global_config.spectrum_scan if global_config else None
        
        if scan_config1 and scan_config1.enabled and len(subplot1_plot_data) > 1:
                try:
                    # 扫描谱线
                    scanned = self.spectrum_scanners[1].scan_last_plot(subplot1_plot_data)
                    
                    # 应用堆叠偏移
                    if scan_config1.stack_offset > 0:
                        self.spectrum_scanners[1].set_stack_offset(scan_config1.stack_offset)
                    
                    # 应用自定义偏移
                    if scan_config1.individual_offsets:
                        self.spectrum_scanners[1].apply_custom_offsets(scan_config1.individual_offsets)
                    
                    # 应用映射（如果提供）
                    if scan_config1.custom_mappings:
                        aligned = self.spectrum_scanners[1].apply_mappings(
                            scan_config1.custom_mappings, 
                            interpolation=True,
                            common_x=sample_indices
                        )
                        # 更新绘图数据
                        for idx, item in enumerate(subplot1_plot_data):
                            if idx < len(aligned):
                                aligned_item = aligned[idx]
                                item['y'] = aligned_item.get('y', item['y'])
                                item['x'] = aligned_item.get('x', item['x'])
                                # 重新绘制
                                if idx < len(ax2.lines):
                                    ax2.lines[idx].set_xdata(item['x'])
                                    ax2.lines[idx].set_ydata(item['y'])
                except Exception as e:
                    print(f"NMF子图1谱线扫描失败: {e}")
        
        ax2.set_xticks(sample_indices)
        ax2.set_xticklabels(self.sample_labels, rotation=45, ha='right', fontsize=style_params['tick_font_size']) 
        # 图例将在后面根据子图样式设置
        
        # 使用GUI中的标题控制参数
        bottom_title_fontsize = style_params.get('nmf_bottom_title_fontsize', style_params['title_font_size'])
        bottom_title_pad = style_params.get('nmf_bottom_title_pad', 10.0)
        bottom_title_show = style_params.get('nmf_bottom_title_show', True)
        
        if bottom_title_show:
            ax2.set_title(bottom_title, fontsize=bottom_title_fontsize, pad=bottom_title_pad)
        
        # 使用GUI中的下图X轴标题控制参数
        bottom_xlabel_fontsize = style_params.get('nmf_bottom_xlabel_fontsize', style_params['label_font_size'])
        bottom_xlabel_pad = style_params.get('nmf_bottom_xlabel_pad', 10.0)
        bottom_xlabel_show = style_params.get('nmf_bottom_xlabel_show', True)
        
        if bottom_xlabel_show:
            ax2.set_xlabel(bottom_xlabel, fontsize=bottom_xlabel_fontsize, labelpad=bottom_xlabel_pad)
        
        # 使用GUI中的下图Y轴标题控制参数
        bottom_ylabel_fontsize = style_params.get('nmf_bottom_ylabel_fontsize', style_params['label_font_size'])
        bottom_ylabel_pad = style_params.get('nmf_bottom_ylabel_pad', 10.0)
        bottom_ylabel_show = style_params.get('nmf_bottom_ylabel_show', True)
        
        if bottom_ylabel_show:
            ax2.set_ylabel(bottom_ylabel, fontsize=bottom_ylabel_fontsize, labelpad=bottom_ylabel_pad)
        
        ax2.tick_params(labelsize=style_params['tick_font_size'])
        
        # 应用多子图样式控制（优先使用子图独立样式，否则使用全局样式）
        from src.core.style_applier import apply_publication_style_to_axes
        
        # 子图0样式 - 尝试从样式匹配窗口获取控制器
        subplot0_controller = None
        try:
            if self.parent():
                parent = self.parent()
                if hasattr(parent, '_style_matching_window') and parent._style_matching_window:
                    style_window = parent._style_matching_window
                    if hasattr(style_window, 'get_subplot_controller'):
                        subplot0_controller = style_window.get_subplot_controller("NMFResultWindow", 0)
        except Exception as e:
            # 如果获取控制器失败，使用全局配置
            print(f"获取子图控制器失败: {e}")
            subplot0_controller = None
        
        if subplot0_controller:
            subplot0_controller.apply_style_to_axes(ax1)
        elif global_config:
            # 使用全局样式
            apply_publication_style_to_axes(ax1, global_config)
        else:
            # 最后回退
            apply_publication_style_to_axes(ax1)
        
        # 子图1样式 - 优先使用子图独立样式，否则使用全局样式
        subplot1_controller = None
        try:
            if self.parent():
                parent = self.parent()
                if hasattr(parent, '_style_matching_window') and parent._style_matching_window:
                    style_window = parent._style_matching_window
                    if hasattr(style_window, 'get_subplot_controller'):
                        subplot1_controller = style_window.get_subplot_controller("NMFResultWindow", 1)
        except Exception as e:
            # 如果获取控制器失败，使用全局配置
            print(f"获取子图1控制器失败: {e}")
            subplot1_controller = None
        
        if subplot1_controller:
            subplot1_controller.apply_style_to_axes(ax2)
        elif global_config:
            # 使用全局样式
            apply_publication_style_to_axes(ax2, global_config)
        else:
            # 最后回退
            apply_publication_style_to_axes(ax2)
        
        # 图例设置（使用子图样式或主菜单参数）
        # 子图0图例
        if subplot0_controller:
            config0 = subplot0_controller.get_style_config()
            ps0 = config0.publication_style
            if ps0.show_legend:
                from matplotlib.font_manager import FontProperties
                legend_font = FontProperties()
                font_family = ps0.font_family
                if font_family == 'SimHei':
                    legend_font.set_family('sans-serif')
                else:
                    legend_font.set_family(font_family)
                legend_font.set_size(ps0.legend_fontsize)
                
                ax1.legend(loc=ps0.legend_loc, fontsize=ps0.legend_fontsize, 
                          frameon=ps0.legend_frame, prop=legend_font,
                          ncol=ps0.legend_ncol, columnspacing=ps0.legend_columnspacing,
                          labelspacing=ps0.legend_labelspacing, handlelength=ps0.legend_handlelength)
        else:
            # 回退到全局样式
            if style_params.get('show_legend', True):
                legend_fontsize = style_params.get('legend_fontsize', style_params.get('legend_font_size', 10))
                legend_frame = style_params.get('legend_frame', True)
                legend_loc = style_params.get('legend_loc', 'best')
                
                from matplotlib.font_manager import FontProperties
                legend_font = FontProperties()
                font_family = style_params.get('font_family', 'Times New Roman')
                if font_family == 'SimHei':
                    legend_font.set_family('sans-serif')
                else:
                    legend_font.set_family(font_family)
                legend_font.set_size(legend_fontsize)
                
                legend_ncol = style_params.get('legend_ncol', 1)
                legend_columnspacing = style_params.get('legend_columnspacing', 2.0)
                legend_labelspacing = style_params.get('legend_labelspacing', 0.5)
                legend_handlelength = style_params.get('legend_handlelength', 2.0)
                
                ax1.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                          ncol=legend_ncol, columnspacing=legend_columnspacing,
                          labelspacing=legend_labelspacing, handlelength=legend_handlelength)
        
        # 子图1图例 - 使用全局配置
        if global_config:
            ps1 = global_config.publication_style
            if ps1.show_legend:
                from matplotlib.font_manager import FontProperties
                legend_font = FontProperties()
                font_family = ps1.font_family
                if font_family == 'SimHei':
                    legend_font.set_family('sans-serif')
                else:
                    legend_font.set_family(font_family)
                legend_font.set_size(ps1.legend_fontsize)
                
                ax2.legend(loc=ps1.legend_loc, fontsize=ps1.legend_fontsize,
                          frameon=ps1.legend_frame, prop=legend_font,
                          ncol=ps1.legend_ncol, columnspacing=ps1.legend_columnspacing,
                          labelspacing=ps1.legend_labelspacing, handlelength=ps1.legend_handlelength)
        else:
            # 回退到全局样式
            if style_params.get('show_legend', True):
                legend_fontsize = style_params.get('legend_fontsize', style_params.get('legend_font_size', 10))
                legend_frame = style_params.get('legend_frame', True)
                legend_loc = style_params.get('legend_loc', 'best')
                
                from matplotlib.font_manager import FontProperties
                legend_font = FontProperties()
                font_family = style_params.get('font_family', 'Times New Roman')
                if font_family == 'SimHei':
                    legend_font.set_family('sans-serif')
                else:
                    legend_font.set_family(font_family)
                legend_font.set_size(legend_fontsize)
                
                legend_ncol = style_params.get('legend_ncol', 1)
                legend_columnspacing = style_params.get('legend_columnspacing', 2.0)
                legend_labelspacing = style_params.get('legend_labelspacing', 0.5)
                legend_handlelength = style_params.get('legend_handlelength', 2.0)
                
                ax2.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                          ncol=legend_ncol, columnspacing=legend_columnspacing,
                          labelspacing=legend_labelspacing, handlelength=legend_handlelength)
        
        # 添加纵横比控制
        aspect_ratio = style_params.get('aspect_ratio', 0.0)
        if aspect_ratio > 0:
            ax1.set_box_aspect(aspect_ratio)
            ax2.set_box_aspect(aspect_ratio)
        else:
            ax1.set_aspect('auto')
            ax2.set_aspect('auto')
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
            fig.tight_layout()
        self.canvas.draw()


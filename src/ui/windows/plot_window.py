import os
import warnings
from collections import defaultdict

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from scipy.signal import find_peaks

from src.core.preprocessor import DataPreProcessor
from src.core.peak_detection_helper import detect_and_plot_peaks as unified_detect_and_plot_peaks
from src.ui.canvas import MplCanvas


class MplPlotWindow(QDialog):
    def __init__(self, group_name, initial_geometry=None, parent=None):
        super().__init__(parent)
        self.group_name = group_name
        self.setWindowTitle(f"光谱图 - 组别: {group_name}")
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        
        # 如果没有指定初始位置，自动计算一个远离主菜单的位置
        if initial_geometry is None:
            initial_geometry = self._calculate_off_screen_position(parent)
        
        self.setGeometry(*initial_geometry)
        self.main_layout = QVBoxLayout(self)
        
        # 尺寸在 update_plot 中根据 params 调整
        self.canvas = MplCanvas(self)
        self.main_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(200, 150)

        self.last_geometry = initial_geometry
        self.moveEvent = self._update_geometry_on_move
        self.resizeEvent = self._update_geometry_on_resize
        
        # 存储当前绘制的数据和 Axes 对象，用于叠加绘图
        self.current_plot_data = defaultdict(lambda: {'x': np.array([]), 'y': np.array([]), 'label': '', 'color': 'gray', 'type': 'Individual'})
        self.current_ax = self.canvas.axes
        
        # 初始化标题状态
        self.has_title = False
    
    def _calculate_off_screen_position(self, parent=None):
        """
        计算一个远离主菜单的窗口位置（屏幕下方，不遮挡主界面）
        
        Returns:
            tuple: (x, y, width, height) 窗口几何信息
        """
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QScreen
        
        # 获取主屏幕
        app = QApplication.instance()
        if app is None:
            return (100, 100, 1000, 600)  # 默认位置
        
        screen = app.primaryScreen()
        if screen is None:
            return (100, 100, 1000, 600)  # 默认位置
        
        # 获取屏幕可用区域
        available_geometry = screen.availableGeometry()
        screen_width = available_geometry.width()
        screen_height = available_geometry.height()
        
        # 获取主窗口位置（如果提供了parent）
        if parent and hasattr(parent, 'geometry'):
            main_geometry = parent.geometry()
            main_x = main_geometry.x()
            main_y = main_geometry.y()
            main_width = main_geometry.width()
            main_height = main_geometry.height()
        else:
            # 如果没有parent，假设主窗口在左上角
            main_x = 0
            main_y = 0
            main_width = 1200
            main_height = 900
        
        # 窗口尺寸
        window_width = 1000
        window_height = 600
        
        # 策略：将窗口放在屏幕下方，尽量不遮挡主窗口
        # 优先放在主窗口右侧下方，如果空间不够则放在屏幕下方居中
        
        # 尝试放在主窗口右侧下方
        x_right = main_x + main_width + 50
        y_below = main_y + main_height + 50
        
        # 如果右侧空间足够，使用右侧位置
        if x_right + window_width <= screen_width and y_below + window_height <= screen_height:
            x = x_right
            y = y_below
        else:
            # 否则放在屏幕下方居中（但确保不遮挡主窗口）
            x = (screen_width - window_width) // 2
            # 放在屏幕下方，但确保在屏幕内
            y = screen_height - window_height - 50
            
            # 如果主窗口在下方，则放在主窗口上方
            if main_y + main_height > screen_height - window_height - 100:
                y = max(50, main_y - window_height - 50)
        
        # 最终检查：确保窗口完全在屏幕内
        if x + window_width > screen_width:
            x = screen_width - window_width - 50
        if y + window_height > screen_height:
            y = screen_height - window_height - 50
        
        # 确保位置不为负
        x = max(50, x)
        y = max(50, y)
        
        return (x, y, window_width, window_height)

    def _update_geometry_on_move(self, event):
        current_rect = self.geometry()
        self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        super().moveEvent(event)

    def _update_geometry_on_resize(self, event):
        current_rect = self.geometry()
        self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        
        # 与数据处理.py保持一致：不调整figure大小，让matplotlib自动适应窗口
        # 使用subplots_adjust代替tight_layout以避免警告
        try:
            # 先尝试tight_layout，如果失败则使用subplots_adjust
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                self.canvas.figure.tight_layout(pad=1.0)
        except:
            # 如果tight_layout失败，使用subplots_adjust作为后备
            try:
                self.canvas.figure.subplots_adjust(
                    left=0.12, right=0.95, bottom=0.12, top=0.95
                )
            except:
                pass
        try:
            self.canvas.draw()
        except:
            pass
        
        super().resizeEvent(event)

    def detect_and_plot_peaks(self, ax, x_data, y_detect, y_final, plot_params, color='blue'):
        """
        通用的波峰检测和绘制函数
        使用统一的峰值检测辅助函数
        x_data: X轴数据（波数）
        y_detect: 用于检测的Y数据（去除偏移）
        y_final: 用于绘制的Y数据（包含偏移）
        plot_params: 绘图参数字典
        color: 线条颜色（用于标记颜色默认值）
        """
        # 使用统一的峰值检测函数
        unified_detect_and_plot_peaks(ax, x_data, y_detect, y_final, plot_params, color)

    def update_plot(self, plot_params):
        # 延迟设置字体（首次绘图时）
        if not hasattr(self, '_fonts_setup'):
            from src.utils.fonts import setup_matplotlib_fonts
            setup_matplotlib_fonts()
            self._fonts_setup = True
        """
        核心绘图逻辑 - 保持与数据处理.py一致的绘图方式
        使用ax.cla()而不是figure.clf()，保持布局一致性
        """
        # 使用现有的axes，只清除内容（与数据处理.py保持一致）
        ax = self.canvas.axes
        
        # 检查是否手动缩放过（与数据处理.py保持一致）
        try:
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()
            # 检查是否是默认范围之外的缩放
            is_zoomed = not np.allclose(current_xlim, self.canvas.default_xlim) or \
                        not np.allclose(current_ylim, self.canvas.default_ylim)
        except AttributeError:
            is_zoomed = False
            current_xlim = None
            current_ylim = None
        
        # 只清除axes内容，保持axes对象和布局（与数据处理.py一致）
        ax.cla()
        
        # 清空旧数据引用
        self.current_plot_data.clear()
        self.current_ax = ax

        # --- 2. 提取基础参数 ---
        grouped_files_data = plot_params.get('grouped_files_data', [])
        if not grouped_files_data:
            # 如果没有提供数据，尝试从当前绘图数据重建（用于样式更新）
            # 但更安全的方式是返回，让调用者重新读取数据
            print("警告: update_plot 缺少 grouped_files_data，无法更新绘图")
            return
        control_data_list = plot_params.get('control_data_list', []) 
        individual_y_params = plot_params.get('individual_y_params', {}) 
        
        # --- 3. 提取显示/模式参数 ---
        plot_mode = plot_params.get('plot_mode', 'Normal Overlay')
        show_y_values = plot_params.get('show_y_values', True)
        is_derivative = plot_params['is_derivative']
        x_axis_invert = plot_params['x_axis_invert'] 
        
        global_stack_offset = plot_params.get('global_stack_offset', 0.5)
        global_scale_factor = plot_params.get('global_scale_factor', 1.0)
        
        # --- 4. 提取预处理参数 ---
        qc_enabled = plot_params.get('qc_enabled', False)
        qc_threshold = plot_params.get('qc_threshold', 5.0)
        is_baseline_als = plot_params.get('is_baseline_als', False)
        als_lam = plot_params.get('als_lam', 10000)
        als_p = plot_params.get('als_p', 0.005)
        is_baseline = plot_params.get('is_baseline', False) 
        baseline_points = plot_params.get('baseline_points', 50)
        baseline_poly = plot_params.get('baseline_poly', 3)
        is_smoothing = plot_params['is_smoothing']
        smoothing_window = plot_params['smoothing_window']
        smoothing_poly = plot_params['smoothing_poly']
        normalization_mode = plot_params['normalization_mode']
        
        # Bose-Einstein
        is_be_correction = plot_params.get('is_be_correction', False)
        be_temp = plot_params.get('be_temp', 300.0)
        
        # 全局动态变换和整体Y轴偏移
        global_transform_mode = plot_params.get('global_transform_mode', '无')
        global_log_base_text = plot_params.get('global_log_base', '10')
        global_log_base = float(global_log_base_text) if global_log_base_text == '10' else np.e
        global_log_offset = plot_params.get('global_log_offset', 1.0)
        global_sqrt_offset = plot_params.get('global_sqrt_offset', 0.0)
        global_y_offset = plot_params.get('global_y_offset', 0.0)
        
        # --- 5. 提取出版样式参数 ---
        line_width = plot_params['line_width']
        line_style = plot_params['line_style']
        font_family = plot_params['font_family']
        axis_title_fontsize = plot_params['axis_title_fontsize']
        tick_label_fontsize = plot_params['tick_label_fontsize']
        legend_fontsize = plot_params.get('legend_fontsize', 10)
        
        show_legend = plot_params['show_legend']
        legend_frame = plot_params['legend_frame']
        legend_loc = plot_params['legend_loc']
        
        # 图例高级控制参数
        legend_ncol = plot_params.get('legend_ncol', 1)
        legend_columnspacing = plot_params.get('legend_columnspacing', 2.0)
        legend_labelspacing = plot_params.get('legend_labelspacing', 0.5)
        legend_handlelength = plot_params.get('legend_handlelength', 2.0)
        
        show_grid = plot_params['show_grid']
        grid_alpha = plot_params['grid_alpha']
        shadow_alpha = plot_params['shadow_alpha']
        main_title_text = plot_params.get('main_title_text', "").strip()
        
        # Aspect Ratio & Plot Style
        aspect_ratio = plot_params.get('aspect_ratio', 0.0)
        plot_style = plot_params.get('plot_style', 'line') # line, scatter
        
        # 设置字体 (仅影响当前 Figure)
        current_font = 'Times New Roman' if font_family == 'Times New Roman' else font_family
        
        # 使用 Viridis 调色板，或用户自定义
        custom_colors = plot_params.get('custom_colors', ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal', 'darkred'])
        
        # 辅助函数：单条数据预处理（使用统一预处理函数）
        def preprocess_single_spectrum(x, y, file_path=None):
            """
            使用统一的预处理函数进行预处理，支持缓存
            
            Args:
                x: X轴数据
                y: Y轴数据
                file_path: 文件路径（用于缓存，可选）
            
            Returns:
                预处理后的Y数据，如果QC失败返回None
            """
            # 准备预处理参数
            preprocess_params = {
                'qc_enabled': qc_enabled,
                'qc_threshold': qc_threshold,
                'is_be_correction': is_be_correction,
                'be_temp': be_temp,
                'is_smoothing': is_smoothing,
                'smoothing_window': smoothing_window,
                'smoothing_poly': smoothing_poly,
                'is_baseline_als': is_baseline_als,
                'als_lam': als_lam,
                'als_p': als_p,
                'is_baseline_poly': False,  # 不使用多项式基线校正
                'baseline_points': 50,
                'baseline_poly': 3,
                'normalization_mode': normalization_mode,
                'global_transform_mode': global_transform_mode,
                'global_log_base': global_log_base,
                'global_log_offset': global_log_offset,
                'global_sqrt_offset': global_sqrt_offset,
                'is_quadratic_fit': plot_params.get('is_quadratic_fit', False),
                'quadratic_degree': plot_params.get('quadratic_degree', 2),
                'is_derivative': is_derivative,
                'global_y_offset': global_y_offset,
            }
            
            # 检查缓存（如果提供了文件路径）
            if file_path and hasattr(self, 'parent') and hasattr(self.parent, 'plot_data_cache'):
                cached_data = self.parent.plot_data_cache.get_preprocess_data(file_path, preprocess_params)
                if cached_data is not None:
                    x_cached, y_cached = cached_data
                    # 检查X轴是否匹配（简单检查长度）
                    if len(x_cached) == len(x):
                        return y_cached
            
            # 使用统一预处理函数
            y_processed = DataPreProcessor.preprocess_spectrum(x, y, preprocess_params)
            
            # QC检查（统一函数内部已处理，但这里需要返回None如果失败）
            if qc_enabled and np.max(y_processed) < qc_threshold:
                return None
            
            # 缓存结果（如果提供了文件路径）
            if file_path and hasattr(self, 'parent') and hasattr(self.parent, 'plot_data_cache'):
                self.parent.plot_data_cache.cache_preprocess_data(file_path, preprocess_params, (x.copy(), y_processed.copy()))
            
            return y_processed

        # ==========================================
        # A. 预处理所有数据（对照组+组内数据），归一化前处理
        # ==========================================
        max_y_value = -np.inf 
        min_y_value = np.inf
        all_data_before_norm = []
        
        # 准备预处理参数（用于统一预处理函数）
        preprocess_params = {
            'qc_enabled': qc_enabled,
            'qc_threshold': qc_threshold,
            'is_be_correction': is_be_correction,
            'be_temp': be_temp,
            'is_smoothing': is_smoothing,
            'smoothing_window': smoothing_window,
            'smoothing_poly': smoothing_poly,
            'is_baseline_als': is_baseline_als,
            'als_lam': als_lam,
            'als_p': als_p,
            'is_baseline_poly': False,
            'baseline_points': baseline_points,
            'baseline_poly': baseline_poly,
            'normalization_mode': 'None',  # 归一化在后面统一处理
            'global_transform_mode': global_transform_mode,
            'global_log_base': global_log_base_text,
            'global_log_offset': global_log_offset,
            'global_sqrt_offset': global_sqrt_offset,
            'is_quadratic_fit': plot_params.get('is_quadratic_fit', False),
            'quadratic_degree': plot_params.get('quadratic_degree', 2),
            'is_derivative': False,  # 二次导数在归一化后处理
            'global_y_offset': 0.0,  # Y轴偏移在归一化后处理
        }
        
        control_data_before_norm = []
        for i, control_data in enumerate(control_data_list):
            x_c = control_data['df']['Wavenumber'].values
            y_c = control_data['df']['Intensity'].values
            
            # 使用统一预处理函数（归一化前）
            temp_y = DataPreProcessor.preprocess_spectrum(x_c, y_c, preprocess_params)
            
            # QC检查
            if qc_enabled and (temp_y is None or np.max(temp_y) < qc_threshold):
                continue
            
            base_name = os.path.splitext(control_data['filename'])[0]
            control_data_before_norm.append({
                'x': x_c,
                'y': temp_y,
                'base_name': base_name,
                'label': control_data['label'],
                'type': 'control',
                'index': i
            })
            all_data_before_norm.append(temp_y)
        
        group_data_before_norm = []
        for file_path, x_data, y_data in grouped_files_data:
            # 使用统一预处理函数（归一化前）
            y_proc = DataPreProcessor.preprocess_spectrum(x_data, y_data, preprocess_params)
            
            # QC检查
            if qc_enabled and (y_proc is None or np.max(y_proc) < qc_threshold):
                continue
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            group_data_before_norm.append({
                'x': x_data,
                'y': y_proc,
                'base_name': base_name,
                'file_path': file_path,
                'type': 'group'
            })
            all_data_before_norm.append(y_proc)
        
        # 3. 一起归一化（如果启用）
        if normalization_mode != 'none' and all_data_before_norm:
            all_y_array = np.array(all_data_before_norm)  # (n_samples, n_features)
            
            if normalization_mode == 'max':
                max_vals = np.max(all_y_array, axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1
                all_y_array = all_y_array / max_vals
            elif normalization_mode == 'area':
                areas = np.trapezoid(all_y_array, axis=1)
                areas = areas[:, np.newaxis]
                areas[areas == 0] = 1
                all_y_array = all_y_array / areas
            elif normalization_mode == 'snv':
                means = np.mean(all_y_array, axis=1, keepdims=True)
                stds = np.std(all_y_array, axis=1, keepdims=True)
                stds[stds == 0] = 1
                all_y_array = (all_y_array - means) / stds
            
            idx = 0
            for item in control_data_before_norm:
                item['y'] = all_y_array[idx]
                idx += 1
            for item in group_data_before_norm:
                item['y'] = all_y_array[idx]
                idx += 1
        
        # ==========================================
        # B. 处理对照组（归一化后）
        # ==========================================
        control_plot_data = []
        for item in control_data_before_norm:
            x_c = item['x']
            temp_y = item['y']
            base_name = item['base_name']
            i = item['index']
            
            ind_params = individual_y_params.get(base_name, {'scale': 1.0, 'offset': 0.0, 'transform': 'none', 'transform_params': {}})
            
            if global_transform_mode == '对数变换 (Log)':
                temp_y = DataPreProcessor.apply_log_transform(temp_y, base=global_log_base, offset=global_log_offset)
            elif global_transform_mode == '平方根变换 (Sqrt)':
                temp_y = DataPreProcessor.apply_sqrt_transform(temp_y, offset=global_sqrt_offset)
            
            transform_mode = ind_params.get('transform', 'none')
            transform_params = ind_params.get('transform_params', {})
            
            if transform_mode == 'log':
                temp_y = DataPreProcessor.apply_log_transform(temp_y, 
                    base=transform_params.get('base', 10),
                    offset=transform_params.get('offset', 1.0))
            elif transform_mode == 'sqrt':
                temp_y = DataPreProcessor.apply_sqrt_transform(temp_y,
                    offset=transform_params.get('offset', 0.0))
            
            temp_y = temp_y * global_scale_factor * ind_params['scale']
            
            if is_derivative:
                d1 = np.gradient(temp_y, x_c)
                temp_y = np.gradient(d1, x_c)
            
            temp_y = temp_y + global_y_offset
            
            # 应用独立偏移（如果启用谱线扫描）
            spectrum_scan_enabled = plot_params.get('spectrum_scan_enabled', False)
            individual_offsets = plot_params.get('individual_offsets', {})
            custom_offset = 0.0
            if spectrum_scan_enabled and individual_offsets:
                # 从individual_offsets中获取该谱线的偏移
                label = item.get('label', '')
                if label in individual_offsets:
                    custom_offset = individual_offsets[label]
                # 如果没有找到标签匹配，尝试使用索引
                elif str(i) in individual_offsets:
                    custom_offset = individual_offsets[str(i)]
            
            final_y = temp_y + ind_params['offset'] + (i * global_stack_offset) + custom_offset 
            
            file_colors = plot_params.get('file_colors', {})
            if base_name in file_colors:
                color = file_colors[base_name]
            else:
                color = custom_colors[i % len(custom_colors)]
            
            label = item['label'] + " (Ref)"
            control_plot_data.append((x_c, final_y, label, color))
            
            if plot_style == 'line':
                ax.plot(x_c, final_y, label=label, color=color, linestyle='--', linewidth=line_width, alpha=0.7)
            else:  # scatter
                ax.plot(x_c, final_y, label=label, color=color, marker='.', linestyle='', markersize=line_width*3, alpha=0.7)

            self.current_plot_data[base_name] = {'x': x_c, 'y': final_y, 'label': label, 'color': color, 'type': 'Ref'}
            
            max_y_value = max(max_y_value, np.max(final_y))
            min_y_value = min(min_y_value, np.min(final_y))

        # ==========================================
        # C. 处理分组数据（归一化后）
        # ==========================================
        processed_group_data = []
        for item in group_data_before_norm:
            x_data = item['x']
            y_clean = item['y']
            base_name = item['base_name']
            file_path = item['file_path']
            
            label = plot_params['legend_names'].get(base_name, base_name)
            ind_params = individual_y_params.get(base_name, {'scale': 1.0, 'offset': 0.0, 'transform': 'none', 'transform_params': {}})
            
            y_transformed = y_clean.copy()
            if global_transform_mode == '对数变换 (Log)':
                y_transformed = DataPreProcessor.apply_log_transform(y_transformed, base=global_log_base, offset=global_log_offset)
            elif global_transform_mode == '平方根变换 (Sqrt)':
                y_transformed = DataPreProcessor.apply_sqrt_transform(y_transformed, offset=global_sqrt_offset)
            
            transform_mode = ind_params.get('transform', 'none')
            transform_params = ind_params.get('transform_params', {})
            
            if transform_mode == 'log':
                y_transformed = DataPreProcessor.apply_log_transform(y_transformed,
                    base=transform_params.get('base', 10),
                    offset=transform_params.get('offset', 1.0))
            elif transform_mode == 'sqrt':
                y_transformed = DataPreProcessor.apply_sqrt_transform(y_transformed,
                    offset=transform_params.get('offset', 0.0))
            
            processed_group_data.append({
                'x': x_data,
                'y_raw_processed': y_transformed,
                'ind_scale': ind_params['scale'],
                'ind_offset': ind_params['offset'],
                'label': label,
                'file_path': file_path,
                'base_name': base_name
            })
            
        if not processed_group_data and not control_data_list:
            ax.text(0.5, 0.5, "No valid data (Check QC threshold / X-range)", transform=ax.transAxes, ha='center')
            self.canvas.draw()
            return

        # ==========================================
        # C. 根据模式绘图
        # ==========================================
        current_plot_index = len(control_data_list)

        if plot_mode == 'Mean + Shadow' and processed_group_data:
            common_x = processed_group_data[0]['x']
            all_y = []
            for item in processed_group_data:
                y_scaled = item['y_raw_processed'] * item['ind_scale']
                all_y.append(y_scaled)
            
            all_y = np.array(all_y)
            mean_y = np.mean(all_y, axis=0)
            std_y = np.std(all_y, axis=0)
            
            mean_y *= global_scale_factor
            std_y *= global_scale_factor
            
            if is_derivative:
                d1 = np.gradient(mean_y, common_x)
                mean_y = np.gradient(d1, common_x)
                std_y = None
            
            mean_y = mean_y + global_y_offset
            
            color = custom_colors[current_plot_index % len(custom_colors)]
            
            rename_map = plot_params.get('legend_names', {})
            base_name = self.group_name
            if base_name in rename_map and rename_map[base_name]:
                base_display_name = rename_map[base_name]
            else:
                base_display_name = base_name
            
            mean_label_key = f"{base_name} Mean"
            std_label_key = f"{base_name} Std Dev"
            
            if mean_label_key in rename_map and rename_map[mean_label_key]:
                mean_label = rename_map[mean_label_key]
            else:
                mean_label = f"{base_display_name} Mean"
            
            if std_label_key in rename_map and rename_map[std_label_key]:
                std_label = rename_map[std_label_key]
            else:
                std_label = f"{base_display_name} Std Dev"
            
            group_color_params = plot_params.get('group_colors', {})
            if self.group_name in group_color_params:
                color = group_color_params[self.group_name]
            else:
                color = custom_colors[current_plot_index % len(custom_colors)]
            
            if is_derivative:
                ax.plot(common_x, mean_y, color=color, linewidth=line_width, label=mean_label)
            else:
                ax.plot(common_x, mean_y, color=color, linewidth=line_width, label=mean_label)
                if std_y is not None:
                    # 确保 alpha 值在 0-1 范围内
                    safe_alpha = max(0.0, min(1.0, shadow_alpha))
                    ax.fill_between(common_x, mean_y - std_y, mean_y + std_y, color=color, alpha=safe_alpha, label=std_label)
            
            self.current_plot_data[self.group_name + "_Mean"] = {'x': common_x, 'y': mean_y, 'label': f"{self.group_name} Mean", 'color': color, 'type': 'Mean'}
            
            if plot_params.get('peak_detection_enabled', False) and not is_derivative:
                self.detect_and_plot_peaks(ax, common_x, mean_y, mean_y, plot_params, color=color)
            
            if is_derivative:
                max_y_value = max(max_y_value, np.max(mean_y))
                min_y_value = min(min_y_value, np.min(mean_y))
            else:
                max_y_value = max(max_y_value, np.max(mean_y + std_y))
                min_y_value = min(min_y_value, np.min(mean_y - std_y))

        else:
            for i, item in enumerate(processed_group_data):
                y_val = item['y_raw_processed'] * global_scale_factor * item['ind_scale']
                
                if is_derivative:
                    d1 = np.gradient(y_val, item['x'])
                    y_val = np.gradient(d1, item['x'])
                
                y_val = y_val + global_y_offset
                
                stack_idx = i + current_plot_index
                # 应用独立偏移（如果启用谱线扫描）
                spectrum_scan_enabled = plot_params.get('spectrum_scan_enabled', False)
                individual_offsets = plot_params.get('individual_offsets', {})
                custom_offset = 0.0
                if spectrum_scan_enabled and individual_offsets:
                    # 从individual_offsets中获取该谱线的偏移
                    label = item.get('label', '')
                    if label in individual_offsets:
                        custom_offset = individual_offsets[label]
                    # 如果没有找到标签匹配，尝试使用索引
                    elif str(stack_idx) in individual_offsets:
                        custom_offset = individual_offsets[str(stack_idx)]
                
                y_final = y_val + item['ind_offset'] + (stack_idx * global_stack_offset) + custom_offset
                
                base_name = item.get('base_name', os.path.splitext(os.path.basename(item.get('file_path', '')))[0] if 'file_path' in item else item.get('label', ''))
                
                file_colors = plot_params.get('file_colors', {})
                if base_name in file_colors:
                    color = file_colors[base_name]
                else:
                    color = custom_colors[stack_idx % len(custom_colors)]
                
                # 保存 final_y 和 color 到 item 中，供峰值匹配使用
                item['final_y'] = y_final
                item['color'] = color
                
                if plot_style == 'line':
                    ax.plot(item['x'], y_final, label=item['label'], color=color, linewidth=line_width, linestyle=line_style)
                else:  # scatter
                    ax.plot(item['x'], y_final, label=item['label'], color=color, marker='.', linestyle='', markersize=line_width*3)


                if plot_params.get('peak_detection_enabled', False) and not is_derivative:
                    y_detect = y_val
                    self.detect_and_plot_peaks(ax, item['x'], y_detect, y_final, plot_params, color)
                    
                self.current_plot_data[item['label']] = {'x': item['x'], 'y': y_final, 'label': item['label'], 'color': color, 'type': 'Individual'}
                
                max_y_value = max(max_y_value, np.max(y_final))
                min_y_value = min(min_y_value, np.min(y_final))

        # --- 6. 坐标轴设置 ---
        if x_axis_invert:
            ax.invert_xaxis()
            
        aspect_ratio = plot_params.get('aspect_ratio', 0.0)
        if aspect_ratio > 0:
            ax.set_box_aspect(aspect_ratio) 
        else:
            ax.set_aspect('auto')

        if is_zoomed:
            ax.set_xlim(current_xlim)
            ax.set_ylim(current_ylim)
        else:
            if max_y_value != -np.inf and min_y_value != np.inf:
                y_range = max_y_value - min_y_value
                new_ylim = (min_y_value - y_range * 0.05, max_y_value + y_range * 0.05)
                ax.set_ylim(new_ylim[0], new_ylim[1])
            
            self.canvas.default_xlim = ax.get_xlim()
            self.canvas.default_ylim = ax.get_ylim()

        vertical_lines = plot_params.get('vertical_lines', [])
        vertical_line_color = plot_params.get('vertical_line_color', 'gray')
        vertical_line_width = plot_params.get('vertical_line_width', 0.8)
        vertical_line_style = plot_params.get('vertical_line_style', ':')
        vertical_line_alpha = plot_params.get('vertical_line_alpha', 0.7)
        
        for line_x in vertical_lines:
            ax.axvline(x=line_x, color=vertical_line_color, linestyle=vertical_line_style, 
                      linewidth=vertical_line_width, alpha=vertical_line_alpha)

        # 绘制RRUFF光谱和参考线
        rruff_spectra = plot_params.get('rruff_spectra', [])
        if rruff_spectra:
            from scipy.interpolate import interp1d
            
            # 获取当前数据的X轴范围（用于插值对齐）
            # 从processed_group_data或control_plot_data获取X轴
            ref_x_data = None
            if processed_group_data:
                ref_x_data = processed_group_data[0]['x']
            elif control_plot_data:
                ref_x_data = control_plot_data[0][0]  # control_plot_data是(x, y, label, color)元组
            
            if ref_x_data is None:
                # 如果没有数据，使用当前axes的X轴范围
                xlim = ax.get_xlim()
                ref_x_data = np.linspace(xlim[0], xlim[1], 1000)
            
            current_x_min = ref_x_data.min()
            current_x_max = ref_x_data.max()
            
            # 获取堆叠偏移和样式参数
            rruff_color_index = len(processed_group_data) if processed_group_data else (len(control_data_list) if control_data_list else 0)
            rruff_colors = plot_params.get('custom_colors', ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal', 'darkred'])
            
            for rruff_idx, rruff_data in enumerate(rruff_spectra):
                rruff_x = rruff_data['x']
                rruff_y = rruff_data['y']
                rruff_name = rruff_data['name']
                matches = rruff_data.get('matches', [])
                
                # 插值对齐到当前X轴
                if len(rruff_x) > 1:
                    # 确定插值范围（取交集）
                    interp_x_min = max(current_x_min, rruff_x.min())
                    interp_x_max = min(current_x_max, rruff_x.max())
                    
                    if interp_x_min < interp_x_max:
                        # 创建插值函数
                        f_interp = interp1d(rruff_x, rruff_y, kind='linear', fill_value=0, bounds_error=False)
                        
                        # 使用参考X轴进行插值
                        mask = (ref_x_data >= interp_x_min) & (ref_x_data <= interp_x_max)
                        interp_x = ref_x_data[mask]
                        interp_y = f_interp(interp_x)
                        
                        if len(interp_x) == 0:
                            continue
                        
                        # 应用堆叠偏移
                        # 检查是否是组合匹配的物相
                        is_combination_phase = rruff_data.get('is_combination_phase', False)
                        combination_stack_offset = rruff_data.get('stack_offset', 0.0)
                        
                        rruff_ref_line_offset = plot_params.get('rruff_ref_line_offset', 0.0)
                        stack_idx = rruff_color_index + rruff_idx
                        
                        if is_combination_phase:
                            # 组合匹配的物相：使用预计算的堆叠偏移
                            rruff_y_final = interp_y + combination_stack_offset
                        elif rruff_ref_line_offset != 0.0:
                            # 批量绘图窗口：使用参考线偏移来分离不同RRUFF光谱
                            rruff_y_final = interp_y + (rruff_idx * rruff_ref_line_offset)
                        else:
                            # 主菜单：使用正常的堆叠偏移
                            rruff_y_final = interp_y + (stack_idx * global_stack_offset)
                        
                        # 选择颜色
                        rruff_color = rruff_colors[stack_idx % len(rruff_colors)]
                        
                        # 更新Y轴范围以包含RRUFF光谱
                        if len(rruff_y_final) > 0:
                            max_y_value = max(max_y_value, np.max(rruff_y_final))
                            min_y_value = min(min_y_value, np.min(rruff_y_final))
                        
                        # 绘制RRUFF光谱（使用实线）
                        if plot_style == 'line':
                            ax.plot(interp_x, rruff_y_final, label=f"RRUFF: {rruff_name}", 
                                   color=rruff_color, linewidth=line_width, linestyle='-', alpha=0.7)
                        else:  # scatter
                            ax.plot(interp_x, rruff_y_final, label=f"RRUFF: {rruff_name}", 
                                   color=rruff_color, marker='.', linestyle='', markersize=line_width*3, alpha=0.7)
                        
                        # 绘制参考线连接匹配的峰值（使用匹配线样式）
                        match_line_color = plot_params.get('match_line_color', 'red')
                        match_line_width = plot_params.get('match_line_width', 1.0)
                        match_line_style = plot_params.get('match_line_style', '-')
                        match_line_alpha = plot_params.get('match_line_alpha', 0.8)
                        # 默认启用参考线，除非明确禁用
                        rruff_ref_lines_enabled = plot_params.get('rruff_ref_lines_enabled', True)
                        if matches and rruff_ref_lines_enabled:
                            # 使用匹配线样式参数（而不是垂直参考线样式）
                            ref_line_color = match_line_color  # 使用匹配线颜色
                            ref_line_style = match_line_style  # 使用匹配线样式
                            ref_line_width = match_line_width  # 使用匹配线宽度
                            ref_line_alpha = match_line_alpha  # 使用匹配线透明度
                            
                            # 获取当前光谱的峰值位置
                            data_items = processed_group_data if processed_group_data else []
                            if not data_items and control_plot_data:
                                # 如果没有processed_group_data，使用control_plot_data
                                for x_c, y_c, label_c, color_c in control_plot_data:
                                    # 检测峰值（使用与主菜单一致的峰值检测参数）
                                    from scipy.signal import find_peaks
                                    
                                    # 使用主菜单的峰值检测参数
                                    peak_height = plot_params.get('peak_height_threshold', 0.0)
                                    peak_distance = plot_params.get('peak_distance_min', 10)
                                    peak_prominence = plot_params.get('peak_prominence', None)
                                    
                                    # 计算智能阈值
                                    y_max = np.max(y_c) if len(y_c) > 0 else 0
                                    y_min = np.min(y_c) if len(y_c) > 0 else 0
                                    y_range = y_max - y_min
                                    
                                    peak_kwargs = {}
                                    if peak_height == 0 or (peak_height > y_range and y_range > 0):
                                        if y_max > 0:
                                            peak_height = y_max * 0.02
                                        else:
                                            peak_height = 0
                                    if peak_height > 0 or peak_height < 0:
                                        peak_kwargs['height'] = peak_height
                                    
                                    if peak_distance == 0 or peak_distance > len(y_c) * 0.5:
                                        peak_distance = max(1, int(len(y_c) * 0.02))
                                    if peak_distance > 0:
                                        peak_kwargs['distance'] = peak_distance
                                    
                                    if peak_prominence is not None and peak_prominence > 0:
                                        if peak_prominence > y_range and y_range > 0:
                                            peak_prominence = y_range * 0.02
                                        peak_kwargs['prominence'] = peak_prominence
                                    
                                    try:
                                        peaks, _ = find_peaks(y_c, **peak_kwargs)
                                    except:
                                        # 如果参数错误，使用默认参数
                                        peaks, _ = find_peaks(y_c, 
                                                            height=y_max * 0.02 if y_max > 0 else 0,
                                                            distance=max(1, int(len(y_c) * 0.02)))
                                    
                                    query_peaks_x = x_c[peaks] if len(peaks) > 0 else np.array([])
                                    
                                    # 绘制匹配的参考线
                                    for match in matches:
                                        query_peak, lib_peak, distance = match
                                        # 直接使用匹配结果中的峰值位置，不需要再查找
                                        # 但需要找到对应的Y坐标
                                        query_y_idx = np.argmin(np.abs(x_c - query_peak))
                                        query_y = y_c[query_y_idx]
                                        
                                        lib_y_idx = np.argmin(np.abs(interp_x - lib_peak))
                                        lib_y = rruff_y_final[lib_y_idx] if lib_y_idx < len(rruff_y_final) else rruff_y_final[-1]
                                        
                                        # 绘制参考线（使用RRUFF光谱颜色）
                                        ax.plot([query_peak, lib_peak], [query_y, lib_y], 
                                               color=ref_line_color, linestyle=ref_line_style, 
                                               linewidth=ref_line_width, alpha=ref_line_alpha)
                                    break
                            else:
                                # 使用processed_group_data
                                for item in data_items:
                                    # 检测当前光谱的峰值（使用与主菜单一致的峰值检测参数）
                                    from scipy.signal import find_peaks
                                    y_detect = item['y_raw_processed']
                                    
                                    # 使用主菜单的峰值检测参数
                                    peak_height = plot_params.get('peak_height_threshold', 0.0)
                                    peak_distance = plot_params.get('peak_distance_min', 10)
                                    peak_prominence = plot_params.get('peak_prominence', None)
                                    
                                    # 计算智能阈值
                                    y_max = np.max(y_detect) if len(y_detect) > 0 else 0
                                    y_min = np.min(y_detect) if len(y_detect) > 0 else 0
                                    y_range = y_max - y_min
                                    
                                    peak_kwargs = {}
                                    if peak_height == 0 or (peak_height > y_range and y_range > 0):
                                        if y_max > 0:
                                            peak_height = y_max * 0.02
                                        else:
                                            peak_height = 0
                                    if peak_height > 0 or peak_height < 0:
                                        peak_kwargs['height'] = peak_height
                                    
                                    if peak_distance == 0 or peak_distance > len(y_detect) * 0.5:
                                        peak_distance = max(1, int(len(y_detect) * 0.02))
                                    if peak_distance > 0:
                                        peak_kwargs['distance'] = peak_distance
                                    
                                    if peak_prominence is not None and peak_prominence > 0:
                                        if peak_prominence > y_range and y_range > 0:
                                            peak_prominence = y_range * 0.02
                                        peak_kwargs['prominence'] = peak_prominence
                                    
                                    try:
                                        peaks, _ = find_peaks(y_detect, **peak_kwargs)
                                    except:
                                        # 如果参数错误，使用默认参数
                                        peaks, _ = find_peaks(y_detect, 
                                                            height=y_max * 0.02 if y_max > 0 else 0,
                                                            distance=max(1, int(len(y_detect) * 0.02)))
                                    
                                    query_peaks_x = item['x'][peaks] if len(peaks) > 0 else np.array([])
                                    
                                    # 绘制匹配的参考线
                                    for match in matches:
                                        query_peak, lib_peak, distance = match
                                        # 直接使用匹配结果中的峰值位置
                                        # 获取Y坐标（考虑所有偏移和缩放）
                                        query_y_idx = np.argmin(np.abs(item['x'] - query_peak))
                                        y_val = item['y_raw_processed'][query_y_idx] * global_scale_factor * item['ind_scale']
                                        if is_derivative:
                                            # 对于导数，需要重新计算
                                            y_val = item['y_raw_processed'][query_y_idx]
                                        y_val = y_val + global_y_offset
                                        stack_idx_item = current_plot_index + data_items.index(item)
                                        query_y = y_val + item['ind_offset'] + (stack_idx_item * global_stack_offset)
                                        
                                        lib_y_idx = np.argmin(np.abs(interp_x - lib_peak))
                                        lib_y = rruff_y_final[lib_y_idx] if lib_y_idx < len(rruff_y_final) else rruff_y_final[-1]
                                        
                                        # 绘制参考线（使用RRUFF光谱颜色）
                                        ax.plot([query_peak, lib_peak], [query_y, lib_y], 
                                               color=ref_line_color, linestyle=ref_line_style, 
                                               linewidth=ref_line_width, alpha=ref_line_alpha)
                                    
                                    # 只处理第一个数据（如果有多个数据，只连接第一个）
                                    break
            
            # 在绘制完所有RRUFF光谱后，重新调整Y轴范围（如果未缩放）
            if not is_zoomed and rruff_spectra:
                if max_y_value != -np.inf and min_y_value != np.inf:
                    y_range = max_y_value - min_y_value
                    if y_range > 0:  # 只有当范围有效时才更新
                        new_ylim = (min_y_value - y_range * 0.05, max_y_value + y_range * 0.05)
                        ax.set_ylim(new_ylim[0], new_ylim[1])
                        # 更新默认Y轴范围
                        self.canvas.default_ylim = ax.get_ylim()

        ylabel_final = "2nd Derivative" if is_derivative else plot_params.get('ylabel_text', 'Intensity')
        if is_be_correction:
             ylabel_final = f"BE Corrected {ylabel_final} @ {be_temp}K"

        xlabel_fontsize = plot_params.get('xlabel_fontsize', axis_title_fontsize)
        xlabel_pad = plot_params.get('xlabel_pad', 10.0)
        xlabel_show = plot_params.get('xlabel_show', True)
        
        if xlabel_show:
            ax.set_xlabel(plot_params.get('xlabel_text', r"Wavenumber ($\mathrm{cm^{-1}}$)"), fontsize=xlabel_fontsize, labelpad=xlabel_pad, fontfamily=current_font)
        
        ylabel_fontsize = plot_params.get('ylabel_fontsize', axis_title_fontsize)
        ylabel_pad = plot_params.get('ylabel_pad', 10.0)
        ylabel_show = plot_params.get('ylabel_show', True)
        
        if ylabel_show:
            ax.set_ylabel(ylabel_final, fontsize=ylabel_fontsize, labelpad=ylabel_pad, fontfamily=current_font)
        
        # 是否隐藏 X/Y 轴数值
        show_x_values = plot_params.get('show_x_values', True)
        if not show_y_values:
            ax.set_yticks([])
        if not show_x_values:
            ax.set_xticks([])
        
        tick_direction = plot_params['tick_direction']
        tick_len_major = plot_params['tick_len_major']
        tick_len_minor = plot_params['tick_len_minor']
        tick_width = plot_params['tick_width']
        
        ax.tick_params(labelsize=tick_label_fontsize, direction=tick_direction, width=tick_width)
        ax.tick_params(which='major', length=tick_len_major)
        ax.tick_params(which='minor', length=tick_len_minor)
        
        for side in ['top', 'right', 'left', 'bottom']:
            if side in plot_params['border_sides']:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(plot_params['border_linewidth'])
            else:
                ax.spines[side].set_visible(False)
                
        # ==========================================
        # D. 应用峰值匹配（如果启用）
        # ==========================================
        peak_matching_enabled = plot_params.get('peak_matching_enabled', False)
        if peak_matching_enabled:
            try:
                from src.core.peak_matcher import PeakMatcher
                from scipy.interpolate import interp1d
                
                peak_matcher = PeakMatcher(tolerance=plot_params.get('peak_matching_tolerance', 5.0))
                peak_matching_mode = plot_params.get('peak_matching_mode', 'all_matched')
                peak_matching_reference_index = plot_params.get('peak_matching_reference_index', -1)
                
                # 获取峰值匹配样式参数
                marker_shape = plot_params.get('peak_matching_marker_shape', 'v')
                marker_size = plot_params.get('peak_matching_marker_size', 8.0)
                marker_distance = plot_params.get('peak_matching_marker_distance', 0.0)
                marker_rotation = plot_params.get('peak_matching_marker_rotation', 0.0)
                show_connection_lines = plot_params.get('peak_matching_show_connection_lines', False)
                use_spectrum_color_for_connection = plot_params.get('peak_matching_use_spectrum_color_for_connection', True)
                connection_line_color = plot_params.get('peak_matching_connection_line_color', 'red')
                connection_line_width = plot_params.get('peak_matching_connection_line_width', 1.0)
                connection_line_style = plot_params.get('peak_matching_connection_line_style', '-')
                connection_line_alpha = plot_params.get('peak_matching_connection_line_alpha', 0.8)
                show_peak_labels = plot_params.get('peak_matching_show_peak_labels', False)
                label_fontsize = plot_params.get('peak_matching_label_fontsize', 10.0)
                label_color = plot_params.get('peak_matching_label_color', 'black')
                label_rotation = plot_params.get('peak_matching_label_rotation', 0.0)
                label_distance = plot_params.get('peak_matching_label_distance', 5.0)
                
                # 准备光谱数据列表（包括对照组和组内数据）
                spectra_list = []
                
                # 添加对照组数据
                for x_c, y_c, label_c, color_c in control_plot_data:
                    spectra_list.append({
                        'x': x_c,
                        'y': y_c,
                        'label': label_c,
                        'color': color_c
                    })
                
                # 添加组内数据
                for idx, item in enumerate(processed_group_data):
                    x_data = item['x']
                    # 计算 final_y（如果不存在）
                    if 'final_y' not in item:
                        y_val = item['y_raw_processed'] * global_scale_factor * item.get('ind_scale', 1.0)
                        if is_derivative:
                            d1 = np.gradient(y_val, x_data)
                            y_val = np.gradient(d1, x_data)
                        y_val = y_val + global_y_offset
                        stack_idx = len(control_plot_data) + idx
                        y_final = y_val + item.get('ind_offset', 0.0) + (stack_idx * global_stack_offset)
                    else:
                        y_final = item['final_y']
                    
                    label = item['label']
                    # 获取颜色
                    base_name = item.get('base_name', '')
                    file_colors = plot_params.get('file_colors', {})
                    if base_name in file_colors:
                        color = file_colors[base_name]
                    else:
                        stack_idx = len(control_plot_data) + idx
                        color = custom_colors[stack_idx % len(custom_colors)]
                    
                    spectra_list.append({
                        'x': x_data,
                        'y': y_final,
                        'label': label,
                        'color': color
                    })
                
                # 执行峰值匹配
                if len(spectra_list) > 1:
                    match_result = peak_matcher.match_multiple_spectra(
                        spectra_list,
                        reference_index=peak_matching_reference_index,
                        mode=peak_matching_mode
                    )
                    
                    # 绘制匹配结果
                    matches = match_result.get('matches', {})
                    for idx, match_info in matches.items():
                        if idx < len(spectra_list):
                            spectrum = spectra_list[idx]
                            color = spectrum.get('color', 'blue')
                            positions = match_info.get('positions', np.array([]))
                            
                            if len(positions) > 0:
                                # 获取对应的Y值（需要插值）
                                x_data = spectrum.get('x', np.array([]))
                                y_data = spectrum.get('y', np.array([]))
                                
                                if len(x_data) > 0 and len(y_data) > 0:
                                    interp_func = interp1d(x_data, y_data, 
                                                         kind='linear', 
                                                         bounds_error=False, 
                                                         fill_value=0.0)
                                    match_y = interp_func(positions)
                                    
                                    # 应用标记距离偏移
                                    match_y_with_offset = match_y + marker_distance
                                    
                                    # 绘制连接线（如果启用）
                                    if show_connection_lines and idx > 0:
                                        # 找到参考光谱的匹配位置
                                        ref_idx = peak_matching_reference_index if peak_matching_reference_index >= 0 else len(spectra_list) - 1
                                        if ref_idx < len(spectra_list) and ref_idx != idx:
                                            ref_spectrum = spectra_list[ref_idx]
                                            ref_x_data = ref_spectrum.get('x', np.array([]))
                                            ref_y_data = ref_spectrum.get('y', np.array([]))
                                            if len(ref_x_data) > 0 and len(ref_y_data) > 0:
                                                ref_interp_func = interp1d(ref_x_data, ref_y_data,
                                                                          kind='linear',
                                                                          bounds_error=False,
                                                                          fill_value=0.0)
                                                ref_match_y = ref_interp_func(positions)
                                                # 绘制连接线
                                                for pos, y1, y2 in zip(positions, match_y_with_offset, ref_match_y):
                                                    # 如果使用谱线颜色，则使用当前谱线的颜色；否则使用统一颜色
                                                    line_color = color if use_spectrum_color_for_connection else connection_line_color
                                                    ax.plot([pos, pos], [y1, y2],
                                                           color=line_color,
                                                           linewidth=connection_line_width,
                                                           linestyle=connection_line_style,
                                                           alpha=connection_line_alpha,
                                                           zorder=9,
                                                           label='_nolegend_')
                                    
                                    # 绘制匹配的峰值标记
                                    # 注意：matplotlib的plot不支持rotation，需要单独绘制每个标记
                                    if marker_rotation == 0.0:
                                        # 无旋转时直接使用plot
                                        ax.plot(positions, match_y_with_offset,
                                               marker=marker_shape,
                                               markersize=marker_size,
                                               color=color,
                                               linestyle='',
                                               zorder=10,
                                               alpha=0.7,
                                               label='_nolegend_')
                                    else:
                                        # 有旋转时需要单独绘制每个标记
                                        from matplotlib import transforms
                                        for pos, y_val in zip(positions, match_y_with_offset):
                                            t = transforms.Affine2D().rotate_deg(marker_rotation) + ax.transData
                                            ax.plot(pos, y_val,
                                                   marker=marker_shape,
                                                   markersize=marker_size,
                                                   color=color,
                                                   linestyle='',
                                                   zorder=10,
                                                   alpha=0.7,
                                                   transform=t,
                                                   label='_nolegend_')
                                    
                                    # 绘制峰值数字（如果启用）
                                    if show_peak_labels:
                                        for pos, y_val in zip(positions, match_y_with_offset):
                                            ax.text(pos, y_val + label_distance,
                                                   f'{pos:.1f}',
                                                   fontsize=label_fontsize,
                                                   color=label_color,
                                                   rotation=label_rotation,
                                                   ha='center',
                                                   va='bottom',
                                                   zorder=11,
                                                   label='_nolegend_')
            except Exception as e:
                print(f"峰值匹配失败: {e}")
                import traceback
                traceback.print_exc()
        
        # ==========================================
        # E. 应用谱线扫描（如果启用）
        # ==========================================
        spectrum_scan_enabled = plot_params.get('spectrum_scan_enabled', False)
        # 注意：谱线扫描的偏移已经在上面计算y_final时应用了
        # 这里保留代码结构，但实际偏移应用在y_final计算处
        if spectrum_scan_enabled:
            try:
                from src.core.spectrum_scanner import SpectrumScanner
                
                spectrum_scanner = SpectrumScanner()
                stack_offset = plot_params.get('stack_offset', 0.5)
                individual_offsets = plot_params.get('individual_offsets', {})
                custom_mappings = plot_params.get('custom_mappings', [])
                
                # 应用堆叠偏移（如果启用）
                if stack_offset != 0.0 and processed_group_data:
                    # 堆叠偏移已经在y_final计算时应用，这里不需要额外处理
                    pass
                
                # 应用自定义映射（如果提供）
                if custom_mappings and len(processed_group_data) > 1:
                    # 准备光谱数据用于映射
                    scan_spectra_list = []
                    for item in processed_group_data:
                        scan_spectra_list.append({
                            'x': item['x'],
                            'y': item['y_raw_processed'],
                            'label': item['label']
                        })
                    
                    # 使用谱线扫描对齐光谱
                    scanned_spectra = spectrum_scanner.scan_last_plot(scan_spectra_list)
                    aligned_spectra = spectrum_scanner.apply_mappings(
                        custom_mappings, interpolation=True
                    )
                    # 更新processed_group_data中的Y值
                    for i, item in enumerate(processed_group_data):
                        if i < len(aligned_spectra):
                            aligned_item = aligned_spectra[i]
                            # 更新原始处理后的Y值（偏移会在后面计算y_final时应用）
                            item['y_raw_processed'] = aligned_item.get('y', item['y_raw_processed'])
            except Exception as e:
                print(f"谱线扫描失败: {e}")
                import traceback
                traceback.print_exc()
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
            
        if show_legend:
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            if font_family != 'SimHei':
                legend_font.set_family(font_family)
            else:
                legend_font.set_family('sans-serif')
            legend_font.set_size(legend_fontsize)
            
            ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                     ncol=legend_ncol, columnspacing=legend_columnspacing, 
                     labelspacing=legend_labelspacing, handlelength=legend_handlelength)
            
        main_title_stripped = main_title_text.strip()
        main_title_fontsize = plot_params.get('main_title_fontsize', axis_title_fontsize)
        main_title_pad = plot_params.get('main_title_pad', 10.0)
        main_title_show = plot_params.get('main_title_show', True)
        
        if main_title_stripped != "" and main_title_show:
            final_title = main_title_stripped
            ax.set_title(
                final_title, 
                fontsize=main_title_fontsize, 
                fontfamily=current_font,
                pad=main_title_pad
            )
        
        # 使用subplots_adjust代替tight_layout以避免警告
        try:
            # 先尝试tight_layout，如果失败则使用subplots_adjust
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                self.canvas.figure.tight_layout(pad=1.0)
        except:
            # 如果tight_layout失败，使用subplots_adjust作为后备
            try:
                self.canvas.figure.subplots_adjust(
                    left=0.12, right=0.95, bottom=0.12, top=0.95
                )
            except:
                pass  # 如果都失败，继续执行
        
        self.canvas.draw()
        
        # 不自动显示窗口，保持窗口位置和可见性状态
        # 如果窗口已经存在，只更新绘图内容，不改变窗口状态
        # if not self.isVisible():
        #     self.show()
        
        self.update()


"""
统一的峰值检测辅助函数
所有窗口都应该使用这个模块进行峰值检测，避免代码重复
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from matplotlib.axes import Axes

from src.core.peak_matcher import PeakMatcher


def detect_and_plot_peaks(ax: Axes, x_data: np.ndarray, y_detect: np.ndarray, 
                         y_final: np.ndarray, plot_params: Dict[str, Any], 
                         color: str = 'blue') -> None:
    """
    统一的峰值检测和绘制函数
    
    Args:
        ax: matplotlib axes对象
        x_data: X轴数据（波数）
        y_detect: 用于检测的Y数据（去除偏移）
        y_final: 用于绘制的Y数据（包含偏移）
        plot_params: 绘图参数字典，应包含峰值检测相关参数
        color: 线条颜色（用于标记颜色默认值）
    """
    if not plot_params.get('peak_detection_enabled', False):
        return
    
    try:
        # 使用统一的峰值检测方法
        peak_matcher = PeakMatcher()
        peaks, properties = peak_matcher.detect_peaks(
            x_data, y_detect,
            height=plot_params.get('peak_height_threshold', 0.0),
            distance=plot_params.get('peak_distance_min', 10),
            prominence=plot_params.get('peak_prominence', None),
            width=plot_params.get('peak_width', None),
            wlen=plot_params.get('peak_wlen', None),
            rel_height=plot_params.get('peak_rel_height', None)
        )
        
        if len(peaks) > 0:
            # 获取标记样式参数
            peak_marker_shape = plot_params.get('peak_marker_shape', 'x')
            peak_marker_size = plot_params.get('peak_marker_size', 10)
            peak_marker_color = plot_params.get('peak_marker_color', None)
            # 如果未指定颜色，使用线条颜色
            if peak_marker_color is None or peak_marker_color == '':
                peak_marker_color = color
            
            # 绘制峰值标记
            ax.plot(x_data[peaks], y_final[peaks], peak_marker_shape, 
                   color=peak_marker_color, markersize=peak_marker_size)
            
            # 显示波数值
            if plot_params.get('peak_show_label', True):
                peak_x_coords = x_data[peaks]
                peak_y_coords = y_final[peaks]
                
                # 获取标签样式参数
                label_font = plot_params.get('peak_label_font', 'Times New Roman')
                label_size = plot_params.get('peak_label_size', 10)
                label_color = plot_params.get('peak_label_color', 'black')
                label_bold = plot_params.get('peak_label_bold', False)
                label_rotation = plot_params.get('peak_label_rotation', 0.0)
                
                # 构建字体属性
                font_props = {
                    'fontsize': label_size,
                    'color': label_color,
                    'fontfamily': label_font,
                    'ha': 'center',
                    'va': 'bottom'
                }
                if label_bold:
                    font_props['weight'] = 'bold'
                if label_rotation != 0:
                    font_props['rotation'] = label_rotation
                
                # 为每个峰值添加波数标签
                for px, py in zip(peak_x_coords, peak_y_coords):
                    # 格式化波数（保留1位小数）
                    wavenumber_str = f"{px:.1f}"
                    ax.text(px, py, wavenumber_str, **font_props)
    except Exception as e:
        # 如果峰值检测失败，打印错误信息以便调试
        print(f"波峰检测失败: {e}")
        import traceback
        traceback.print_exc()


def get_peak_detection_params_from_config(config) -> Dict[str, Any]:
    """
    从 PlotConfig 对象中提取峰值检测参数
    
    Args:
        config: PlotConfig 对象
        
    Returns:
        包含峰值检测参数的字典
    """
    pd = config.peak_detection
    return {
        'peak_detection_enabled': pd.enabled,
        'peak_height_threshold': pd.height_threshold,
        'peak_distance_min': pd.distance_min,
        'peak_prominence': pd.prominence,
        'peak_width': pd.width,
        'peak_wlen': pd.wlen,
        'peak_rel_height': pd.rel_height,
        'peak_show_label': pd.show_label,
        'peak_label_font': pd.label_font,
        'peak_label_size': pd.label_size,
        'peak_label_color': pd.label_color,
        'peak_label_bold': pd.label_bold,
        'peak_label_rotation': pd.label_rotation,
        'peak_marker_shape': pd.marker_shape,
        'peak_marker_size': pd.marker_size,
        'peak_marker_color': pd.marker_color,
    }


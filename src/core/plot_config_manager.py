"""
通用绘图配置管理器
统一管理所有绘图相关的通用设置，包括：
- 出版质量样式控制
- 峰值检测与匹配
- 谱线扫描与堆叠偏移
- 物理验证
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from PyQt6.QtCore import QSettings
import numpy as np
from scipy.signal import find_peaks


@dataclass
class PublicationStyleConfig:
    """出版质量样式配置"""
    # Figure
    fig_width: float = 10.0
    fig_height: float = 6.0
    fig_dpi: int = 300
    aspect_ratio: float = 0.6
    
    # Font
    font_family: str = 'Times New Roman'
    axis_title_fontsize: int = 20
    tick_label_fontsize: int = 16
    legend_fontsize: int = 10
    title_fontsize: int = 18
    
    # Lines
    line_width: float = 1.2
    line_style: str = '-'
    
    # Ticks
    tick_direction: str = 'in'
    tick_len_major: int = 8
    tick_len_minor: int = 4
    tick_width: float = 1.0
    tick_top: bool = True
    tick_bottom: bool = True
    tick_left: bool = True
    tick_right: bool = True
    
    # Grid
    show_grid: bool = True
    grid_alpha: float = 0.2
    shadow_alpha: float = 0.25
    
    # Spines
    spine_top: bool = True
    spine_bottom: bool = True
    spine_left: bool = True
    spine_right: bool = True
    spine_width: float = 2.0
    
    # Legend
    show_legend: bool = True
    legend_frame: bool = True
    legend_loc: str = 'best'
    legend_ncol: int = 1
    legend_columnspacing: float = 2.0
    legend_labelspacing: float = 0.5
    legend_handlelength: float = 2.0
    
    # 标题控制（新增）
    # X轴标题
    xlabel_text: str = r"Wavenumber ($\mathrm{cm^{-1}}$)"
    xlabel_show: bool = True
    xlabel_fontsize: int = 20
    xlabel_pad: float = 10.0
    
    # Y轴标题
    ylabel_text: str = "Intensity"
    ylabel_show: bool = True
    ylabel_fontsize: int = 20
    ylabel_pad: float = 10.0
    
    # 主标题
    title_text: str = ""
    title_show: bool = True
    title_fontsize: int = 18
    title_pad: float = 10.0
    
    # 坐标轴显示控制（新增）
    x_axis_invert: bool = False
    show_x_values: bool = True
    show_y_values: bool = True
    show_bottom_xaxis: bool = True  # 显示下X轴
    show_left_yaxis: bool = True  # 显示左Y轴
    show_top_xaxis: bool = False  # 显示上X轴
    show_right_yaxis: bool = False  # 显示右Y轴


@dataclass
class PeakDetectionConfig:
    """峰值检测配置"""
    enabled: bool = False
    height_threshold: float = 0.0
    distance_min: int = 10
    prominence: Optional[float] = None
    width: Optional[float] = None
    wlen: Optional[int] = None
    rel_height: Optional[float] = None
    
    # 峰值显示
    show_label: bool = True
    label_font: str = 'Arial'
    label_size: int = 10
    label_color: str = 'black'
    label_bold: bool = False
    label_rotation: float = 0.0
    marker_shape: str = 'v'
    marker_size: int = 8


@dataclass
class PeakMatchingConfig:
    """峰值匹配配置"""
    enabled: bool = False
    mode: str = 'all_matched'  # 'all_peaks', 'matched_only', 'all_matched', 'top_display'
    tolerance: float = 5.0  # cm^-1
    reference_index: int = -1  # -1表示最后一个光谱作为基准
    
    # 匹配模式说明：
    # 'all_peaks': 只显示最下面谱峰检测的所有峰值
    # 'matched_only': 只显示最下面匹配到的谱峰
    # 'all_matched': 显示所有谱线都匹配的谱峰
    # 'top_display': 在最上方谱线显示最下方谱峰匹配到的峰值以供观察
    
    # 标记样式
    marker_shape: str = 'v'  # 标记形状：'v', 'o', 's', '^', 'D', '*', '+', 'x'
    marker_size: float = 8.0  # 标记大小
    marker_distance: float = 0.0  # 标记离谱线的距离（Y轴偏移）
    marker_rotation: float = 0.0  # 标记旋转角度（度）
    
    # 谱线连接
    show_connection_lines: bool = False  # 是否显示连接匹配峰值的谱线
    use_spectrum_color_for_connection: bool = True  # 是否使用各自谱线颜色（True=使用谱线颜色，False=使用统一颜色）
    connection_line_color: str = 'red'  # 连接线颜色（仅在use_spectrum_color_for_connection=False时使用）
    connection_line_width: float = 1.0  # 连接线宽度
    connection_line_style: str = '-'  # 连接线样式：'-', '--', ':', '-.'
    connection_line_alpha: float = 0.8  # 连接线透明度
    
    # 峰值数字显示
    show_peak_labels: bool = False  # 是否显示峰值数字
    label_fontsize: float = 10.0  # 标签字体大小
    label_color: str = 'black'  # 标签颜色
    label_rotation: float = 0.0  # 标签旋转角度（度）
    label_distance: float = 5.0  # 标签离谱线的距离（像素）


@dataclass
class SpectrumScanConfig:
    """谱线扫描配置"""
    enabled: bool = False
    scan_last_plot: bool = True  # 扫描最后一次绘图的所有谱线
    custom_mappings: List[Tuple[int, int]] = field(default_factory=list)  # 指定匹配关系 [(source_idx, target_idx), ...]
    
    # 堆叠偏移
    stack_offset: float = 0.5
    individual_offsets: Dict[str, float] = field(default_factory=dict)  # 每根线的独立偏移


@dataclass
class PlotConfig:
    """完整的绘图配置"""
    publication_style: PublicationStyleConfig = field(default_factory=PublicationStyleConfig)
    peak_detection: PeakDetectionConfig = field(default_factory=PeakDetectionConfig)
    peak_matching: PeakMatchingConfig = field(default_factory=PeakMatchingConfig)
    spectrum_scan: SpectrumScanConfig = field(default_factory=SpectrumScanConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于保存和传递）"""
        return {
            'publication_style': {
                'fig_width': self.publication_style.fig_width,
                'fig_height': self.publication_style.fig_height,
                'fig_dpi': self.publication_style.fig_dpi,
                'aspect_ratio': self.publication_style.aspect_ratio,
                'font_family': self.publication_style.font_family,
                'axis_title_fontsize': self.publication_style.axis_title_fontsize,
                'tick_label_fontsize': self.publication_style.tick_label_fontsize,
                'legend_fontsize': self.publication_style.legend_fontsize,
                'title_fontsize': self.publication_style.title_fontsize,
                'line_width': self.publication_style.line_width,
                'line_style': self.publication_style.line_style,
                'tick_direction': self.publication_style.tick_direction,
                'tick_len_major': self.publication_style.tick_len_major,
                'tick_len_minor': self.publication_style.tick_len_minor,
                'tick_width': self.publication_style.tick_width,
                'show_grid': self.publication_style.show_grid,
                'grid_alpha': self.publication_style.grid_alpha,
                'shadow_alpha': self.publication_style.shadow_alpha,
                'spine_top': self.publication_style.spine_top,
                'spine_bottom': self.publication_style.spine_bottom,
                'spine_left': self.publication_style.spine_left,
                'spine_right': self.publication_style.spine_right,
                'spine_width': self.publication_style.spine_width,
                'show_legend': self.publication_style.show_legend,
                'legend_frame': self.publication_style.legend_frame,
                'legend_loc': self.publication_style.legend_loc,
                'legend_ncol': self.publication_style.legend_ncol,
                'legend_columnspacing': self.publication_style.legend_columnspacing,
                'legend_labelspacing': self.publication_style.legend_labelspacing,
                'legend_handlelength': self.publication_style.legend_handlelength,
                'xlabel_text': self.publication_style.xlabel_text,
                'xlabel_show': self.publication_style.xlabel_show,
                'xlabel_fontsize': self.publication_style.xlabel_fontsize,
                'xlabel_pad': self.publication_style.xlabel_pad,
                'ylabel_text': self.publication_style.ylabel_text,
                'ylabel_show': self.publication_style.ylabel_show,
                'ylabel_fontsize': self.publication_style.ylabel_fontsize,
                'ylabel_pad': self.publication_style.ylabel_pad,
                'title_text': self.publication_style.title_text,
                'title_show': self.publication_style.title_show,
                'title_fontsize': self.publication_style.title_fontsize,
                'title_pad': self.publication_style.title_pad,
                'x_axis_invert': self.publication_style.x_axis_invert,
                'show_x_values': self.publication_style.show_x_values,
                'show_y_values': self.publication_style.show_y_values,
            },
            'peak_detection': {
                'enabled': self.peak_detection.enabled,
                'height_threshold': self.peak_detection.height_threshold,
                'distance_min': self.peak_detection.distance_min,
                'prominence': self.peak_detection.prominence,
                'width': self.peak_detection.width,
                'wlen': self.peak_detection.wlen,
                'rel_height': self.peak_detection.rel_height,
                'show_label': self.peak_detection.show_label,
                'label_font': self.peak_detection.label_font,
                'label_size': self.peak_detection.label_size,
                'label_color': self.peak_detection.label_color,
                'label_bold': self.peak_detection.label_bold,
                'label_rotation': self.peak_detection.label_rotation,
                'marker_shape': self.peak_detection.marker_shape,
                'marker_size': self.peak_detection.marker_size,
            },
            'peak_matching': {
                'enabled': self.peak_matching.enabled,
                'mode': self.peak_matching.mode,
                'tolerance': self.peak_matching.tolerance,
                'reference_index': self.peak_matching.reference_index,
                'marker_shape': self.peak_matching.marker_shape,
                'marker_size': self.peak_matching.marker_size,
                'marker_distance': self.peak_matching.marker_distance,
                'marker_rotation': self.peak_matching.marker_rotation,
                'show_connection_lines': self.peak_matching.show_connection_lines,
                'use_spectrum_color_for_connection': self.peak_matching.use_spectrum_color_for_connection,
                'connection_line_color': self.peak_matching.connection_line_color,
                'connection_line_width': self.peak_matching.connection_line_width,
                'connection_line_style': self.peak_matching.connection_line_style,
                'connection_line_alpha': self.peak_matching.connection_line_alpha,
                'show_peak_labels': self.peak_matching.show_peak_labels,
                'label_fontsize': self.peak_matching.label_fontsize,
                'label_color': self.peak_matching.label_color,
                'label_rotation': self.peak_matching.label_rotation,
                'label_distance': self.peak_matching.label_distance,
            },
            'spectrum_scan': {
                'enabled': self.spectrum_scan.enabled,
                'scan_last_plot': self.spectrum_scan.scan_last_plot,
                'custom_mappings': self.spectrum_scan.custom_mappings,
                'stack_offset': self.spectrum_scan.stack_offset,
                'individual_offsets': self.spectrum_scan.individual_offsets,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlotConfig':
        """从字典创建配置对象"""
        config = cls()
        
        if 'publication_style' in data:
            ps = data['publication_style']
            config.publication_style = PublicationStyleConfig(**{k: v for k, v in ps.items() if hasattr(PublicationStyleConfig, k)})
        
        if 'peak_detection' in data:
            pd = data['peak_detection']
            config.peak_detection = PeakDetectionConfig(**{k: v for k, v in pd.items() if hasattr(PeakDetectionConfig, k)})
        
        if 'peak_matching' in data:
            pm = data['peak_matching']
            config.peak_matching = PeakMatchingConfig(**{k: v for k, v in pm.items() if hasattr(PeakMatchingConfig, k)})
        
        if 'spectrum_scan' in data:
            ss = data['spectrum_scan']
            config.spectrum_scan = SpectrumScanConfig(**{k: v for k, v in ss.items() if hasattr(SpectrumScanConfig, k)})
        
        return config


class PlotConfigManager:
    """绘图配置管理器（单例模式）"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.settings = QSettings("GTLab", "SpectraPro_v4")
            self.config = PlotConfig()
            self._load_config()
            PlotConfigManager._initialized = True
    
    def _load_config(self):
        """从QSettings加载配置"""
        # 加载出版质量样式
        ps = self.config.publication_style
        ps.fig_width = float(self.settings.value("plot_config/fig_width", ps.fig_width))
        ps.fig_height = float(self.settings.value("plot_config/fig_height", ps.fig_height))
        ps.fig_dpi = int(self.settings.value("plot_config/fig_dpi", ps.fig_dpi))
        ps.aspect_ratio = float(self.settings.value("plot_config/aspect_ratio", ps.aspect_ratio))
        ps.font_family = str(self.settings.value("plot_config/font_family", ps.font_family))
        ps.axis_title_fontsize = int(self.settings.value("plot_config/axis_title_fontsize", ps.axis_title_fontsize))
        ps.tick_label_fontsize = int(self.settings.value("plot_config/tick_label_fontsize", ps.tick_label_fontsize))
        ps.legend_fontsize = int(self.settings.value("plot_config/legend_fontsize", ps.legend_fontsize))
        ps.title_fontsize = int(self.settings.value("plot_config/title_fontsize", ps.title_fontsize))
        ps.line_width = float(self.settings.value("plot_config/line_width", ps.line_width))
        ps.line_style = str(self.settings.value("plot_config/line_style", ps.line_style))
        ps.tick_direction = str(self.settings.value("plot_config/tick_direction", ps.tick_direction))
        ps.tick_len_major = int(self.settings.value("plot_config/tick_len_major", ps.tick_len_major))
        ps.tick_len_minor = int(self.settings.value("plot_config/tick_len_minor", ps.tick_len_minor))
        ps.tick_width = float(self.settings.value("plot_config/tick_width", ps.tick_width))
        ps.show_grid = self.settings.value("plot_config/show_grid", ps.show_grid, type=bool)
        ps.grid_alpha = float(self.settings.value("plot_config/grid_alpha", ps.grid_alpha))
        ps.shadow_alpha = float(self.settings.value("plot_config/shadow_alpha", ps.shadow_alpha))
        ps.spine_top = self.settings.value("plot_config/spine_top", ps.spine_top, type=bool)
        ps.spine_bottom = self.settings.value("plot_config/spine_bottom", ps.spine_bottom, type=bool)
        ps.spine_left = self.settings.value("plot_config/spine_left", ps.spine_left, type=bool)
        ps.spine_right = self.settings.value("plot_config/spine_right", ps.spine_right, type=bool)
        ps.spine_width = float(self.settings.value("plot_config/spine_width", ps.spine_width))
        ps.show_legend = self.settings.value("plot_config/show_legend", ps.show_legend, type=bool)
        ps.legend_frame = self.settings.value("plot_config/legend_frame", ps.legend_frame, type=bool)
        ps.legend_loc = str(self.settings.value("plot_config/legend_loc", ps.legend_loc))
        ps.legend_ncol = int(self.settings.value("plot_config/legend_ncol", ps.legend_ncol))
        ps.legend_columnspacing = float(self.settings.value("plot_config/legend_columnspacing", ps.legend_columnspacing))
        ps.legend_labelspacing = float(self.settings.value("plot_config/legend_labelspacing", ps.legend_labelspacing))
        ps.legend_handlelength = float(self.settings.value("plot_config/legend_handlelength", ps.legend_handlelength))
        
        # 标题控制
        ps.xlabel_text = str(self.settings.value("plot_config/xlabel_text", ps.xlabel_text))
        ps.xlabel_show = self.settings.value("plot_config/xlabel_show", ps.xlabel_show, type=bool)
        ps.xlabel_fontsize = int(self.settings.value("plot_config/xlabel_fontsize", ps.xlabel_fontsize))
        ps.xlabel_pad = float(self.settings.value("plot_config/xlabel_pad", ps.xlabel_pad))
        ps.ylabel_text = str(self.settings.value("plot_config/ylabel_text", ps.ylabel_text))
        ps.ylabel_show = self.settings.value("plot_config/ylabel_show", ps.ylabel_show, type=bool)
        ps.ylabel_fontsize = int(self.settings.value("plot_config/ylabel_fontsize", ps.ylabel_fontsize))
        ps.ylabel_pad = float(self.settings.value("plot_config/ylabel_pad", ps.ylabel_pad))
        ps.title_text = str(self.settings.value("plot_config/title_text", ps.title_text))
        ps.title_show = self.settings.value("plot_config/title_show", ps.title_show, type=bool)
        ps.title_fontsize = int(self.settings.value("plot_config/title_fontsize", ps.title_fontsize))
        ps.title_pad = float(self.settings.value("plot_config/title_pad", ps.title_pad))
        
        # 坐标轴显示控制
        ps.x_axis_invert = self.settings.value("plot_config/x_axis_invert", ps.x_axis_invert, type=bool)
        ps.show_x_values = self.settings.value("plot_config/show_x_values", ps.show_x_values, type=bool)
        ps.show_y_values = self.settings.value("plot_config/show_y_values", ps.show_y_values, type=bool)
        
        # 加载峰值检测
        pd = self.config.peak_detection
        pd.enabled = self.settings.value("plot_config/peak_detection_enabled", pd.enabled, type=bool)
        pd.height_threshold = float(self.settings.value("plot_config/peak_height_threshold", pd.height_threshold))
        pd.distance_min = int(self.settings.value("plot_config/peak_distance_min", pd.distance_min))
        pd.prominence = self.settings.value("plot_config/peak_prominence", pd.prominence)
        if pd.prominence is not None:
            pd.prominence = float(pd.prominence)
        pd.width = self.settings.value("plot_config/peak_width", pd.width)
        if pd.width is not None:
            pd.width = float(pd.width)
        pd.wlen = self.settings.value("plot_config/peak_wlen", pd.wlen)
        if pd.wlen is not None:
            pd.wlen = int(pd.wlen)
        pd.rel_height = self.settings.value("plot_config/peak_rel_height", pd.rel_height)
        if pd.rel_height is not None:
            pd.rel_height = float(pd.rel_height)
        
        # 加载峰值匹配
        pm = self.config.peak_matching
        pm.enabled = self.settings.value("plot_config/peak_matching_enabled", pm.enabled, type=bool)
        pm.mode = str(self.settings.value("plot_config/peak_matching_mode", pm.mode))
        pm.tolerance = float(self.settings.value("plot_config/peak_matching_tolerance", pm.tolerance))
        pm.reference_index = int(self.settings.value("plot_config/peak_matching_reference_index", pm.reference_index))
        pm.marker_shape = str(self.settings.value("plot_config/peak_matching_marker_shape", pm.marker_shape))
        pm.marker_size = float(self.settings.value("plot_config/peak_matching_marker_size", pm.marker_size))
        pm.marker_distance = float(self.settings.value("plot_config/peak_matching_marker_distance", pm.marker_distance))
        pm.marker_rotation = float(self.settings.value("plot_config/peak_matching_marker_rotation", pm.marker_rotation))
        pm.show_connection_lines = self.settings.value("plot_config/peak_matching_show_connection_lines", pm.show_connection_lines, type=bool)
        pm.use_spectrum_color_for_connection = self.settings.value("plot_config/peak_matching_use_spectrum_color_for_connection", pm.use_spectrum_color_for_connection, type=bool)
        pm.connection_line_color = str(self.settings.value("plot_config/peak_matching_connection_line_color", pm.connection_line_color))
        pm.connection_line_width = float(self.settings.value("plot_config/peak_matching_connection_line_width", pm.connection_line_width))
        pm.connection_line_style = str(self.settings.value("plot_config/peak_matching_connection_line_style", pm.connection_line_style))
        pm.connection_line_alpha = float(self.settings.value("plot_config/peak_matching_connection_line_alpha", pm.connection_line_alpha))
        pm.show_peak_labels = self.settings.value("plot_config/peak_matching_show_peak_labels", pm.show_peak_labels, type=bool)
        pm.label_fontsize = float(self.settings.value("plot_config/peak_matching_label_fontsize", pm.label_fontsize))
        pm.label_color = str(self.settings.value("plot_config/peak_matching_label_color", pm.label_color))
        pm.label_rotation = float(self.settings.value("plot_config/peak_matching_label_rotation", pm.label_rotation))
        pm.label_distance = float(self.settings.value("plot_config/peak_matching_label_distance", pm.label_distance))
        
        # 加载谱线扫描
        ss = self.config.spectrum_scan
        ss.enabled = self.settings.value("plot_config/spectrum_scan_enabled", ss.enabled, type=bool)
        ss.scan_last_plot = self.settings.value("plot_config/spectrum_scan_last_plot", ss.scan_last_plot, type=bool)
        ss.stack_offset = float(self.settings.value("plot_config/stack_offset", ss.stack_offset))
    
    def save_config(self):
        """保存配置到QSettings"""
        ps = self.config.publication_style
        self.settings.setValue("plot_config/fig_width", ps.fig_width)
        self.settings.setValue("plot_config/fig_height", ps.fig_height)
        self.settings.setValue("plot_config/fig_dpi", ps.fig_dpi)
        self.settings.setValue("plot_config/aspect_ratio", ps.aspect_ratio)
        self.settings.setValue("plot_config/font_family", ps.font_family)
        self.settings.setValue("plot_config/axis_title_fontsize", ps.axis_title_fontsize)
        self.settings.setValue("plot_config/tick_label_fontsize", ps.tick_label_fontsize)
        self.settings.setValue("plot_config/legend_fontsize", ps.legend_fontsize)
        self.settings.setValue("plot_config/title_fontsize", ps.title_fontsize)
        self.settings.setValue("plot_config/line_width", ps.line_width)
        self.settings.setValue("plot_config/line_style", ps.line_style)
        self.settings.setValue("plot_config/tick_direction", ps.tick_direction)
        self.settings.setValue("plot_config/tick_len_major", ps.tick_len_major)
        self.settings.setValue("plot_config/tick_len_minor", ps.tick_len_minor)
        self.settings.setValue("plot_config/tick_width", ps.tick_width)
        self.settings.setValue("plot_config/show_grid", ps.show_grid)
        self.settings.setValue("plot_config/grid_alpha", ps.grid_alpha)
        self.settings.setValue("plot_config/shadow_alpha", ps.shadow_alpha)
        self.settings.setValue("plot_config/spine_top", ps.spine_top)
        self.settings.setValue("plot_config/spine_bottom", ps.spine_bottom)
        self.settings.setValue("plot_config/spine_left", ps.spine_left)
        self.settings.setValue("plot_config/spine_right", ps.spine_right)
        self.settings.setValue("plot_config/spine_width", ps.spine_width)
        self.settings.setValue("plot_config/show_legend", ps.show_legend)
        self.settings.setValue("plot_config/legend_frame", ps.legend_frame)
        self.settings.setValue("plot_config/legend_loc", ps.legend_loc)
        self.settings.setValue("plot_config/legend_ncol", ps.legend_ncol)
        self.settings.setValue("plot_config/legend_columnspacing", ps.legend_columnspacing)
        self.settings.setValue("plot_config/legend_labelspacing", ps.legend_labelspacing)
        self.settings.setValue("plot_config/legend_handlelength", ps.legend_handlelength)
        
        # 标题控制
        self.settings.setValue("plot_config/xlabel_text", ps.xlabel_text)
        self.settings.setValue("plot_config/xlabel_show", ps.xlabel_show)
        self.settings.setValue("plot_config/xlabel_fontsize", ps.xlabel_fontsize)
        self.settings.setValue("plot_config/xlabel_pad", ps.xlabel_pad)
        self.settings.setValue("plot_config/ylabel_text", ps.ylabel_text)
        self.settings.setValue("plot_config/ylabel_show", ps.ylabel_show)
        self.settings.setValue("plot_config/ylabel_fontsize", ps.ylabel_fontsize)
        self.settings.setValue("plot_config/ylabel_pad", ps.ylabel_pad)
        self.settings.setValue("plot_config/title_text", ps.title_text)
        self.settings.setValue("plot_config/title_show", ps.title_show)
        self.settings.setValue("plot_config/title_fontsize", ps.title_fontsize)
        self.settings.setValue("plot_config/title_pad", ps.title_pad)
        
        # 坐标轴显示控制
        self.settings.setValue("plot_config/x_axis_invert", ps.x_axis_invert)
        self.settings.setValue("plot_config/show_x_values", ps.show_x_values)
        self.settings.setValue("plot_config/show_y_values", ps.show_y_values)
        
        # 峰值检测
        pd = self.config.peak_detection
        self.settings.setValue("plot_config/peak_detection_enabled", pd.enabled)
        self.settings.setValue("plot_config/peak_height_threshold", pd.height_threshold)
        self.settings.setValue("plot_config/peak_distance_min", pd.distance_min)
        self.settings.setValue("plot_config/peak_prominence", pd.prominence)
        self.settings.setValue("plot_config/peak_width", pd.width)
        self.settings.setValue("plot_config/peak_wlen", pd.wlen)
        self.settings.setValue("plot_config/peak_rel_height", pd.rel_height)
        
        # 峰值匹配
        pm = self.config.peak_matching
        self.settings.setValue("plot_config/peak_matching_enabled", pm.enabled)
        self.settings.setValue("plot_config/peak_matching_mode", pm.mode)
        self.settings.setValue("plot_config/peak_matching_tolerance", pm.tolerance)
        self.settings.setValue("plot_config/peak_matching_reference_index", pm.reference_index)
        self.settings.setValue("plot_config/peak_matching_marker_shape", pm.marker_shape)
        self.settings.setValue("plot_config/peak_matching_marker_size", pm.marker_size)
        self.settings.setValue("plot_config/peak_matching_marker_distance", pm.marker_distance)
        self.settings.setValue("plot_config/peak_matching_marker_rotation", pm.marker_rotation)
        self.settings.setValue("plot_config/peak_matching_show_connection_lines", pm.show_connection_lines)
        self.settings.setValue("plot_config/peak_matching_use_spectrum_color_for_connection", pm.use_spectrum_color_for_connection)
        self.settings.setValue("plot_config/peak_matching_connection_line_color", pm.connection_line_color)
        self.settings.setValue("plot_config/peak_matching_connection_line_width", pm.connection_line_width)
        self.settings.setValue("plot_config/peak_matching_connection_line_style", pm.connection_line_style)
        self.settings.setValue("plot_config/peak_matching_connection_line_alpha", pm.connection_line_alpha)
        self.settings.setValue("plot_config/peak_matching_show_peak_labels", pm.show_peak_labels)
        self.settings.setValue("plot_config/peak_matching_label_fontsize", pm.label_fontsize)
        self.settings.setValue("plot_config/peak_matching_label_color", pm.label_color)
        self.settings.setValue("plot_config/peak_matching_label_rotation", pm.label_rotation)
        self.settings.setValue("plot_config/peak_matching_label_distance", pm.label_distance)
        
        # 谱线扫描
        ss = self.config.spectrum_scan
        self.settings.setValue("plot_config/spectrum_scan_enabled", ss.enabled)
        self.settings.setValue("plot_config/spectrum_scan_last_plot", ss.scan_last_plot)
        self.settings.setValue("plot_config/stack_offset", ss.stack_offset)
        
        self.settings.sync()
    
    def get_config(self) -> PlotConfig:
        """获取当前配置"""
        return self.config
    
    def update_config(self, config: PlotConfig):
        """更新配置"""
        self.config = config
        self.save_config()


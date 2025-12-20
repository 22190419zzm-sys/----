"""
通用绘图接口
为所有绘图窗口提供统一的接口，方便扩展新绘图类型
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.core.plot_config_manager import PlotConfig, PlotConfigManager
from src.core.peak_matcher import PeakMatcher
from src.core.spectrum_scanner import SpectrumScanner, StackOffsetManager


class IPlotRenderer(ABC):
    """绘图渲染器接口"""
    
    @abstractmethod
    def render(self, ax: Axes, plot_data: List[Dict[str, Any]], 
              config: PlotConfig) -> None:
        """
        渲染绘图
        
        Args:
            ax: matplotlib axes对象
            plot_data: 绘图数据列表
            config: 绘图配置
        """
        pass


class BasePlotRenderer(IPlotRenderer):
    """基础绘图渲染器（提供通用功能）"""
    
    def __init__(self):
        self.config_manager = PlotConfigManager()
        self.peak_matcher = PeakMatcher()
        self.spectrum_scanner = SpectrumScanner()
        self.stack_offset_manager = StackOffsetManager()
    
    def apply_publication_style(self, ax: Axes, config: PlotConfig):
        """应用出版质量样式"""
        ps = config.publication_style
        
        # 设置字体
        font_family = ps.font_family
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = font_family
        plt.rcParams['mathtext.it'] = f'{font_family}:italic'
        plt.rcParams['mathtext.bf'] = f'{font_family}:bold'
        
        # 设置标签字体
        ax.xaxis.label.set_fontsize(ps.xlabel_fontsize)
        ax.yaxis.label.set_fontsize(ps.ylabel_fontsize)
        ax.title.set_fontsize(ps.title_fontsize)
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
        ax.title.set_fontfamily(font_family)
        
        # 设置标题（如果启用）
        if ps.xlabel_show:
            ax.set_xlabel(ps.xlabel_text, fontsize=ps.xlabel_fontsize, 
                         labelpad=ps.xlabel_pad, fontfamily=font_family)
        if ps.ylabel_show:
            ax.set_ylabel(ps.ylabel_text, fontsize=ps.ylabel_fontsize,
                         labelpad=ps.ylabel_pad, fontfamily=font_family)
        if ps.title_show and ps.title_text:
            ax.set_title(ps.title_text, fontsize=ps.title_fontsize,
                        pad=ps.title_pad, fontfamily=font_family)
        
        # 刻度设置
        ax.tick_params(axis='both', which='major',
                      direction=ps.tick_direction,
                      length=ps.tick_len_major,
                      width=ps.tick_width,
                      labelsize=ps.tick_label_fontsize,
                      top=ps.spine_top,
                      right=ps.spine_right)
        ax.tick_params(axis='both', which='minor',
                      direction=ps.tick_direction,
                      length=ps.tick_len_minor,
                      width=ps.tick_width,
                      top=ps.spine_top,
                      right=ps.spine_right)
        
        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(ps.tick_label_fontsize)
        
        # 边框设置
        ax.spines['top'].set_visible(ps.spine_top)
        ax.spines['bottom'].set_visible(ps.spine_bottom)
        ax.spines['left'].set_visible(ps.spine_left)
        ax.spines['right'].set_visible(ps.spine_right)
        
        for spine in ax.spines.values():
            spine.set_linewidth(ps.spine_width)
        
        # 网格设置
        if ps.show_grid:
            ax.grid(True, alpha=ps.grid_alpha, linestyle='-')
        else:
            ax.grid(False)
        
        # 图例设置
        legend = ax.get_legend()
        if legend:
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            legend_font.set_family(font_family)
            legend_font.set_size(ps.legend_fontsize)
            for text in legend.get_texts():
                text.set_fontproperties(legend_font)
            legend.set_frame_on(ps.legend_frame)
    
    def apply_peak_detection(self, ax: Axes, x_data: np.ndarray, y_data: np.ndarray,
                            color: str, config: PlotConfig):
        """应用峰值检测"""
        if not config.peak_detection.enabled:
            return
        
        pd = config.peak_detection
        peak_indices, properties = self.peak_matcher.detect_peaks(
            x_data, y_data,
            height=pd.height_threshold,
            distance=pd.distance_min,
            prominence=pd.prominence,
            width=pd.width,
            wlen=pd.wlen,
            rel_height=pd.rel_height
        )
        
        if len(peak_indices) > 0:
            peak_x = x_data[peak_indices]
            peak_y = y_data[peak_indices]
            
            # 绘制峰值标记
            ax.plot(peak_x, peak_y, marker=pd.marker_shape, 
                   markersize=pd.marker_size, color=color, 
                   linestyle='', zorder=10)
            
            # 绘制峰值标签
            if pd.show_label:
                for px, py in zip(peak_x, peak_y):
                    ax.text(px, py, f'{px:.1f}',
                           fontsize=pd.label_size,
                           color=pd.label_color,
                           fontweight='bold' if pd.label_bold else 'normal',
                           rotation=pd.label_rotation,
                           ha='center', va='bottom',
                           fontfamily=pd.label_font)
    
    def apply_peak_matching(self, ax: Axes, spectra_data: List[Dict[str, Any]],
                           config: PlotConfig):
        """应用峰值匹配"""
        if not config.peak_matching.enabled:
            return
        
        pm = config.peak_matching
        self.peak_matcher.tolerance = pm.tolerance
        
        # 准备光谱数据
        spectra_list = []
        for spec in spectra_data:
            spectra_list.append({
                'x': spec.get('x', np.array([])),
                'y': spec.get('y', np.array([])),
                'color': spec.get('color', 'blue'),
                'label': spec.get('label', '')
            })
        
        # 执行匹配
        match_result = self.peak_matcher.match_multiple_spectra(
            spectra_list,
            reference_index=pm.reference_index,
            mode=pm.mode
        )
        
        # 绘制匹配结果
        matches = match_result.get('matches', {})
        for idx, match_info in matches.items():
            if idx < len(spectra_data):
                spectrum = spectra_data[idx]
                color = spectrum.get('color', 'blue')
                positions = match_info.get('positions', np.array([]))
                
                if len(positions) > 0:
                    # 获取对应的Y值（需要插值）
                    x_data = spectrum.get('x', np.array([]))
                    y_data = spectrum.get('y', np.array([]))
                    
                    if len(x_data) > 0 and len(y_data) > 0:
                        from scipy.interpolate import interp1d
                        interp_func = interp1d(x_data, y_data, 
                                             kind='linear', 
                                             bounds_error=False, 
                                             fill_value=0.0)
                        match_y = interp_func(positions)
                        
                        # 绘制匹配的峰值
                        ax.plot(positions, match_y, 
                               marker='v', markersize=8,
                               color=color, linestyle='',
                               zorder=10, alpha=0.7)
    
    def apply_stack_offset(self, plot_data: List[Dict[str, Any]], 
                          config: PlotConfig) -> List[Dict[str, Any]]:
        """应用堆叠偏移"""
        ss = config.spectrum_scan
        self.stack_offset_manager.default_offset = ss.stack_offset
        
        # 应用独立偏移
        for label, offset in ss.individual_offsets.items():
            self.stack_offset_manager.set_individual_offset(label, offset)
        
        # 应用堆叠偏移
        result = []
        for i, data in enumerate(plot_data):
            label = data.get('label', f'Spectrum {i}')
            total_offset = self.stack_offset_manager.get_total_offset(
                label, i, ss.stack_offset
            )
            
            modified_data = data.copy()
            modified_data['y'] = np.array(data.get('y', [])) + total_offset
            modified_data['stack_offset'] = total_offset
            result.append(modified_data)
        
        return result
    
    def scan_spectra(self, plot_data: List[Dict[str, Any]], 
                    config: PlotConfig) -> List[Dict[str, Any]]:
        """扫描谱线"""
        if not config.spectrum_scan.enabled:
            return plot_data
        
        ss = config.spectrum_scan
        
        if ss.scan_last_plot:
            # 扫描最后一次绘图
            scanned = self.spectrum_scanner.scan_last_plot(plot_data)
            
            # 应用自定义映射
            if ss.custom_mappings:
                aligned = self.spectrum_scanner.apply_mappings(
                    ss.custom_mappings, interpolation=True
                )
                return aligned
            
            return scanned
        
        return plot_data


class StandardPlotRenderer(BasePlotRenderer):
    """标准绘图渲染器（用于普通线图）"""
    
    def render(self, ax: Axes, plot_data: List[Dict[str, Any]], 
              config: PlotConfig) -> None:
        """渲染标准线图"""
        # 应用堆叠偏移
        processed_data = self.apply_stack_offset(plot_data, config)
        
        # 扫描谱线（如果需要）
        processed_data = self.scan_spectra(processed_data, config)
        
        # 绘制谱线
        for data in processed_data:
            x = data.get('x', np.array([]))
            y = data.get('y', np.array([]))
            color = data.get('color', 'blue')
            label = data.get('label', '')
            linewidth = config.publication_style.line_width
            linestyle = config.publication_style.line_style
            
            if len(x) > 0 and len(y) > 0:
                ax.plot(x, y, color=color, label=label,
                       linewidth=linewidth, linestyle=linestyle)
                
                # 应用峰值检测
                self.apply_peak_detection(ax, x, y, color, config)
        
        # 应用峰值匹配
        self.apply_peak_matching(ax, processed_data, config)
        
        # 应用出版质量样式
        self.apply_publication_style(ax, config)
        
        # 图例
        if config.publication_style.show_legend:
            ax.legend(loc=config.publication_style.legend_loc,
                     ncol=config.publication_style.legend_ncol,
                     frameon=config.publication_style.legend_frame,
                     columnspacing=config.publication_style.legend_columnspacing,
                     labelspacing=config.publication_style.legend_labelspacing,
                     handlelength=config.publication_style.legend_handlelength)


class WaterfallPlotRenderer(BasePlotRenderer):
    """瀑布图渲染器（用于堆叠图）"""
    
    def render(self, ax: Axes, plot_data: List[Dict[str, Any]], 
              config: PlotConfig) -> None:
        """渲染瀑布图"""
        # 应用堆叠偏移
        processed_data = self.apply_stack_offset(plot_data, config)
        
        # 扫描谱线（如果需要）
        processed_data = self.scan_spectra(processed_data, config)
        
        # 绘制堆叠谱线
        for data in processed_data:
            x = data.get('x', np.array([]))
            y = data.get('y', np.array([]))
            color = data.get('color', 'blue')
            label = data.get('label', '')
            linewidth = config.publication_style.line_width
            linestyle = config.publication_style.line_style
            
            if len(x) > 0 and len(y) > 0:
                ax.plot(x, y, color=color, label=label,
                       linewidth=linewidth, linestyle=linestyle)
                
                # 应用峰值检测
                self.apply_peak_detection(ax, x, y, color, config)
        
        # 应用峰值匹配
        self.apply_peak_matching(ax, processed_data, config)
        
        # 应用出版质量样式
        self.apply_publication_style(ax, config)
        
        # 图例
        if config.publication_style.show_legend:
            ax.legend(loc=config.publication_style.legend_loc,
                     ncol=config.publication_style.legend_ncol,
                     frameon=config.publication_style.legend_frame)


class ShadowPlotRenderer(BasePlotRenderer):
    """阴影图渲染器（用于带阴影区域的图）"""
    
    def render(self, ax: Axes, plot_data: List[Dict[str, Any]], 
              config: PlotConfig) -> None:
        """渲染阴影图"""
        # 应用堆叠偏移
        processed_data = self.apply_stack_offset(plot_data, config)
        
        # 扫描谱线（如果需要）
        processed_data = self.scan_spectra(processed_data, config)
        
        # 绘制带阴影的谱线
        for data in processed_data:
            x = data.get('x', np.array([]))
            y = data.get('y', np.array([]))
            y_upper = data.get('y_upper', None)
            y_lower = data.get('y_lower', None)
            color = data.get('color', 'blue')
            label = data.get('label', '')
            linewidth = config.publication_style.line_width
            shadow_alpha = config.publication_style.shadow_alpha
            
            # 确保 alpha 值在 0-1 范围内
            safe_alpha = max(0.0, min(1.0, shadow_alpha))
            
            if len(x) > 0 and len(y) > 0:
                # 绘制主线条
                ax.plot(x, y, color=color, label=label, linewidth=linewidth)
                
                # 绘制阴影区域
                if y_upper is not None and y_lower is not None:
                    ax.fill_between(x, y_lower, y_upper, 
                                   color=color, alpha=safe_alpha)
                
                # 应用峰值检测
                self.apply_peak_detection(ax, x, y, color, config)
        
        # 应用峰值匹配
        self.apply_peak_matching(ax, processed_data, config)
        
        # 应用出版质量样式
        self.apply_publication_style(ax, config)
        
        # 图例
        if config.publication_style.show_legend:
            ax.legend(loc=config.publication_style.legend_loc,
                     ncol=config.publication_style.legend_ncol,
                     frameon=config.publication_style.legend_frame)


class MeanShadowPlotRenderer(BasePlotRenderer):
    """均值+阴影图渲染器（用于组内平均光谱带标准差阴影）"""
    
    def render(self, ax: Axes, plot_data: List[Dict[str, Any]], 
              config: PlotConfig) -> None:
        """渲染均值+阴影图"""
        if not plot_data:
            return
        
        # 对于均值+阴影图，plot_data 应该包含均值、标准差等信息
        # 如果只有一个数据项，它应该包含 mean, std 等信息
        for data in plot_data:
            x = data.get('x', np.array([]))
            y_mean = data.get('y', np.array([]))  # 均值
            y_std = data.get('y_std', None)  # 标准差
            color = data.get('color', 'blue')
            label = data.get('label', 'Mean')
            std_label = data.get('std_label', 'Std Dev')
            linewidth = config.publication_style.line_width
            shadow_alpha = config.publication_style.shadow_alpha
            
            # 确保 alpha 值在 0-1 范围内
            safe_alpha = max(0.0, min(1.0, shadow_alpha))
            
            if len(x) > 0 and len(y_mean) > 0:
                # 绘制均值线
                ax.plot(x, y_mean, color=color, label=label, linewidth=linewidth)
                
                # 绘制阴影区域（如果有标准差）
                if y_std is not None and len(y_std) > 0:
                    y_upper = y_mean + y_std
                    y_lower = y_mean - y_std
                    ax.fill_between(x, y_lower, y_upper, 
                                   color=color, alpha=safe_alpha, label=std_label)
                
                # 应用峰值检测
                self.apply_peak_detection(ax, x, y_mean, color, config)
        
        # 应用峰值匹配（如果有多个数据项）
        if len(plot_data) > 1:
            self.apply_peak_matching(ax, plot_data, config)
        
        # 应用出版质量样式
        self.apply_publication_style(ax, config)
        
        # 图例
        if config.publication_style.show_legend:
            ax.legend(loc=config.publication_style.legend_loc,
                     ncol=config.publication_style.legend_ncol,
                     frameon=config.publication_style.legend_frame,
                     columnspacing=config.publication_style.legend_columnspacing,
                     labelspacing=config.publication_style.legend_labelspacing,
                     handlelength=config.publication_style.legend_handlelength)


# 渲染器注册表
PLOT_RENDERERS = {
    'standard': StandardPlotRenderer,
    'waterfall': WaterfallPlotRenderer,
    'shadow': ShadowPlotRenderer,
    'mean_shadow': MeanShadowPlotRenderer,
}


def get_renderer(renderer_type: str = 'standard') -> IPlotRenderer:
    """获取绘图渲染器"""
    renderer_class = PLOT_RENDERERS.get(renderer_type, StandardPlotRenderer)
    return renderer_class()


def register_renderer(renderer_type: str, renderer_class: type):
    """注册新的绘图渲染器"""
    PLOT_RENDERERS[renderer_type] = renderer_class


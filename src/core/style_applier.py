"""
统一的样式应用辅助函数
所有绘图窗口都应该使用这个模块应用样式，确保样式一致性
"""
from typing import Dict, Any, Optional
import numpy as np
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties

from src.core.plot_config_manager import PlotConfigManager, PlotConfig


def apply_publication_style_to_axes(ax: Axes, config: Optional[PlotConfig] = None) -> None:
    """
    统一应用出版质量样式到matplotlib axes
    
    Args:
        ax: matplotlib axes对象
        config: PlotConfig对象，如果为None则从PlotConfigManager获取
    """
    if config is None:
        config_manager = PlotConfigManager()
        config = config_manager.get_config()
    
    ps = config.publication_style
    
    # 设置字体
    font_family = ps.font_family
    current_font = 'Times New Roman' if font_family == 'Times New Roman' else font_family
    
    # 设置标题（如果启用）
    if ps.xlabel_show:
        ax.set_xlabel(ps.xlabel_text, fontsize=ps.xlabel_fontsize, 
                     labelpad=ps.xlabel_pad, fontfamily=current_font)
    if ps.ylabel_show:
        ax.set_ylabel(ps.ylabel_text, fontsize=ps.ylabel_fontsize,
                     labelpad=ps.ylabel_pad, fontfamily=current_font)
    if ps.title_show and ps.title_text:
        ax.set_title(ps.title_text, fontsize=ps.title_fontsize,
                    pad=ps.title_pad, fontfamily=current_font)
    
    # 刻度设置
    ax.tick_params(axis='both', which='major',
                  direction=ps.tick_direction,
                  length=ps.tick_len_major,
                  width=ps.tick_width,
                  labelsize=ps.tick_label_fontsize,
                  top=ps.tick_top,
                  bottom=ps.tick_bottom,
                  left=ps.tick_left,
                  right=ps.tick_right,
                  labeltop=ps.tick_top,
                  labelbottom=ps.tick_bottom,
                  labelleft=ps.tick_left,
                  labelright=ps.tick_right)
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
    
    # 图例设置（如果图例已存在）
    legend = ax.get_legend()
    if legend:
        legend_font = FontProperties()
        legend_font.set_family(current_font)
        legend_font.set_size(ps.legend_fontsize)
        for text in legend.get_texts():
            text.set_fontproperties(legend_font)
        legend.set_frame_on(ps.legend_frame)


def get_style_params_from_config(config: Optional[PlotConfig] = None) -> Dict[str, Any]:
    """
    从PlotConfig对象中提取样式参数字典（用于向后兼容）
    
    Args:
        config: PlotConfig对象，如果为None则从PlotConfigManager获取
        
    Returns:
        包含样式参数的字典
    """
    if config is None:
        config_manager = PlotConfigManager()
        config = config_manager.get_config()
    
    ps = config.publication_style
    
    return {
        'fig_width': ps.fig_width,
        'fig_height': ps.fig_height,
        'fig_dpi': ps.fig_dpi,
        'font_family': ps.font_family,
        'axis_title_fontsize': ps.axis_title_fontsize,
        'tick_label_fontsize': ps.tick_label_fontsize,
        'legend_fontsize': ps.legend_fontsize,
        'line_width': ps.line_width,
        'line_style': ps.line_style,
        'tick_direction': ps.tick_direction,
        'tick_len_major': ps.tick_len_major,
        'tick_len_minor': ps.tick_len_minor,
        'tick_width': ps.tick_width,
        'show_grid': ps.show_grid,
        'grid_alpha': ps.grid_alpha,
        'shadow_alpha': ps.shadow_alpha,
        'spine_top': ps.spine_top,
        'spine_bottom': ps.spine_bottom,
        'spine_left': ps.spine_left,
        'spine_right': ps.spine_right,
        'spine_width': ps.spine_width,
        'show_legend': ps.show_legend,
        'legend_frame': ps.legend_frame,
        'legend_loc': ps.legend_loc,
        'legend_ncol': ps.legend_ncol,
        'legend_columnspacing': ps.legend_columnspacing,
        'legend_labelspacing': ps.legend_labelspacing,
        'legend_handlelength': ps.legend_handlelength,
        'aspect_ratio': ps.aspect_ratio,
        'xlabel_text': ps.xlabel_text,
        'xlabel_show': ps.xlabel_show,
        'xlabel_fontsize': ps.xlabel_fontsize,
        'xlabel_pad': ps.xlabel_pad,
        'ylabel_text': ps.ylabel_text,
        'ylabel_show': ps.ylabel_show,
        'ylabel_fontsize': ps.ylabel_fontsize,
        'ylabel_pad': ps.ylabel_pad,
        'title_text': ps.title_text,
        'title_show': ps.title_show,
        'title_fontsize': ps.title_fontsize,
        'title_pad': ps.title_pad,
    }


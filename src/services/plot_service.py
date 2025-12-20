"""
绘图服务骨架：
- PlotSpec：描述通用绘图数据/样式
- PlotService：将分析结果转换为 PlotSpec（主图可复用）
当前为占位实现，后续可扩展到 3D/交互式绘图。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TraceSpec:
    x: Any
    y: Any
    label: str = ""
    style: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlotSpec:
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    traces: List[TraceSpec] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)  # 轴范围、网格、图例等


class PlotService:
    """将算法输出转换为通用 PlotSpec，供主绘图组件使用。"""

    def to_plot_spec(self, title: str, xlabel: str, ylabel: str, series: List[Dict[str, Any]]) -> PlotSpec:
        traces = []
        for item in series:
            traces.append(
                TraceSpec(
                    x=item.get("x"),
                    y=item.get("y"),
                    label=item.get("label", ""),
                    style=item.get("style", {}),
                )
            )
        return PlotSpec(title=title, xlabel=xlabel, ylabel=ylabel, traces=traces, layout={})


# 默认实例
plot_service = PlotService()


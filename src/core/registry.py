"""
轻量级注册表，统一管理预处理函数、模型与绘图风格，便于插件式扩展。

用途：
- register_preprocessor(name, callable): 注册预处理步骤
- register_model(name, cls_or_callable): 注册模型/变换器
- register_plot_style(name, callable): 注册绘图样式（可选，用于非 GUI 场景）

获取：
- get_preprocessors() / get_models() / get_plot_styles()

测试：
- reset_registry() 仅用于测试清空注册表。
"""

from typing import Callable, Dict

_preprocessors: Dict[str, Callable] = {}
_models: Dict[str, Callable] = {}
_plot_styles: Dict[str, Callable] = {}


def register_preprocessor(name: str, func: Callable):
    """注册预处理函数，名称统一小写。"""
    _preprocessors[name.lower()] = func


def register_model(name: str, factory: Callable):
    """注册模型/变换器构造器或类。"""
    _models[name.lower()] = factory


def register_plot_style(name: str, func: Callable):
    """注册绘图样式生成器/应用函数。"""
    _plot_styles[name.lower()] = func


def get_preprocessors() -> Dict[str, Callable]:
    return dict(_preprocessors)


def get_models() -> Dict[str, Callable]:
    return dict(_models)


def get_plot_styles() -> Dict[str, Callable]:
    return dict(_plot_styles)


def reset_registry():
    """仅测试使用：清空注册表，避免跨测试污染。"""
    _preprocessors.clear()
    _models.clear()
    _plot_styles.clear()


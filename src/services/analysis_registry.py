"""
分析器注册表：用于解耦 UI 与具体算法实现。
- register(name, cls_or_factory): 注册分析器
- create(name, **kwargs): 创建分析器实例
- list(): 查看已注册的分析器 key
"""
from typing import Callable, Dict, Any


class AnalysisRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]):
        self._registry[name] = factory

    def create(self, name: str, **kwargs):
        if name not in self._registry:
            raise KeyError(f"Analyzer '{name}' not registered.")
        return self._registry[name](**kwargs)

    def list(self):
        return list(self._registry.keys())


# 默认全局实例，UI/服务层可直接导入
registry = AnalysisRegistry()


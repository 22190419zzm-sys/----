"""
延迟导入工具：实现模块的延迟加载
"""
import sys
from typing import Any, Optional


class LazyModule:
    """延迟加载的模块包装器"""
    
    def __init__(self, module_name: str):
        """
        初始化延迟模块
        
        Args:
            module_name: 模块名称（如 'sklearn.decomposition'）
        """
        self._module_name = module_name
        self._module: Optional[Any] = None
        self._imported = False
    
    def _import(self):
        """实际导入模块"""
        if not self._imported:
            try:
                self._module = __import__(self._module_name, fromlist=[''])
                self._imported = True
            except ImportError as e:
                raise ImportError(f"Failed to import {self._module_name}: {e}")
    
    def __getattr__(self, name: str):
        """获取模块属性时触发导入"""
        if not self._imported:
            self._import()
        return getattr(self._module, name)
    
    def __dir__(self):
        """返回模块的属性列表"""
        if not self._imported:
            self._import()
        return dir(self._module)


# 延迟导入的模块字典
_lazy_modules = {}


def lazy_import(module_name: str):
    """
    延迟导入模块
    
    Args:
        module_name: 模块名称
    
    Returns:
        LazyModule实例
    
    Example:
        sklearn = lazy_import('sklearn')
        # 此时sklearn还未导入
        pca = sklearn.decomposition.PCA()  # 此时才真正导入
    """
    if module_name not in _lazy_modules:
        _lazy_modules[module_name] = LazyModule(module_name)
    return _lazy_modules[module_name]


# 预定义的延迟导入模块
sklearn = lazy_import('sklearn')
sklearn_pipeline = lazy_import('sklearn.pipeline')
sklearn_decomposition = lazy_import('sklearn.decomposition')
sklearn_neural_network = lazy_import('sklearn.neural_network')
sklearn_base = lazy_import('sklearn.base')


"""
缓存工具模块：提供数据预处理和绘图结果的缓存机制
"""
import os
import hashlib
import pickle
import json
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Optional
import numpy as np


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径，默认为用户目录下的 .spectra_cache
        """
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), '.spectra_cache')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存（用于快速访问）
        self._memory_cache = {}
        self._max_memory_cache_size = 100  # 最多缓存100个结果
        
    def _get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        # 将参数序列化为字符串
        key_data = {
            'args': [str(arg) for arg in args],
            'kwargs': {k: str(v) for k, v in sorted(kwargs.items())}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_file_path(self, cache_key: str, suffix: str = '.pkl') -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}{suffix}"
    
    def get(self, cache_key: str) -> Optional[Any]:
        """
        从缓存中获取数据（先检查内存缓存，再检查磁盘缓存）
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的数据，如果不存在则返回None
        """
        # 先检查内存缓存
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # 检查磁盘缓存
        cache_file = self._get_file_path(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                # 存入内存缓存
                self._set_memory_cache(cache_key, data)
                return data
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_key}: {e}")
                return None
        
        return None
    
    def set(self, cache_key: str, data: Any):
        """
        将数据存入缓存（同时存入内存和磁盘）
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
        """
        # 存入内存缓存
        self._set_memory_cache(cache_key, data)
        
        # 存入磁盘缓存
        cache_file = self._get_file_path(cache_key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_key}: {e}")
    
    def _set_memory_cache(self, cache_key: str, data: Any):
        """设置内存缓存（带大小限制）"""
        if len(self._memory_cache) >= self._max_memory_cache_size:
            # 删除最旧的缓存（简单策略：删除第一个）
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[cache_key] = data
    
    def clear(self, cache_key: Optional[str] = None):
        """
        清除缓存
        
        Args:
            cache_key: 要清除的缓存键，如果为None则清除所有缓存
        """
        if cache_key is None:
            # 清除所有缓存
            self._memory_cache.clear()
            # 清除磁盘缓存
            for cache_file in self.cache_dir.glob('*.pkl'):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
        else:
            # 清除指定缓存
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            cache_file = self._get_file_path(cache_key)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    pass
    
    def get_cache_size(self) -> int:
        """获取缓存文件数量"""
        return len(list(self.cache_dir.glob('*.pkl')))


# 全局缓存管理器实例
_cache_manager = CacheManager()


def cached(cache_key_func: Optional[Callable] = None, use_cache: bool = True):
    """
    缓存装饰器
    
    Args:
        cache_key_func: 用于生成缓存键的函数，如果为None则使用默认方法
        use_cache: 是否使用缓存，默认为True
    
    Example:
        @cached()
        def expensive_function(x, y):
            # 耗时操作
            return result
        
        @cached(cache_key_func=lambda x, y: f"key_{x}_{y}")
        def another_function(x, y):
            return result
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not use_cache:
                return func(*args, **kwargs)
            
            # 生成缓存键
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # 使用默认方法：函数名 + 参数哈希
                key_data = {
                    'func_name': func.__name__,
                    'args': [str(arg) for arg in args],
                    'kwargs': {k: str(v) for k, v in sorted(kwargs.items())}
                }
                key_str = json.dumps(key_data, sort_keys=True)
                cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # 检查缓存
            cached_result = _cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            _cache_manager.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器实例"""
    return _cache_manager


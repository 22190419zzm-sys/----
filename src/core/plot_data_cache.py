"""
绘图数据缓存模块
用于缓存文件读取和预处理结果，提升绘图性能
"""
import os
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FileCacheEntry:
    """文件缓存条目"""
    file_path: str
    mtime: float  # 文件修改时间
    data: Tuple  # (x, y) 数据
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PreprocessCacheEntry:
    """预处理缓存条目"""
    preprocess_hash: str  # 预处理参数哈希
    data: Any  # 预处理后的数据
    timestamp: datetime = field(default_factory=datetime.now)


class PlotDataCache:
    """绘图数据缓存管理器"""
    
    def __init__(self, max_cache_size: int = 100):
        """
        初始化缓存管理器
        
        Args:
            max_cache_size: 最大缓存条目数
        """
        self.max_cache_size = max_cache_size
        self.file_cache: Dict[str, FileCacheEntry] = {}
        self.preprocess_cache: Dict[str, PreprocessCacheEntry] = {}
        self.group_cache: Dict[str, Dict] = {}  # 分组结果缓存
        
    def _get_file_hash(self, file_path: str) -> str:
        """获取文件哈希（基于路径和修改时间）"""
        try:
            mtime = os.path.getmtime(file_path)
            return hashlib.md5(f"{file_path}:{mtime}".encode()).hexdigest()
        except:
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def _get_preprocess_hash(self, preprocess_params: Dict) -> str:
        """获取预处理参数哈希"""
        # 只包含影响预处理结果的参数
        relevant_params = {
            'qc_enabled': preprocess_params.get('qc_enabled', False),
            'qc_threshold': preprocess_params.get('qc_threshold', 5.0),
            'is_be_correction': preprocess_params.get('is_be_correction', False),
            'be_temp': preprocess_params.get('be_temp', 300.0),
            'is_smoothing': preprocess_params.get('is_smoothing', False),
            'smoothing_window': preprocess_params.get('smoothing_window', 15),
            'smoothing_poly': preprocess_params.get('smoothing_poly', 3),
            'is_baseline_als': preprocess_params.get('is_baseline_als', False),
            'als_lam': preprocess_params.get('als_lam', 10000),
            'als_p': preprocess_params.get('als_p', 0.005),
            'is_baseline_poly': preprocess_params.get('is_baseline_poly', False),
            'baseline_points': preprocess_params.get('baseline_points', 50),
            'baseline_poly': preprocess_params.get('baseline_poly', 3),
            'normalization_mode': preprocess_params.get('normalization_mode', 'None'),
            'global_transform_mode': preprocess_params.get('global_transform_mode', '无'),
            'global_log_base': preprocess_params.get('global_log_base', '10'),
            'global_log_offset': preprocess_params.get('global_log_offset', 1.0),
            'global_sqrt_offset': preprocess_params.get('global_sqrt_offset', 0.0),
            'is_quadratic_fit': preprocess_params.get('is_quadratic_fit', False),
            'quadratic_degree': preprocess_params.get('quadratic_degree', 2),
            'is_derivative': preprocess_params.get('is_derivative', False),
            'global_y_offset': preprocess_params.get('global_y_offset', 0.0),
            'x_min_phys': preprocess_params.get('x_min_phys'),
            'x_max_phys': preprocess_params.get('x_max_phys'),
        }
        params_str = json.dumps(relevant_params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _get_group_hash(self, file_list: List[str], n_chars: int) -> str:
        """获取分组哈希"""
        file_list_str = ','.join(sorted(file_list))
        return hashlib.md5(f"{file_list_str}:{n_chars}".encode()).hexdigest()
    
    def get_file_data(self, file_path: str) -> Optional[Tuple]:
        """
        获取文件数据（从缓存或需要重新读取）
        
        Returns:
            如果缓存有效返回 (x, y)，否则返回 None
        """
        if not os.path.exists(file_path):
            return None
        
        file_hash = self._get_file_hash(file_path)
        
        # 检查缓存
        if file_hash in self.file_cache:
            entry = self.file_cache[file_hash]
            # 验证文件是否改变
            try:
                current_mtime = os.path.getmtime(file_path)
                if current_mtime == entry.mtime:
                    return entry.data
            except:
                pass
        
        return None
    
    def cache_file_data(self, file_path: str, data: Tuple):
        """缓存文件数据"""
        if not os.path.exists(file_path):
            return
        
        file_hash = self._get_file_hash(file_path)
        try:
            mtime = os.path.getmtime(file_path)
        except:
            mtime = 0.0
        
        self.file_cache[file_hash] = FileCacheEntry(
            file_path=file_path,
            mtime=mtime,
            data=data
        )
        
        # 限制缓存大小
        if len(self.file_cache) > self.max_cache_size:
            # 删除最旧的条目
            oldest_key = min(self.file_cache.keys(), 
                           key=lambda k: self.file_cache[k].timestamp)
            del self.file_cache[oldest_key]
    
    def get_preprocess_data(self, file_path: str, preprocess_params: Dict) -> Optional[Any]:
        """
        获取预处理数据（从缓存）
        
        Returns:
            如果缓存有效返回预处理后的数据，否则返回 None
        """
        file_hash = self._get_file_hash(file_path)
        preprocess_hash = self._get_preprocess_hash(preprocess_params)
        cache_key = f"{file_hash}:{preprocess_hash}"
        
        if cache_key in self.preprocess_cache:
            return self.preprocess_cache[cache_key].data
        
        return None
    
    def cache_preprocess_data(self, file_path: str, preprocess_params: Dict, data: Any):
        """缓存预处理数据"""
        file_hash = self._get_file_hash(file_path)
        preprocess_hash = self._get_preprocess_hash(preprocess_params)
        cache_key = f"{file_hash}:{preprocess_hash}"
        
        self.preprocess_cache[cache_key] = PreprocessCacheEntry(
            preprocess_hash=preprocess_hash,
            data=data
        )
        
        # 限制缓存大小
        if len(self.preprocess_cache) > self.max_cache_size:
            oldest_key = min(self.preprocess_cache.keys(),
                           key=lambda k: self.preprocess_cache[k].timestamp)
            del self.preprocess_cache[oldest_key]
    
    def get_group_data(self, file_list: List[str], n_chars: int) -> Optional[Dict]:
        """获取分组数据（从缓存）"""
        group_hash = self._get_group_hash(file_list, n_chars)
        return self.group_cache.get(group_hash)
    
    def cache_group_data(self, file_list: List[str], n_chars: int, groups: Dict):
        """缓存分组数据"""
        group_hash = self._get_group_hash(file_list, n_chars)
        self.group_cache[group_hash] = groups
        
        # 限制缓存大小
        if len(self.group_cache) > self.max_cache_size:
            # 删除最旧的条目（简单策略：删除第一个）
            if self.group_cache:
                first_key = next(iter(self.group_cache))
                del self.group_cache[first_key]
    
    def clear_cache(self):
        """清空所有缓存"""
        self.file_cache.clear()
        self.preprocess_cache.clear()
        self.group_cache.clear()
    
    def clear_preprocess_cache(self):
        """清空预处理缓存（保留文件缓存）"""
        self.preprocess_cache.clear()
    
    def clear_file_cache(self, file_path: Optional[str] = None):
        """清空文件缓存（如果指定文件路径，只清除该文件的缓存）"""
        if file_path:
            file_hash = self._get_file_hash(file_path)
            if file_hash in self.file_cache:
                del self.file_cache[file_hash]
            # 清除相关的预处理缓存
            keys_to_remove = [k for k in self.preprocess_cache.keys() if k.startswith(file_hash)]
            for k in keys_to_remove:
                del self.preprocess_cache[k]
        else:
            self.file_cache.clear()
            self.preprocess_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'file_cache_size': len(self.file_cache),
            'preprocess_cache_size': len(self.preprocess_cache),
            'group_cache_size': len(self.group_cache),
            'total_size': len(self.file_cache) + len(self.preprocess_cache) + len(self.group_cache)
        }


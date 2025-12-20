"""
峰值匹配模块
实现多模式峰值匹配功能，支持所有绘图类型
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


class PeakMatcher:
    """峰值匹配器"""
    
    def __init__(self, tolerance: float = 5.0):
        """
        Args:
            tolerance: 峰值匹配容差（cm^-1）
        """
        self.tolerance = tolerance
    
    def detect_peaks(self, x_data: np.ndarray, y_data: np.ndarray, 
                     height: float = 0.0, distance: int = 10,
                     prominence: Optional[float] = None,
                     width: Optional[float] = None,
                     wlen: Optional[int] = None,
                     rel_height: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """
        检测峰值
        
        Returns:
            (peak_indices, peak_properties)
        """
        peak_kwargs = {}
        
        # 智能调整height参数
        y_max = np.max(y_data)
        y_min = np.min(y_data)
        y_range = y_max - y_min
        
        if height == 0:
            if y_max > 0:
                height = y_max * 0.0001
            else:
                y_mean = np.mean(y_data)
                y_std = np.std(y_data)
                height = abs(y_mean) + y_std * 0.05
        
        if height > y_range * 2 and y_range > 0:
            height = y_max * 0.0001
        
        peak_kwargs['height'] = height
        
        # 智能调整distance参数
        if distance == 0:
            distance = max(1, int(len(y_data) * 0.001))
        if distance > len(y_data) * 0.5:
            distance = max(1, int(len(y_data) * 0.001))
        distance = max(1, distance)
        
        # 如果height很小，不使用distance限制
        use_distance = True
        if height < 0 or (y_max > 0 and height < y_max * 0.001):
            use_distance = False
        elif distance == 1:
            use_distance = False
        
        if use_distance:
            peak_kwargs['distance'] = distance
        
        # 添加可选参数
        if prominence is not None and prominence != 0:
            if prominence > y_range * 2 and y_range > 0:
                prominence = y_range * 0.001
            peak_kwargs['prominence'] = prominence
        
        if width is not None and width > 0:
            peak_kwargs['width'] = width
        
        if wlen is not None and wlen > 0:
            peak_kwargs['wlen'] = wlen
        
        if rel_height is not None and rel_height > 0:
            peak_kwargs['rel_height'] = rel_height
        
        try:
            peak_indices, properties = find_peaks(y_data, **peak_kwargs)
            return peak_indices, properties
        except Exception as e:
            print(f"峰值检测失败: {e}")
            return np.array([]), {}
    
    def match_peaks(self, reference_peaks: np.ndarray, target_peaks: np.ndarray,
                   reference_x: np.ndarray, target_x: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        匹配两个光谱的峰值
        
        Args:
            reference_peaks: 参考光谱的峰值索引
            target_peaks: 目标光谱的峰值索引
            reference_x: 参考光谱的X轴数据
            target_x: 目标光谱的X轴数据
        
        Returns:
            List of (ref_idx, target_idx, distance) 匹配对列表
        """
        if len(reference_peaks) == 0 or len(target_peaks) == 0:
            return []
        
        reference_positions = reference_x[reference_peaks]
        target_positions = target_x[target_peaks]
        
        matches = []
        used_target_indices = set()
        
        # 对每个参考峰值，找到最近的目标峰值
        for ref_idx, ref_pos in enumerate(reference_positions):
            distances = np.abs(target_positions - ref_pos)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            if min_dist <= self.tolerance and min_dist_idx not in used_target_indices:
                matches.append((reference_peaks[ref_idx], target_peaks[min_dist_idx], min_dist))
                used_target_indices.add(min_dist_idx)
        
        return matches
    
    def match_multiple_spectra(self, spectra_data: List[Dict], 
                              reference_index: int = -1,
                              mode: str = 'all_matched') -> Dict:
        """
        匹配多个光谱的峰值
        
        Args:
            spectra_data: 光谱数据列表，每个元素包含 {'x': x_data, 'y': y_data, 'color': color, 'label': label}
            reference_index: 参考光谱索引（-1表示最后一个）
            mode: 匹配模式
                - 'all_peaks': 只显示最下面谱峰检测的所有峰值
                - 'matched_only': 只显示最下面匹配到的谱峰
                - 'all_matched': 显示所有谱线都匹配的谱峰
                - 'top_display': 在最上方谱线显示最下方谱峰匹配到的峰值
        
        Returns:
            匹配结果字典
        """
        if len(spectra_data) == 0:
            return {}
        
        # 确定参考光谱
        if reference_index < 0:
            reference_index = len(spectra_data) + reference_index
        
        if reference_index >= len(spectra_data):
            reference_index = len(spectra_data) - 1
        
        reference_spectrum = spectra_data[reference_index]
        
        # 检测参考光谱的峰值
        ref_x = reference_spectrum['x']
        ref_y = reference_spectrum['y']
        
        # 使用默认参数检测峰值（实际应该从配置中获取）
        ref_peaks, _ = self.detect_peaks(ref_x, ref_y)
        
        if len(ref_peaks) == 0:
            return {
                'reference_index': reference_index,
                'reference_peaks': [],
                'matches': {},
                'mode': mode
            }
        
        # 检测所有光谱的峰值
        all_peaks = {}
        for i, spectrum in enumerate(spectra_data):
            x = spectrum['x']
            y = spectrum['y']
            peaks, _ = self.detect_peaks(x, y)
            all_peaks[i] = peaks
        
        # 根据模式进行匹配
        matches = {}
        
        if mode == 'all_peaks':
            # 只显示参考光谱的所有峰值
            matches[reference_index] = {
                'peaks': ref_peaks,
                'positions': ref_x[ref_peaks],
                'matched': True
            }
        
        elif mode == 'matched_only':
            # 只显示匹配到的峰值
            for i, spectrum in enumerate(spectra_data):
                if i == reference_index:
                    matches[i] = {
                        'peaks': ref_peaks,
                        'positions': ref_x[ref_peaks],
                        'matched': True
                    }
                else:
                    target_peaks = all_peaks.get(i, np.array([]))
                    if len(target_peaks) > 0:
                        match_pairs = self.match_peaks(ref_peaks, target_peaks, ref_x, spectrum['x'])
                        if match_pairs:
                            matched_ref_indices = [pair[0] for pair in match_pairs]
                            matches[i] = {
                                'peaks': np.array(matched_ref_indices),
                                'positions': ref_x[matched_ref_indices],
                                'matched': True,
                                'match_pairs': match_pairs
                            }
        
        elif mode == 'all_matched':
            # 显示所有谱线都匹配的峰值
            # 找到所有光谱都匹配的峰值
            common_peaks = set(range(len(ref_peaks)))
            
            for i, spectrum in enumerate(spectra_data):
                if i == reference_index:
                    continue
                
                target_peaks = all_peaks.get(i, np.array([]))
                if len(target_peaks) > 0:
                    match_pairs = self.match_peaks(ref_peaks, target_peaks, ref_x, spectrum['x'])
                    matched_ref_indices = {pair[0] for pair in match_pairs}
                    common_peaks &= matched_ref_indices
            
            if common_peaks:
                common_peak_indices = np.array([ref_peaks[i] for i in common_peaks])
                for i in range(len(spectra_data)):
                    matches[i] = {
                        'peaks': common_peak_indices,
                        'positions': ref_x[common_peak_indices],
                        'matched': True
                    }
        
        elif mode == 'top_display':
            # 在最上方谱线显示最下方谱峰匹配到的峰值
            top_index = 0
            matches[top_index] = {
                'peaks': ref_peaks,
                'positions': ref_x[ref_peaks],
                'matched': True,
                'display_mode': 'top_display'
            }
        
        return {
            'reference_index': reference_index,
            'reference_peaks': ref_peaks,
            'reference_positions': ref_x[ref_peaks],
            'all_peaks': all_peaks,
            'matches': matches,
            'mode': mode
        }


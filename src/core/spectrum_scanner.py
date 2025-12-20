"""
谱线扫描与堆叠偏移模块
支持扫描最后一次绘图的所有谱线，并可微调每根线的距离和指定匹配关系
"""
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy.interpolate import interp1d


class SpectrumScanner:
    """谱线扫描器"""
    
    def __init__(self):
        self.last_plot_data: List[Dict[str, Any]] = []
        self.scanned_spectra: List[Dict[str, Any]] = []
    
    def scan_last_plot(self, plot_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        扫描最后一次绘图的所有谱线
        
        Args:
            plot_data: 绘图数据列表，每个元素包含 {'x': x_data, 'y': y_data, 'label': label, 'color': color, ...}
        
        Returns:
            扫描后的谱线数据
        """
        self.last_plot_data = plot_data
        self.scanned_spectra = []
        
        for i, data in enumerate(plot_data):
            spectrum = {
                'index': i,
                'x': np.array(data.get('x', [])),
                'y': np.array(data.get('y', [])),
                'label': data.get('label', f'Spectrum {i+1}'),
                'color': data.get('color', 'blue'),
                'linewidth': data.get('linewidth', 1.0),
                'linestyle': data.get('linestyle', '-'),
                'offset': 0.0,  # 初始偏移为0
            }
            self.scanned_spectra.append(spectrum)
        
        return self.scanned_spectra
    
    def get_scanned_spectra(self) -> List[Dict[str, Any]]:
        """获取扫描的谱线数据"""
        return self.scanned_spectra
    
    def set_individual_offset(self, index: int, offset: float):
        """设置单个谱线的偏移"""
        if 0 <= index < len(self.scanned_spectra):
            self.scanned_spectra[index]['offset'] = offset
    
    def set_stack_offset(self, offset: float):
        """设置堆叠偏移（应用到所有谱线）"""
        for i, spectrum in enumerate(self.scanned_spectra):
            # 堆叠偏移：每个谱线按索引递增偏移
            spectrum['offset'] = i * offset
    
    def apply_custom_offsets(self, offsets: Dict):
        """
        应用自定义偏移（覆盖堆叠偏移）
        
        Args:
            offsets: 偏移字典，可以是 {index: offset} 或 {label: offset}
        """
        for key, offset in offsets.items():
            # 支持整数索引或字符串标签
            if isinstance(key, int):
                # 整数索引
                if 0 <= key < len(self.scanned_spectra):
                    self.scanned_spectra[key]['offset'] = offset
            elif isinstance(key, str):
                # 字符串标签，查找匹配的谱线
                for i, spectrum in enumerate(self.scanned_spectra):
                    if spectrum.get('label', '') == key:
                        self.scanned_spectra[i]['offset'] = offset
                        break
    
    def create_mapping(self, source_indices: List[int], target_indices: List[int]) -> List[Tuple[int, int]]:
        """
        创建谱线匹配映射
        
        Args:
            source_indices: 源谱线索引列表
            target_indices: 目标谱线索引列表
        
        Returns:
            匹配对列表 [(source_idx, target_idx), ...]
        """
        if len(source_indices) != len(target_indices):
            raise ValueError("源谱线索引和目标谱线索引数量必须相同")
        
        mappings = []
        for src_idx, tgt_idx in zip(source_indices, target_indices):
            if 0 <= src_idx < len(self.scanned_spectra) and 0 <= tgt_idx < len(self.scanned_spectra):
                mappings.append((src_idx, tgt_idx))
        
        return mappings
    
    def apply_mappings(self, mappings: List[Tuple[int, int]], 
                      interpolation: bool = True,
                      common_x: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        应用匹配映射，将源谱线对齐到目标谱线
        
        Args:
            mappings: 匹配对列表
            interpolation: 是否进行插值对齐
            common_x: 公共X轴（如果提供，所有谱线将插值到此X轴）
        
        Returns:
            对齐后的谱线数据
        """
        if len(self.scanned_spectra) == 0:
            return []
        
        # 确定公共X轴
        if common_x is None:
            # 使用所有谱线的X轴范围的并集
            all_x_min = min(spec['x'].min() for spec in self.scanned_spectra)
            all_x_max = max(spec['x'].max() for spec in self.scanned_spectra)
            # 使用最密集的采样
            min_dx = min(np.min(np.diff(spec['x'])) for spec in self.scanned_spectra if len(spec['x']) > 1)
            common_x = np.arange(all_x_min, all_x_max + min_dx, min_dx)
        
        aligned_spectra = []
        
        for src_idx, tgt_idx in mappings:
            source_spec = self.scanned_spectra[src_idx]
            target_spec = self.scanned_spectra[tgt_idx]
            
            if interpolation:
                # 插值到公共X轴
                if len(source_spec['x']) > 1:
                    interp_func = interp1d(source_spec['x'], source_spec['y'], 
                                         kind='linear', bounds_error=False, fill_value=0.0)
                    aligned_y = interp_func(common_x)
                else:
                    aligned_y = np.zeros_like(common_x)
                
                aligned_spectrum = {
                    'index': src_idx,
                    'x': common_x,
                    'y': aligned_y,
                    'label': source_spec['label'],
                    'color': source_spec['color'],
                    'linewidth': source_spec['linewidth'],
                    'linestyle': source_spec['linestyle'],
                    'offset': source_spec['offset'],
                    'target_index': tgt_idx,
                    'target_label': target_spec['label'],
                }
            else:
                # 不插值，直接使用原始数据
                aligned_spectrum = source_spec.copy()
                aligned_spectrum['target_index'] = tgt_idx
                aligned_spectrum['target_label'] = target_spec['label']
            
            aligned_spectra.append(aligned_spectrum)
        
        return aligned_spectra
    
    def get_spectrum_info(self, index: int) -> Optional[Dict[str, Any]]:
        """获取指定索引的谱线信息"""
        if 0 <= index < len(self.scanned_spectra):
            return self.scanned_spectra[index].copy()
        return None
    
    def clear(self):
        """清除扫描数据"""
        self.last_plot_data = []
        self.scanned_spectra = []


class StackOffsetManager:
    """堆叠偏移管理器"""
    
    def __init__(self, default_offset: float = 0.5):
        self.default_offset = default_offset
        self.individual_offsets: Dict[str, float] = {}
    
    def calculate_stack_offset(self, index: int, base_offset: Optional[float] = None) -> float:
        """
        计算堆叠偏移
        
        Args:
            index: 谱线索引
            base_offset: 基础偏移（如果为None，使用default_offset）
        
        Returns:
            计算后的偏移值
        """
        if base_offset is None:
            base_offset = self.default_offset
        
        return index * base_offset
    
    def get_total_offset(self, label: str, index: int, base_offset: Optional[float] = None) -> float:
        """
        获取总偏移（堆叠偏移 + 独立偏移）
        
        Args:
            label: 谱线标签（用于查找独立偏移）
            index: 谱线索引（用于计算堆叠偏移）
            base_offset: 基础偏移
        
        Returns:
            总偏移值
        """
        stack_offset = self.calculate_stack_offset(index, base_offset)
        individual_offset = self.individual_offsets.get(label, 0.0)
        return stack_offset + individual_offset
    
    def set_individual_offset(self, label: str, offset: float):
        """设置独立偏移"""
        self.individual_offsets[label] = offset
    
    def clear_individual_offsets(self):
        """清除所有独立偏移"""
        self.individual_offsets.clear()
    
    def apply_to_spectra(self, spectra: List[Dict[str, Any]], 
                        base_offset: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        将堆叠偏移应用到谱线列表
        
        Args:
            spectra: 谱线数据列表
            base_offset: 基础偏移
        
        Returns:
            应用偏移后的谱线数据
        """
        result = []
        for i, spectrum in enumerate(spectra):
            label = spectrum.get('label', f'Spectrum {i}')
            total_offset = self.get_total_offset(label, i, base_offset)
            
            modified_spectrum = spectrum.copy()
            modified_spectrum['y'] = spectrum['y'] + total_offset
            modified_spectrum['stack_offset'] = total_offset
            result.append(modified_spectrum)
        
        return result


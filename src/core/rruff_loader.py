"""
RRUFF库加载器和峰值匹配模块
支持加载RRUFF标准格式的光谱库文件，并进行峰值匹配识别
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
from io import StringIO


class RRUFFLibraryLoader:
    """RRUFF标准库加载器"""
    
    def __init__(self, library_folder=None, preprocess_params=None):
        """
        Args:
            library_folder: RRUFF库文件夹路径
            preprocess_params: 预处理参数字典，如果提供则对加载的光谱应用预处理
        """
        self.library_folder = library_folder
        self.preprocess_params = preprocess_params or {}
        self.peak_detection_params = {}  # 峰值检测参数（与主菜单一致）
        self.library_spectra = {}  # {name: {'x': wavenumbers, 'y': spectrum, 'y_raw': raw_spectrum, 'peaks': peaks, 'metadata': metadata}}
        if library_folder:
            self.load_library()
    
    def load_library(self, library_folder=None, preprocess_params=None, progress_callback=None, max_workers=None):
        """
        加载RRUFF库中的所有光谱文件（使用多线程并行加载）
        
        Args:
            library_folder: 库文件夹路径（如果提供则更新self.library_folder）
            preprocess_params: 预处理参数字典（如果提供则更新self.preprocess_params）
            progress_callback: 进度回调函数 callback(current, total, filename)
            max_workers: 最大工作线程数（默认使用CPU核心数）
        """
        if library_folder:
            self.library_folder = library_folder
        if preprocess_params is not None:
            self.preprocess_params = preprocess_params
        
        if not self.library_folder or not os.path.isdir(self.library_folder):
            return
        
        self.library_spectra.clear()
        
        # 支持多种文件格式
        files = glob.glob(os.path.join(self.library_folder, '*.txt')) + \
                glob.glob(os.path.join(self.library_folder, '*.csv')) + \
                glob.glob(os.path.join(self.library_folder, '*.dat'))
        
        total_files = len(files)
        if total_files == 0:
            return
        
        # 使用多线程并行加载
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)  # 最多8个线程，避免过多线程导致性能下降
        
        # 准备加载函数（需要传入实例的预处理参数）
        def load_single_file(file_path):
            """加载单个文件的函数"""
            try:
                # 读取文件内容，自动过滤所有无效行
                try:
                    # 先读取所有行
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    # 如果UTF-8失败，尝试其他编码
                    try:
                        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                            lines = f.readlines()
                    except:
                        return None
                except:
                    return None
                
                # 过滤掉所有无效行
                valid_data_rows = []
                for line in lines:
                    line_stripped = line.strip()
                    
                    # 跳过空行
                    if not line_stripped:
                        continue
                    
                    # 跳过包含特殊标记的行（##END=, ##, 注释等）
                    if (line_stripped.startswith('##') or 
                        line_stripped.startswith('#') or
                        '##END=' in line_stripped or
                        line_stripped.startswith('END')):
                        continue
                    
                    # 尝试解析这一行，检查是否包含有效的数字数据
                    try:
                        # 尝试分割行（支持空格、制表符、逗号等分隔符）
                        parts = line_stripped.replace(',', ' ').split()
                        if len(parts) < 2:
                            continue
                        
                        # 尝试将前两列转换为浮点数
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                        
                        # 检查是否为有效数字（不是NaN或Inf）
                        if not (np.isnan(x_val) or np.isnan(y_val) or 
                                np.isinf(x_val) or np.isinf(y_val)):
                            # 检查波数范围是否合理（通常在0-100000 cm^-1之间）
                            if 0 < x_val < 100000:
                                valid_data_rows.append(line)
                    except (ValueError, TypeError, IndexError):
                        # 如果无法解析，跳过这一行
                        continue
                
                # 如果没有任何有效数据行，返回None
                if len(valid_data_rows) == 0:
                    return None
                
                # 使用StringIO将有效行转换为DataFrame
                valid_content = ''.join(valid_data_rows)
                df = pd.read_csv(StringIO(valid_content), header=None, sep=None, engine='python')
                
                if df.shape[1] < 2:
                    return None
                
                # 尝试转换为浮点数
                try:
                    x = df.iloc[:, 0].values.astype(float)
                    y_raw = df.iloc[:, 1].values.astype(float)
                except (ValueError, TypeError):
                    return None
                
                # 移除NaN和无效值
                valid_mask = ~(np.isnan(x) | np.isnan(y_raw))
                x = x[valid_mask]
                y_raw = y_raw[valid_mask]
                
                if len(x) == 0:
                    return None
                
                # 确保X轴是降序（拉曼光谱通常从高波数到低波数）
                if len(x) > 1 and x[0] < x[-1]:
                    x = x[::-1]
                    y_raw = y_raw[::-1]
                
                # 应用预处理（如果提供了预处理参数）
                y = self._apply_preprocessing(x, y_raw.copy())
                
                # 检查预处理后的数据是否有效
                if len(y) == 0 or np.all(np.isnan(y)) or np.max(y) <= 0:
                    return None
                
                # 检测峰值（在预处理后的数据上，使用峰值检测参数）
                peaks = self._detect_peaks(x, y, peak_detection_params=self.peak_detection_params)
                
                # 提取文件名作为标识
                name = os.path.splitext(os.path.basename(file_path))[0]
                
                return {
                    'name': name,
                    'data': {
                        'x': x,
                        'y': y,  # 预处理后的数据
                        'y_raw': y_raw,  # 原始数据
                        'peaks': peaks,
                        'file_path': file_path,
                        'metadata': {'name': name}
                    }
                }
            except Exception as e:
                print(f"加载RRUFF库光谱失败 {file_path}: {e}")
                return None
        
        # 使用线程池并行加载
        loaded_count = 0
        successful_count = 0
        failed_count = 0
        failed_reasons = {}  # 统计失败原因
        # 使用锁确保线程安全（虽然Python字典操作在GIL保护下是线程安全的，但为了保险起见）
        from threading import Lock
        spectra_lock = Lock()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(load_single_file, file_path): file_path 
                             for file_path in files}
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                loaded_count += 1
                
                try:
                    result = future.result()
                    if result is not None:
                        # 使用锁保护字典操作（虽然通常不需要，但为了保险）
                        with spectra_lock:
                            # 检查是否已经有同名光谱（避免重复）
                            name = result['name']
                            
                            # 过滤processed版本：如果存在xxx-processed和xxx-raw，只保留xxx-raw
                            # 检查是否是processed版本
                            is_processed = 'processed' in name.lower() or '-processed' in name.lower()
                            if is_processed:
                                # 尝试找到对应的raw版本
                                raw_name = name.replace('-processed', '').replace('processed', 'raw')
                                if raw_name in self.library_spectra:
                                    # 如果raw版本已存在，跳过processed版本
                                    successful_count += 1
                                    continue
                            
                            # 如果当前是raw版本，检查是否有对应的processed版本需要删除
                            if 'raw' in name.lower() or '-raw' in name.lower():
                                processed_name = name.replace('-raw', '').replace('raw', 'processed')
                                if processed_name in self.library_spectra:
                                    # 删除processed版本
                                    del self.library_spectra[processed_name]
                            
                            if name in self.library_spectra:
                                # 如果已存在同名光谱，使用更完整的路径作为key
                                name = os.path.basename(file_path)
                            self.library_spectra[name] = result['data']
                        successful_count += 1
                    else:
                        failed_count += 1
                        failed_reasons['返回None'] = failed_reasons.get('返回None', 0) + 1
                except Exception as e:
                    error_msg = str(e)[:100]  # 截取前100个字符
                    failed_reasons[error_msg] = failed_reasons.get(error_msg, 0) + 1
                    if failed_count < 10:  # 只打印前10个错误，避免输出太多
                        print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
                    failed_count += 1
                
                # 更新进度
                if progress_callback:
                    try:
                        progress_callback(loaded_count, total_files, os.path.basename(file_path))
                    except:
                        pass
        
        # 打印加载统计信息
        final_count = len(self.library_spectra)
        print(f"RRUFF库加载完成: 总文件数 {total_files}, 成功加载 {successful_count}, 失败 {failed_count}, 最终光谱数 {final_count}")
        
        # 打印失败原因统计（前10个最常见的原因）
        if failed_reasons:
            print(f"\n失败原因统计（前10个）:")
            sorted_reasons = sorted(failed_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
            for reason, count in sorted_reasons:
                print(f"  {reason}: {count} 次")
        
        # 如果最终数量与成功数量不一致，打印警告
        if final_count != successful_count:
            print(f"警告: 最终光谱数 {final_count} 与成功加载数 {successful_count} 不一致！")
        
        # 如果成功数量明显少于文件数量，打印警告
        if successful_count < total_files * 0.1:  # 如果成功率低于10%
            print(f"\n警告: 成功率过低 ({successful_count}/{total_files} = {successful_count/total_files*100:.1f}%)")
            print(f"可能的原因:")
            print(f"  1. 文件格式不兼容")
            print(f"  2. QC检查阈值过高（当前阈值: {self.preprocess_params.get('qc_threshold', '未设置')}）")
            print(f"  3. 文件编码问题")
            print(f"  4. 数据格式问题（需要至少2列数据）")
    
    def _auto_detect_skip_rows(self, file_path):
        """
        自动检测应该跳过的行数（已废弃）
        
        注意：现在load_single_file已经自动过滤所有无效行（开头、中间、末尾），
        这个函数保留是为了兼容性，但不再需要实际检测，直接返回0即可。
        
        Returns:
            skip_rows: 总是返回0，因为过滤在load_single_file中进行
        """
        # 由于load_single_file现在会自动过滤所有无效行，这里直接返回0
        return 0
    
    def _apply_preprocessing(self, x, y):
        """
        应用预处理到光谱数据
        
        Args:
            x: 波数数组
            y: 强度数组
            
        Returns:
            y_processed: 预处理后的强度数组
        """
        if not self.preprocess_params:
            return y
        
        from src.core.preprocessor import DataPreProcessor
        
        y_proc = y.astype(float)
        
        # 获取预处理参数
        qc_enabled = self.preprocess_params.get('qc_enabled', False)
        qc_threshold = self.preprocess_params.get('qc_threshold', 5.0)
        is_be_correction = self.preprocess_params.get('is_be_correction', False)
        be_temp = self.preprocess_params.get('be_temp', 300.0)
        is_smoothing = self.preprocess_params.get('is_smoothing', False)
        smoothing_window = self.preprocess_params.get('smoothing_window', 15)
        smoothing_poly = self.preprocess_params.get('smoothing_poly', 3)
        is_baseline_als = self.preprocess_params.get('is_baseline_als', False)
        als_lam = self.preprocess_params.get('als_lam', 10000)
        als_p = self.preprocess_params.get('als_p', 0.005)
        normalization_mode = self.preprocess_params.get('normalization_mode', 'None')
        global_transform_mode = self.preprocess_params.get('global_transform_mode', '无')
        global_log_base_text = self.preprocess_params.get('global_log_base', '10')
        global_log_base = float(global_log_base_text) if global_log_base_text == '10' else np.e
        global_log_offset = self.preprocess_params.get('global_log_offset', 1.0)
        global_sqrt_offset = self.preprocess_params.get('global_sqrt_offset', 0.0)
        global_y_offset = self.preprocess_params.get('global_y_offset', 0.0)
        is_derivative = self.preprocess_params.get('is_derivative', False)
        
        # QC检查
        if qc_enabled and np.max(y_proc) < qc_threshold:
            return y_proc
        
        # 1. BE校正
        if is_be_correction:
            y_proc = DataPreProcessor.apply_bose_einstein_correction(x, y_proc, be_temp)
        
        # 2. 平滑
        if is_smoothing:
            y_proc = DataPreProcessor.apply_smoothing(y_proc, smoothing_window, smoothing_poly)
        
        # 3. 基线校正
        if is_baseline_als:
            b = DataPreProcessor.apply_baseline_als(y_proc, als_lam, als_p)
            y_proc = y_proc - b
            y_proc[y_proc < 0] = 0
        
        # 4. 归一化
        if normalization_mode == 'max':
            y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
        elif normalization_mode == 'area':
            y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
        elif normalization_mode == 'snv':
            y_proc = DataPreProcessor.apply_snv(y_proc)
        
        # 5. 全局动态变换
        if global_transform_mode == '对数变换 (Log)':
            y_proc = DataPreProcessor.apply_log_transform(y_proc, base=global_log_base, offset=global_log_offset)
        elif global_transform_mode == '平方根变换 (Sqrt)':
            y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=global_sqrt_offset)
        
        # 6. 二次导数
        if is_derivative:
            d1 = np.gradient(y_proc, x)
            y_proc = np.gradient(d1, x)
        
        # 7. 整体Y轴偏移
        y_proc = y_proc + global_y_offset
        
        return y_proc
    
    def update_preprocessing(self, preprocess_params, peak_detection_params=None, progress_callback=None):
        """
        更新预处理参数并重新处理所有已加载的光谱
        
        Args:
            preprocess_params: 预处理参数字典
            peak_detection_params: 峰值检测参数字典（如果提供，将使用这些参数检测峰值）
            progress_callback: 进度回调函数 callback(current, total, message)
        
        Returns:
            bool: 如果参数真正改变并重新处理了光谱，返回True；否则返回False
        """
        # 检查参数是否真正改变
        preprocess_changed = (self.preprocess_params != preprocess_params)
        peak_detection_changed = (self.peak_detection_params != (peak_detection_params or {}))
        
        # 如果参数没有改变，直接返回
        if not preprocess_changed and not peak_detection_changed:
            return False
        
        self.preprocess_params = preprocess_params
        self.peak_detection_params = peak_detection_params or {}
        
        # 获取需要处理的光谱列表
        spectra_to_process = []
        if preprocess_changed:
            # 预处理参数改变，需要重新处理所有光谱
            for name, spectrum in self.library_spectra.items():
                if 'y_raw' in spectrum:
                    spectra_to_process.append((name, spectrum, 'full'))
        elif peak_detection_changed:
            # 只有峰值检测参数改变，只重新检测峰值（不重新预处理，更快）
            for name, spectrum in self.library_spectra.items():
                if 'y' in spectrum:
                    spectra_to_process.append((name, spectrum, 'peaks_only'))
        
        total = len(spectra_to_process)
        
        # 只在参数改变时才重新处理光谱
        # 如果只有峰值检测参数改变，只重新检测峰值（更快）
        for idx, (name, spectrum, process_type) in enumerate(spectra_to_process):
            # 更新进度
            if progress_callback:
                try:
                    if process_type == 'full':
                        message = f"重新处理: {name[:30]}..."
                    else:
                        message = f"重新检测峰值: {name[:30]}..."
                    progress_callback(idx + 1, total, message)
                except:
                    pass
            
            if process_type == 'full':
                # 预处理参数改变，需要重新处理所有光谱
                spectrum['y'] = self._apply_preprocessing(spectrum['x'], spectrum['y_raw'].copy())
                # 重新检测峰值
                spectrum['peaks'] = self._detect_peaks(spectrum['x'], spectrum['y'], peak_detection_params=self.peak_detection_params)
            else:
                # 只有峰值检测参数改变，只重新检测峰值（不重新预处理，更快）
                # 使用已有的预处理后的光谱重新检测峰值
                spectrum['peaks'] = self._detect_peaks(spectrum['x'], spectrum['y'], peak_detection_params=self.peak_detection_params)
        
        # 最终进度
        if progress_callback:
            try:
                progress_callback(total, total, "完成")
            except:
                pass
        
        return True
    
    def _detect_peaks(self, x, y, prominence_factor=0.01, distance_factor=0.01, peak_detection_params=None):
        """
        检测光谱峰值（使用主菜单的峰值检测参数，确保与查询光谱一致）
        
        Args:
            x: 波数数组
            y: 强度数组
            prominence_factor: 峰值突出度因子（相对于最大值，默认0.01=1%，仅在peak_detection_params未提供时使用）
            distance_factor: 峰值距离因子（相对于数据长度，默认0.01=1%，仅在peak_detection_params未提供时使用）
            peak_detection_params: 峰值检测参数字典（如果提供，将使用这些参数，与主菜单一致）
        
        Returns:
            peaks: 峰值索引数组
            peak_wavenumbers: 峰值波数数组
        """
        if len(y) == 0:
            return np.array([]), np.array([])
        
        y_max = np.max(y)
        y_min = np.min(y)
        y_range = y_max - y_min
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        # 如果提供了峰值检测参数，使用它们（与主菜单一致）
        if peak_detection_params:
            peak_height = peak_detection_params.get('peak_height_threshold', 0.0)
            peak_distance = peak_detection_params.get('peak_distance_min', 10)
            peak_prominence = peak_detection_params.get('peak_prominence', None)
            peak_width = peak_detection_params.get('peak_width', None)
            peak_wlen = peak_detection_params.get('peak_wlen', None)
            peak_rel_height = peak_detection_params.get('peak_rel_height', None)
            
            peak_kwargs = {}
            
            # 处理height参数（与主菜单逻辑一致）
            if peak_height == 0:
                if y_max > 0:
                    peak_height = y_max * 0.0001  # 0.01%
                else:
                    peak_height = abs(y_mean) + y_std * 0.05
            if peak_height > y_range * 2 and y_range > 0:
                peak_height = y_max * 0.0001
            if peak_height != 0:
                peak_kwargs['height'] = peak_height
            
            # 处理distance参数
            if peak_distance == 0:
                peak_distance = max(1, int(len(y) * 0.001))  # 0.1%
            if peak_distance > len(y) * 0.5:
                peak_distance = max(1, int(len(y) * 0.001))
            peak_distance = max(1, peak_distance)
            
            # 如果height是负数或极小值，不使用distance
            if 'height' in peak_kwargs:
                height_val = peak_kwargs['height']
                if height_val < 0 or (y_max > 0 and height_val < y_max * 0.001):
                    # 不使用distance
                    pass
                else:
                    peak_kwargs['distance'] = peak_distance
            else:
                peak_kwargs['distance'] = peak_distance
            
            # 处理prominence参数
            if peak_prominence is not None and peak_prominence != 0:
                if peak_prominence > y_range * 2 and y_range > 0:
                    peak_prominence = y_range * 0.001
                peak_kwargs['prominence'] = peak_prominence
            
            # 处理其他参数
            if peak_width is not None and peak_width > 0:
                peak_kwargs['width'] = peak_width
            if peak_wlen is not None and peak_wlen > 0:
                if peak_wlen > len(y) * 0.5:
                    peak_wlen = max(1, int(len(y) * 0.3))
                peak_kwargs['wlen'] = peak_wlen
            if peak_rel_height is not None and peak_rel_height > 0:
                peak_kwargs['rel_height'] = peak_rel_height
            
            # 确保至少有一个参数
            if len(peak_kwargs) == 0:
                if y_max > 0:
                    peak_kwargs = {'height': y_max * 0.0001}
                else:
                    peak_kwargs = {'height': abs(y_mean) + y_std * 0.05}
            
            try:
                peaks, properties = find_peaks(y, **peak_kwargs)
                peak_wavenumbers = x[peaks]
                return peaks, peak_wavenumbers
            except:
                # 如果失败，使用默认方法
                pass
        
        # 如果没有提供峰值检测参数，使用默认方法（向后兼容）
        if y_range > 0:
            prominence_threshold = max(y_range * prominence_factor, y_max * 0.005)
        else:
            prominence_threshold = abs(y_max) * prominence_factor if y_max != 0 else 0.01
        
        distance_threshold = max(1, int(len(y) * distance_factor))
        
        try:
            try:
                peaks, properties = find_peaks(
                    y,
                    prominence=prominence_threshold,
                    distance=distance_threshold
                )
            except:
                height_threshold = y_max * 0.005 if y_max > 0 else 0
                peaks, properties = find_peaks(
                    y,
                    height=height_threshold,
                    distance=distance_threshold
                )
            peak_wavenumbers = x[peaks]
            return peaks, peak_wavenumbers
        except:
            return np.array([]), np.array([])
    
    def get_all_spectra_names(self):
        """获取所有已加载的光谱名称列表"""
        return list(self.library_spectra.keys())
    
    def get_spectrum(self, name):
        """获取指定名称的光谱数据"""
        return self.library_spectra.get(name)
    
    def remove_spectrum(self, name):
        """从库中移除指定光谱"""
        if name in self.library_spectra:
            del self.library_spectra[name]
    
    def get_filtered_library(self, excluded_names=None):
        """
        获取过滤后的库（排除指定名称）
        
        Args:
            excluded_names: 要排除的光谱名称列表
        
        Returns:
            filtered_spectra: 过滤后的光谱字典
        """
        if excluded_names is None:
            excluded_names = []
        
        filtered = {}
        for name, spectrum in self.library_spectra.items():
            if name not in excluded_names:
                filtered[name] = spectrum
        return filtered


class PeakMatcher:
    """峰值匹配器：匹配实验光谱峰值与RRUFF库峰值"""
    
    def __init__(self, tolerance=5.0):
        """
        Args:
            tolerance: 峰值匹配容差（cm^-1）
        """
        self.tolerance = tolerance
    
    def match_peaks(self, query_peaks, library_peaks, tolerance=None):
        """
        匹配查询峰值与库峰值（统一匹配逻辑，不区分自身匹配）
        
        Args:
            query_peaks: 查询光谱的峰值波数数组
            library_peaks: 库光谱的峰值波数数组
            tolerance: 匹配容差（如果提供则覆盖self.tolerance）
        
        Returns:
            matches: 匹配的峰值对列表 [(query_peak, library_peak, distance), ...]
            match_score: 匹配分数（匹配数/总查询峰值数）
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        if len(query_peaks) == 0 and len(library_peaks) == 0:
            # 如果两者都没有峰值，返回100%匹配（可能是相同的光谱）
            return [], 1.0
        
        if len(query_peaks) == 0 or len(library_peaks) == 0:
            # 如果只有一个有峰值，返回0匹配
            return [], 0.0
        
        matches = []
        matched_lib_indices = set()
        
        # 对查询峰值进行排序，优先匹配最接近的峰值
        sorted_query_indices = np.argsort(query_peaks)
        
        for q_idx in sorted_query_indices:
            q_peak = query_peaks[q_idx]
            # 找到最近的库峰值
            distances = np.abs(library_peaks - q_peak)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            if min_dist <= tolerance and min_idx not in matched_lib_indices:
                matches.append((q_peak, library_peaks[min_idx], min_dist))
                matched_lib_indices.add(min_idx)
        
        # 统一匹配分数计算：使用对称匹配分数
        # 如果峰值数量相同且所有峰值都匹配，返回100%
        if len(query_peaks) == len(library_peaks) and len(matches) == len(query_peaks):
            match_score = 1.0
        else:
            # 使用对称匹配分数：匹配数 / max(查询峰值数, 库峰值数)
            total_peaks = max(len(query_peaks), len(library_peaks))
            match_score = len(matches) / total_peaks if total_peaks > 0 else 0.0
        
        return matches, match_score
    
    def find_best_matches(self, query_wavenumbers, query_spectrum, query_peaks, library_loader, top_k=5, excluded_names=None, progress_callback=None, max_workers=None):
        """
        在库中查找最佳匹配的光谱
        
        Args:
            query_wavenumbers: 查询光谱的波数数组
            query_spectrum: 查询光谱的强度数组
            query_peaks: 查询光谱的峰值波数数组
            library_loader: RRUFFLibraryLoader实例
            top_k: 返回前k个最佳匹配
            excluded_names: 要排除的光谱名称列表
        
        Returns:
            best_matches: 列表 [(name, match_score, matches, spectrum_data), ...]
        """
        if not library_loader.library_spectra:
            return []
        
        # 获取过滤后的库
        filtered_library = library_loader.get_filtered_library(excluded_names)
        
        match_results = []
        total_items = len(filtered_library)
        processed_count = 0
        
        for name, lib_data in filtered_library.items():
            processed_count += 1
            if progress_callback:
                try:
                    progress_callback(processed_count, total_items, name)
                except:
                    pass
            lib_peaks = lib_data['peaks'][1]  # 获取峰值波数数组
            lib_x = lib_data['x']
            lib_y = lib_data['y']
            
            # 峰值匹配（统一匹配逻辑，不区分自身匹配）
            matches, peak_match_score = self.match_peaks(query_peaks, lib_peaks)
            
            # 计算光谱相似度（使用相关系数）
            spectrum_similarity = 0.0
            try:
                # 插值对齐到公共波数范围
                from scipy.interpolate import interp1d
                common_x_min = max(query_wavenumbers.min(), lib_x.min())
                common_x_max = min(query_wavenumbers.max(), lib_x.max())
                
                if common_x_min < common_x_max:
                    # 创建公共波数轴
                    common_x = np.linspace(common_x_min, common_x_max, min(len(query_wavenumbers), len(lib_x)))
                    
                    # 插值对齐
                    f_query = interp1d(query_wavenumbers, query_spectrum, kind='linear', fill_value=0, bounds_error=False)
                    f_lib = interp1d(lib_x, lib_y, kind='linear', fill_value=0, bounds_error=False)
                    
                    query_aligned = f_query(common_x)
                    lib_aligned = f_lib(common_x)
                    
                    # 计算相关系数
                    if np.std(query_aligned) > 0 and np.std(lib_aligned) > 0:
                        correlation = np.corrcoef(query_aligned, lib_aligned)[0, 1]
                        spectrum_similarity = max(0.0, correlation)  # 确保非负
            except:
                pass
            
            # 综合匹配分数：峰值匹配分数和光谱相似度的加权平均
            # 峰值匹配权重0.6，光谱相似度权重0.4
            combined_score = 0.6 * peak_match_score + 0.4 * spectrum_similarity
            
            match_results.append({
                'name': name,
                'match_score': combined_score,
                'peak_match_score': peak_match_score,
                'spectrum_similarity': spectrum_similarity,
                'matches': matches,
                'spectrum_data': lib_data
            })
        
        # 按综合匹配分数排序
        match_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        return match_results[:top_k]
    
    def find_best_combination_matches(self, query_wavenumbers, query_spectrum, query_peaks, library_loader, 
                                      max_phases=3, top_k=10, excluded_names=None, use_gpu=False, progress_callback=None, 
                                      min_peak_coverage=0.8):
        """
        查找最佳的多物相组合匹配（将多个RRUFF光谱组合来匹配查询光谱）
        
        Args:
            query_wavenumbers: 查询光谱的波数数组
            query_spectrum: 查询光谱的强度数组
            query_peaks: 查询光谱的峰值波数数组
            library_loader: RRUFFLibraryLoader实例
            max_phases: 最大物相数量（组合中最多包含的物相数）
            top_k: 返回前k个最佳匹配组合
            excluded_names: 要排除的光谱名称列表
            use_gpu: 是否使用GPU加速（需要安装cupy或torch）
        
        Returns:
            best_combinations: 列表 [{'phases': [name1, name2, ...], 'ratios': [r1, r2, ...], 
                                     'match_score': score, 'peak_match_score': peak_score, 
                                     'spectrum_similarity': sim, 'combined_peaks': peaks, 
                                     'combined_spectrum': spectrum}, ...]
        """
        if not library_loader.library_spectra:
            return []
        
        # 尝试导入GPU库
        xp = np  # 默认使用numpy
        if use_gpu:
            try:
                import cupy as cp
                # 检查CuPy是否能正常使用GPU
                try:
                    _ = cp.array([1, 2, 3])  # 测试GPU是否可用
                    xp = cp
                    print("使用GPU加速（CuPy）")
                except Exception as e:
                    print(f"CuPy GPU不可用: {e}，使用CPU")
            except ImportError:
                try:
                    import torch
                    if torch.cuda.is_available():
                        # 测试PyTorch GPU是否可用
                        try:
                            _ = torch.tensor([1, 2, 3]).cuda()
                            xp = torch
                            print("使用GPU加速（PyTorch）")
                        except Exception as e:
                            print(f"PyTorch GPU不可用: {e}，使用CPU")
                    else:
                        print("GPU不可用，使用CPU")
                except ImportError:
                    print("GPU库未安装，使用CPU")
        else:
            # 即使use_gpu=False，也尝试检测GPU可用性并提示用户
            try:
                import cupy as cp
                try:
                    _ = cp.array([1, 2, 3])
                    print("提示：检测到可用GPU（CuPy），可以在匹配时启用GPU加速")
                except:
                    pass
            except ImportError:
                try:
                    import torch
                    if torch.cuda.is_available():
                        print("提示：检测到可用GPU（PyTorch），可以在匹配时启用GPU加速")
                except:
                    pass
        
        from scipy.interpolate import interp1d
        from itertools import combinations
        
        # 获取过滤后的库
        filtered_library = library_loader.get_filtered_library(excluded_names)
        library_names = list(filtered_library.keys())
        
        if len(library_names) == 0:
            return []
        
        # 先获取单物相匹配结果，选择前N个作为候选
        # 使用并行处理加速单物相匹配，增加候选数量以确保不遗漏重要匹配
        single_matches = self.find_best_matches(query_wavenumbers, query_spectrum, query_peaks, 
                                               library_loader, top_k=min(100, len(library_names)), 
                                               excluded_names=excluded_names,
                                               progress_callback=None,  # 组合匹配时不显示单物相进度
                                               max_workers=min(32, multiprocessing.cpu_count()))  # 充分利用32线程
        
        # 优先选择峰值匹配数最多的候选（而不是只看综合分数）
        # 按峰值匹配数排序，然后按综合分数排序
        single_matches_sorted = sorted(single_matches, 
                                      key=lambda m: (m.get('peak_match_score', 0.0), m.get('match_score', 0.0)), 
                                      reverse=True)
        candidate_names = [m['name'] for m in single_matches_sorted[:min(50, len(single_matches_sorted))]]
        
        # 如果候选太少，使用所有库光谱
        if len(candidate_names) < max_phases:
            candidate_names = library_names[:min(50, len(library_names))]  # 增加候选数量以提高匹配精度
        
        # 预计算所有候选光谱的插值结果（避免重复插值）
        # 优化：对于CPU模式，直接存储numpy数组；对于GPU模式，存储GPU数组
        precomputed_spectra = {}
        precomputed_peaks = {}
        
        # 预转换查询光谱（避免在循环中重复转换）
        query_spectrum_np = query_spectrum  # numpy版本
        query_spectrum_gpu = query_spectrum  # GPU版本（如果使用GPU）
        
        if use_gpu and xp != np:
            if xp == cp:
                query_spectrum_gpu = cp.asarray(query_spectrum)
            elif xp == torch:
                query_spectrum_gpu = torch.from_numpy(query_spectrum).float().cuda()
        
        for phase_name in candidate_names:
            lib_data = filtered_library[phase_name]
            lib_x = lib_data['x']
            lib_y = lib_data['y']
            lib_peaks = lib_data['peaks'][1]
            
            # 插值对齐到查询光谱的波数轴
            f_interp = interp1d(lib_x, lib_y, kind='linear', fill_value=0, bounds_error=False)
            lib_y_aligned = f_interp(query_wavenumbers)
            
            # 对于CPU模式，直接存储numpy数组（避免不必要的GPU转换）
            # 对于GPU模式，存储GPU数组以便后续使用
            if use_gpu and xp != np:
                if xp == cp:
                    lib_y_aligned_gpu = cp.asarray(lib_y_aligned)
                    # CPU和GPU版本都存储（CPU版本用于NNLS求解）
                    precomputed_spectra[phase_name] = {'cpu': lib_y_aligned, 'gpu': lib_y_aligned_gpu}
                elif xp == torch:
                    lib_y_aligned_gpu = torch.from_numpy(lib_y_aligned).float().cuda()
                    precomputed_spectra[phase_name] = {'cpu': lib_y_aligned, 'gpu': lib_y_aligned_gpu}
            else:
                # CPU模式：只存储numpy数组
                precomputed_spectra[phase_name] = lib_y_aligned
            
            precomputed_peaks[phase_name] = lib_peaks
        
        combination_results = []
        
        # 使用生成器分批生成组合，避免内存爆炸
        # 不一次性生成所有组合，而是使用生成器按需生成
        from itertools import combinations, islice
        from math import comb
        
        # 多物相组合匹配：从1物相开始，自动调整物相数量
        # 增加组合数限制，确保有足够的多物相组合
        max_total_combinations = 300 if use_gpu else 200  # 总组合数限制
        max_combinations_per_phase = 150 if use_gpu else 100  # 每相位的最大组合数
        
        def generate_combinations_batch():
            """生成器：分批生成组合，避免一次性加载所有组合到内存"""
            total_generated = 0
            
            # 从1物相开始，自动调整物相数量（根据峰值数量动态调整）
            # 优先生成能匹配更多峰值的组合
            for n_phases in range(1, max_phases + 1):
                if n_phases > len(candidate_names):
                    break
                
                # 计算组合数
                total_combinations = comb(len(candidate_names), n_phases)
                
                # 如果组合数太多，分批生成（使用islice避免一次性生成）
                if total_combinations > max_combinations_per_phase:
                    # 使用islice分批生成，避免一次性生成所有组合
                    combo_gen = combinations(candidate_names, n_phases)
                    batch_size = max_combinations_per_phase
                    
                    while total_generated < max_total_combinations:
                        # 使用islice每次只取一批，不一次性生成所有
                        batch = list(islice(combo_gen, batch_size))
                        if not batch:
                            break
                        for phase_combo in batch:
                            if total_generated >= max_total_combinations:
                                return
                            yield (n_phases, phase_combo)
                            total_generated += 1
                        if len(batch) < batch_size:
                            break
                else:
                    # 组合数不多，直接使用生成器（不转换为列表）
                    for phase_combo in combinations(candidate_names, n_phases):
                        if total_generated >= max_total_combinations:
                            return
                        yield (n_phases, phase_combo)
                        total_generated += 1
                
                if total_generated >= max_total_combinations:
                    break
        
        # 不转换为列表，直接使用生成器，在并行处理时按需生成
        combinations_generator = generate_combinations_batch()
        
        # 估算总组合数（用于进度显示），但不实际生成所有组合
        # 从1物相开始估算
        estimated_total = 0
        for n_phases in range(1, min(max_phases + 1, 5)):  # 估算前4个物相组合（1,2,3,4物相）
            if n_phases <= len(candidate_names):
                total_comb = comb(len(candidate_names), n_phases)
                estimated_total += min(total_comb, max_combinations_per_phase)
        estimated_total = min(estimated_total, max_total_combinations)
        
        # 使用多线程/多进程加速组合匹配
        import threading
        
        def process_single_combination_fast(args):
            """优化的单组合处理函数（使用线性最小二乘而非迭代优化，减少内存复制）"""
            n_phases, phase_combo = args
            try:
                # 使用预计算的光谱数据，避免重复插值
                phase_spectra_np = []
                phase_peaks_list = []
                
                for name in phase_combo:
                    spec = precomputed_spectra[name]
                    # 根据存储格式提取numpy数组
                    if isinstance(spec, dict):
                        # GPU模式：使用CPU版本进行NNLS求解
                        spec_np = spec['cpu']
                    else:
                        # CPU模式：直接使用
                        spec_np = spec
                    phase_spectra_np.append(spec_np)
                    phase_peaks_list.append(precomputed_peaks[name])
                
                # 构建矩阵：A @ ratios = query_spectrum
                # 使用线性最小二乘求解（比迭代优化快得多）
                A = np.column_stack(phase_spectra_np)
                
                # 使用非负最小二乘（NNLS）求解
                from scipy.optimize import nnls
                optimal_ratios, residual = nnls(A, query_spectrum_np)
                
                # 归一化比例
                ratio_sum = np.sum(optimal_ratios)
                if ratio_sum > 1e-10:  # 避免除零
                    optimal_ratios = optimal_ratios / ratio_sum
                else:
                    optimal_ratios = np.ones(n_phases) / n_phases
                
                # 构建组合光谱（直接使用numpy矩阵乘法，避免循环）
                combined_spectrum_np = A @ optimal_ratios
                
                # 合并峰值（所有物相的峰值）
                all_combined_peaks = np.concatenate(phase_peaks_list)
                all_combined_peaks = np.unique(np.sort(all_combined_peaks))
                
                # 计算峰值匹配分数
                matches, peak_match_score = self.match_peaks(query_peaks, all_combined_peaks)

                # 统计已匹配 / 未匹配的查询峰值（用于后续在界面上高亮或提示）
                if len(matches) > 0:
                    matched_query_peaks = np.array([m[0] for m in matches])
                    unmatched_query_peaks = np.setdiff1d(query_peaks, matched_query_peaks)
                else:
                    matched_query_peaks = np.array([], dtype=float)
                    unmatched_query_peaks = np.array(query_peaks, copy=True)
                
                # 计算光谱相似度（使用向量化操作加速）
                spectrum_similarity = 0.0
                try:
                    query_std = np.std(query_spectrum)
                    combined_std = np.std(combined_spectrum_np)
                    if query_std > 0 and combined_std > 0:
                        # 使用向量化计算相关系数
                        query_norm = (query_spectrum - np.mean(query_spectrum)) / query_std
                        combined_norm = (combined_spectrum_np - np.mean(combined_spectrum_np)) / combined_std
                        correlation = np.dot(query_norm, combined_norm) / len(query_spectrum)
                        spectrum_similarity = max(0.0, correlation)
                except:
                    pass
                
                # 计算综合匹配分数
                # 优先匹配所有峰值：如果还有未匹配峰值，降低分数；如果所有峰值都匹配，给予奖励
                num_unmatched = len(unmatched_query_peaks)
                num_total_peaks = len(query_peaks)
                num_matched = len(matched_query_peaks)
                
                # 峰值覆盖率奖励：优先奖励能匹配更多峰值的组合
                peak_coverage_bonus = 0.0
                if num_total_peaks > 0:
                    peak_coverage_ratio = num_matched / num_total_peaks
                    # 如果所有峰值都匹配，给予额外奖励
                    if num_unmatched == 0:
                        peak_coverage_bonus = 0.3  # 额外奖励30%（提高奖励）
                    elif peak_coverage_ratio >= 0.95:
                        peak_coverage_bonus = 0.15  # 95%以上匹配，奖励15%
                    elif peak_coverage_ratio >= 0.9:
                        peak_coverage_bonus = 0.1  # 90%以上匹配，奖励10%
                    elif peak_coverage_ratio < 0.8:
                        # 如果匹配率低于80%，惩罚
                        peak_coverage_bonus = -0.15 * (0.8 - peak_coverage_ratio)
                
                # 多物相组合奖励：如果使用多物相且匹配效果好，给予额外奖励
                multi_phase_bonus = 0.0
                if n_phases > 1 and peak_coverage_ratio >= 0.9:
                    # 多物相组合且匹配效果好，给予额外奖励（鼓励使用多物相）
                    multi_phase_bonus = 0.05 * min(n_phases - 1, 3) / 3  # 最多奖励5%，随物相数增加
                
                # 基础分数：峰值匹配权重0.7，光谱相似度权重0.3（更重视峰值匹配）
                base_score = 0.7 * peak_match_score + 0.3 * spectrum_similarity
                # 加上峰值覆盖率奖励和多物相奖励
                combined_score = base_score + peak_coverage_bonus + multi_phase_bonus
                # 确保分数在合理范围内
                combined_score = max(0.0, min(1.0, combined_score))
                
                return {
                    'phases': list(phase_combo),
                    'ratios': optimal_ratios.tolist(),
                    'match_score': combined_score,
                    'peak_match_score': peak_match_score,
                    'spectrum_similarity': spectrum_similarity,
                    'combined_peaks': all_combined_peaks,
                    'combined_spectrum': combined_spectrum_np,
                    'matches': matches,
                    'matched_peaks': matched_query_peaks,
                    'unmatched_peaks': unmatched_query_peaks,
                    'num_matched_peaks': len(matched_query_peaks),
                    'num_unmatched_peaks': num_unmatched,
                }
            except Exception as e:
                # 如果组合失败，返回None
                return None
        
        # 使用进程池并行处理（避免GIL限制，更适合CPU密集型任务）
        # 优化并行处理配置
        # Windows上ProcessPoolExecutor可能有性能问题，优先使用ThreadPoolExecutor
        import threading
        import sys
        
        # 计算最优worker数量
        cpu_count = multiprocessing.cpu_count()
        # 使用估算的总组合数来计算worker数量（因为生成器不能直接获取长度）
        total_combinations = estimated_total
        
        # 初始化锁（用于线程安全的结果收集）
        results_lock = threading.Lock()
        
        if use_gpu and xp != np:
            # GPU模式：使用线程池，充分利用GPU并行能力
            # GPU模式下，线程数可以更多（因为GPU操作是异步的）
            max_workers = min(max_total_combinations, max(32, cpu_count * 2))
            executor_class = ThreadPoolExecutor
        else:
            # CPU模式：Windows上使用线程池，Linux/Mac可以使用进程池
            # 对于CPU密集型任务，Windows上线程池可能更高效（避免进程间通信开销）
            if sys.platform == 'win32':
                # Windows: 使用线程池，避免进程间通信开销
                max_workers = min(max_total_combinations, max(cpu_count, 16))
                executor_class = ThreadPoolExecutor
            else:
                # Linux/Mac: 使用进程池，充分利用多核
                max_workers = min(max_total_combinations, max(cpu_count, 16))
                executor_class = ProcessPoolExecutor
        
        # progress_callback 已在函数参数中定义
        
        # 分批提交任务，避免一次性提交太多任务导致内存爆炸
        # 使用生成器按需生成组合，而不是一次性生成所有组合
        batch_size = max_workers * 2  # 每批提交的任务数（减少批次大小，降低内存压力）
        
        with executor_class(max_workers=max_workers) as executor:
            # 分批提交和处理任务
            future_to_combo = {}
            processed_count = 0
            submitted_count = 0
            generator_exhausted = False
            
            # 从生成器中按需获取组合并提交任务
            while not generator_exhausted:
                # 提交一批新任务（从生成器中获取）
                while len(future_to_combo) < batch_size and not generator_exhausted:
                    try:
                        combo = next(combinations_generator)
                        future = executor.submit(process_single_combination_fast, combo)
                        future_to_combo[future] = combo
                        submitted_count += 1
                    except StopIteration:
                        generator_exhausted = True
                        break
                
                # 处理已完成的任务（不等待所有任务完成）
                completed_futures = []
                for future in list(future_to_combo.keys()):
                    if future.done():
                        completed_futures.append(future)
                
                for future in completed_futures:
                    processed_count += 1
                    
                    # 更新进度（使用估算值）
                    if progress_callback:
                        try:
                            progress_callback(processed_count, estimated_total, 
                                           f"组合 {processed_count}/{estimated_total} (已提交: {submitted_count})")
                        except:
                            pass
                    
                    try:
                        result = future.result()
                        if result is not None:
                            # 线程池模式需要锁，进程池模式不需要
                            if executor_class == ThreadPoolExecutor:
                                with results_lock:
                                    combination_results.append(result)
                            else:
                                combination_results.append(result)
                            
                            # 提前终止：如果top_k不为None且找到很好的匹配（>0.95），且已收集足够的结果
                            if top_k is not None and result['match_score'] > 0.95 and len(combination_results) >= top_k * 2:
                                # 取消剩余任务
                                for f in future_to_combo:
                                    if not f.done():
                                        f.cancel()
                                generator_exhausted = True
                                break
                    except Exception as e:
                        # 忽略单个组合的错误，继续处理其他组合
                        pass
                    
                    del future_to_combo[future]
                
                # 如果已经提前终止，跳出循环
                if top_k is not None and len(combination_results) >= top_k * 2:
                    break
                
                # 如果生成器已耗尽且没有待处理任务，退出循环
                if generator_exhausted and len(future_to_combo) == 0:
                    break
            
            # 处理剩余的任务
            for future in as_completed(future_to_combo):
                processed_count += 1
                
                # 更新进度
                if progress_callback:
                    try:
                        progress_callback(processed_count, estimated_total, 
                                       f"组合 {processed_count}/{estimated_total}")
                    except:
                        pass
                
                try:
                    result = future.result()
                    if result is not None:
                        # 线程池模式需要锁，进程池模式不需要
                        if executor_class == ThreadPoolExecutor:
                            with results_lock:
                                combination_results.append(result)
                        else:
                            combination_results.append(result)
                except Exception as e:
                    # 忽略单个组合的错误
                    pass
        
        # 按匹配分数排序，优先显示分数高的
        # 排序规则：1) 匹配分数（越高越好，降序），2) 未匹配峰值数（越少越好）
        # 确保结果按分数从大到小排序
        def get_sort_key(x):
            """获取排序键：优先匹配分数高的，然后未匹配峰值数少的"""
            match_score = x.get('match_score', 0.0)
            unmatched_count = x.get('num_unmatched_peaks')
            if unmatched_count is None:
                unmatched_peaks = x.get('unmatched_peaks', [])
                unmatched_count = len(unmatched_peaks) if isinstance(unmatched_peaks, (list, np.ndarray)) else 0
            return (-match_score, unmatched_count)  # 匹配分数降序，未匹配峰值数升序
        
        combination_results.sort(key=get_sort_key, reverse=False)
        
        # 如果top_k为None，返回所有结果（不限制数量）
        if top_k is None:
            return combination_results
        
        # 否则返回前top_k个结果
        return combination_results[:top_k]


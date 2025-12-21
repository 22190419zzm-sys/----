"""File service for scanning, grouping, and skip rows detection"""

import os
import glob
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.utils.helpers import group_files_by_name
from src.utils.skip_rows_detector import SkipRowsDetector


class FileService:
    """Service class for file operations: scanning, grouping, and skip rows detection"""
    
    def __init__(self):
        self.skip_rows_detection_results = {}
    
    def scan_folder(self, folder_path: str) -> Dict[str, any]:
        """
        扫描文件夹，返回文件列表和分组信息
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            dict: 包含以下键的字典
                - 'files': List[str] - 所有文件路径列表
                - 'groups': Dict[str, List[str]] - 分组后的文件字典
                - 'skip_rows': int - 检测到的跳过行数
                - 'skip_rows_info': str - 跳过行数检测信息
        """
        if not os.path.isdir(folder_path):
            return {
                'files': [],
                'groups': {},
                'skip_rows': 0,
                'skip_rows_info': '文件夹不存在'
            }
        
        # 扫描文件
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        file_list = sorted(csv_files + txt_files)
        
        # 检测跳过行数
        skip_rows_info = self.detect_skip_rows(folder_path)
        skip_rows = skip_rows_info.get('skip_rows', 0)
        
        return {
            'files': file_list,
            'groups': {},  # 分组需要额外的参数（n_chars），由调用者使用 group_files 方法
            'skip_rows': skip_rows,
            'skip_rows_info': skip_rows_info.get('info', '')
        }
    
    def group_files(self, file_list: List[str], n_chars: int = 5) -> Dict[str, List[str]]:
        """
        按文件名前缀分组文件
        
        Args:
            file_list: 文件路径列表
            n_chars: 用于分组的前缀字符数
            
        Returns:
            Dict[str, List[str]]: 分组后的文件字典，键为组名，值为文件列表
        """
        return group_files_by_name(file_list, n_chars)
    
    def detect_skip_rows(self, folder_path: str, sample_count: int = 3) -> Dict[str, any]:
        """
        检测文件夹中文件的跳过行数
        
        Args:
            folder_path: 文件夹路径
            sample_count: 检测的样本文件数量（默认3个）
            
        Returns:
            dict: 包含以下键的字典
                - 'skip_rows': int - 最常见的跳过行数
                - 'info': str - 检测信息文本
                - 'results': Dict[str, dict] - 每个文件的检测结果
        """
        if not os.path.isdir(folder_path):
            return {
                'skip_rows': 0,
                'info': '文件夹不存在',
                'results': {}
            }
        
        try:
            # 获取所有CSV和TXT文件
            csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
            txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
            all_files = csv_files + txt_files
            
            if not all_files:
                return {
                    'skip_rows': 0,
                    'info': '未找到数据文件',
                    'results': {}
                }
            
            # 只检测前几个文件（减少检测时间）
            sample_files = all_files[:sample_count]
            
            # 检查缓存：只检测修改时间改变的文件
            files_to_detect = []
            
            for file_path in sample_files:
                if file_path not in self.skip_rows_detection_results:
                    files_to_detect.append(file_path)
                else:
                    # 检查文件是否改变
                    try:
                        cached_mtime = self.skip_rows_detection_results[file_path].get('mtime', 0)
                        current_mtime = os.path.getmtime(file_path)
                        if current_mtime != cached_mtime:
                            files_to_detect.append(file_path)
                    except:
                        files_to_detect.append(file_path)
            
            # 只检测需要检测的文件
            if files_to_detect:
                new_results = SkipRowsDetector.detect_multiple_files(files_to_detect)
                # 更新缓存
                for file_path, info in new_results.items():
                    try:
                        info['mtime'] = os.path.getmtime(file_path)
                    except:
                        info['mtime'] = 0
                    self.skip_rows_detection_results[file_path] = info
                results = self.skip_rows_detection_results
            else:
                # 使用缓存结果
                results = {k: v for k, v in self.skip_rows_detection_results.items() if k in sample_files}
            
            if not results:
                return {
                    'skip_rows': 0,
                    'info': '检测失败',
                    'results': {}
                }
            
            # 统计跳过行数
            skip_rows_list = [info['skip_rows'] for info in results.values()]
            most_common_skip = max(set(skip_rows_list), key=skip_rows_list.count) if skip_rows_list else 0
            
            # 获取第一个文件的预览
            first_file = list(results.keys())[0]
            first_info = results[first_file]
            
            # 格式化显示信息
            preview_text = first_info.get('preview', '')[:50].replace('\n', ' ') if first_info.get('preview') else "N/A"
            middle_text = first_info.get('middle', '')[:50].replace('\n', ' ') if first_info.get('middle') else "N/A"
            end_text = first_info.get('end', '')[:50].replace('\n', ' ') if first_info.get('end') else "N/A"
            
            detected_count = len(files_to_detect) if files_to_detect else 0
            cached_count = len(sample_files) - detected_count
            
            info_text = (
                f"检测结果: 跳过 {most_common_skip} 行 (检测 {detected_count} 个文件，使用缓存 {cached_count} 个)\n"
                f"前: {preview_text}...\n"
                f"中: {middle_text}...\n"
                f"后: {end_text}..."
            )
            
            return {
                'skip_rows': most_common_skip,
                'info': info_text,
                'results': results
            }
            
        except Exception as e:
            return {
                'skip_rows': 0,
                'info': f'检测失败 - {str(e)}',
                'results': {}
            }
    
    def scan_and_load_legend_rename_data(self, folder_path: str, n_chars: int = 5, 
                                         target_groups: Optional[List[str]] = None) -> Dict[str, any]:
        """
        扫描文件并为图例重命名准备数据（包括瀑布图的组名）
        
        Args:
            folder_path: 文件夹路径
            n_chars: 用于分组的前缀字符数
            target_groups: 指定的组名列表（如果提供，只返回这些组）
            
        Returns:
            dict: 包含以下键的字典
                - 'files': List[str] - 所有文件路径列表
                - 'groups': Dict[str, List[str]] - 分组后的文件字典
                - 'files_in_groups': set - 组中包含的所有文件集合
        """
        if not os.path.isdir(folder_path):
            return {
                'files': [],
                'groups': {},
                'files_in_groups': set()
            }
        
        # 扫描文件
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        file_list_full = sorted(csv_files + txt_files)
        
        # 扫描分组
        groups = self.group_files(file_list_full, n_chars)
        
        # 筛选指定组（如果设置了）
        if target_groups:
            target_gs = [x.strip() for x in target_groups if x.strip()]
            if target_gs:
                groups = {k: v for k, v in groups.items() if k in target_gs}
        
        # 收集所有组中的文件，避免重复添加
        files_in_groups = set()
        for g_files in groups.values():
            files_in_groups.update(g_files)
        
        return {
            'files': file_list_full,
            'groups': groups,
            'files_in_groups': files_in_groups
        }


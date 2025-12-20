"""
跳过行数自动检测工具
检测CSV/TXT文件的前中后部分，确定应该跳过的行数
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class SkipRowsDetector:
    """跳过行数检测器"""
    
    @staticmethod
    def detect_skip_rows(file_path: str, max_check_lines: int = 20) -> int:
        """
        自动检测应该跳过的行数
        
        Args:
            file_path: 文件路径
            max_check_lines: 最多检查的行数
            
        Returns:
            skip_rows: 应该跳过的行数
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    lines = f.readlines()
            except:
                return 0
        except:
            return 0
        
        if not lines:
            return 0
        
        # 检查前中后三部分
        total_lines = len(lines)
        check_ranges = [
            (0, min(max_check_lines, total_lines)),  # 前部
            (max(0, total_lines // 2 - max_check_lines // 2), 
             min(total_lines, total_lines // 2 + max_check_lines // 2)),  # 中部
            (max(0, total_lines - max_check_lines), total_lines)  # 后部
        ]
        
        best_skip = 0
        best_score = -1
        
        # 对每个可能的跳过行数进行评分
        for skip_rows in range(min(max_check_lines, total_lines)):
            score = 0
            
            # 检查前中后三部分
            for start, end in check_ranges:
                if skip_rows >= end:
                    continue
                
                check_start = max(skip_rows, start)
                if check_start >= end:
                    continue
                
                # 尝试读取这部分数据
                try:
                    test_df = pd.read_csv(
                        file_path, 
                        header=None, 
                        skiprows=skip_rows,
                        nrows=min(5, end - check_start),
                        sep=None, 
                        engine='python'
                    )
                    
                    if test_df.shape[1] >= 2:
                        # 尝试转换为浮点数
                        try:
                            x_col = test_df.iloc[:, 0].astype(float)
                            y_col = test_df.iloc[:, 1].astype(float)
                            
                            # 检查是否有有效数据
                            valid_x = x_col[~np.isnan(x_col)]
                            valid_y = y_col[~np.isnan(y_col)]
                            
                            if len(valid_x) > 0 and len(valid_y) > 0:
                                # 检查是否在合理范围内
                                x_mean = np.mean(valid_x)
                                if 0 < x_mean < 100000:  # 合理的波数范围
                                    score += len(valid_x)
                        except:
                            pass
                except:
                    pass
            
            if score > best_score:
                best_score = score
                best_skip = skip_rows
        
        return best_skip
    
    @staticmethod
    def detect_multiple_files(file_paths: List[str]) -> Dict[str, Dict]:
        """
        检测多个文件的跳过行数
        
        Returns:
            {file_path: {'skip_rows': int, 'preview': str, 'middle': str, 'end': str}}
        """
        results = {}
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            
            skip_rows = SkipRowsDetector.detect_skip_rows(file_path)
            
            # 获取前中后预览
            preview_lines = []
            middle_lines = []
            end_lines = []
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    all_lines = f.readlines()
                
                total = len(all_lines)
                
                # 前3行（跳过skip_rows后）
                start_idx = skip_rows
                preview_lines = all_lines[start_idx:start_idx+3] if start_idx < total else []
                
                # 中间3行
                mid_idx = total // 2
                middle_lines = all_lines[mid_idx:mid_idx+3] if mid_idx < total else []
                
                # 后3行
                end_idx = max(0, total - 3)
                end_lines = all_lines[end_idx:] if end_idx < total else []
                
            except:
                pass
            
            results[file_path] = {
                'skip_rows': skip_rows,
                'preview': ''.join(preview_lines[:3]),
                'middle': ''.join(middle_lines[:3]),
                'end': ''.join(end_lines[:3])
            }
        
        return results


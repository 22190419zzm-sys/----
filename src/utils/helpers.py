import os
from collections import defaultdict


def natural_sort_key(s):
    """
    自然排序键函数：正确处理字符串中的数字
    例如："10mg" 会排在 "2mg" 之后，而不是之前
    
    Args:
        s: 字符串
    
    Returns:
        tuple: 用于排序的键
    """
    import re
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    return [convert(c) for c in re.split(r'(\d+)', str(s))]


def group_files_by_name(file_list, n_chars=5):
    grouped_files = defaultdict(list)
    ignore_suffixes = []

    for file_path in file_list:
        filename = os.path.basename(file_path)
        
        # 1. 移除忽略的后缀（如 _residual, _baseline 等）
        processed_name = filename
        for suf in ignore_suffixes:
            if processed_name.endswith(suf):
                processed_name = processed_name[:-len(suf)]
        
        # 2. 截取前 n_chars 作为分组键（如果 n_chars>0），否则使用完整文件名
        if n_chars > 0:
            group_key = processed_name[:n_chars]
        else:
            group_key = processed_name
            
        grouped_files[group_key].append(file_path)
            
    return grouped_files


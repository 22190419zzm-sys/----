import glob
import os
import numpy as np
import pandas as pd


class SpectralMatcher:
    """光谱库匹配器：使用余弦相似度匹配残差谱与标准库"""
    def __init__(self, library_folder):
        """
        Args:
            library_folder: 标准库文件夹路径
        """
        self.library_folder = library_folder
        self.library_spectra = {}  # {name: (wavenumbers, spectrum)}
        self.load_library()
    
    def load_library(self):
        """加载标准库中的所有光谱"""
        if not os.path.isdir(self.library_folder):
            return
        
        files = glob.glob(os.path.join(self.library_folder, '*.txt')) + \
                glob.glob(os.path.join(self.library_folder, '*.csv'))
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, header=None, skiprows=2)
                if df.shape[1] < 2:
                    continue
                x = df.iloc[:, 0].values.astype(float)
                y = df.iloc[:, 1].values.astype(float)
                
                name = os.path.splitext(os.path.basename(file_path))[0]
                self.library_spectra[name] = (x, y)
            except Exception as e:
                print(f"加载标准库光谱失败 {file_path}: {e}")
                continue
    
    def match(self, query_wavenumbers, query_spectrum, top_k=3):
        """
        匹配查询光谱与标准库
        
        Args:
            query_wavenumbers: 查询光谱的波数数组
            query_spectrum: 查询光谱数组
            top_k: 返回前k个匹配结果
        
        Returns:
            matches: 列表 [(name, similarity_score), ...]
        """
        if not self.library_spectra:
            return []
        
        matches = []
        
        for name, (lib_x, lib_y) in self.library_spectra.items():
            # 插值对齐到查询光谱的波数轴
            from scipy.interpolate import interp1d
            f_interp = interp1d(lib_x, lib_y, kind='linear', fill_value=0, bounds_error=False)
            lib_y_aligned = f_interp(query_wavenumbers)
            
            # 计算余弦相似度
            # 归一化
            query_norm = query_spectrum / (np.linalg.norm(query_spectrum) + 1e-10)
            lib_norm = lib_y_aligned / (np.linalg.norm(lib_y_aligned) + 1e-10)
            
            similarity = np.dot(query_norm, lib_norm)
            matches.append((name, similarity))
        
        # 按相似度排序
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k]


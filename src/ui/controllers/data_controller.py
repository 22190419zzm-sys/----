import os
import numpy as np
import pandas as pd

from src.utils.helpers import group_files_by_name
from src.utils.skip_rows_detector import SkipRowsDetector


class DataController:
    """数据读取与分组逻辑的可复用控制器（与 GUI 解耦，便于测试/扩展）。"""

    def read_data(self, file_path, skip_rows, x_min_phys=None, x_max_phys=None):
        """
        读取单个光谱文件并按物理范围截断。
        自动检测并跳过无效行，从第一个包含数字数据的行开始读取。

        Args:
            file_path: 文件路径
            skip_rows: 读取时跳过的头部行数（如果为-1，则自动检测）
            x_min_phys: 物理下限
            x_max_phys: 物理上限

        Returns:
            (x, y) 截断后的波数与强度
        """
        try:
            # 如果skip_rows为-1，自动检测跳过行数
            if skip_rows == -1:
                skip_rows = self._auto_detect_skip_rows(file_path)
            
            try:
                df = pd.read_csv(file_path, header=None, skiprows=skip_rows, sep=None, engine='python')
            except Exception:
                df = pd.read_csv(file_path, header=None, skiprows=skip_rows)

            if df.shape[1] < 2:
                raise ValueError("数据列不足2列")

            x = df.iloc[:, 0].values.astype(float)
            y = df.iloc[:, 1].values.astype(float)

            # 强制 X 降序 (Wavenumber 高->低)
            if len(x) > 1 and x[0] < x[-1]:
                x = x[::-1]
                y = y[::-1]

            # 物理截断
            mask = np.ones_like(x, dtype=bool)
            if x_min_phys is not None:
                mask &= (x >= x_min_phys)
            if x_max_phys is not None:
                mask &= (x <= x_max_phys)

            if not np.any(mask):
                raise ValueError(f"文件 {os.path.basename(file_path)} 在 X-Range [{x_min_phys}-{x_max_phys}] 内无数据。")

            x = x[mask]
            y = y[mask]
            return x, y
        except Exception as exc:  # pragma: no cover - 打印错误路径方便定位
            print(f"Error reading file {file_path}: {exc}")
            raise
    
    def _auto_detect_skip_rows(self, file_path):
        """
        自动检测应该跳过的行数，从第一个包含有效数字数据的行开始
        
        Returns:
            skip_rows: 应该跳过的行数
        """
        return SkipRowsDetector.detect_skip_rows(file_path)

    def parse_region_weights(self, weights_str, wavenumbers):
        """
        解析区域权重字符串并生成权重向量。

        Args:
            weights_str: 形如 "800-1000:0.1, 1000-1200:1.0"
            wavenumbers: 波数数组
        """
        if not weights_str or not weights_str.strip():
            return np.ones(len(wavenumbers))

        weight_vector = np.ones(len(wavenumbers))
        try:
            parts = weights_str.split(',')
            for part in parts:
                part = part.strip()
                if ':' not in part:
                    continue
                range_str, weight_str = part.split(':', 1)
                min_w, max_w = map(float, range_str.strip().split('-'))
                weight = float(weight_str.strip())
                mask = (wavenumbers >= min_w) & (wavenumbers <= max_w)
                weight_vector[mask] = weight
        except Exception as exc:  # pragma: no cover - 解析异常时回退
            print(f"警告：区域权重解析失败: {exc}，使用默认权重（全1）")
            return np.ones(len(wavenumbers))
        return weight_vector

    def load_and_average_data(self, file_list, n_chars, skip_rows, x_min_phys=None, x_max_phys=None):
        """
        将重复样本（如 sample-1, sample-2）分组并计算平均光谱。

        Returns:
            averaged_data: {group: {'x','y','label','files'}}
            common_x: 公共波数轴
        """
        grouped_files = group_files_by_name(file_list, n_chars)
        averaged_data = {}
        common_x = None

        for group_key, files_in_group in grouped_files.items():
            group_spectra = []
            group_x_list = []

            for file_path in files_in_group:
                try:
                    x, y = self.read_data(file_path, skip_rows, x_min_phys, x_max_phys)
                    group_x_list.append(x)
                    group_spectra.append(y)
                except Exception as exc:
                    print(f"警告：跳过文件 {os.path.basename(file_path)}: {exc}")
                    continue

            if not group_spectra:
                continue

            if common_x is None:
                common_x = group_x_list[0]
            else:
                aligned_spectra = []
                for x_local, y_local in zip(group_x_list, group_spectra):
                    if len(x_local) == len(common_x) and np.allclose(x_local, common_x):
                        aligned_spectra.append(y_local)
                    else:
                        from scipy.interpolate import interp1d
                        f_interp = interp1d(x_local, y_local, kind='linear', fill_value=0, bounds_error=False)
                        aligned_spectra.append(f_interp(common_x))
                group_spectra = aligned_spectra

            group_matrix = np.array(group_spectra)
            y_averaged = np.mean(group_matrix, axis=0)

            averaged_data[group_key] = {
                'x': common_x,
                'y': y_averaged,
                'label': group_key,
                'files': files_in_group
            }

        return averaged_data, common_x


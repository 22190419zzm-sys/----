import numpy as np
from scipy import sparse
from scipy.linalg import svd
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

from .registry import register_preprocessor

# 物理常数 (用于 Bose-Einstein 校正)
C_H = 6.62607015e-34  # 普朗克常数 (J*s)
C_C = 2.99792458e10  # 光速 (cm/s)
C_K = 1.380649e-23   # 玻尔兹曼常数 (J/K)
C_CM_TO_HZ = C_C     # 波数 cm^-1 到 频率 Hz 的转换因子


class DataPreProcessor:
    """Includes Bose-Einstein Correction, AsLS Baseline, and Smoothing."""
    @staticmethod
    def apply_smoothing(y_data, window_length, polyorder):
        if window_length < polyorder + 2: return y_data
        if window_length % 2 == 0: window_length += 1
        return savgol_filter(y_data, window_length, polyorder)

    @staticmethod
    def apply_baseline_als(y_data, lam, p, niter=10):
        L = len(y_data)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        z = np.zeros(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y_data)
            w = p * (y_data > z) + (1-p) * (y_data < z)
        return z

    @staticmethod
    def apply_baseline_correction(x_data, y_data, n_points=50, poly_order=3):
        """
        简单双曲线：分段取低百分位拟合多项式基线。
        - 将光谱分成 n_points 段，取每段 5% 分位作为基线锚点
        - 拟合 poly_order 阶多项式并减去
        """
        x = np.asarray(x_data)
        y = np.asarray(y_data)
        if x.size == 0 or y.size == 0:
            return y
        n_points = int(max(poly_order + 1, min(n_points, len(x))))
        if n_points < poly_order + 1:
            return y

        edges = np.linspace(0, len(x), n_points + 1, dtype=int)
        anchor_x, anchor_y = [], []
        for i in range(n_points):
            start, end = edges[i], edges[i + 1]
            if end <= start:
                continue
            seg_x = x[start:end]
            seg_y = y[start:end]
            if seg_y.size == 0:
                continue
            anchor_x.append(float(seg_x.mean()))
            anchor_y.append(float(np.percentile(seg_y, 5)))

        if len(anchor_x) < poly_order + 1:
            return y

        coeffs = np.polyfit(anchor_x, anchor_y, poly_order)
        baseline = np.polyval(coeffs, x)
        return y - baseline

    @staticmethod
    def apply_normalization(y_data, norm_mode='max'):
        if norm_mode == 'max':
            max_val = np.max(y_data)
            return y_data / max_val if max_val != 0 else y_data
        elif norm_mode == 'area':
            area = np.trapezoid(y_data)
            return y_data / area if area != 0 else y_data
        elif norm_mode == 'snv':
            return DataPreProcessor.apply_snv(y_data)
        return y_data

    @staticmethod
    def apply_snv(y_data):
        mean = np.mean(y_data)
        std = np.std(y_data)
        return (y_data - mean) / std if std != 0 else y_data

    @staticmethod
    def apply_log_transform(y_data, base=10, offset=1.0):
        y_shifted = np.maximum(y_data + offset, 1e-10)
        if base == 10: return np.log10(y_shifted)
        elif base == np.e or base == 'e': return np.log(y_shifted)
        else: return np.log(y_shifted) / np.log(base)

    @staticmethod
    def apply_sqrt_transform(y_data, offset=0.0):
        return np.sqrt(np.maximum(y_data + offset, 0.0))

    @staticmethod
    def apply_bose_einstein_correction(x_data, y_data, temp_k):
        """Corrects for thermal population effects: I_corr = I_meas / (n(nu) + 1)."""
        exp_arg = (C_H * x_data * C_CM_TO_HZ) / (C_K * temp_k)
        exp_val = np.exp(exp_arg)
        mask = exp_val > 1.000001
        n_nu = np.zeros_like(x_data)
        n_nu[mask] = 1.0 / (exp_val[mask] - 1.0)
        be_factor = n_nu + 1.0
        y_corr = np.zeros_like(y_data)
        valid_mask = be_factor != 0
        y_corr[valid_mask] = y_data[valid_mask] / be_factor[valid_mask]
        return y_corr

    @staticmethod
    def svd_denoise(matrix, k):
        """
        使用 SVD 去噪：保留前 k 个主成分，去除随机噪声
        
        Args:
            matrix: 输入数据矩阵 (n_samples, n_features)
            k: 保留的主成分数量
        
        Returns:
            denoised_matrix: 去噪后的数据矩阵
        """
        if k <= 0 or k > min(matrix.shape):
            return matrix
        
        # 执行 SVD
        U, s, Vt = svd(matrix, full_matrices=False)
        
        # 只保留前 k 个成分
        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]
        
        # 重构数据
        denoised_matrix = U_k @ np.diag(s_k) @ Vt_k
        
        # 确保非负（对于光谱数据）
        denoised_matrix = np.maximum(denoised_matrix, 0)
        
        return denoised_matrix
    
    @staticmethod
    def apply_quadratic_fit(x_data, y_data, degree=2):
        """
        应用二次函数（多项式）拟合作为预处理步骤
        
        Args:
            x_data: X轴数据（波数）
            y_data: Y轴数据（强度）
            degree: 多项式阶数（默认2，即二次函数）
        
        Returns:
            y_fitted: 拟合后的Y数据
        """
        if len(x_data) < degree + 1:
            return y_data
        
        try:
            # 使用numpy的polyfit进行多项式拟合
            coeffs = np.polyfit(x_data, y_data, degree)
            # 计算拟合值
            y_fitted = np.polyval(coeffs, x_data)
            return y_fitted
        except:
            return y_data
    
    @staticmethod
    def preprocess_spectrum(x_data, y_data, preprocess_params):
        """
        统一的预处理函数，包含所有预处理步骤
        
        预处理顺序：
        1. QC检查（如果启用）
        2. BE校正（如果启用）
        3. 平滑（如果启用）
        4. 基线校正（AsLS或多项式，如果启用）
        5. 归一化（如果启用）
        6. 全局动态范围压缩（对数/平方根变换，如果启用）
        7. 二次函数拟合（如果启用）
        8. 二次导数（如果启用）
        9. 整体Y轴偏移（最后一步）
        
        Args:
            x_data: X轴数据（波数）
            y_data: Y轴数据（强度）
            preprocess_params: 预处理参数字典，包含：
                - qc_enabled: bool, QC检查是否启用
                - qc_threshold: float, QC阈值
                - is_be_correction: bool, BE校正是否启用
                - be_temp: float, BE温度
                - is_smoothing: bool, 平滑是否启用
                - smoothing_window: int, 平滑窗口
                - smoothing_poly: int, 平滑多项式阶数
                - is_baseline_als: bool, AsLS基线校正是否启用
                - als_lam: float, AsLS lambda参数
                - als_p: float, AsLS p参数
                - is_baseline_poly: bool, 多项式基线校正是否启用
                - baseline_points: int, 基线采样点数
                - baseline_poly: int, 基线多项式阶数
                - normalization_mode: str, 归一化模式 ('None', 'max', 'area', 'snv')
                - global_transform_mode: str, 全局变换模式 ('无', '对数变换 (Log)', '平方根变换 (Sqrt)')
                - global_log_base: str, 对数底数 ('10', 'e')
                - global_log_offset: float, 对数偏移
                - global_sqrt_offset: float, 平方根偏移
                - is_quadratic_fit: bool, 二次函数拟合是否启用
                - quadratic_degree: int, 二次函数阶数（默认2）
                - is_derivative: bool, 二次导数是否启用
                - global_y_offset: float, 整体Y轴偏移
        
        Returns:
            y_processed: 预处理后的Y数据
        """
        y_proc = np.asarray(y_data, dtype=float).copy()
        x_proc = np.asarray(x_data, dtype=float)
        
        # 1. QC检查
        if preprocess_params.get('qc_enabled', False):
            qc_threshold = preprocess_params.get('qc_threshold', 5.0)
            if np.max(y_proc) < qc_threshold:
                # QC失败，返回原始数据（或零数组）
                return y_proc
        
        # 2. BE校正
        if preprocess_params.get('is_be_correction', False):
            be_temp = preprocess_params.get('be_temp', 300.0)
            y_proc = DataPreProcessor.apply_bose_einstein_correction(x_proc, y_proc, be_temp)
        
        # 3. 平滑
        if preprocess_params.get('is_smoothing', False):
            smoothing_window = preprocess_params.get('smoothing_window', 15)
            smoothing_poly = preprocess_params.get('smoothing_poly', 3)
            y_proc = DataPreProcessor.apply_smoothing(y_proc, smoothing_window, smoothing_poly)
        
        # 4. 基线校正（优先AsLS）
        if preprocess_params.get('is_baseline_als', False):
            als_lam = preprocess_params.get('als_lam', 10000)
            als_p = preprocess_params.get('als_p', 0.005)
            b = DataPreProcessor.apply_baseline_als(y_proc, als_lam, als_p)
            y_proc = y_proc - b
            y_proc[y_proc < 0] = 0
        elif preprocess_params.get('is_baseline_poly', False):
            baseline_points = preprocess_params.get('baseline_points', 50)
            baseline_poly = preprocess_params.get('baseline_poly', 3)
            y_proc = DataPreProcessor.apply_baseline_correction(x_proc, y_proc, baseline_points, baseline_poly)
        
        # 5. 归一化
        normalization_mode = preprocess_params.get('normalization_mode', 'None')
        if normalization_mode == 'max':
            y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
        elif normalization_mode == 'area':
            y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
        elif normalization_mode == 'snv':
            y_proc = DataPreProcessor.apply_snv(y_proc)
        
        # 6. 全局动态范围压缩
        global_transform_mode = preprocess_params.get('global_transform_mode', '无')
        if global_transform_mode == '对数变换 (Log)':
            global_log_base = preprocess_params.get('global_log_base', '10')
            base = float(global_log_base) if global_log_base == '10' else np.e
            global_log_offset = preprocess_params.get('global_log_offset', 1.0)
            y_proc = DataPreProcessor.apply_log_transform(y_proc, base=base, offset=global_log_offset)
        elif global_transform_mode == '平方根变换 (Sqrt)':
            global_sqrt_offset = preprocess_params.get('global_sqrt_offset', 0.0)
            y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=global_sqrt_offset)
        
        # 7. 二次函数拟合（新增，在全局变换之后）
        if preprocess_params.get('is_quadratic_fit', False):
            quadratic_degree = preprocess_params.get('quadratic_degree', 2)
            y_proc = DataPreProcessor.apply_quadratic_fit(x_proc, y_proc, degree=quadratic_degree)
        
        # 8. 二次导数
        if preprocess_params.get('is_derivative', False):
            d1 = np.gradient(y_proc, x_proc)
            y_proc = np.gradient(d1, x_proc)
        
        # 9. 整体Y轴偏移（最后一步）
        global_y_offset = preprocess_params.get('global_y_offset', 0.0)
        y_proc = y_proc + global_y_offset
        
        return y_proc


# 注册默认预处理函数，便于插件式扩展
register_preprocessor("smoothing", DataPreProcessor.apply_smoothing)
register_preprocessor("baseline_als", DataPreProcessor.apply_baseline_als)
register_preprocessor("baseline_poly", DataPreProcessor.apply_baseline_correction)
register_preprocessor("normalization", DataPreProcessor.apply_normalization)
register_preprocessor("snv", DataPreProcessor.apply_snv)
register_preprocessor("log_transform", DataPreProcessor.apply_log_transform)
register_preprocessor("sqrt_transform", DataPreProcessor.apply_sqrt_transform)
register_preprocessor("bose_einstein", DataPreProcessor.apply_bose_einstein_correction)
register_preprocessor("svd_denoise", DataPreProcessor.svd_denoise)


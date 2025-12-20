import os
import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """合成数据生成器：基于纯组分光谱生成混合光谱（用于数据增强）"""
    def __init__(self, wavenumbers):
        """
        Args:
            wavenumbers: 波数数组
        """
        self.wavenumbers = wavenumbers
        self.common_x = wavenumbers  # 兼容性：支持两种命名
        self.pure_spectra = {}  # 存储纯组分光谱 {name: spectrum_array}
        self.pure_spectra_full = []  # 存储完整光谱信息 [(name, x, y), ...]
    
    def load_pure_spectrum(self, file_path, name):
        """
        加载纯组分光谱
        
        支持多种文件格式：
        - 无头部文件（直接两列：波数，强度）
        - 有头部文件（自动跳过头部行）
        - 支持 .txt, .csv 格式
        """
        try:
            # 尝试不同的 skiprows 值（0, 1, 2），以适应不同的文件格式
            df = None
            skiprows_list = [0, 1, 2]
            
            for skiprows in skiprows_list:
                try:
                    # 尝试读取文件
                    df = pd.read_csv(file_path, header=None, skiprows=skiprows, sep=None, engine='python')
                    
                    # 检查是否有足够的列
                    if df.shape[1] < 2:
                        continue
                    
                    # 尝试转换为浮点数（如果失败，说明可能还有头部）
                    try:
                        x_original = df.iloc[:, 0].values.astype(float)
                        y_original = df.iloc[:, 1].values.astype(float)
                        
                        # 检查是否有有效数据
                        if len(x_original) > 0 and not np.isnan(x_original).all():
                            break  # 成功读取，退出循环
                    except (ValueError, TypeError):
                        continue  # 转换失败，尝试下一个 skiprows
                        
                except Exception:
                    continue  # 读取失败，尝试下一个 skiprows
            
            # 如果所有尝试都失败
            if df is None or df.shape[1] < 2:
                print(f"警告：无法读取文件 {file_path}，尝试了 skiprows={skiprows_list}")
                return False
            
            # 确保数据是浮点数
            try:
                x_original = df.iloc[:, 0].values.astype(float)
                y_original = df.iloc[:, 1].values.astype(float)
            except (ValueError, TypeError) as e:
                print(f"警告：文件 {file_path} 包含非数值数据: {e}")
                return False
            
            # 移除 NaN 值
            valid_mask = ~(np.isnan(x_original) | np.isnan(y_original))
            if not np.any(valid_mask):
                print(f"警告：文件 {file_path} 没有有效数据")
                return False
            
            x_original = x_original[valid_mask]
            y_original = y_original[valid_mask]
            
            # 确保 X 轴是降序（拉曼光谱通常从高波数到低波数）
            if len(x_original) > 1 and x_original[0] < x_original[-1]:
                x_original = x_original[::-1]
                y_original = y_original[::-1]
            
            # 保存原始数据（用于高级增强）
            x_aligned = x_original.copy()
            y_aligned = y_original.copy()
            
            # 如果X轴不一致，需要插值对齐
            if len(x_original) != len(self.wavenumbers) or not np.allclose(x_original, self.wavenumbers, rtol=1e-3):
                from scipy.interpolate import interp1d
                f_interp = interp1d(x_original, y_original, kind='linear', fill_value=0, bounds_error=False)
                y_aligned = f_interp(self.wavenumbers)
                x_aligned = self.wavenumbers.copy()
            
            # 保存对齐后的数据（用于简单方法）
            self.pure_spectra[name] = y_aligned
            
            # 同时保存完整信息用于高级增强（保存对齐后的数据）
            self.pure_spectra_full.append((name, x_aligned, y_aligned))
            print(f"成功加载纯组分光谱: {name} (来自 {os.path.basename(file_path)})")
            return True
        except Exception as e:
            print(f"加载纯组分光谱失败 {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_shift_and_stretch(self, spectrum, max_shift=3.0, max_stretch=1.005):
        """对光谱应用随机偏移和轻微拉伸（硬负样本挖掘）"""
        from scipy.interpolate import interp1d
        length = len(spectrum)
        original_indices = np.arange(length)
        shift = np.random.uniform(-max_shift, max_shift)
        stretch = np.random.uniform(1.0, max_stretch)
        center = length / 2.0
        shifted_indices = (original_indices - center) * stretch + center + shift
        
        interp_func = interp1d(original_indices, spectrum, kind='linear', 
                              bounds_error=False, fill_value=(spectrum[0], spectrum[-1]))
        new_spectrum = interp_func(shifted_indices)
        return np.nan_to_num(new_spectrum, nan=spectrum[0])
    
    def _add_selective_suppression(self, spectrum, suppression_prob=0.2, strength_range=(0.3, 0.8)):
        """选择性峰抑制：模拟部分峰被淹没或重叠的情况"""
        suppressed_spectrum = spectrum.copy()
        suppression_mask = np.random.random(len(spectrum)) < suppression_prob
        if np.any(suppression_mask):
            suppression_strength = np.random.uniform(strength_range[0], strength_range[1])
            suppressed_spectrum[suppression_mask] *= suppression_strength
        return suppressed_spectrum
    
    def _add_noise(self, spectrum, noise_level=0.01):
        """添加高斯噪音"""
        noise = np.random.normal(0, noise_level * np.max(spectrum), spectrum.shape)
        return spectrum + noise
    
    def generate_mixture(self, components, ratios, noise_level=0.01, drift_level=0.05, complexity=1.0):
        """
        生成一条混合光谱（集成复杂增强）
        
        Args:
            components: 组分列表，每个元素为 (name, x, y) 元组
            ratios: 比例列表，与 components 对应
            noise_level: 噪声水平
            drift_level: 基线漂移水平
            complexity: 复杂度因子（0-1），控制增强强度
        
        Returns:
            final_spectrum: 生成的混合光谱
        """
        if len(components) != len(ratios):
            raise ValueError("Component list and ratio list must have the same length.")
            
        final_spectrum = np.zeros_like(self.wavenumbers)
        
        for comp, ratio in zip(components, ratios):
            y_data = comp[2]  # 获取强度数据
            
            # [增强 1] 对矿物和有机物都应用随机偏移和拉伸
            if np.random.random() < 0.5 * complexity:
                y_data = self._add_shift_and_stretch(y_data, max_shift=2.0 * complexity)
            
            final_spectrum += y_data * ratio
        
        # [增强 2] 选择性峰抑制 (模拟信号淹没)
        if np.random.random() < 0.3 * complexity:
            final_spectrum = self._add_selective_suppression(final_spectrum, suppression_prob=0.1 * complexity)
            
        # [增强 3] 基线漂移 (使用多项式基线，简化处理)
        if drift_level > 0:
            x = np.linspace(0, 1, len(self.wavenumbers))
            degree = int(10 * complexity) + 1  # 复杂度越高，多项式次数越高
            coeffs = np.random.uniform(-drift_level, drift_level, degree + 1)
            baseline = np.polyval(coeffs, x)
            final_spectrum += baseline
            
        final_spectrum = self._add_noise(final_spectrum, noise_level)
        
        return np.maximum(final_spectrum, 0)
    
    def generate_synthetic_spectrum(self, ratios, noise_level=0.01, baseline_drift=0.0):
        """
        生成合成光谱
        
        Args:
            ratios: 字典 {component_name: ratio}，例如 {'mineral': 0.7, 'organic': 0.3}
            noise_level: 高斯噪声水平（相对于最大强度）
            baseline_drift: 基线漂移幅度
        
        Returns:
            synthetic_spectrum: 合成光谱数组
        """
        if not self.pure_spectra:
            raise ValueError("未加载纯组分光谱")
        
        # 线性混合
        synthetic = np.zeros_like(self.wavenumbers, dtype=float)
        for name, ratio in ratios.items():
            if name in self.pure_spectra:
                synthetic += ratio * self.pure_spectra[name]
        
        # 添加高斯噪声
        if noise_level > 0:
            max_intensity = np.max(np.abs(synthetic))
            noise = np.random.normal(0, noise_level * max_intensity, len(synthetic))
            synthetic += noise
        
        # 添加基线漂移
        if baseline_drift > 0:
            drift = np.random.uniform(-baseline_drift, baseline_drift) * np.ones_like(synthetic)
            synthetic += drift
        
        # 确保非负
        synthetic = np.maximum(synthetic, 0)
        
        return synthetic
    
    def generate_batch(self, n_samples, ratio_ranges, noise_level=0.01, baseline_drift=0.0, complexity=1.0, use_advanced=True):
        """
        批量生成合成光谱
        
        Args:
            n_samples: 生成样本数
            ratio_ranges: 字典 {component_name: (min_ratio, max_ratio)}
            noise_level: 噪声水平
            baseline_drift: 基线漂移幅度
            complexity: 复杂度因子（0-1），控制增强强度
            use_advanced: 是否使用高级增强方法（shift/stretch/suppression）
        
        Returns:
            X_synthetic: 合成光谱矩阵 (n_samples, n_features)
            ratios_used: 使用的比例列表
        """
        X_synthetic = []
        ratios_used = []
        
        # 准备组分列表（用于高级方法）
        component_names = list(ratio_ranges.keys())
        components = []
        for name in component_names:
            # 查找对应的完整光谱信息
            for comp_info in self.pure_spectra_full:
                if comp_info[0] == name:
                    # 如果X轴不一致，需要插值对齐
                    if len(comp_info[1]) != len(self.wavenumbers) or not np.allclose(comp_info[1], self.wavenumbers):
                        from scipy.interpolate import interp1d
                        f_interp = interp1d(comp_info[1], comp_info[2], kind='linear', fill_value=0, bounds_error=False)
                        y_aligned = f_interp(self.wavenumbers)
                        components.append((name, self.wavenumbers, y_aligned))
                    else:
                        components.append((name, comp_info[1], comp_info[2]))
                    break
        
        if not components:
            # 回退到简单方法
            components = None
        
        for i in range(n_samples):
            # 随机生成比例
            ratios = {}
            ratio_list = []
            for name, (min_r, max_r) in ratio_ranges.items():
                ratio_val = np.random.uniform(min_r, max_r)
                ratios[name] = ratio_val
                ratio_list.append(ratio_val)
            
            # 归一化比例（确保总和为1）
            total = sum(ratios.values())
            if total > 0:
                ratios = {k: v/total for k, v in ratios.items()}
                ratio_list = [r/total for r in ratio_list]
            
            # 使用高级方法或简单方法
            if use_advanced and components and len(components) == len(ratio_list):
                synthetic = self.generate_mixture(components, ratio_list, noise_level, baseline_drift, complexity)
            else:
                synthetic = self.generate_synthetic_spectrum(ratios, noise_level, baseline_drift)
            
            X_synthetic.append(synthetic)
            ratios_used.append(ratios)
        
        return np.array(X_synthetic), ratios_used


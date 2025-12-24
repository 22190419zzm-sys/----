# flake8: noqa
import os
import sys
import glob
import json
import pickle
import math
import random
import itertools
import traceback
import warnings
import re
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from pathlib import Path
from importlib import util
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from src.core.peak_detection_helper import detect_and_plot_peaks as unified_detect_and_plot_peaks
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Deep Learning features will be disabled.")

from PyQt6.QtCore import Qt, QPoint, QSize, QSettings, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, 
    QGroupBox, QCheckBox, QLineEdit, QSpinBox, 
    QDoubleSpinBox, QPushButton, QFormLayout, 
    QLabel, QMessageBox, QTextEdit, QWidget,
    QToolButton, QSizePolicy, QHBoxLayout, QScrollArea, QSpacerItem,
    QComboBox, QFileDialog, QTabWidget, QGridLayout, QFrame,
    QListWidget, QListWidgetItem, QAbstractItemView, QMenu,
    QRadioButton, QButtonGroup, QColorDialog, QTableWidget, QTableWidgetItem,
    QHeaderView
)

from src.config.constants import C_H, C_C, C_K, C_CM_TO_HZ
from src.config.plot_config import PlotStyleConfig
from src.utils.fonts import setup_matplotlib_fonts
from src.utils.helpers import natural_sort_key, group_files_by_name
from src.core.preprocessor import DataPreProcessor
from src.core.generators import SyntheticDataGenerator
from src.core.matcher import SpectralMatcher
from src.core.transformers import AutoencoderTransformer, NonNegativeTransformer, AdaptiveMineralFilter
from src.ui.widgets.custom_widgets import CollapsibleGroupBox, SmartDoubleSpinBox
from src.ui.canvas import MplCanvas
from src.ui.windows.nmf_window import NMFResultWindow
from src.ui.windows.plot_window import MplPlotWindow


class DAEComparisonWindow(QDialog):
    """Deep Autoencoder 降噪前后对比窗口"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Deep Autoencoder: 降噪前后对比")
        # 使用Window类型而不是Dialog，这样最小化后能显示窗口名称
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        # 移除最小尺寸限制，允许随意调整大小
        self.setMinimumSize(400, 300)
        
        self.main_layout = QVBoxLayout(self)
        
        # 图表区域（两个子图并排）
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(14, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)
        
        # 存储数据
        self.wavenumbers = None
        self.y_raw = None
        self.y_clean = None
    
    def set_data(self, wavenumbers, y_raw, y_clean):
        """
        设置数据并更新绘图
        
        Args:
            wavenumbers: 波数数组
            y_raw: 原始噪声输入
            y_clean: 降噪后输出
        """
        self.wavenumbers = wavenumbers
        self.y_raw = y_raw
        self.y_clean = y_clean
        self.update_plot()
    
    def update_plot(self):
        """更新对比图"""
        if self.wavenumbers is None or self.y_raw is None or self.y_clean is None:
            return
        
        self.figure.clear()
        
        # 创建两个子图
        ax1 = self.figure.add_subplot(121)  # 原始输入
        ax2 = self.figure.add_subplot(122)  # 降噪输出
        
        # 绘制原始输入
        ax1.plot(self.wavenumbers, self.y_raw, 'b-', linewidth=1.5, label='Noisy Input')
        ax1.set_xlabel("Wavenumber (cm⁻¹)", fontfamily='Times New Roman', fontsize=16)
        ax1.set_ylabel("Intensity", fontfamily='Times New Roman', fontsize=16)
        ax1.set_title("Original Noisy Input", fontfamily='Times New Roman', fontsize=18, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 应用发表级别样式
        self._apply_publication_style(ax1)
        
        # 绘制降噪输出
        ax2.plot(self.wavenumbers, self.y_clean, 'r-', linewidth=1.5, label='Clean Output')
        ax2.set_xlabel("Wavenumber (cm⁻¹)", fontfamily='Times New Roman', fontsize=16)
        ax2.set_ylabel("Intensity", fontfamily='Times New Roman', fontsize=16)
        ax2.set_title("Denoised Output (Deep AE)", fontfamily='Times New Roman', fontsize=18, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 应用发表级别样式
        self._apply_publication_style(ax2)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _apply_publication_style(self, ax):
        """应用发表级别绘图样式"""
        font_family = 'Times New Roman'
        
        ax.tick_params(axis='both', which='major', 
                      direction='in',
                      length=8,
                      width=1.0,
                      labelsize=14,
                      top=True,
                      right=True)
        ax.tick_params(axis='both', which='minor',
                      direction='in',
                      length=4,
                      width=1.0,
                      top=True,
                      right=True)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(14)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_visible(True)
        
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
        ax.title.set_fontfamily(font_family)

# ----------------------------------------------------
# ⚙️ 【自定义 Transformer：非负转换器】
# ============================================================================
# Core Algorithms (Refactored for Academic Publication Standards)
# ============================================================================

class NonNegativeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.array(X)
        X[X < 0] = 0
        return X

# --- PyTorch Deep Autoencoder ---
if TORCH_AVAILABLE:
    class DeepSpectralAE(nn.Module):
        def __init__(self, n_features, n_components=6, dropout_rate=0.2):
            super(DeepSpectralAE, self).__init__()
            # Encoder: Compress to latent space with dropout for regularization
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 128), nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, n_components), nn.ReLU() # Force non-negative latent
            )
            # Decoder: Reconstruct with dropout
            self.decoder = nn.Sequential(
                nn.Linear(n_components, 64), nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, n_features)
            )
        
        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon, z
        
        def compute_loss(self, x, x_recon, z, l1_lambda=0.01):
            """Compute combined loss: MSE reconstruction + L1 sparsity regularization"""
            mse_loss = nn.functional.mse_loss(x_recon, x)
            l1_loss = torch.mean(torch.abs(z))  # Encourage sparse latent representation
            total_loss = mse_loss + l1_lambda * l1_loss
            return total_loss, mse_loss, l1_loss

class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    """Hybrid Transformer: Uses PyTorch if available, falls back to sklearn MLP."""
    def __init__(self, n_components=6, hidden_nodes=128, max_iter=1000, use_deep=True,
                 l1_lambda=0.01, learning_rate=0.001, batch_size=32, n_epochs=200, 
                 dropout_rate=0.2, normalize=True, random_state=42):
        self.n_components = n_components
        self.hidden_nodes = hidden_nodes
        self.max_iter = max_iter
        self.use_deep = use_deep and TORCH_AVAILABLE
        self.l1_lambda = l1_lambda
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout_rate = dropout_rate
        self.normalize = normalize
        self.random_state = random_state
        self.model = None
        self.mean_ = None
        self.std_ = None
        self.n_features = None  # 保存训练时的特征维度，用于维度对齐
    
    def _set_random_seed(self):
        """Set random seeds for reproducibility"""
        if TORCH_AVAILABLE:
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            # For deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(self.random_state)
        import random
        random.seed(self.random_state)
    
    def _normalize(self, X):
        """Normalize data to zero mean and unit variance"""
        if self.mean_ is None:
            self.mean_ = np.mean(X, axis=0, keepdims=True)
            self.std_ = np.std(X, axis=0, keepdims=True)
            self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        return (X - self.mean_) / self.std_
    
    def _denormalize(self, X):
        """Denormalize data back to original scale"""
        if self.mean_ is None:
            return X
        return X * self.std_ + self.mean_
    
    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float32)  # Explicitly convert to float32
        
        # 保存训练时的特征维度（在归一化之前保存原始维度）
        self.n_features = X.shape[1]
        
        # Normalize data for better training stability
        if self.normalize:
            X = self._normalize(X)
        
        if self.use_deep:
            n_features = X.shape[1]
            n_samples = X.shape[0]
            
            # Set random seeds for reproducibility
            self._set_random_seed()
            
            self.model = DeepSpectralAE(n_features, self.n_components, self.dropout_rate)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            # Remove verbose parameter (deprecated in newer PyTorch versions)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                             patience=10)
            
            # Convert to tensor
            data_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Improved training loop with batching and early stopping
            self.model.train()
            best_loss = float('inf')
            patience_counter = 0
            patience = 20  # Early stopping patience
            
            # Create a random number generator with fixed seed for reproducible shuffling
            rng = np.random.RandomState(self.random_state)
            
            for epoch in range(self.n_epochs):
                # Shuffle data for each epoch (reproducible)
                indices = rng.permutation(n_samples)
                total_loss = 0.0
                n_batches = 0
                
                # Mini-batch training
                for i in range(0, n_samples, self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    batch_data = data_tensor[batch_indices]
                    
                    optimizer.zero_grad()
                    recon, z = self.model(batch_data)
                    loss, mse_loss, l1_loss = self.model.compute_loss(batch_data, recon, z, self.l1_lambda)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1
                
                avg_loss = total_loss / n_batches if n_batches > 0 else total_loss
                scheduler.step(avg_loss)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break  # Early stop if no improvement
        else:
            # Fallback to sklearn
            self.model = MLPRegressor(hidden_layer_sizes=(self.n_components, self.hidden_nodes),
                                      activation='relu', max_iter=self.max_iter, random_state=42,
                                      alpha=1e-4, learning_rate='adaptive')
            self.model.fit(X, X)
        return self
    
    def transform(self, X):
        X = np.array(X, dtype=np.float32)  # Explicitly convert to float32
        
        # Normalize if normalization was used during training
        if self.normalize and self.mean_ is not None:
            X = self._normalize(X)
        
        if self.use_deep:
            self.model.eval()  # Set to evaluation mode (disables dropout)
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                _, z = self.model(X_tensor)
            return z.numpy()
        else:
            # Manually extract latent layer for sklearn MLP
            w1 = self.model.coefs_[0]
            b1 = self.model.intercepts_[0]
            z = np.maximum(X @ w1 + b1, 0)
            return z[:, :self.n_components]
    
    def inverse_transform(self, H_encoded):
        """Decode latent representation back to original space."""
        H_encoded = np.array(H_encoded, dtype=np.float32)  # Explicitly convert to float32
        if self.use_deep:
            self.model.eval()  # Set to evaluation mode
            with torch.no_grad():
                H_tensor = torch.tensor(H_encoded, dtype=torch.float32)
                X_recon = self.model.decoder(H_tensor)
            X_recon_np = X_recon.numpy()
            
            # Denormalize if normalization was used during training
            if self.normalize and self.mean_ is not None:
                X_recon_np = self._denormalize(X_recon_np)
            
            return X_recon_np
        else:
            # Manual decoding for sklearn MLP
            encoded = H_encoded
            if len(self.model.coefs_) >= 2:
                hidden_input = encoded @ self.model.coefs_[1] + self.model.intercepts_[1]
                hidden_output = np.maximum(hidden_input, 0)
                if len(self.model.coefs_) >= 3:
                    output = hidden_output @ self.model.coefs_[2] + self.model.intercepts_[2]
                else:
                    output = hidden_output
            else:
                output = encoded
            
            # Denormalize if normalization was used
            if self.normalize and self.mean_ is not None:
                output = self._denormalize(output)
            
            return output

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
    def apply_normalization(y_data, norm_mode='max'):
        if norm_mode == 'max':
            max_val = np.max(y_data)
            return y_data / max_val if max_val != 0 else y_data
        elif norm_mode == 'area':
            area = np.trapezoid(y_data)
            return y_data / area if area != 0 else y_data
        elif norm_mode == 'snv':
            mean = np.mean(y_data)
            std = np.std(y_data)
            return (y_data - mean) / std if std != 0 else y_data
        return y_data

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

class AdaptiveMineralFilter(BaseEstimator, TransformerMixin):
    """
    Implements Iterative Re-weighted PCA (Robust PCA variant).
    Solves X = L (Background) + S (Signal) + N (Noise).
    """
    def __init__(self, n_components=5, max_iter=2, contamination=0.1, organic_ranges=[(2800, 3050), (1600, 1750)]):
        self.n_components = n_components
        self.max_iter = max_iter
        self.contamination = contamination
        self.organic_ranges = organic_ranges
        self.background_pca = None
    
    def fit(self, X, y=None, wavenumbers=None):
        if wavenumbers is None:
            self.background_pca = PCA(n_components=self.n_components).fit(X)
            return self
        
        # 1. Mask organic regions to learn background only from mineral regions
        mask = np.ones(len(wavenumbers), dtype=bool)
        for start, end in self.organic_ranges:
            mask &= ~((wavenumbers >= start) & (wavenumbers <= end))
        X_masked = X[:, mask]
        
        # 2. Iterative Robust Learning
        pca_temp = PCA(n_components=self.n_components)
        pca_temp.fit(X_masked)
        
        # Calculate reconstruction error to find outliers (organic-rich samples)
        X_rec = pca_temp.inverse_transform(pca_temp.transform(X_masked))
        residuals = np.sum((X_masked - X_rec)**2, axis=1)
        
        # Keep only the cleanest samples (pure background)
        cutoff = np.percentile(residuals, 100 * (1 - self.contamination))
        clean_indices = residuals <= cutoff
        X_clean = X[clean_indices]
        
        # 3. Final fit on clean background
        self.background_pca = PCA(n_components=self.n_components)
        self.background_pca.fit(X_clean)
        return self
    
    def transform(self, X):
        if self.background_pca is None: return X
        X_bg = self.background_pca.inverse_transform(self.background_pca.transform(X))
        return X - X_bg  # Return Residuals (The Signal)

    def get_explanation(self, x_spectrum):
        x = x_spectrum.reshape(1, -1)
        bg = self.background_pca.inverse_transform(self.background_pca.transform(x)).flatten()
        return x.flatten(), bg, x.flatten() - bg

# ----------------------------------------------------
# 📊 【Matplotlib Canvas 和 Plot Window 类】
# ----------------------------------------------------
class PlotStyleConfig:
    """通用的绘图样式配置类，用于统一管理样式参数"""
    def __init__(self, parent_dialog=None):
        self.parent_dialog = parent_dialog
        self.settings = QSettings("GTLab", "SpectraPro_v4")
        
    def get_default_style_params(self):
        """获取默认样式参数"""
        return {
            # Figure
            'fig_width': 10.0,
            'fig_height': 6.0,
            'fig_dpi': 300,
            'aspect_ratio': 0.6,
            
            # Font
            'font_family': 'Times New Roman',
            'axis_title_fontsize': 20,
            'tick_label_fontsize': 16,
            'legend_fontsize': 10,
            'title_fontsize': 18,
            
            # Lines
            'line_width': 1.2,
            'line_style': '-',
            'marker_size': 4,
            'marker_style': 'o',
            
            # Ticks
            'tick_direction': 'in',
            'tick_len_major': 8,
            'tick_len_minor': 4,
            'tick_width': 1.0,
            
            # Grid
            'show_grid': True,
            'grid_alpha': 0.2,
            'grid_linestyle': '-',
            
            # Spines
            'spine_top': True,
            'spine_bottom': True,
            'spine_left': True,
            'spine_right': True,
            'spine_width': 2.0,
            
            # Legend
            'show_legend': True,
            'legend_frame': True,
            'legend_loc': 'best',
            
            # Colors
            'color_raw': 'gray',
            'color_fit': 'blue',
            'color_residual': 'black',
            
            # Text labels
            'title_text': '',
            'validation_title_fontsize': 18,
            'validation_title_pad': 10.0,
            'validation_title_show': True,
            'xlabel_text': 'Wavenumber (cm⁻¹)',
            'validation_xlabel_fontsize': 20,
            'validation_xlabel_pad': 10.0,
            'validation_xlabel_show': True,
            'ylabel_main_text': 'Intensity',
            'ylabel_residual_text': 'Residuals',
            'validation_ylabel_fontsize': 20,
            'validation_ylabel_pad': 10.0,
            'validation_ylabel_show': True,
            'legend_raw_label': 'Raw Low-Conc. Spectrum',
            'legend_fit_label': 'Fitted Organic Contribution',
            'show_label_a': True,
            'show_label_b': True,
            'label_a_text': '(A)',
            'label_b_text': '(B)',
        }
    
    def load_style_params(self, window_name):
        """从QSettings加载样式参数"""
        params = self.get_default_style_params()
        prefix = f"{window_name}/style/"
        
        for key in params.keys():
            value = self.settings.value(f"{prefix}{key}", params[key])
            # 类型转换
            if isinstance(params[key], bool):
                params[key] = value == 'true' if isinstance(value, str) else bool(value)
            elif isinstance(params[key], int):
                params[key] = int(value) if value is not None else params[key]
            elif isinstance(params[key], float):
                params[key] = float(value) if value is not None else params[key]
            else:
                params[key] = value if value is not None else params[key]
        
        return params
    
    def save_style_params(self, window_name, params):
        """保存样式参数到QSettings"""
        prefix = f"{window_name}/style/"
        for key, value in params.items():
            self.settings.setValue(f"{prefix}{key}", value)
        self.settings.sync()
    
    def apply_style_to_axes(self, ax, params):
        """将样式参数应用到matplotlib axes（发表级别质量）"""
        # 强制使用 Times New Roman 字体（发表级别要求）
        font_family = 'Times New Roman'
        axis_title_fontsize = params.get('axis_title_fontsize', 20)
        tick_label_fontsize = params.get('tick_label_fontsize', 16)
        
        # 启用 LaTeX 数学格式支持
        plt.rcParams['text.usetex'] = False  # 如果系统有 LaTeX，可以设为 True
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
        
        # 设置标签字体（强制 Times New Roman）
        ax.xaxis.label.set_fontsize(axis_title_fontsize)
        ax.yaxis.label.set_fontsize(axis_title_fontsize)
        ax.title.set_fontsize(params.get('title_fontsize', 18))
        ax.xaxis.label.set_fontfamily(font_family)
        ax.yaxis.label.set_fontfamily(font_family)
        ax.title.set_fontfamily(font_family)
        
        # 发表级别刻度设置：direction='in', top=True, right=True
        ax.tick_params(axis='both', which='major', 
                      direction='in',  # 强制向内
                      length=params.get('tick_len_major', 8),
                      width=params.get('tick_width', 1.0),
                      labelsize=tick_label_fontsize,
                      top=True,  # 顶部刻度
                      right=True)  # 右侧刻度
        ax.tick_params(axis='both', which='minor',
                      direction='in',  # 强制向内
                      length=params.get('tick_len_minor', 4),
                      width=params.get('tick_width', 1.0),
                      top=True,  # 顶部刻度
                      right=True)  # 右侧刻度
        
        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(tick_label_fontsize)
        
        # 发表级别边框设置：linewidth=1.5，所有边框可见
        ax.spines['top'].set_visible(True)  # 强制显示顶部边框
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)  # 强制显示右侧边框
        
        spine_width = 1.5  # 发表级别标准：1.5
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        
        # 网格设置
        if params.get('show_grid', True):
            ax.grid(True, alpha=params.get('grid_alpha', 0.2),
                   linestyle=params.get('grid_linestyle', '-'))
        else:
            ax.grid(False)
        
        # 图例设置（强制 Times New Roman）
        legend = ax.get_legend()
        if legend:
            legend_fontsize = params.get('legend_fontsize', 10)
            try:
                legend.set_fontsize(legend_fontsize)
            except AttributeError:
                for text in legend.get_texts():
                    text.set_fontsize(legend_fontsize)
            
            # 强制使用 Times New Roman
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            legend_font.set_family(font_family)
            legend_font.set_size(legend_fontsize)
            for text in legend.get_texts():
                text.set_fontproperties(legend_font)
            legend.set_frame_on(params.get('legend_frame', True))
            if params.get('legend_loc'):
                legend.set_loc(params.get('legend_loc', 'best'))

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.figure = fig
        self.default_xlim = (0, 1)
        self.default_ylim = (0, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

class NMFResultWindow(QDialog):
    """[新增] NMF 分析结果独立窗口（参考4.py，所有参数在NMF分析中）"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 900)
        self.main_layout = QVBoxLayout(self)
        
        # 创建水平布局，左侧是图表，右侧是控制面板
        content_layout = QHBoxLayout()
        
        # 左侧：图表区域
        left_panel = QVBoxLayout()
        self.canvas = MplCanvas(self, width=12, height=9, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        h_layout = QHBoxLayout()
        self.export_button = QPushButton("导出 NMF 结果 (W & H)")
        self.export_button.clicked.connect(self.export_data)
        h_layout.addStretch(1)
        h_layout.addWidget(self.export_button)
        h_layout.addStretch(1)
        
        left_panel.addLayout(h_layout)
        left_panel.addWidget(self.toolbar)
        left_panel.addWidget(self.canvas)
        
        # 右侧：控制面板
        right_panel = QVBoxLayout()
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel)
        right_panel_widget.setMaximumWidth(250)
        right_panel_widget.setMinimumWidth(200)
        
        # 目标组分选择组
        target_group = QGroupBox("目标组分选择")
        target_layout = QVBoxLayout(target_group)
        
        self.target_component_button_group = QButtonGroup()
        self.target_component_radios = []  # 存储所有单选按钮
        
        target_layout.addWidget(QLabel("请选择目标信号组分："))
        
        # 初始时没有组分，set_data时会更新
        self.target_component_container = QWidget()
        self.target_component_layout = QVBoxLayout(self.target_component_container)
        self.target_component_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.target_component_container)
        
        target_layout.addStretch(1)
        right_panel.addWidget(target_group)
        right_panel.addStretch(1)
        
        # 将左右面板添加到内容布局
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        content_layout.addWidget(left_widget, stretch=3)
        content_layout.addWidget(right_panel_widget, stretch=0)
        
        self.main_layout.addLayout(content_layout)
        
        self.W = None
        self.H = None
        self.common_x = None
        self.sample_labels = []
        self.style_params = {}
        self.n_components = 0
        self.target_component_index = 0  # 默认选择第一个组分

    def set_data(self, W, H, common_x, style_params, sample_labels):
        self.W = W
        self.H = H
        self.common_x = common_x
        self.sample_labels = sample_labels
        self.style_params = style_params
        self.n_components = H.shape[0] if H is not None else 0
        
        # 更新目标组分选择UI
        self._update_target_component_radios()
        
        self.plot_results(style_params)
    
    def _update_target_component_radios(self):
        """更新目标组分选择单选按钮"""
        # 清除旧的单选按钮
        for radio in self.target_component_radios:
            self.target_component_button_group.removeButton(radio)
            radio.deleteLater()
        self.target_component_radios.clear()
        
        # 清除布局
        while self.target_component_layout.count():
            item = self.target_component_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 创建新的单选按钮
        if self.n_components > 0:
            # 获取NMF组分图例重命名（如果有）
            nmf_legend_names = self.style_params.get('nmf_legend_names', {})
            
            for i in range(self.n_components):
                comp_label = f"Component {i+1}"
                display_label = nmf_legend_names.get(comp_label, comp_label)
                
                radio = QRadioButton(display_label)
                radio.setChecked(i == self.target_component_index)  # 默认选择第一个
                self.target_component_button_group.addButton(radio, i)
                self.target_component_radios.append(radio)
                self.target_component_layout.addWidget(radio)
                
                # 连接信号，当选择改变时更新索引并通知父窗口
                radio.toggled.connect(lambda checked, idx=i: self._on_target_component_changed(idx) if checked else None)
    
    def _on_target_component_changed(self, index):
        """当目标组分选择改变时调用"""
        self.target_component_index = index
        # 通知父窗口（如果存在）
        parent = self.parent()
        if parent and hasattr(parent, 'update_nmf_target_component'):
            parent.update_nmf_target_component(index)
    
    def get_target_component_index(self):
        """返回当前选中的目标组分索引"""
        return self.target_component_index
        
    def export_data(self):
        if self.W is None or self.H is None:
            QMessageBox.warning(self, "警告", "没有数据可以导出。")
            return
            
        save_dir = QFileDialog.getExistingDirectory(self, "选择 NMF 结果保存目录")
        if not save_dir: return

        # 导出 H (Spectra)
        h_df = pd.DataFrame(self.H.T, index=self.common_x, columns=[f"Component_{i+1}" for i in range(self.H.shape[0])])
        h_df.index.name = "Wavenumber"
        h_df.to_csv(os.path.join(save_dir, "NMF_H_Components.csv"))
        
        # 导出 W (Weights)
        w_df = pd.DataFrame(self.W, columns=[f"Weight_Comp_{i+1}" for i in range(self.W.shape[1])])
        w_df.index = self.sample_labels
        w_df.index.name = "Sample Name"
        w_df.to_csv(os.path.join(save_dir, "NMF_W_Weights.csv"))
        
        QMessageBox.information(self, "完成", f"NMF 结果已导出到 {save_dir}。")
    
    def closeEvent(self, event):
        """窗口关闭时，保存目标组分选择到父窗口"""
        parent = self.parent()
        if parent and hasattr(parent, 'update_nmf_target_component'):
            parent.update_nmf_target_component(self.target_component_index)
        super().closeEvent(event)

    def plot_results(self, style_params):
        """绘制NMF结果（参考4.py，每次都用fig.clear()）"""
        # 使用现有的figure，只清除内容（与4.py保持一致）
        fig = self.canvas.figure
        fig.clear()  # 使用clear而不是clf，保持窗口状态
        
        # 确保 Matplotlib 有足够的空间
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        n_components = self.H.shape[0]
        
        # Comp Colors
        c1_color = style_params['comp1_color']
        c2_color = style_params['comp2_color']
        colors = [c1_color, c2_color] + ['green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal', 'darkred']

        # 提取绘图参数
        is_derivative = style_params.get('is_derivative', False)
        global_stack_offset = style_params.get('global_stack_offset', 0.0)
        global_scale_factor = style_params.get('global_scale_factor', 1.0)
        individual_y_params = style_params.get('individual_y_params', {})
        control_data_list = style_params.get('control_data_list', [])
        
        # 获取NMF组分图例重命名
        nmf_legend_names = style_params.get('nmf_legend_names', {})
        
        # 绘制 H (Components/Spectra)
        for i in range(n_components):
            comp_color = colors[i % len(colors)]
            y_data = self.H[i].copy()
            
            # 应用动态范围压缩预处理（在对数/平方根变换之前）
            comp_label = f"Component {i+1}"
            ind_params = individual_y_params.get(comp_label, {'scale': 1.0, 'offset': 0.0, 'transform': 'none', 'transform_params': {}})
            
            transform_mode = ind_params.get('transform', 'none')
            transform_params = ind_params.get('transform_params', {})
            
            if transform_mode == 'log':
                y_data = DataPreProcessor.apply_log_transform(y_data,
                    base=transform_params.get('base', 10),
                    offset=transform_params.get('offset', 1.0))
            elif transform_mode == 'sqrt':
                y_data = DataPreProcessor.apply_sqrt_transform(y_data,
                    offset=transform_params.get('offset', 0.0))
            
            # 应用二阶导数（如果启用）
            if is_derivative:
                y_data = np.gradient(np.gradient(y_data))
            
            # 应用全局缩放
            y_data = y_data * global_scale_factor
            
            # 应用独立Y轴参数（如果存在）
            y_data = y_data * ind_params['scale'] + ind_params['offset']
            
            # 应用堆叠偏移
            y_final = y_data + (i * global_stack_offset)
            
            # 使用重命名后的图例名称（如果存在），否则使用默认名称
            display_label = nmf_legend_names.get(comp_label, comp_label)
            
            ax1.plot(self.common_x, y_final, 
                     label=display_label, 
                     color=comp_color, 
                     linewidth=style_params['comp_line_width'],
                     linestyle=style_params['comp_line_style'])
            
            # NMF组分的峰值检测（如果启用）
            if style_params.get('peak_detection_enabled', False):
                # 获取峰值检测参数（从父窗口）
                parent = self.parent()
                if parent and hasattr(parent, 'peak_check') and parent.peak_check.isChecked():
                    peak_params = {
                        'peak_detection_enabled': True,
                        'peak_height_threshold': parent.peak_height_spin.value() if hasattr(parent, 'peak_height_spin') else 0.0,
                        'peak_distance_min': parent.peak_distance_spin.value() if hasattr(parent, 'peak_distance_spin') else 10,
                        'peak_prominence': parent.peak_prominence_spin.value() if hasattr(parent, 'peak_prominence_spin') else 0.0,
                        'peak_width': parent.peak_width_spin.value() if hasattr(parent, 'peak_width_spin') else None,
                        'peak_wlen': parent.peak_wlen_spin.value() if hasattr(parent, 'peak_wlen_spin') else None,
                        'peak_rel_height': parent.peak_rel_height_spin.value() if hasattr(parent, 'peak_rel_height_spin') else None,
                        'peak_show_label': parent.peak_show_label_check.isChecked() if hasattr(parent, 'peak_show_label_check') else True,
                        'peak_label_font': parent.peak_label_font_combo.currentText() if hasattr(parent, 'peak_label_font_combo') else 'Times New Roman',
                        'peak_label_size': parent.peak_label_size_spin.value() if hasattr(parent, 'peak_label_size_spin') else 10,
                        'peak_label_color': parent.peak_label_color_input.text().strip() or 'black' if hasattr(parent, 'peak_label_color_input') else 'black',
                        'peak_label_bold': parent.peak_label_bold_check.isChecked() if hasattr(parent, 'peak_label_bold_check') else False,
                        'peak_label_rotation': parent.peak_label_rotation_spin.value() if hasattr(parent, 'peak_label_rotation_spin') else 0.0,
                        'peak_marker_shape': parent.peak_marker_shape_combo.currentText() if hasattr(parent, 'peak_marker_shape_combo') else 'x',
                        'peak_marker_size': parent.peak_marker_size_spin.value() if hasattr(parent, 'peak_marker_size_spin') else 10,
                        'peak_marker_color': parent.peak_marker_color_input.text().strip() or '' if hasattr(parent, 'peak_marker_color_input') else '',
                    }
                    # 使用MplPlotWindow的detect_and_plot_peaks方法
                    # 由于NMFResultWindow没有这个方法，我们需要创建一个临时实例或直接调用函数
                    # 这里我们直接调用find_peaks并绘制
                    try:
                        y_detect = self.H[i]
                        y_max = np.max(y_detect)
                        y_min = np.min(y_detect)
                        y_range = y_max - y_min
                        
                        peak_kwargs = {}
                        peak_height = peak_params.get('peak_height_threshold', 0.0)
                        if peak_height == 0 or (peak_height > y_range and y_range > 0):
                            if y_max > 0:
                                peak_height = y_max * 0.05
                            else:
                                peak_height = abs(np.mean(y_detect)) + np.std(y_detect) * 0.5
                        if peak_height > 0 or peak_height < 0:
                            peak_kwargs['height'] = peak_height
                        
                        peak_distance = peak_params.get('peak_distance_min', 10)
                        if peak_distance == 0 or peak_distance > len(y_detect) * 0.5:
                            peak_distance = max(1, int(len(y_detect) * 0.03))
                        if peak_distance > 0:
                            peak_kwargs['distance'] = peak_distance
                        
                        if len(peak_kwargs) == 0:
                            peak_kwargs = {'height': max(np.mean(y_detect), y_max * 0.05) if y_max > 0 else 0, 'distance': max(1, int(len(y_detect) * 0.03))}
                        
                        peaks, _ = find_peaks(y_detect, **peak_kwargs)
                        
                        if len(peaks) > 0:
                            marker_shape = peak_params.get('peak_marker_shape', 'x')
                            marker_size = peak_params.get('peak_marker_size', 10)
                            marker_color = peak_params.get('peak_marker_color', '') or comp_color
                            
                            ax1.plot(self.common_x[peaks], self.H[i][peaks], marker_shape, 
                                   color=marker_color, markersize=marker_size)
                            
                            if peak_params.get('peak_show_label', True):
                                label_font = peak_params.get('peak_label_font', 'Times New Roman')
                                label_size = peak_params.get('peak_label_size', 10)
                                label_color = peak_params.get('peak_label_color', 'black')
                                label_bold = peak_params.get('peak_label_bold', False)
                                label_rotation = peak_params.get('peak_label_rotation', 0.0)
                                
                                font_props = {'fontsize': label_size, 'color': label_color, 'fontfamily': label_font, 'ha': 'center', 'va': 'bottom'}
                                if label_bold:
                                    font_props['weight'] = 'bold'
                                if label_rotation != 0:
                                    font_props['rotation'] = label_rotation
                                
                                for px, py in zip(self.common_x[peaks], self.H[i][peaks]):
                                    wavenumber_str = f"{px:.1f}"
                                    ax1.text(px, py, wavenumber_str, **font_props)
                    except Exception as e:
                        print(f"NMF峰值检测失败: {e}")
        
        # 绘制对照组（如果存在）
        if control_data_list:
            control_colors = ['black', 'darkblue', 'darkred', 'darkgreen', 'darkmagenta']
            for idx, ctrl_data in enumerate(control_data_list):
                ctrl_y = ctrl_data['y'].copy()
                
                # 应用二阶导数（如果启用）
                if is_derivative:
                    ctrl_y = np.gradient(np.gradient(ctrl_y))
                
                # 应用全局缩放
                ctrl_y = ctrl_y * global_scale_factor
                
                # 应用独立Y轴参数（如果存在）
                ctrl_label = ctrl_data['label']
                ind_params = individual_y_params.get(ctrl_label, {'scale': 1.0, 'offset': 0.0})
                ctrl_y = ctrl_y * ind_params['scale'] + ind_params['offset']
                
                # 应用堆叠偏移（对照组放在最后）
                ctrl_y_final = ctrl_y + (n_components * global_stack_offset)
                
                ctrl_color = control_colors[idx % len(control_colors)]
                ax1.plot(ctrl_data['x'], ctrl_y_final,
                        label=f"{ctrl_label} (Ref)",
                        color=ctrl_color,
                        linewidth=style_params['comp_line_width'],
                        linestyle='--',  # 对照组用虚线
                        alpha=0.7)
            
        # 绘制垂直参考线（如果存在）
        vertical_lines = style_params.get('vertical_lines', [])
        if vertical_lines:
            vertical_line_color = style_params.get('vertical_line_color', '#034DFB')
            vertical_line_style = style_params.get('vertical_line_style', '--')
            vertical_line_width = style_params.get('vertical_line_width', 0.8)
            vertical_line_alpha = style_params.get('vertical_line_alpha', 0.8)
            for line_x in vertical_lines:
                ax1.axvline(x=line_x, color=vertical_line_color, linestyle=vertical_line_style,
                          linewidth=vertical_line_width, alpha=vertical_line_alpha)
        
        if style_params['x_axis_invert']: ax1.invert_xaxis()
        ax1.legend(fontsize=style_params['legend_font_size'])
        # 使用自定义标题和轴标签
        top_title = style_params.get('nmf_top_title', 'Extracted Spectra (Components)')
        bottom_title = style_params.get('nmf_bottom_title', 'Concentration Weights (vs. Sample)')
        top_xlabel = style_params.get('nmf_top_xlabel', 'Wavenumber ($\\mathrm{cm^{-1}}$)')
        top_ylabel = style_params.get('nmf_top_ylabel', 'Intensity (Arb. Unit)')
        bottom_xlabel = style_params.get('nmf_bottom_xlabel', 'Sample Name')
        bottom_ylabel = style_params.get('nmf_bottom_ylabel', 'Weight (Arb. Unit)')
        
        # 使用GUI中的标题控制参数
        top_title_fontsize = style_params.get('nmf_top_title_fontsize', style_params['title_font_size'])
        top_title_pad = style_params.get('nmf_top_title_pad', 10.0)
        top_title_show = style_params.get('nmf_top_title_show', True)
        
        if top_title_show:
            ax1.set_title(top_title, fontsize=top_title_fontsize, pad=top_title_pad)
        
        # 使用GUI中的上图X轴标题控制参数
        top_xlabel_fontsize = style_params.get('nmf_top_xlabel_fontsize', style_params['label_font_size'])
        top_xlabel_pad = style_params.get('nmf_top_xlabel_pad', 10.0)
        top_xlabel_show = style_params.get('nmf_top_xlabel_show', True)
        
        if top_xlabel_show:
            ax1.set_xlabel(top_xlabel, fontsize=top_xlabel_fontsize, labelpad=top_xlabel_pad)
        
        # 使用GUI中的上图Y轴标题控制参数
        top_ylabel_fontsize = style_params.get('nmf_top_ylabel_fontsize', style_params['label_font_size'])
        top_ylabel_pad = style_params.get('nmf_top_ylabel_pad', 10.0)
        top_ylabel_show = style_params.get('nmf_top_ylabel_show', True)
        
        if top_ylabel_show:
            ax1.set_ylabel(top_ylabel, fontsize=top_ylabel_fontsize, labelpad=top_ylabel_pad)
        
        ax1.tick_params(labelsize=style_params['tick_font_size'])

        # 绘制 W (Weights/Concentrations)
        sample_indices = np.arange(len(self.sample_labels))
        
        for i in range(n_components):
            ax2.plot(sample_indices, self.W[:, i], 
                     marker=style_params['weight_marker_style'], 
                     markersize=style_params['weight_marker_size'],
                     linestyle=style_params['weight_line_style'],
                     linewidth=style_params['weight_line_width'],
                     label=f"Comp {i+1} Weight", 
                     color=colors[i % len(colors)])
        
        ax2.set_xticks(sample_indices)
        ax2.set_xticklabels(self.sample_labels, rotation=45, ha='right', fontsize=style_params['tick_font_size']) 
        ax2.legend(fontsize=style_params['legend_font_size'])
        
        # 使用GUI中的标题控制参数
        bottom_title_fontsize = style_params.get('nmf_bottom_title_fontsize', style_params['title_font_size'])
        bottom_title_pad = style_params.get('nmf_bottom_title_pad', 10.0)
        bottom_title_show = style_params.get('nmf_bottom_title_show', True)
        
        if bottom_title_show:
            ax2.set_title(bottom_title, fontsize=bottom_title_fontsize, pad=bottom_title_pad)
        
        # 使用GUI中的下图X轴标题控制参数
        bottom_xlabel_fontsize = style_params.get('nmf_bottom_xlabel_fontsize', style_params['label_font_size'])
        bottom_xlabel_pad = style_params.get('nmf_bottom_xlabel_pad', 10.0)
        bottom_xlabel_show = style_params.get('nmf_bottom_xlabel_show', True)
        
        if bottom_xlabel_show:
            ax2.set_xlabel(bottom_xlabel, fontsize=bottom_xlabel_fontsize, labelpad=bottom_xlabel_pad)
        
        # 使用GUI中的下图Y轴标题控制参数
        bottom_ylabel_fontsize = style_params.get('nmf_bottom_ylabel_fontsize', style_params['label_font_size'])
        bottom_ylabel_pad = style_params.get('nmf_bottom_ylabel_pad', 10.0)
        bottom_ylabel_show = style_params.get('nmf_bottom_ylabel_show', True)
        
        if bottom_ylabel_show:
            ax2.set_ylabel(bottom_ylabel, fontsize=bottom_ylabel_fontsize, labelpad=bottom_ylabel_pad)
        
        ax2.tick_params(labelsize=style_params['tick_font_size'])
        
        # 应用主菜单的出版质量样式控制参数
        # 字体设置
        font_family = style_params.get('font_family', 'Times New Roman')
        current_font = 'Times New Roman' if font_family == 'Times New Roman' else font_family
        
        # 刻度样式
        tick_direction = style_params.get('tick_direction', 'in')
        tick_len_major = style_params.get('tick_len_major', 8)
        tick_len_minor = style_params.get('tick_len_minor', 4)
        tick_width = style_params.get('tick_width', 1.0)
        tick_label_fontsize = style_params.get('tick_label_fontsize', style_params['tick_font_size'])
        
        ax1.tick_params(labelsize=tick_label_fontsize, direction=tick_direction, width=tick_width, labelfontfamily=current_font)
        ax1.tick_params(which='major', length=tick_len_major)
        ax1.tick_params(which='minor', length=tick_len_minor)
        ax2.tick_params(labelsize=tick_label_fontsize, direction=tick_direction, width=tick_width, labelfontfamily=current_font)
        ax2.tick_params(which='major', length=tick_len_major)
        ax2.tick_params(which='minor', length=tick_len_minor)
        
        # 边框设置 (Spines)
        border_sides = []
        if style_params.get('spine_top', True): border_sides.append('top')
        if style_params.get('spine_bottom', True): border_sides.append('bottom')
        if style_params.get('spine_left', True): border_sides.append('left')
        if style_params.get('spine_right', True): border_sides.append('right')
        border_linewidth = style_params.get('spine_width', 2.0)
        
        for side in ['top', 'right', 'left', 'bottom']:
            if side in border_sides:
                ax1.spines[side].set_visible(True)
                ax1.spines[side].set_linewidth(border_linewidth)
                ax2.spines[side].set_visible(True)
                ax2.spines[side].set_linewidth(border_linewidth)
            else:
                ax1.spines[side].set_visible(False)
                ax2.spines[side].set_visible(False)
        
        # 网格设置
        if style_params.get('show_grid', False):
            ax1.grid(True, alpha=style_params.get('grid_alpha', 0.3))
            ax2.grid(True, alpha=style_params.get('grid_alpha', 0.3))
        
        # 图例设置（使用主菜单参数）
        if style_params.get('show_legend', True):
            legend_fontsize = style_params.get('legend_fontsize', style_params['legend_font_size'])
            legend_frame = style_params.get('legend_frame', True)
            legend_loc = style_params.get('legend_loc', 'best')
            
            # 设置图例字体（支持中文）
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            if font_family == 'SimHei':
                legend_font.set_family('sans-serif')
            else:
                legend_font.set_family(font_family)
            legend_font.set_size(legend_fontsize)
            
            legend_ncol = style_params.get('legend_ncol', 1)
            legend_columnspacing = style_params.get('legend_columnspacing', 2.0)
            legend_labelspacing = style_params.get('legend_labelspacing', 0.5)
            legend_handlelength = style_params.get('legend_handlelength', 2.0)
            
            ax1.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                      ncol=legend_ncol, columnspacing=legend_columnspacing, 
                      labelspacing=legend_labelspacing, handlelength=legend_handlelength)
            ax2.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                      ncol=legend_ncol, columnspacing=legend_columnspacing, 
                      labelspacing=legend_labelspacing, handlelength=legend_handlelength)
        
        # 添加纵横比控制
        aspect_ratio = style_params.get('aspect_ratio', 0.0)
        if aspect_ratio > 0:
            ax1.set_box_aspect(aspect_ratio)
            ax2.set_box_aspect(aspect_ratio)
        else:
            ax1.set_aspect('auto')
            ax2.set_aspect('auto')
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
            fig.tight_layout()
        self.canvas.draw()


class MplPlotWindow(QDialog):
    def __init__(self, group_name, initial_geometry=(100, 100, 1000, 600), parent=None):
        super().__init__(parent)
        self.group_name = group_name
        self.setWindowTitle(f"光谱图 - 组别: {group_name}")
        self.setGeometry(*initial_geometry) 
        self.main_layout = QVBoxLayout(self)
        
        # 尺寸在 update_plot 中根据 params 调整
        self.canvas = MplCanvas(self) 
        self.main_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)
        self.setMinimumSize(400, 300) 

        self.last_geometry = initial_geometry
        self.moveEvent = self._update_geometry_on_move
        self.resizeEvent = self._update_geometry_on_resize
        
        # 存储当前绘制的数据和 Axes 对象，用于叠加绘图
        self.current_plot_data = defaultdict(lambda: {'x': np.array([]), 'y': np.array([]), 'label': '', 'color': 'gray', 'type': 'Individual'})
        self.current_ax = self.canvas.axes
        
        # 初始化标题状态
        self.has_title = False

    def _update_geometry_on_move(self, event):
        current_rect = self.geometry()
        self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        super().moveEvent(event)

    def _update_geometry_on_resize(self, event):
        current_rect = self.geometry()
        self.last_geometry = (current_rect.x(), current_rect.y(), current_rect.width(), current_rect.height())
        
        # 与数据处理.py保持一致：不调整figure大小，让matplotlib自动适应窗口
        # tight_layout会自动调整布局以适应窗口大小
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
                self.canvas.figure.tight_layout()
            self.canvas.draw()
        except:
            pass
        
        super().resizeEvent(event)

    def detect_and_plot_peaks(self, ax, x_data, y_detect, y_final, plot_params, color='blue'):
        """
        通用的波峰检测和绘制函数
        使用统一的峰值检测辅助函数
        x_data: X轴数据（波数）
        y_detect: 用于检测的Y数据（去除偏移）
        y_final: 用于绘制的Y数据（包含偏移）
        plot_params: 绘图参数字典
        color: 线条颜色（用于标记颜色默认值）
        """
        # 使用统一的峰值检测函数
        unified_detect_and_plot_peaks(ax, x_data, y_detect, y_final, plot_params, color)

    def update_plot(self, plot_params):
        """
        核心绘图逻辑 - 保持与数据处理.py一致的绘图方式
        使用ax.cla()而不是figure.clf()，保持布局一致性
        """
        # 使用现有的axes，只清除内容（与数据处理.py保持一致）
        ax = self.canvas.axes
        
        # 检查是否手动缩放过（与数据处理.py保持一致）
        try:
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()
            # 检查是否是默认范围之外的缩放
            is_zoomed = not np.allclose(current_xlim, self.canvas.default_xlim) or \
                        not np.allclose(current_ylim, self.canvas.default_ylim)
        except AttributeError:
            is_zoomed = False
            current_xlim = None
            current_ylim = None
        
        # 只清除axes内容，保持axes对象和布局（与数据处理.py一致）
        ax.cla()
        
        # 清空旧数据引用
        self.current_plot_data.clear()
        self.current_ax = ax

        # --- 2. 提取基础参数 ---
        grouped_files_data = plot_params.get('grouped_files_data', [])
        if not grouped_files_data:
            # 如果没有提供数据，尝试从当前绘图数据重建（用于样式更新）
            # 但更安全的方式是返回，让调用者重新读取数据
            print("警告: update_plot 缺少 grouped_files_data，无法更新绘图")
            return
        control_data_list = plot_params.get('control_data_list', []) 
        individual_y_params = plot_params.get('individual_y_params', {}) 
        
        # --- 3. 提取显示/模式参数 ---
        plot_mode = plot_params.get('plot_mode', 'Normal Overlay')
        show_y_values = plot_params.get('show_y_values', True)
        is_derivative = plot_params['is_derivative']
        x_axis_invert = plot_params['x_axis_invert'] 
        
        global_stack_offset = plot_params['global_stack_offset']
        global_scale_factor = plot_params['global_scale_factor']
        
        # --- 4. 提取预处理参数 ---
        qc_enabled = plot_params.get('qc_enabled', False)
        qc_threshold = plot_params.get('qc_threshold', 5.0)
        is_baseline_als = plot_params.get('is_baseline_als', False)
        als_lam = plot_params.get('als_lam', 10000)
        als_p = plot_params.get('als_p', 0.005)
        is_baseline = plot_params.get('is_baseline', False) 
        baseline_points = plot_params.get('baseline_points', 50)
        baseline_poly = plot_params.get('baseline_poly', 3)
        is_smoothing = plot_params['is_smoothing']
        smoothing_window = plot_params['smoothing_window']
        smoothing_poly = plot_params['smoothing_poly']
        normalization_mode = plot_params['normalization_mode']
        
        # Bose-Einstein
        is_be_correction = plot_params.get('is_be_correction', False)
        be_temp = plot_params.get('be_temp', 300.0)
        
        # 全局动态变换和整体Y轴偏移
        global_transform_mode = plot_params.get('global_transform_mode', '无')
        global_log_base_text = plot_params.get('global_log_base', '10')
        global_log_base = float(global_log_base_text) if global_log_base_text == '10' else np.e
        global_log_offset = plot_params.get('global_log_offset', 1.0)
        global_sqrt_offset = plot_params.get('global_sqrt_offset', 0.0)
        global_y_offset = plot_params.get('global_y_offset', 0.0)
        
        # --- 5. 提取出版样式参数 ---
        line_width = plot_params['line_width']
        line_style = plot_params['line_style']
        font_family = plot_params['font_family']
        axis_title_fontsize = plot_params['axis_title_fontsize']
        tick_label_fontsize = plot_params['tick_label_fontsize']
        legend_fontsize = plot_params.get('legend_fontsize', 10)
        
        show_legend = plot_params['show_legend']
        legend_frame = plot_params['legend_frame']
        legend_loc = plot_params['legend_loc']
        
        # 图例高级控制参数
        legend_ncol = plot_params.get('legend_ncol', 1)
        legend_columnspacing = plot_params.get('legend_columnspacing', 2.0)
        legend_labelspacing = plot_params.get('legend_labelspacing', 0.5)
        legend_handlelength = plot_params.get('legend_handlelength', 2.0)
        
        show_grid = plot_params['show_grid']
        grid_alpha = plot_params['grid_alpha']
        shadow_alpha = plot_params['shadow_alpha']
        main_title_text = plot_params.get('main_title_text', "").strip()
        
        # Aspect Ratio & Plot Style
        aspect_ratio = plot_params.get('aspect_ratio', 0.0)
        plot_style = plot_params.get('plot_style', 'line') # line, scatter
        
        # 设置字体 (仅影响当前 Figure)
        current_font = 'Times New Roman' if font_family == 'Times New Roman' else font_family
        
        # 使用 Viridis 调色板，或用户自定义
        custom_colors = plot_params.get('custom_colors', ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal', 'darkred'])
        
        # 辅助函数：单条数据预处理
        # 注意：此处必须使用 DataPreProcessor 中定义的静态方法
        def preprocess_single_spectrum(x, y):
            y_proc = y.astype(float)
            
            # QC Check
            if qc_enabled and np.max(y_proc) < qc_threshold:
                return None 

            # 1. BE 校正 (前置处理)
            if is_be_correction:
                y_proc = DataPreProcessor.apply_bose_einstein_correction(x, y_proc, be_temp)

            # 平滑
            if is_smoothing:
                y_proc = DataPreProcessor.apply_smoothing(y_proc, smoothing_window, smoothing_poly)
            
            # 基线校正 (优先 AsLS)
            if is_baseline_als:
                b = DataPreProcessor.apply_baseline_als(y_proc, als_lam, als_p)
                y_proc = y_proc - b
                y_proc[y_proc < 0] = 0 
            # 注意：旧版基线校正方法已移除，is_baseline硬编码为False
            
            # 归一化
            if normalization_mode == 'max':
                y_proc = DataPreProcessor.apply_normalization(y_proc, 'max')
            elif normalization_mode == 'area':
                y_proc = DataPreProcessor.apply_normalization(y_proc, 'area')
            elif normalization_mode == 'snv':
                y_proc = DataPreProcessor.apply_snv(y_proc)
            
            # 全局动态范围压缩（如果启用）- 在归一化之后
            if global_transform_mode == '对数变换 (Log)':
                y_proc = DataPreProcessor.apply_log_transform(y_proc, base=global_log_base, offset=global_log_offset)
            elif global_transform_mode == '平方根变换 (Sqrt)':
                y_proc = DataPreProcessor.apply_sqrt_transform(y_proc, offset=global_sqrt_offset)
            
            # 二次导数（如果启用）- 在全局动态变换之后
            if is_derivative:
                d1 = np.gradient(y_proc, x)
                y_proc = np.gradient(d1, x)
            
            # 整体Y轴偏移（预处理最后一步，在二次导数之后）
            y_proc = y_proc + global_y_offset
            
            return y_proc

        # ==========================================
        # A. 预处理所有数据（对照组+组内数据），归一化前处理
        # ==========================================
        # 跟踪Y值的范围（与数据处理.py保持一致）
        max_y_value = -np.inf 
        min_y_value = np.inf
        
        # 收集所有数据（对照组+组内数据），先进行归一化前的预处理
        all_data_before_norm = []  # 存储归一化前的数据
        
        # 1. 处理对照组（归一化前）
        control_data_before_norm = []
        for i, control_data in enumerate(control_data_list):
            x_c = control_data['df']['Wavenumber'].values
            y_c = control_data['df']['Intensity'].values
            
            # 对照文件应用预处理（归一化前）
            temp_y = y_c.astype(float)
            if is_be_correction: temp_y = DataPreProcessor.apply_bose_einstein_correction(x_c, temp_y, be_temp)
            if is_smoothing: temp_y = DataPreProcessor.apply_smoothing(temp_y, smoothing_window, smoothing_poly)
            if is_baseline_als: 
                b = DataPreProcessor.apply_baseline_als(temp_y, als_lam, als_p)
                temp_y = temp_y - b
                temp_y[temp_y < 0] = 0
            
            # 注意：全局动态变换、二次导数和整体Y轴偏移在归一化后统一应用
            
            base_name = os.path.splitext(control_data['filename'])[0]
            control_data_before_norm.append({
                'x': x_c,
                'y': temp_y,
                'base_name': base_name,
                'label': control_data['label'],
                'type': 'control',
                'index': i
            })
            all_data_before_norm.append(temp_y)
        
        # 2. 处理组内数据（归一化前）
        group_data_before_norm = []
        for file_path, x_data, y_data in grouped_files_data:
            y_proc = y_data.astype(float)
            
            # QC Check
            if qc_enabled and np.max(y_proc) < qc_threshold:
                continue
            
            # BE校正
            if is_be_correction:
                y_proc = DataPreProcessor.apply_bose_einstein_correction(x_data, y_proc, be_temp)
            
            # 平滑
            if is_smoothing:
                y_proc = DataPreProcessor.apply_smoothing(y_proc, smoothing_window, smoothing_poly)
            
            # 基线校正
            if is_baseline_als:
                b = DataPreProcessor.apply_baseline_als(y_proc, als_lam, als_p)
                y_proc = y_proc - b
                y_proc[y_proc < 0] = 0
            
            # 注意：全局动态变换、二次导数和整体Y轴偏移在归一化后统一应用
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            group_data_before_norm.append({
                'x': x_data,
                'y': y_proc,
                'base_name': base_name,
                'file_path': file_path,
                'type': 'group'
            })
            all_data_before_norm.append(y_proc)
        
        # 3. 一起归一化（如果启用）
        if normalization_mode != 'none' and all_data_before_norm:
            # 收集所有数据到一个数组进行归一化
            all_y_array = np.array(all_data_before_norm)  # (n_samples, n_features)
            
            if normalization_mode == 'max':
                # Max归一化：每个样本独立归一化到最大值
                max_vals = np.max(all_y_array, axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1  # 避免除零
                all_y_array = all_y_array / max_vals
            elif normalization_mode == 'area':
                # Area归一化：每个样本独立归一化到面积
                # np.trapezoid替代已弃用的np.trapz
                areas = np.trapezoid(all_y_array, axis=1)  # (n_samples,)
                areas = areas[:, np.newaxis]  # 转换为 (n_samples, 1) 以匹配广播
                areas[areas == 0] = 1  # 避免除零
                all_y_array = all_y_array / areas
            elif normalization_mode == 'snv':
                # SNV归一化：每个样本独立标准化
                means = np.mean(all_y_array, axis=1, keepdims=True)
                stds = np.std(all_y_array, axis=1, keepdims=True)
                stds[stds == 0] = 1  # 避免除零
                all_y_array = (all_y_array - means) / stds
            
            # 将归一化后的数据分配回去
            idx = 0
            for item in control_data_before_norm:
                item['y'] = all_y_array[idx]
                idx += 1
            for item in group_data_before_norm:
                item['y'] = all_y_array[idx]
                idx += 1
        
        # ==========================================
        # B. 处理对照组（归一化后）
        # ==========================================
        control_plot_data = []
        for item in control_data_before_norm:
            x_c = item['x']
            temp_y = item['y']
            base_name = item['base_name']
            i = item['index']
            
            # 获取独立参数
            ind_params = individual_y_params.get(base_name, {'scale': 1.0, 'offset': 0.0, 'transform': 'none', 'transform_params': {}})
            
            # 1. 全局动态范围压缩（如果启用）- 在归一化之后
            if global_transform_mode == '对数变换 (Log)':
                temp_y = DataPreProcessor.apply_log_transform(temp_y, base=global_log_base, offset=global_log_offset)
            elif global_transform_mode == '平方根变换 (Sqrt)':
                temp_y = DataPreProcessor.apply_sqrt_transform(temp_y, offset=global_sqrt_offset)
            
            # 2. 应用独立动态范围压缩预处理（在对数/平方根变换之前）
            transform_mode = ind_params.get('transform', 'none')
            transform_params = ind_params.get('transform_params', {})
            
            if transform_mode == 'log':
                temp_y = DataPreProcessor.apply_log_transform(temp_y, 
                    base=transform_params.get('base', 10),
                    offset=transform_params.get('offset', 1.0))
            elif transform_mode == 'sqrt':
                temp_y = DataPreProcessor.apply_sqrt_transform(temp_y,
                    offset=transform_params.get('offset', 0.0))
            
            # 3. 应用缩放
            temp_y = temp_y * global_scale_factor * ind_params['scale']
            
            # 4. 二次导数（如果启用）- 在全局动态变换之后
            if is_derivative:
                d1 = np.gradient(temp_y, x_c)
                temp_y = np.gradient(d1, x_c)
            
            # 5. 整体Y轴偏移（预处理最后一步，在二次导数之后）
            temp_y = temp_y + global_y_offset
            
            # 6. 应用独立偏移和堆叠偏移
            final_y = temp_y + ind_params['offset'] + (i * global_stack_offset) 
            
            # 优先使用individual_y_params中指定的颜色
            file_colors = plot_params.get('file_colors', {})
            if base_name in file_colors:
                color = file_colors[base_name]
            else:
                color = custom_colors[i % len(custom_colors)]
            
            label = item['label'] + " (Ref)"
            control_plot_data.append((x_c, final_y, label, color))
            
            # 绘制：使用 line 或 scatter
            if plot_style == 'line':
                ax.plot(x_c, final_y, label=label, color=color, linestyle='--', linewidth=line_width, alpha=0.7)
            else: # scatter
                ax.plot(x_c, final_y, label=label, color=color, marker='.', linestyle='', markersize=line_width*3, alpha=0.7)

            # 存储数据以备叠加
            self.current_plot_data[base_name] = {'x': x_c, 'y': final_y, 'label': label, 'color': color, 'type': 'Ref'}
            
            # 更新Y值范围
            max_y_value = max(max_y_value, np.max(final_y))
            min_y_value = min(min_y_value, np.min(final_y))

        # ==========================================
        # C. 处理分组数据（归一化后）
        # ==========================================
        processed_group_data = []
        for item in group_data_before_norm:
            x_data = item['x']
            y_clean = item['y']  # 已经归一化
            base_name = item['base_name']
            file_path = item['file_path']
            
            # 检查是否有重命名
            label = plot_params['legend_names'].get(base_name, base_name)
            ind_params = individual_y_params.get(base_name, {'scale': 1.0, 'offset': 0.0, 'transform': 'none', 'transform_params': {}})
            
            # 1. 全局动态范围压缩（如果启用）- 在归一化之后
            y_transformed = y_clean.copy()
            if global_transform_mode == '对数变换 (Log)':
                y_transformed = DataPreProcessor.apply_log_transform(y_transformed, base=global_log_base, offset=global_log_offset)
            elif global_transform_mode == '平方根变换 (Sqrt)':
                y_transformed = DataPreProcessor.apply_sqrt_transform(y_transformed, offset=global_sqrt_offset)
            
            # 2. 应用独立动态范围压缩预处理（在对数/平方根变换之前）
            transform_mode = ind_params.get('transform', 'none')
            transform_params = ind_params.get('transform_params', {})
            
            if transform_mode == 'log':
                y_transformed = DataPreProcessor.apply_log_transform(y_transformed,
                    base=transform_params.get('base', 10),
                    offset=transform_params.get('offset', 1.0))
            elif transform_mode == 'sqrt':
                y_transformed = DataPreProcessor.apply_sqrt_transform(y_transformed,
                    offset=transform_params.get('offset', 0.0))
            
            processed_group_data.append({
                'x': x_data,
                'y_raw_processed': y_transformed, # 已应用全局和独立动态范围压缩，但未缩放、未偏移、未求导、未应用整体Y轴偏移
                'ind_scale': ind_params['scale'],
                'ind_offset': ind_params['offset'],
                'label': label, # 使用重命名后的标签
                'file_path': file_path,  # 添加文件路径，用于获取颜色
                'base_name': base_name  # 添加基础名称，用于获取颜色
            })
            
        if not processed_group_data and not control_data_list:
            ax.text(0.5, 0.5, "No valid data (Check QC threshold / X-range)", transform=ax.transAxes, ha='center')
            self.canvas.draw()
            return

        # ==========================================
        # C. 根据模式绘图
        # ==========================================
        current_plot_index = len(control_data_list) # 接着对照组的索引

        # 模式 1: Mean + Shadow (平均值+阴影)
        if plot_mode == 'Mean + Shadow' and processed_group_data:
            common_x = processed_group_data[0]['x']
            all_y = []
            for item in processed_group_data:
                y_scaled = item['y_raw_processed'] * item['ind_scale']
                all_y.append(y_scaled)
            
            all_y = np.array(all_y)
            mean_y = np.mean(all_y, axis=0)
            std_y = np.std(all_y, axis=0)
            
            mean_y *= global_scale_factor
            std_y *= global_scale_factor
            
            # 二次导数（如果启用）- 在全局动态变换之后
            if is_derivative:
                d1 = np.gradient(mean_y, common_x)
                mean_y = np.gradient(d1, common_x)
                std_y = None  # 二次导数模式下不显示标准差阴影
            
            # 整体Y轴偏移（预处理最后一步，在二次导数之后）
            mean_y = mean_y + global_y_offset
            
            # 应用堆叠偏移（Mean + Shadow 模式也应该支持堆叠）
            mean_y = mean_y + (current_plot_index * global_stack_offset)
            
            color = custom_colors[current_plot_index % len(custom_colors)]
            
            # 使用重命名后的图例名称（如果有）
            rename_map = plot_params.get('legend_names', {})
            base_name = self.group_name
            
            # 检查是否有基础组名重命名
            if base_name in rename_map and rename_map[base_name]:
                base_display_name = rename_map[base_name]
            else:
                base_display_name = base_name
            
            # 检查是否有完整的图例名称重命名
            mean_label_key = f"{base_name} Mean"
            std_label_key = f"{base_name} Std Dev"
            
            if mean_label_key in rename_map and rename_map[mean_label_key]:
                mean_label = rename_map[mean_label_key]
            else:
                mean_label = f"{base_display_name} Mean"
            
            if std_label_key in rename_map and rename_map[std_label_key]:
                std_label = rename_map[std_label_key]
            else:
                std_label = f"{base_display_name} Std Dev"
            
            # 获取该组的颜色（从individual_y_params中获取，如果没有则使用custom_colors）
            group_color_params = plot_params.get('group_colors', {})
            if self.group_name in group_color_params:
                color = group_color_params[self.group_name]
            else:
                # 使用custom_colors，确保颜色与图例一致
                color = custom_colors[current_plot_index % len(custom_colors)]
            
            # 绘制
            if is_derivative:
                ax.plot(common_x, mean_y, color=color, linewidth=line_width, label=mean_label)
            else:
                ax.plot(common_x, mean_y, color=color, linewidth=line_width, label=mean_label)
                # 检查是否显示阴影（从样式配置获取）
                show_shadow = plot_params.get('show_shadow', True)
                if show_shadow and std_y is not None:
                    # 阴影颜色与线条颜色一致，确保图例颜色也一致
                    ax.fill_between(common_x, mean_y - std_y, mean_y + std_y, color=color, alpha=shadow_alpha, label=std_label)
            
            # 存储均值数据 (用于可能的叠加拟合)
            self.current_plot_data[self.group_name + "_Mean"] = {'x': common_x, 'y': mean_y, 'label': f"{self.group_name} Mean", 'color': color, 'type': 'Mean'}
            
            # 峰值检测（Mean + Shadow模式）
            if plot_params.get('peak_detection_enabled', False) and not is_derivative:
                # 对于Mean模式，在均值线上检测峰值
                self.detect_and_plot_peaks(ax, common_x, mean_y, mean_y, plot_params, color=color)
            
            # 更新Y值范围
            if is_derivative:
                max_y_value = max(max_y_value, np.max(mean_y))
                min_y_value = min(min_y_value, np.min(mean_y))
            else:
                max_y_value = max(max_y_value, np.max(mean_y + std_y))
                min_y_value = min(min_y_value, np.min(mean_y - std_y))


        # 模式 2/3: Waterfall / Normal Overlay
        else:
            for i, item in enumerate(processed_group_data):
                y_val = item['y_raw_processed'] * global_scale_factor * item['ind_scale']
                
                # 二次导数（如果启用）- 在全局动态变换之后
                if is_derivative:
                    d1 = np.gradient(y_val, item['x'])
                    y_val = np.gradient(d1, item['x'])
                
                # 整体Y轴偏移（预处理最后一步，在二次导数之后）
                y_val = y_val + global_y_offset
                
                stack_idx = i + current_plot_index
                y_final = y_val + item['ind_offset'] + (stack_idx * global_stack_offset)
                
                # 优先使用individual_y_params中指定的颜色
                base_name = item.get('base_name', os.path.splitext(os.path.basename(item.get('file_path', '')))[0] if 'file_path' in item else item.get('label', ''))
                
                # 从plot_params中获取文件颜色映射（在run_plot_logic中构建）
                file_colors = plot_params.get('file_colors', {})
                if base_name in file_colors:
                    color = file_colors[base_name]
                else:
                    color = custom_colors[stack_idx % len(custom_colors)]
                
                # **使用用户定义的 Line Style**
                if plot_style == 'line':
                    ax.plot(item['x'], y_final, label=item['label'], color=color, linewidth=line_width, linestyle=line_style)
                else: # scatter
                    ax.plot(item['x'], y_final, label=item['label'], color=color, marker='.', linestyle='', markersize=line_width*3)

                # 瀑布图模式下添加尾部标签
                if plot_mode == 'Waterfall (Stacked)':
                    ax.text(item['x'][0], y_final[0], item['label'], fontsize=legend_fontsize-1, va='center', color=color)

                # 峰值检测（使用通用函数）
                if plot_params.get('peak_detection_enabled', False) and not is_derivative:
                    y_detect = y_val # 检测用的 Y (去除偏移)
                    self.detect_and_plot_peaks(ax, item['x'], y_detect, y_final, plot_params, color)
                    
                # 存储数据以备叠加
                self.current_plot_data[item['label']] = {'x': item['x'], 'y': y_final, 'label': item['label'], 'color': color, 'type': 'Individual'}
                
                # 更新Y值范围
                max_y_value = max(max_y_value, np.max(y_final))
                min_y_value = min(min_y_value, np.min(y_final))


        # --- 6. 坐标轴设置 ---
        if x_axis_invert:
            ax.invert_xaxis()
            
        # --- Aspect Ratio 修正 3 ---
        aspect_ratio = plot_params.get('aspect_ratio', 0.0)
        if aspect_ratio > 0:
            ax.set_box_aspect(aspect_ratio) 
        else:
            ax.set_aspect('auto')
        # ---------------------------

        # 坐标轴范围设置（与数据处理.py保持一致，不固定范围）
        if is_zoomed:
            # 用户手动缩放过，恢复之前的范围
            ax.set_xlim(current_xlim) 
            ax.set_ylim(current_ylim)
        else:
            # 自动设置Y轴范围（与数据处理.py保持一致）
            if max_y_value != -np.inf and min_y_value != np.inf:
                y_range = max_y_value - min_y_value
                # 自动设置范围，并留出 5% 边距
                new_ylim = (min_y_value - y_range * 0.05, max_y_value + y_range * 0.05)
                ax.set_ylim(new_ylim[0], new_ylim[1])
            
            # 保存默认范围
            self.canvas.default_xlim = ax.get_xlim()
            self.canvas.default_ylim = ax.get_ylim()

        # 垂直线（使用可自定义的样式）
        vertical_lines = plot_params.get('vertical_lines', [])
        vertical_line_color = plot_params.get('vertical_line_color', 'gray')
        vertical_line_width = plot_params.get('vertical_line_width', 0.8)
        vertical_line_style = plot_params.get('vertical_line_style', ':')
        vertical_line_alpha = plot_params.get('vertical_line_alpha', 0.7)
        
        for line_x in vertical_lines:
            ax.axvline(x=line_x, color=vertical_line_color, linestyle=vertical_line_style, 
                      linewidth=vertical_line_width, alpha=vertical_line_alpha)

        # 标签
        ylabel_final = "2nd Derivative" if is_derivative else plot_params['ylabel_text']
        # 注意：BE校正后仍然使用样式配置中的Y轴标题，不强制修改
        # 如果需要显示BE校正信息，可以在标题或图例中说明

        # 使用GUI中的X轴标题控制参数
        xlabel_fontsize = plot_params.get('xlabel_fontsize', axis_title_fontsize)
        xlabel_pad = plot_params.get('xlabel_pad', 10.0)
        xlabel_show = plot_params.get('xlabel_show', True)
        
        if xlabel_show:
            ax.set_xlabel(plot_params['xlabel_text'], fontsize=xlabel_fontsize, labelpad=xlabel_pad, fontfamily=current_font)
        
        # 使用GUI中的Y轴标题控制参数
        ylabel_fontsize = plot_params.get('ylabel_fontsize', axis_title_fontsize)
        ylabel_pad = plot_params.get('ylabel_pad', 10.0)
        ylabel_show = plot_params.get('ylabel_show', True)
        
        if ylabel_show:
            ax.set_ylabel(ylabel_final, fontsize=ylabel_fontsize, labelpad=ylabel_pad, fontfamily=current_font)
        
        # 是否隐藏 Y 轴数值
        if not show_y_values:
            ax.set_yticks([])
        
        # Ticks 样式
        tick_direction = plot_params['tick_direction']
        tick_len_major = plot_params['tick_len_major']
        tick_len_minor = plot_params['tick_len_minor']
        tick_width = plot_params['tick_width']
        
        ax.tick_params(labelsize=tick_label_fontsize, direction=tick_direction, width=tick_width)
        ax.tick_params(which='major', length=tick_len_major)
        ax.tick_params(which='minor', length=tick_len_minor)
        
        # 边框设置 (Spines)
        for side in ['top', 'right', 'left', 'bottom']:
            if side in plot_params['border_sides']:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(plot_params['border_linewidth'])
            else:
                ax.spines[side].set_visible(False)
                
        # 网格
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
            
        # 图例 - 使用完整的图例控制参数
        if show_legend and plot_mode != 'Waterfall (Stacked)':
            # 设置图例字体
            from matplotlib.font_manager import FontProperties
            legend_font = FontProperties()
            if font_family != 'SimHei':
                legend_font.set_family(font_family)
            else:
                legend_font.set_family('sans-serif')
            legend_font.set_size(legend_fontsize)
            
            ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frame, prop=legend_font,
                     ncol=legend_ncol, columnspacing=legend_columnspacing, 
                     labelspacing=legend_labelspacing, handlelength=legend_handlelength)
            
        # --- 7. 设置标题（在所有绘图完成后）---
        # -------------------------------------------------------------------
        # 修正 2: 统一设置主标题逻辑（在绘图完成后设置，避免被覆盖）
        # 修复：如果用户输入空格，则不显示标题（即使有group_name）
        # -------------------------------------------------------------------
        main_title_stripped = main_title_text.strip()
        # 只有当用户明确输入了非空标题时才显示，否则不显示标题
        # 使用GUI中的标题控制参数
        main_title_fontsize = plot_params.get('main_title_fontsize', axis_title_fontsize)
        main_title_pad = plot_params.get('main_title_pad', 10.0)
        main_title_show = plot_params.get('main_title_show', True)
        
        if main_title_stripped != "" and main_title_show:
            final_title = main_title_stripped
            ax.set_title(
                final_title, 
                fontsize=main_title_fontsize, 
                fontfamily=current_font,
                pad=main_title_pad
            )
        # 如果用户输入空格或留空，则不设置标题（不显示group_name）
        # -------------------------------------------------------------------
        
        # --- 8. 最终布局和渲染（与数据处理.py保持一致）---
        
        # 使用tight_layout自动调整布局（与数据处理.py一致）
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
            self.canvas.figure.tight_layout()
        
        # 最终渲染
        self.canvas.draw()
        
        # 确保窗口可见
        if not self.isVisible():
            self.show()
        
        self.update() # 强制 Qt 窗口刷新


# -----------------------------------------------------------------
# 🚀 【GUI 配置与运行部分 - 基于 PyQt6】
# -----------------------------------------------------------------


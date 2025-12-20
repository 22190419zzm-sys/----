# src/core 概览与快速使用

本目录包含与 GUI 解耦的核心算法与工具，适合脚本化/批量场景直接复用。

## 模块与类

- `preprocessor.py`  
  - `DataPreProcessor`: 平滑、AsLS 基线校正、归一化（max/area/SNV）、对数/平方根变换、Bose-Einstein 校正、SVD 去噪。
- `transformers.py`  
  - `NonNegativeTransformer`: 将负值截断为 0。  
  - `AutoencoderTransformer`: 深度自编码器（PyTorch，可回退 sklearn MLP）。  
  - `AdaptiveMineralFilter`: 基于 PCA 的鲁棒背景抑制（矿物/有机分离）。  
  - 以上模型在导入时会注册到 `registry.py`，便于插件式扩展。
- `generators.py`  
  - `SyntheticDataGenerator`: 加载纯组分并生成混合/增强光谱（噪声、基线漂移、峰抑制、偏移/拉伸）。
- `matcher.py`  
  - `SpectralMatcher`: 余弦相似度匹配查询谱与标准库。
- `registry.py`  
  - 轻量注册表，支持动态注册预处理/模型/绘图风格：`register_*` / `get_*`。

## 最小示例：预处理 + 自编码器（回退 sklearn）

```python
import numpy as np
from src.core.preprocessor import DataPreProcessor
from src.core.transformers import AutoencoderTransformer

X = np.random.rand(50, 200).astype(np.float32)

# 预处理：AsLS + SNV
baseline = DataPreProcessor.apply_baseline_als(X[0], lam=1e4, p=0.01)
X_snv = DataPreProcessor.apply_normalization(X, norm_mode='snv')

# 自编码器（若无 torch 自动回退 sklearn）
ae = AutoencoderTransformer(n_components=6, use_deep=False, max_iter=500)
ae.fit(X_snv)
H = ae.transform(X_snv)
X_rec = ae.inverse_transform(H)
```

## 使用注册表做插件式扩展

```python
from src.core.registry import register_preprocessor, get_preprocessors

def my_clip(y, min_v=0):
    return (y - min_v).clip(min=0)

register_preprocessor("clip", my_clip)
print(get_preprocessors().keys())  # 包含 clip 和内置步骤
```

## 批量生成/匹配光谱

```python
from src.core.generators import SyntheticDataGenerator
from src.core.matcher import SpectralMatcher
import numpy as np

wavenumbers = np.linspace(4000, 400, 1200)
gen = SyntheticDataGenerator(wavenumbers)
gen.load_pure_spectrum("纯组分文件夹/50-2.txt", name="mineral")
gen.load_pure_spectrum("纯组分文件夹/s-water-1.txt", name="water")
X_syn, ratios = gen.generate_batch(
    n_samples=5,
    ratio_ranges={"mineral": (0.5, 0.9), "water": (0.1, 0.5)},
    noise_level=0.01,
    baseline_drift=0.05,
    complexity=0.7,
)

matcher = SpectralMatcher("纯组分文件夹")
matches = matcher.match(wavenumbers, X_syn[0], top_k=3)
```

## 运行提示

- 若需深度自编码器，请安装 `torch`；否则自动回退 sklearn MLP。
- 所有函数/类均为纯 Python，无 GUI 依赖，便于在服务器/CI 环境运行。


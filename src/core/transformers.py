import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

from .registry import register_model

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Deep Learning features will be disabled.")


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


# 将核心模型注册到插件式注册表，方便动态扩展/替换
register_model("autoencoder", AutoencoderTransformer)
register_model("nonnegative", NonNegativeTransformer)
register_model("adaptive_mineral_filter", AdaptiveMineralFilter)


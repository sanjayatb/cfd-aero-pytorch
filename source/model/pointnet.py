from typing import Dict, List

from source.config.dto import Config
from source.model.base import Model
from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from neuralop.models import FNO1d
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointNet(Model):
    def __init__(self, config):
        super(SimplePointNet, self).__init__(config)

        emb_dims = config.parameters.model.emb_dims
        dropout = config.parameters.model.dropout

        # **Dynamically create Conv1D layers**
        conv_layers = config.parameters.model.conv_layers  # List of channels e.g. [3, 256, 512, emb_dims]
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        print(conv_layers)
        for i in range(len(conv_layers) - 1):
            self.conv_layers.append(nn.Conv1d(conv_layers[i], conv_layers[i + 1], kernel_size=1, bias=False))
            self.bn_layers.append(nn.BatchNorm1d(conv_layers[i + 1]))

        # **Dynamically create Fully Connected layers**
        fc_layers = config.parameters.model.fc_layers  # List of channels e.g. [emb_dims, 512, 256, 1]
        print(fc_layers)
        self.fc_layers = nn.ModuleList()
        self.fc_bn_layers = nn.ModuleList()

        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)

        for i in range(len(fc_layers) - 1):
            self.fc_layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            if i < len(fc_layers) - 2:  # No BatchNorm on final layer
                if dataset_conf.target_col_alias == "Pressure":
                    self.fc_bn_layers.append(nn.LayerNorm(fc_layers[i + 1]))
                else:
                    self.fc_bn_layers.append(nn.BatchNorm1d(fc_layers[i + 1]))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)

        # (B, 3, N) → Conv layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.leaky_relu(bn(conv(x)))  # x: (B, C, N)

        if dataset_conf.target_col_alias == "Pressure":
            # → pointwise MLP
            x = x.transpose(1, 2)  # (B, N, C)

            for i, fc in enumerate(self.fc_layers):
                x = fc(x)  # (B, N, C')
                if i < len(self.fc_layers) - 1:
                    x = F.leaky_relu(self.fc_bn_layers[i](x))
                    x = self.dropout(x)

            return x  # (B, N, 1)

        else:
            # → global pooling + FC
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, C)
            for i, fc in enumerate(self.fc_layers):
                x = fc(x)
                if i < len(self.fc_layers) - 1:
                    x = F.leaky_relu(self.fc_bn_layers[i](x))
                    x = self.dropout(x)
            return x  # (B, 1)


class ShallowMLP(Model):
    def __init__(self, config):
        super(ShallowMLP, self).__init__(config)
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.mean(x, dim=-1)  # Global Average Pooling (B, 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PointNetFNO(Model):
    def __init__(self, config):
        super(PointNetFNO, self).__init__(config)

        emb_dims = config.parameters.model.emb_dims
        dropout = config.parameters.model.dropout

        # **Dynamically create Conv1D & FNO layers**
        conv_layers = config.parameters.model.conv_layers

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(len(conv_layers) - 1):
            if i == len(conv_layers) - 2:  # Use FNO instead of last Conv1D layer
                self.conv_layers.append(
                    FNO1d(
                        n_modes_width=16,  # Number of Fourier modes
                        n_modes_height=16,  # Still required even for 1D case
                        hidden_channels=conv_layers[i]  # Match hidden channels with Conv1D input
                    )
                )
                self.bn_layers.append(nn.BatchNorm1d(conv_layers[i]))  # BatchNorm for input size
            else:
                self.conv_layers.append(nn.Conv1d(conv_layers[i], conv_layers[i + 1], kernel_size=1, bias=False))
                self.bn_layers.append(nn.BatchNorm1d(conv_layers[i + 1]))

        # **Dynamically create Fully Connected layers**
        fc_layers = config.parameters.model.fc_layers
        self.fc_layers = nn.ModuleList()
        self.fc_bn_layers = nn.ModuleList()

        for i in range(len(fc_layers) - 1):
            self.fc_layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            if i < len(fc_layers) - 2:  # No BatchNorm on final layer
                self.fc_bn_layers.append(nn.BatchNorm1d(fc_layers[i + 1]))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # **Apply Conv1D & FNO layers**
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            if isinstance(conv, FNO1d):  # If FNO layer, reshape input
                x = x.permute(0, 2, 1)  # Convert to (B, N, C) for FNO
                x = conv(x)
                x = x.permute(0, 2, 1)  # Convert back to (B, C, N)
            else:
                x = F.leaky_relu(bn(conv(x)))  # Apply Conv1D

        # **Global Feature Pooling**
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)

        # **Apply Fully Connected layers**
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:
                x = F.leaky_relu(self.fc_bn_layers[i](x))
                x = self.dropout(x)

        return x  # (B, 1)


class RegPointNet(Model):
    """
    PointNet-based multi-task model: predicts point-wise pressure and global drag.
    """

    def __init__(self, config: Config):
        super(RegPointNet, self).__init__(config)

        emb_dims = config.parameters.model.emb_dims
        dropout = config.parameters.model.dropout

        # Shared feature extractor (backbone)
        self.conv1 = nn.Conv1d(3, 512, 1, bias=False)
        self.conv2 = nn.Conv1d(512, 1024, 1, bias=False)
        self.conv3 = nn.Conv1d(1024, 1024, 1, bias=False)
        self.conv4 = nn.Conv1d(1024, 1024, 1, bias=False)
        self.conv5 = nn.Conv1d(1024, 1024, 1, bias=False)
        self.conv6 = nn.Conv1d(1024, emb_dims, 1, bias=False)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(emb_dims)

        # Shortcut (residual)
        self.conv_shortcut = nn.Conv1d(3, emb_dims, 1, bias=False)
        self.bn_shortcut = nn.BatchNorm1d(emb_dims)

        self.dropout = nn.Dropout(p=dropout)

        # --------- Point-wise Pressure Head ---------
        self.pressure_head = nn.Sequential(
            nn.Conv1d(emb_dims, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 1, 1)  # (B, 1, N)
        )

        # --------- Global Drag Head (after pooling) ---------
        self.drag_head = nn.Sequential(
            nn.Linear(emb_dims, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)  # (B, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 3, N)
        Returns:
            pressure: (B, N)
            drag: (B, 1)
        """
        shortcut = self.bn_shortcut(self.conv_shortcut(x))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x + shortcut  # Residual connection

        # Point-wise prediction (no pooling)
        pressure = self.pressure_head(x).squeeze(1)  # (B, N)

        # Global prediction via pooled features
        pooled = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, emb_dims)
        drag = self.drag_head(pooled)  # (B, 1)

        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
        if dataset_conf.target_col_alias == "Pressure":
            return pressure

        return drag



class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        assert in_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.qkv = nn.Linear(in_dim, in_dim * 3)  # (batch, num_points, 3*in_dim)
        self.fc = nn.Linear(in_dim, in_dim)
        self.scale = self.head_dim ** -0.5  # Scale for stability

    def forward(self, x):
        batch_size, num_points, in_dim = x.shape  # (B, N, D)

        # Generate Q, K, V and split into heads
        qkv = self.qkv(x).reshape(batch_size, num_points, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each has shape (B, N, num_heads, head_dim)

        # Transpose for attention computation: (B, num_heads, N, head_dim)
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]

        window_size = min(q.shape[-2], 1024)

        q_blocks = F.unfold(q.permute(0, 3, 2, 1), kernel_size=(window_size, 1)).permute(0, 3, 2, 1)
        k_blocks = F.unfold(k.permute(0, 3, 2, 1), kernel_size=(window_size, 1)).permute(0, 3, 2, 1)
        v_blocks = F.unfold(v.permute(0, 3, 2, 1), kernel_size=(window_size, 1)).permute(0, 3, 2, 1)

        # Compute attention only within blocks
        attn_weights = torch.matmul(q_blocks, k_blocks.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_blocks)

        # Merge heads back: (B, N, D)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, num_points, in_dim)

        return self.fc(attn_output) + x  # Residual connection


class PointNetWithSelfAttention(Model):
    def __init__(self, config):
        super(PointNetWithSelfAttention, self).__init__(config)

        emb_dims = config.parameters.model.emb_dims
        dropout = config.parameters.model.dropout

        # Convolutional layers
        self.conv1 = nn.Conv1d(3, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(1024, emb_dims, kernel_size=1, bias=False)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(emb_dims)

        # Dropout layers
        self.dropout = nn.Dropout(p=dropout)

        # Residual shortcut
        self.conv_shortcut = nn.Conv1d(3, emb_dims, kernel_size=1, bias=False)
        self.bn_shortcut = nn.BatchNorm1d(emb_dims)

        # Self-Attention
        self.self_attention = SelfAttention(emb_dims)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.final_linear = nn.Linear(64, 1)

    def forward(self, x):
        shortcut = self.bn_shortcut(self.conv_shortcut(x))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        # Adding residual connection
        x = x + shortcut

        # Apply Self-Attention
        x = self.self_attention(x.permute(0, 2, 1))  # (batch, num_points, emb_dims)
        x = x.permute(0, 2, 1)  # Back to (batch, emb_dims, num_points)

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # Global pooling

        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = F.relu(self.bn7(self.fc3(x)))
        x = F.relu(self.bn8(self.fc4(x)))
        x = self.final_linear(x)

        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for point clouds"""

    def __init__(self, in_dim):
        super(PositionalEncoding, self).__init__()
        self.linear = nn.Linear(
            3, in_dim
        )  # Convert (x, y, z) coordinates to embedding space

    def forward(self, x):
        """
        x.shape: (batch_size, num_points, 3)
        Output: (batch_size, num_points, in_dim)
        """
        return self.linear(x)


class PointNetWithTorchTransformer(Model):
    def __init__(self, config):
        super(PointNetWithTorchTransformer, self).__init__(config)

        emb_dims = config.parameters.model.emb_dims
        dropout = config.parameters.model.dropout
        num_heads = 8
        num_layers = 4

        # Positional Encoding (before any transformations)
        self.pos_encoding = PositionalEncoding(emb_dims)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=emb_dims,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            activation="relu",
            batch_first=True,  # Ensures batch dimension is first
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected regression layers
        self.fc1 = nn.Linear(emb_dims, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.final_linear = nn.Linear(64, 1)

    def forward(self, x):
        """
        x.shape = (batch_size, 3, num_points)
        Transformer expects (batch_size, num_points, emb_dims)
        """

        batch_size, _, num_points = x.shape  # x.shape: (batch, 3, num_points)

        # **Fix the shape before positional encoding**
        x = x.permute(0, 2, 1)  # Change to (batch_size, num_points, 3)

        # Apply Positional Encoding
        x = self.pos_encoding(x)  # (batch_size, num_points, emb_dims)

        # Pass through Transformer Encoder
        x = self.transformer(x)  # Shape remains (batch_size, num_points, emb_dims)

        # Pooling to reduce dimensionality
        x = x.mean(dim=1)  # (batch_size, emb_dims)

        # Fully Connected Regression Layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = self.final_linear(x)  # Output shape: (batch_size, 1)

        return x


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock1D, self).__init__()

        # 1x1 Convolution (PointNet-style)
        self.conv1x1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels // 4)

        # 3x1 Convolution
        self.conv3x1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels // 4)

        # 5x1 Convolution
        self.conv5x1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels // 4)

        # Max Pooling branch
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm1d(out_channels // 4)

    def forward(self, x):
        branch1 = F.relu(self.bn1(self.conv1x1(x)))
        branch2 = F.relu(self.bn2(self.conv3x1(x)))
        branch3 = F.relu(self.bn3(self.conv5x1(x)))
        branch4 = F.relu(self.bn4(self.conv_pool(x)))

        # Concatenate along the channel dimension
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionPointNet(Model):
    def __init__(self, config):
        super(InceptionPointNet, self).__init__(config)

        emb_dims = config.parameters.model.emb_dims
        dropout = config.parameters.model.dropout

        # Inception Blocks for Feature Extraction
        self.inc1 = InceptionBlock1D(3, 512)
        self.inc2 = InceptionBlock1D(512, 1024)
        self.inc3 = InceptionBlock1D(1024, 1024)
        self.inc4 = InceptionBlock1D(1024, emb_dims)

        # Batch Normalization for Residual Connection
        self.shortcut = nn.Conv1d(3, emb_dims, kernel_size=1, bias=False)
        self.bn_shortcut = nn.BatchNorm1d(emb_dims)

        # Fully Connected Layers
        self.fc1 = nn.Linear(emb_dims, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.final_fc = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        shortcut = self.bn_shortcut(self.shortcut(x))  # Residual Connection

        x = self.inc1(x)
        x = self.inc2(x)
        x = self.inc3(x)
        x = self.inc4(x)

        x = x + shortcut  # Apply residual connection

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.final_fc(x)

        return x


class PointTransformerV3Regressor(Model):
    """Wrapper around the PointTransformerV3 backbone for CFD regression tasks."""

    def __init__(self, config: Config):
        super().__init__(config)
        pt_cfg = dict(getattr(config.parameters.model, "point_transformer", {}) or {})

        backbone_cls, offset2bincount = self._load_point_transformer()
        self._offset2bincount = offset2bincount

        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
        self.target_alias = dataset_conf.target_col_alias
        self.num_points = config.parameters.data.num_points
        self.grid_size = float(pt_cfg.get("grid_size", 0.01))
        self.use_all_features = bool(pt_cfg.get("use_all_features", False))

        backbone_kwargs = self._build_backbone_kwargs(pt_cfg)
        self.in_channels = backbone_kwargs["in_channels"]
        self.cls_mode = backbone_kwargs.get("cls_mode", False)
        self.out_channels = (
            backbone_kwargs["enc_channels"][-1]
            if self.cls_mode
            else backbone_kwargs["dec_channels"][0]
        )

        try:
            self.backbone = backbone_cls(**backbone_kwargs)
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Failed to initialise PointTransformerV3. Install optional "
                "dependencies such as spconv (CUDA build) and torch_scatter "
                "before using this model."
            ) from exc

        head_cfg = pt_cfg.get("head", {})
        drag_cfg = head_cfg.get("drag", {})
        pressure_cfg = head_cfg.get("pressure", {})

        self.drag_pooling = drag_cfg.get("pooling", "mean").lower()
        drag_hidden = drag_cfg.get("hidden_dims", [256, 128])
        drag_dropout = drag_cfg.get("dropout", config.parameters.model.dropout)
        self.drag_head = self._build_mlp(
            [self.out_channels, *drag_hidden, 1], drag_dropout
        )

        if self.target_alias == "Pressure":
            pressure_hidden = pressure_cfg.get("hidden_dims", [128])
            pressure_dropout = pressure_cfg.get(
                "dropout", config.parameters.model.dropout
            )
            self.pressure_head = self._build_mlp(
                [self.out_channels, *pressure_hidden, 1], pressure_dropout
            )
        else:
            self.pressure_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cls_mode:
            raise NotImplementedError(
                "PointTransformerV3Regressor currently expects cls_mode=False"
            )

        batch_size, _, num_points = x.shape
        feats = x.transpose(1, 2).contiguous()

        coords = feats[..., :3]
        if self.use_all_features:
            feat = feats[..., : self.in_channels]
        else:
            feat = coords

        if feat.shape[-1] < self.in_channels:
            pad = feat.new_zeros((batch_size, num_points, self.in_channels - feat.shape[-1]))
            feat = torch.cat([feat, pad], dim=-1)
        elif feat.shape[-1] > self.in_channels:
            feat = feat[..., : self.in_channels]

        batch_index = (
            torch.arange(batch_size, device=x.device, dtype=torch.long)
            .repeat_interleave(num_points)
        )

        point_dict = {
            "coord": coords.reshape(-1, coords.shape[-1]),
            "feat": feat.reshape(-1, self.in_channels),
            "batch": batch_index,
            "grid_size": self.grid_size,
        }

        point = self.backbone(point_dict)
        feat = point.feat

        if hasattr(point, "offset") and point.offset is not None:
            counts = self._offset2bincount(point.offset)
        else:  # pragma: no cover - fallback for unexpected inputs
            counts = torch.full(
                (batch_size,),
                num_points,
                device=x.device,
                dtype=torch.long,
            )

        features = self._restore_batch(feat, counts, batch_size, num_points)

        if self.target_alias == "Pressure":
            return self._forward_pressure(features)

        pooled = self._pool_features(features)
        return self.drag_head(pooled).squeeze(-1)

    def _restore_batch(
        self,
        feat: torch.Tensor,
        counts: torch.Tensor,
        batch_size: int,
        default_points: int,
    ) -> torch.Tensor:
        counts_list = counts.detach().cpu().tolist()
        chunks = torch.split(feat, counts_list)
        target_points = self.num_points or max(counts_list + [default_points])

        restored: List[torch.Tensor] = []
        for chunk in chunks:
            if chunk.shape[0] < target_points:
                pad = chunk.new_zeros((target_points - chunk.shape[0], chunk.shape[1]))
                chunk = torch.cat([chunk, pad], dim=0)
            elif chunk.shape[0] > target_points:
                chunk = chunk[:target_points]
            restored.append(chunk)

        if len(restored) != batch_size:
            raise RuntimeError(
                f"Recovered {len(restored)} samples, expected {batch_size}."
            )
        return torch.stack(restored, dim=0)

    def _forward_pressure(self, features: torch.Tensor) -> torch.Tensor:
        if self.pressure_head is None:
            raise RuntimeError("Pressure head is not initialised for this dataset")
        batch_size, num_points, channels = features.shape
        outputs = self.pressure_head(features.view(batch_size * num_points, channels))
        return outputs.view(batch_size, num_points, -1)

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.drag_pooling == "max":
            return features.max(dim=1).values
        return features.mean(dim=1)

    @staticmethod
    def _build_mlp(channels: List[int], dropout: float) -> nn.Sequential:
        layers: List[nn.Module] = []
        for idx in range(len(channels) - 1):
            layers.append(nn.Linear(channels[idx], channels[idx + 1]))
            if idx < len(channels) - 2:
                layers.append(nn.GELU())
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _build_backbone_kwargs(self, cfg: Dict) -> Dict:
        order = cfg.get("order", ("z", "z-trans", "hilbert", "hilbert-trans"))
        if isinstance(order, str):
            order = (order,)
        else:
            order = tuple(order)

        def _tuple_key(key, default):
            value = cfg.get(key, default)
            return tuple(value) if isinstance(value, (list, tuple)) else tuple(default)

        kwargs = {
            "in_channels": int(cfg.get("in_channels", 3)),
            "order": order,
            "stride": _tuple_key("stride", (2, 2, 2, 2)),
            "enc_depths": _tuple_key("enc_depths", (2, 2, 2, 6, 2)),
            "enc_channels": _tuple_key("enc_channels", (32, 64, 128, 256, 512)),
            "enc_num_head": _tuple_key("enc_num_head", (2, 4, 8, 16, 32)),
            "enc_patch_size": _tuple_key(
                "enc_patch_size", (1024, 1024, 1024, 1024, 1024)
            ),
            "dec_depths": _tuple_key("dec_depths", (2, 2, 2, 2)),
            "dec_channels": _tuple_key("dec_channels", (64, 64, 128, 256)),
            "dec_num_head": _tuple_key("dec_num_head", (4, 4, 8, 16)),
            "dec_patch_size": _tuple_key("dec_patch_size", (1024, 1024, 1024, 1024)),
            "mlp_ratio": cfg.get("mlp_ratio", 4),
            "qkv_bias": cfg.get("qkv_bias", True),
            "qk_scale": cfg.get("qk_scale"),
            "attn_drop": cfg.get("attn_drop", 0.0),
            "proj_drop": cfg.get("proj_drop", 0.0),
            "drop_path": cfg.get("drop_path", 0.1),
            "pre_norm": cfg.get("pre_norm", True),
            "shuffle_orders": cfg.get("shuffle_orders", True),
            "enable_rpe": cfg.get("enable_rpe", False),
            "enable_flash": cfg.get("enable_flash", False),
            "upcast_attention": cfg.get("upcast_attention", False),
            "upcast_softmax": cfg.get("upcast_softmax", False),
            "cls_mode": cfg.get("cls_mode", False),
            "pdnorm_bn": cfg.get("pdnorm_bn", False),
            "pdnorm_ln": cfg.get("pdnorm_ln", False),
            "pdnorm_decouple": cfg.get("pdnorm_decouple", True),
            "pdnorm_adaptive": cfg.get("pdnorm_adaptive", False),
            "pdnorm_affine": cfg.get("pdnorm_affine", True),
            "pdnorm_conditions": tuple(
                cfg.get(
                    "pdnorm_conditions",
                    ("ScanNet", "S3DIS", "Structured3D"),
                )
            ),
        }
        return kwargs

    @staticmethod
    def _load_point_transformer():
        try:
            from source.model.point_transformer_v3 import PointTransformerV3
            from source.model.point_transformer_v3.model import offset2bincount
        except ModuleNotFoundError as exc:  # pragma: no cover - informative failure
            missing = exc.name or "unknown"
            raise RuntimeError(
                "PointTransformerV3 dependencies are missing. "
                f"Install the optional package '{missing}' (and related CUDA/CPU builds) "
                "before selecting PointTransformerV3Regressor."
            ) from exc
        except Exception as exc:  # pragma: no cover - informative failure
            raise RuntimeError(
                "PointTransformerV3 dependencies could not be imported. "
                "Verify that timm, torch-scatter, spconv, addict, and flash-attn (optional) "
                "are installed and compatible with your current platform."
            ) from exc
        return PointTransformerV3, offset2bincount

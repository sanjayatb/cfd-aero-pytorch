from source.config.dto import Config
from source.model.base import Model
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from neuralop.models import FNO1d
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

        for i in range(len(fc_layers) - 1):
            self.fc_layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            if i < len(fc_layers) - 2:  # No BatchNorm on final layer
                self.fc_bn_layers.append(nn.BatchNorm1d(fc_layers[i + 1]))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # **Apply dynamically created Conv1D layers**
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.leaky_relu(bn(conv(x)))  # (B, Channels, N)

        # **Global Feature Pooling**
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, emb_dims)

        # **Apply dynamically created Fully Connected layers**
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:  # No activation on final layer
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
    PointNet-based regression model for 3D point cloud data.

    Args:
        args (dict): Configuration parameters including 'emb_dims' for embedding dimensions and 'dropout' rate.

    Methods:
        forward(x): Forward pass through the network.
    """

    def __init__(self, config: Config):
        """
        Initialize the RegPointNet model for regression tasks with enhanced complexity,
        including additional layers and residual connections.

        Parameters:
            emb_dims (int): Dimensionality of the embedding space.
            dropout (float): Dropout probability.
        """
        super(RegPointNet, self).__init__(config)

        # Convolutional layers
        self.conv1 = nn.Conv1d(3, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
        self.conv6 = nn.Conv1d(
            1024, config.parameters.model.emb_dims, kernel_size=1, bias=False
        )

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(config.parameters.model.emb_dims)

        # Dropout layers
        self.dropout_conv = nn.Dropout(p=config.parameters.model.dropout)
        self.dropout_linear = nn.Dropout(p=config.parameters.model.dropout)

        # Residual connection layer
        self.conv_shortcut = nn.Conv1d(
            3, config.parameters.model.emb_dims, kernel_size=1, bias=False
        )
        self.bn_shortcut = nn.BatchNorm1d(config.parameters.model.emb_dims)

        # Linear layers for regression output
        self.linear1 = nn.Linear(config.parameters.model.emb_dims, 512, bias=False)
        self.bn7 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn8 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)  # Output one scalar value
        self.bn9 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128, 64)  # Output one scalar value
        self.bn10 = nn.BatchNorm1d(64)
        self.final_linear = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, 3, num_points).

        Returns:
            Tensor: Output tensor of the predicted scalar value.
        """
        shortcut = self.bn_shortcut(self.conv_shortcut(x))

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn6(self.conv6(x)))
        # Adding the residual connection
        x = x + shortcut

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.bn7(self.linear1(x)))
        x = F.relu(self.bn8(self.linear2(x)))
        x = F.relu(self.bn9(self.linear3(x)))
        x = F.relu(self.bn10(self.linear4(x)))
        features = x
        x = self.final_linear(x)

        # return x, features
        return x


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

import torch.nn as nn
import torch.nn.functional as F
import torch

from source.config.dto import Config
from source.model.base import Model
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.scale = 1.0 / (in_dim ** 0.5)
        self.fc = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.fc(out) + x  # Residual connection


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


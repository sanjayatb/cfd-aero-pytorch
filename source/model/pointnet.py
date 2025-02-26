import torch.nn as nn
import torch.nn.functional as F

from source.config.dto import Config
from source.model.base import Model


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
        self.conv6 = nn.Conv1d(1024, config.parameters.model.emb_dims, kernel_size=1, bias=False)

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
        self.conv_shortcut = nn.Conv1d(3, config.parameters.model.emb_dims, kernel_size=1, bias=False)
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

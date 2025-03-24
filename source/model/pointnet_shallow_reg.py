from source.model.base import Model
import torch.nn as nn
import torch.nn.functional as F


class PointNetShallowReg(Model):
    def __init__(self, config):
        super(PointNetShallowReg, self).__init__(config)

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

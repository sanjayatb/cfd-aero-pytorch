from torch import nn

from source.config.dto import Config


class Model(nn.Module):
    config: Config

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

    def forward(self, x):
        pass

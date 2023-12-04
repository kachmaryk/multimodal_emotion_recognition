import os
import sys
import requests
from requests.adapters import HTTPAdapter

import torch
from torch import nn
from torch.nn import functional as F


class MultiModalFastAi:
    """

    """
    def __init__(self,
                 num_classes=None,
                 device=None):
        # Set simple attributes
        self.num_classes = num_classes

        self.out_fast_ai_head = nn.Sequential(
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(in_features=1024, out_features=512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5, inplace=False),

            nn.Linear(in_features=512, out_features=self.num_classes, bias=False)
        )

        self.device = torch.device('cpu')

        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        x = self.out_fast_ai_head(x)

        return x

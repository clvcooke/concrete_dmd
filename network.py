import torch
import torch.nn as nn
from dmd import DMDSpatial


class DigitNet(nn.Module):
    def __init__(self, input_size=784, dmd_count=1, temperature=1):
        super(DigitNet, self).__init__()
        self.input_size = input_size
        self.dmd_count = dmd_count
        self.simple_mlp = nn.Sequential(
            nn.Linear(self.dmd_count, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=-1)
        )
        self.dmds = [DMDSpatial(input_size, 1, temperature) for _ in range(self.dmd_count)]

    def forward(self, x):
        # sensed is (B, self.dmd_count)
        sensed = torch.stack([dmd(x) for dmd in self.dmds], dim=1)
        classified = self.simple_mlp(sensed)
        return classified

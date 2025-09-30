import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_pre_activation: bool = False) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_pre_activation = is_pre_activation
        stride = 1 if self.in_channels == self.out_channels else 2
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=stride, padding=1
        )        
        self.bn1 = nn.BatchNorm2d(num_features=self.in_channels if is_pre_activation else self.out_channels)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=self.out_channels)
        self.downsample = (
            None
            if self.in_channels == self.out_channels
            else nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._pre_activation(x) if self.is_pre_activation else self._post_activation(x)

    def _post_activation(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        z = x if self.downsample is None else self.downsample(x)
        out += z
        out = F.relu(out)

        return out

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        z = x if self.downsample is None else self.downsample(x)
        out += z

        return out


if __name__ == "__main__":
    net = ResnetBlock(in_channels=3, out_channels=6, is_pre_activation=True)
    img = torch.rand((1, 3, 224, 224))
    out = net(img)

    print(out.shape)

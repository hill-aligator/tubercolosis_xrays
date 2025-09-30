import torch.nn as nn
import torch

from resnet_block import ResnetBlock

resnet_18_blocks = [(64, 2), (128, 2), (256, 2), (512, 2)]


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        block_params: list[tuple[int, int]] = resnet_18_blocks,
        is_pre_activation: bool = True,
        is_original_start: bool = False,
    ) -> None:
        super().__init__()

        self.start = self._create_start(in_channels, block_params[0][0], is_original_start)
        self.blocks = nn.ModuleList()

        for i, (channels, num_resnet_blocks) in enumerate(block_params):
            is_last = i == len(block_params) - 1
            self._create_block(channels, num_resnet_blocks, is_pre_activation, is_last)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=block_params[-1][0], out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.start(x)
        for block in self.blocks:
            out = block(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _create_start(self, in_channels: int, out_channels: int, is_original_start: bool) -> nn.Sequential:
        if is_original_start:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            half_out_channels = out_channels // 2

            return nn.Sequential(
                nn.Conv2d(3, half_out_channels, 3, stride=2, padding=1),  
                nn.BatchNorm2d(half_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_out_channels, half_out_channels, 3, stride=1, padding=1),  
                nn.BatchNorm2d(half_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_out_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def _create_block(self, channels: int, num_resnet_blocks: int, is_pre_activation: bool, is_last: bool) -> None:
        for i in range(num_resnet_blocks):
            out_channels = channels if i < num_resnet_blocks - 1 or is_last else channels * 2
            self.blocks.append(
                ResnetBlock(
                    in_channels=channels,
                    out_channels=out_channels,
                    is_pre_activation=is_pre_activation,
                )
            )


if __name__ == "__main__":
    resnet = ResNet(in_channels=3, num_classes=1000, is_original_start=True)
    img = torch.rand((1, 3, 224, 224))
    out = resnet(img)

    print(out.shape)

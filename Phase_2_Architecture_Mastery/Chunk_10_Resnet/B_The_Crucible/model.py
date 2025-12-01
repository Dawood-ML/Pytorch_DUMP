"""
Resnet (CIFAR-100)

KEY ARCHITECTURAL CHANGES FROM STANDARD IMAGENET RESNET:
1. First Conv: 3x3 kernel, stride 1, padding 1 (preserves 32x32 spatial dim).
   (Standard ResNet uses 7x7 stride 2, which is too aggressive for small images).
2. No MaxPool after first conv.
3. Three stages of residual blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    The building block of ResNet-18 and ResNet-34 structure
    
    Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
                |                                       ^
                |_______________________________________|
                              Skip Connection
    """
    expansion = 1 # Expansion factor for channel depth (used in deeper ResNets like ResNet-50)

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # THe Main Path (F(x))
        # 1. First conv (potentiallyy strided for downsampleing)
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=planes, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # The shortcut path (x)
        # If input shape != output shape (due to stride or channel change),
        # we need a 1x1 conv to align them.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=self.expansion * planes,
                    kernel_size=1, stride=stride,
                    bias=True
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            # 2. Residual connection
            # "The Magic Addition": Adding the original input (processed by shortcut)
            out += self.shortcut(x)

            # 3. Final Activation
            out = F.relu()
            return out



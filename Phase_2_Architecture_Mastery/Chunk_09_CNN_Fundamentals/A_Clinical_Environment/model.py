import torch
import torch.nn as nn

class SimpleVGG(nn.Module):
    """
    A simple VGG-style CNN for Cifar 10
    """
    def __init__(self, num_classes=10):
        super(SimpleVGG, self).__init__()

        # BLock 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        # Block 2
        self.block2 =  nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Block 3
        self.block3 =  nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        # Classifier Head
        # After 3 max-pooling layers of stride 2, the 32x32 image becomes 4x4
        # so the flattened size is 128 (channels) * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.classifier(self.block3(self.block2(self.block1(x))))
        return x
    
if __name__ == '__main__':
    # Test the model with a dummy input
    dummy_input = torch.randn(64, 3, 32, 32) # (batch_size, channels, height, width)
    model = SimpleVGG(num_classes=10)
    output = model(dummy_input)
    print("Model created successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
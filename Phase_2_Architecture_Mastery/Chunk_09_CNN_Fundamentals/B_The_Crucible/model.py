import torch 
import torch.nn as nn

class VGG_XRay(nn.Module):
    """
    VGG-style network for grayscale medical Images.
    
    1. First Conv layer `in_channels=1` for grayscale.
    2. Classifier output `num_classes=2`,
    3. 128x128 inputs images 
    """    

    def __init__(self, num_classes = 2): # Two classes ( NORMAL  /  PNEUMONIA )
        super(VGG_XRay, self).__init__()

         # We need more pooling layers for a larger input image.
        self.features = nn.Sequential(
              # Block 1: 128x128 -> 64x64
            nn.Conv2d(1, 32,
                    kernel_size=3,
                    padding=1), # in_channels = 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32,32,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            

            # BLock 2 64x64 -> 32x32
            nn.Conv2d(32, 64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3 : 32x32 -> 16x16
            nn.Conv2d(64, 128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),            
            
            # Block 4 : 16x16 -> 8x8
            nn.Conv2d(128, 256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
         )

        #  After 4 max-pools, 128 -> 64 -> 32 -> 16 -> 8
        # Flattened size is 256 * 8 * 8
        
        self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 8 * 8, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
    
if __name__ == '__main__':
    dummy_input = torch.randint(16, 1, 128, 128) # Test with adpated dimensions
    model = VGG_XRay(num_classes=2)
    output = model(dummy_input)
    print("X-Ray model created successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
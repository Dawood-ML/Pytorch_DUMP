# Now, we arrive at the architectural heart of this lesson. 
# We will build our "baby" ResNet. The core innovation you must internalize is the `ResidualBlock`. 
# Everything else is just stacking these blocks together.


# Let's focus on the why. 
# A "plain" deep network tries to learn a direct mapping, H(x). 
# As it gets deeper, gradients vanish, and performance degrades. 
# A ResNet block reframes the problem: it learns a residual function, 
# F(x), and the final output is H(x) = F(x) + x. The + x is the "skip connection." 
# This simple addition allows gradients to flow unimpeded through the identity path (x), 
# enabling the training of dramatically deeper networks.

# imports 
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Conceptual Understanding
"""
The ResidualBlock solves the degradation problem in deep networks.
As networks get deeper, plain CNNs often see their performance get worse
due to vanishing gradients, making it hard for early layers to learn.

Key intuition: It's easier for a network to learn to push a residual (F(x))
to zero than to learn an identity mapping (H(x) = x).

Why it works: The skip connection (the '+ x' part) creates a direct path for
the gradient to flow backward through the network. This "gradient highway"
ensures that even the earliest layers receive a strong training signal.

When to use: When building deep CNNs (typically > 20 layers). It has become
the default building block for most modern computer vision architectures.
When NOT to use: For very shallow networks, the complexity might be unnecessary.
A plain CNN might suffice and be faster.

For more information : https://youtu.be/Q1JCrG1bJ-A?si=3P06b8JyZseIvrsv
"""

####################################################################################
#
# 2. Minimal and Production implementation (The Residual Block)
class ResidualBlock(nn.Module):
    """The Fundamental building vlock of a resnet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # This is the main convolution path.
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # This is the skip connection (or identity mapping).
        # It needs to handle two cases:
        # 1. If the dimensions are the same (stride=1, in_channels = out_channels):
        #    We do nothing, just add the input to the output.
        # 2. If the dimensions change (stride > 1 or in_Channels != out_channels):
        #    We need to project the input 'x' to match the output dimensions of conv_path.
        #    This is done with a 1x1 convolution.
        self.skip_connection = nn.Sequential()
        if stride !=1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
            # The magic happens here: input 'x' is added tot eh output of the conv path.
            out   =  self.conv_path(x)
            skip  =  self.skip_connection(x)
            out = out + skip
            return F.relu(out)
##########################################################################################
#
##########################################################################################
#    
# 3. NOw comes the real resnet implementation ( PRODUCTION IMPLEMENTATION [ The Full ResNet Model ] )
class ResNet(nn.Module):
    """A simplified ResNet model for Cifar-10"""
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64 # Initial number of channels after the first conv layer
        
        # Initial convolutional layer to process the input image
        self.conv1  = nn.Sequential(
            nn.Conv2d(3, 64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        
        # Stacjing the residual blocks
        self.layer1 = self._make_layer(block, 
                                       64,  
                                       num_blocks[0], 
                                       stride=1)

        self.layer2 = self._make_layer(block, 
                                       128, 
                                       num_blocks[1], 
                                       stride=2)

        self.layer3 = self._make_layer(block, 
                                       256, 
                                       num_blocks[2], 
                                       stride=2)

        self.layer4 = self._make_layer(block, 
                                       512, 
                                       num_blocks[3], 
                                       stride=2)
        # A global averaging pooling layer followd by the final classifier
        # AdaptiveAvgPool2D is great because it creates a fixed-size output
        # tensor regardless of the input's spatial dimensions.
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Helper Function to create a layer of residual blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.fc(torch.flatten(self.avg_pool(self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(x)))))), 1))
        return out
    
# A common configuration for CIFAR-10 is ResNet-18
def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

# Standalone Testing
# This allows us to verify the model's architecture and forward pass.
if __name__ == "__main__":
    # 4. Visualization (of shapes)
    print("Testing ResNet-18 Architecture")

    # Create a dummy input tensor with the shape of a single batch from our dataloader
    # (batch_size, channels, height, width)
    dummy_input = torch.randn(64, 3, 32, 32)

    # Instantiate the model
    model = ResNet18()

    # Pass the dummy input through the model
    output = model(dummy_input)
    print(model)

    # Check the output shape
    print(f"\nInput shape : {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") # Should be [batch_Size, num_classes], e.g., [64,10]

    # Verify the output shape is correct
    assert output.shape == (64, 10), "The output shape is incorrect!"

    print("\nModel architecture seems correct. Forward pass successful.")

    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters : {num_params:,}")


# SUMMARY BELOW
######################################################################################
# 1. ResidualBlock is the Star: This class is the core concept. 
# Notice the two paths: conv_path and skip_connection. 
# The most critical part is the if stride != 1 or in_channels != out_channels: logic. 
# This is how we handle changes in dimensionality. W
# hen we want to downsample the image (halve its height/width), 
# we use a stride=2 convolution. This means the output of conv_path will be smaller than the input x. 
# To make them addable, we must also downsample x in the skip_connection using a 1x1 convolution with the same stride.

# 2. _make_layer Helper: This is a standard ResNet design pattern. 
# Instead of writing self.block1 = ResidualBlock(...), self.block2 = ResidualBlock(...) over and over, 
# this function programmatically creates a sequence of blocks for us. 
# It elegantly handles setting the stride to 2 for the first block in a layer (to downsample) and 1 for all subsequent blocks.

# 3. The ResNet Class: This assembles the pieces. It starts with a standard conv layer, 
# then stacks the layers made by our helper function, and finishes with AdaptiveAvgPool2d and a linear classifier. 
# This structure is highly modular and scalable.

# 4. ResNet18(): We create a simple function to return a specific configuration of our ResNet class. 
# The list [2, 2, 2, 2] means we create 2 blocks in each of the 4 layers, 
# giving us a total of 1 (initial conv) + 2*2 + 2*2 + 2*2 + 2*2 + 1 (fc) = 18 effective layers.

# 5. Sanity Check (if __name__ == '__main__':): This is our self-test. 
# We create a fake batch of data with the correct input dimensions (64, 3, 32, 32) and pass it through the model. 
# If the output shape is (64, 10), we know all our layers connect correctly and 
# the tensor dimensions are handled properly throughout the network. 
# This simple test saves hours of debugging during training.

################################################################################################################################
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

# 2. Minimal and Production implementation (The Residual Block)
class ResidualBlock(nn.Module):
    """The Fundamental building vlock of a resnet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # This is the main convolution path.
        self
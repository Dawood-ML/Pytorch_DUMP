# ============================================================================
# PYTORCH FUNDAMENTALS: CHUNKS 1-10 COMPLETE GUIDE
# From Zero to Project-Ready in PyTorch
# ============================================================================

"""
This is your complete foundation in PyTorch. Work through each chunk sequentially.
Each chunk builds on previous ones. Code is production-ready and follows best practices.

SETUP INSTRUCTIONS:
1. Create a new conda/venv environment: python -m venv pytorch_env
2. Activate: source pytorch_env/bin/activate (Linux/Mac) or pytorch_env\Scripts\activate (Windows)
3. Install: pip install torch torchvision torchaudio matplotlib numpy scikit-learn tensorboard
4. Create a project directory and organize code by chunks
5. Use Jupyter notebooks for exploration, .py files for production code

Each chunk has:
- Concept explanation
- From-scratch implementation
- PyTorch API way
- Common mistakes and debugging
- Practice exercises
"""
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
# ============================================================================
# CHUNK 1: TENSOR FUNDAMENTALS
# Goal: Master PyTorch tensors - the foundation of everything
# ============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("CHUNK 1: TENSOR FUNDAMENTALS")
print("="*80)

# 1.1 TENSOR CREATION
print("\n1.1 Tensor Creation")
print("-" * 40)

# From Python lists
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(f"From list: {tensor_from_list}")

# From numpy arrays
numpy_array = np.array([1, 2, 3, 4, 5])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"From numpy: {tensor_from_numpy}")

# Zeros, ones, empty
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
empty = torch.empty(2, 2)  # Uninitialized values
print(f"\nZeros shape: {zeros.shape}")
print(f"Ones shape: {ones.shape}")

# Random tensors
random_uniform = torch.rand(3, 3)  # Uniform [0, 1)
random_normal = torch.randn(3, 3)  # Normal distribution (μ=0, σ=1)
random_int = torch.randint(0, 10, (3, 3))  # Random integers
print(f"\nRandom uniform:\n{random_uniform}")

# Range tensors
arange = torch.arange(0, 10, 2)  # start, end, step
linspace = torch.linspace(0, 1, 5)  # start, end, num_points
print(f"\nArange: {arange}")
print(f"Linspace: {linspace}")

# Identity matrix
identity = torch.eye(4)
print(f"\nIdentity matrix:\n{identity}")

# 1.2 TENSOR PROPERTIES
print("\n1.2 Tensor Properties")
print("-" * 40)

x = torch.randn(3, 4, 5)
print(f"Shape: {x.shape}")
print(f"Size: {x.size()}")  # Same as shape
print(f"Dtype: {x.dtype}")
print(f"Device: {x.device}")
print(f"Requires grad: {x.requires_grad}")
print(f"Number of elements: {x.numel()}")
print(f"Number of dimensions: {x.ndim}")

# 1.3 INDEXING AND SLICING
print("\n1.3 Indexing and Slicing")
print("-" * 40)

x = torch.arange(24).reshape(4, 6)
print(f"Original tensor:\n{x}")

# Basic indexing
print(f"\nFirst row: {x[0]}")
print(f"Last column: {x[:, -1]}")
print(f"Element at (2, 3): {x[2, 3]}")

# Slicing
print(f"\nFirst 2 rows, first 3 columns:\n{x[:2, :3]}")
print(f"Every other row:\n{x[::2]}")

# Boolean indexing
mask = x > 10
print(f"\nElements > 10: {x[mask]}")

# Advanced indexing
indices = torch.tensor([0, 2])
print(f"\nRows 0 and 2:\n{x[indices]}")

# 1.4 RESHAPING AND VIEW
print("\n1.4 Reshaping and View")
print("-" * 40)

x = torch.arange(12)
print(f"Original: {x}")

# Reshape (creates new tensor if needed)
reshaped = x.reshape(3, 4)
print(f"\nReshape to (3, 4):\n{reshaped}")

# View (must be contiguous)
view = x.view(2, 6)
print(f"\nView as (2, 6):\n{view}")

# Flatten
flattened = reshaped.flatten()
print(f"\nFlattened: {flattened}")

# Squeeze and unsqueeze
x = torch.randn(1, 3, 1, 4)
print(f"\nOriginal shape: {x.shape}")
squeezed = x.squeeze()  # Remove dimensions of size 1
print(f"Squeezed shape: {squeezed.shape}")
unsqueezed = squeezed.unsqueeze(0)  # Add dimension at position 0
print(f"Unsqueezed shape: {unsqueezed.shape}")

# Transpose and permute
x = torch.randn(2, 3, 4)
transposed = x.transpose(0, 1)  # Swap dimensions 0 and 1
permuted = x.permute(2, 0, 1)  # Reorder dimensions
print(f"\nOriginal: {x.shape}")
print(f"Transposed: {transposed.shape}")
print(f"Permuted: {permuted.shape}")

# 1.5 TENSOR OPERATIONS
print("\n1.5 Tensor Operations")
print("-" * 40)

# Element-wise operations
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"Addition: {x + y}")
print(f"Subtraction: {x - y}")
print(f"Multiplication: {x * y}")
print(f"Division: {x / y}")
print(f"Power: {x ** 2}")

# In-place operations (end with _)
x.add_(1)  # Modifies x in-place
print(f"After add_(1): {x}")

# Reduction operations
x = torch.randn(3, 4)
print(f"\nTensor:\n{x}")
print(f"Sum: {x.sum()}")
print(f"Mean: {x.mean()}")
print(f"Max: {x.max()}")
print(f"Min: {x.min()}")
print(f"Std: {x.std()}")

# Along specific dimension
print(f"\nSum along dim 0: {x.sum(dim=0)}")  # Column sums
print(f"Mean along dim 1: {x.mean(dim=1)}")  # Row means

# Matrix operations
a = torch.randn(3, 4)
b = torch.randn(4, 5)
matmul = torch.matmul(a, b)  # or a @ b
print(f"\nMatrix multiplication shape: {matmul.shape}")

# 1.6 BROADCASTING
print("\n1.6 Broadcasting")
print("-" * 40)

# PyTorch automatically broadcasts tensors of different shapes
x = torch.randn(3, 1)
y = torch.randn(1, 4)
result = x + y  # (3, 1) + (1, 4) -> (3, 4)
print(f"x shape: {x.shape}, y shape: {y.shape}")
print(f"Result shape: {result.shape}")

# Broadcasting rules:
# 1. If tensors have different number of dimensions, prepend 1s to smaller
# 2. Dimensions are compatible if they're equal or one of them is 1
# 3. Tensors are broadcastable if all dimensions are compatible

# Examples
x = torch.randn(5, 3, 4)
y = torch.randn(3, 1)
result = x + y  # y is broadcasted to (5, 3, 4)
print(f"\nx: {x.shape}, y: {y.shape}, result: {result.shape}")

# 1.7 GPU OPERATIONS
print("\n1.7 GPU Operations")
print("-" * 40)

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Move tensor to device
x = torch.randn(3, 4)
x_gpu = x.to(device)
print(f"CPU tensor device: {x.device}")
print(f"GPU tensor device: {x_gpu.device}")

# Create tensor directly on device
x_gpu = torch.randn(3, 4, device=device)

# Move back to CPU
x_cpu = x_gpu.cpu()

# IMPORTANT: Operations must be on same device
# x_cpu + x_gpu  # This will error!

# 1.8 TENSOR VS NUMPY
print("\n1.8 Tensor vs NumPy")
print("-" * 40)

# NumPy to Tensor
numpy_arr = np.array([1, 2, 3, 4, 5])
tensor = torch.from_numpy(numpy_arr)
print(f"NumPy array: {numpy_arr}")
print(f"Tensor: {tensor}")

# Tensor to NumPy (only works for CPU tensors)
tensor = torch.tensor([1, 2, 3, 4, 5])
numpy_arr = tensor.numpy()
print(f"Tensor: {tensor}")
print(f"NumPy array: {numpy_arr}")

# WARNING: They share memory!
tensor[0] = 100
print(f"After modifying tensor: {numpy_arr}")  # NumPy array also changed!

# To avoid sharing memory, use .clone()
tensor = torch.tensor([1, 2, 3, 4, 5])
tensor_copy = tensor.clone()
numpy_arr = tensor_copy.numpy()

# Key differences:
# - Tensors can be on GPU, NumPy is CPU-only
# - Tensors support autograd (automatic differentiation)
# - Tensors are optimized for deep learning operations

# 1.9 COMMON MISTAKES
print("\n1.9 Common Mistakes")
print("-" * 40)

# Mistake 1: Mixing devices
try:
    x_cpu = torch.randn(3)
    x_gpu = torch.randn(3, device='cuda' if torch.cuda.is_available() else 'cpu')
    # result = x_cpu + x_gpu  # RuntimeError!
    print("Don't mix CPU and GPU tensors!")
except RuntimeError as e:
    print(f"Error: {e}")

# Mistake 2: Shape mismatches
try:
    a = torch.randn(3, 4)
    b = torch.randn(5, 6)
    # result = a + b  # RuntimeError!
    print("Shapes must be compatible for operations!")
except RuntimeError as e:
    print(f"Error: {e}")

# Mistake 3: Modifying view affects original
x = torch.randn(4, 4)
view = x.view(16)
view[0] = 999
print(f"\nOriginal tensor affected by view modification: {x[0, 0]}")

# Mistake 4: Forgetting to detach when converting to NumPy
x = torch.randn(3, requires_grad=True)
try:
    # arr = x.numpy()  # RuntimeError if requires_grad=True
    arr = x.detach().numpy()  # Correct way
    print("Remember to detach() tensors with gradients before converting to NumPy")
except RuntimeError as e:
    print(f"Error: {e}")


# ============================================================================
# CHUNK 1 EXERCISES
# ============================================================================

print("\n" + "="*80)
print("CHUNK 1 EXERCISES")
print("="*80)

"""
1. Create a 5x5 tensor with random values and:
   - Extract the diagonal elements
   - Replace all values > 0.5 with 1 and <= 0.5 with 0
   - Calculate the sum of each row and column

2. Create two tensors of shape (100, 50) and (50, 200):
   - Perform matrix multiplication
   - Time the operation on CPU vs GPU (if available)
   - Verify results are the same

3. Implement a function that normalizes a tensor to have mean=0 and std=1:
   - Handle different dimensions (batch dimension should not be normalized together)
   - Test with tensors of different shapes

4. Create a checkerboard pattern tensor (alternating 0s and 1s):
   - Use broadcasting, no loops
   - Make it work for any size

5. Slice a 4D tensor (B, C, H, W) to extract:
   - All channels of the first image
   - Center crop of all images
   - Every other image
"""
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# ============================================================================
# CHUNK 2: AUTOGRAD AND BACKPROPAGATION
# Goal: Understand automatic differentiation - the heart of deep learning
# ============================================================================

print("\n" + "="*80)
print("CHUNK 2: AUTOGRAD AND BACKPROPAGATION")
print("="*80)

import torch

# 2.1 REQUIRES_GRAD
print("\n2.1 requires_grad")
print("-" * 40)

# Tensors track computations when requires_grad=True
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

print(f"x: {x}")
print(f"x.requires_grad: {x.requires_grad}")

# Perform operations
z = x + y
w = z * z
loss = w.sum()

print(f"\nloss: {loss}")
print(f"loss.requires_grad: {loss.requires_grad}")

# 2.2 COMPUTATIONAL GRAPH
print("\n2.2 Computational Graph")
print("-" * 40)

# Every operation creates a computational graph
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = x²
z = 2 * y   # z = 2x²
loss = z.sum()  # loss = 2x²

print(f"x: {x.item()}")
print(f"y: {y.item()}")
print(f"z: {z.item()}")
print(f"loss: {loss.item()}")

# The graph tracks the relationship:
# x -> y=x² -> z=2y -> loss=sum(z)
print(f"\nloss.grad_fn: {loss.grad_fn}")  # SumBackward
print(f"z.grad_fn: {z.grad_fn}")  # MulBackward
print(f"y.grad_fn: {y.grad_fn}")  # PowBackward

# 2.3 BACKWARD PASS
print("\n2.3 Backward Pass")
print("-" * 40)

# Compute gradients
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
loss = y.sum()

print(f"Before backward: x.grad = {x.grad}")

loss.backward()  # Compute gradients

print(f"After backward: x.grad = {x.grad}")

# Mathematical verification:
# loss = x², dloss/dx = 2x = 2*2 = 4
print(f"Expected gradient: {2 * x.item()}")

# 2.4 GRADIENT ACCUMULATION
print("\n2.4 Gradient Accumulation")
print("-" * 40)

x = torch.tensor([1.0], requires_grad=True)

# First backward pass
y = x ** 2
y.backward()
print(f"After first backward: x.grad = {x.grad}")

# Second backward pass WITHOUT zeroing gradients
y = x ** 3
y.backward()
print(f"After second backward: x.grad = {x.grad}")  # Accumulated!

# IMPORTANT: Always zero gradients before backward pass
x.grad.zero_()
y = x ** 3
y.backward()
print(f"After zeroing and backward: x.grad = {x.grad}")

# 2.5 DETACHING AND NO_GRAD
print("\n2.5 Detaching and no_grad")
print("-" * 40)

x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# Stop tracking operations
y_detached = y.detach()
print(f"y.requires_grad: {y.requires_grad}")
print(f"y_detached.requires_grad: {y_detached.requires_grad}")

# Context manager for no gradient tracking
with torch.no_grad():
    y = x ** 2
    print(f"Inside no_grad, y.requires_grad: {y.requires_grad}")

# Use no_grad for inference to save memory
x = torch.randn(1000, 1000, requires_grad=True)
with torch.no_grad():
    y = x @ x.T  # No computational graph created
    
# 2.6 MANUAL GRADIENT COMPUTATION (from scratch)
print("\n2.6 Manual Gradient Computation")
print("-" * 40)

def manual_linear_regression():
    """Linear regression: y = wx + b, implemented manually"""
    
    # Data
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_true = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    
    # Parameters (no autograd for manual implementation)
    w = torch.tensor([[0.0]], requires_grad=False)
    b = torch.tensor([[0.0]], requires_grad=False)
    
    learning_rate = 0.01
    epochs = 100
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = X @ w + b
        
        # Loss (MSE)
        loss = ((y_pred - y_true) ** 2).mean()
        
        # Manual backward pass
        # dloss/dy_pred = 2(y_pred - y_true) / n
        grad_y_pred = 2 * (y_pred - y_true) / y_true.size(0)
        
        # dy_pred/dw = X (chain rule)
        grad_w = X.T @ grad_y_pred
        
        # dy_pred/db = 1
        grad_b = grad_y_pred.sum()
        
        # Update parameters
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")
    
    return w, b

print("Manual implementation:")
w_manual, b_manual = manual_linear_regression()

# 2.7 AUTOGRAD LINEAR REGRESSION
print("\n2.7 Autograd Linear Regression")
print("-" * 40)

def autograd_linear_regression():
    """Same linear regression using PyTorch autograd"""
    
    # Data
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_true = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    
    # Parameters (with autograd)
    w = torch.zeros((1, 1), requires_grad=True)
    b = torch.zeros((1, 1), requires_grad=True)
    
    learning_rate = 0.01
    epochs = 100
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = X @ w + b
        
        # Loss
        loss = ((y_pred - y_true) ** 2).mean()
        
        # Backward pass (automatic!)
        loss.backward()
        
        # Update parameters (no_grad to prevent tracking)
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            
            # Zero gradients
            w.grad.zero_()
            b.grad.zero_()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")
    
    return w, b

print("\nAutograd implementation:")
w_auto, b_auto = autograd_linear_regression()

# 2.8 HIGHER-ORDER DERIVATIVES
print("\n2.8 Higher-Order Derivatives")
print("-" * 40)

# Second derivatives
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3  # y = x³

# First derivative: dy/dx = 3x²
y.backward(create_graph=True)  # create_graph=True to compute second derivative
print(f"First derivative (dy/dx): {x.grad}")

# Second derivative: d²y/dx² = 6x
x.grad.backward()
print(f"Second derivative (d²y/dx²): 6 * {x.item()} = {6 * x.item()}")

# 2.9 GRADIENT CHECKING
print("\n2.9 Gradient Checking")
print("-" * 40)

def numerical_gradient(f, x, eps=1e-5):
    """Compute gradient numerically using finite differences"""
    x_plus = x + eps
    x_minus = x - eps
    return (f(x_plus) - f(x_minus)) / (2 * eps)

# Define a function
def f(x):
    return (x ** 3).sum()

# Compute gradient with autograd
x = torch.tensor([2.0], requires_grad=True)
y = f(x)
y.backward()
autograd_grad = x.grad.item()

# Compute gradient numerically
x_no_grad = torch.tensor([2.0])
numerical_grad = numerical_gradient(f, x_no_grad).item()

print(f"Autograd gradient: {autograd_grad}")
print(f"Numerical gradient: {numerical_grad}")
print(f"Difference: {abs(autograd_grad - numerical_grad):.2e}")

# 2.10 COMMON MISTAKES
print("\n2.10 Common Mistakes")
print("-" * 40)

# Mistake 1: Forgetting to zero gradients
print("Mistake 1: Not zeroing gradients")
x = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"Iteration {i+1}: x.grad = {x.grad.item()}")  # Accumulates!

# Correct way
x = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"Iteration {i+1} (with zero): x.grad = {x.grad.item()}")
    x.grad.zero_()

# Mistake 2: In-place operations
print("\nMistake 2: In-place operations")
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
# y += 1  # RuntimeError! In-place operation on leaf variable
y = y + 1  # Correct way
y.backward()

# Mistake 3: Backward on non-scalar
print("\nMistake 3: Backward on non-scalar")
x = torch.randn(3, requires_grad=True)
y = x ** 2
# y.backward()  # RuntimeError! grad can be implicitly created only for scalar outputs
y.backward(torch.ones_like(y))  # Must provide gradient argument

# Mistake 4: Using tensor with gradients in NumPy
print("\nMistake 4: Converting to NumPy without detaching")
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
# arr = y.numpy()  # RuntimeError!
arr = y.detach().numpy()  # Correct
print(f"Converted to NumPy: {arr}")


# ============================================================================
# CHUNK 2 EXERCISES
# ============================================================================

print("\n" + "="*80)
print("CHUNK 2 EXERCISES")
print("="*80)

"""
1. Implement polynomial regression (y = w3*x³ + w2*x² + w1*x + b):
   - First manually compute gradients
   - Then use autograd
   - Verify they match

2. Create a function that computes gradients of:
   - f(x, y) = x²y + y³
   - Verify using numerical gradients
   - Compute second derivatives

3. Implement a simple neural network layer manually:
   - y = relu(Wx + b)
   - Compute forward and backward passes
   - Handle batch inputs

4. Debug gradient flow:
   - Create a computation with detached tensors
   - Identify where gradients stop flowing
   - Fix the computation

5. Implement gradient clipping:
   - Clip gradients by norm
   - Clip gradients by value
   - Show when/why this is needed
"""
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# ============================================================================
# CHUNK 3: THE nn.MODULE PHILOSOPHY
# Goal: Master PyTorch's module system - the building blocks of neural networks
# ============================================================================

print("\n" + "="*80)
print("CHUNK 3: THE nn.MODULE PHILOSOPHY")
print("="*80)

import torch
import torch.nn as nn

# 3.1 YOUR FIRST MODULE
print("\n3.1 Your First Module")
print("-" * 40)

class SimpleLinear(nn.Module):
    """A simple linear layer: y = Wx + b"""
    
    def __init__(self, in_features, out_features):
        super().__init__()  # ALWAYS call parent __init__
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        """Defines the computation performed at every call"""
        return x @ self.weight.T + self.bias

# Create and test
layer = SimpleLinear(10, 5)
x = torch.randn(32, 10)  # Batch of 32, 10 features
output = layer(x)  # Calls forward()
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Parameters: {list(layer.parameters())}")

# 3.2 UNDERSTANDING nn.MODULE
print("\n3.2 Understanding nn.Module")
print("-" * 40)

class ExplainedModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # These are automatically registered as parameters
        self.weight = nn.Parameter(torch.randn(5, 5))
        
        # Submodules are automatically registered
        self.linear = nn.Linear(5, 3)
        
        # Regular attributes are NOT parameters
        self.some_value = 42
    
    def forward(self, x):
        x = x @ self.weight
        x = self.linear(x)
        return x

model = ExplainedModule()
print("All parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

print("\nAll modules:")
for name, module in model.named_modules():
    print(f"  {name}: {module.__class__.__name__}")

# 3.3 BUILT-IN LAYERS
print("\n3.3 Built-in Layers")
print("-" * 40)

# Linear layer
linear = nn.Linear(10, 5)  # in_features=10, out_features=5
print(f"Linear weight shape: {linear.weight.shape}")  # (5, 10)
print(f"Linear bias shape: {linear.bias.shape}")  # (5,)

# Activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

x = torch.randn(5)
print(f"\nInput: {x}")
print(f"ReLU: {relu(x)}")
print(f"Sigmoid: {sigmoid(x)}")

# Dropout (different in train vs eval mode)
dropout = nn.Dropout(p=0.5)
x = torch.ones(10)
print(f"\nInput: {x}")
dropout.train()  # Training mode
print(f"Dropout (train): {dropout(x)}")
dropout.eval()  # Evaluation mode
print(f"Dropout (eval): {dropout(x)}")

# Batch normalization
batch_norm = nn.BatchNorm1d(10)  # 10 features
x = torch.randn(32, 10)  # Batch of 32
output = batch_norm(x)
print(f"\nBatchNorm input mean: {x.mean(0)[:3]}")
print(f"BatchNorm output mean: {output.mean(0)[:3]}")  # ~0

# 3.4 MODULE COMPOSITION
print("\n3.4 Module Composition")
print("-" * 40)

class MLPBlock(nn.Module):
    """A single MLP block with layer norm and dropout"""
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    """Multi-layer perceptron built from blocks"""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer (no activation/dropout)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Sequential container
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create and test
model = MLP(input_dim=10, hidden_dims=[64, 32], output_dim=5)
x = torch.randn(16, 10)
output = model(x)
print(f"MLP architecture:\n{model}")
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

# 3.5 PARAMETER INITIALIZATION
print("\n3.5 Parameter Initialization")
print("-" * 40)

class InitializedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Custom weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                # Zero bias
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = InitializedNetwork()

# Common initialization methods
linear = nn.Linear(10, 5)

# Xavier initialization (good for tanh, sigmoid)
nn.init.xavier_uniform_(linear.weight)

# Kaiming/He initialization (good for ReLU)
nn.init.kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')

# Constant initialization
nn.init.constant_(linear.bias, 0)

# Normal/Uniform initialization
nn.init.normal_(linear.weight, mean=0, std=0.01)
nn.init.uniform_(linear.weight, a=-0.1, b=0.1)

print("Initialization methods applied successfully")

# 3.6 NESTED MODULES AND MODULE LISTS
print("\n3.6 Nested Modules and ModuleLists")
print("-" * 40)

class ResidualBlock(nn.Module):
    """A residual block with skip connection"""
    
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.layers(x)  # Skip connection

class DynamicNetwork(nn.Module):
    """Network with variable number of blocks"""
    
    def __init__(self, input_dim, hidden_dim, n_blocks, output_dim):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # ModuleList for variable number of blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        # Pass through all blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        return x

model = DynamicNetwork(input_dim=10, hidden_dim=64, n_blocks=5, output_dim=3)
print(f"Number of residual blocks: {len(model.blocks)}")

x = torch.randn(8, 10)
output = model(x)
print(f"Output shape: {output.shape}")

# ModuleDict for named submodules
class MultiTaskNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.backbone = nn.Linear(input_dim, 128)
        
        # ModuleDict for multiple task heads
        self.heads = nn.ModuleDict({
            'classification': nn.Linear(128, 10),
            'regression': nn.Linear(128, 1),
            'segmentation': nn.Linear(128, 5)
        })
    
    def forward(self, x, task='classification'):
        features = torch.relu(self.backbone(x))
        return self.heads[task](features)

model = MultiTaskNetwork(input_dim=20)
x = torch.randn(16, 20)
classification_output = model(x, task='classification')
regression_output = model(x, task='regression')
print(f"\nClassification output shape: {classification_output.shape}")
print(f"Regression output shape: {regression_output.shape}")

# 3.7 TRAIN VS EVAL MODE
print("\n3.7 Train vs Eval Mode")
print("-" * 40)

class NetworkWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(20, 5)
        self.batch_norm = nn.BatchNorm1d(5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        return x

model = NetworkWithDropout()
x = torch.randn(32, 10)

# Training mode (dropout active, batch norm updates statistics)
model.train()
train_output = model(x)
print(f"Training mode: {model.training}")
print(f"Output mean: {train_output.mean().item():.4f}")

# Evaluation mode (dropout inactive, batch norm uses running statistics)
model.eval()
eval_output = model(x)
print(f"\nEvaluation mode: {model.training}")
print(f"Output mean: {eval_output.mean().item():.4f}")

# CRITICAL: Always set mode appropriately!
# Training: model.train()
# Inference: model.eval()

# 3.8 SAVING AND LOADING MODELS
print("\n3.8 Saving and Loading Models")
print("-" * 40)

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

# Create and train model
model = SimpleModel(10, 5)
optimizer = torch.optim.Adam(model.parameters())

# Method 1: Save entire model (not recommended)
# torch.save(model, 'model.pth')
# model = torch.load('model.pth')

# Method 2: Save state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Load state dict
new_model = SimpleModel(10, 5)
new_model.load_state_dict(torch.load('model_weights.pth'))

# Save checkpoint with optimizer state
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.5,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")

# 3.9 BUILDING A COMPLETE MLP FROM SCRATCH
print("\n3.9 Building a Complete MLP from Scratch")
print("-" * 40)

class FlexibleMLP(nn.Module):
    """
    Flexible MLP with customizable architecture.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function name
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 activation='relu', dropout=0.0, batch_norm=False):
        super().__init__()
        
        # Store config
        self.config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm
        }
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

# Create model
model = FlexibleMLP(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    activation='relu',
    dropout=0.2,
    batch_norm=True
)

print(model)
print(f"\nParameters: {model.count_parameters()}")

# Test forward pass
x = torch.randn(32, 784)
output = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

# 3.10 COMMON MISTAKES
print("\n3.10 Common Mistakes")
print("-" * 40)

# Mistake 1: Forgetting super().__init__()
print("Mistake 1: Forgetting super().__init__()")
try:
    class BadModule(nn.Module):
        def __init__(self):
            # super().__init__()  # FORGOT THIS!
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    # model = BadModule()  # Will have issues!
    print("Always call super().__init__() first!")
except Exception as e:
    print(f"Error: {e}")

# Mistake 2: Not returning anything in forward
print("\nMistake 2: Not returning in forward()")
class BadForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        x = self.linear(x)
        # Forgot to return!

model = BadForward()
output = model(torch.randn(5, 10))
print(f"Output is None: {output is None}")

# Mistake 3: Using regular Python list instead of nn.ModuleList
print("\nMistake 3: Using list instead of ModuleList")
class BadList(nn.Module):
    def __init__(self):
        super().__init__()
        # This won't register parameters!
        self.layers = [nn.Linear(10, 10) for _ in range(3)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

bad_model = BadList()
print(f"Parameters registered: {len(list(bad_model.parameters()))}")  # 0!

class GoodList(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ModuleList
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

good_model = GoodList()
print(f"Parameters registered: {len(list(good_model.parameters()))}")  # 6 (3 weights + 3 biases)

# Mistake 4: Modifying input in-place
print("\nMistake 4: In-place operations on inputs")
class InPlaceModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x += 1  # DON'T DO THIS! Modifies input
        x = x + 1  # Correct way
        return x

# Mistake 5: Not setting train/eval mode
print("\nMistake 5: Not setting train/eval mode")
print("ALWAYS use model.train() for training and model.eval() for inference!")


# ============================================================================
# CHUNK 3 EXERCISES
# ============================================================================

print("\n" + "="*80)
print("CHUNK 3 EXERCISES")
print("="*80)

"""
1. Build a custom layer that implements:
   - Layer normalization from scratch
   - Compare with nn.LayerNorm
   - Test on different input shapes

2. Create a ResNet-style block with:
   - Skip connections
   - Batch normalization
   - Optional downsampling
   - Test gradient flow

3. Implement a multi-head network:
   - Shared backbone
   - Multiple task-specific heads
   - Ability to freeze backbone
   - Test on dummy data

4. Build a custom initialization scheme:
   - Implement orthogonal initialization
   - Apply to different layer types
   - Compare with PyTorch built-ins

5. Create a model with conditional computation:
   - Different paths based on input
   - Gating mechanisms
   - Ensure all parameters are used
"""
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# ============================================================================
# CHUNK 4: TRAINING LOOP FUNDAMENTALS
# Goal: Master the canonical PyTorch training loop
# ============================================================================

print("\n" + "="*80)
print("CHUNK 4: TRAINING LOOP FUNDAMENTALS")
print("="*80)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 4.1 THE CANONICAL TRAINING LOOP
print("\n4.1 The Canonical Training Loop")
print("-" * 40)

def basic_training_loop():
    """The fundamental PyTorch training pattern"""
    
    # Create dummy data
    X_train = torch.randn(1000, 20)
    y_train = torch.randint(0, 2, (1000,))
    
    # Create model
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    batch_size = 32
    
    for epoch in range(num_epochs):
        # Training mode
        model.train()
        
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            # Get batch
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print progress
        avg_loss = epoch_loss / (len(X_train) / batch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Running basic training loop:")
basic_training_loop()

# 4.2 TRAINING WITH DATALOADER
print("\n4.2 Training with DataLoader")
print("-" * 40)

def training_with_dataloader():
    """Better training loop using DataLoader"""
    
    # Create dataset
    X = torch.randn(1000, 20)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model, loss, optimizer
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            
            # Update
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Running training with DataLoader:")
training_with_dataloader()

# 4.3 VALIDATION LOOP
print("\n4.3 Validation Loop")
print("-" * 40)

def train_with_validation():
    """Training loop with validation"""
    
    # Create train and val datasets
    X_train = torch.randn(800, 20)
    y_train = torch.randint(0, 2, (800,))
    X_val = torch.randn(200, 20)
    y_val = torch.randint(0, 2, (200,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model, loss, optimizer
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 5
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        # Validation phase
        model.eval()  # Important!
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # No gradients needed
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        # Print metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print("Running training with validation:")
train_with_validation()

# 4.4 CHECKPOINTING
print("\n4.4 Checkpointing")
print("-" * 40)

def train_with_checkpointing():
    """Training loop with model checkpointing"""
    
    # Setup (same as before)
    X_train = torch.randn(800, 20)
    y_train = torch.randint(0, 2, (800,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Tracking
    best_loss = float('inf')
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, 'best_model.pth')
            print(f"  → Saved checkpoint (best loss: {best_loss:.4f})")

print("Running training with checkpointing:")
train_with_checkpointing()

# 4.5 LOSS FUNCTIONS
print("\n4.5 Loss Functions")
print("-" * 40)

# Classification losses
print("Classification losses:")

# Binary cross-entropy
bce_loss = nn.BCEWithLogitsLoss()  # Includes sigmoid
logits = torch.randn(10, 1)
targets = torch.randint(0, 2, (10, 1)).float()
loss = bce_loss(logits, targets)
print(f"BCE Loss: {loss.item():.4f}")

# Cross-entropy (multi-class)
ce_loss = nn.CrossEntropyLoss()  # Includes softmax
logits = torch.randn(10, 5)  # 10 samples, 5 classes
targets = torch.randint(0, 5, (10,))
loss = ce_loss(logits, targets)
print(f"CE Loss: {loss.item():.4f}")

# Regression losses
print("\nRegression losses:")

# Mean Squared Error
mse_loss = nn.MSELoss()
predictions = torch.randn(10, 1)
targets = torch.randn(10, 1)
loss = mse_loss(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")

# Mean Absolute Error
mae_loss = nn.L1Loss()
loss = mae_loss(predictions, targets)
print(f"MAE Loss: {loss.item():.4f}")

# Smooth L1 Loss (Huber)
smooth_l1_loss = nn.SmoothL1Loss()
loss = smooth_l1_loss(predictions, targets)
print(f"Smooth L1 Loss: {loss.item():.4f}")

# 4.6 OPTIMIZERS
print("\n4.6 Optimizers")
print("-" * 40)

model = nn.Linear(10, 5)

# SGD
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with weight decay)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Different learning rates for different parameters
optimizer = optim.Adam([
    {'params': model.weight, 'lr': 0.001},
    {'params': model.bias, 'lr': 0.01}
])

print("Optimizers created successfully")

# 4.7 GRADIENT CLIPPING
print("\n4.7 Gradient Clipping")
print("-" * 40)

def train_with_gradient_clipping():
    """Training with gradient clipping"""
    
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping by norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Or gradient clipping by value
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training with gradient clipping:")
train_with_gradient_clipping()

# 4.8 COMPLETE MNIST EXAMPLE
print("\n4.8 Complete MNIST-style Training")
print("-" * 40)

class MNISTNet(nn.Module):
    """Simple CNN for MNIST-like data"""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_mnist_style():
    """Complete training pipeline"""
    
    # Create synthetic MNIST-like data
    X_train = torch.randn(5000, 784)
    y_train = torch.randint(0, 10, (5000,))
    X_val = torch.randn(1000, 784)
    y_val = torch.randint(0, 10, (1000,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 3
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        # Metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mnist_model.pth')
            print(f"  → Saved best model (acc={best_val_acc:.2f}%)")

print("Training MNIST-style model:")
train_mnist_style()

# 4.9 COMMON MISTAKES
print("\n4.9 Common Mistakes")
print("-" * 40)

print("Mistake 1: Forgetting to zero gradients")
print("  Always call optimizer.zero_grad() before backward()")

print("\nMistake 2: Not setting train/eval mode")
print("  Use model.train() for training, model.eval() for validation")

print("\nMistake 3: Using gradients during validation")
print("  Always wrap validation in 'with torch.no_grad():'")

print("\nMistake 4: Moving data to wrong device")
print("  Both model and data must be on same device (CPU/GPU)")

print("\nMistake 5: Not shuffling training data")
print("  Use shuffle=True in training DataLoader")


# ============================================================================
# CHUNK 4 EXERCISES
# ============================================================================

print("\n" + "="*80)
print("CHUNK 4 EXERCISES")
print("="*80)

"""
1. Implement early stopping:
   - Stop training if validation loss doesn't improve
   - Save best model
   - Restore best weights at end

2. Add learning rate warmup:
   - Gradually increase LR for first N epochs
   - Then use normal schedule
   - Plot LR over time

3. Implement mixed precision training:
   - Use torch.cuda.amp
   - Compare speed and memory usage
   - Verify accuracy is maintained

4. Create a training logger:
   - Log metrics to file
   - Plot training curves
   - Save hyperparameters

5. Implement curriculum learning:
   - Start with easy examples
   - Gradually increase difficulty
   - Measure impact on convergence
"""
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# ============================================================================
# CHUNK 5: DATA LOADING AND PREPROCESSING
# Goal: Master PyTorch's data pipeline
# ============================================================================

print("\n" + "="*80)
print("CHUNK 5: DATA LOADING AND PREPROCESSING")
print("="*80)

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# 5.1 CUSTOM DATASET
print("\n5.1 Custom Dataset")
print("-" * 40)

class SimpleDataset(Dataset):
    """
    Custom dataset must implement:
    - __init__: Initialize dataset
    - __len__: Return dataset size
    - __getitem__: Return one sample
    """
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# Create dataset
data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))
dataset = SimpleDataset(data, labels)

print(f"Dataset length: {len(dataset)}")
sample, label = dataset[0]
print(f"First sample shape: {sample.shape}, label: {label}")

# 5.2 DATALOADER
print("\n5.2 DataLoader")
print("-" * 40)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,  # Number of subprocesses for data loading
    pin_memory=True  # Faster data transfer to GPU
)

print(f"Number of batches: {len(dataloader)}")

# Iterate through batches
for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape={batch_data.shape}, labels shape={batch_labels.shape}")
    if batch_idx >= 2:  # Just show first 3 batches
        break

# 5.3 DATA SPLITTING
print("\n5.3 Data Splitting")
print("-" * 40)

# Random split
dataset = SimpleDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Total samples: {len(dataset)}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5.4 TRANSFORMS
print("\n5.4 Transforms")
print("-" * 40)

# Common transforms for images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Transforms for tensors
tensor_transform = transforms.Compose([
    transforms.Lambda(lambda x: x * 2 + 1),  # Custom transform
    transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
])

print("Transforms created")

# 5.5 CUSTOM TRANSFORMS
print("\n5.5 Custom Transforms")
print("-" * 40)

class Normalize:
    """Normalize to mean=0, std=1"""
    
    def __init__(self):
        pass
    
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-8)

class AddGaussianNoise:
    """Add Gaussian noise to tensor"""
    
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

# Use custom transforms
custom_transform = transforms.Compose([
    Normalize(),
    AddGaussianNoise(mean=0, std=0.05)
])

x = torch.randn(10)
x_transformed = custom_transform(x)
print(f"Original mean: {x.mean():.4f}, std: {x.std():.4f}")
print(f"Transformed mean: {x_transformed.mean():.4f}, std: {x_transformed.std():.4f}")

# 5.6 ADVANCED DATASET
print("\n5.6 Advanced Dataset with Caching")
print("-" * 40)

class CachedDataset(Dataset):
    """Dataset with lazy loading and caching"""
    
    def __init__(self, data_size, feature_dim, cache=True):
        self.data_size = data_size
        self.feature_dim = feature_dim
        self.cache = cache
        self._cache = {} if cache else None
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        # Check cache first
        if self.cache and idx in self._cache:
            return self._cache[idx]
        
        # Generate sample (simulate expensive operation)
        sample = torch.randn(self.feature_dim)
        label = torch.randint(0, 2, (1,)).item()
        
        # Cache if enabled
        if self.cache:
            self._cache[idx] = (sample, label)
        
        return sample, label

dataset = CachedDataset(data_size=100, feature_dim=10, cache=True)
print(f"Dataset length: {len(dataset)}")

# First access (generates data)
sample1, label1 = dataset[0]
# Second access (from cache)
sample2, label2 = dataset[0]
print(f"Cache working: {torch.equal(sample1, sample2)}")

# 5.7 COLLATE FUNCTION
print("\n5.7 Custom Collate Function")
print("-" * 40)

def custom_collate(batch):
    """
    Custom collate function for variable-length sequences.
    Batch is a list of (data, label) tuples.
    """
    # Separate data and labels
    data_list = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad sequences to same length
    max_len = max([d.size(0) for d in data_list])
    padded_data = []
    
    for data in data_list:
        if data.size(0) < max_len:
            padding = torch.zeros(max_len - data.size(0), *data.shape[1:])
            data = torch.cat([data, padding], dim=0)
        padded_data.append(data)
    
    # Stack into batch
    batched_data = torch.stack(padded_data)
    batched_labels = torch.tensor(labels)
    
    return batched_data, batched_labels

# Dataset with variable-length sequences
class VariableLengthDataset(Dataset):
    def __init__(self, size):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random length between 5 and 15
        length = torch.randint(5, 16, (1,)).item()
        data = torch.randn(length, 10)
        label = torch.randint(0, 2, (1,)).item()
        return data, label

var_dataset = VariableLengthDataset(100)
var_loader = DataLoader(var_dataset, batch_size=4, collate_fn=custom_collate)

for batch_data, batch_labels in var_loader:
    print(f"Batch shape: {batch_data.shape}")  # All same length now!
    break

# 5.8 DATA AUGMENTATION
print("\n5.8 Data Augmentation")
print("-" * 40)

class AugmentedDataset(Dataset):
    """Dataset with on-the-fly augmentation"""
    
    def __init__(self, data, labels, augment=True):
        self.data = data
        self.labels = labels
        self.augment = augment
        
        # Augmentation pipeline
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)
            ], p=0.5),
            transforms.RandomApply([
                transforms.Lambda(lambda x: x * torch.rand(1).item() * 0.5 + 0.75)
            ], p=0.5)
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.augment:
            sample = self.transform(sample)
        
        return sample, label

# Create augmented dataset
data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))

train_dataset = AugmentedDataset(data, labels, augment=True)
val_dataset = AugmentedDataset(data, labels, augment=False)

print("Augmented datasets created")

# 5.9 SAMPLER
print("\n5.9 Custom Sampler")
print("-" * 40)

from torch.utils.data import Sampler

class BalancedSampler(Sampler):
    """Sample equal number from each class"""
    
    def __init__(self, labels, samples_per_class):
        self.labels = labels
        self.samples_per_class = samples_per_class
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            label_item = label.item() if torch.is_tensor(label) else label
            if label_item not in self.class_indices:
                self.class_indices[label_item] = []
            self.class_indices[label_item].append(idx)
    
    def __iter__(self):
        indices = []
        for class_idx, class_indices in self.class_indices.items():
            # Sample from this class
            if len(class_indices) >= self.samples_per_class:
                sampled = torch.randperm(len(class_indices))[:self.samples_per_class]
                sampled = [class_indices[i] for i in sampled]
            else:
                # Oversample if needed
                sampled = class_indices * (self.samples_per_class // len(class_indices) + 1)
                sampled = sampled[:self.samples_per_class]
            indices.extend(sampled)
        
        # Shuffle all indices
        indices = torch.tensor(indices)[torch.randperm(len(indices))].tolist()
        return iter(indices)
    
    def __len__(self):
        return len(self.class_indices) * self.samples_per_class

# Create imbalanced dataset
data = torch.randn(100, 10)
labels = torch.cat([torch.zeros(80), torch.ones(20)]).long()  # Imbalanced

dataset = SimpleDataset(data, labels)
sampler = BalancedSampler(labels, samples_per_class=20)

loader = DataLoader(dataset, batch_size=16, sampler=sampler)

# Check class distribution
all_labels = []
for _, batch_labels in loader:
    all_labels.extend(batch_labels.tolist())

print(f"Class 0 count: {all_labels.count(0)}")
print(f"Class 1 count: {all_labels.count(1)}")

# 5.10 COMMON MISTAKES
print("\n5.10 Common Mistakes")
print("-" * 40)

print("Mistake 1: Not shuffling training data")
print("  Always use shuffle=True for training DataLoader")

print("\nMistake 2: Using too many workers")
print("  Start with num_workers=0, increase if data loading is bottleneck")

print("\nMistake 3: Augmenting validation data")
print("  Only augment training data, not validation/test")

print("\nMistake 4: Not using pin_memory")
print("  Use pin_memory=True for faster CPU to GPU transfer")

print("\nMistake 5: Returning tensors without batch dimension")
print("  DataLoader expects samples, adds batch dimension automatically")


# ============================================================================
# CHUNK 5 EXERCISES
# ============================================================================

print("\n" + "="*80)
print("CHUNK 5 EXERCISES")
print("="*80)

"""
1. Create a dataset for time series:
   - Sliding window approach
   - Variable window sizes
   - Handle padding

2. Implement data augmentation for text:
   - Random word replacement
   - Sentence shuffling
   - Back-translation simulation

3. Build a multi-modal dataset:
   - Load image and text pairs
   - Handle missing modalities
   - Custom collate function

4. Create a weighted sampler:
   - Sample inversely proportional to class frequency
   - Handle class imbalance
   - Verify distribution

5. Implement prefetching:
   - Load next batch while training
   - Measure speedup
   - Handle GPU memory
"""
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# ============================================================================
# CHUNK 6: CONVOLUTIONAL NEURAL NETWORKS
# Goal: Master CNNs for computer vision
# ============================================================================

print("\n" + "="*80)
print("CHUNK 6: CONVOLUTIONAL NEURAL NETWORKS")
print("="*80)

import torch
import torch.nn as nn
import torch.nn.functional as F

# 6.1 CONVOLUTION BASICS
print("\n6.1 Convolution Basics")
print("-" * 40)

# 2D Convolution
conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=16,    # 16 filters
    kernel_size=3,      # 3x3 kernel
    stride=1,           # Move by 1 pixel
    padding=1,          # Pad to maintain size
    bias=True
)

# Input: (batch, channels, height, width)
x = torch.randn(8, 3, 32, 32)  # Batch of 8 RGB images (32x32)
output = conv(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Conv weight shape: {conv.weight.shape}")  # (out_ch, in_ch, kh, kw)

# Output size formula: (W - K + 2P) / S + 1
# W=32, K=3, P=1, S=1: (32 - 3 + 2*1) / 1 + 1 = 32

# 6.2 POOLING LAYERS
print("\n6.2 Pooling Layers")
print("-" * 40)

# Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by 2
x = torch.randn(8, 16, 32, 32)
output = max_pool(x)
print(f"Max pool input: {x.shape}")
print(f"Max pool output: {output.shape}")  # 32/2 = 16

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
output = avg_pool(x)
print(f"Avg pool output: {output.shape}")

# Adaptive pooling (output size is fixed regardless of input)
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (batch, channels, 1, 1)
x = torch.randn(8, 512, 7, 7)
output = adaptive_pool(x)
print(f"Adaptive pool input: {x.shape}")
print(f"Adaptive pool output: {output.shape}")

# 6.3 SIMPLE CNN
print("\n6.3 Simple CNN")
print("-" * 40)

class SimpleCNN(nn.Module):
    """Basic CNN for image classification"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 poolings: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv blocks
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128*4*4)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

model = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 32, 32)
output = model(x)
print(f"SimpleCNN output shape: {output.shape}")

# 6.4 BATCH NORMALIZATION
print("\n6.4 Batch Normalization")
print("-" * 40)

class CNNWithBatchNorm(nn.Module):
    """CNN with batch normalization"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Conv + BN + ReLU blocks
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.fc(x)
        
        return x

model = CNNWithBatchNorm()
x = torch.randn(8, 3, 32, 32)
output = model(x)
print(f"CNN with BatchNorm output: {output.shape}")

# 6.5 RESIDUAL CONNECTIONS
print("\n6.5 Residual Connections")
print("-" * 40)

class ResidualBlock(nn.Module):
    """Basic residual block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)  # Skip connection
        out = F.relu(out)
        
        return out

# Test residual block
block = ResidualBlock(64, 64)
x = torch.randn(4, 64, 32, 32)
output = block(x)
print(f"Residual block output: {output.shape}")

# 6.6 BUILDING RESNET-STYLE NETWORK
print("\n6.6 ResNet-Style Network")
print("-" * 40)

class ResNet(nn.Module):
    """Simple ResNet architecture"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        # Classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block might downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

model = ResNet(num_classes=10)
x = torch.randn(4, 3, 224, 224)
output = model(x)
print(f"ResNet output: {output.shape}")

# 6.7 RECEPTIVE FIELD
print("\n6.7 Understanding Receptive Field")
print("-" * 40)

def calculate_receptive_field(layers):
    """
    Calculate receptive field for a sequence of conv/pool layers.
    Each layer is (kernel_size, stride)
    """
    rf = 1  # Receptive field
    stride_product = 1
    
    for kernel_size, stride in layers:
        rf = rf + (kernel_size - 1) * stride_product
        stride_product *= stride
    
    return rf

# Example: VGG-style network
layers = [
    (3, 1),  # conv3x3
    (3, 1),  # conv3x3
    (2, 2),  # pool2x2
    (3, 1),  # conv3x3
    (3, 1),  # conv3x3
    (2, 2),  # pool2x2
]

rf = calculate_receptive_field(layers)
print(f"Receptive field: {rf}x{rf}")

# 6.8 DEPTHWISE SEPARABLE CONVOLUTIONS
print("\n6.8 Depthwise Separable Convolutions")
print("-" * 40)

class DepthwiseSeparableConv(nn.Module):
    """Efficient convolution used in MobileNet"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Depthwise convolution (separate filter per channel)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels  # Key: groups=in_channels
        )
        
        # Pointwise convolution (1x1 to combine)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

# Compare parameters
standard_conv = nn.Conv2d(64, 128, 3, padding=1)
depthwise_sep = DepthwiseSeparableConv(64, 128)

standard_params = sum(p.numel() for p in standard_conv.parameters())
depthwise_params = sum(p.numel() for p in depthwise_sep.parameters())

print(f"Standard conv parameters: {standard_params}")
print(f"Depthwise separable parameters: {depthwise_params}")
print(f"Reduction: {standard_params/depthwise_params:.2f}x")

# 6.9 VISUALIZATION HOOKS
print("\n6.9 Feature Map Visualization")
print("-" * 40)

class FeatureExtractor:
    """Extract intermediate feature maps"""
    
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.features = {}
        self.hooks = []
        
        # Register hooks
        for name, layer in model.named_modules():
            if name in layers:
                hook = layer.register_forward_hook(self.save_output(name))
                self.hooks.append(hook)
    
    def save_output(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# Example usage
model = SimpleCNN()
extractor = FeatureExtractor(model, ['conv1', 'conv2'])

x = torch.randn(1, 3, 32, 32)
_ = model(x)

print(f"Feature maps extracted:")
for name, features in extractor.features.items():
    print(f"  {name}: {features.shape}")

extractor.remove_hooks()

# 6.10 COMMON MISTAKES
print("\n6.10 Common Mistakes")
print("-" * 40)

print("Mistake 1: Wrong input dimensions")
print("  CNNs expect (batch, channels, height, width)")
print("  NOT (batch, height, width, channels)")

print("\nMistake 2: Forgetting to flatten before FC layers")
print("  Use x.view(x.size(0), -1) or x.flatten(1)")

print("\nMistake 3: Not matching padding for same-size output")
print("  For kernel_size=3: use padding=1")
print("  For kernel_size=5: use padding=2")

print("\nMistake 4: Using BatchNorm before activation")
print("  Correct order: Conv -> BN -> Activation")

print("\nMistake 5: Not using adaptive pooling")
print("  Use AdaptiveAvgPool2d for flexible input sizes")


# ============================================================================
# CHUNK 6 EXERCISES
# ============================================================================

print("\n" + "="*80)
print("CHUNK 6 EXERCISES")
print("="*80)

"""
1. Implement AlexNet architecture:
   - Build from scratch
   - Train on dummy data
   - Compare with torchvision version

2. Create a multi-scale CNN:
   - Extract features at different resolutions
   - Concatenate multi-scale features
   - Test on images

3. Implement Squeeze-and-Excitation block:
   - Channel-wise attention
   - Integrate into CNN
   - Measure improvement

4. Build a fully convolutional network:
   - No fully connected layers
   - Global average pooling
   - Handle variable input sizes

5. Visualize learned filters:
   - Extract conv1 filters
   - Plot as images
   - Apply filters to image
"""
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# ============================================================================
# CHUNK 7: OPTIMIZATION AND LEARNING RATE SCHEDULING
# Goal: Master training optimization techniques
# ============================================================================

print("\n" + "="*80)
print("CHUNK 7: OPTIMIZATION AND LEARNING RATE SCHEDULING")
print("="*80)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR

# 7.1 OPTIMIZER COMPARISON
print("\n7.1 Optimizer Comparison")
print("-" * 40)

def compare_optimizers():
    """Compare different optimizers on same task"""
    
    # Simple model
    model = nn.Linear(10, 1)
    
    # Data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Loss
    criterion = nn.MSELoss()
    
    # Different optimizers
    optimizers_dict = {
        'SGD': optim.SGD(model.parameters(), lr=0.01),
        'SGD+Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'Adam': optim.Adam(model.parameters(), lr=0.01),
        'AdamW': optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01),
        'RMSprop': optim.RMSprop(model.parameters(), lr=0.01),
    }
    
    for name, optimizer in optimizers_dict.items():
        # Reset model
        model = nn.Linear(10, 1)
        optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
        
        # Train briefly
        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print(f"{name:15s} - Final loss: {losses[-1]:.6f}")

compare_optimizers()

# 7.2 SGD WITH MOMENTUM
print("\n7.2 SGD with Momentum")
print("-" * 40)

# Without momentum
optimizer_sgd = optim.SGD(
    model.parameters(),
    lr=0.01
)

# With momentum (smooths updates)
optimizer_momentum = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

# With Nesterov momentum (look-ahead gradient)
optimizer_nesterov = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True
)

print("SGD variants created")

# 7.3 ADAM VARIANTS
print("\n7.3 Adam Variants")
print("-" * 40)

model = nn.Linear(10, 1)

# Standard Adam
optimizer_adam = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),  # Exponential decay rates
    eps=1e-8
)

# AdamW (Adam with decoupled weight decay)
optimizer_adamw = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01  # Better than L2 regularization
)

# Different learning rates for different layers
optimizer_custom = optim.Adam([
    {'params': model.weight, 'lr': 0.001},
    {'params': model.bias, 'lr': 0.01}
])

print("Adam variants created")

# 7.4 LEARNING RATE SCHEDULERS
print("\n7.4 Learning Rate Schedulers")
print("-" * 40)

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Step LR: Reduce LR by gamma every step_size epochs
scheduler_step = StepLR(optimizer, step_size=10, gamma=0.1)

# Multi-step LR: Reduce at specific epochs
scheduler_multistep = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)

# Exponential LR: Multiply by gamma each epoch
scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing: Smooth decrease following cosine
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# Reduce on plateau: Reduce when metric plateaus
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)

print("Schedulers created")

# 7.5 ONE CYCLE POLICY
print("\n7.5 One Cycle Policy")
print("-" * 40)

def train_with_onecycle():
    """Training with OneCycleLR (fast convergence)"""
    
    # Model and data
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    X = torch.randn(1000, 20)
    y = torch.randn(1000, 1)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=10,
        steps_per_epoch=len(loader),
        pct_start=0.3,  # Warmup for 30% of training
        anneal_strategy='cos'
    )
    
    criterion = nn.MSELoss()
    
    # Training
    for epoch in range(10):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step after each batch!
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, LR: {current_lr:.6f}")

print("Training with OneCycleLR:")
train_with_onecycle()

# 7.6 GRADIENT ACCUMULATION
print("\n7.6 Gradient Accumulation")
print("-" * 40)

def train_with_gradient_accumulation():
    """Simulate larger batch size via gradient accumulation"""
    
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=10
    )
    
    accumulation_steps = 4  # Effective batch size = 10 * 4 = 40
    
    optimizer.zero_grad()
    for i, (batch_X, batch_y) in enumerate(loader):
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update only every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    print("Gradient accumulation completed")

train_with_gradient_accumulation()

# 7.7 GRADIENT CLIPPING
print("\n7.7 Gradient Clipping")
print("-" * 40)

model = nn.LSTM(10, 50, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

X = torch.randn(20, 32, 10)  # (seq_len, batch, features)
targets = torch.randn(20, 32, 50)
criterion = nn.MSELoss()

# Forward and backward
optimizer.zero_grad()
outputs, _ = model(X)
loss = criterion(outputs, targets)
loss.backward()

# Gradient clipping by norm
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Or gradient clipping by value
# torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Check gradient norms
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f"Gradient norm after clipping: {total_norm:.4f}")

optimizer.step()

# 7.8 WEIGHT DECAY AND REGULARIZATION
print("\n7.8 Weight Decay and Regularization")
print("-" * 40)

# Weight decay (L2 regularization)
optimizer_l2 = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# AdamW (decoupled weight decay - better!)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Manual L2 regularization (if needed)
def train_with_manual_l2():
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    l2_lambda = 0.01
    
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Add L2 penalty
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += l2_lambda * l2_reg
    
    loss.backward()
    optimizer.step()
    
    print(f"Loss with L2: {loss.item():.4f}")

train_with_manual_l2()

# 7.9 WARMUP STRATEGIES
print("\n7.9 Learning Rate Warmup")
print("-" * 40)

class WarmupScheduler:
    """Custom warmup + cosine annealing scheduler"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr + (self.max_lr - self.base_lr) * \
                 self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr + (self.max_lr - self.base_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

# Test warmup scheduler
import numpy as np
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters())
scheduler = WarmupScheduler(optimizer, warmup_epochs=10, total_epochs=100,
                           base_lr=0, max_lr=0.001)

print("Learning rates over epochs:")
for epoch in range(20):
    lr = scheduler.step()
    if epoch % 5 == 0:
        print(f"  Epoch {epoch}: LR = {lr:.6f}")

# 7.10 MONITORING TRAINING
print("\n7.10 Monitoring Training")
print("-" * 40)

class TrainingMonitor:
    """Monitor gradients, weights, and learning rate"""
    
    def __init__(self, model):
        self.model = model
    
    def check_gradients(self):
        """Check for vanishing/exploding gradients"""
        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
        
        if grad_norms:
            avg_norm = sum(grad_norms) / len(grad_norms)
            max_norm = max(grad_norms)
            min_norm = min(grad_norms)
            
            print(f"Gradient norms - Avg: {avg_norm:.6f}, Max: {max_norm:.6f}, Min: {min_norm:.6f}")
            
            if max_norm > 10:
                print("  WARNING: Possible exploding gradients!")
            if max_norm < 1e-7:
                print("  WARNING: Possible vanishing gradients!")
    
    def check_weights(self):
        """Check weight statistics"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                mean = param.data.mean().item()
                std = param.data.std().item()
                print(f"{name}: mean={mean:.6f}, std={std:.6f}")

# Test monitor
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

X = torch.randn(32, 10)
y = torch.randn(32, 1)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
monitor = TrainingMonitor(model)

# One training step
optimizer.zero_grad()
outputs = model(X)
loss = criterion(outputs, y)
loss.backward()
optimizer.step()

print("Gradient check:")
monitor.check_gradients()

print("\nWeight check:")
monitor.check_weights()


# ============================================================================
# CHUNK 7 EXERCISES
# ============================================================================

print("\n" + "="*80)
print("CHUNK 7 EXERCISES")
print("="*80)

"""
1. Implement custom optimizer:
   - Create SGD from scratch
   - Add momentum
   - Compare with PyTorch SGD

2. Build adaptive learning rate finder:
   - Implement LR range test
   - Plot loss vs LR
   - Find optimal LR

3. Create custom scheduler:
   - Implement cyclic LR
   - Add warmup restarts
   - Test on dummy task

4. Implement layer-wise learning rates:
   - Different LR for each layer
   - Discriminative fine-tuning
   - Compare with uniform LR

5. Build optimization diagnostics:
   - Track gradient flow
   - Monitor weight updates
   - Detect training issues
"""
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# ============================================================================
# CHUNK 8: DEBUGGING AND VISUALIZATION
# Goal: Master debugging techniques and visualization
# ============================================================================

print("\n" + "="*80)
print("CHUNK 8: DEBUGGING AND VISUALIZATION")
print("="*80)

import torch
import torch.nn as nn

# 8.1 COMMON ERRORS AND FIXES
print("\n8.1 Common Errors and Fixes")
print("-" * 40)

print("Error 1: Shape mismatch")
print("  Problem: RuntimeError: size mismatch")
print("  Solution: Check input/output shapes at each layer")
print("  Use: print(x.shape) after each operation")

print("\nError 2: Gradient flow issues")
print("  Problem: Loss not decreasing")
print("  Solution: Check if requires_grad=True, verify gradients exist")
print("  Use: print(param.grad) for each parameter")

print("\nError 3: NaN or Inf values")
print("  Problem: Loss becomes NaN")
print("  Solution: Check learning rate, use gradient clipping")
print("  Use: torch.isnan(loss), torch.isinf(loss)")

print("\nError 4: GPU out of memory")
print("  Problem: CUDA out of memory")
print("  Solution: Reduce batch size, use gradient accumulation")
print("  Use: torch.cuda.empty_cache()")

# 8.2 GRADIENT CHECKING
print("\n8.2 Gradient Checking")
print("-" * 40)

def gradient_check(model, X, y, criterion, epsilon=1e-5):
    """Verify gradients using finite differences"""
    
    # Get analytical gradient
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    
    # Check each parameter
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        analytical_grad = param.grad.clone()
        
        # Compute numerical gradient for first element
        param_flat = param.data.view(-1)
        numerical_grad = torch.zeros_like(param_flat)
        
        for i in range(min(10, len(param_flat))):  # Check first 10 elements
            # +epsilon
            param_flat[i] += epsilon
            output_plus = model(X)
            loss_plus = criterion(output_plus, y)
            
            # -epsilon
            param_flat[i] -= 2 * epsilon
            output_minus = model(X)
            loss_minus = criterion(output_minus, y)
            
            # Numerical gradient
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore
            param_flat[i] += epsilon
        
        # Compare
        analytical_flat = analytical_grad.view(-1)[:10]
        diff = torch.abs(analytical_flat - numerical_grad[:10]).max()
        
        print(f"{name}: max difference = {diff:.2e}")
        if diff > 1e-4:
            print(f"  WARNING: Large gradient difference!")

# Test gradient checking
model = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1))
X = torch.randn(10, 5)
y = torch.randn(10, 1)
criterion = nn.MSELoss()

print("Gradient check results:")
gradient_check(model, X, y, criterion)

# 8.3 DETECTING NAN/INF
print("\n8.3 Detecting NaN/Inf")
print("-" * 40)

def check_for_nan_inf(tensor, name="tensor"):
    """Check if tensor contains NaN or Inf"""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan:
        print(f"WARNING: {name} contains NaN values!")
        return True
    if has_inf:
        print(f"WARNING: {name} contains Inf values!")
        return True
    return False

# Hook to detect NaN/Inf during forward pass
def register_nan_hooks(model):
    """Register hooks to detect NaN/Inf in activations"""
    
    def hook_fn(module, input, output):
        check_for_nan_inf(output, name=f"{module.__class__.__name__} output")
    
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            module.register_forward_hook(hook_fn)

# Test
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
register_nan_hooks(model)

X = torch.randn(4, 10)
output = model(X)
print("NaN/Inf check completed")

# 8.4 VISUALIZING GRADIENTS
print("\n8.4 Visualizing Gradients")
print("-" * 40)

def plot_grad_flow(named_parameters):
    """Plot gradient flow through network"""
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in named_parameters:
        if param.grad is not None and "bias" not in name:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
    
    print("Gradient Flow:")
    for layer, ave_grad, max_grad in zip(layers, ave_grads, max_grads):
        print(f"  {layer:20s}: avg={ave_grad:.6f}, max={max_grad:.6f}")

# Test gradient flow
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

X = torch.randn(32, 10)
y = torch.randn(32, 1)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters())
optimizer.zero_grad()
output = model(X)
loss = criterion(output, y)
loss.backward()

plot_grad_flow(model.named_parameters())

# 8.5 MODEL INSPECTION
print("\n8.5 Model Inspection")
print("-" * 40)

def inspect_model(model):
    """Print detailed model information"""
    
    print("Model Architecture:")
    print(model)
    
    print("\nModel Parameters:")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"  {name:30s}: {param.shape} ({num_params:,} params)")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Memory estimate
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / 1024**2
    
    print(f"\nModel size: {total_size:.2f} MB")

# Test model inspection
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

inspect_model(model)

# 8.6 DEBUGGING TRAINING LOOP
print("\n8.6 Debugging Training Loop")
print("-" * 40)

class DebugTrainer:
    """Training loop with extensive debugging"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Check input
            if check_for_nan_inf(data, "input data"):
                print(f"NaN in input at batch {batch_idx}")
                continue
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Check output
            if check_for_nan_inf(output, "model output"):
                print(f"NaN in output at batch {batch_idx}")
                continue
            
            loss = self.criterion(output, target)
            
            # Check loss
            if check_for_nan_inf(loss, "loss"):
                print(f"NaN in loss at batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Check gradients
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if check_for_nan_inf(param.grad, f"{name} gradient"):
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"NaN in gradients at batch {batch_idx}")
                continue
            
            self.optimizer.step()
            losses.append(loss.item())
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        return sum(losses) / len(losses) if losses else float('inf')

# 8.7 TENSORBOARD LOGGING
print("\n8.7 TensorBoard Logging")
print("-" * 40)

from torch.utils.tensorboard import SummaryWriter

def train_with_tensorboard():
    """Training with TensorBoard logging"""
    
    # Create writer
    writer = SummaryWriter('runs/experiment_1')
    
    # Model and data
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=10
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    for epoch in range(5):
        for i, (batch_X, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # Log to TensorBoard
            global_step = epoch * len(loader) + i
            writer.add_scalar('Loss/train', loss.item(), global_step)
            
            # Log learning rate
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR', lr, global_step)
        
        # Log weights histogram
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}.grad', param.grad, epoch)
    
    writer.close()
    print("TensorBoard logs saved to runs/experiment_1")
    print("View with: tensorboard --logdir=runs")

train_with_tensorboard()

# 8.8 PROFILING
print("\n8.8 Profiling")
print("-" * 40)

def profile_model(model, input_shape, num_iterations=100):
    """Profile model performance"""
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    x = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Profile forward pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations
    fps = 1.0 / avg_time
    
    print(f"Average forward pass time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory allocated: {memory_allocated:.2f} MB")
        print(f"GPU Memory reserved: {memory_reserved:.2f} MB")

# Test profiling
model = nn.Sequential(
    nn.Linear(224*224*3, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
)
print("Profiling model:")
profile_model(model, (1, 224*224*3))

# 8.9 DEBUGGING CHECKLIST
print("\n8.9 Debugging Checklist")
print("-" * 40)

debugging_checklist = """
BEFORE TRAINING:
[ ] Check data shapes (batch_size, channels, height, width)
[ ] Verify labels are in correct range
[ ] Ensure model and data are on same device
[ ] Check model output shape matches target shape
[ ] Verify loss function is appropriate
[ ] Test forward pass with dummy data

DURING TRAINING:
[ ] Monitor loss - should decrease
[ ] Check for NaN/Inf in loss
[ ] Verify gradients are not vanishing/exploding
[ ] Monitor learning rate
[ ] Check validation loss - should track train loss
[ ] Ensure model switches between train/eval modes
[ ] Verify optimizer.zero_grad() is called

AFTER TRAINING:
[ ] Compare train and validation metrics
[ ] Check for overfitting (train much better than val)
[ ] Verify model predictions make sense
[ ] Test on held-out data
[ ] Check confusion matrix
[ ] Visualize predictions
"""

print(debugging_checklist)

# 8.10 COMMON MISTAKES
print("\n8.10 Common Debugging Mistakes")
print("-" * 40)

print("Mistake 1: Not checking data")
print("  Always visualize a few samples before training")

print("\nMistake 2: Training on one batch")
print("  Overfit on one batch first to verify model can learn")

print("\nMistake 3: Not logging enough")
print("  Log everything: loss, metrics, gradients, weights")

print("\nMistake 4: Ignoring warnings")
print("  PyTorch warnings often indicate real problems")

print("\nMistake 5: Not using reproducibility")
print("  Set seeds: torch.manual_seed(), np.random.seed()")


# ============================================================================
# CHUNK 8 EXERCISES
# ============================================================================

print("\n" + "="*80)
print("CHUNK 8 EXERCISES")
print("="*80)

"""
1. Implement gradient flow visualization:
   - Plot gradient magnitudes per layer
   - Color-code by magnitude
   - Track over epochs

2. Create debugging hooks:
   - Register hooks to save activations
   - Detect dead neurons (always zero)
   - Find exploding activations

3. Build training diagnostics dashboard:
   - Real-time loss plotting
   - Gradient statistics
   - Memory usage tracking

4. Implement model comparison:
   - Compare multiple models
   - Track all metrics
   - Generate comparison report

5. Create reproducibility framework:
   - Set all random seeds
   - Save full environment
   - Version control everything
"""

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# ============================================================================
# CHUNK 9: TRANSFER LEARNING BASICS
# Goal: Learn to use pretrained models
# ============================================================================

print("\n" + "="*80)
print("CHUNK 9: TRANSFER LEARNING BASICS")
print("="*80)

import torch
import torch.nn as nn
import torchvision.models as models

# 9.1 LOADING PRETRAINED MODELS
print("\n9.1 Loading Pretrained Models")
print("-" * 40)

# Load pretrained ResNet18
resnet18 = models.resnet18(pretrained=True)
print(f"ResNet18 loaded: {type(resnet18)}")

# Available models in torchvision
print("\nPopular pretrained models:")
print("  - resnet18, resnet34, resnet50, resnet101, resnet152")
print("  - vgg16, vgg19")
print("  - densenet121, densenet169, densenet201")
print("  - mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large")
print("  - efficientnet_b0 through efficientnet_b7")

# 9.2 FEATURE EXTRACTION
print("\n9.2 Feature Extraction")
print("-" * 40)

class FeatureExtractor(nn.Module):
    """Use pretrained model as feature extractor"""
    
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        
        # Remove last layer
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        
        # Freeze pretrained weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Add custom classifier
        num_features = pretrained_model.fc.in_features
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Create feature extractor
base_model = models.resnet18(pretrained=True)
model = FeatureExtractor(base_model, num_classes=10)

# Test
x = torch.randn(4, 3, 224, 224)
output = model(x)
print(f"Feature extractor output: {output.shape}")

# Check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,}")

# 9.3 FINE-TUNING
print("\n9.3 Fine-tuning")
print("-" * 40)

class FineTuneModel(nn.Module):
    """Fine-tune pretrained model"""
    
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.model = pretrained_model
        
        # Replace last layer
        num_features = pretrained_model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except classifier"""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_layers(self, n):
        """Unfreeze last n layers"""
        # Get all layer names
        layers = list(self.model.named_parameters())
        
        # Freeze all first
        for name, param in layers:
            param.requires_grad = False
        
        # Unfreeze last n
        for name, param in layers[-n:]:
            param.requires_grad = True

# Create fine-tune model
base_model = models.resnet18(pretrained=True)
model = FineTuneModel(base_model, num_classes=5)

print("Initial state:")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable:,}")

model.freeze_backbone()
print("\nAfter freezing backbone:")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable:,}")

model.unfreeze_last_n_layers(10)
print("\nAfter unfreezing last 10 layers:")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable:,}")

# 9.4 DISCRIMINATIVE LEARNING RATES
print("\n9.4 Discriminative Learning Rates")
print("-" * 40)

def get_discriminative_params(model, base_lr=1e-3, layer_lr_decay=0.95):
    """Create parameter groups with decreasing learning rates"""
    
    param_groups = []
    
    # Group parameters by layer
    layers = list(model.named_parameters())
    n_layers = len(layers)
    
    for i, (name, param) in enumerate(layers):
        # Learning rate decreases for earlier layers
        lr = base_lr * (layer_lr_decay ** (n_layers - i - 1))
        param_groups.append({'params': param, 'lr': lr})
    
    return param_groups

# Create optimizer with discriminative LR
base_model = models.resnet18(pretrained=True)
model = FineTuneModel(base_model, num_classes=10)

param_groups = get_discriminative_params(model, base_lr=1e-3)
optimizer = torch.optim.Adam(param_groups)

print(f"Created {len(param_groups)} parameter groups")
print(f"First layer LR: {param_groups[0]['lr']:.6f}")
print(f"Last layer LR: {param_groups[-1]['lr']:.6f}")

# 9.5 GRADUAL UNFREEZING
print("\n9.5 Gradual Unfreezing")
print("-" * 40)

def gradual_unfreeze_training():
    """Training with gradual unfreezing"""
    
    # Setup
    base_model = models.resnet18(pretrained=True)
    model = FineTuneModel(base_model, num_classes=5)
    
    # Dummy data
    X = torch.randn(100, 3, 224, 224)
    y = torch.randint(0, 5, (100,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=10
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Train only classifier (5 epochs)
    print("Phase 1: Training classifier only")
    model.freeze_backbone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(2):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Phase 2: Unfreeze last few layers (5 epochs)
    print("\nPhase 2: Unfreezing last layers")
    model.unfreeze_last_n_layers(20)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    
    for epoch in range(2):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Phase 3: Unfreeze all (5 epochs)
    print("\nPhase 3: Fine-tuning all layers")
    model.unfreeze_backbone()
    param_groups = get_discriminative_params(model, base_lr=1e-4)
    optimizer = torch.optim.Adam(param_groups)
    
    for epoch in range(2):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")

gradual_unfreeze_training()

# 9.6 CUSTOM PRETRAINED MODELS
print("\n9.6 Using Custom Pretrained Models")
print("-" * 40)

# Load model from checkpoint
def load_pretrained_custom(checkpoint_path, model_class, num_classes_pretrained):
    """Load custom pretrained model"""
    
    # Create model with pretrained architecture
    model = model_class(num_classes=num_classes_pretrained)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

# Adapt to new task
def adapt_model(pretrained_model, num_classes_new):
    """Adapt pretrained model to new task"""
    
    # Get the classifier layer
    if hasattr(pretrained_model, 'fc'):
        in_features = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Linear(in_features, num_classes_new)
    elif hasattr(pretrained_model, 'classifier'):
        if isinstance(pretrained_model.classifier, nn.Sequential):
            in_features = pretrained_model.classifier[-1].in_features
            pretrained_model.classifier[-1] = nn.Linear(in_features, num_classes_new)
        else:
            in_features = pretrained_model.classifier.in_features
            pretrained_model.classifier = nn.Linear(in_features, num_classes_new)
    
    return pretrained_model

print("Custom pretrained model functions defined")

# 9.7 DATA AUGMENTATION FOR TRANSFER LEARNING
print("\n9.7 Data Augmentation for Transfer Learning")
print("-" * 40)

from torchvision import transforms

# Training transforms (with augmentation)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

print("Transforms for transfer learning created")
print("  Train: Augmentation + Normalization (ImageNet stats)")
print("  Val: Only Resize + Normalization")

# 9.8 TRANSFER LEARNING BEST PRACTICES
print("\n9.8 Transfer Learning Best Practices")
print("-" * 40)

best_practices = """
1. START WITH FROZEN BACKBONE:
   - Train only classifier first
   - Use higher learning rate (1e-3)
   - 5-10 epochs

2. GRADUALLY UNFREEZE:
   - Unfreeze last few layers
   - Lower learning rate (1e-4)
   - 5-10 epochs

3. FINE-TUNE ALL LAYERS:
   - Unfreeze everything
   - Use discriminative learning rates
   - Even lower LR for backbone (1e-5)

4. DATA AUGMENTATION:
   - Use same normalization as pretrained model
   - Add task-specific augmentations
   - More augmentation = better generalization

5. LEARNING RATE:
   - Classifier: 10-100x higher than backbone
   - Use warmup for stability
   - Monitor validation loss closely

6. BATCH SIZE:
   - Smaller batch size often works better
   - 16-32 is common
   - Use gradient accumulation if needed

7. EARLY STOPPING:
   - Monitor validation metrics
   - Patience of 5-10 epochs
   - Save best model
"""

print(best_practices)

# 9.9 COMPLETE TRANSFER LEARNING EXAMPLE
print("\n9.9 Complete Transfer Learning Example")
print("-" * 40)

def transfer_learning_pipeline():
    """Complete transfer learning pipeline"""
    
    # 1. Load pretrained model
    base_model = models.resnet18(pretrained=True)
    model = FineTuneModel(base_model, num_classes=5)
    
    # 2. Freeze backbone
    model.freeze_backbone()
    
    # 3. Prepare data (dummy)
    X_train = torch.randn(100, 3, 224, 224)
    y_train = torch.randint(0, 5, (100,))
    X_val = torch.randn(20, 3, 224, 224)
    y_val = torch.randint(0, 5, (20,))
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=16, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=16, shuffle=False
    )
    
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 4. Stage 1: Train classifier
    print("Stage 1: Training classifier")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    
    for epoch in range(2):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"  Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}")
    
    # 5. Stage 2: Fine-tune all
    print("\nStage 2: Fine-tuning all layers")
    model.unfreeze_backbone()
    param_groups = get_discriminative_params(model, base_lr=1e-4)
    optimizer = torch.optim.Adam(param_groups)
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(5):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"  Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_transfer_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered")
                break
    
    print("\nTransfer learning complete!")
    return model

model = transfer_learning_pipeline()

# 9.10 COMMON MISTAKES
print("\n9.10 Common Transfer Learning Mistakes")
print("-" * 40)

print("Mistake 1: Using wrong learning rate")
print("  Pretrained weights need lower LR than random init")

print("\nMistake 2: Not normalizing correctly")
print("  Must use same normalization as pretrained model")

print("\nMistake 3: Training all layers from start")
print("  Always freeze backbone initially")

print("\nMistake 4: Not checking input size")
print("  Most models expect 224x224 or 299x299")

print("\nMistake 5: Forgetting to unfreeze layers")
print("  Need to gradually unfreeze for best results")


# ============================================================================
# CHUNK 9 EXERCISES
# ============================================================================

print("\n" + "="*80)
print("CHUNK 9 EXERCISES")
print("="*80)

"""
1. Compare transfer learning strategies:
   - Feature extraction only
   - Full fine-tuning
   - Gradual unfreezing
   - Measure final accuracy

2. Implement multi-stage training:
   - Stage 1: Classifier only
   - Stage 2: Last block
   - Stage 3: All layers
   - Track metrics at each stage

3. Create custom backbone:
   - Train model on task A
   - Use as backbone for task B
   - Compare with ImageNet pretrained

4. Implement domain adaptation:
   - Pretrained on natural images
   - Adapt to medical images
   - Use appropriate augmentations

5. Build ensemble of pretrained models:
   - Load multiple architectures
   - Combine predictions
   - Compare with single model
"""








#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
# ============================================================================
# CHUNK 10: MODEL EVALUATION AND METRICS
# Dataset: Food-101 (101 food categories, 1000 images each)
# Goal: Master comprehensive model evaluation beyond simple accuracy
# ============================================================================

# ============================================================================
# PART 1: SETUP AND IMPORTS
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_auc_score,
    top_k_accuracy_score
)
from collections import defaultdict
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# PART 2: DATASET EXPLORATION (EDA)
# ============================================================================

# Download and prepare Food-101 dataset
data_dir = './data/food101'
Path(data_dir).mkdir(parents=True, exist_ok=True)

# Data transforms for training and evaluation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load Food-101 dataset
print("Loading Food-101 dataset...")
train_dataset = datasets.Food101(root=data_dir, split='train', 
                                 transform=train_transform, download=True)
test_dataset = datasets.Food101(root=data_dir, split='test', 
                                transform=eval_transform, download=True)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"\nFirst 10 classes: {train_dataset.classes[:10]}")

# ============================================================================
# EDA: Dataset Statistics
# ============================================================================

def analyze_dataset(dataset, name="Dataset"):
    """Comprehensive dataset analysis"""
    print(f"\n{'='*60}")
    print(f"{name} Analysis")
    print(f"{'='*60}")
    
    # Class distribution
    class_counts = defaultdict(int)
    for _, label in dataset:
        class_counts[label] += 1
    
    counts = np.array(list(class_counts.values()))
    print(f"Total samples: {len(dataset)}")
    print(f"Classes: {len(class_counts)}")
    print(f"Min samples per class: {counts.min()}")
    print(f"Max samples per class: {counts.max()}")
    print(f"Mean samples per class: {counts.mean():.2f}")
    print(f"Std samples per class: {counts.std():.2f}")
    
    return class_counts

train_class_counts = analyze_dataset(train_dataset, "Training Set")
test_class_counts = analyze_dataset(test_dataset, "Test Set")

# ============================================================================
# EDA: Visualize Sample Images
# ============================================================================

def show_samples(dataset, class_names, n_samples=10):
    """Display random samples from dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    # Get random indices
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Denormalize
        img = img.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"{class_names[label]}", fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('food101_samples.png', dpi=150, bbox_inches='tight')
    print("\nSample images saved to 'food101_samples.png'")
    plt.close()

show_samples(test_dataset, train_dataset.classes)

# ============================================================================
# PART 3: MODEL ARCHITECTURE (Transfer Learning)
# ============================================================================

class Food101Classifier(nn.Module):
    """ResNet18 with custom head for Food-101"""
    
    def __init__(self, num_classes=101, pretrained=True):
        super(Food101Classifier, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
        
        # Freeze early layers (optional - usually better to fine-tune all)
        # for param in list(self.backbone.parameters())[:-20]:
        #     param.requires_grad = False
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ============================================================================
# PART 4: TRAINING UTILITIES
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================================
# PART 5: TRAINING LOOP (Quick version for demonstration)
# ============================================================================

# Use subset for faster training (remove this for full training)
USE_SUBSET = True  # Set to False for full dataset
if USE_SUBSET:
    train_indices = list(range(0, 10100))  # 100 samples per class
    test_indices = list(range(0, 2525))    # 25 samples per class
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)
    print(f"\nUsing subset: {len(train_dataset)} train, {len(test_dataset)} test")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                         num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

# Initialize model
model = Food101Classifier(num_classes=101, pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training loop
print("\nTraining model...")
num_epochs = 5  # Increase for better results
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                       optimizer, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), 'food101_model.pth')
print("\nModel saved to 'food101_model.pth'")

# ============================================================================
# PART 6: COMPREHENSIVE EVALUATION METRICS
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation toolkit"""
    
    def __init__(self, model, dataloader, device, class_names):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Collect predictions
        self.all_labels = []
        self.all_predictions = []
        self.all_probabilities = []
        
    def collect_predictions(self):
        """Collect all predictions and labels"""
        print("\nCollecting predictions...")
        self.model.eval()
        
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                self.all_labels.extend(labels.cpu().numpy())
                self.all_predictions.extend(outputs.argmax(1).cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
        
        self.all_labels = np.array(self.all_labels)
        self.all_predictions = np.array(self.all_predictions)
        self.all_probabilities = np.array(self.all_probabilities)
        
        print(f"Collected {len(self.all_labels)} predictions")
    
    def compute_accuracy_metrics(self):
        """Compute various accuracy metrics"""
        print("\n" + "="*60)
        print("ACCURACY METRICS")
        print("="*60)
        
        # Top-1 accuracy
        top1_acc = 100 * np.mean(self.all_predictions == self.all_labels)
        print(f"Top-1 Accuracy: {top1_acc:.2f}%")
        
        # Top-5 accuracy
        top5_acc = top_k_accuracy_score(self.all_labels, 
                                        self.all_probabilities, k=5)
        print(f"Top-5 Accuracy: {100 * top5_acc:.2f}%")
        
        # Top-10 accuracy
        top10_acc = top_k_accuracy_score(self.all_labels, 
                                         self.all_probabilities, k=10)
        print(f"Top-10 Accuracy: {100 * top10_acc:.2f}%")
        
        return {
            'top1': top1_acc,
            'top5': 100 * top5_acc,
            'top10': 100 * top10_acc
        }
    
    def compute_per_class_metrics(self):
        """Compute precision, recall, F1 per class"""
        print("\n" + "="*60)
        print("PER-CLASS METRICS")
        print("="*60)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels, self.all_predictions, average=None, 
            zero_division=0
        )
        
        # Create per-class results
        class_metrics = []
        for i in range(self.num_classes):
            class_metrics.append({
                'class': self.class_names[i],
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            })
        
        # Sort by F1 score
        class_metrics_sorted = sorted(class_metrics, key=lambda x: x['f1'])
        
        print("\nWorst 5 classes (by F1 score):")
        for i, metrics in enumerate(class_metrics_sorted[:5]):
            print(f"{i+1}. {metrics['class']:25s} - "
                  f"F1: {metrics['f1']:.3f}, "
                  f"Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}")
        
        print("\nBest 5 classes (by F1 score):")
        for i, metrics in enumerate(class_metrics_sorted[-5:][::-1]):
            print(f"{i+1}. {metrics['class']:25s} - "
                  f"F1: {metrics['f1']:.3f}, "
                  f"Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}")
        
        # Macro and weighted averages
        print("\nAggregated Metrics:")
        print(f"Macro Precision: {np.mean(precision):.3f}")
        print(f"Macro Recall: {np.mean(recall):.3f}")
        print(f"Macro F1: {np.mean(f1):.3f}")
        
        weighted_f1 = np.sum(f1 * support) / np.sum(support)
        print(f"Weighted F1: {weighted_f1:.3f}")
        
        return class_metrics
    
    def plot_confusion_matrix(self, top_k=20):
        """Plot confusion matrix for top-k classes"""
        print(f"\nPlotting confusion matrix for top {top_k} classes...")
        
        # Find top-k most common classes
        unique, counts = np.unique(self.all_labels, return_counts=True)
        top_classes = unique[np.argsort(counts)[-top_k:]]
        
        # Filter data
        mask = np.isin(self.all_labels, top_classes)
        labels_filtered = self.all_labels[mask]
        preds_filtered = self.all_predictions[mask]
        
        # Compute confusion matrix
        cm = confusion_matrix(labels_filtered, preds_filtered, 
                            labels=top_classes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Absolute counts
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=[self.class_names[i] for i in top_classes],
                   yticklabels=[self.class_names[i] for i in top_classes])
        ax1.set_title(f'Confusion Matrix (Counts) - Top {top_k} Classes')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
        plt.setp(ax1.yaxis.get_majorticklabels(), fontsize=8)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='RdYlGn', 
                   ax=ax2, vmin=0, vmax=1,
                   xticklabels=[self.class_names[i] for i in top_classes],
                   yticklabels=[self.class_names[i] for i in top_classes])
        ax2.set_title(f'Confusion Matrix (Normalized) - Top {top_k} Classes')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
        plt.setp(ax2.yaxis.get_majorticklabels(), fontsize=8)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("Confusion matrix saved to 'confusion_matrix.png'")
        plt.close()
    
    def analyze_confidence(self):
        """Analyze prediction confidence"""
        print("\n" + "="*60)
        print("CONFIDENCE ANALYSIS")
        print("="*60)
        
        # Get max probability for each prediction
        confidences = np.max(self.all_probabilities, axis=1)
        correct_mask = self.all_predictions == self.all_labels
        
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        print(f"Correct predictions:")
        print(f"  Mean confidence: {correct_confidences.mean():.3f}")
        print(f"  Std confidence: {correct_confidences.std():.3f}")
        print(f"  Min confidence: {correct_confidences.min():.3f}")
        print(f"  Max confidence: {correct_confidences.max():.3f}")
        
        print(f"\nIncorrect predictions:")
        print(f"  Mean confidence: {incorrect_confidences.mean():.3f}")
        print(f"  Std confidence: {incorrect_confidences.std():.3f}")
        print(f"  Min confidence: {incorrect_confidences.min():.3f}")
        print(f"  Max confidence: {incorrect_confidences.max():.3f}")
        
        # Plot confidence distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(correct_confidences, bins=50, alpha=0.7, label='Correct', 
               color='green', density=True)
        ax.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect',
               color='red', density=True)
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Density')
        ax.set_title('Prediction Confidence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
        print("\nConfidence distribution saved to 'confidence_distribution.png'")
        plt.close()
    
    def find_worst_predictions(self, n=10):
        """Find most confident wrong predictions"""
        print("\n" + "="*60)
        print(f"TOP {n} MOST CONFIDENT WRONG PREDICTIONS")
        print("="*60)
        
        # Get incorrect predictions
        incorrect_mask = self.all_predictions != self.all_labels
        incorrect_indices = np.where(incorrect_mask)[0]
        
        # Get confidences for incorrect
        incorrect_confidences = np.max(self.all_probabilities[incorrect_mask], 
                                      axis=1)
        
        # Sort by confidence
        worst_indices = incorrect_indices[np.argsort(incorrect_confidences)[::-1][:n]]
        
        for i, idx in enumerate(worst_indices):
            true_label = self.all_labels[idx]
            pred_label = self.all_predictions[idx]
            confidence = np.max(self.all_probabilities[idx])
            
            print(f"\n{i+1}. Confidence: {confidence:.3f}")
            print(f"   True: {self.class_names[true_label]}")
            print(f"   Predicted: {self.class_names[pred_label]}")
            
            # Show top-5 predictions
            top5_idx = np.argsort(self.all_probabilities[idx])[-5:][::-1]
            print(f"   Top-5 predictions:")
            for j, class_idx in enumerate(top5_idx):
                print(f"      {j+1}. {self.class_names[class_idx]}: "
                      f"{self.all_probabilities[idx][class_idx]:.3f}")
    
    def evaluate_all(self):
        """Run all evaluation metrics"""
        self.collect_predictions()
        acc_metrics = self.compute_accuracy_metrics()
        class_metrics = self.compute_per_class_metrics()
        self.plot_confusion_matrix(top_k=20)
        self.analyze_confidence()
        self.find_worst_predictions(n=10)
        
        return {
            'accuracy_metrics': acc_metrics,
            'class_metrics': class_metrics
        }

# ============================================================================
# PART 7: RUN EVALUATION
# ============================================================================

evaluator = ModelEvaluator(model, test_loader, device, 
                          train_dataset.dataset.classes if USE_SUBSET 
                          else train_dataset.classes)
results = evaluator.evaluate_all()

# ============================================================================
# PART 8: MODEL SAVING AND LOADING BEST PRACTICES
# ============================================================================

print("\n" + "="*60)
print("MODEL SAVING AND LOADING")
print("="*60)

# Method 1: Save just state_dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')
print("Saved: model_weights.pth")

# Method 2: Save entire model (not recommended - less portable)
torch.save(model, 'model_complete.pth')
print("Saved: model_complete.pth")

# Method 3: Save checkpoint with training info (best for resuming training)
checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'history': history,
    'best_acc': max(history['val_acc'])
}
torch.save(checkpoint, 'checkpoint.pth')
print("Saved: checkpoint.pth")

# Loading examples
print("\nLoading model (example):")
# Load state dict
loaded_model = Food101Classifier(num_classes=101, pretrained=False)
loaded_model.load_state_dict(torch.load('model_weights.pth'))
loaded_model.to(device)
loaded_model.eval()
print("Model loaded successfully")

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
loaded_model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")

# ============================================================================
# PART 9: INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """Production-ready inference pipeline"""
    
    def __init__(self, model_path, class_names, device):
        self.device = device
        self.class_names = class_names
        self.model = Food101Classifier(num_classes=len(class_names), 
                                      pretrained=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image, top_k=5):
        """Predict single image with top-k results"""
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
        
        # Get top-k predictions
        probs, indices = probabilities[0].topk(top_k)
        
        results = []
        for prob, idx in zip(probs, indices):
            results.append({
                'class': self.class_names[idx.item()],
                'probability': prob.item(),
                'confidence': f"{100 * prob.item():.2f}%"
            })
        
        return results
    
    def predict_batch(self, images, batch_size=32):
        """Predict batch of images efficiently"""
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack([self.transform(img) for img in batch])
            batch_tensor = batch_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            predictions = probabilities.argmax(1).cpu().numpy()
            confidences = probabilities.max(1)[0].cpu().numpy()
            
            for pred, conf in zip(predictions, confidences):
                all_results.append({
                    'class': self.class_names[pred],
                    'confidence': conf
                })
        
        return all_results

print("\n" + "="*60)
print("INFERENCE PIPELINE EXAMPLE")
print("="*60)

# Create inference pipeline
pipeline = InferencePipeline('model_weights.pth', 
                           train_dataset.dataset.classes if USE_SUBSET 
                           else train_dataset.classes,
                           device)

# Test on sample from test set
sample_img, sample_label = test_dataset[0] if not USE_SUBSET else test_dataset.dataset[test_indices[0]]
# Convert tensor back to PIL for inference demo
from PIL import Image
img_np = sample_img.numpy().transpose(1, 2, 0)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_np = std * img_np + mean
img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
pil_img = Image.fromarray(img_np)

# Predict
predictions = pipeline.predict(pil_img, top_k=5)
print(f"\nTrue label: {train_dataset.dataset.classes[sample_label] if USE_SUBSET else train_dataset.classes[sample_label]}")
print("\nTop-5 predictions:")
for i, pred in enumerate(predictions):
    print(f"{i+1}. {pred['class']:25s} - {pred['confidence']}")

# ============================================================================
# PART 10: KEY TAKEAWAYS AND NEXT STEPS
# ============================================================================

print("\n" + "="*60)
print("KEY TAKEAWAYS")
print("="*60)
print("""
1. ACCURACY IS NOT ENOUGH
   - Top-1 accuracy hides per-class performance
   - Use top-5, top-10 for multi-class problems
   - Per-class metrics reveal true model behavior

2. CONFIDENCE MATTERS
   - High confidence + wrong = dangerous in production
   - Calibrate confidence (temperature scaling, etc.)
   - Set thresholds for uncertain predictions

3. CONFUSION MATRIX IS YOUR FRIEND
   - Shows where model gets confused
   - Reveals similar class problems
   - Guides data augmentation strategy

4. PER-CLASS ANALYSIS IS CRITICAL
   - Some classes always fail (imbalanced data, hard examples)
   - Focus training on weak classes
   - Consider class weights in loss

5. MODEL SAVING BEST PRACTICES
   - Save state_dict, not entire model (portability)
   - Save optimizer state for training resumption
   - Keep best model + last model checkpoints

6. PRODUCTION INFERENCE
   - Batch processing for efficiency
   - Handle edge cases (corrupted images, wrong format)
   - Monitor confidence distribution in production
   - Set up A/B testing infrastructure

NEXT STEPS:
- Implement ensemble predictions (average 3-5 models)
- Add test-time augmentation (flip, crop variations)
- Try different architectures (EfficientNet, Vision Transformer)
- Implement uncertainty quantification (MC Dropout)
- Deploy with model monitoring (drift detection)
""")
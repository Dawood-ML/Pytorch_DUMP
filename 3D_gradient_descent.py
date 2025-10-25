import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# --- 1. Define the Loss Function and its Gradient ---

# A simple quadratic loss surface (a stretched bowl shape)
def loss_function(w1, w2):
    """
    Calculates the loss for given weights w1 and w2.
    The minimum is at (0, 0).
    """
    return w1**2 + 3 * w2**2

def gradient(w1, w2):
    """
    Calculates the gradient (partial derivatives) of the loss function.
    This tells us the direction of steepest ascent.
    """
    return np.array([2 * w1, 6 * w2])

# --- 2. Set up the 3D Plot ---

# Create the figure and a 3D subplot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25) # Make room for sliders

# Generate a grid of points to plot the surface
w1_vals = np.linspace(-4, 4, 100)
w2_vals = np.linspace(-4, 4, 100)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = loss_function(W1, W2)

# Plot the loss surface with transparency
ax.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.6, edgecolor='none', rstride=5, cstride=5)

# Add contour lines on the 'floor' to help visualize the shape
ax.contour(W1, W2, Z, zdir='z', offset=Z.min() - 2, cmap='viridis', alpha=0.7)

# Set labels for the axes
ax.set_xlabel('Weight 1 ($w_1$)', fontsize=12, labelpad=10)
ax.set_ylabel('Weight 2 ($w_2$)', fontsize=12, labelpad=10)
ax.set_zlabel('Loss', fontsize=12, labelpad=10)
ax.set_title('Interactive 3D Gradient Descent', fontsize=18, fontweight='bold', pad=20)
ax.view_init(elev=30., azim=-60) # Set an initial viewing angle

# --- 3. Implement Gradient Descent and Plot the Path ---

# Initial parameters
initial_learning_rate = 0.25
initial_steps = 20
start_point = np.array([20, -20]) # The starting position for the algorithm

def calculate_path(start_w, learning_rate, num_steps):
    """
    Performs the gradient descent algorithm to find the path down the surface.
    """
    w = start_w.copy()
    # Path stores (w1, w2, loss) for each step
    path = [np.append(w, loss_function(w[0], w[1]))]
    
    for _ in range(num_steps):
        grad = gradient(w[0], w[1])
        w = w - learning_rate * grad # The core gradient descent update rule
        path.append(np.append(w, loss_function(w[0], w[1])))
        
    return np.array(path)

# Calculate the initial path
initial_path = calculate_path(start_point, initial_learning_rate, initial_steps)

# Plot the initial path as a red line with dots
path_plot, = ax.plot(initial_path[:, 0], initial_path[:, 1], initial_path[:, 2], 
                     'r-o', linewidth=2, markersize=4, zorder=10, 
                     label='Descent Path')
# Plot the start and end points
start_plot, = ax.plot([initial_path[0, 0]], [initial_path[0, 1]], [initial_path[0, 2]],
                      'g*', markersize=15, zorder=11, label='Start')
end_plot, = ax.plot([initial_path[-1, 0]], [initial_path[-1, 1]], [initial_path[-1, 2]],
                    'kX', markersize=10, zorder=11, label='End')
ax.legend()


# --- 4. Create Interactive Sliders ---

# Define the axes for the sliders
ax_lr = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_steps = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Create the slider objects
slider_lr = Slider(
    ax=ax_lr,
    label='Learning Rate (α)',
    valmin=0.01,
    valmax=0.33, # A high learning rate to show divergence
    valinit=initial_learning_rate,
)

slider_steps = Slider(
    ax=ax_steps,
    label='Number of Steps',
    valmin=1,
    valmax=50,
    valinit=initial_steps,
    valstep=1 # Steps must be integers
)


# --- 5. Define the Update Function for Interactivity ---

def update(val):
    """
    This function is called every time a slider is moved.
    """
    # Get current values from the sliders
    current_lr = slider_lr.val
    current_steps = int(slider_steps.val)
    
    # Recalculate the gradient descent path
    new_path = calculate_path(start_point, current_lr, current_steps)
    
    # Update the plot data for the path line
    path_plot.set_data_3d(new_path[:, 0], new_path[:, 1], new_path[:, 2])
    
    # Update the end point marker
    end_plot.set_data_3d([new_path[-1, 0]], [new_path[-1, 1]], [new_path[-1, 2]])
    
    # Redraw the plot to show the changes
    fig.canvas.draw_idle()

# Register the update function with the sliders
slider_lr.on_changed(update)
slider_steps.on_changed(update)

# --- 6. Show the Plot ---
print("Plot window opened. Close the window to exit the script.")
print("Drag the sliders to see how the learning rate and number of steps affect the path.")
plt.show()
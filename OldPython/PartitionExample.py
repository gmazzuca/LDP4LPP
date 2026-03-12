import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

def plot_discrete_partition(matrix):
    """
    Plots a matrix as a 3D plane partition (stacks of cubes) with correct styling.
    """
    matrix = np.array(matrix)
    rows, cols = matrix.shape
    
    # Handle empty partition case
    if matrix.size == 0 or matrix.max() == 0:
        print("Partition is empty.")
        return

    max_height = matrix.max()
    
    # 1. Create the Voxel Grid
    # voxels[x, y, z] is True if a cube exists at that coordinate
    # We swap rows/cols mapping to match standard matrix visualization (row=x, col=y)
    voxels = np.zeros((rows, cols, max_height), dtype=bool)
    
    # 2. Create the Color Grid
    # This must be a 4D array: (x, y, z, 4) for RGBA values
    colors = np.zeros((rows, cols, max_height, 4), dtype=float)
    
    # Colormap
    cmap = plt.cm.viridis
    
    # Fill voxels and colors
    for r in range(rows):
        for c in range(cols):
            h = matrix[r, c]
            if h > 0:
                # Fill the boolean grid
                voxels[r, c, :h] = True
                
                # Determine color based on stack height (normalized)
                col = cmap(h / max_height)
                
                # Assign this color to all cubes in this stack
                colors[r, c, :h] = col

    # 3. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use Orthographic projection for the classic "Plane Partition" look
    ax.set_proj_type('ortho')
    
    # Render voxels
    # edgecolors='k' adds the black outlines to cubes
    ax.voxels(voxels, facecolors=colors, edgecolors='k', linewidth=0.5, shade=True)
    
    # 4. Adjust View for "Corner" Perspective
    # elev=35, azim=45 is the standard isometric-like view
    ax.view_init(elev=35, azim=45)
    
    # Formatting
    ax.set_xlabel('Rows (M)')
    ax.set_ylabel('Cols (N)')
    ax.set_zlabel('Height')
    ax.set_title(f'Discrete Plane Partition (Max Height={max_height})')
    
    # Invert axes to match matrix layout (optional, usually looks better for partitions)
    ax.invert_xaxis() 
    
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# Using the partition from Figure 1 caption in the paper
example_partition = [
    [8, 5, 4, 4, 2, 1],
    [6, 5, 3, 3, 2, 1],
    [4, 3, 2, 2, 1, 0],
    [4, 2, 1, 1, 0, 0],
    [3, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0]
]

plot_discrete_partition(example_partition)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the 2D Gaussian function
def gaussian_2d(x, y, sigma_x, sigma_y, x0=0, y0=0):
    """
    2D Gaussian function with different widths along x and y axes
    
    Parameters:
    - x, y: coordinate arrays
    - sigma_x: standard deviation along x-axis (width)
    - sigma_y: standard deviation along y-axis (width)
    - x0, y0: center coordinates (default at origin)
    
    Returns:
    - z: Gaussian values (peak intensity = 1)
    """
    z = np.exp(-((x - x0)**2 / (sigma_x**2) + (y - y0)**2 / (sigma_y**2)))
    return z

# Parameters
sigma_x = 67.5e-3  # Width along x-axis, temporal axis
sigma_y = 5.1674 # Width along y-axis, spectral axis

y0 = 803.1577

# Create grid
x = np.linspace(-2.5*sigma_x, 2.5*sigma_x, 100)
y = np.linspace(-2.5*sigma_y, 2.5*sigma_y, 100)+y0
X, Y = np.meshgrid(x, y)

# Calculate Gaussian values
Z = gaussian_2d(X, Y, sigma_x, sigma_y, y0=y0)

# Create 3D plot
fig = plt.figure()

# 3D surface plot
ax1 = fig.add_subplot(111, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Wavelength (nm)')
ax1.set_zlabel('Intensity')
ax1.set_title(f'3D Gaussian (σ_x={sigma_x}, σ_y={sigma_y})')
fig.colorbar(surf, ax=ax1, shrink=0.5)

plt.tight_layout()
plt.savefig('gaussian_3d.png', dpi=150, bbox_inches='tight')
plt.show()

# Verify projections
print(f"Projection on X-axis (y=0): Gaussian with σ={sigma_x}")
print(f"Projection on Y-axis (x=0): Gaussian with σ={sigma_y}")


# Plot the projections
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))

# X-axis projection (slice at y=0)
z_x = gaussian_2d(x, 0, sigma_x, sigma_y)
ax3.plot(x, z_x, 'b-', linewidth=2)
ax3.set_xlabel('Time (ps)')
ax3.set_ylabel('Field Intensity')
ax3.set_title(f'X-axis projection (σ_x={sigma_x})')
ax3.grid(True, alpha=0.3)

# Y-axis projection (slice at x=0)
z_y = gaussian_2d(0, y, sigma_x, sigma_y, y0=y0)
ax4.plot(y, z_y, 'r-', linewidth=2)
ax4.set_xlabel('Wavelength (nm)')
ax4.set_ylabel('Intensity')
ax4.set_title(f'Y-axis projection (σ_y={sigma_y})')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gaussian_projections.png', dpi=150, bbox_inches='tight')

plt.show()
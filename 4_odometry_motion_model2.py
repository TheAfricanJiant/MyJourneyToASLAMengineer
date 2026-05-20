import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow, Ellipse
from mpl_toolkits.mplot3d import Axes3D

def gaussian_belief3d(x_pos, y_pos, robot_mean, sigma, theta):
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    image_path = os.path.join(assets_dir, "4_odometry_motion_model2.png")

    # Compute 2D Gaussian belief
    def gaussian(pos, mean, sigma):
        return np.exp(-0.5 * ((pos - mean) / sigma) ** 2)
    
    bx = gaussian(x_pos, robot_mean[0], sigma[0])
    by = gaussian(y_pos, robot_mean[1], sigma[1])
    
    belief = np.outer(bx, by)      # Correct order: x then y
    belief /= belief.sum()

    # ====================== Figure Setup ======================
    fig = plt.figure(figsize=(16, 8))
    
    # ==================== LEFT: 3D Gaussian Surface ====================
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    
    X, Y = np.meshgrid(x_pos, y_pos)
    Z = belief.T * 1500   # Increased scale for better visibility
    
    # 3D Surface - Light color (almost white) with color gradient
    surf = ax3d.plot_surface(X, Y, Z, cmap='YlGnBu', alpha=0.85, 
                            linewidth=0, antialiased=True)

    ax3d.set_title('3D Gaussian Belief (Position Uncertainty)', fontsize=14)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Probability Density')

    # Black grid lines on 3D plot
    ax3d.xaxis._axinfo['grid'].update(color='black', linestyle='--', linewidth=0.6, alpha=0.7)
    ax3d.yaxis._axinfo['grid'].update(color='black', linestyle='--', linewidth=0.6, alpha=0.7)
    ax3d.zaxis._axinfo['grid'].update(color='black', linestyle='--', linewidth=0.6, alpha=0.7)
    
    fig.colorbar(surf, ax=ax3d, shrink=0.5, label='Probability')

    # ==================== RIGHT: 2D Top-down View ====================
    ax2d = fig.add_subplot(1, 2, 2)
    
    # Heatmap - light background
    im = ax2d.imshow(belief.T, origin='lower',
                     extent=[x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()],
                     cmap='bone', alpha=0.9)
    
    # Black contours (centered correctly now)
    ax2d.contour(X, Y, belief.T, levels=15, colors='black', linewidths=1.1, alpha=0.75)
    
    plt.colorbar(im, ax=ax2d, label='Probability Density')

    # ==================== Robot ====================
    robot_radius = 3.8
    
    # White filled circle with black border
    circle = Circle(robot_mean, robot_radius, facecolor='white', edgecolor='black', 
                    linewidth=2.5, alpha=0.95, zorder=5)
    ax2d.add_patch(circle)

    # Orientation arrow
    arrow_len = 8.0
    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)
    arrow = Arrow(robot_mean[0], robot_mean[1], dx, dy, 
                  width=4.5, facecolor='black', edgecolor='white', 
                  linewidth=1.5, zorder=6)
    ax2d.add_patch(arrow)

    # Uncertainty ellipses (black)
    for n, alpha in zip([1, 2, 3], [0.9, 0.6, 0.35]):
        ell = Ellipse(xy=robot_mean, width=sigma[0]*2*n, height=sigma[1]*2*n,
                      angle=0, edgecolor='black', facecolor='none', 
                      linewidth=2, alpha=alpha, linestyle='--', zorder=4)
        ax2d.add_patch(ell)

    # Styling
    ax2d.set_title('Top-down View: Robot + Belief Contours', fontsize=14)
    ax2d.set_xlabel('X Position')
    ax2d.set_ylabel('Y Position')
    ax2d.grid(True, color='black', linestyle='--', alpha=0.5)
    
    # Info text
    info = f"Robot @ ({robot_mean[0]:.1f}, {robot_mean[1]:.1f})\n" \
           f"Heading: {np.degrees(theta):.0f}°\n" \
           f"σ = {sigma[0]:.1f}"
    
    ax2d.text(4, 92, info, color='black', fontsize=11, fontweight='bold',
              bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()


# ========================== Usage ==========================
x_pos = np.arange(0, 100)
y_pos = np.arange(0, 100)

gaussian_belief3d(
    x_pos, y_pos,
    robot_mean=(50, 50),
    sigma=(5.5, 5.5),
    theta=np.pi/4
)
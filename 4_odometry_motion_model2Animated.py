import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# ====================== Parameters ======================
x_pos = np.arange(0, 100)
y_pos = np.arange(0, 100)

sigma = (5.5, 5.5)
theta = np.pi / 4          # fixed orientation for simplicity

# Robot movement path (you can change this)
num_frames = 60
t = np.linspace(0, 2*np.pi, num_frames)
robot_x = 40 + 25 * np.cos(t)          # circular movement
robot_y = 45 + 20 * np.sin(t) * 0.6    # slightly elliptical

# ====================== Setup Figure ======================
fig = plt.figure(figsize=(16, 8))

ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax2d = fig.add_subplot(1, 2, 2)

# Pre-create objects for updating
X, Y = np.meshgrid(x_pos, y_pos)
surf = None
contour = None
circle = None
arrow = None
ellipses = []

def init():
    global surf, contour, circle, arrow, ellipses
    
    # Clear previous
    ax3d.clear()
    ax2d.clear()
    
    # 3D Plot
    ax3d.set_title('3D Gaussian Belief', fontsize=14)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Probability')
    
    # Black grid
    ax3d.xaxis._axinfo['grid'].update(color='black', linestyle='--', alpha=0.6)
    ax3d.yaxis._axinfo['grid'].update(color='black', linestyle='--', alpha=0.6)
    ax3d.zaxis._axinfo['grid'].update(color='black', linestyle='--', alpha=0.6)
    
    # 2D Plot
    ax2d.set_title('Top-down View: Robot + Belief', fontsize=14)
    ax2d.set_xlabel('X Position')
    ax2d.set_ylabel('Y Position')
    ax2d.grid(True, color='black', linestyle='--', alpha=0.5)
    
    return []

def update(frame):
    global surf, contour, circle, arrow, ellipses
    
    # Current robot position
    mean = (robot_x[frame], robot_y[frame])
    
    # Compute Gaussian
    def g(pos, mu, sig):
        return np.exp(-0.5 * ((pos - mu) / sig) ** 2)
    
    bx = g(x_pos, mean[0], sigma[0])
    by = g(y_pos, mean[1], sigma[1])
    belief = np.outer(bx, by)
    belief /= belief.sum()
    Z = belief.T * 1500

    # Clear previous artists
    ax3d.clear()
    ax2d.clear()
    
    # === 3D Plot ===
    surf = ax3d.plot_surface(X, Y, Z, cmap='YlGnBu', alpha=0.85, linewidth=0, antialiased=True)
    
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Probability')
    ax3d.set_title('3D Gaussian Belief (Position Uncertainty)')
    
    # === 2D Plot ===
    ax2d.imshow(belief.T, origin='lower',
                extent=[0, 99, 0, 99], cmap='bone', alpha=0.9)
    
    ax2d.contour(X, Y, belief.T, levels=15, colors='black', linewidths=1.1, alpha=0.75)
    
    # Robot (White fill + Black border)
    circle = Circle(mean, 3.8, facecolor='white', edgecolor='black', 
                    linewidth=2.5, zorder=5)
    ax2d.add_patch(circle)
    
    # Arrow
    arrow_len = 8.0
    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)
    arrow = Arrow(mean[0], mean[1], dx, dy, width=4.5,
                  facecolor='black', edgecolor='white', linewidth=1.5, zorder=6)
    ax2d.add_patch(arrow)
    
    # Uncertainty ellipses
    for n, alpha in zip([1, 2, 3], [0.9, 0.6, 0.35]):
        ell = Ellipse(xy=mean, width=sigma[0]*2*n, height=sigma[1]*2*n,
                      angle=0, edgecolor='black', facecolor='none',
                      linewidth=2, alpha=alpha, linestyle='--', zorder=4)
        ax2d.add_patch(ell)
    
    ax2d.set_xlim(0, 99)
    ax2d.set_ylim(0, 99)
    
    return []

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, 
                    interval=80, blit=False, repeat=True)

# Save as GIF
print("Saving animation as GIF... (this may take 10-30 seconds)")
ani.save('robot_gaussian_belief.gif', writer=PillowWriter(fps=12))

print("✅ Animation saved successfully as 'robot_gaussian_belief.gif'")
plt.show()
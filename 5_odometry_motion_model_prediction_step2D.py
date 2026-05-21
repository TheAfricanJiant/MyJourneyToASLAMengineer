import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# ====================== Parameters ======================
x_pos = np.arange(0, 100)
y_pos = np.arange(0, 100)

# Starting values - Much sharper Gaussian
mean = np.array([35.0, 45.0])
sigma = np.array([2.8, 2.8])        # Much thinner at the start (~1/15 of plane)
theta = np.pi / 4

num_frames = 85
np.random.seed(42)

# Robot path
t = np.linspace(0, 4*np.pi, num_frames)
true_x = 35 + 32 * (t / (4*np.pi))
true_y = 45 + 18 * np.sin(t * 0.65)

# ====================== Figure Setup ======================
fig = plt.figure(figsize=(18, 9))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1.35], height_ratios=[1, 0.28])  # 2D plot bigger

ax3d = fig.add_subplot(gs[0, 0], projection='3d')
ax2d = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])

X, Y = np.meshgrid(x_pos, y_pos)

def gaussian_belief(mean, sigma):
    def gaussian(pos, mu, sig):
        return np.exp(-0.5 * ((pos - mu) / sig) ** 2)
    
    bx = gaussian(x_pos, mean[0], sigma[0])
    by = gaussian(y_pos, mean[1], sigma[1])
    belief = np.outer(bx, by)
    belief /= belief.sum()
    return belief

def update(frame):
    global mean, sigma, theta
    
    ax3d.clear()
    ax2d.clear()
    ax_table.clear()
    
    delta_rot1 = delta_trans = delta_rot2 = 0.0
    noisy_rot1 = noisy_trans = noisy_rot2 = 0.0
    
    if frame > 0:
        prev = np.array([true_x[frame-1], true_y[frame-1]])
        curr = np.array([true_x[frame], true_y[frame]])
        
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        delta_trans = np.hypot(dx, dy)
        delta_rot1 = np.arctan2(dy, dx) - theta
        delta_rot2 = 0.0
        
        noisy_rot1 = delta_rot1 + np.random.normal(0, 0.085)
        noisy_trans = delta_trans + np.random.normal(0, 0.14)
        noisy_rot2 = delta_rot2 + np.random.normal(0, 0.085)
        
        mean[0] += noisy_trans * np.cos(theta + noisy_rot1)
        mean[1] += noisy_trans * np.sin(theta + noisy_rot1)
        theta += noisy_rot1 + noisy_rot2
        
        sigma += 0.38
    
    belief = gaussian_belief(mean, sigma)
    Z = belief.T * 2800   # Increased multiplier for taller peak
    
    # ==================== 3D Plot ====================
    surf = ax3d.plot_surface(X, Y, Z, cmap='YlGnBu', alpha=0.85, 
                            linewidth=0, antialiased=True)
    
    ax3d.set_title('3D Gaussian Belief (Prediction Step)', fontsize=14)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Probability Density')
    ax3d.view_init(elev=35, azim=-50)
    
    ax3d.xaxis._axinfo['grid'].update(color='black', linestyle='--', linewidth=0.6, alpha=0.7)
    ax3d.yaxis._axinfo['grid'].update(color='black', linestyle='--', linewidth=0.6, alpha=0.7)
    ax3d.zaxis._axinfo['grid'].update(color='black', linestyle='--', linewidth=0.6, alpha=0.7)
    
    # ==================== 2D Plot - Larger area, smaller robot ====================
    im = ax2d.imshow(belief.T, origin='lower',
                     extent=[0, 99, 0, 99], cmap='bone', alpha=0.9)
    
    ax2d.contour(X, Y, belief.T, levels=15, colors='black', linewidths=1.1, alpha=0.75)
    
    # Smaller robot
    robot_radius = 2.6
    circle = Circle(mean, robot_radius, facecolor='white', edgecolor='black', 
                    linewidth=2.5, alpha=0.95, zorder=5)
    ax2d.add_patch(circle)

    arrow_len = 6.0
    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)
    arrow = Arrow(mean[0], mean[1], dx, dy, width=3.5, facecolor='black', 
                  edgecolor='white', linewidth=1.5, zorder=6)
    ax2d.add_patch(arrow)

    for n, alpha in zip([1, 2, 3], [0.9, 0.6, 0.35]):
        ell = Ellipse(xy=mean, width=sigma[0]*2*n, height=sigma[1]*2*n,
                      angle=0, edgecolor='black', facecolor='none', 
                      linewidth=2, alpha=alpha, linestyle='--', zorder=4)
        ax2d.add_patch(ell)

    ax2d.set_title('Top-down View: Robot + Belief Contours', fontsize=14)
    ax2d.set_xlabel('X Position')
    ax2d.set_ylabel('Y Position')
    ax2d.grid(True, color='black', linestyle='--', alpha=0.5)
    ax2d.set_xlim(0, 99)
    ax2d.set_ylim(0, 99)
    
    # Table
    ax_table.axis('off')
    table_text = f"""
Frame: {frame:2d}    |    Pose: ({mean[0]:.1f}, {mean[1]:.1f}, {np.degrees(theta):.0f}°)

δ_rot1   = {np.degrees(delta_rot1):6.2f}°   (noisy: {np.degrees(noisy_rot1):6.2f}°)
δ_trans  = {delta_trans:6.2f}    (noisy: {noisy_trans:6.2f})
δ_rot2   = {np.degrees(delta_rot2):6.2f}°   (noisy: {np.degrees(noisy_rot2):6.2f}°)

σ = ({sigma[0]:.2f}, {sigma[1]:.2f})     ← Uncertainty grows (Prediction Step)
"""
    ax_table.text(0.5, 0.5, table_text, ha='center', va='center', fontsize=12.5,
                  bbox=dict(facecolor='lightyellow', alpha=0.95, edgecolor='black', boxstyle='round,pad=0.8'))
    
    return []

# ====================== Animation ======================
ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

print("Saving animation...")
ani.save('robot_odometry_prediction_step.gif', writer=PillowWriter(fps=10))

print("✅ Animation saved successfully!")
plt.show()
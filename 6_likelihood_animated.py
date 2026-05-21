import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from map import GridMap

# ========================== Gaussian ==========================
def gaussian(dist, sigma=2.5):
    return np.exp(-0.5 * (dist / sigma) ** 2)

# ========================== Setup ==========================
env = GridMap(width=100, height=100)
map_grid = env.get_map()

# Fixed robot pose
robot_x, robot_y = 45.0, 35.0
robot_theta = np.pi / 4

# ========================== Plot Setup ==========================
fig, axs = plt.subplots(1, 2, figsize=(17, 8))

# Left: Map + Scanning Beams
axs[0].imshow(map_grid, origin='lower', cmap='gray', alpha=0.85)
axs[0].set_title('(a) Environment Map + Scanning Laser Beams')
axs[0].set_xlim(0, 99)
axs[0].set_ylim(0, 99)

# Right: Dynamic Active Likelihood (only currently seen endpoints)
axs[1].imshow(map_grid, origin='lower', cmap='gray', alpha=0.7)
axs[1].set_title('(b) Active Gaussians - Currently Seen Obstacles')
dynamic_im = axs[1].imshow(np.zeros_like(map_grid, dtype=float), 
                          origin='lower', extent=[0, 99, 0, 99],
                          cmap='hot', alpha=0.0)  # will be updated

# Fixed Robot
for ax in [axs[0], axs[1]]:
    ax.add_patch(Circle((robot_x, robot_y), 3.5, facecolor='lime', edgecolor='black', linewidth=2.5))
    ax.arrow(robot_x, robot_y, 6*np.cos(robot_theta), 6*np.sin(robot_theta),
             head_width=3, color='white', linewidth=2.5)

rays = []
active_gaussians = None

# ========================== Animation Function ==========================
def update(frame):
    global rays, active_gaussians
    
    # Clear old rays
    for ray in rays:
        ray.remove()
    rays.clear()
    
    # Sweep
    sweep_offset = frame * 0.18
    
    num_rays = 36
    max_range = 50
    hit_points = []   # store endpoints for Gaussians
    
    for i in range(num_rays):
        angle = robot_theta + i * (2 * np.pi / num_rays) + sweep_offset
        hit = max_range
        for r in np.linspace(0, max_range, 200):
            px = robot_x + r * np.cos(angle)
            py = robot_y + r * np.sin(angle)
            if env.is_occupied(px, py):
                hit = r
                hit_points.append((px, py))
                break
        
        ex = robot_x + hit * np.cos(angle)
        ey = robot_y + hit * np.sin(angle)
        
        ray = axs[0].plot([robot_x, ex], [robot_y, ey], 
                         color='red', linewidth=2.8, alpha=0.92)[0]
        rays.append(ray)
    
    # === Dynamic Active Gaussians on Right Plot ===
    if active_gaussians is not None:
        active_gaussians.remove()
    
    field = np.zeros_like(map_grid, dtype=float)
    sigma = 2.8
    step = 1
    for hx, hy in hit_points:
        xx, yy = np.meshgrid(np.arange(0, 100, step), np.arange(0, 100, step))
        dists = np.sqrt((xx - hx)**2 + (yy - hy)**2)
        field += gaussian(dists, sigma)
    
    if len(hit_points) > 0:
        field = field / field.max()
    
    active_gaussians = axs[1].imshow(field, origin='lower', extent=[0, 99, 0, 99],
                                    cmap='hot', alpha=0.85)
    
    return rays + [active_gaussians]

# Create animation
ani = FuncAnimation(fig, update, frames=90, interval=70, blit=False, repeat=True)

plt.tight_layout()

# Save as GIF
print("Saving animation as 'active_laser_gaussians.gif'...")
ani.save('active_laser_gaussians.gif', writer='pillow', fps=18)
print("✅ GIF saved successfully!")

plt.show()
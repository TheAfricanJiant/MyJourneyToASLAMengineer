import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from map import GridMap

# ========================== Gaussian ==========================
def gaussian(dist, sigma=1.8):
    return np.exp(-0.5 * (dist / sigma) ** 2)

# ========================== Setup ==========================
env = GridMap(width=100, height=100)
map_grid = env.get_map()

robot_x, robot_y = 45.0, 35.0
robot_theta = np.pi / 4

# ========================== Sensor Readings ==========================
num_rays = 36
max_range = 50
sensor_readings = []
ray_angles = []

for i in range(num_rays):
    angle = robot_theta + i * (2 * np.pi / num_rays)
    ray_angles.append(angle)
    hit = max_range
    for r in np.linspace(0, max_range, 250):
        if env.is_occupied(robot_x + r*np.cos(angle), robot_y + r*np.sin(angle)):
            hit = r
            break
    sensor_readings.append(hit)

# ========================== Likelihood Field ==========================
print("Computing Likelihood Field...")
step = 1
xx, yy = np.meshgrid(np.arange(0, 100, step), np.arange(0, 100, step))
likelihood = np.zeros_like(xx, dtype=float)

for y in range(0, 100):
    for x in range(0, 100):
        if map_grid[y, x] == 1:  # obstacle
            dists = np.sqrt((xx - x)**2 + (yy - y)**2)
            likelihood += gaussian(dists, sigma=2.2)

likelihood = likelihood / likelihood.max()

# ========================== Plot ==========================
fig, axs = plt.subplots(1, 2, figsize=(17, 8))

# Left: Map + Measurements (Improved visibility)
axs[0].imshow(map_grid, origin='lower', cmap='gray', alpha=0.85)
for i, dist in enumerate(sensor_readings):
    angle = ray_angles[i]
    ex = robot_x + dist * np.cos(angle)
    ey = robot_y + dist * np.sin(angle)
    # Bright red beams with better visibility
    axs[0].plot([robot_x, ex], [robot_y, ey], 
                color='red', linewidth=2.5, alpha=0.9)

axs[0].add_patch(Circle((robot_x, robot_y), 3.5, facecolor='lime', edgecolor='black', linewidth=2.5))
axs[0].arrow(robot_x, robot_y, 6*np.cos(robot_theta), 6*np.sin(robot_theta),
             head_width=3, color='white', linewidth=2.5)

axs[0].set_title('(a) Environment Map + Laser Beams')
axs[0].set_xlim(0, 99)
axs[0].set_ylim(0, 99)

# Right: Likelihood Field
im = axs[1].imshow(likelihood, origin='lower', extent=[0, 99, 0, 99],
                   cmap='bone', alpha=0.95)
axs[1].set_title('(b) Likelihood Field')
plt.colorbar(im, ax=axs[1])

# Robot on likelihood field too
axs[1].add_patch(Circle((robot_x, robot_y), 3.5, facecolor='lime', edgecolor='black', linewidth=2.5))
axs[1].arrow(robot_x, robot_y, 6*np.cos(robot_theta), 6*np.sin(robot_theta),
             head_width=3, color='white', linewidth=2.5)

plt.tight_layout()
plt.show()
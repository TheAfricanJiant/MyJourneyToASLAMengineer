import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from map import GridMap

# ========================== Setup ==========================
env = GridMap(width=100, height=100)
map_grid = env.get_map()

# Robot true position
robot_x, robot_y = 45.0, 35.0
robot_theta = np.pi / 4

# Simulate laser scanner (12 rays)
num_rays = 12
max_range = 50.0
sensor_readings = []

for i in range(num_rays):
    angle = robot_theta + i * (2*np.pi / num_rays) # evenly spaced rays around the robot
    # Simple ray casting simulation
    for r in np.linspace(0, max_range, 200):
        x = robot_x + r * np.cos(angle)
        y = robot_y + r * np.sin(angle)
        if env.is_occupied(x, y):
            sensor_readings.append(r)
            break
    else:
        sensor_readings.append(max_range)

# ====================== Plot ======================
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Left: Map + Robot + Sensor Rays
axs[0].imshow(map_grid, origin='lower', cmap='gray', alpha=0.7)
axs[0].set_title('Environment Map + Sensor Measurements')

# Plot sensor rays
for i, dist in enumerate(sensor_readings):
    angle = robot_theta + i * (2*np.pi / num_rays)
    ex = robot_x + dist * np.cos(angle)
    ey = robot_y + dist * np.sin(angle)
    axs[0].plot([robot_x, ex], [robot_y, ey], 'r-', alpha=0.6, linewidth=1.2)

# Robot
axs[0].add_patch(Circle((robot_x, robot_y), 3, color='red', alpha=0.9))
axs[0].arrow(robot_x, robot_y, 6*np.cos(robot_theta), 6*np.sin(robot_theta),
             head_width=3, color='yellow', length_includes_head=True)

axs[0].set_xlim(0, 99)
axs[0].set_ylim(0, 99)
axs[0].grid(True, alpha=0.3)

# Right: Likelihood Field (to be filled in next step)
axs[1].imshow(map_grid, origin='lower', cmap='gray', alpha=0.3)
axs[1].set_title('Likelihood Field p(z | x)  [Coming Next]')

plt.tight_layout()
plt.show()
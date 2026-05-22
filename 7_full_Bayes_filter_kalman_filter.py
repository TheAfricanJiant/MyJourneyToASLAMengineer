import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from map import GridMap

# ====================== Parameters ======================
np.random.seed(42)
env = GridMap(width=100, height=100)

num_frames = 80
max_range  = 50
num_rays   = 36

x_pos = np.arange(0, 100, dtype=float)
y_pos = np.arange(0, 100, dtype=float)
X, Y  = np.meshgrid(x_pos, y_pos)   # X[i,j]=j (x-coord), Y[i,j]=i (y-coord)
map_grid = env.get_map()

# ====================== Wall-avoiding path ======================
# Each segment was hand-checked against every wall in map.py.
# The robot follows this ground-truth path; Bayes filter estimates it.
waypoints = np.array([
    [15, 30],   # start – left side, below horizontal wall (y=20-24)
    [15, 10],   # bottom-left corner
    [70, 10],   # along bottom  (y=10 clears all internal walls)
    [90, 10],   # bottom-right  (y=10 clears obstacle at y=15-19,x=80-84)
    [90, 55],   # right corridor (x=90 > wall at x=80-84; y=55 < wall y=60-64)
    [90, 90],   # top-right     (col 90 outside wall grid[60:65, 60:90])
    [50, 90],   # top middle
    [15, 90],   # top-left      (x=15 clears obstacle at x=20-29)
    [15, 50],   # back to start area
], dtype=float)

seg_lens = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
cum_dist  = np.concatenate([[0], np.cumsum(seg_lens)])
t_wp      = cum_dist / cum_dist[-1]
t_path    = np.linspace(0, 1, num_frames)
true_x    = np.interp(t_path, t_wp, waypoints[:, 0])
true_y    = np.interp(t_path, t_wp, waypoints[:, 1])

# Initial pose
mean  = np.array([true_x[0], true_y[0]])
sigma = np.array([3.0, 3.0])
theta = np.arctan2(true_y[1] - true_y[0], true_x[1] - true_x[0])

# ====================== Likelihood field ======================
print("Precomputing Likelihood Field...")
likelihood_field = np.zeros((100, 100), dtype=float)
for wy in range(100):
    for wx in range(100):
        if map_grid[wy, wx] == 1:
            dists = np.sqrt((X - wx)**2 + (Y - wy)**2)
            likelihood_field += np.exp(-0.5 * (dists / 2.2)**2)
likelihood_field /= likelihood_field.max()

# ====================== Figure ======================
fig = plt.figure(figsize=(21, 10))
gs  = gridspec.GridSpec(2, 3, width_ratios=[1.1, 1.1, 1.2], height_ratios=[1, 0.22])
ax3d    = fig.add_subplot(gs[0, 0], projection='3d')
ax2d    = fig.add_subplot(gs[0, 1])
ax_live = fig.add_subplot(gs[0, 2])
ax_tbl  = fig.add_subplot(gs[1, :])

# ── helpers ─────────────────────────────────────────────────────────────────

def gaussian_belief(mean, sigma):
    """
    Uses np.outer (same as the reference file).
    belief[i, j] = bx[i] * by[j]  →  i indexes x, j indexes y.
    Always use belief.T when displaying, so the peak sits at
    (row = mean[1], col = mean[0]) → visual (x=mean[0], y=mean[1]). ✓
    """
    bx = np.exp(-0.5 * ((x_pos - mean[0]) / sigma[0]) ** 2)
    by = np.exp(-0.5 * ((y_pos - mean[1]) / sigma[1]) ** 2)
    belief = np.outer(bx, by)
    belief /= belief.sum() + 1e-12
    return belief

def compute_sensor_likelihood(pos, theta):
    total = 0.0
    for i in range(num_rays):
        angle = theta + i * (2 * np.pi / num_rays)
        hit = max_range
        for r in np.linspace(0, max_range, 100):
            px = pos[0] + r * np.cos(angle)
            py = pos[1] + r * np.sin(angle)
            if env.is_occupied(px, py):
                hit = r
                break
        hx = int(np.clip(pos[0] + hit * np.cos(angle), 0, 99))
        hy = int(np.clip(pos[1] + hit * np.sin(angle), 0, 99))
        total += likelihood_field[hy, hx]      # grid is [y, x]
    return total / num_rays + 0.01

# ── animation ───────────────────────────────────────────────────────────────

def update(frame):
    global mean, sigma, theta

    ax3d.clear(); ax2d.clear(); ax_live.clear(); ax_tbl.clear()

    if frame > 0:
        # ── PREDICTION: apply noisy odometry, uncertainty grows ──
        dx = true_x[frame] - true_x[frame - 1]
        dy = true_y[frame] - true_y[frame - 1]
        delta_trans = np.hypot(dx, dy)
        delta_rot1  = np.arctan2(dy, dx) - theta

        noisy_rot1  = delta_rot1 + np.random.normal(0, 0.085)
        noisy_trans = delta_trans + np.random.normal(0, 0.14)
        noisy_rot2  = np.random.normal(0, 0.085)

        mean[0] += noisy_trans * np.cos(theta + noisy_rot1)
        mean[1] += noisy_trans * np.sin(theta + noisy_rot1)
        theta    += noisy_rot1 + noisy_rot2
        sigma    += 0.33                        # uncertainty grows after prediction

        # ── CORRECTION: sensor match → shrink uncertainty ──
        lik    = compute_sensor_likelihood(mean, theta)
        sigma *= (0.87 + 0.13 * lik)           # high match → more shrinkage
        sigma  = np.maximum(sigma, 1.7)

    belief = gaussian_belief(mean, sigma)       # belief[i,j]: i=x index, j=y index
    bT     = belief.T                           # bT[i,j]:    i=y index, j=x index  ✓

    # ── subplot 1: 3D surface ────────────────────────────────────────────────
    ax3d.plot_surface(X, Y, bT * 2800, cmap='YlGnBu', alpha=0.85)
    ax3d.set_title('3D Belief  (Prediction + Correction)', fontsize=13)
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Prob')
    ax3d.view_init(elev=35, azim=-50)

    # ── subplot 2: 2-D heatmap + robot  (no ellipses) ───────────────────────
    # bT displayed with origin='lower': row i → y=i from bottom, col j → x=j.
    # bT peaks at row=mean[1], col=mean[0]  → visual (x=mean[0], y=mean[1]) ✓
    ax2d.imshow(bT, origin='lower', extent=[0, 99, 0, 99], cmap='bone', alpha=0.9)
    ax2d.contour(X, Y, bT, levels=12, colors='black', linewidths=0.8, alpha=0.7)

    ax2d.add_patch(Circle(mean, 3.0, facecolor='lime', edgecolor='black',
                           linewidth=2.5, zorder=5))
    ax2d.add_patch(Arrow(mean[0], mean[1], 7*np.cos(theta), 7*np.sin(theta),
                          width=3.5, facecolor='red', edgecolor='black',
                          linewidth=2, zorder=6))

    ax2d.set_title('Belief heatmap — Gaussian centred on robot')
    ax2d.set_xlim(0, 99); ax2d.set_ylim(0, 99)

    # ── subplot 3: map + sensor beams + hit-point Gaussians ─────────────────
    ax_live.imshow(map_grid, origin='lower', cmap='gray', alpha=0.75)

    hit_points = []
    for i in range(num_rays):
        angle = theta + i * (2 * np.pi / num_rays)
        hit   = max_range
        for r in np.linspace(0, max_range, 150):
            px = mean[0] + r * np.cos(angle)
            py = mean[1] + r * np.sin(angle)
            if env.is_occupied(px, py):
                hit = r
                break
        hx = mean[0] + hit * np.cos(angle)
        hy = mean[1] + hit * np.sin(angle)
        hit_points.append((hx, hy))
        ax_live.plot([mean[0], hx], [mean[1], hy],
                     color='red', linewidth=2.0, alpha=0.85)

    # Gaussian blobs at each sensor hit point
    field = np.zeros((100, 100), dtype=float)
    for hx, hy in hit_points:
        dists = np.sqrt((X - hx)**2 + (Y - hy)**2)
        field += np.exp(-0.5 * (dists / 2.8)**2)
    if field.max() > 0:
        field /= field.max()

    # field[i,j] peaks at i=hy (row=y), j=hx (col=x) → no .T needed
    ax_live.imshow(field, origin='lower', extent=[0, 99, 0, 99],
                   cmap='hot', alpha=0.75)

    ax_live.add_patch(Circle(mean, 3.0, facecolor='lime', edgecolor='black',
                              linewidth=2.5, zorder=5))
    ax_live.add_patch(Arrow(mean[0], mean[1], 7*np.cos(theta), 7*np.sin(theta),
                             width=3.5, facecolor='red', edgecolor='black',
                             linewidth=2, zorder=6))
    ax_live.set_title('Sensor beams  (beams stop at walls → correction step)')

    # ── info table ───────────────────────────────────────────────────────────
    ax_tbl.axis('off')
    lik = compute_sensor_likelihood(mean, theta)
    ax_tbl.text(
        0.5, 0.5,
        f"Frame {frame:2d}  |  Pose ({mean[0]:.1f}, {mean[1]:.1f}, {np.degrees(theta):.0f}°)\n"
        f"σ = ({sigma[0]:.2f}, {sigma[1]:.2f})   Likelihood = {lik:.3f}   "
        f"← correction shrinks σ when beams match the map well",
        ha='center', va='center', fontsize=12.5,
        bbox=dict(facecolor='lightyellow', alpha=0.95,
                  edgecolor='black', boxstyle='round,pad=1'))
    return []

ani = FuncAnimation(fig, update, frames=num_frames, interval=130, blit=False)
ani.save('full_bayes_filter_final.gif', writer=PillowWriter(fps=8))
print("✅  Saved as full_bayes_filter_final.gif")
plt.show()
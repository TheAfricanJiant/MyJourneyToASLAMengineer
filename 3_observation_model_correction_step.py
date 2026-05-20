#Author: Tambu Precious Takum 
#Date: 2026-05-19
# Exercise 3 - Correction Step (slide 10-11)

# Full Bayes Filter:
# Prediction:  bel_bar(xt) = ∫ p(xt|ut,xt-1) * bel(xt-1) dxt-1
# Correction:  bel(xt)     = η * p(zt|xt) * bel_bar(xt)

import os
import numpy as np
import matplotlib.pyplot as plt

assets_dir = "assets"
os.makedirs(assets_dir, exist_ok=True)

positions = np.arange(100)

def gaussian_belief(postions, mean, sigma):
    belief = np.exp(-0.5 * ((positions - mean) / sigma) ** 2)
    return belief / belief.sum()

def predict(belief, move_command, motion_noise):
    new_belief = np.zeros(len(belief))
    for x_prev in range(len(belief)):
        if belief[x_prev] < 1e-10: 
            continue # skip positions with near -zero probability 
        
        expected_pos = x_prev + move_command
        motion_model = gaussian_belief(positions, expected_pos, motion_noise)
        new_belief += belief[x_prev] * motion_model
    return new_belief / new_belief.sum()

def correct(belief, sensor_reading, sensor_noise):
    """
    Correction step.
    belief        : predicted belief bel_bar(xt)
    sensor_reading: what the sensor measured (zt)
    sensor_noise  : how accurate the sensor is (sigma)

    p(zt | xt) - likelihood of seeing zt if robot is at xt
    """
    
    # Sensor model: how likely is this reading at each position?
    sensor_model = gaussian_belief(positions, sensor_reading, sensor_noise)
    
    # Correction: multiply belief by the sensor likelihood
    # η is just the normalization constant
    new_belief = sensor_model * belief
    return new_belief / new_belief.sum() # Normalize

# ---- Run the full predict → correct cycle ----

# Start confident at position 20
belief = gaussian_belief(positions, mean=20, sigma=3)

plt.figure(figsize=(12, 8))
plt.plot(positions, belief, label="Start (postion 20)", color="blue")

# --- Step 1: Move but dont sense ---
belief = predict(belief, move_command=20, motion_noise=5)
plt.plot(positions, belief, label="After predict (belief spreads)", 
         color="orange", linestyle="--")

# --- Step 2: Sensor sees position 42 ---
# Robot is roughly at 40 but sensor is noisy
belief = correct(belief, sensor_reading=42, sensor_noise=3)
plt.plot(positions, belief, label="After correct (sensor at 42)", 
         color="red")

# --- Step 3: Move again ---
belief = predict(belief, move_command=20, motion_noise=5)
plt.plot(positions, belief, label="After 2nd predict (spreads again)", 
         color="purple", linestyle="--")


# --- Step 4: Sensor sees position 63 ---
belief = correct(belief, sensor_reading=63, sensor_noise=3)
plt.plot(positions, belief, label="After 2nd correct (sensor at 63)", 
         color="green")

### comment this out to see what happens when the sensor breaks and gives a wrong reading.
## --- Step 5: Move again ---
belief = predict(belief, move_command=20, motion_noise=5)
plt.plot(positions, belief, label="After 3rd predict (spreads again)",
         color="brown", linestyle="--")

## --- Step 6: Sensor sees position 96 after it got broken ---
belief = correct(belief, sensor_reading=96, sensor_noise=3)
plt.plot(positions, belief, label="After 3rd correct (brokensensor at 96)", 
         color="black")

###end

plt.title("Full Bayes Filter: Predict → Correct → Predict → Correct\n"
          "bel(xt) = η * p(zt|xt) * bel_bar(xt)")
plt.xlabel("Position")
plt.ylabel("Probability")
plt.legend()
#plt.savefig(os.path.join(assets_dir, "3_observation_model_correction_step_sensor.png"))
plt.savefig(os.path.join(assets_dir, "3_observation_model_correction_step_broken_sensor.png"))
plt.show()

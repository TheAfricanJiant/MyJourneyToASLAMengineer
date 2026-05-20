#Author: Tambu Precious Takum 
#Date: 2026-05-19
#Prediction step (slides 10-11)

#About: Implement the motion model. Move your robot forward with Gaussian noise added. Watch how the belief spreads out over time.

############### Theory ###############
# Below is the prediciton step of the Bayes filter. There corrction step is next
# bel(xt​)=∫p(xt​∣ut​,xt−1​) *  bel(xt−1​)dxt−1
#          [motion model]  [previous belief]
# - The motion model p(xt | ut, xt-1) describes how the robot's state changes in response to control inputs (ut) and the previous state (xt-1). This is great and was possisble due to markov assumption.

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
    """
    Prediction step - move the robot and spread the belief.
    belief       : current belief distribution
    move_command : how far we want to move (u)
    motion_noise : how noisy our motion is (sigma)
    """
    new_belief = np.zeros(len(belief))
    
    for x_prev in range(len(belief)):
        if belief[x_prev] < 1e-10: 
            continue # skip positions with near -zero probability 
        
        #For each previous postion, wherer could we end up?
        # This is (p(x_t | u_t, x_{t-1}) - the motion model
        expected_pos = x_prev + move_command
        motion_model = gaussian_belief(positions, expected_pos, motion_noise)
        new_belief += belief[x_prev] * motion_model
    
    return new_belief / new_belief.sum() # Normalize



    # Start with robot fairly confident it's at position 20
belief = gaussian_belief(positions, mean=20, sigma=3)

plt.figure(figsize=(10, 6))
plt.plot(positions, belief, label="Before move", color="blue")

# Command the robot to move 20 steps forward
# with some motion noise
belief = predict(belief, move_command=20, motion_noise=3)
plt.plot(positions, belief, label="After 1 move", color="orange")

# Move again
belief = predict(belief, move_command=20, motion_noise=3)
plt.plot(positions, belief, label="After 2 moves", color="red")

#move again 
belief = predict(belief=belief, move_command=20, motion_noise=3)
plt.plot(positions, belief, label="After 3 moves", color="green")

plt.title("Prediction Step - Belief spreads with each move")
plt.xlabel("Position")
plt.ylabel("Probability")
plt.legend()
plt.savefig(os.path.join(assets_dir, "2_motion_model_prediction_step1.png"))
plt.show()
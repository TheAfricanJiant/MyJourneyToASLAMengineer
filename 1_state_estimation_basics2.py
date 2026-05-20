#Author: Tambu Precious Takum 
#Date: 2026-05-19
#State Estimation basics (slides 2-3)


# Run that first. You should see a flat line - meaning the robot has no idea where it is.
#Then try this - place the robot at a known position and represent that as a Gaussian belief. 

import os
import numpy as np
import matplotlib.pyplot as plt

assets_dir = "assets"
os.makedirs(assets_dir, exist_ok=True)

def gaussian_belief(positions, mean, sigma):
    belief = np.exp(-0.5 * ((positions - mean) / sigma) ** 2) #bell curve
    return belief / belief.sum() #Normalize so that it sums to 1
    
    #Notes
    # - So unlike the previous example, this code assumes robot is arround the mean
    #   but not just unceertain about it.  It forms a bel curve around the mean.  
 
positions = np.arange(100)

belief = gaussian_belief(positions=positions, mean=50, sigma=5) 
#sigma is measuiring how far each position is from the mean. 

plt.plot(positions, belief)
plt.title("Belief - Robot thinks it's near position 50")
plt.xlabel("Position")
plt.ylabel("Belief")
plt.savefig(os.path.join(assets_dir, "1_state_estimation_basics2.png"))
plt.show()
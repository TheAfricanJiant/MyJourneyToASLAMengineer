#Author: Tambu Precious Takum 
#Date: 2026-05-19
#State Estimation basics (slides 2-3)

################### Some Theory ##############################

#The goal of state estimation is to estimate the state x of a system given observations Z and controls u. 
#Goal p(x | z, u)
#Simulate a simple 1D robot on a line. Write a function that represents a belief as a probability distribution over positions.
#bel(Xt) = p(Xt | Z1:t, U1:t)

import os
import numpy as np
import matplotlib.pyplot as plt

assets_dir = "assets"
os.makedirs(assets_dir, exist_ok=True)

# Represent the world as 100 positions (0 to 99) 
positions = np.arange(100)

# Start with a uniform belief - robot could be anywhere
belief = np.ones(100) / 100 # all positions equally likely

# Plot the initial belief
plt.plot(positions, belief)
plt.title("Initial Belief - Uniform Distribution")
plt.xlabel("position")
plt.ylabel("probability")
plt.savefig(os.path.join(assets_dir, "1_state_estimation_basics1.png"))
plt.show()
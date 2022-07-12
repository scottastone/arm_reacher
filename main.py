from arm import Arm
import numpy as np
from tqdm import trange

N_EPISODES = 500
STEPS_PER_EPISODE = 1000

arm = Arm("assets/arm.xml")

for episode in trange(N_EPISODES):
    #print(f"Episode: {episode}")
    for step in range(STEPS_PER_EPISODE):
        obs, reward, done, info = arm.step((0.9, -1))
        #print(f"{arm._physics.time():>16.4}: {info}")
        if done:
            arm.reset()
            break

    arm.reset()
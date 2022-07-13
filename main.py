from arm import Arm
import numpy as np
from tqdm import trange

N_EPISODES = 500
STEPS_PER_EPISODE = 100

arm = Arm(xml_path="assets/arm.xml")

for episode in trange(N_EPISODES):
    #print(f"Episode: {episode}")
    for step in range(STEPS_PER_EPISODE):
        obs, reward, done, info = arm.step(np.random.uniform(-1, 1, 2))
        #print(f"{arm._physics.time():>16.4}: {info}")
        #arm.render()
        if done:
            arm.reset()
            break

    arm.reset()
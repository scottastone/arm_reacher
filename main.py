from arm import Arm
import matplotlib.pyplot as plt
import numpy as np
import tqdm

arm = Arm("assets/arm.xml")

for _ in tqdm.tqdm(range(100000)):
    obs, reward, done, info = arm.step((1,-.1))
    #print(f"{arm._physics.named.data.geom_xpos['finger', :2]}, {arm._target_contact}")
    #print(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}")
    #arm.render()
from arm import Arm
import matplotlib.pyplot as plt
import numpy as np

arm = Arm("arm.xml")

for _ in range(25):
    obs, reward, done, info = arm.step(np.ones((2,)))
    print(f"{arm._physics.named.data.geom_xpos['finger', :2]}, {arm._target_contact}")
    #print(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}")
    arm.render()
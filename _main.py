import dm_control
from dm_control import suite
from dm_control import mujoco
import matplotlib.pyplot as plt
import numpy as np



def get_observation(physics):
    pass

def reset(physics):
    with physics.reset_context():
        initialize_episode(physics)

    return physics.position()

def initialize_episode(physics):
    # set the timestep TODO: find out how?
    # physics.named.model.timestep = 1/60

    # set the target size
    physics.named.model.geom_size['target', 0] = .015

    # set target position
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(.05, .20)
    physics.named.model.geom_pos['target', 'x'] = .05 #radius * np.sin(angle)
    physics.named.model.geom_pos['target', 'y'] = .05 #radius * np.cos(angle)

physics = mujoco.Physics.from_xml_path("arm.xml")
pos = reset(physics)

print(f"after reset: {physics.named.model.geom_pos['target', 'x']}, {physics.named.model.geom_pos['target', 'y']}, {pos}")
pixels = physics.render(height=320, width=640, camera_id="fixed")

# step
physics.step()

# check if we got a reward
get_observation(physics)

plt.imshow(pixels)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from rorlkit.torch.data_management.rosbag_data import PosesGetter


train_dataset_info = {
    'bag_files': ['../dataset/bagfile_stirfry/stir_fry_slow.bag',
                  '../dataset/bagfile_stirfry/stir_fry_fast.bag',
                  '../dataset/bagfile_stirfry/stir_fry_maxspeed.bag'],
    'topics': ['/cartesian/left_hand/reference',
               '/cartesian/right_hand/reference']
}

pg = PosesGetter(**train_dataset_info)
x, y, z = pg.get_xyz(1, 0)
# print(x)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'gray')
# colors = np.arange(0, len(x))
ax.scatter3D(x, y, z, s=6, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plt.plot(y, 'g^')
plt.show()

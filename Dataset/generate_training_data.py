import generate_surface
import numpy as np
from scipy.interpolate import interp2d
from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt

# Create a random pattern
SubsetSize = 256 # must be even
I, _, _, _, _ = generate_surface.rough_surface(sigma=25, l0=3, dx=1, m=np.max([SubsetSize, SubsetSize]),
                                               type='gaussian')
I += np.abs(np.min(I))  # Convert to non-negative values (i.e., potential intensity maps)
I_interp = np.pad(I, 2)        # pad 0s

# Mesh of data
xp = np.arange(0, SubsetSize)+2
xxp = np.arange(0, SubsetSize + 4)
[Xp_subset, Yp_subset] = np.meshgrid(xp, xp)
xp0 = np.arange(1, SubsetSize/8 + 1, 1/8) + 3
xxp0 = np.arange(0, SubsetSize // 8 + 4)

# Random displacement
f = np.random.rand(SubsetSize//8+4, SubsetSize//8+4) * 2 - 1
g = np.random.rand(SubsetSize//8+4, SubsetSize//8+4) * 2 - 1

# Interpolate random displacement map
interp_x = interp2d(xxp0, xxp0, f)
disp_x = interp_x(xp0, xp0)
interp_y = interp2d(xxp0, xxp0, g)
disp_y = interp_y(xp0, xp0)

# Set boundaries to 0
disp_x[0:2, :] = 0
disp_x[:, 0:2] = 0
disp_x[SubsetSize - 2: SubsetSize - 1, :] = 0
disp_x[:, SubsetSize - 2: SubsetSize - 1] = 0
disp_y[0:2, :] = 0
disp_y[:, 0:2] = 0
disp_y[SubsetSize - 2: SubsetSize - 1, :] = 0
disp_y[:, SubsetSize - 2: SubsetSize - 1] = 0


# Create displaced pattern by interpolation/re-mapping
x = (Xp_subset.transpose() - disp_x).flatten()
y = (Yp_subset.transpose() - disp_y).flatten()
I_disp = np.reshape(map_coordinates(I_interp, [x.ravel(), y.ravel()], order=3, mode='nearest'), (SubsetSize, SubsetSize))




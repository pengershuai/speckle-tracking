import generate_surface
import numpy as np
from scipy.interpolate import interp2d
from scipy.ndimage import map_coordinates
import os
import pandas as pd
import csv
import speckle_quality as sq

import matplotlib.pyplot as plt

ifIntensity = 1  # Set to 0 if only want displacement, 1 if want to apply intensity
displacement = 3  # max pixel displacement

SubsetSize = 256  # must be even
imgSize = 10  # number of images to generate
n = 10  # number of iterations for each image

# surface parameters
sigma = 10
l0 = 4
dx = 6

# Create directories
try:
    os.mkdir("Data")
except OSError as error:
    print(error)

# List of file names
files = []

for img in range(1, imgSize + 1):
    # Create a random pattern
    I, _, _, _, _ = generate_surface.rough_surface(sigma=sigma, l0=l0, dx=dx, m=np.max([SubsetSize, SubsetSize]),
                                                   type='gaussian')
    I += np.abs(np.min(I))  # Convert to non-negative values (i.e., potential intensity maps)
    I_interp = np.pad(I, 2)        # pad 0s

    # Output speckle information
    # speckle_size = sq.autocorrelation(I, dx, report=True)['FWHM']
    # sq.visibility_std(I, int(speckle_size * 3), report=True)

    # Mesh of data
    xp = np.arange(0, SubsetSize)+2
    xxp = np.arange(0, SubsetSize + 4)
    [Xp_subset, Yp_subset] = np.meshgrid(xp, xp)
    xp0 = np.arange(1, SubsetSize/8 + 1, 1/8) + 3
    xxp0 = np.arange(0, SubsetSize // 8 + 4)

    for l in range(1, n + 1):
        # Random displacement
        f = np.random.rand(SubsetSize//8+4, SubsetSize//8+4) * 2 * displacement - displacement
        g = np.random.rand(SubsetSize//8+4, SubsetSize//8+4) * 2 * displacement - displacement

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

        # Write to .csv
        path = 'Data/'
        name_ref = 'Ref' + str(img) + '_' + str(l) + '.csv'
        name_def = 'Def' + str(img) + '_' + str(l) + '.csv'
        name_dispx = 'Dispx' + str(img) + '_' + str(l) + '.csv'
        name_dispy = 'Dispy' + str(img) + '_' + str(l) + '.csv'

        pd.DataFrame(I).to_csv(path + name_ref, header=None, index=None)
        pd.DataFrame(I_disp).to_csv(path + name_def, header=None, index=None)
        pd.DataFrame(disp_x).to_csv(path + name_dispx, header=None, index=None)
        pd.DataFrame(disp_y).to_csv(path + name_dispy, header=None, index=None)

        # if applying intensity
        if ifIntensity:
            # Random intensity
            h = (np.random.rand(SubsetSize, SubsetSize) + 1) / 2
            I_int = np.multiply(I_disp, h)
            name_int = 'Int' + str(img) + '_' + str(l) + '.csv'
            pd.DataFrame(I_int).to_csv(path + name_int, header=None, index=None)
            files.append([name_ref, name_int, name_dispx, name_dispy])
        else:
            files.append([name_ref, name_def, name_dispx, name_dispy])

# write annotation file
with open('Data/Train_annotations.csv', 'w') as csvfile:
# with open('Data/Test_annotations.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(files)
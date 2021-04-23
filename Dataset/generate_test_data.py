import numpy as np
import generate_surface
import generate_star_displacement
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates
import speckle_quality


# image size
m = 512
n = 512

# Star displacement field (positive x shift means down, positive y shift means to the right)
d = generate_star_displacement.star_displacement(m, n, amplitude=3)
#d = 2 * np.ones((m, m))   # test
#plt.imshow(d, cmap='RdBu_r'), plt.show()
GT = Image.fromarray(d)
GT.save('displacement.tif')


# Original speckle pattern
z, dx, RMS, PSD, ACF = generate_surface.rough_surface(sigma=10, l0=2, dx=1, m=np.max([m, n])+2, type='gaussian')



z = z[0:m, 0:n]
z = z + np.abs(np.min(z))
im = Image.fromarray(z)
im.save('Star1.tif')

ac = speckle_quality.autocorrelation(z, 1, report=True)
v, v_map = speckle_quality.visibility_std(z, int(np.ceil(ac['FWHM']))*3, report=True)

# Displaced speckle pattern
x, y = np.meshgrid(np.arange(0, z.shape[0]), np.arange(0, z.shape[1]))
x_new = (x.transpose() - d).flatten()
y_new = (y.transpose() - 0).flatten()
z_new = np.reshape(map_coordinates(z, [x_new.ravel(), y_new.ravel()], order=3, mode='nearest'), (m, n))
z_new_noT = z_new
im_new_noT = Image.fromarray(z_new_noT)
im_new_noT.save('Star2.tif')

Tx, Ty = np.meshgrid(np.linspace(1, 0.1, n), np.linspace(1, 0.1, m))  # Additional attenuation
z_new = z_new*Ty
# __, T = speckle_quality.darkfield_transmission(z,z_new,50)
# z_new = z_new / T
im_new = Image.fromarray(z_new)
im_new.save('Star3.tif')




# m = 501
# n = 2000
# d = generate_star_displacement.star_displacement(m, n, amplitude=0.5)
# im = Image.open('Noiseless_frames/Star1.tif')
# z_test = np.array(im)
# z_test = z_test[0:m, 0:m]
# d = d[0:m, 0:m]
#
# x, y = np.meshgrid(np.arange(0, m), np.arange(0, m))
# im = Image.fromarray(z_test)
# im.save('Real_speckle/Star1.tif')
# x_new = (x.transpose() - d).flatten()
# y_new = (y.transpose() - 0).flatten()
# z_new = np.reshape(map_coordinates(z_test, [x_new.ravel(), y_new.ravel()], order=3, mode='nearest'), (m, m))
# im_new = Image.fromarray(z_new)
# im_new.save('Real_speckle/Star2.tif')

#plt.imshow(z), plt.colorbar(), plt.show()
#plt.imshow(z_test), plt.colorbar(), plt.show()
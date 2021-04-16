import numpy as np
from numpy.fft import fftshift as fftshift
import matplotlib.pyplot as plt


def star_displacement(n, m, amplitude=0.5):
    """
    Create a star pattern displacement field
    :param n: int
        pixels in x
    :param m: int
        pixels in y
    :param amplitude: float
        sinus wave amplitude, i.e., maximum displacement
    :return: 2d (nxm) array of float
        displacement map
    """

    # displacement data
    data = np.zeros((n, m))

    xi = np.arange(0, n) - n/2

    # If in physical units rather than pixel
    # xi = np.arange(0, n) * dx
    # xi = fftshift(xi)  # shift 0 to center of array
    # ind0 = np.where(xi == 0)[0][0]  # find pixel index of central pixel
    # xi[0:ind0] = xi[0:ind0] - xi[ind0 - 1] - xi[ind0 + 1]  # adjust to -max/2:max/2


    for i in range(0, data.shape[1]):
        # wavelength ranges from 10 pixel to 150/2000*m at the end
        wavelength = 10 + (150 / 2000 * m - 10) / m * i
        #wavelength = 30 + (450 / 2000 * m - 30) / m * i
        data[:, i] = amplitude * np.cos(2 * np.pi * xi / wavelength)

    #plt.imshow(data, cmap='RdBu_r'), plt.colorbar(), plt.show()

    return data

if __name__ == '__main__':
    d = star_displacement(500, 2000, 1)
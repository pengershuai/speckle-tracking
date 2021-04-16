import numpy as np
from numpy.fft import fftshift, ifftshift, ifft2
import speckle_quality
import timeit

def cart2pol(x, y):
    """
    Convert cartesian to polar coordinates
    :param x: array
        position values (x)
    :param y: array
        position values (x)
    :return: 2 arrays
        rho: radius values
        phi: angle values
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """
    Convert polar to cartesian coordinates
    :param rho: array
        radius values
    :param phi: array
        angle values
    :return: 2 arrays
        x: x position values
        y: y position values
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def rough_surface(sigma, l0, dx, m, type='gaussian', fr=0):
    """
    Generate a rough surface
    :param sigma: float
        RMS height [m]
    :param l0: float
        correlation length [m]
    :param dx: float
        pixel size [m]
    :param m: int
        number of pixels in each dimension
    :param type: string
        type of autocovariance/power spectral density used for the surface: exponential or gaussian (default).
    :param fr: float, optional
        frequency cutoff parameter [1/m]
    :return z: mxm 2darray of float
        surface height [m] map
    :return dx: float
        pixel size [m]
    :return RMS: float
        RMS height [m]
    :return PSD: dict
        Prescribed power spectral density
    "return ACOV: dict
        Prescribed autocovariance function
    """

    # Check if type is existing
    if not type.lower() in ['gaussian', 'exponential']:
        raise NotImplementedError('Unknown PSD type! Please use exponential or gaussian (default).')

    # Ensure even number of pixels
    m = int(m)
    if np.mod(m, 2):
        m = m - 1

    # Size of the image volume
    Lx = m * dx

    # position vector
    x = np.arange(0, m) * dx
    x = fftshift(x)  # shift 0 to center of array
    ind0 = np.where(x == 0)[0][0]  # find pixel index of central pixel
    x[0:ind0] = x[0:ind0] - x[ind0 - 1] - x[ind0 + 1]  # adjust to -max/2:max/2
    r = np.sqrt(x ** 2 + x ** 2)  # radius

    # Frequency (f = 1 / dx)
    fx = np.arange(0, m) / m / dx
    fx = fftshift(fx)
    ind0 = np.where(fx == 0)[0][0]
    fx[0:ind0] = fx[0:ind0] - fx[ind0 - 1] - fx[ind0 + 1]

    # 2D matrix of q values
    [fxx, fyy] = np.meshgrid(fx, fx)
    [rho, _] = cart2pol(fxx, fyy)
    del fxx, fyy

    # Prescribed autocovariance function = sigma ** 2 * autocorrelation function
    if type.lower() == 'gaussian':
        acf = sigma ** 2 * np.exp(- (r / l0) ** 2)
    elif type.lower() == 'exponential':
        acf = sigma ** 2 * np.exp(- r / l0)

    ACF = {
        "r": r,      # radial position
        "acf": acf   # autocovariance function
    }

    # Corresponding power spectral density, which will be prescribed
    # No explicit calculation via ifft2 is needed since the analytical solution is known.
    rho[rho == 0] = fr
    if type.lower() == 'gaussian':
        S_2D = np.pi * l0 ** 2 * np.exp(-(np.pi * l0 * rho) ** 2)
    elif type.lower() == 'exponential':
        S_2D = 2 * np.pi * l0 ** 2 / (1 + (2 * np.pi * l0 * rho) ** 2) ** (3 / 2)

    # applying RMS
    S_2D[int(m / 2), int(m / 2)] = 0  # remove mean
    RMS_2D = np.sqrt((sum(sum(S_2D))) / (Lx * Lx)) # remove RMS in case S_2D is not normalized
    alfa = sigma / RMS_2D
    S_2D = S_2D * alfa ** 2

    # Azimuthal averaging
    res = 100  # resolution in frequency space
    frmin = np.log10(np.sqrt(((1 / Lx) ** 2 + (1 / Lx) ** 2)))
    frmax = np.log10(np.sqrt(fx[-1] ** 2 + fx[-1] ** 2))  # Nyquist
    f = 10 ** np.linspace(frmin, frmax, res)

    # Flatten 2d arrays to make it easier and faster
    rho = rho.flatten()
    S_2D = S_2D.flatten()

    # THIS PART NEEDS TO BE IMPROVED! TOO SLOW
    # Find elements in each bin and average PSD
    S_ave = np.zeros(len(f))
    ind = [[] for _ in range(len(f))]
    for j in range(0, len(f) - 1):
        ind[j] = np.where(np.logical_and(rho >= f[j], rho < f[j + 1]))[0]
        data = [S_2D[i] for i in ind[j]]
        if data:
            S_ave[j] = np.nanmean(S_2D[ind[j]])
        else:
            S_ave[j] = np.nan

    ind = ~np.isnan(S_ave)
    S = S_ave[ind]
    f = f[ind]

    # Bring back S_2D to correct shape since it will be provided as function output
    S_2D = S_2D.reshape((m, m))

    # reversing operation: PSD to fft
    Bq = np.sqrt(S_2D / (dx ** 2 / (m * m)))

    # apply conjugate symmetry to magnitude
    Bq[0, 0] = 0
    Bq[0, m // 2] = 0
    Bq[m // 2, m // 2] = 0
    Bq[m // 2, 0] = 0
    Bq[1::, 1:m // 2] = np.rot90(Bq[1::, m // 2 + 1::], 2)
    Bq[0, 1:m // 2] = np.flip(Bq[0, m // 2 + 1::], 0)
    Bq[m // 2 + 1::, 0] = np.flip(Bq[1:m // 2, 0], 0)
    Bq[m // 2 + 1::, m // 2] = np.flip(Bq[1:m // 2, m // 2], 0)

    # defining a random phase between -pi and phi (due to fftshift, otherwise between 0 and 2pi)
    phi = np.random.uniform(low=- np.pi, high=np.pi, size=(m, m))

    # apply conjugate symmetry to phase
    phi[0, 0] = 0
    phi[0, m // 2] = 0
    phi[m // 2, m // 2] = 0
    phi[m // 2, 0] = 0
    phi[1::, 1:m // 2] = -np.rot90(phi[1::, m // 2 + 1::], 2)
    phi[0, 1:m // 2] = -np.flip(phi[0, m // 2 + 1::], 0)
    phi[m // 2 + 1::, 0] = -np.flip(phi[1:m // 2, 0], 0)
    phi[m // 2 + 1::, m // 2] = -np.flip(phi[1:m // 2, m // 2], 0)

    # Combine magnitude and phase in cartesian coordinates
    [a, b] = pol2cart(Bq, phi)
    Hm = a + 1j * b

    # Calculate surface by inverse Fourier transformation
    z = ifft2(ifftshift(Hm)).real

    # Power spectral density dictionary, which will be provided as function output
    PSD = {
        "fx": fx,        # frequency x
        "fy": fx,        # frequency y
        "psd_xy": S_2D,  # 2D power spectral density
        "f": f,          # frequency radial
        "psd": S         # azimuthally averaged power spectral density
    }

    # RMS height (must/will match sigma of the prescribed ACF/PSD)
    RMS = np.std(z)

    return z, dx, RMS, PSD, ACF


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12})

    z, dx, RMS, PSD, ACF = rough_surface(sigma=40e-6, l0=20e-6, dx=2e-6, m=2 ** 11, type='gaussian')

    ACF_im = speckle_quality.autocorrelation(z, dx, res=5000, report=True)
    PSD_im = speckle_quality.power_spectral_density(z, dx)

    plt.imshow(z * 1e6, extent=[1e6*ACF['r'][0], 1e6*ACF['r'][-1],
                                1e6*ACF['r'][0], 1e6*ACF['r'][-1]],
               cmap='RdBu_r')
    clb = plt.colorbar()
    clb.set_label('Height [um]')
    plt.xlabel('x-position [um]')
    plt.ylabel('y-position [um]')
    plt.show()

    plt.loglog(PSD['f']*1e-6, PSD['psd'], 'k-', linewidth=2), plt.loglog(PSD_im['f']*1e-6, PSD_im['psd'], 'rs')
    plt.xlim(5e-4, 7e-2), plt.ylim(1e-30, 1e-15)
    plt.ylabel('Power spectral density')
    plt.xlabel('Frequency[1/um]')
    plt.legend(['Prescribed', 'Image'])
    plt.show()

    plt.plot(ACF['r']*1e6, ACF['acf'] / RMS ** 2, 'k-'), plt.plot(ACF_im['r']*1e6, ACF_im['acf'], 'r:', linewidth=2)
    plt.xlim(0, 3 * ACF_im['FWHM']*1e6)
    plt.ylabel('Autocorrelation function')
    plt.xlabel('Displacement [um]')
    plt.legend(['Prescribed', 'Image'])
    plt.show()
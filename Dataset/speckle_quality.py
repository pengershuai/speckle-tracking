#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:17:21 2019
Evaluation of the speckle quality
@author: Sebastian Meyer
"""

import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm
from scipy.ndimage.filters import minimum_filter as minf2D
from scipy.ndimage.filters import maximum_filter as maxf2D
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
from numpy.fft import fftshift, fft2
from scipy.signal.windows import tukey, hann, kaiser, bartlett, boxcar
from scipy.integrate import trapz
from quantiphy import Quantity


# Convert cartesian to polar coordinates
def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def mean_variance(image, kernel_size):
    """
    Compute local mean and variance via convolution
    :param image: NxN array of float
        2D intensity image
    :param kernel_size: int
        size [pixel] of the kernel used for calculating mean and variance
    :return mean: float
        mean value of the mean intensity map
    :return variance: float
        mean value of the intensity variance map
    """
    image = image.astype(float)  # convert to float to avoid issues, e.g. if the image is only of type uint16,
    # taking image**2 might become too large
    mean_filter = 1 / (kernel_size ** 2) * np.ones((kernel_size, kernel_size))  # kernel for local mean
    mean = fftconvolve(image, mean_filter, mode="valid")  # mean of image
    mean2 = fftconvolve(pow(image, 2), mean_filter, mode="valid")  # mean of squared image
    variance = np.mean(mean2 - pow(mean, 2))  # std identity
    return np.mean(mean), variance  # std identity


def visibility_std(image, kernel_size, report=False, display=False):
    """
    Compute the speckle viability within a sliding window according to: sigma/mean
    :param image: tuple of NxM float/int
        2D speckle intensity image
    :param kernel_size: int
        size (quadratic) of the sliding window
    :param report: logical, optional
        report the value of the median visibility. Default is "False"
    :param display: logical, optional
        create a plot of visibility map. Default is "False"
    :return v: float
        median value of the intensity map
    :return v_map: tuple of NxM float
        2D visibility map
    """
    image = image.astype(float)  # convert to float to avoid issues, e.g. if the image is only of type uint16,
    # taking image**2 might become too large
    mean_filter = 1 / (kernel_size ** 2) * np.ones((kernel_size, kernel_size))  # kernel for local mean
    mean = fftconvolve(image, mean_filter, mode="valid")  # mean of image
    mean2 = fftconvolve(pow(image, 2), mean_filter, mode="valid")  # mean of squared image
    v = 100 * np.mean(np.sqrt(mean2 - pow(mean, 2)) / mean)  # std identity
    v_map = 100 * np.sqrt(mean2 - pow(mean, 2)) / mean
    if report:
        print('Speckle visbility: %5.2f %%' % (v))
    if display:
        plt.imshow(v_map, cmap='RdBu_r', vmin=np.mean(v_map) - 3 * np.std(v_map),
                   vmax=np.mean(v_map) + 3 * np.std(v_map))
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                        labelleft=False)
        clb = plt.colorbar()
        clb.set_label('Visbility [%]')
        plt.show()
    return v, v_map


def visibility_minmax(image, kernel_size):
    """
    Compute the speckle viability within a sliding window according to: (max-min)/(max+min)
    :param image: tuple of NxM float/int
        2D speckle intensity image
    :param kernel_size: int
        size (quadratic) of the sliding window
    :return v: float
        median value of the intensity map
    :return v_map: tuple of NxM float
        2D visibility map
    """
    max = maxf2D(image, size=(kernel_size, kernel_size))  # maximum value in window
    min = minf2D(image, size=(kernel_size, kernel_size))  # minimum value in window
    return np.mean((max - min) / (max + min)), (max - min) / (max + min)


def visibility_vs_pixel(image, kernel_size):
    """
    Compute the speckle viability for a continuously increasing pixel size
    :param image: tuple of NxM float/int
        2D speckle intensity image
    :param kernel_size: int
        size (quadratic) of the sliding window
    :return: visibility_minmax (v[:, 2]), visibility_std (v[:, 1]) for different re-binning factors (v[:, 0])
    """
    max_bin_factor = round(kernel_size / 4)  # ROI must be maximum 4x4 pixels
    v = np.zeros((max_bin_factor, 3))  # Initialize output vector
    for bin_factor in tqdm(range(1, max_bin_factor + 1), ascii=True, desc="Binning:"):
        i = int(image.shape[0] - np.floor(image.shape[0] / bin_factor) * bin_factor)  # determine the number of
        # pixels which have to be cut in order to allow a correct re-binning
        image_binned = bin_ndarray(image[0:image.shape[0] - i, 0:image.shape[0] - i],
                                   new_shape=(
                                       round(image[0:image.shape[0] - i, 0:image.shape[0] - i].shape[0] / bin_factor),
                                       round(image[0:image.shape[0] - i, 0:image.shape[0] - i].shape[1] / bin_factor)),
                                   operation='sum')  # re-bin image by summing the pxiel values

        v[bin_factor - 1, 0] = bin_factor  # re-binning factor
        v[bin_factor - 1, 1] = visibility_std(image_binned, round(kernel_size / bin_factor))[0]  # visibility
        v[bin_factor - 1, 2] = visibility_minmax(image_binned, round(kernel_size / bin_factor))[0]  # visibility
    return v


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
    averaging.
    :param ndarray: ndarray
        input array
    :param new_shape: tuple with the same dimensions as input array
        requested shape
    :param operation: 'sum' or 'mean'
        operation used to combined pixel values:
    :return:
    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape,
                                                     ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def darkfield_transmission(Ir, Is, kernel_size):
    mean_filter = 1 / (kernel_size ** 2) * np.ones((kernel_size, kernel_size))  # kernel for local mean
    mean_Ir = fftconvolve(Ir, mean_filter, mode="same")  # mean of image
    mean2_Ir = fftconvolve(pow(Ir, 2), mean_filter, mode="same")  # mean of squared image
    std_Ir = np.sqrt(mean2_Ir - pow(mean_Ir, 2))
    mean_Is = fftconvolve(Is, mean_filter, mode="same")  # mean of image
    mean2_Is = fftconvolve(pow(Is, 2), mean_filter, mode="same")  # mean of squared image
    std_Is = np.sqrt(mean2_Is - pow(mean_Is, 2))
    return (std_Is / mean_Is) / (std_Ir / mean_Ir), mean_Is / mean_Ir


def patched_autocorrelation(image, dx, patch_size, mode='same', report=False, plot=False):
    """
    Compute the azimuthally averaged autocorrelation using an average of several small patches
    :param image: 2darray of float/int
        input image
    :param dx: float
        pixel size [m]
    :param patch_size: tuple of 2 int
        size of the patch
    :param mode: string, optional
        mode of the fftconvolve
    :param report: logical, optional
        report the FWHM of the autocorrelation function. Default is "False"
    :param plot: logical, optional
        create a plot of the autocorrelation function. Default is "False"
    :return: dict
        azimutally averaged autocorrelation function
    """
    patches = view_as_windows(image, patch_size, (int(patch_size[0] / 2), int(patch_size[1] / 2)))
    N, M, _, _ = patches.shape

    first = True
    for n in range(0, N):
        for m in range(0, M):
            image = patches[n, m, :, :]
            image = image - np.mean(image)
            ac2D = fftconvolve(image, image[::-1, ::-1], mode=mode) / np.var(image) / (image.shape[0] * image.shape[1])
            if first:
                first = False
                ac2D_t = ac2D
            else:
                ac2D_t = ac2D_t + ac2D
    ac2D_t = ac2D_t / (N * M)
    ac = azi_av_acf(ac2D_t, dx)
    if report:
        print('FWHM:', Quantity(ac['FWHM'], 'm'))
        print('Correlation length:', Quantity(ac['l0'], 'm'))
    if plot:
        fig, ax = plt.subplots(1, 1)
        plt.plot(ac['r'], ac['ac'], color='#011F5B', linewidth=4)
        ax.set(xlabel='Displacement [m]', ylabel='Autocorrelation')
        ax.grid(linestyle='--', linewidth='0.5', color='black')
        plt.xlim(0, int(2 * ac['FWHM']))
        plt.show()
    return ac


def autocorrelation(image, dx, mode='same', res=5000, report=False, plot=False):
    """
    Compute the azimuthally averaged autocorrelation
    :param image: 2darray of float/int
        input image
    :param dx: float
        pixel size [m]
    :param mode: string
        one of 'same' (default), 'full' or 'valid' (see help for fftconvolve for more info)
    :param res: int, optional
        resolution for the azimutal averaging
    :param plot: logical, optional
        create a plot of the autocorrelation function. Default is "False"
    :param report: logical, optional
        report the FWHM of the autocorrelation function. Default is "False"
    :return: autocorrelation function
    """
    image = image - np.mean(image)
    ac2D = fftconvolve(image, image[::-1, ::-1], mode=mode) / np.var(image) / (image.shape[0] * image.shape[1])
    ac = azi_av_acf(ac2D, dx, res)
    if report:
        print('FWHM:', Quantity(ac['FWHM'], 'm'))
        print('Correlation length:', Quantity(ac['l0'], 'm'))
    if plot:
        fig, ax = plt.subplots(1, 1)
        plt.plot(ac['r'], ac['acf'], color='#011F5B', linewidth=4)
        ax.set(xlabel='Displacement [m]', ylabel='Autocorrelation')
        ax.grid(linestyle='--', linewidth='0.5', color='black')
        plt.xlim(0, 4 * ac['FWHM'])
        plt.show()

    return ac


def azi_av_acf(acf2D, dx, res):
    """
    Perform a azimuthal averaging of the autocorrelation function
    :param acf2D: 2darray of float/int
        2D autocorrelation, but can be actually any 2D image
    :param dx: float
        pixel size [m]
    :param res: int
        resolution of radial binning
    :return: dictonary
        azimutally averaged autocorrelation function
        r : radial position [m]
        acf: value of the autocorrelation function
        FWHM: FWHM of autocorrelation [m]
        l0: correlation length (i.e., at exp(-1)) of autocorrelation [m]
    """
    if not np.equal(acf2D.shape[0], acf2D.shape[1]):
        raise NotImplementedError('matrix must be square!')

    m = acf2D.shape[0]

    # Positions
    xi = np.arange(0, m) * dx
    xi = fftshift(xi)  # shift 0 to center of array
    ind0 = np.where(xi == 0)[0][0]  # find pixel index of central pixel
    xi[0:ind0] = xi[0:ind0] - xi[ind0 - 1] - xi[ind0 + 1]  # adjust to -max/2:max/2

    # 2D matrix of xi values and corresponding radius map
    [xx, yy] = np.meshgrid(xi, xi)
    [rho, _] = cart2pol(xx, yy)

    xrmax = np.log10(np.sqrt(xi[-1] ** 2 + xi[-1] ** 2))  # Nyquist
    r = np.linspace(0, 10 ** xrmax, res)

    # Flatten 2d arrays to make it easier and faster
    rho = rho.flatten()
    acf2D = acf2D.flatten()

    acf_ave = np.zeros(len(r))
    ind = [[] for _ in range(len(r))]
    for j in range(0, len(r) - 1):
        ind[j] = np.where(np.logical_and(rho >= r[j], rho < r[j + 1]))[0]
        data = [acf2D[i] for i in ind[j]]
        if data:
            acf_ave[j] = np.nanmean(acf2D[ind[j]])
        else:
            acf_ave[j] = np.nan

    ind = ~np.isnan(acf_ave)
    acf_ave = acf_ave[ind]
    r = r[ind]

    acf_FWHM = 2 * np.interp(0.5, acf_ave[::-1], r[::-1], right=99, left=98)
    acf_cl = np.interp(np.exp(-1), acf_ave[::-1], r[::-1], right=99, left=98)

    return {"r": r, "acf": acf_ave, "FWHM": acf_FWHM, "l0": acf_cl}


def power_spectral_density(z, dx, win='Welch', res=100):
    """
    Compute the azimuthally averaged power_spectral_density
    :param z: 2d array of float
        input image
    :param dx: float
        pixel size [m]
    :param win: string, optional
        window type used for calculation. Can be tukey, hann, kaiser, bartlett or welch (default).
    :param res: int
    resolution of radial binning
    :return: dict
        power_spectral_density
    """
    if not np.equal(z.shape[0], z.shape[1]):
        raise NotImplementedError('matrix must be square!')

    listOfWindows = ["tukey", "hann", "kaiser", "bartlett", "welch"]
    if not win.lower() in listOfWindows:
        raise NotImplementedError('Unknown window type! Please use tukey, hann, kaiser, bartlett or welch.')

    m = z.shape[0]
    if np.mod(m, 2):
        z = z[1::, 1::]
        m = m - 1

    # Window function
    if win.lower() == 'tukey':
        win = tukey(m, 0.25).reshape(-1, 1) * tukey(m, 0.25).reshape(1, -1)
    elif win.lower() == 'hann':
        win = hann(m).reshape(-1, 1) * hann(m).reshape(1, -1)
    elif win.lower() == 'kaiser':
        win = kaiser(m, 10).reshape(-1, 1) * kaiser(m, 10).reshape(1, -1)
    elif win.lower() == 'bartlett':
        win = bartlett(m).reshape(-1, 1) * bartlett(m).reshape(1, -1)
    elif win.lower() == 'welch':
        win = (1 - ((np.arange(0, m) - ((m - 1) / 2)) / ((m + 1) / 2)) ** 2).reshape(-1, 1) * \
              (1 - ((np.arange(0, m) - ((m - 1) / 2)) / ((m + 1) / 2)) ** 2).reshape(1, -1)  # Welch

    recInt = trapz(
        trapz((boxcar(m).reshape(-1, 1) * boxcar(m).reshape(1, -1)) ** 2, np.arange(0, m), axis=1),
        np.arange(0, m), axis=0)
    winInt = trapz(trapz(win ** 2, np.arange(0, m), axis=1), np.arange(0, m), axis=0)

    U = winInt / recInt  # Normalization constant
    z_win = z * win
    Hm = fftshift(fft2(z_win))
    Cq = (1 / U) * (dx ** 2 / (m * m) * (abs(Hm) ** 2))
    Cq[m // 2, m // 2] = 0

    PSD = azi_av_psd(Cq, dx, res)

    return PSD


def azi_av_psd(psd2D, dx, res):
    """
    Perform a azimuthal averaging of the power spectral density
    :param psd2D: 2darray of float/int
        2D power spectral density
    :param dx: float
        pixel size [m]
    :param res: int
    resolution of radial binning
    :return: dictonary
        azimutally averaged power spectral density
        r : frequencies [1/m]
        psd : power spectral density
    """
    if not np.equal(psd2D.shape[0], psd2D.shape[1]):
        raise NotImplementedError('matrix must be square!')

    m = psd2D.shape[0]
    Lx = m * dx

    # Frequencies
    fx = np.arange(0, m) / m / dx
    fx = fftshift(fx)
    ind0 = np.where(fx == 0)[0][0]
    fx[0:ind0] = fx[0:ind0] - fx[ind0 - 1] - fx[ind0 + 1]

    # 2D matrix of q values
    [fxx, fyy] = np.meshgrid(fx, fx)
    [rho, _] = cart2pol(fxx, fyy)
    del fxx, fyy

    # Radial Averaging : DATA WILL BE USEFUL FOR CHECKING RESULTS
    rhof = np.floor(rho)
    frmin = np.log10(np.sqrt(((1 / Lx) ** 2 + (1 / Lx) ** 2)))
    frmax = np.log10(np.sqrt(fx[-1] ** 2 + fx[-1] ** 2))  # Nyquist
    f = np.floor(10 ** np.linspace(frmin, frmax, res))

    # Flatten 2d arrays to make it easier and faster
    rhof = rhof.flatten()
    psd2D = psd2D.flatten()

    psd_ave = np.zeros(len(f))
    ind = [[] for _ in range(len(f))]
    for j in range(0, len(f) - 1):
        ind[j] = np.where(np.logical_and(rhof >= f[j], rhof < f[j + 1]))[0]
        data = [psd2D[i] for i in ind[j]]
        if data:
            psd_ave[j] = np.nanmean(psd2D[ind[j]])
        else:
            psd_ave[j] = np.nan

    ind = ~np.isnan(psd_ave)
    psd = psd_ave[ind]
    f = f[ind]

    return {"f": f, "psd": psd}

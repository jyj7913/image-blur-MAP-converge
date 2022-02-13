from math import sqrt
import cv2
import numpy as np
from util import *
from scipy.ndimage import correlate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from skimage import color, data, restoration


def deblur_single_level(blurred, k_size):
    lambda_l = 0.00064
    alpha = 0.1
    labmda_k = 0.001
    num_iters = 20

    # initial psf
    psf = np.zeros(shape=k_size)
    psf[(k_size[0]+1)//2][(k_size[1]+1)//2] = 1

    # prepare gradient maps of blurred
    # gaus2 = gauss2d(shape=(60, 60), sigma=10)
    # blur_f = np.fft.fft2(blurred)
    # gaus2_f = psf2otf(gaus2, blurred.shape)
    # img_flt_f = blur_f * gaus2_f
    # blurred = np.real(np.fft.ifft2(img_flt_f))

    sx = np.array([[0, -1, 1]])
    sy = np.array([[0, -1, 1]]).T

    bx = correlate(blurred, sx)
    by = correlate(blurred, sy)

    blurred_g = np.array((bx, by))
    blurred_g = np.transpose(blurred_g, (1, 2, 0))

    for i in range(num_iters):
        latent_g = deconv_sps(blurred_g, psf, lambda_l, alpha)
        energy, data, prior_l, prior_k = energy_func(
            latent_g, blurred_g, psf, lambda_l, alpha, labmda_k)
        print(i)
        print(f'{i} {energy}    {data}  {prior_l}   {prior_k}')
        psf = estimate_psf(blurred_g, latent_g, psf.shape, labmda_k)

    return psf, latent_g


def energy_func(latent, blurred, psf, lambda_l, alpha, lambda_k):
    tau = 0.01

    K = psf2otf(psf, blurred.shape)
    K = np.transpose(np.array([K, K]), (1, 2, 0))
    b = np.real(np.fft.ifft2(np.fft.fft2(latent)*K))
    diff = b - blurred
    data = np.linalg.norm(diff) ** 2

    w = np.maximum(np.abs(latent), tau) ** (alpha-2)
    prior_l = np.sum(w * (latent**2))
    prior_k = np.sum(psf ** 2)

    energy = data + lambda_l * prior_l + lambda_k * prior_k
    return energy, data, prior_l, prior_k


def deconv_sps(blurred, psf, lambda_l, alpha, num_iters=15):
    tau = 0.01

    B = np.fft.fft2(blurred)
    K = psf2otf(psf, (blurred.shape[0], blurred.shape[1]))
    K = np.transpose(np.array([K, K]), (1, 2, 0))
    L = (np.conjugate(K) * B) / (np.conjugate(K) * K + lambda_l)
    latent = np.real(np.fft.ifft2(L))

    for i in range(num_iters):
        w = np.maximum(np.abs(latent), tau) ** (alpha-2)
        latent = deconv_L2(blurred, latent, psf, lambda_l, w, 5)

    return latent


def deconv_L2(blurred, latent, psf, lambda_l, weight, n_iters):
    img_size = blurred.shape
    psf_f = psf2otf(psf, img_size)
    b1 = np.real(np.fft.ifft2(np.fft.fft2(
        blurred[:, :, 0]) * np.conjugate(psf_f)))
    b2 = np.real(np.fft.ifft2(np.fft.fft2(
        blurred[:, :, 1]) * np.conjugate(psf_f)))
    b = np.transpose(np.array([b1, b2]), (1, 2, 0))

    p = [lambda_l, psf_f, weight]
    latent = conjgrad(latent, b, n_iters, 1e-4, Ax, p)
    return latent


def Ax(x, p):
    y1 = np.real(np.fft.ifft2(np.conjugate(
        p[1]) * p[1] * np.fft.fft2(x[:, :, 0])))
    y2 = np.real(np.fft.ifft2(np.conjugate(
        p[1]) * p[1] * np.fft.fft2(x[:, :, 1])))

    y = np.transpose(np.array([y1, y2]), (1, 2, 0))
    y = y + p[0] * p[2] * x
    return y


def conjgrad(x, b, n_iters, tol, Ax_func, func_param):
    r = b - Ax_func(x, func_param)
    p = r
    rsold = np.sum(r ** 2)

    for i in range(n_iters):
        Ap = Ax_func(p, func_param)
        alpha = rsold/np.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(r ** 2)
        if sqrt(rsnew) < tol:
            break
        p = r + rsnew / rsold * p
        rsold = rsnew

    return x


def estimate_psf(blurred, latent, psf_size, lambda_k):
    B = np.fft.fft2(blurred)
    L = np.fft.fft2(latent)
    Bx = B[:, :, 0]
    By = B[:, :, 1]
    Lx = L[:, :, 0]
    Ly = L[:, :, 1]

    Lap = psf2otf(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
                  (blurred.shape[0], blurred.shape[1]))
    K = (np.conjugate(Lx) * Bx + np.conjugate(Ly) * By) / \
        (np.conjugate(Lx) * Lx + np.conjugate(Ly) * Ly + Lap * lambda_k)
    psf = otf2psf(K, psf_size)
    psf = psf / np.sum(psf)
    psf = np.real(psf)
    psf = np.greater(psf, np.max(psf) * 0.05) * psf
    psf = psf / np.sum(psf)

    return psf


# def find_optimal_no_blur(blurred, lambda_l, alpha):
#     LUT = make_LUT(lambda_l, alpha)
#     bx = blurred[:, :, 0]
#     by = blurred[:, :, 1]

#     lx = interp1d(LUT[0], LUT[1], kind='linear', fill_value='extrapolate')(bx)
#     lx = lx.reshape(bx.shape)

#     ly = interp1d(LUT[0], LUT[1], kind='linear', fill_value='extrapolate')(by)
#     ly = ly.reshape(by.shape)
#     l = np.array(lx, ly)
#     l = np.transpose(l, (1, 2, 0))
#     return l


# def make_LUT(lambda_l, alpha):
#     v = np.arange(-1, 1, 0.1/256)
#     b, l = np.meshgrid(v, v)
#     tau = 0.01

#     w = np.maximum(np.abs(l), tau) ** (alpha-2)
#     prior_l = w*l**2

#     energies = (b - l) ** 2 + lambda_l * prior_l
#     min_energies, min_indices = np.min(energies, 0), np.argmin(energies, 0)
#     LUT = [v, l[min_indices]]

#     return LUT


def showResult(img, psf, deblur):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
    plt.gray()

    for a in (ax[0], ax[1], ax[2]):
        a.axis('off')

    ax[0].imshow(img)
    ax[0].set_title('Original Data')

    ax[1].imshow(psf)
    ax[1].set_title('Kernel')

    ax[2].imshow(deblur)
    ax[2].set_title('Restoration using\nRichardson-Lucy')

    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    plt.show()
    return 0


def main():
    x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    blurred = cv2.imread('picassoBlurImage.PNG', IMREAD_GRAYSCALE) / 255.
    _out = cv2.imread('picassoOut.png', IMREAD_GRAYSCALE) / 255.

    psf = np.load('picasso51kernel4.npy', allow_pickle=True)
    # latent = deconv_sps(np.transpose(
    # np.array([blurred, blurred]), (1, 2, 0)), psf, 0.00064, 0.1)

    # psf, latent = deblur_single_level(blurred, (51, 51))

    # plt.imshow(psf, cmap='gray')
    # plt.show()

    deconv = restoration.wiener(blurred, psf, 0.1)

    plt.imshow(deconv, cmap='gray')
    plt.show()
    # np.save('picasso51kernel4', psf)

    # cv2.imwrite('picassolatent04.png', latent[:, :, 0])
    # cv2.imwrite('picassolatent14.png', latent[:, :, 1])
    # cv2.imwrite('picassokernel4.png', psf)
    # print(np.max(psf))

    # cv2.imshow("51*51 latent0", latent[:, :, 0])
    # cv2.imshow("51*51 latent1", latent[:, :, 1])
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    main()

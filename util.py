from cv2 import BORDER_REFLECT, IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
import pandas as pd
from scipy.ndimage import convolve
import scipy.stats
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

LAMBDA1 = 0.1
LAMBDA2 = 15
GAMMA = 8


class psfModel(nn.Module):
    def __init__(self, psf, img, latent):
        super().__init__()
        weights = psf

        pad_img = padding(img, 1)
        pad_lat = padding(latent, 1)
        sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        img_x = conv2dn(pad_img, sx)
        img_y = conv2dn(pad_img, sy)
        pad_img_x = padding(img_x, 1)
        pad_img_y = padding(img_y, 1)
        img_xx = conv2dn(pad_img_x, sx)
        img_xy = conv2dn(pad_img_x, sy)
        img_yy = conv2dn(pad_img_y, sy)
        self.img_grad = [torch.Tensor(img), torch.Tensor(img_x), torch.Tensor(
            img_y), torch.Tensor(img_xx), torch.Tensor(img_xy), torch.Tensor(img_yy)]

        lat_x = conv2dn(pad_lat, sx)
        lat_y = conv2dn(pad_lat, sy)
        pad_lat_x = padding(lat_x, 1)
        pad_lat_y = padding(lat_y, 1)
        lat_xx = conv2dn(pad_lat_x, sx)
        lat_xy = conv2dn(pad_lat_x, sy)
        lat_yy = conv2dn(pad_lat_y, sy)
        self.lat_grad = [torch.Tensor(padding(latent, psf.shape[0]//2)), torch.Tensor(padding(lat_x, psf.shape[0]//2)), torch.Tensor(padding(lat_y, psf.shape[0]//2)),
                         torch.Tensor(padding(lat_xx, psf.shape[0]//2)), torch.Tensor(padding(lat_xy, psf.shape[0]//2)), torch.Tensor(padding(lat_yy, psf.shape[0]//2))]

        self.w = [0, 1, 1, 2, 2, 2]

        self.weights = nn.Parameter(torch.Tensor(weights), requires_grad=True)

    def forward(self):
        psf = self.weights
        sum = 0
        for i in range(5):
            temp = conv2dt(self.lat_grad[i], psf) - self.img_grad[i]
            temp = torch.norm(temp).item() ** 2
            temp *= 50 / (2 ** self.w[i])
            sum += temp
        sum += torch.norm(psf, p=1).item()

        return sum


def training_loop(model, optimizer, n=100):
    loss_fn = nn.MSELoss()
    for param in model.parameters():
        param.requires_grad_(True)

    for i in range(n):
        preds = model()
        loss = loss_fn(torch.Tensor([preds]), torch.Tensor([0]))
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    loss.requires_grad_(False)

    for param in model.parameters():
        param.requires_grad_(False)
        return param.data


class psiModel(nn.Module):
    def __init__(self, psi, img, latent, omega):
        super().__init__()
        weight_x = psi[0]
        weight_y = psi[1]
        pad_img = padding(img, 1)
        pad_lat = padding(latent, 1)
        self.img = torch.Tensor(pad_img)
        self.latent = torch.Tensor(pad_lat)
        self.omega = torch.Tensor(omega)
        self.sx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.sy = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        eq_x = torch.zeros(size=img.shape)
        eq_y = torch.zeros(size=img.shape)

        for i in range(len(eq_x)):
            for j in range(len(eq_x[i])):
                eq_x[i][j] = eq(weight_x[i][j])
                eq_y[i][j] = eq(weight_y[i][j])

        self.eq_x = eq_x
        self.eq_y = eq_y

        self.weight_x = nn.Parameter(
            torch.Tensor(weight_x), requires_grad=True)
        self.weight_y = nn.Parameter(
            torch.Tensor(weight_y), requires_grad=True)

    def forward(self):
        psi_x = self.weight_x
        psi_y = self.weight_y
        lam1x = self.eq_x
        lam1x = torch.norm(lam1x, p=1)
        lam1y = self.eq_y
        lam1y = torch.norm(lam1y, p=1)

        lam2x = psi_x - conv2dt(self.img, self.sx)
        lam2x = torch.mul(lam2x, self.omega)
        lam2x = LAMBDA2 * torch.norm(lam2x).item() ** 2

        lam2y = psi_y - conv2dt(self.img, self.sy)
        lam2y = torch.mul(lam2y, self.omega)
        lam2y = LAMBDA2 * torch.norm(lam2y).item() ** 2

        gamx = psi_x - conv2dt(self.latent, self.sx)
        gamx = GAMMA * torch.norm(gamx).item() ** 2

        gamy = psi_y - conv2dt(self.latent, self.sy)
        gamy = GAMMA * torch.norm(gamy).item() ** 2

        ret = lam1x + lam1y + lam2x + lam2y + gamx + gamy

        return ret


class psiModel2(nn.Module):
    def __init__(self, psi, img, latent, omega):
        super().__init__()
        weight_x = psi[0]
        weight_y = psi[1]
        pad_img = padding(img, 1)
        pad_lat = padding(latent, 1)
        self.img = torch.Tensor(pad_img)
        self.latent = torch.Tensor(pad_lat)
        self.omega = torch.Tensor(omega)
        self.sx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.sy = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        eq_x = torch.zeros(size=img.shape)
        eq_y = torch.zeros(size=img.shape)

        for i in range(len(eq_x)):
            for j in range(len(eq_x[i])):
                eq_x[i][j] = eq(weight_x[i][j])
                eq_y[i][j] = eq(weight_y[i][j])

        self.eq_x = eq_x
        self.eq_y = eq_y

        self.conv1 = nn.Conv2d(1, 1, img.shape)
        self.conv2 = nn.Conv2d(1, 1, img.shape)

        with torch.no_grad():
            self.conv1.weight = nn.Parameter(torch.unsqueeze(
                torch.unsqueeze(torch.Tensor(weight_x), 0), 0), requires_grad=True)
            self.conv2.weight = nn.Parameter(torch.unsqueeze(
                torch.unsqueeze(torch.Tensor(weight_y), 0), 0), requires_grad=True)

            # self.conv1.weight.data = self.conv1.weight.data + \
            #     torch.Tensor(weight_x)
            # self.conv2.weight.data = self.conv2.weight.data + \
            #     torch.Tensor(weight_y)

    def forward(self):
        psi_x = torch.squeeze(self.conv1.weight.data)
        psi_y = torch.squeeze(self.conv2.weight.data)
        lam1x = self.eq_x
        lam1x = torch.norm(lam1x, p=1)
        lam1y = self.eq_y
        lam1y = torch.norm(lam1y, p=1)

        lam2x = psi_x - conv2dt(self.img, self.sx)
        lam2x = torch.mul(lam2x, self.omega)
        lam2x = LAMBDA2 * torch.norm(lam2x).item() ** 2

        lam2y = psi_y - conv2dt(self.img, self.sy)
        lam2y = torch.mul(lam2y, self.omega)
        lam2y = LAMBDA2 * torch.norm(lam2y).item() ** 2

        gamx = psi_x - conv2dt(self.latent, self.sx)
        gamx = GAMMA * torch.norm(gamx).item() ** 2

        gamy = psi_y - conv2dt(self.latent, self.sy)
        gamy = GAMMA * torch.norm(gamy).item() ** 2

        ret = lam1x + lam1y + lam2x + lam2y + gamx + gamy

        return ret


def training_loop_psi(model, optimizer, n=15):
    loss_fn = nn.MSELoss()
    for param in model.parameters():
        param.requires_grad_(True)

    for i in range(n):
        preds = model()
        loss = loss_fn(preds, torch.Tensor([0]))
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    loss.requires_grad_(False)
    ret = []
    for i, param in enumerate(model.parameters()):
        param.requires_grad_(False)
        if i % 2 == 0:
            ret.append(torch.squeeze(param.data).cpu().detach().numpy())
    return ret


# def psf2otf(flt, img_shape):
#     flt_top_half = flt.shape[0]//2
#     flt_bottom_half = flt.shape[0] - flt_top_half
#     flt_left_half = flt.shape[1]//2
#     flt_right_half = flt.shape[1] - flt_left_half
#     flt_padded = np.zeros(img_shape, dtype=flt.dtype)
#     flt_padded[:flt_bottom_half,
#                :flt_right_half] = flt[flt_top_half:, flt_left_half:]
#     flt_padded[:flt_bottom_half, img_shape[1] -
#                flt_left_half:] = flt[flt_top_half:, :flt_left_half]
#     flt_padded[img_shape[0]-flt_top_half:,
#                :flt_right_half] = flt[:flt_top_half, flt_left_half:]
#     flt_padded[img_shape[0]-flt_top_half:, img_shape[1] -
#                flt_left_half:] = flt[:flt_top_half, :flt_left_half]
#     return np.fft.fft2(flt_padded)


def conv2dn(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def conv2dt(a, f):
    ret = F.conv2d(Variable(a.view(1, 1, a.shape[0], a.shape[1])),
                   Variable(f.view(1, 1, f.shape[0], f.shape[1])))
    ret.squeeze_()
    return ret


def padding(img, window_param):
    ret = cv2.copyMakeBorder(
        img, window_param, window_param, window_param, window_param, cv2.BORDER_REFLECT)
    return ret


def computOmega(img, window_param):
    padded = padding(img, window_param)
    res = np.zeros(shape=img.shape)
    for i in np.arange(window_param, img.shape[0]+window_param):
        for j in np.arange(window_param, img.shape[1]+window_param):
            temp_img = padded[i:i+window_param*2+1, j:j+window_param*2+1]
            st = np.std(temp_img)
            if st < 5/255.:
                res[i-window_param][j-window_param] = 1
    return res


def psi_x(psi, a):
    energy = LAMBDA1 * abs(eq(psi))
    energy += LAMBDA2 * a[2] * ((psi - a[0])**2)
    energy += GAMMA * ((psi - a[1])**2)
    return energy


def psi_y(psi, a):
    energy = LAMBDA1 * abs(eq(psi))
    energy += LAMBDA2 * a[2] * ((psi - a[0])**2)
    energy += GAMMA * ((psi - a[1])**2)
    return energy


def trii(img_shape):
    x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    fx = psf2otf(x, img_shape)
    fy = psf2otf(y, img_shape)

    fxx = fx * fx
    fyy = fy * fy
    fxy = fx * fy

    xout = (50/(2**1)) * (fx * np.conjugate(fx))
    yout = (50/(2**1)) * (fy * np.conjugate(fy))
    xxout = (50/(2**2)) * (fxx * np.conjugate(fxx))
    yyout = (50/(2**2)) * (fyy * np.conjugate(fyy))
    xyout = (50/(2**2)) * (fxy * np.conjugate(fxy))

    return xout + yout + xxout + yyout + xyout


def eq(x):
    if abs(x) > 1.885:
        return -(6.1e-4 * (x**2) + 5.0)
    else:
        return -2.7 * abs(x)


def gauss(n=11, sigma=1):
    r = np.arange(0, n, dtype=np.float32) - (n-1.)/2.
    r = np.exp(-r**2./(2.*sigma**2))
    return r / np.sum(r)


def gauss2d(shape=(11, 11), sigma=1):
    g1 = gauss(shape[0], sigma).reshape([shape[0], 1])
    g2 = gauss(shape[1], sigma).reshape([1, shape[1]])
    return np.matmul(g1, g2)


def psf2otf(psf, otf_size):
    # calculate otf from psf with size >= psf size

    if psf.any():  # if any psf element is non-zero
        # pad PSF with zeros up to image size
        pad_size = ((0, otf_size[0]-psf.shape[0]),
                    (0, otf_size[1]-psf.shape[1]))
        psf_padded = np.pad(psf, pad_size, 'constant')

        # circularly shift psf
        psf_padded = np.roll(
            psf_padded, -int(np.floor(psf.shape[0]/2)), axis=0)
        psf_padded = np.roll(
            psf_padded, -int(np.floor(psf.shape[1]/2)), axis=1)

       # calculate otf
        otf = np.fft.fftn(psf_padded)
        # this condition depends on psf size
        num_small = np.log2(psf.shape[0])*4*np.spacing(1)
        if np.max(abs(otf.imag))/np.max(abs(otf)) <= num_small:
            otf = otf.real
    else:  # if all psf elements are zero
        otf = np.zeros(otf_size)
    return otf


def otf2psf(otf, psf_size):
    # calculate psf from otf with size <= otf size

    if otf.any():  # if any otf element is non-zero
        # calculate psf
        psf = np.fft.ifftn(otf)
        # this condition depends on psf size
        num_small = np.log2(otf.shape[0])*4*np.spacing(1)
        if np.max(abs(psf.imag))/np.max(abs(psf)) <= num_small:
            psf = psf.real

        # circularly shift psf
        psf = np.roll(psf, int(np.floor(psf_size[0]/2)), axis=0)
        psf = np.roll(psf, int(np.floor(psf_size[1]/2)), axis=1)

        # crop psf
        psf = psf[0:psf_size[0], 0:psf_size[1]]
    else:  # if all otf elements are zero
        psf = np.zeros(psf_size)
    return psf

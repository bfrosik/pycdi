from pycohere.lib.cohlib import cohlib
import cupy as cp
import cupyx as cpx
import math
import numpy as np


class cplib(cohlib):
    def set_device(dev_id):
        cp.cuda.Device(dev_id).use()

    def set_backend(proc):
        pass

    def to_numpy(arr):
        return cp.array(arr).T

    def from_numpy(arr):
        return cp.asnumpy(arr.T)

    def asarray(arr, dtype):
        return arr.astype(dtype)

    def random(shape, **kwargs):
        import time
        import os

        cp.random.seed(int(time.time() * 1000000) * os.getpid() + os.getpid())
        return cp.random.random(shape, dtype=cp.complex64)

    def fftshift(arr):
        return cp.fft.fftshift(arr)

    def ifftshift(arr):
        return cp.fft.fftshift(arr)

    def shift(arr, sft):
        return cp.roll(arr, sft)

    def fft(arr):
        return cp.fft.fftn(arr)

    def ifft(arr):
        return cp.fft.ifftn(arr)

    def fftconvolve(arr1, arr2):
        return cpx.scipy.signal.convolve(arr1, arr2)
      #  return cpx.scipy.ndimage.convolve(arr1, arr2)

    def where(cond, x, y):
        return cp.where(cond, x, y)

    def dims(arr):
        # get array dimensions
        return arr.shape

    def abs(arr):
        return cp.abs(arr)

    def sqrt(arr):
        return cp.sqrt(arr)

    def square(arr):
        return cp.square(arr)

    def sum(arr):
        return cp.sum(arr)

    def real(arr):
        return cp.real(arr)

    def imag(arr):
        return cp.imag(arr)

    def max(arr):
        return cp.amax(arr)

    def print(arr, **kwargs):
        print(arr)

    def replace(lhs, cond, rhs):
        return cp.select(cond, lhs, rhs)

    def arctan2(arr1, arr2):
        return cp.atan2(arr1, arr2)

    def flip(arr, axis=None):
        if axis is None:
            raise NotImplementedError
        else:
            return cp.flip(arr, axis)

    def full(shape, fill_value, **kwargs):
        return cp.full(shape, fill_value)

    def gaussian(shape, sigmas, **kwargs):
        inarr = cp.full(shape, 1.0)
        return cpx.scipy.ndimage.gaussian_filter(inarr, sigmas)

    def center_of_mass(inarr):
        return cpx.scipy.ndimage.center_of_mass(inarr)
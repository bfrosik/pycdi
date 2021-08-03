from pycohere.lib.cohlib import cohlib
import numpy as np
import scipy


class nplib(cohlib):

    def set_device(dev_id):
        pass

    def set_backend(proc):
        pass

    def to_numpy(arr):
        return arr

    def load(filename):
        return np.load(filename)

    def from_numpy(arr):
        return arr

    def save(filename, arr):
        np.save(filename, arr)

    def dtype(arr):
        return arr.dtype

    def size(arr):
        return arr.size

    def hasnan(arr):
        return np.any(np.isnan(arr))

    def random(shape, **kwargs):
        import time
        import os

       # rng = np.random.default_rng(time.time()* 10000 * os.getpid() + os.getpid())
        if len(shape) == 1:
            return np.random.rand(shape[0])
        elif len(shape) == 2:
            return np.random.rand(shape[0], shape[1])
        elif len(shape) == 3:
            return np.random.rand(shape[0], shape[1], shape[2])

    def fftshift(arr):
        return np.fft.fftshift(arr)

    def ifftshift(arr):
        return np.fft.fftshift(arr)

    def shift(arr, sft):
        sft = [int(s) for s in sft]
        return np.roll(arr, sft)

    def fft(arr):
        return np.fft.fftn(arr)

    def ifft(arr):
        return np.fft.ifftn(arr)

    def fftconvolve(arr1, arr2):
        return scipy.ndimage.convolve(arr1, arr2)

    def where(cond, x, y):
        return np.where(cond, x, y)

    def dims(arr):
        # get array dimensions
        return arr.shape

    def absolute(arr):
        return np.absolute(arr)

    def sqrt(arr):
        return np.sqrt(arr)

    def square(arr):
        return np.square(arr)

    def sum(arr, axis=None):
        return np.sum(arr, axis)

    def real(arr):
        return np.real(arr)

    def imag(arr):
        return np.imag(arr)

    def amax(arr):
        return np.amax(arr)

    def maximum(arr1, arr2):
        return np.maximum(arr1, arr2)

    def argmax(arr, axis=None):
        return np.argmax(arr, axis)

    def ceil(arr):
        return cp.ceil(arr)

    def fix(arr):
        return np.fix(arr)

    def print(arr, **kwargs):
        print(arr)

    def angle(arr):
        return np.angle(arr)

    def flip(arr, axis=None):
        return np.flip(arr, axis)

    def full(shape, fill_value, **kwargs):
        return np.full(shape, fill_value)

    def gaussian(shape, sigmas, **kwargs):
        inarr = np.full(shape, 1.0)
        return scipy.ndimage.gaussian_filter(inarr, sigmas)

    def center_of_mass(inarr):
        return scipy.ndimage.center_of_mass(np.absolute(inarr))

    def exp(arr):
        return np.exp(arr)

    def conj(arr):
        return np.conj(arr)

    def save(file, arr):
        np.save(file, arr)

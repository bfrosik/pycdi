from pycohere.lib.cohlib import cohlib
import cupy as cp
import cupyx as cpx
import math
import numpy as np
from cupyx.scipy import ndimage


class cplib(cohlib):
    def array(obj):
        return cp.array(obj)

    def dot(arr1, arr2):
        return cp.dot(arr1, arr2)

    def set_device(dev_id):
        cp.cuda.Device(dev_id).use()

    def set_backend(proc):
        pass

    def to_numpy(arr):
        return cp.asnumpy(arr).T

    def from_numpy(arr):
        return cp.array(arr.T)

    def save(filename, arr):
        cp.save(filename, arr)

    def load(filename):
        return cp.load(filename)

    def dtype(arr):
        return arr.dtype

    def size(arr):
        return arr.size

    def hasnan(arr):
        return cp.any(cp.isnan(arr))
    
    def copy(arr):
        return cp.copy(arr)

    def random(shape, **kwargs):
        import time
        import os

        seed = np.array([time.time()* 10000 * os.getpid(), os.getpid()])
        rs = cp.random.RandomState(seed=seed)
        return cp.random.random(shape, dtype=cp.float32) + 1j * cp.random.random(shape, dtype=cp.float32)

    def fftshift(arr):
        return cp.fft.fftshift(arr)

    def ifftshift(arr):
        return cp.fft.fftshift(arr)

    def shift(arr, sft):
        sft = [int(s) for s in sft]
        return cp.roll(arr, sft)

    def fft(arr):
        return cp.fft.fftn(arr)

    def ifft(arr):
        return cp.fft.ifftn(arr)

    def fftconvolve(arr1, arr2):
        #return cpx.scipy.signal.convolve(arr1, arr2)
        return ndimage.convolve(arr1, arr2)

    def where(cond, x, y):
        return cp.where(cond, x, y)

    def dims(arr):
        # get array dimensions
        return arr.shape

    def absolute(arr):
        return cp.absolute(arr)

    def sqrt(arr):
        return cp.sqrt(arr)

    def square(arr):
        return cp.square(arr)

    def sum(arr, axis=None):
        return cp.sum(arr, axis)

    def real(arr):
        return cp.real(arr)

    def imag(arr):
        return cp.imag(arr)

    def amax(arr):
        return cp.amax(arr)

    def argmax(arr, axis=None):
        return cp.argmax(arr, axis)

    def unravel_index(indices, shape):
        return cp.unravel_index(indices, shape)

    def maximum(arr1, arr2):
        return cp.maximum(arr1, arr2)

    def ceil(arr):
        return cp.ceil(arr)

    def fix(arr):
        return cp.fix(arr)

    def round(val):
        return cp.round(val)

    def print(arr, **kwargs):
        print(arr)

    def angle(arr):
        return cp.angle(arr)

    def flip(arr, axis=None):
        return cp.flip(arr, axis)

    def tile(arr, rep):
        return cp.tile(arr, rep)

    def full(shape, fill_value, **kwargs):
        return cp.full(shape, fill_value)

    def gaussian(shape, sigmas, **kwargs):
        grid = np.full(shape, 1.0)
        for i in range(len(shape)):
            # prepare indexes for tile and transpose
            tile_shape = list(shape)
            tile_shape.pop(i)
            tile_shape.append(1)
            trans_shape = list(range(len(shape) - 1))
            trans_shape.insert(i, len(shape) - 1)

            multiplier = - 0.5 / pow(sigmas[i], 2)
            line = np.linspace(-(shape[i] - 1) / 2.0, (shape[i] - 1) / 2.0, shape[i])
            gi = np.tile(line, tile_shape)
            gi = np.transpose(gi, tuple(trans_shape))
            exponent = np.power(gi, 2) * multiplier
            gi = np.exp(exponent)
            grid = grid * gi

        grid_total = np.sum(grid)
        return grid / grid_total


    def center_of_mass(inarr):
        return cpx.scipy.ndimage.center_of_mass(cp.absolute(inarr))

    def meshgrid(*xi):
        return cp.meshgrid(*xi)

    def exp(arr):
        return cp.exp(arr)

    def conj(arr):
        return cp.conj(arr)

    def save(file, arr):
        cp.save(file, arr)

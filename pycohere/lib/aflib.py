from pycohere.lib.cohlib import cohlib
import arrayfire as af
import math
import numpy as np


class aflib(cohlib):
    def iifftshift(arr):
        return af.shift(arr, math.ceil(arr.dims()[0] / 2) + 1, math.ceil(arr.dims()[1] / 2) + 1,
                        math.ceil(arr.dims()[2] / 2) + 1)

    def set_device(dev_id):
        af.device.set_device(dev_id)

    def set_backend(proc):
        af.set_backend(proc)

    def to_numpy(arr):
        return arr.to_ndarray().T

    def from_numpy(arr):
        return af.np_to_af_array(arr.T)

    def asarray(arr, dtype):
        if dtype == np.float32:
            return arr.as_type(af.Dtype.c32)
        elif dtype == np.float64:
            return arr.as_type(af.Dtype.c64)
        else:
            return arr

    def random(shape, **kwargs):
        import time
        import os

        dims = [None, None, None, None]
        for i in range(len(shape)):
            dims[i] = shape[i]

        if 'dtype' in kwargs:
            if kwargs['dtype'] == np.float32:
                dtype = af.Dtype.c32
            elif kwargs['dtype'] == np.float64:
                dtype = af.Dtype.c64
            else:
                dtype = af.Dtype.c32
        else:
            dtype = af.Dtype.c32

        eng = af.random.Random_Engine(engine_type=af.RANDOM_ENGINE.DEFAULT,
                                      seed=int(time.time() * 1000000) * os.getpid() + os.getpid())
        return af.random.randn(dims[0], dims[1], dims[2], dims[3], dtype=dtype, engine=eng)

    def fftshift(arr):
        raise NotImplementedError

    def ifftshift(arr):
        raise NotImplementedError

    def shift(arr, sft):
        raise NotImplementedError

    def fft(arr):
        raise NotImplementedError

    def ifft(arr):
        raise NotImplementedError

    def fftconvolve(arr1, arr2):
        return af.fft_convolve(arr1, arr2)

    def where(cond, x, y):
        return af.select(cond, x, y)

    def dims(arr):
        # get array dimensions
        return arr.dims()

    def abs(arr):
        return af.abs(arr)

    def sqrt(arr):
        return af.sqrt(arr)

    def square(arr):
        return af.pow(arr, 2)

    def sum(arr):
        return af.sum(arr)

    def real(arr):
        return af.real(arr)

    def imag(arr):
        return af.imag(arr)

    def max(arr):
        return af.max(arr)

    def print(arr, **kwargs):
        af.display(arr)

    def replace(lhs, cond, rhs):
        return af.replace(lhs, cond, rhs)

    def arctan2(arr1, arr2):
        return af.atan2(arr1, arr2)

    def flip(arr, axis=None):
        if axis is None:
            raise NotImplementedError
        else:
            return af.flip(arr, axis)

    def full(shape, fill_value, dtype=None):
        dims = [None, None, None, None]
        for i in range(len(shape)):
            dims[i] = shape[i]
        return af.constant(fill_value, dims[0], dims[1], dims[2], dims[3], dtype=dtype)


class aflib1(aflib):
    def fftshift(arr):
        return af.shift(arr, math.ceil(arr.dims()[0] / 2) - 1)

    def ifftshift(arr):
        return af.shift(arr, math.ceil(arr.dims()[0] / 2))

    def shift(arr, sft):
        return af.shift(arr, math.ceil(sft[0]))

    def fft(arr):
        return af.fft(arr)

    def ifft(arr):
        return af.ifft(arr)

    # def random(dims, **kwargs):
    #     import time
    #     import os
    #
    #     if 'dtype' in kwargs:
    #         print('in random, dtype', kwargs['dtype'])
    #         dtype = kwargs['dtype']
    #     else:
    #         print ('no dtype')
    #         dtype = af.Dtype.c32
    #     eng = af.random.Random_Engine(engine_type=af.RANDOM_ENGINE.DEFAULT,
    #                                   seed=int(time.time() * 1000000) * os.getpid() + os.getpid())
    #     return af.random.randn(dims[0], dtype=dtype, engine=eng)

    def flip(arr, axis=None):
        if axis is None:
            return af.flip(arr, 0)
        else:
            return af.flip(arr, axis)


class aflib2(aflib):
    def fftshift(arr):
        return af.shift(arr, math.ceil(arr.dims()[0] / 2) - 1, math.ceil(arr.dims()[1] / 2) - 1)

    def ifftshift(arr):
        return af.shift(arr, math.ceil(arr.dims()[0] / 2), math.ceil(arr.dims()[1] / 2))

    def shift(arr, sft):
        return af.shift(arr, math.ceil(sft[0]), math.ceil(sft[1]))

    def fft(arr):
        return af.fft2(arr)

    def ifft(arr):
        return af.ifft2(arr)

    # def random(dims, **kwargs):
    #     import time
    #     import os
    #
    #     if 'dtype' in kwargs:
    #         if kwargs['dtype'] == np.float32:
    #             dtype = af.Dtype.f32
    #         elif kwargs['dtype'] == np.float64:
    #             dtype = af.Dtype.f64
    #         else:
    #             dtype = af.Dtype.c32
    #     else:
    #         dtype = af.Dtype.c32
    #     eng = af.random.Random_Engine(engine_type=af.RANDOM_ENGINE.DEFAULT,
    #                                   seed=int(time.time() * 1000000) * os.getpid() + os.getpid())
    #     return af.random.randn(dims[0], dims[1], dtype=dtype, engine=eng)
    #
    def flip(arr, axis=None):
        if axis is None:
            return af.flip(af.flip(arr, 0), 1)
        else:
            return af.flip(arr, axis)


class aflib3(aflib):
    def fftshift(arr):
        return af.shift(arr, math.ceil(arr.dims()[0] / 2) - 1, math.ceil(arr.dims()[1] / 2) - 1,
                        math.ceil(arr.dims()[2] / 2) - 1)

    def ifftshift(arr):
        return af.shift(arr, math.ceil(arr.dims()[0] / 2), math.ceil(arr.dims()[1] / 2), math.ceil(arr.dims()[2] / 2))

    def shift(arr, sft):
        return af.shift(arr, math.ceil(sft[0]), math.ceil(sft[1]), math.ceil(sft[2]))

    def fft(arr):
        return af.fft3(arr)

    def ifft(arr):
        return af.ifft3(arr)

    # def random(shape, **kwargs):
    #     import time
    #     import os
    #
    #     dims = [None, None, None, None]
    #     for i in range(len(shape)):
    #         dims[i] = shape[i]
    #
    #     if 'dtype' in kwargs:
    #         if kwargs['dtype'] == np.float32:
    #             dtype = af.Dtype.f32
    #         elif kwargs['dtype'] == np.float64:
    #             dtype = af.Dtype.f64
    #         else:
    #             dtype = af.Dtype.c32
    #     else:
    #         dtype = af.Dtype.c32
    #
    #     eng = af.random.Random_Engine(engine_type=af.RANDOM_ENGINE.DEFAULT,
    #                                   seed=int(time.time() * 1000000) * os.getpid() + os.getpid())
    #     return af.random.randn(dims[0], dims[1], dims[2], dims[3], dtype=dtype, engine=eng)
    #
    def flip(arr, axis=None):
        if axis is None:
            return af.flip(af.flip(af.flip(arr, 0), 1), 2)
        else:
            return af.flip(arr, axis)


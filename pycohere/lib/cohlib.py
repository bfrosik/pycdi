class cohlib:
    # Interface
    def set_device(dev_id):
        pass

    def set_backend(proc):
        pass

    def to_numpy(arr):
        pass

    def from_numpy(arr):
        pass

    def asarray(arr, dtype):
        pass

    def fftshift(arr):
        pass

    def ifftshift(arr):
        pass

    def shift(arr, sft):
        pass

    def fft(arr):
        pass

    def ifft(arr):
        pass

    def fftconvolve(arr1, arr2):
        pass

    def where(cond, x, y):
        pass

    def dims(arr):
        # get array dimensions
        pass

    def abs(arr):
        pass

    def square(arr):
        pass

    def sqrt(arr):
        pass

    def sum(arr):
        pass

    def real(arr):
        pass

    def imag(arr):
        pass

    def max(arr):
        pass

    def random(shape, **kwargs):
        pass

    def full(shape, fill_value, **kwargs):
        pass

    def print(arr, **kwargs):
        pass

    def replace(lhs, cond, rhs):
        #    Select elements from one of two arrays based on condition.
        #
        #    Parameters:
        #    lhs : af.Array or scalar
        #
        #    numerical array whose elements are replaced with rhs when conditional element is False
        #
        #    cond : af.Array
        #
        #    Conditional array
        #
        #    rhs : af.Array or scalar
        #
        #    numerical array whose elements are picked when conditional element is False
        pass

    def arctan2(arr1, arr2):
        pass

    def flip(arr, axis):
        pass

    def gaussian(dims, sigmas, **kwargs):
        pass

    def center_of_mass(arr):
        pass
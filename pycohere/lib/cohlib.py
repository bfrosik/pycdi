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

    def dtype(arr):
        pass

    def size(arr):
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

    def absolute(arr):
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

    def arctan2(arr1, arr2):
        pass

    def flip(arr, axis):
        pass

    def gaussian(dims, sigmas, **kwargs):
        pass

    def center_of_mass(arr):
        pass

class cohlib:
    # Interface
    def is_af(self):
        return False

    def set_device(dev_id):
        pass

    def set_backend(proc):
        pass

    def to_numpy(arr):
        pass

    def save(filename, arr):
        pass

    def load(filename):
        pass

    def from_numpy(arr):
        pass

    def dtype(arr):
        pass

    def size(arr):
        pass

    def hasnan(arr):
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

    def sum(arr, axis=None):
        pass

    def real(arr):
        pass

    def imag(arr):
        pass

    def amax(arr):
        pass

    def argmax(arr, axis=None):
        pass

    def maximum(arr1, arr2):
        pass

    def ceil(arr):
        pass

    def fix(arr):
        pass

    def random(shape, **kwargs):
        pass

    def full(shape, fill_value, **kwargs):
        pass

    def print(arr, **kwargs):
        pass

    def angle(arr):
        pass

    def flip(arr, axis):
        pass

    def gaussian(dims, sigmas, **kwargs):
        pass

    def center_of_mass(arr):
        pass

    def exp(arr):
        pass

    def conj(arr):
        pass

    def save(file, arr):
        pass

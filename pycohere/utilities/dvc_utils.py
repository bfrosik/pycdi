import math
import arrayfire as af


def set_lib(dlib):
    global dvclib
    dvclib = dlib


def gaussian(dims, sigmas, alpha=1):
    """
    Calculates Gaussian distribution grid in ginven dimensions.

    Parameters
    ----------
    shape : tuple
        shape of the grid
    sigmas : list
        sigmas in all dimensions
    alpha : float
        a multiplier

    Returns
    -------
    grid : ndarray
        Gaussian distribution grid
    """
    grid = af.data.constant(1.0, dims[0], dims[1], dims[2], dtype=af.Dtype.f64)
    for i in range(len(sigmas)):
        multiplier = - 0.5 * alpha / pow(sigmas[i], 2)
        exponent = dvclib.square((af.data.range(dims[0], dims[1], dims[2], dim=i) - (dims[i] - 1) / 2.0).as_type(af.Dtype.f64)) * multiplier
        grid = grid * af.arith.exp(exponent)

    grid_tot = af.sum(af.sum(af.sum(grid, dim=0), dim=1), dim=2)
    grid_total = af.tile(grid_tot, dims[0], dims[1], dims[2])
    grid = grid / grid_total
    return grid


def crop_center(arr, shape):
    nparr = af.shift(arr, math.ceil(arr.dims()[0] / 2), math.ceil(arr.dims()[1] / 2),
                     math.ceil(arr.dims()[2] / 2)).copy()

    dims = nparr.dims()
    principio = []
    finem = []
    for i in range(3):
        principio.append(int((dims[i] - shape[i]) / 2))
        finem.append(principio[i] + shape[i])
    if arr.numdims() == 1:
        return arr[principio[0]: finem[0]]
    elif arr.numdims() == 2:
        return arr[principio[0]: finem[0], principio[1]: finem[1]]
    elif arr.numdims() == 3:
        cropped = nparr[principio[0]: finem[0], principio[1]: finem[1], principio[2]: finem[2]]
    else:
        raise NotImplementedError
    return cropped


def center_of_mass(input):
    arr = af.abs(input)
    normalizer = af.sum(arr)
    t_dims = list(arr.dims())
    mod_dims = [1] * len(t_dims)
    com = []

    for dim in range(len(t_dims)):
        # swap
        mod_dims[dim] = t_dims[dim]
        t_dims[dim] = 1
        grid = af.iota(mod_dims[0], mod_dims[1], mod_dims[2], tile_dims=t_dims)
#        print(grid)
        com.append(af.sum(grid * arr) / normalizer)
        # swap back
        t_dims[dim] = mod_dims[dim]
        mod_dims[dim] = 1

    return com


def get_max(arr):
    import arrayfire as af

    return af.imax(arr)

import numpy as np
from scipy import ndimage
import arrayfire as af

def center_of_mass(inp):
    arr = af.abs(inp)
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

    grid1 = af.iota(mod_dims[0], mod_dims[1], mod_dims[2])

    return com

input = np.array(([[0,0,0,0,0],
[0,1,1,0,0],
[0,1,1,0,0],
[0,1,1,0,0]],[[0,0,0,0,0],
[0,1,1,0,0],
[0,1,1,0,0],
[0,1,1,0,0]],
[[0,0,0,0,0],
[0,1,1,0,0],
[0,1,1,0,0],
[0,1,1,0,0]]))

# normalizer = np.sum(input)

grids = np.ogrid[[slice(0, i) for i in input.shape]]
print (grids)

#load input on device
arr = af.np_to_af_array(input.T)

print (center_of_mass(arr), ndimage.measurements.center_of_mass(input))


normalizer = af.sum(arr)
t_dims = list(arr.dims())
mod_dims = [1] * len(t_dims)

for dim in range(len(t_dims)):
    # swap
    mod_dims[dim] = t_dims[dim]
    t_dims[dim] = 1
    print(mod_dims, t_dims)
    grid = af.iota(mod_dims[0], mod_dims[1], mod_dims[2], tile_dims=t_dims)
    af.display(grid)



af.display(af.iota(mod_dims[0], mod_dims[1], mod_dims[2]))

# results = [np.sum(input * grids[dir].astype(float)) / normalizer
#            for dir in range(input.ndim)]
#
# if numpy.isscalar(results[0]):
#     return tuple(results)
#
# return [tuple(v) for v in numpy.array(results).T]

# d_type
# multiplier = - 0.5 * alpha / pow(sgma[0], 2);
# af::array
# exponent = pow((range(data_dim, 0) - (data_dim[0] - 1) / 2.0).as(f64), 2)*multiplier;
# af::array
# grid = exp(exponent);
#
# // add
# grid in other
# dimensions
# for (uint i = 1; i < nD; i++)
# {
#     multiplier = - 0.5 * alpha / pow(sgma[i], 2);
# exponent = pow((range(data_dim, i) - (data_dim[i] - 1) / 2.0).as(f64), 2)*multiplier;
# af::array
# gi = exp(exponent);
# grid = grid * gi;
# }
# d_type
# grid_total = sum < d_type > (grid);
# grid = grid / grid_total;
# return grid;
#
#

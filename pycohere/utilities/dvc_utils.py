def set_lib(dlib):
    global dvclib
    dvclib = dlib


def crop_center(arr, shape):
    dims = dvclib.dims(arr)
    principio = []
    finem = []
    for i in range(3):
        principio.append(int((dims[i] - shape[i]) / 2))
        finem.append(principio[i] + shape[i])
    if len(shape) == 1:
        return arr[principio[0]: finem[0]]
    elif len(shape) == 2:
        return arr[principio[0]: finem[0], principio[1]: finem[1]]
    elif len(shape) == 3:
        cropped = arr[principio[0]: finem[0], principio[1]: finem[1], principio[2]: finem[2]]
    else:
        raise NotImplementedError
    return cropped


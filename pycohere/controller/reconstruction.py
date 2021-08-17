# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This module controls a single reconstruction process.
"""

import numpy as np
import os
import importlib
import pycohere.controller.rec as calc
import pycohere.utilities.utils as ut
from pycohere.controller.params import Params

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['single_rec',
           'reconstruction']


def set_lib(pkg, ndim=None):
    global devlib
    if pkg == 'af':
        if ndim == 1:
            devlib = importlib.import_module('pycohere.lib.aflib').aflib1
        elif ndim == 2:
            devlib = importlib.import_module('pycohere.lib.aflib').aflib2
        elif ndim == 3:
            devlib = importlib.import_module('pycohere.lib.aflib').aflib3
        else:
            raise NotImplementedError
    elif pkg == 'cp':
        devlib = importlib.import_module('pycohere.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('pycohere.lib.nplib').nplib
    calc.set_lib(devlib, pkg=='af')


def single_rec(proc, save_dir, data, pars, dev, continue_dir=None):
    """
    This function starts reconstruction and returns results.

    Parameters
    ----------
    proc : str
        a string indicating the processor type (cpu, cuda or opencl)

    data : numpy array
        data array

    pars : Object
        Params object containing parsed configuration

    dev : int
        id defining the GPU this reconstruction will be utilizing, or -1 if running cpu or the gpu assignment is left to OS

    image : numpy array
        reconstructed image for further reconstruction, or None if initial

    support : numpy array
        support of previous reconstructed image, or None

    coh : numpy array
        coherence of previous reconstructed images, or None

    Returns
    -------
    image : numpy array
        reconstructed image
    support : numpy array
        support of reconstructed images
    coh : numpy array
        coherence of reconstructed images
    er : list
        a vector containing errors for each iteration
    reciprocal : ndarray
        the array converted to reciprocal space
    flow : ndarray
        info to scientist/developer; a list of functions  that can run in one iterations (excluding inactive features)
    iter_array : ndarray
        info to scientist/developer; an array of 0s and 1s, 1 meaning the function in flow will be executed in iteration, 0 otherwise
    """
    # devlib.set_backend(proc)
    # if dev != -1:
    #     devlib.set_device(dev)
    #
    # worker = calc.Rec(pars, devlib.from_numpy(devlib.absolute(data)), continue_dir)
    # ret_code = worker.iterate()
    # if ret_code != 0:
    #     worker.save_res(save_dir)
    #
    # return ret_code
    #

def reconstruction(lib, conf_file, datafile, dir, dev):
    """
    Controls single reconstruction.

    This function checks whether the reconstruction is continuation or initial reconstruction. If continuation, the arrays of image, support, coherence are read from cont_directory, otherwise they are initialized to None.
    It starts thr reconstruction and saves results.

    Parameters
    ----------
    proc : str
        a string indicating the processor type (cpu, cuda or opencl)

    conf_file : str
        configuration file name

    datafile : str
        data file name

    dir : str
        a parent directory that holds the reconstructions. It can be experiment directory or scan directory.

    dev : int
        id defining the GPU this reconstruction will be utilizing, or -1 if running cpu or the gpu assignment is left to OS


    Returns
    -------
    nothing
    """
    pars = Params(conf_file)
    er_msg = pars.set_params()
    if er_msg is not None:
        return er_msg

    if lib == 'af' or lib == 'cpu' or lib == 'opencl' or lib == 'cuda':
        if datafile.endswith('tif') or datafile.endswith('tiff'):
            try:
                data = ut.read_tif(datafile)
            except:
                print ('could not load data file', datafile)
                return
        elif datafile.endswith('npy'):
            try:
                data = np.load(datafile)
            except:
                print ('could not load data file', datafile)
                return
        else:
            print ('no data file found')
            return
        print('data shape', data.shape)
        set_lib('af', len(data.shape))
        if lib != 'af':
            devlib.set_backend(lib)
    else:
        set_lib(lib)

    if not pars.cont:
        continue_dir = None
    else:
        continue_dir = pars.continue_dir
    print('cnt dir', continue_dir)

    try:
        save_dir = pars.save_dir
    except AttributeError:
        filename = conf_file.split('/')[-1]
        save_dir = os.path.join(dir, filename.replace('config_rec', 'results'))

    worker = calc.Rec(pars, datafile)

    try:
        worker.init_dev(dev[0])
    except EnvironmentError:
        print ('cannot load on device ID', dev)
        return
    except ValueError:
        print('could not load data file', datafile)
        return

    worker.init(continue_dir)
    ret_code = worker.iterate()
    if ret_code == 0:
        worker.save_res(save_dir)



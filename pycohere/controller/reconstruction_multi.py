# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This module controls the multi reconstruction process.
"""

import os
import numpy as np
import importlib
import pycohere.utilities.utils as ut
import pycohere.controller.rec as calc
from multiprocessing import Pool, Queue
from functools import partial
from pycohere.controller.params import Params


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['single_rec_process',
           'assign_gpu',
           'multi_rec',
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


def single_rec_process(metric_type, gen, rec_attrs):
    """
    This function runs a single reconstruction process.

    Parameters
    ----------
    proc : str
        string defining library used 'cpu' or 'opencl' or 'cuda'

    pars : Object
        Params object containing parsed configuration

    data : numpy array
        data array

    req_metric : str
        defines metric that will be used if GA is utilized

    dirs : list
        tuple of two elements: directory that contain results of previous run or None, and directory where the results of this processing will be saved

    Returns
    -------
    metric : float
        a calculated characteristic of the image array defined by the metric
    """
    worker, prev_dir, save_dir = rec_attrs
    if worker.init_dev(gpu) < 0:
        worker = None
        metric = None
    else:
        worker.init(prev_dir, gen)

        if gen is not None and gen > 0:
            worker.breed()

        ret_code = worker.iterate()
        if ret_code == 0:
            worker.save_res(save_dir)
            metric = worker.get_metric(metric_type)
        else:    # bad reconstruction
            metric = None
        worker = None    # TODO check if this clear the GPU
    return metric


def assign_gpu(*args):
    """
    This function dequeues GPU id from given queue and makes it global, thus associating it with the process.

    Parameters
    ----------
    q : Queue
        a queue holding GPU ids assigned to consequitive processes

    Returns
    -------
    nothing
    """
    q = args[0]
    global gpu
    gpu = q.get()


def multi_rec(save_dir, devices, workers, prev_dirs, metric_type='chi', gen=None):
    """
    This function controls the multiple reconstructions.

    Parameters
    ----------
    save_dir : str
        a directory where the subdirectories will be created to save all the results for multiple reconstructions

    proc : str
        a string indicating the processor type (cpu, cuda or opencl)

    data : numpy array
        data array

    pars : Object
        Params object containing parsed configuration

    devices : list
        list of GPUs available for this reconstructions

    previous_dirs : list
        list directories that hols results of previous reconstructions if it is continuation

    metric : str
        a metric defining algorithm by which to evaluate the image array

    Returns
    -------
    save_dirs : list
        list of directories where to save results
    evals : list
        list of evaluation results of image arrays
    """
    evals = []

    def collect_result(result):
        for r in result:
            evals.append(r)

    iterable = []
    save_dirs = []

    for i in range(len(workers)):
        save_sub = os.path.join(save_dir, str(i))
        save_dirs.append(save_sub)
        iterable.append((workers[i], prev_dirs[i], save_sub))
    func = partial(single_rec_process, metric_type, gen)
    q = Queue()
    for device in devices:
        q.put(device)
    with Pool(processes=len(devices), initializer=assign_gpu, initargs=(q,)) as pool:
        pool.map_async(func, iterable, callback=collect_result)
        q.close()
        pool.close()
        pool.join()
        pool.terminate()

    # remove the unsuccessful reconstructions
    for i, e in reversed(list(enumerate(evals))):
        if e is None:
            evals.pop(i)
            save_dirs.pop(i)

    return save_dirs, evals


def reconstruction(lib, conf_file, datafile, dir, devices):
    """
    This function controls multiple reconstructions.

    Parameters
    ----------
    proc : str
        processor to run on (cpu, opencl, or cuda)

    conf_file : str
        configuration file with reconstruction parameters

    datafile : str
        name of the file with initial data

    dir : str
        a parent directory that holds the reconstructions. It can be experiment directory or scan directory.

    devices : list
        list of GPUs available for this reconstructions

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

    try:
        reconstructions = pars.reconstructions
    except:
        reconstructions = 1

    prev_dirs = []
    if pars.cont:
        continue_dir = pars.continue_dir
        for sub in os.listdir(continue_dir):
            image, support, coh = ut.read_results(os.path.join(continue_dir, sub) + '/')
            if image is not None:
                prev_dirs.append(sub)
    else:
        for _ in range(reconstructions):
            prev_dirs.append(None)
    try:
        save_dir = pars.save_dir
    except AttributeError:
        filename = conf_file.split('/')[-1]
        save_dir = os.path.join(dir, filename.replace('config_rec', 'results'))

    workers = [calc.Rec(pars, datafile) for _ in range(reconstructions)]

    save_dirs, evals = multi_rec(save_dir, devices, workers, prev_dirs)

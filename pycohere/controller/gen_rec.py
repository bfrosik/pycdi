# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
This module controls the genetic algoritm process.
"""

import numpy as np
import os
import pycohere.controller.reconstruction as single
import pycohere.controller.reconstruction_multi as multi
import pycohere.utilities.utils as ut
import multiprocessing as mp
import shutil
import importlib
import pycohere.controller.rec as calc
from pycohere.controller.params import Params


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Generation.next_gen',
           'Generation.get_data',
           'Generation.get_gmask',
           'Generation.order,'
           'Generation.breed_one',
           'Generation.breed',
           'reconstruction']


def set_lib(pkg, ndim=None):
    global dvclib
    if pkg == 'af':
        if ndim == 1:
            dvclib = importlib.import_module('pycohere.lib.aflib').aflib1
        elif ndim == 2:
            dvclib = importlib.import_module('pycohere.lib.aflib').aflib2
        elif ndim == 3:
            dvclib = importlib.import_module('pycohere.lib.aflib').aflib3
        else:
            raise NotImplementedError
    elif pkg == 'cp':
        dvclib = importlib.import_module('pycohere.lib.cplib').cplib
    elif pkg == 'np':
        dvclib = importlib.import_module('pycohere.lib.nplib').nplib
    calc.set_lib(dvclib, pkg=='af')


class Generation:
    """
    This class holds fields relevant to generations according to configuration and encapsulates generation functionality.
    """

    def __init__(self, pars):
        """
        Constructor, parses GA parameters from configuration file and saves as class members.
        """
        self.current_gen = 0
        self.generations = pars.generations
        self.metrics = tuple(pars.metrics)
        self.ga_reconstructions = pars.ga_reconstructions
        self.ga_support_thresholds = tuple(pars.ga_support_thresholds)
        self.ga_support_sigmas = tuple(pars.ga_support_sigmas)
        self.breed_modes = tuple(pars.breed_modes)
        self.low_resolution_generations = pars.low_resolution_generations
        if self.low_resolution_generations > 0:
            self.low_resolution_alg = pars.low_resolution_alg
            self.sigmas = pars.ga_low_resolution_sigmas


    def next_gen(self):
        """
        Advances to the next generation.

        Parameters
        ----------
        none

        Returns
        -------
        nothing
        """
        self.current_gen += 1


    def get_data(self, data):
        """
        If low resolution feature is enabled (i.e. when ga_low_resolution_sigmas parameter is defined) the data is modified in each generation that has the sigma provided in the ga_low_resolution_sigmas list. The data is multiplied by gaussian distribution calculated using the defined sigma.

        Parameters
        ----------
        data : ndarray
            experiment data array, prepared and formatted

        Returns
        -------
        data : ndarray
            modified data
        """
        if self.current_gen >= self.low_resolution_generations:
            return data
        else:
            gmask = self.get_gmask(data.shape)
            return data * gmask

    def get_gmask(self, shape):
        """
        It calculates the gaussian distribution for low resolution generations.

        Parameters
        ----------
        shape : tuple
            shape of the data array

        Returns
        -------
        mask : ndarray
            array of the same size as data containing Gaussian distribution accross all dimensions for low resolution generations, or array of 1s for other generations.
        """
        if self.low_resolution_alg == 'GAUSS':
            if self.sigmas[self.current_gen] < 1.0:
                ut.gaussian(shape, self.sigmas[self.current_gen])
            else:
                return np.ones(shape)


    def order_dirs(self, dirs, evals):
        """
        Orders results in generation directory in subdirectories numbered from 0 and up, the best result stored in the '0' subdirectory. The ranking is done by numbers in evals list, which are the results of the generation's metric to the image array.

        Parameters
        ----------
        dirs : list
            list of directories where the reconstruction results files are saved
        evals : list
            list of evaluation of the results in the directories from the dirs list. The evaluation is a number calculated for metric configured for this generation

        Returns
        -------
        nothing
        """
        metric = self.metrics[self.current_gen]
        # ranks keeps indexes of reconstructions from best to worst
        # for most of the metric types the minimum of the metric is best, but for
        # 'summed_phase' and 'area' it is oposite, so reversing the order
        ranks = np.argsort(evals).tolist()
        if metric == 'summed_phase' or metric == 'area':
            ranks.reverse()

        # all the generation directories are in the same parent directory
        parent_dir = os.path.abspath(os.path.join(dirs[0], os.pardir))
        rank_dirs = []
        # append "_<rank>" to each result directory name
        for i in range(len(ranks)):
            dest = os.path.join(parent_dir, str(i) + '_' + str(ranks[i]))
            shutil.move(dirs[i], dest)
            rank_dirs.append(dest)

        # remove the number preceding rank from each directory name, so the directories are numbered
        # according to rank
        for dir in rank_dirs:
            last_sub = os.path.basename(dir)
            dest = os.path.join(parent_dir, last_sub.split('_')[-1])
            shutil.move(dir, dest)


    def order_processes(self, proc_metrics):
        """
        Orders results in generation directory in subdirectories numbered from 0 and up, the best result stored in the '0' subdirectory. The ranking is done by numbers in evals list, which are the results of the generation's metric to the image array.

        Parameters
        ----------
        dirs : list
            list of directories where the reconstruction results files are saved
        evals : list
            list of evaluation of the results in the directories from the dirs list. The evaluation is a number calculated for metric configured for this generation

        Returns
        -------
        nothing
        """
        ranked_proc = sorted(proc_metrics.items(), key=lambda x: x[1], reverse=False)
        metric_type = self.metrics[self.current_gen]
        # ranks keeps indexes of reconstructions from best to worst
        # for most of the metric types the minimum of the metric is best, but for
        # 'summed_phase' and 'area' it is oposite, so reversing the order
        # procs, metrics = zip(*proc_metrics)
        # ranks = np.argsort(metrics).tolist()
        if metric_type == 'summed_phase' or metric_type == 'area':
            ranked_proc.reverse()
        # proc_ranks = list(zip(procs, ranks))
        return ranked_proc


    def cull(self, lst):
        print('breeding generation ', (self.current_gen + 1))

        if len(lst) <= self.ga_reconstructions[self.current_gen]:
            return lst
        else:
            return lst[0:self.ga_reconstructions[self.current_gen]]


def reconstruction(lib, conf_file, datafile, dir, devices):
    """
    This function controls reconstruction utilizing genetic algorithm.

    Parameters
    ----------
    proc : str
        processor to run on (cpu, opencl, or cuda)

    conf_file : str
        configuration file with reconstruction parameters

    datafile : str
        name of the file with initial data

    dir : str
        a parent directory that holds the generations. It can be experiment directory or scan directory.

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
        set_lib('af', len(data.shape))
        if lib != 'af':
            dvclib.set_backend(lib)
    else:
        set_lib(lib)

    try:
        reconstructions = pars.reconstructions
    except:
        reconstructions = 1

    gen_obj = Generation(pars)

    try:
        save_dir = pars.save_dir
    except AttributeError:
        filename = conf_file.split('/')[-1]
        save_dir = os.path.join(dir, filename.replace('config_rec', 'results'))
        #    temp_dir = os.path.join(save_dir, 'temp')

    generations = pars.generations

    # init starting values
    # if multiple reconstructions configured (typical for genetic algorithm), use "reconstruction_multi" module
    if reconstructions > 1:
        if pars.ga_fast:  # NEW the number of processes is the same as available GPUs (can be same GPU if can fit more recs)
            reconstructions = min(reconstructions, len(devices))
            workers = [calc.Rec(pars, datafile) for _ in range(reconstructions)]
            #            for worker in workers:
            #                worker.init_dev(devices.pop())
            processes = {}

            for worker in workers:
                worker_qin = mp.Queue()
                worker_qout = mp.Queue()
                process = mp.Process(target=worker.fast_ga, args=(worker_qin, worker_qout))
                process.start()
                processes[process.pid] = [worker_qin, worker_qout]

            prev_dirs = None
            for g in range(generations):
                if g == 0:
                    for pid in processes:
                        worker_qin = processes[pid][0]
                        worker_qin.put(('init_dev', devices.pop()))
                    bad_processes = []
                    for pid in processes:
                        worker_qout = processes[pid][1]
                        ret = worker_qout.get()
                        if ret < 0:
                            worker_qin = processes[pid][0]
                            worker_qin.put('done')
                            bad_processes.append(pid)
                    # remove bad processes from dict (in the future we may reuse them)
                    for pid in bad_processes:
                        processes.pop(pid)
                for pid in processes:
                    worker_qin = processes[pid][0]
                    if prev_dirs is None:
                        prev_dir = None
                    else:
                        prev_dir = prev_dirs[pid]
                    worker_qin.put(('init', prev_dir, g))
                for pid in processes:
                    worker_qout = processes[pid][1]
                    ret = worker_qout.get()
                if g > 0:
                    for pid in processes:
                        worker_qin = processes[pid][0]
                        worker_qin.put('breed')
                    for pid in processes:
                        worker_qout = processes[pid][1]
                        ret = worker_qout.get()
                for pid in processes:
                    worker_qin = processes[pid][0]
                    worker_qin.put('iterate')
                bad_processes = []
                for pid in processes:
                    worker_qout = processes[pid][1]
                    ret = worker_qout.get()
                    if ret < 0:
                        worker_qin = processes[pid][0]
                        worker_qin.put('done')
                        bad_processes.append(pid)
                # remove bad processes from dict (in the future we may reuse them)
                for pid in bad_processes:
                    processes.pop(pid)
                # get metric, i.e the goodness of reconstruction from each run
                proc_metrics = {}
                for pid in processes:
                    worker_qin = processes[pid][0]
                    metric_type = pars.metrics[g]
                    worker_qin.put(('get_metric', metric_type))
                for pid in processes:
                    worker_qout = processes[pid][1]
                    metric = worker_qout.get()
                    proc_metrics[pid] = metric
                # order processes by metric
                proc_ranks = gen_obj.order_processes(proc_metrics)
                # cull
                culled_proc_ranks = gen_obj.cull(proc_ranks)
                # remove culled processes from list (in the future we may reuse them)
                for i in range(len(culled_proc_ranks), len(proc_ranks)):
                    pid = proc_ranks[i][0]
                    worker_qin = processes[pid][0]
                    worker_qin.put('done')
                    processes.pop(pid)
                # save results, we may modify it later to save only some
                gen_save_dir = os.path.join(save_dir, 'g_' + str(g))
                prev_dirs = {}
                for i in range(len(culled_proc_ranks)):
                    pid = culled_proc_ranks[i][0]
                    worker_qin = processes[pid][0]
                    worker_qin.put(('save_res', os.path.join(gen_save_dir, str(i))))
                    prev_dirs[pid] = os.path.join(gen_save_dir, str(i))
                for pid in processes:
                    worker_qout = processes[pid][1]
                    ret = worker_qout.get()
                if len(processes) == 0:
                    break
                gen_obj.next_gen()
            for pid in processes:
                worker_qin = processes[pid][0]
                worker_qin.put('done')
        else:
            rec = multi
            prev_dirs = []
            for _ in range(reconstructions):
                prev_dirs.append(None)
            for g in range(generations):
                gen_save_dir = os.path.join(save_dir, 'g_' + str(g))
                metric_type = gen_obj.metrics[g]
                workers = [calc.Rec(pars, datafile) for _ in range(len(prev_dirs))]
                prev_dirs, evals = rec.multi_rec(gen_save_dir, devices, workers, prev_dirs, metric_type, g)

                # results are saved in a list of directories - save_dir
                # it will be ranked, and moved to temporary ranked directories
                gen_obj.order_dirs(prev_dirs, evals)
                prev_dirs = gen_obj.cull(prev_dirs)
                gen_obj.next_gen()
    else:
        image = None
        support = None
        coh = None
        rec = single

        for g in range(generations):
            gen_data = gen_obj.get_data(data)
            ret_code = rec.single_rec(lib, os.path.join(save_dir, 'g_' + str(g)), gen_data, pars,
                                                                               devices[0], image, support, coh)
            if ret_code != 0:
                return

            gen_obj.next_gen()

    print('done gen')

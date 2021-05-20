# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
This module controls the reconstruction process. The user has to provide parameters such as type of processor, data, and configuration.
The processor specifies which library will be used by FM (Fast Module) that performs the processor intensive calculations. The module can be run on cpu, or gpu. Depending on the gpu hardware and library, one can use opencl or cuda library.
"""

import numpy as np
import time
import pycohere.utilities.utils as ut
import pycohere.utilities.dvc_utils as dvut
import pycohere.controller.op_flow as of
import importlib

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['fast_module_reconstruction', ]


def set_lib(pkg, ndim):
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
    dvut.set_lib(devlib)


def get_norm(arr):
    return np.sum(np.pow(np.abs(arr), 2))


class Pcdi:
    def __init__(self, params, coh_arr=None):
        self.params = params
        self.kernel = coh_arr

    def af_init(self, data):
        self.dims = devlib.dims(data)
        centered = devlib.ifftshift(data).copy()
        self.roi_data = dvut.crop_center(centered, self.params.partial_coherence_roi)
        if self.params.partial_coherence_normalize:
            self.sum_roi_data = devlib.sum(devlib.square(self.roi_data))
        if self.kernel is None:
            self.kernel = devlib.full(self.params.partial_coherence_roi, 0.5, dtype=data.dtype())

    def set_previous(self, abs_amplitudes):
        centered = devlib.ifftshift(abs_amplitudes).copy()
        self.roi_amplitudes_prev = dvut.crop_center(centered, self.params.partial_coherence_roi)

    def apply_partial_coherence(self, abs_amplitudes):
        abs_amplitudes_2 = devlib.square(abs_amplitudes)
        converged_2 = devlib.fftconvolve(abs_amplitudes_2, self.kernel)
        converged_2 = devlib.where(converged_2 < 0, 0, converged_2)
        converged = devlib.sqrt(converged_2)
        return converged

    def update_partial_coherence(self, abs_amplitudes):
        centered = devlib.ifftshift(abs_amplitudes).copy()
        roi_amplitudes = dvut.crop_center(centered, self.params.partial_coherence_roi)
        roi_combined_amp = 2 * roi_amplitudes - self.roi_amplitudes_prev
        if self.params.partial_coherence_normalize:
            amplitudes_2 = devlib.square(roi_combined_amp)
            sum_ampl = devlib.sum(amplitudes_2)
            ratio = self.sum_roi_data / sum_ampl
#            print(ratio, self.sum_roi_data, sum_ampl)
            amplitudes = devlib.sqrt(amplitudes_2 * ratio)
        else:
            amplitudes = roi_combined_amp

        if self.params.partial_coherence_type == "LUCY":
            self.lucy_deconvolution(devlib.square(amplitudes), devlib.square(self.roi_data),
                                    self.params.partial_coherence_iteration_num)

    def lucy_deconvolution(self, amplitudes, data, iterations):
        data_mirror = devlib.flip(data).copy()
        for i in range(self.params.partial_coherence_iteration_num):
            conv = devlib.fftconvolve(self.kernel, data)
            devlib.where(conv == 0, 1.0, conv)
            relative_blurr = amplitudes / conv
            self.kernel = self.kernel * devlib.fftconvolve(relative_blurr, data_mirror)
        self.kernel = devlib.real(self.kernel)
        coh_sum = devlib.sum(devlib.abs(self.kernel))
        self.kernel = devlib.abs(self.kernel) / coh_sum
        # devlib.print(self.kernel)


class Support:
    def __init__(self, params, dims):
        self.params = params
        support_area = params.support_area
        init_support = []
        for i in range(len(support_area)):
            if type(support_area[0]) == int:
                init_support.append(support_area[i])
            else:
                init_support.append(int(support_area[i] * dims[i]))

        # create initial support as ndarray
        center = np.ones(init_support, dtype=int)
        self.support = ut.get_zero_padded_centered(center, dims)
        # The sigma can change if resolution trigger is active. When it
        # changes the distribution has to be recalculated using the given sigma
        # At some iteration the low resolution become inactive, and the sigma
        # is set to support_sigma. The prev_sigma is kept to check if sigma changed
        # and thus the distribution must be updated
        self.distribution = None
        self.prev_sigma = 0

    def af_init(self):
        self.support = devlib.from_numpy(self.support)
        self.dims = devlib.dims(self.support)

    def get_support(self):
        return self.support

    def get_distribution(self, dims, sigma):
        sigmas = []
        for i in range(len(dims)):
            sigmas.append(dims[i] / (2.0 * np.pi * self.params.support_sigma))
        dist = devlib.gaussian(dims, sigmas)

        return dist

    def gauss_conv_fft(self, ds_image_abs):
        image_sum = devlib.sum(ds_image_abs)
        shifted = devlib.ifftshift(ds_image_abs)
        rs_amplitudes = devlib.fft(shifted)
        rs_amplitudes_cent = devlib.ifftshift(rs_amplitudes)
        amp_dist = rs_amplitudes_cent * self.distribution
        shifted = devlib.ifftshift(amp_dist)
        convag_compl = devlib.ifft(shifted)
        convag = devlib.ifftshift(convag_compl)
        convag = devlib.real(convag)
        convag = devlib.where(convag > 0, convag, 0.0)
        correction = image_sum / devlib.sum(convag)
        convag *= correction
        return convag

    def update_amp(self, ds_image, sigma):
        if sigma != self.prev_sigma:
            self.distribution = self.get_distribution(self.dims, sigma)
            self.prev_sigma = sigma
        convag = self.gauss_conv_fft(devlib.abs(ds_image))
        max_convag = devlib.max(convag)
        convag = convag / max_convag
        self.support = devlib.where(self.support > 0, 0, self.support)
        self.support = devlib.where(convag < self.params.support_threshold, self.support, 1)

    def update_phase(self, ds_image):
        phase = devlib.arctan2(devlib.imag(ds_image), devlib.real(ds_image))
        phase_condition = (phase > self.params.phase_min) & (phase < self.params.phase_max)
        self.support *= phase_condition


class Rec:
    def __init__(self, params, shape, first_run):
        iter_functions = [self.next,
                          self.resolution_trigger,
                          self.shrink_wrap_trigger,
                          self.phase_support_trigger,
                          self.to_reciprocal_space,
                          self.pcdi_trigger,
                          self.pcdi,
                          self.no_pcdi,
                          self.set_prev_pcdi_trigger,
                          self.to_direct_space,
                          self.er,
                          self.hio,
                          self.twin_trigger,
                          self.average_trigger,
                          self.progress_trigger]

        flow_items_list = []
        for f in iter_functions:
            flow_items_list.append(f.__name__)

        flow = of.get_flow_arr(params, flow_items_list, first_run)

        self.flow = []
        (op_no, iter_no) = flow.shape
        for i in range(iter_no):
            for j in range(op_no):
                if flow[j, i] == 1:
                    self.flow.append(iter_functions[j])

        self.aver = None
        self.iter = -1
        self.errs = []
        self.params = params
        self.sigma = self.params.support_sigma
        self.support_obj = Support(self.params, shape)
        if self.params.is_pcdi:
            self.pcdi_obj = Pcdi(self.params)


    def dev_init(self, proc, device, data, first):
        devlib.set_backend(proc)
        if device != -1:
            devlib.set_device(device)

        data_r = devlib.from_numpy(data)
        self.data = devlib.abs(data_r)
#        print('data norm', self.get_norm(self.data))
        if self.params.ll_sigmas is not None:
            self.iter_data = self.data.copy()
        else:
            self.iter_data = self.data

        if self.params.is_pcdi:
            self.pcdi_obj.af_init(self.data)
        self.support_obj.af_init()

        self.dims = self.data.dims()
        self.ds_image = devlib.random(self.dims, dtype=data.dtype)

        norm_data = self.get_norm(self.data)
        num_points = self.data.elements()
        if (first):
            max_data = devlib.max(self.data)
            self.ds_image *= self.get_norm(self.ds_image) * max_data

            # temp = self.support_obj.get_support().copy()
            # self.ds_image = devlib.asarray(temp, dtype=data.dtype)

            print ('ds_image af type', self.ds_image.dtype())
            self.ds_image *= self.support_obj.get_support()

        # print('in init , data type, norm', self.data.type(), self.get_norm(self.data))
        # print('ds_image norm', self.get_norm(self.ds_image))

    def iterate(self):
        start_t = time.time()
        for f in self.flow:
            f()

        print('iterate took ', (time.time() - start_t), ' sec')

        if self.aver is not None:
            ratio = self.get_ratio(devlib.from_numpy(self.aver), devlib.abs(self.ds_image))
            self.ds_image *= ratio / self.aver_iter

        return devlib.to_numpy(self.ds_image), devlib.to_numpy(self.support_obj.get_support()), self.errs

    def next(self):
#        print('******** next')
        self.iter = self.iter + 1
        # the sigma used when recalculating support and data can be modified
        # by resolution trigger. So set the params to the configured values at the beginning
        # of iteration
        if self.params.ll_sigmas is not None:
            self.sigma = self.params.support_sigma
        if self.params.ll_dets is not None:
            self.iter_data = self.data

    def resolution_trigger(self):
#        print('******** res_trig')
        if self.params.ll_dets is not None:
            sigmas = []
            for i in range(len(self.dims)):
                sigmas.append(self.dims[i] * self.params.ll_dets[self.iter])
            distribution = devlib.gaussian(self.dims, sigmas)
            max_el = devlib.max(distribution)
            distribution = distribution / max_el
            data_shifted = devlib.ifftshift(self.data)
            masked = distribution * data_shifted
            self.iter_data = devlib.ifftshift(masked)

        if self.params.ll_sigmas is not None:
            self.sigma = self.params.ll_sigmas[self.iter]

    def shrink_wrap_trigger(self):
#        print('******* shrink wrap')
        self.support_obj.update_amp(self.ds_image, self.sigma)
        print ('shrink wrap, support norm', get_norm(self.support_obj.get_support()))

    def phase_support_trigger(self):
#        print('****** phase trig')
        self.support_obj.update_phase(self.ds_image)

    def to_reciprocal_space(self):
#        print('******** to recip')
        dims = self.ds_image.dims()
        self.rs_amplitudes = devlib.ifft(self.ds_image) * self.data.elements()
#        print('ampl norm', self.get_norm(self.rs_amplitudes))
        #  devlib.print(self.rs_amplitudes)

    def pcdi_trigger(self):
#        print('**** pcdi trigger')
        self.pcdi_obj.update_partial_coherence(devlib.abs(self.rs_amplitudes).copy())

    def pcdi(self):
#        print('**** pcdi')
        abs_amplitudes = devlib.abs(self.rs_amplitudes).copy()
        converged = self.pcdi_obj.apply_partial_coherence(abs_amplitudes)
        ratio = self.get_ratio(self.iter_data, devlib.abs(converged))
        error = self.get_norm(
            devlib.where((converged > 0), (devlib.abs(converged) - self.iter_data), 0)) / self.get_norm(self.iter_data)
        self.errs.append(error)
        self.rs_amplitudes *= ratio

    def no_pcdi(self):
#        print('********** no pcdi')
        ratio = self.get_ratio(self.iter_data, devlib.abs(self.rs_amplitudes))
        # consider moving the error operation to cpu
        error = self.get_norm(devlib.where((self.rs_amplitudes > 0), (devlib.abs(self.rs_amplitudes) - self.iter_data),
                                           0)) / self.get_norm(self.iter_data)
#        print('error', error)
        self.errs.append(error)
        self.rs_amplitudes = self.rs_amplitudes * ratio
#        print ('rs_ampl norm', get_norm(self.rs_amplitudes))

    def set_prev_pcdi_trigger(self):
#        print('***** set prev')
        self.pcdi_obj.set_previous(devlib.abs(self.rs_amplitudes).copy())

    def to_direct_space(self):
#        print('********** to direct')
        dims = self.rs_amplitudes.dims()
        self.ds_image_raw = devlib.fft(self.rs_amplitudes) / self.data.elements()
#        print('image_raw norm', self.get_norm(self.ds_image_raw))
        #   devlib.print(self.ds_image_raw)

    def er(self):
        print('********** er')
        self.ds_image = self.ds_image_raw * self.support_obj.get_support()
#        print('image norm', self.get_norm(self.ds_image))

    def hio(self):
        print('*********** hio')
#        print('image, image_raw norm', self.get_norm(self.ds_image), self.get_norm(self.ds_image_raw))
        #adj_calc_image = self.ds_image_raw
        adj_calc_image = self.ds_image_raw * self.params.beta
#        print('adj_calc_image', self.get_norm(adj_calc_image))
        combined_image = self.ds_image - adj_calc_image
#        print('combined image norm', self.get_norm(combined_image))
        support = self.support_obj.get_support()
#        print('support', self.get_norm(support))
        self.ds_image = devlib.where((support > 0), self.ds_image_raw, combined_image)
#        print('image norm', self.get_norm(self.ds_image))

    def twin_trigger(self):
#        print('******** twin trig')
        # mass center self.ds_image
        com = devlib.center_of_mass(self.ds_image)
        print('com',com)
        self.ds_image = devlib.shift(self.ds_image, com)
        dims = self.ds_image.shape
        half_x = int((dims[0] + 1) / 2)
        half_y = int((dims[1] + 1) / 2)
        if self.params.twin_halves[0] == 0:
            self.ds_image[half_x:, :, :] = 0
        else:
            self.ds_image[: half_x, :, :] = 0
        if self.params.twin_halves[1] == 0:
            self.ds_image[:, half_y:, :] = 0
        else:
            self.ds_image[:, : half_y, :] = 0
#        print ('ds_image norm', get_norm(self.ds_image))

    def average_trigger(self):
#        print('------aver trig')
        if self.aver is None:
            self.aver = devlib.to_numpy(devlib.abs(self.ds_image))
            self.aver_iter = 1
        else:
            self.aver = self.aver + devlib.to_numpy(devlib.abs(self.ds_image))
            self.aver_iter += 1

    def progress_trigger(self):
        print('------iter', self.iter, '   error ', self.errs[-1])

    def get_ratio(self, divident, divisor):
        divisor_copy = divisor.copy()
        devlib.replace(divisor_copy, divisor_copy != 0.0, 1.0)
        ratio = divident / divisor_copy
        return ratio

    def get_norm(self, arr):
        return devlib.sum(devlib.square(devlib.abs(arr)))


def get_norm(arr):
    return devlib.sum(devlib.square(devlib.abs(arr)))


def fast_module_reconstruction(proc, device, params, data, image=None, support=None, coherence=None):
    """
    This function calls a bridge method corresponding to the requested processor type. The bridge method is an access
    to the CFM (Calc Fast Module). When reconstruction is completed the function retrieves results from the CFM.
    The data received is max centered and the array is ordered "C". The CFM requires data zero-frequency component at
    the center of the spectrum and "F" array order. Thus the data is modified at the beginning.
    Parameters
    ----------
    proc : str
        a string indicating the processor type/library, chices are: cpu, cuda, opencl
    device : int
        device id assigned to this reconstruction
    params : Paramas object
        object containing reconstruction parameters
    data : ndarray
        np array containing pre-processed, formatted experiment data
    image : ndarray
        initial image to continue reconstruction or None if random initial image
    support : ndarray
        support corresponding to image if continuation or None
    coherence : ndarray
       coherence corresponding to image if continuation and active pcdi feature or None

    Returns
    -------
    image : ndarray
        reconstructed image
    support : ndarray
        support for reconstructed image
    coherence : ndarray
        coherence for reconstructed image or None if pcdi inactive
    er : list
        a vector containing errors for each iteration
    flow : ndarray
        info to scientist/developer; a list of functions  that can run in one iterations (excluding inactive features)
    iter_array : ndarray
        info to scientist/developer; an array of 0s and 1s, 1 meaning the function in flow will be executed in iteration, 0 otherwise
    """
    d_type = np.float32
    data = np.fft.fftshift(data).astype(d_type)
#    print(data.shape)
    set_lib('af', len(data.shape))

    worker = Rec(params, data.shape, (image is None))
    worker.dev_init(proc, device, data, (image is None))
    image, support, err = worker.iterate()

    mx = np.abs(image).max()
    image = image / mx

    return image, support, None, err

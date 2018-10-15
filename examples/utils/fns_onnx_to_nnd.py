#!/usr/bin/env python3

# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation
#
# The source code contained or described herein and all documents related to the source code ("Material") are owned by
# Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel
# or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted,
# transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual property right is granted to
# or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement,
# estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel
# in writing.
#
#
# For details about script please contact following people:
#  * [Version: 1.0] Walkowiak, Marcin <marcin.walkowiak@intel.com>

# Use only limited part of this file. It is written for python 2.7.
from nnd_converter import NndFile

import functools
import glob
import onnx
import operator
import os.path

from array import array

onnx_data_infos = {
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    "convolution_W":                 ('conv1_weights',       [16, 3, 9, 9]),
    "convolution_B":                 ('conv1_bias',          [16]),
    "InstanceNormalization_scale":   ('inst_norm1_weights',  [16],               'S'),
    "InstanceNormalization_B":       ('inst_norm1_bias',     [16],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    "convolution1_W":                ('conv2_weights',       [32, 16, 3, 3]),
    "convolution1_B":                ('conv2_bias',          [32]),
    "InstanceNormalization_scale1":  ('inst_norm2_weights',  [32],               'S'),
    "InstanceNormalization_B1":      ('inst_norm2_bias',     [32],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    "convolution2_W":                ('conv3_weights',       [64, 32, 3, 3]),
    "convolution2_B":                ('conv3_bias',          [64]),
    "InstanceNormalization_scale2":  ('inst_norm3_weights',  [64],               'S'),
    "InstanceNormalization_B2":      ('inst_norm3_bias',     [64],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    "convolution3_W":                ('conv4_weights',       [64, 64, 3, 3]),
    "convolution3_B":                ('conv4_bias',          [64]),
    "InstanceNormalization_scale3":  ('inst_norm4_weights',  [64],               'S'),
    "InstanceNormalization_B3":      ('inst_norm4_bias',     [64],               'S'),
    "convolution4_W":                ('conv5_weights',       [64, 64, 3, 3]),
    "convolution4_B":                ('conv5_bias',          [64]),
    "InstanceNormalization_scale4":  ('inst_norm5_weights',  [64],               'S'),
    "InstanceNormalization_B4":      ('inst_norm5_bias',     [64],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    "convolution5_W":                ('conv6_weights',       [64, 64, 3, 3]),
    "convolution5_B":                ('conv6_bias',          [64]),
    "InstanceNormalization_scale5":  ('inst_norm6_weights',  [64],               'S'),
    "InstanceNormalization_B5":      ('inst_norm6_bias',     [64],               'S'),
    "convolution6_W":                ('conv7_weights',       [64, 64, 3, 3]),
    "convolution6_B":                ('conv7_bias',          [64]),
    "InstanceNormalization_scale6":  ('inst_norm7_weights',  [64],               'S'),
    "InstanceNormalization_B6":      ('inst_norm7_bias',     [64],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    "convolution7_W":                ('conv8_weights',       [64, 64, 3, 3]),
    "convolution7_B":                ('conv8_bias',          [64]),
    "InstanceNormalization_scale7":  ('inst_norm8_weights',  [64],               'S'),
    "InstanceNormalization_B7":      ('inst_norm8_bias',     [64],               'S'),
    "convolution8_W":                ('conv9_weights',       [64, 64, 3, 3]),
    "convolution8_B":                ('conv9_bias',          [64]),
    "InstanceNormalization_scale8":  ('inst_norm9_weights',  [64],               'S'),
    "InstanceNormalization_B8":      ('inst_norm9_bias',     [64],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    "convolution9_W":                ('conv10_weights',      [64, 64, 3, 3]),
    "convolution9_B":                ('conv10_bias',         [64]),
    "InstanceNormalization_scale9":  ('inst_norm10_weights', [64],               'S'),
    "InstanceNormalization_B9":      ('inst_norm10_bias',    [64],               'S'),
    "convolution10_W":               ('conv11_weights',      [64, 64, 3, 3]),
    "convolution10_B":               ('conv11_bias',         [64]),
    "InstanceNormalization_scale10": ('inst_norm11_weights', [64],               'S'),
    "InstanceNormalization_B10":     ('inst_norm11_bias',    [64],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    "convolution11_W":               ('conv12_weights',      [64, 64, 3, 3]),
    "convolution11_B":               ('conv12_bias',         [64]),
    "InstanceNormalization_scale11": ('inst_norm12_weights', [64],               'S'),
    "InstanceNormalization_B11":     ('inst_norm12_bias',    [64],               'S'),
    "convolution12_W":               ('conv13_weights',      [64, 64, 3, 3]),
    "convolution12_B":               ('conv13_bias',         [64]),
    "InstanceNormalization_scale12": ('inst_norm13_weights', [64],               'S'),
    "InstanceNormalization_B12":     ('inst_norm13_bias',    [64],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    "convolution13_W":               ('deconv1_weights',     [64, 32, 3, 3],     'D'),
    "convolution13_B":               ('deconv1_bias',        [32]),
    "InstanceNormalization_scale13": ('inst_norm14_weights', [32],               'S'),
    "InstanceNormalization_B13":     ('inst_norm14_bias',    [32],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    "convolution14_W":               ('deconv2_weights',     [32, 16, 3, 3],     'D'),
    "convolution14_B":               ('deconv2_bias',        [16]),
    "InstanceNormalization_scale14": ('inst_norm15_weights', [16],               'S'),
    "InstanceNormalization_B14":     ('inst_norm15_bias',    [16],               'S'),
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    "convolution15_W":               ('conv14_weights',      [3, 16, 9, 9]),
    "convolution15_B":               ('conv14_bias',         [3]),
    "Mul_B":                         ('scale2_weights',      [3, 1, 1]),
    "scale_B":                       ('scale2_bias',         [3, 1, 1]),
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
}

print('Started')

onnx_files = [(file, os.path.splitext(os.path.basename(file))[0], os.path.abspath(file))
              for file in glob.glob('*.onnx')]

for onnx_file, onnx_name, onnx_path in onnx_files:
    print(' -- Processing {0:<20} file...'.format('"{0}"'.format(onnx_file)))

    onnx_model = onnx.load(onnx_path)
    onnx_data = onnx_model.graph.initializer

    out_nnd_dir = onnx_name
    if len(onnx_data) > 0:
        os.makedirs(out_nnd_dir, exist_ok=True)


    # Save mean .nnd file.
    mean = array('f', [103.93900299072266, 116.77899932861328, 123.68000030517578])
    out_nnd_file = os.path.join(out_nnd_dir, 'fns_instance_norm_mean.nnd')
    print('        Saving: (dims: {0}) "{1}".'.format([3], out_nnd_file))
    out_nnd = NndFile.from_iterable(out_nnd_file, NndFile.DT_FP32, mean)
    out_nnd.to_file()

    for onnx_data_col in onnx_data:
        print('     -- Processing {0:<35} collection / tensor ( {1} )...'
              .format('"{0}"'.format(onnx_data_col.name), str(onnx_data_col.dims)))

        col_dims   = onnx_data_col.dims
        col_idx    = 0
        col_values = []

        if onnx_data_col.name not in onnx_data_infos:
            print("WARNING: Encountered collection does not have information entry! Omitting!")
            continue

        col_info   = onnx_data_infos[onnx_data_col.name]
        if col_dims != col_info[1]:
            print("WARNING: Expected dimensions are different than encountered! Omitting!")
            continue

        # Transform dimensions if necessary.
        permute_bf = False

        if 1 < len(col_dims) < 4 and len(col_info) >= 3:
            if 'B' in col_info[2]:
                col_dims = [functools.reduce(operator.mul, col_dims)]

        if len(col_dims) == 1 and len(col_info) >= 3:
            if 'S' in col_info[2]:
                col_dims = [1, col_dims[0], 1, 1]

        while 1 < len(col_dims) < 4:
            col_dims.insert(0, 1)

        if len(col_dims) == 4 and len(col_info) >= 3:
            if 'D' in col_info[2]:
                permute_bf = True

        # Save .nnd files.
        out_nnd_file = os.path.join(out_nnd_dir, col_info[0] + '.nnd')
        if len(col_dims) == 4:
            col_values = [[[array('f', onnx_data_col.float_data[y * col_dims[3]:(y + 1) * col_dims[3]])
                            for y in range(f * col_dims[2], (f + 1) * col_dims[2])]
                           for f in range(b * col_dims[1], (b + 1) * col_dims[1])]
                          for b in range(col_dims[0])]

            if permute_bf:
                col_dims[0], col_dims[1] = col_dims[1], col_dims[0]
                col_values = [[col_b_values[f]
                               for col_b_values in col_values]
                              for f in range(col_dims[0])]

            print('        Saving: (dims: {0}) "{1}".'.format(col_dims, out_nnd_file))
            out_nnd = NndFile.from_primitive_dump_raw_vals(out_nnd_file, col_values)
            out_nnd.to_file()
        elif len(col_dims) == 1:
            print('        Saving: (dims: {0}) "{1}".'.format(col_dims, out_nnd_file))
            out_nnd = NndFile.from_iterable(out_nnd_file, NndFile.DT_FP32, onnx_data_col.float_data)
            out_nnd.to_file()
        else:
            print("WARNING: Unexpected number of dimensions in data! Omitting!")

print('Finished')

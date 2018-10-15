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

from coremltools.models.utils import load_spec
from winmltools import convert_coreml
from winmltools.utils import load_model, save_model, save_text

import glob
import os.path


core_ml_files = [(file, os.path.splitext(os.path.basename(file))[0], os.path.abspath(file))
                 for file in glob.glob('*.mlmodel')]
onnx_files    = [(file, os.path.splitext(os.path.basename(file))[0], os.path.abspath(file))
                 for file in glob.glob('*.onnx')]

print('Started')
# print(core_ml_files)
# print(onnx_files)

for core_ml_file, core_ml_name, core_ml_path in core_ml_files:
    print(' -- Processing "{0}" file...'.format(core_ml_file))

    core_ml_model = load_spec(core_ml_path)
    onnx_model    = convert_coreml(core_ml_model, name=core_ml_name)
    save_model(onnx_model, os.path.splitext(core_ml_path)[0] + '.onnx')
    save_text(onnx_model, os.path.splitext(core_ml_path)[0] + '.dump.txt')

for onnx_file, onnx_name, onnx_path in onnx_files:
    print(' -- Processing "{0}" file...'.format(onnx_file))

    onnx_model = load_model(onnx_path)
    save_text(onnx_model, os.path.splitext(onnx_path)[0] + '.dump.txt')


print('Finished')
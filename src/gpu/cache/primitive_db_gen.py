#!/usr/bin/env python3


# Copyright (c) 2016-2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# To add new kernel please add a .cl file to kernels directory
# the database name will be the part of the file name up to first '.' character
# the trailing characters are a tag to allow multiple primitive implementations

import os.path
import sys


# Workaround for C2026 in Visual Studio (until limit of around 64K).
def split_into_sized_literals(s, size):
    return '\n'.join(['R"__krnl({})__krnl"'.format(s[i:i + size]) for i in range(0, len(s), size)])

kernels_dir_name = sys.argv[2]
out_file_name = os.path.join(sys.argv[1], 'primitive_db.inc')

with open(out_file_name, 'w') as out_file:
    out_file.write('// This file is autogenerated by primitive_db_gen.py, all changes to this file will be undone\n\n')
    for file_name in os.listdir(kernels_dir_name):
        if file_name.endswith('.cl'):
            print('processing {}'.format(file_name))
            with open(os.path.join(kernels_dir_name, 
                                   file_name), 'r') as kernel_file:
                out_file.write('{{"{}",\n{}}},\n\n'.format(
                    file_name[:file_name.find('.')],
                    split_into_sized_literals(kernel_file.read(), 16000)))
                
    

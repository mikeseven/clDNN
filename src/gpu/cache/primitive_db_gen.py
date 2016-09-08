#!/usr/bin/env python3

# To add new kernel please add a .cl file to kernels directory
# the database name will be the part of the file name up to first '.' character
# the trailing characters are a tag to allow multiple primitive implementations

import os.path
import sys

kernels_dir_name = sys.argv[2]
out_file_name = os.path.join(sys.argv[1], 'primitive_db.inc')

with open(out_file_name, 'w') as out_file:
    out_file.write('// This file is autogenerated by primitive_db_gen.py, all changes to this file will be undone\n\n')
    for file_name in os.listdir(kernels_dir_name):
        if file_name.endswith('.cl'):
            print('processing {}'.format(file_name))
            with open(os.path.join(kernels_dir_name, 
                                   file_name), 'r') as kernel_file:
                out_file.write('{{"{}",\nR"{}"}},\n\n'.format(file_name[:file_name.find('.')],
                                                        kernel_file.read()))
                
    

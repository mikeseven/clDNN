#!/usr/bin/python
import os
import sys
import common


########################################################################################################################
def main():
    branch = common.get_base_hash()
    curr_dir = os.getcwd()
    os.chdir('../../')
    common.create_patch_file(curr_dir, branch, True)
    os.chdir(curr_dir)
    
main()
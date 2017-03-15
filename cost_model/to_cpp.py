#!/usr/bin/env python2.7
import os
import sys

def get_pitches(dim_str):
    int_v = [int(v.strip()) for v in dim_str.split(",")]
    x = int_v[0]
    y = x*int_v[1]
    z = y*int_v[2]
    w = z*int_v[3]
    return x, y ,z, w

def main():
    with open('convolution_kernels.txt') as f_in:
        print ''
        print 'std::vector<KernelSelctor::ConvolutionParams> params_vec;'

        print 'void InitConvParams()'
        print '{'
        param_num = 0
        for line in f_in:
            params_str = 'params%d' % param_num
            param_num += 1

            line_list = line.strip().split('_')

            print ''
            print 'KernelSelctor::ConvolutionParams %s;' % params_str
            print '%s.inputType = Datatype::%s;' % (params_str, line_list[0])
            print '%s.inputLayout = KernelSelctor::DataLayout::%s;' % (params_str, line_list[1])
            print '%s.outputLayout = KernelSelctor::DataLayout::%s;' % (params_str, line_list[2])
            print '%s.activationFunc = ActivationFunction::%s;' % (params_str, line_list[3])
            print '%s.nlParams = {%s, %s};' % (params_str, line_list[4], line_list[5])
            print '%s.inDims = {%s, %s, %s, %s};' % (params_str, line_list[6], line_list[7], line_list[8], line_list[9])
            print '%s.inDesc = {%s, {%s, %s, %s, %s}, false};' % (params_str, line_list[10], line_list[11], line_list[12], line_list[13], line_list[14])
            print '%s.outDims = {%s, %s, %s, %s};' % (params_str, line_list[15], line_list[16], line_list[17], line_list[18])
            print '%s.outDesc = {%s, {%s, %s, %s, %s}, false};' % (params_str, line_list[19], line_list[20], line_list[21], line_list[22], line_list[23])
            print '%s.convParams.filterSize = {%s, %s};' % (params_str, line_list[24], line_list[25])
            print '%s.convParams.padding = {%s, %s};' % (params_str, line_list[26], line_list[27])
            print '%s.convParams.stride = {%s, %s};' % (params_str, line_list[28], line_list[29])

            '''
            if line == '------------':
                params_str = 'params%d' % param_num
                print ''
                print 'KernelSelctor::ConvolutionParams %s;' % params_str
                print '%s.inputLayout = KernelSelctor::DataLayout::bfyx;' % params_str
                print '%s.outputLayout = KernelSelctor::DataLayout::bfyx;' % params_str
                print '%s.activationFunc = ActivationFunction::NONE;' % params_str
                param_num += 1
            if line.startswith('type:'):
                val = line.split(':')[1]
                print '%s.inputType = Datatype::%s;' % (params_str, val)
            if line.startswith('inDims:'):
                val = line.split(':')[1]
                print '%s.inDims = {%s};' % (params_str, val)
                print params_str + '.inDesc = {0, {%d, %d, %d, %d}};' % get_pitches(val)
            if line.startswith('outDims:'):
                val = line.split(':')[1]
                print '%s.outDims = {%s};' % (params_str, val)
                print params_str + '.outDesc = {0, {%d, %d, %d, %d}};' % get_pitches(val)
            if line.startswith('filterSize:'):
                val = line.split(':')[1]
                print '%s.convParams.filterSize = {%s};' % (params_str, val)
            if line.startswith('padding:'):
                val = line.split(':')[1]
                print '%s.convParams.padding = {%s};' % (params_str, val)
            if line.startswith('stride:'):
                val = line.split(':')[1]
                print '%s.convParams.stride = {%s};' % (params_str, val)
            '''

        print ''
        print 'params_vec = {'
        for i in range(0, param_num):
            params_str = 'params%d,' % i
            print '    %s' % params_str
        print '};'
        print '}'


if __name__ == '__main__':
    main()
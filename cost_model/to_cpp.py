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
            types = [
                'input_type',
                'input_layout',
                'output_layout',
                'activation_func',
                'nl_m',
                'nl_n',
                'input_x',
                'input_y',
                'input_feature',
                'input_batch',
                'output_x',
                'output_y',
                'output_feature',
                'output_batch',
                'weights_layout',
                'bias_layout',
                'filter_x',
                'filter_y',
                'pad_x',
                'pad_y',
                'stride_x',
                'stride_y',
                'dilation_x',
                'dilation_y',
            ]

            vals = dict()
            for i,v in enumerate(types):
                vals[v] = line_list[i]

            print ''
            print 'ConvolutionParams %s;' % params_str
            print '%s.inputs.resize(1);' % params_str
            print '%s.layerID = "%s";' % (params_str, params_str)

            print '%s.inputs[0] = { Datatype::%s, DataLayout::%s, PADDED_VAL::ZERO, 0, { %s, %s, %s, %s } };' % \
                  (params_str, vals['input_type'], vals['input_layout'].lower(), vals['input_x'], vals['input_y'], vals['input_feature'], vals['input_batch'])
            print '%s.output    = { Datatype::%s, DataLayout::%s, PADDED_VAL::ZERO, 0, { %s, %s, %s, %s } };' % \
                  (params_str, vals['input_type'], vals['output_layout'].lower(), vals['output_x'], vals['output_y'], vals['output_feature'], vals['output_batch'])
            print '%s.weights   = { WeightsType::%s, WeightsLayout::%s, PADDED_VAL::ZERO, 0, { %s, %s, %s, %s } };' % \
                  (params_str, vals['input_type'], vals['weights_layout'].lower(), vals['filter_x'], vals['filter_y'], vals['input_feature'], vals['output_feature'])

            if vals['bias_layout'] != 'nobias':
                print '%s.bias.resize(1);' % params_str
                print '%s.bias[0]   = { Datatype::%s, DataLayout::%s, PADDED_VAL::ZERO, 0, { %s, 1 } };' % \
                      (params_str, vals['input_type'], vals['bias_layout'].lower(), vals['output_feature'])


            print '%s.activationFunc = ActivationFunction::%s;' % (params_str, vals['activation_func'])
            print '%s.nlParams = {%s, %s};' % (params_str, vals['nl_m'], vals['nl_n'])
            print '%s.convParams.filterSize = {%s, %s};' % (params_str, vals['filter_x'], vals['filter_y'])
            print '%s.convParams.padding = {%s, %s};' % (params_str, vals['pad_x'], vals['pad_y'])
            print '%s.convParams.stride = {%s, %s};' % (params_str, vals['stride_x'], vals['stride_y'])
            print '%s.convParams.dilation = {%s, %s};' % (params_str, vals['dilation_x'], vals['dilation_y'])
            print '%s.convParams.split = 1;' % params_str

        print ''
        print 'params_vec = {'
        for i in range(0, param_num):
            params_str = 'params%d,' % i
            print '    %s' % params_str
        print '};'
        print '}'


if __name__ == '__main__':
    main()
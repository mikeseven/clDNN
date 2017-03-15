#!/usr/bin/env python2.7
import os
import sys
import re

def main():
    # gemm = '0_cost_gt2_gemm_linux.txt'
    # direct = '0_cost_gt2_direct_linux.txt'
    # igk = '0_cost_gt2_igk_linux.txt'

    # gemm_flag = '1_gemm_flag.txt'
    # gemm_no_flag = '1_gemm_no_flag.txt'
    # direct_flag = '1_direct_flag.txt'
    # direct_flag_unroll_1 = '1_direct_flag_unroll_1.txt'
    # direct_flag_unroll_2 = '1_direct_flag_unroll_2.txt'
    # direct_flag_unroll_3 = '1_direct_flag_unroll_3.txt'
    # direct_no_flag = '1_direct_no_flag.txt'
    # direct_no_flag_unroll_1 = '1_direct_no_flag_unroll_1.txt'
    # direct_no_flag_unroll_2 = '1_direct_no_flag_unroll_2.txt'
    # direct_no_flag_unroll_3 = '1_direct_no_flag_unroll_3.txt'
    # igk_flag = '1_igk_flag.txt'
    # igk_no_flag = '1_igk_no_flag.txt'

    gemm_flag_linux = '1_gemm_flag_linux.txt'
    gemm_no_flag_linux = '1_gemm_no_flag_linux.txt'
    direct_flag_linux = '1_direct_flag_linux.txt'
    direct_no_flag_linux = '1_direct_no_flag_linux.txt'
    igk_flag_linux = '1_igk_flag_linux.txt'
    igk_no_flag_linux = '1_igk_no_flag_linux.txt'

    file1 = igk_flag_linux
    file2 = direct_flag_linux
    with open(file1) as res_1:
        with open(file2) as res_2:
            lines_1 = res_1.read().split('\n')
            lines_2 = res_2.read().split('\n')

            total_res_1 = []
            total_res_2 = []

            regex = re.compile('\{0x.*, (.*)\}, .*')
            for i in range(0,len(lines_1)):
                if (lines_1[i].strip().startswith('{0x')):
                    direct_val = regex.match(lines_1[i]).group(1).strip()
                    igk_val = regex.match(lines_2[i]).group(1).strip()
                    if direct_val == 'NOT_SUPPORTED' or igk_val == 'NOT_SUPPORTED':
                        continue
                    time_1 = float(direct_val[:-1])
                    time_2 = float(igk_val[:-1])
                    delta = time_2 - time_1

                    epsilon = 0.02
                    #print delta, igk_val, direct_val
                    if delta > epsilon:
                        total_res_1.append('({}): {}'.format(abs(delta), lines_1[i]))
                    elif delta < -epsilon:
                        total_res_2.append('({}): {}'.format(abs(delta), lines_2[i]))

            print 'Time1: ({})'.format(file1)
            for l in total_res_1:
                print l
            print 'Time2: ({})'.format(file2)
            for l in total_res_2:
                print l

if __name__ == '__main__':
    main()
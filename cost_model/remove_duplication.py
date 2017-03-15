#!/usr/bin/env python2.7
import os
import sys

def main():
    with open('convolution_kernels.txt') as f:
        lines = f.read().split('\n')
        final_lines = []
        s = set()
        for line in lines:
            line = line.strip()
            if line not in s:
                s.add(line)
                final_lines.append(line)

        for line in final_lines:
            print line



if __name__ == '__main__':
    main()
/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/

#include "api/neural.h"

int main( int argc, char* argv[ ] )
{
    extern void example_convolution_backward();
    example_convolution_backward();
    int result = 0;
    return result;
}
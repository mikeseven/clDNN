/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/

#include "api/neural.h"
#include <iostream>

#include "convolution.h" //todo remove

int main( int argc, char* argv[ ] )
{
    neural::attach_conv();
    extern void example_convolution_forward();

    try{
        example_convolution_forward();
    } catch (std::exception &e) {
        std::cerr << e.what();
    }

    return 0;
}
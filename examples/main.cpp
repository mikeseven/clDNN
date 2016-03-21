/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/

#include "api/neural.h"
#include <iostream>

int main( int argc, char* argv[ ] )
{
    extern void example_convolution_forward();

    try{
        example_convolution_forward();
    } catch (std::exception &e) {
        std::cerr << e.what();
    }

    return 0;
}
/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/


#include "gtest/gtest.h"

int main( int argc, char* argv[ ] )
{
    int result;
  ::testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();
    return result;
}
/*
Copyright (c) 2016, Intel Corporation
NeuralIA
*/


#include "gtest/gtest.h"

int main( int argc, char* argv[ ] )
{
    int result;
    {
        ::testing::InitGoogleTest(&argc, argv);

#if _WIN32
        // Safety cleanup.
        system("where /q umdh && del pre_cpu.txt");
        system("where /q umdh && del post_cpu.txt");
        system("where /q umdh && del memdiff_cpu.txt");

        // Get first snapshot.
        system("where /q umdh && umdh -pn:tests.exe -f:pre_cpu.txt");
#endif

        result = RUN_ALL_TESTS();
    }

#if _WIN32
    // Get second snapshot.
    system("where /q umdh && umdh -pn:tests.exe -f:post_cpu.txt");

    // Prepare memory diff.
    system("where /q umdh && umdh pre_cpu.txt post_cpu.txt -f:memdiff_cpu.txt");

    // Cleanup.
    system("where /q umdh && del pre_cpu.txt");
    system("where /q umdh && del post_cpu.txt");
#endif

    return result;
}
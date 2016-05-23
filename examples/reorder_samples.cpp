#include "api/neural.h"
#include <vector>
#include <numeric>
#include <cassert>
#include <iostream>

// memory->memory reshape
void example_009() {
    using namespace neural;
    using namespace std;

    const uint32_t dim_y      = 2, dim_x      = 2, dim_f      = 4, dim_b      = 3;

  	const uint32_t all_size = dim_y*dim_x*dim_f*dim_b;
	auto in_layout = memory::format::yxfb_f32;
    auto out_layout = memory::format::byxf_f32;

    float in_buffer[all_size] =
    {// yxfb
         0,  1,  2,//b row f0 x0 y0
         3,  4,  5,//b row f1 x0 y0
         6,  7,  8,//b row f2 x0 y0
         9, 10, 11,//b row f3 x0 y0

        12, 13, 14,//b row f0 x1 y0
        15, 16, 17,//b row f1 x1 y0
        18, 19, 20,//b row f2 x1 y0
        21, 22, 23,//b row f3 x1 y0

        24, 25, 26,//b row f0 x0 y1
        27, 28, 29,//b row f1 x0 y1
        30, 31, 32,//b row f2 x0 y1
        33, 34, 35,//b row f3 x0 y1

        36, 37, 38,//b row f0 x1 y1
        39, 40, 41,//b row f1 x1 y1
        42, 43, 44,//b row f2 x1 y1
        45, 46, 47 //b row f3 x1 y1
    };

	// input buffer should be initialized with valid data
                                     //b  y  x  f
    neural::vector<uint32_t> in_sizes = { 3, {2, 2}, 4};
                                     //b  f  x  y
    neural::vector<uint32_t> out_sizes= { 2, {4, 2}, 3};

    auto input  = memory::create({engine::reference, in_layout, in_sizes});
    auto output = memory::create({engine::reference, out_layout, out_sizes, true});

    auto reorder    = reorder::create(reorder::arguments{engine::reference,input,output});

    try
    {
        execute({input(in_buffer), reorder}).sync();
    }
    catch (const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
}

//#include "api/neural.h"
//#include <vector>
//#include <numeric>
//#include <cassert>
//#include <iostream>
//
//// memory_obsolete->memory_obsolete reshape
//void example_009() {
//    using namespace neural;
//    using namespace std;
//
//    const uint32_t dim_y      = 2, dim_x      = 2, dim_f      = 4, dim_b      = 3;
//
//  	const uint32_t all_size = dim_y*dim_x*dim_f*dim_b;
//	auto in_layout = memory_obsolete::format::yxfb_f32;
//    auto out_layout = memory_obsolete::format::bfxy_f32;
//
//    float in_buffer[all_size] =
//    {// yxfb
//         0,  1,  2,//b row f0 x0 y0
//         3,  4,  5,//b row f1 x0 y0
//         6,  7,  8,//b row f2 x0 y0
//         9, 10, 11,//b row f3 x0 y0
//
//        12, 13, 14,//b row f0 x1 y0
//        15, 16, 17,//b row f1 x1 y0
//        18, 19, 20,//b row f2 x1 y0
//        21, 22, 23,//b row f3 x1 y0
//
//        24, 25, 26,//b row f0 x0 y1
//        27, 28, 29,//b row f1 x0 y1
//        30, 31, 32,//b row f2 x0 y1
//        33, 34, 35,//b row f3 x0 y1
//
//        36, 37, 38,//b row f0 x1 y1
//        39, 40, 41,//b row f1 x1 y1
//        42, 43, 44,//b row f2 x1 y1
//        45, 46, 47 //b row f3 x1 y1
//    };
//
//    float wyn_buffer[all_size] =
//    {//bfxy x=2, y=2, f=4, b=3
//        0, // b0 f0 x0 y0
//       24, // b0 f0 x0 y1
//       12, // b0 f0 x1 y0
//       36, // b0 f0 x1 y1
//        3, // b0 f1 x0 y0
//       27, // b0 f1 x0 y1
//       15, // b0 f1 x1 y0
//       39, // b0 f1 x1 y1
//
//        6, // b0 f2 x0 y0
//       30, // b0 f2 x0 y1
//       18, // b0 f2 x1 y0
//       42, // b0 f2 x1 y1
//        9, // b0 f3 x0 y0
//       33, // b0 f3 x0 y1
//       21, // b0 f3 x1 y0
//       45, // b0 f3 x1 y1
//
//        1, // b1 f0 x0 y0
//       25, // b1 f0 x0 y1
//       13, // b1 f0 x1 y0
//       37, // b1 f0 x1 y1
//        4, // b1 f1 x0 y0
//       28, // b1 f1 x0 y1
//       16, // b1 f1 x1 y0
//       40, // b1 f1 x1 y1
//
//        7, // b1 f2 x0 y0
//       31, // b1 f2 x0 y1
//       19, // b1 f2 x1 y0
//       43, // b1 f2 x1 y1
//       10, // b1 f3 x0 y0
//       34, // b1 f3 x0 y1
//       22, // b1 f3 x1 y0
//       46, // b1 f3 x1 y1
//
//        //3, // b2 f0 x0 y0 // should fail
//        2, // b2 f0 x0 y0
//       26, // b2 f0 x0 y1
//       14, // b2 f0 x1 y0
//       38, // b2 f0 x1 y1
//        5, // b2 f1 x0 y0
//       29, // b2 f1 x0 y1
//       17, // b2 f1 x1 y0
//       41, // b2 f1 x1 y1
//
//        8, // b2 f2 x0 y0
//       32, // b2 f2 x0 y1
//       20, // b2 f2 x1 y0
//       44, // b2 f2 x1 y1
//       11, // b2 f3 x0 y0
//       35, // b2 f3 x0 y1
//       23, // b2 f3 x1 y0
//       47, // b2 f3 x1 y1
//    };
//
//	// input buffer should be initialized with valid data
//                                     //y  x  f  b
//    std::vector<uint32_t> in_sizes = { 2, 2, 4, 3};
//                                     //b  f  x  y
//    std::vector<uint32_t> out_sizes= { 3, 4, 2, 2};
//
//    auto input  = memory_obsolete::create({engine::cpu, in_layout, in_sizes});
//    auto output = memory_obsolete::create({engine::cpu, out_layout, out_sizes, true});
//
//    auto reorder    = reorder::create(reorder::arguments{engine::reference,input,output});
//
//    try
//    {
//        execute({input(in_buffer), reorder});
//    }
//    catch (const std::exception& ex)
//    {
//        std::cout << ex.what() << std::endl;
//    }
//
//    auto buf_out = static_cast<float*>(output.as<const memory_obsolete&>().pointer);
//
//    for(size_t i = 0; i < dim_y*dim_x*dim_f*dim_b; ++i)
//        assert (buf_out[i] == wyn_buffer[i]);
//
//    //2 pass output as input, should give input ;)
//    auto input2  = memory_obsolete::create({engine::cpu, out_layout, out_sizes});
//    auto output2 = memory_obsolete::create({engine::cpu, in_layout, in_sizes, true});
//
//    try
//    {
//        auto reorder2    = reorder::create({engine::reference,input2,output2});
//        execute({input2(wyn_buffer), reorder2});
//    }
//    catch (const std::exception& ex)
//    {
//        std::cout << ex.what() << std::endl;
//    }
//
//    auto buf_out2 = static_cast<float*>(output2.as<const memory_obsolete&>().pointer);
//
//    for(size_t i = 0; i < dim_y*dim_x*dim_f*dim_b; ++i)
//        assert (buf_out2[i] == in_buffer[i]);
//
//    //throws exception auto in_layout = memory_obsolete::format::yxfb_f32;
//    input2  = memory_obsolete::create({engine::cpu, memory_obsolete::format::yxfb_f64, out_sizes});
//    output2 = memory_obsolete::create({engine::cpu, in_layout, in_sizes, true});
//
//    try
//    {
//        auto reorder2    = reorder::create({engine::reference,input2,output2});
//        execute({input2(wyn_buffer), reorder2});
//    }
//    catch (const std::exception& ex)
//    {
//        std::cout << ex.what() << std::endl;
//    }
//    system("pause");
//
//}

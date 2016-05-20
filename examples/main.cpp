/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "api/neural.h"
#include <iostream>
#include "tests/test_utils/test_utils.h"


#include <cstdint>
#include <iostream>
#include <iomanip>
#include <intrin.h>
#include <array>
#include <random>
#include <algorithm>
#include <set>
#include <map>
#include <Windows.h>
#include <cstddef>
const int64_t   input_width             = 64;   // data = input|output
const int64_t   input_height            = 64;   // data = input|output
const int64_t   input_feature_maps      = 4;
const int64_t   input_view_start_x      = 0;
const int64_t   input_view_start_y      = 0;
const int64_t   output_feature_maps     = 32;
const int64_t   output_view_start_x     = 0;
const int64_t   output_view_start_y     = 0;
const int64_t   output_view_width       = 50;
const int64_t   output_view_height      = 50;
const int64_t   filter_size             = 11;    // filter size is the same for both axes
const int64_t   block_width             = 3;
const int64_t   block_height            = 3;
const int64_t   stride_width            = 4;
const int64_t   stride_height           = 4;
// derived configuration
const int64_t   output_width            = (input_width +stride_width -1)/stride_width;
const int64_t   output_height           = (input_height+stride_height-1)/stride_height;
const auto      core_count              = 1;
//jit_convolution *make_jit_convolution(
//    float *output, uint64_t output_width, uint64_t output_height, uint64_t output_feature_maps
//, float * input, uint64_t  input_width, uint64_t  input_height, uint64_t  input_feature_maps, uint64_t stride_width, uint64_t stride_height
//, uint64_t input_view_start_x, uint64_t input_view_start_y
//, float *filter, uint64_t filter_size
//, uint64_t block_width, uint64_t block_height
//) {
//    if(input_feature_maps!=3) {
//        assert( input_feature_maps%jit_convolution_generic:: input_features_per_iteration==0 && "input feature count is not multiple of input_features_per_iteration");
//        assert(output_feature_maps%jit_convolution_generic::output_features_per_iteration==0 && "output feature count is not multiple of output_features_per_iteration");
//        return new jit_convolution_generic(
//              output, output_width, output_height, output_feature_maps
//            ,  input,  input_width,  input_height,  input_feature_maps, stride_width, stride_height
//            , filter,  filter_size
//            , block_width, block_height
//        );
//    }
//}

// validation code
void validate(
    float *output, uint64_t output_width, uint64_t output_height, uint64_t output_feature_maps, uint64_t output_features_per_iteration
    , uint64_t output_view_x, uint64_t output_view_y, uint64_t output_view_width, uint64_t output_view_height
    , float *input, uint64_t input_width, uint64_t input_height, uint64_t input_feature_maps, uint64_t stride_width, uint64_t stride_height
    , uint64_t input_view_start_x, uint64_t input_view_start_y
    , float *filter, uint64_t filter_size
    , uint64_t batch_size
) {
    std::cout << "validation:";

    auto is_close_enough = [](float valid, float test) -> bool {
        const auto error_relative = 0.015f;
        const auto error_direct = 1e-2f;
        auto d = abs(valid-test);
        if(abs(valid)<error_direct) return d<error_direct;
        else return d/abs(valid)<error_relative;
    };

    const int64_t filter_radius = (filter_size-1)/2;
    const auto output_feature_blocks    = output_feature_maps/output_features_per_iteration;
    for(int64_t b=0; b<batch_size; ++b) {
        for(int64_t y=0; y<output_view_height; ++y) {
            for(int64_t x=0; x<output_view_width; ++x)
                for(int64_t fo=0; fo<output_feature_maps; ++fo) {
                    int64_t foi = fo%output_features_per_iteration;
                    int64_t fob = fo/output_features_per_iteration;
                    float valid = 0.0f;
                    for(int64_t yk=0; yk<filter_size; ++yk) {
                        int64_t ys = input_view_start_y + y*stride_height+yk-filter_radius;
                        if(ys<0 || ys>=input_height) continue;
                        for(int64_t xk=0; xk<filter_size; ++xk) {
                            int64_t xs = input_view_start_x + x*stride_height+xk-filter_radius;
                            if(xs<0 || xs>=input_width) continue;
                            for(int64_t fi=0; fi<input_feature_maps; ++fi) {
                                float value  = input[b + batch_size*(fi + input_feature_maps*(xs + input_width*ys))];
                                float weight = filter[foi + output_features_per_iteration*(fi + input_feature_maps*(fob + output_feature_blocks*(xk + filter_size*yk)))];
                                valid  = static_cast<float>(double(value)*weight + valid);
                            }
#if 0
                            std::cout << "[yo=" << yo << ",xo=" << xo << ",fo=" << fo << ",yk=" << yk << ",xk=" << xk << "]: " << valid;
                            std::cout << std::endl;
#endif
                        }
                    }
                    if(valid<0) valid = 0;
                    auto yo = y+output_view_y;
                    auto xo = x+output_view_x;
                    auto tested = output[b + batch_size*(fo + output_feature_maps*(xo + output_width*yo))];
                    if(!is_close_enough(valid, tested)) {
                        std::cout  << std::endl
                            << "error at [b,x,y,f] = [" << b << "," << xo << "," << yo << "," << fo << "]\n"
                            << "\tvalid  = " << valid << "\n"
                            << "\ttested = " << tested << std::endl;
                        __debugbreak();
                    }
                }
        }
    }
    std::cout << " ok"  << std::endl;
};
template<typename T> T *align(T *pointer, size_t align) {
    return reinterpret_cast<T *>((reinterpret_cast<uintptr_t>(pointer)+align-1)/align*align);
};

int main()
{
    extern void example_convolution_cpu_forward();

    std::cout << "initializing buffers" << std::endl;

    // allocate memory buffers
    const uint64_t batch_size = 24;
    const auto align_size = 64;
    const auto align_size_in_float = align_size/sizeof(float);
    const auto output_buffer_size = output_height*output_width*output_feature_maps*batch_size;
    const auto  input_buffer_size =  input_height* input_width* input_feature_maps*batch_size;
    const auto filter_buffer_size = filter_size*filter_size*output_feature_maps*input_feature_maps;

    std::unique_ptr<float> output_container = std::move(std::unique_ptr<float>(new float[output_buffer_size+align_size_in_float]));
    std::unique_ptr<float>  input_container = std::move(std::unique_ptr<float>(new float[ input_buffer_size+align_size_in_float]));
    std::unique_ptr<float> filter_container = std::move(std::unique_ptr<float>(new float[filter_buffer_size+align_size_in_float]));

    const auto output = align(output_container.get(), align_size);
    const auto  input = align( input_container.get(), align_size);
    const auto filter = align(filter_container.get(), align_size);

    // generate convolution code using jit and create job set
    //auto job_set = make_jit_convolution(
    //      output, output_width, output_height, output_feature_maps
    //    ,  input,  input_width,  input_height,  input_feature_maps, stride_width, stride_height
    //    ,  input_view_start_x, input_view_start_y
    //    , filter, filter_size
    //    , block_width, block_height
    //);

    // initialized inputs & filter with pseudorandom values
    std::mt19937 engine(0xdeadf00d);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    auto lambda = [&]{return distribution(engine);};
    std::fill    (output, output+output_buffer_size, 0.0f);
    std::generate( input,  input+ input_buffer_size, lambda);
    std::generate(filter, filter+filter_buffer_size, lambda);

    validate(
            output, output_width, output_height, output_feature_maps, 24
        , 0, 0, output_width, output_height
        ,  input, input_width, input_height, input_feature_maps, stride_width, stride_height
        , 0, 0
        , filter, filter_size
        , batch_size
    );

    try {
        example_convolution_cpu_forward();
    }
    catch (std::exception &e) {
        std::cerr << e.what();
    }
    catch (...) {
        std::cerr << "Unknown exceptions.";
    }
    return 0;
}

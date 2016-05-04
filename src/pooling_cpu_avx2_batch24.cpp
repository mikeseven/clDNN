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
#include <cstring>
#include <immintrin.h>
#include <mutex>

#include "pooling_cpu_avx2_batch24.h"
#include "pooling.h"

const uint64_t BATCH_ACCEPTED_BLOCK = 24;                         //the batch size that is minimal required for usage with jit version
const uint64_t BATCH_SHIFT = 8;                                   //the size of register used for shifting with batch layout / number if pics/floats that are processed at the same time
const uint64_t BATCH_BLOCKS = BATCH_ACCEPTED_BLOCK / BATCH_SHIFT; //number of registers (blocks to process) in the batch format
const uint64_t BUFFERS_ALIGNMENT = BATCH_SHIFT * sizeof(float);   //required alignment of all buffers used by jit primitives

namespace {
 //todo begin remove
struct InputHeight {};
struct InputWidth {};
struct InputFeats {};
struct Rows {};
struct Cols {};
struct PoolingWidth {};
struct PoolingHeight {};

template <typename T_What, typename T>
struct Value
{
    T value;
    Value(T_What, T value) : value(value) {}

    template <typename T_Other>
    Value(const Value<T_What, T_Other>& other) : value(other.value) {}


    operator T() const { return value; }

    Value<T_What, T> operator+(const Value<T_What, T>& other) { return Value<T_What, T>(T_What(), value + other.value); }
    Value<T_What, T> operator-(const Value<T_What, T>& other) { return Value<T_What, T>(T_What(), value - other.value); }
    Value<T_What, T> operator*(const Value<T_What, T>& other) { return Value<T_What, T>(T_What(), value * other.value); }
    Value<T_What, T> operator/(const Value<T_What, T>& other) { return Value<T_What, T>(T_What(), value / other.value); }
};

template <typename T_Height, typename T_Width>
struct Dimensions2D
{
    Value<T_Height, uint64_t> height;
    Value<T_Width, uint64_t> width;
    Dimensions2D(Value<T_Height, uint64_t> height, Value<T_Width, uint64_t> width)
        : height(height)
        , width(width)
    {}

    uint64_t size() const { return height * width; }
};

template <typename T_Height, typename T_Width, typename T_Feats>
struct Dimensions3D : Dimensions2D<T_Height, T_Width>
{
    Value<T_Feats, uint64_t> feats;
    Dimensions3D(Value<T_Height, uint64_t> height,
                 Value<T_Width, uint64_t> width,
                 Value<T_Feats, uint64_t> feats)
        : Dimensions2D<T_Height, T_Width>(height, width)
        , feats(feats)
    {}
    uint64_t size() const { return Dimensions2D<T_Height, T_Width>::size() * feats; }
};
typedef Dimensions3D<InputHeight, InputWidth, InputFeats> InputDimensions;

template <typename T_What, typename T> Value<T_What, T> make(T val) { return Value<T_What, T>(T_What(), val); }
typedef Dimensions2D<PoolingHeight, PoolingWidth> PoolingDimensions;
struct Stride { Value<Rows, uint64_t> rows; Value<Cols, uint64_t> cols; };
struct PoolingInfo {
    PoolingDimensions dims;
    Stride stride;
};
 //todo end remove

void naive(float* input, float* output, InputDimensions input_dims, PoolingInfo info) {
    uint64_t batch_accs = BATCH_ACCEPTED_BLOCK / BATCH_SHIFT;
    uint64_t in_pixel_size = input_dims.feats * BATCH_ACCEPTED_BLOCK;

    auto feats = input_dims.feats;
    std::memcpy(output, input, input_dims.feats * BATCH_ACCEPTED_BLOCK * sizeof(float));
    for (uint64_t i = 0; i < info.dims.height; ++i)
    {
        for (uint64_t j = 0; j < info.dims.width; ++j)
        {
            if ((i == 0) && (j == 0)) continue;
            auto curr_output = output;
            auto curr_input = input + (i * input_dims.width + j) * in_pixel_size;
            for (uint64_t feat = 0u; feat < input_dims.feats; ++feat)
            {
                for (uint64_t acc_index = 0; acc_index < batch_accs; ++acc_index)
                {
                    auto result = _mm256_max_ps(_mm256_load_ps(curr_output), _mm256_load_ps(curr_input));
                    _mm256_store_ps(curr_output, result);

                    curr_output += BATCH_SHIFT;
                    curr_input += BATCH_SHIFT;
                }
            }
        }
    }
    }

//struct packed_arguments{
//    float* input;
//    float* output;
//    uint64_t input_height;
//    uint64_t input_width;
//    uint64_t pool_out_row;
//    uint64_t pool_out_col;
//    uint64_t pool_b;
//    uint64_t output_height;
//    uint64_t output_width;
//    uint64_t batch_blocks;
//    uint64_t input_pixel_size;
//    uint64_t output_pixel_size;
//    PoolingInfo& info;
//    InputDimensions& in_dims;
//    std::mutex& mtx;
//};
//
//bool pull_job( float* input,
//               float* output,
//               uint64_t input_height,
//               uint64_t input_width,
//               uint64_t pool_out_row,
//               uint64_t pool_out_col,
//               uint64_t pool_b,
//               uint64_t output_height,
//               uint64_t output_width,
//               uint64_t batch_blocks,
//               uint64_t input_pixel_size,
//               uint64_t output_pixel_size,
//               PoolingInfo& info,
//               InputDimensions& in_dims,
//               std::mutex& mtx){
//    uint64_t b = 0u;
//    uint64_t out_row = 0u;
//    uint64_t out_col = 0u;
//    {
//        std::unique_lock<std::mutex> guard(mtx);
//        out_row = pool_out_row;
//        out_col = pool_out_col;
//        b = pool_b;
//        if (out_row >= output_height) return false;
//
//        ++pool_b;
//        if (pool_b >= batch_blocks) {
//            pool_out_col++;
//            pool_b = 0;
//        }
//        if (pool_out_col >= output_width) {
//            pool_b = 0;
//            pool_out_col = 0;
//            ++pool_out_row;
//        }
//    }
//
//    auto curr_out_buffer = output + ((b * output_height + out_row) * output_width + out_col) * output_pixel_size;
//    auto in_base_row = out_row * info.stride.rows;
//    auto in_base_col = out_col * info.stride.cols;
//    auto curr_in_base_buffer = input + ((b * input_height + in_base_row) * input_width + in_base_col) * input_pixel_size;
//
//    naive(curr_in_base_buffer, curr_out_buffer, in_dims, info);
//    return true;
//};
//
//void thread_job(const void* ptr) {
//    auto tmp = reinterpret_cast<const packed_arguments*>(ptr);
//
//    while(pull_job( tmp->input,
//                    tmp->output,
//                    tmp->input_height,
//                    tmp->input_width,
//                    tmp->pool_out_row,
//                    tmp->pool_out_col,
//                    tmp->pool_b,
//                    tmp->output_height,
//                    tmp->output_width,
//                    tmp->batch_blocks,
//                    tmp->input_pixel_size,
//                    tmp->output_pixel_size,
//                    tmp->info,
//                    tmp->in_dims,
//                    tmp->mtx
//                   ) );
//};
}

namespace neural {
pooling_cpu_avx2_batch24::pooling_cpu_avx2_batch24(pooling &arg)
    : is_an_implementation(neural::type_id<pooling_cpu_avx2_batch24>())
    , outer(arg) {};
pooling_cpu_avx2_batch24::~pooling_cpu_avx2_batch24() {};
void pooling_cpu_avx2_batch24::implementation(const void *ptr) {
    auto this_pooling = static_cast<const pooling *>(ptr);
    //auto input        = static_cast<float*>(this_pooling->argument.input[0].primitive.as<const memory&>().pointer);
    //auto output       = static_cast<float*>(this_pooling->argument.output[0].as<const memory&>().pointer);

    auto input = static_cast<float*>(this_pooling->argument.input[0].primitive.as<const memory&>().pointer);
    auto output = static_cast<float*>(this_pooling->argument.output[0].as<const memory&>().pointer);

    auto input_memory_arg  = this_pooling->argument.input[0].primitive.as<const memory&>().argument;
    auto input_buffer_size = input_memory_arg.size;
    auto input_offset      = this_pooling->argument.input_offset;

    auto output_memory_arg = this_pooling->argument.output[0].as<const memory&>().argument;
    auto output_buffer_size= output_memory_arg.size;
    auto output_offset     = this_pooling->argument.output_offset;
    auto output_size       = this_pooling->argument.output_size;

    auto stride            = this_pooling->argument.stride;
    auto window            = this_pooling->argument.size;
    auto padding           = this_pooling->argument.padding;

    int b_pos = 3; // todo typetraits
    int f_pos = 2;
    int x_pos = 1;
    int y_pos = 0;
    //uint64_t width = output->get_length(NN_DATA_COORD_x);
    //uint64_t height = output->get_length(NN_DATA_COORD_y);
    //uint64_t batch_blocks = output->get_length(NN_DATA_COORD_n);
    uint64_t width  = output_size.raw[x_pos];
    uint64_t height = output_size.raw[y_pos];
    uint64_t batch_blocks = output_size.raw[b_pos];

    //assert(width == out_dims.width);
    //assert(height == out_dims.height);
    //assert(output->get_length(NN_DATA_COORD_z) == out_dims.feats);
    //assert(input->get_length(NN_DATA_COORD_z) == out_dims.feats);

    //assert(output->get_length(NN_DATA_COORD_p) == BATCH_ACCEPTED_BLOCK); // wtf? coord p?

    //if (input->get_length() != input->parent->lengths)                    // there are no views
    //    throw std::runtime_error("view on input in max pooling batch 24n");
    //if (output->get_length() != output->parent->lengths)
    //    throw std::runtime_error("view on output in max pooling batch 24n");

    //uint64_t input_width = input->parent->lengths.t[NN_DATA_COORD_x];
    //uint64_t input_height = input->parent->lengths.t[NN_DATA_COORD_y];
    //uint64_t input_pixel_size = BATCH_ACCEPTED_BLOCK * out_dims.feats;
    //uint64_t output_width = output->parent->lengths.t[NN_DATA_COORD_x];
    //uint64_t output_height = output->parent->lengths.t[NN_DATA_COORD_y];
    //uint64_t output_pixel_size = BATCH_ACCEPTED_BLOCK * output->parent->lengths.t[NN_DATA_COORD_z];
    uint64_t input_width       = input_buffer_size.raw[x_pos];
    uint64_t input_height      = input_buffer_size.raw[y_pos];
    uint64_t input_pixel_size  = BATCH_ACCEPTED_BLOCK * output_buffer_size.raw[f_pos]; //todo why out?
    uint64_t output_width      = output_buffer_size.raw[x_pos];
    uint64_t output_height     = output_buffer_size.raw[y_pos];
    uint64_t output_pixel_size = BATCH_ACCEPTED_BLOCK * output_buffer_size.raw[f_pos];

    uint64_t pool_b = 0u;
    uint64_t pool_out_row = 0u;
    uint64_t pool_out_col = 0u;

    InputDimensions in_dims = {make<InputHeight>(input_height),
                               make<InputWidth>(input_width),
                               //make<InputFeats>(out_dims.feats)};
                               make<InputFeats>(output_size.raw[f_pos])}; //todo why out?

    //todo added, what is it?
    PoolingDimensions pd(make<PoolingHeight>(window.raw[1]), make<PoolingWidth>(window.raw[0]));
    Stride            ps({make<Rows>(stride.raw[1]), make<Cols>(stride.raw[0])});
    PoolingInfo info({pd,ps});

    std::mutex mtx; //todo remove

    //packed_arguments args{
    //    input,
    //    output,
    //    input_height,
    //    input_width,
    //    pool_out_row,
    //    pool_out_col,
    //    pool_b,
    //    output_height,
    //    output_width,
    //    batch_blocks,
    //    input_pixel_size,
    //    output_pixel_size,
    //    info,
    //    in_dims,
    //    mtx};

    auto pull_job = [&]{
            uint64_t b = 0u;
            uint64_t out_row = 0u;
            uint64_t out_col = 0u;
            {
                std::unique_lock<std::mutex> guard(mtx);
                out_row = pool_out_row;
                out_col = pool_out_col;
                b = pool_b;
                if (out_row >= output_height) return false;

                ++pool_b;
                if (pool_b >= batch_blocks) {
                    pool_out_col++;
                    pool_b = 0;
                }
                if (pool_out_col >= output_width) {
                    pool_b = 0;
                    pool_out_col = 0;
                    ++pool_out_row;
                }
            }

            auto curr_out_buffer = output + ((b * output_height + out_row) * output_width + out_col) * output_pixel_size;
            auto in_base_row = out_row * info.stride.rows;
            auto in_base_col = out_col * info.stride.cols;
            auto curr_in_base_buffer = input + ((b * input_height + in_base_row) * input_width + in_base_col) * input_pixel_size;

            naive(curr_in_base_buffer, curr_out_buffer, in_dims, info);
            return true;
        };
    size_t num_threads = 1; // todo

    pull_job();
//   // std::vector<nn_multithreaded_request> jobs(device->thread_pool.get_num_threads(), {thread_job, nullptr});
//    std::vector<task> jobs(num_threads, {thread_job, nullptr});
//
//    // device->thread_pool.push_job(jobs);
//    for(auto &x: jobs)
//        x.callback(x.data);
}
namespace{
struct attach{
    attach(){
        auto key = std::make_tuple(engine::cpu, memory::format::yxfb_f32, memory::format::yxfb_f32); //todo is this key ok?
        auto val_fw = pooling_cpu_avx2_batch24::create;
  //      auto val_bw = pooling_cpu_avx2_batch24::create;

        pool_fw_implementation_map.insert( {key, val_fw} );
  //      pool_bw_implementation_map.insert( {key, val_bw} );
    }
    ~attach(){}
};

#ifdef __GNUC__
    __attribute__((constructor))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;
}
}

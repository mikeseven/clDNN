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

#include "convolution_cpu_jit_batch1.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"

#include <thread>

namespace neural {

convolution_cpu_jit_batch1::convolution_cpu_jit_batch1(convolution &arg)
        : is_an_implementation(neural::type_id<convolution_cpu_jit_batch1>())
{
    if(arg.argument.input_offset.raw[0])
        tasks.push_back({nullptr, nullptr});
    /*size_t num_threads = 0;
    size_t outpfmap_length = output_buffer->view_end.t[NN_DATA_COORD_z] - output_buffer->view_begin.t[NN_DATA_COORD_z] + 1;
    size_t outpcol_length = output_buffer->view_end.t[NN_DATA_COORD_x] - output_buffer->view_begin.t[NN_DATA_COORD_x] + 1;
    size_t outprow_length = output_buffer->view_end.t[NN_DATA_COORD_y] - output_buffer->view_begin.t[NN_DATA_COORD_y] + 1;

    size_t output_row_package_size;
    size_t output_fmap_package_size;
    size_t output_col_package_size;

    size_t num_output_fm_items;
    size_t num_output_fm_items_remainder;
    size_t num_col_items;
    size_t num_row_items;
    size_t total_workers;

    auto compute_thread_bounds = [&](size_t row_size, size_t fmap_size, size_t col_size, bool spawn_across_column)
    {
        output_row_package_size = row_size;
        output_fmap_package_size = fmap_size;
        output_col_package_size = (spawn_across_column) ? col_size : outpcol_length;

        num_output_fm_items = outpfmap_length / output_fmap_package_size;
        num_output_fm_items_remainder = outpfmap_length % output_fmap_package_size;

        if(num_output_fm_items_remainder) ++num_output_fm_items;

        num_col_items = outpcol_length / output_col_package_size;
        num_row_items = outprow_length / output_row_package_size;

        if(outprow_length % output_row_package_size)
            throw std::runtime_error("convolution needs aligned output height");

        if(outpcol_length % output_col_package_size)
            throw std::runtime_error("convolution needs aligned output width");

        total_workers = num_output_fm_items * num_row_items * num_col_items;
    };

    compute_thread_bounds(4, convolution_f32_impl::C_slice_size, 0, false);

    if(num_threads * 4 > total_workers)
    {
        if(outpcol_length % 960 == 0) compute_thread_bounds(4, convolution_f32_impl::C_slice_size, 960, true);
        else                          compute_thread_bounds(2, convolution_f32_impl::C_slice_size, 0, false);
    }

    if(num_threads * 4 > total_workers)
    {
        if(outpcol_length % 480 == 0) compute_thread_bounds(4, convolution_f32_impl::C_slice_size, 480, true);
        else                          compute_thread_bounds(2, convolution_f32_impl::C_slice_size, 0, false);
    }

    if(num_threads * 4 > total_workers)
    {
        if(outpcol_length % 240 == 0) compute_thread_bounds(4, convolution_f32_impl::C_slice_size, 240, true);
        else                          compute_thread_bounds(2, convolution_f32_impl::C_slice_size, 0, false);
    }

    if(num_threads * 4 > total_workers)
    {
        if(outpcol_length % 120 == 0) compute_thread_bounds(4, convolution_f32_impl::C_slice_size, 120, true);
        else                          compute_thread_bounds(2, convolution_f32_impl::C_slice_size, 0, false);
    }

    if(num_threads * 4 > total_workers)
    {
        if(outpcol_length % 60 == 0)  compute_thread_bounds(4, convolution_f32_impl::C_slice_size, 60, true);
        else                          compute_thread_bounds(2, convolution_f32_impl::C_slice_size, 0, false);
    }

    if(num_threads * 4 > total_workers)
    {
        if(outpcol_length % 30 == 0)  compute_thread_bounds(4, convolution_f32_impl::C_slice_size, 30, true);
        else                          compute_thread_bounds(2, convolution_f32_impl::C_slice_size, 0, false);
    }

    if(num_threads * 4 > total_workers)
    {
        if(outpcol_length % 30 == 0)  compute_thread_bounds(2, convolution_f32_impl::C_slice_size, 30, true);
        else                          compute_thread_bounds(1, convolution_f32_impl::C_slice_size, 0, false);
    }

    if(num_threads * 4 > total_workers)
    {
        if(outpcol_length % 30 == 0)  {
            std::cout << "minimal case occured" << std::endl;
            compute_thread_bounds(1, convolution_f32_impl::C_slice_size, 30, true);
        }
    }

    // Full cores utilization version.
    std::vector<nn::workload_data<> *> input_views(total_workers);
    std::vector<nn::workload_data<> *> weight_views(total_workers);
    std::vector<nn::workload_data<> *> bias_views(total_workers);
    std::vector<nn::workload_data<> *> output_views(total_workers);

    const auto cpp_master_input = input_buffer;
    const auto cpp_master_output = output_buffer;
    const auto cpp_master_weights = weights_buffer;

    // Fill slave work items.
    for (auto output_fm_item = 0u; output_fm_item < num_output_fm_items; ++output_fm_item)
    {
        for (auto row_item = 0u; row_item < num_row_items; ++row_item)
        {
            for (auto col_item = 0u; col_item < num_col_items; ++col_item)
            { 
                auto item_in_pool = col_item * num_output_fm_items * num_row_items + row_item * num_output_fm_items + output_fm_item;

                // Replace nn_workload_datas pointers with views.
                nn_workload_data_coords_t input_view_begin(
                    0,
                    col_item * output_col_package_size * stride_x,
                    row_item * output_row_package_size * stride_y,
                    0,
                    0,
                    0
                    );
                nn_workload_data_coords_t input_view_end(
                    cpp_master_input->get_length(NN_DATA_COORD_n) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_x) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_y) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_z) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_p) - 1,
                    cpp_master_input->get_length(NN_DATA_COORD_q) - 1
                    );

                nn_workload_data_coords_t output_view_begin(
                    0,
                    col_item * output_col_package_size,
                    row_item * output_row_package_size,
                    output_fm_item * output_fmap_package_size,
                    0,
                    0
                    );
                nn_workload_data_coords_t output_view_end(
                    cpp_master_output->get_length(NN_DATA_COORD_n) - 1,
                    (col_item+1) * output_col_package_size - 1,
                    (row_item+1) * output_row_package_size - 1,
                    (output_fm_item+1) * output_fmap_package_size - 1,
                    cpp_master_output->get_length(NN_DATA_COORD_p) - 1,
                    cpp_master_output->get_length(NN_DATA_COORD_q) - 1
                    );

                nn_workload_data_coords_t weights_view_begin(
                    0,
                    0,
                    0,
                    0,
                    0,
                    output_fm_item
                    );
                nn_workload_data_coords_t weights_view_end(
                    cpp_master_weights->get_length(NN_DATA_COORD_n) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_x) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_y) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_z) - 1,
                    cpp_master_weights->get_length(NN_DATA_COORD_p) - 1,
                    output_fm_item
                    );

                if(output_fm_item+1 == num_output_fm_items && num_output_fm_items_remainder)
                {
                    // Case where we need to process only remaining FMaps.
                    output_view_end.t[NN_DATA_COORD_z] = output_view_begin.t[NN_DATA_COORD_z] + num_output_fm_items_remainder - 1;
                    weights_view_end.t[NN_DATA_COORD_p] = num_output_fm_items_remainder - 1;
                }

                input_views[item_in_pool] =
                    new nn::workload_data<>(const_cast<nn::workload_data<>&>(*cpp_master_input), input_view_begin, input_view_end);

                output_views[item_in_pool] =
                    new nn::workload_data<>(*cpp_master_output, output_view_begin, output_view_end);

                weight_views[item_in_pool] =
                    new nn::workload_data<>(const_cast<nn::workload_data<>&>(*cpp_master_weights), weights_view_begin, weights_view_end);

                // Use biases.
                if (bias_buffer != nullptr)
                {
                    const auto cpp_master_biases = bias_buffer;

                    nn_workload_data_coords_t bias_view_begin(
                        0,
                        output_fm_item * output_fmap_package_size,
                        0,
                        0,
                        0,
                        0
                        );
                    nn_workload_data_coords_t bias_view_end(
                        cpp_master_biases->get_length(NN_DATA_COORD_n) - 1,
                        (output_fm_item+1) * output_fmap_package_size - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_y) - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_z) - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_p) - 1,
                        cpp_master_biases->get_length(NN_DATA_COORD_q) - 1
                        );

                    if(output_fm_item+1 == num_output_fm_items && num_output_fm_items_remainder)
                    {
                        // Case where we need to process only remaining FMaps.
                        bias_view_end.t[NN_DATA_COORD_x] = bias_view_begin.t[NN_DATA_COORD_x] + num_output_fm_items_remainder - 1;
                    }

                    bias_views[item_in_pool] =
                        new nn::workload_data<>(const_cast<nn::workload_data<>&>(*cpp_master_biases), bias_view_begin, bias_view_end);
                } else {
                    bias_views[item_in_pool] = nullptr;
                }
            }
        }
    }

    // Create job vector from created views..
    precompiled_jobs.clear();
    precompiled_jobs.resize(num_threads);
    precompiled_request_handles.clear();
    precompiled_request_handles.resize(num_threads);

    for(auto& joblist : precompiled_request_handles)
        joblist.resize(total_workers/num_threads);

    for(size_t jobs_left = total_workers%num_threads; jobs_left; --jobs_left)
        precompiled_request_handles[jobs_left].resize(precompiled_request_handles[jobs_left].size() + 1);

    size_t item_in_pool = 0;

    for (size_t thread = 0u; thread < num_threads; ++thread)
    {
        for (size_t item_in_thread = 0u; item_in_thread < precompiled_request_handles[thread].size(); ++item_in_thread)
        {
            auto& input_view = input_views[item_in_pool];
            auto& weights = weight_views[item_in_pool];
            auto& bias = bias_views[item_in_pool];
            auto& output_view = output_views[item_in_pool];

            auto input_ptr = static_cast<float*>(input_view->parent->data_buffer);
            auto weights_ptr = static_cast<float*>(weights->parent->data_buffer);
            auto bias_ptr = static_cast<float*>(bias->parent->data_buffer);
            auto output_ptr = static_cast<float*>(output_view->parent->data_buffer);

            size_t input_width = input_view->parent->lengths.t[NN_DATA_COORD_x];
            size_t input_height = input_view->parent->lengths.t[NN_DATA_COORD_y];
            size_t input_depth = input_view->parent->lengths.t[NN_DATA_COORD_z];

            size_t output_width = output_view->parent->lengths.t[NN_DATA_COORD_x];
            size_t output_height = output_view->parent->lengths.t[NN_DATA_COORD_y];
            size_t output_depth = output_view->parent->lengths.t[NN_DATA_COORD_z];

            size_t kernel_width = weights->parent->lengths.t[NN_DATA_COORD_x];
            size_t kernel_height = weights->parent->lengths.t[NN_DATA_COORD_y];

            size_t kernel_input_fmap_view_start = weights->view_begin.t[NN_DATA_COORD_z];

            size_t input_fmap_view_start = input_view->view_begin.t[NN_DATA_COORD_z];
            size_t input_fmap_view_end = input_view->view_end.t[NN_DATA_COORD_z];

            size_t output_fm_view_start = output_view->view_begin.t[NN_DATA_COORD_z];
            size_t output_fm_view_end = output_view->view_end.t[NN_DATA_COORD_z];

            size_t output_row_view_start = output_view->view_begin.t[NN_DATA_COORD_y];
            size_t output_row_view_end = output_view->view_end.t[NN_DATA_COORD_y];

            size_t output_column_view_start = output_view->view_begin.t[NN_DATA_COORD_x];
            size_t output_column_view_end = output_view->view_end.t[NN_DATA_COORD_x];

            size_t input_column_view_start = input_view->view_begin.t[NN_DATA_COORD_x];
            size_t input_row_view_start = input_view->view_begin.t[NN_DATA_COORD_y];

            size_t kernel_out_fmap_view_start = weights->view_begin.t[NN_DATA_COORD_q];

            size_t output_image_view_start = output_view->view_begin.t[NN_DATA_COORD_n];
            size_t output_image_view_end = output_view->view_end.t[NN_DATA_COORD_n];

            size_t kernel_ifmap = weights->parent->lengths.t[NN_DATA_COORD_z];

            size_t output_fmaps_view_length = output_fm_view_end - output_fm_view_start + 1;
            size_t input_fmap_view_length = input_fmap_view_end - input_fmap_view_start + 1;
            size_t block_size = (output_fmaps_view_length == 3) 
                ? 4 
                : ((output_fmaps_view_length == 8)
                   ? 14
                   : 6);

            size_t out_fm_pckg_size = (output_fmaps_view_length == 3) 
                ? 3 
                : ((output_fmaps_view_length == 8)
                   ? 8
                   : 16);

            size_t output_column_view_length = output_column_view_end - output_column_view_start + 1;
            size_t num_blocks_full      = output_column_view_length / block_size;
            size_t partial_block_size   = output_column_view_length % block_size;

            precompiled_request_handles[thread][item_in_thread].input_ptr                    = input_ptr;                 
            precompiled_request_handles[thread][item_in_thread].weights_ptr                  = weights_ptr;
            precompiled_request_handles[thread][item_in_thread].bias_ptr                     = bias_ptr;
            precompiled_request_handles[thread][item_in_thread].output_ptr                   = output_ptr;
            precompiled_request_handles[thread][item_in_thread].input_fmap_view_start        = input_fmap_view_start;
            precompiled_request_handles[thread][item_in_thread].input_fmap_view_end          = input_fmap_view_end;
            precompiled_request_handles[thread][item_in_thread].input_column_view_start      = input_column_view_start - center_offset_x;
            precompiled_request_handles[thread][item_in_thread].input_row_view_start         = input_row_view_start - center_offset_y;
            precompiled_request_handles[thread][item_in_thread].kernel_input_fmap_view_start = kernel_input_fmap_view_start;
            precompiled_request_handles[thread][item_in_thread].kernel_out_fmap_view_start   = kernel_out_fmap_view_start * layer::convolution_f32_impl::C_slice_size;
            precompiled_request_handles[thread][item_in_thread].output_column_view_start     = output_column_view_start;
            precompiled_request_handles[thread][item_in_thread].output_column_view_end       = output_column_view_end;
            precompiled_request_handles[thread][item_in_thread].output_row_view_start        = output_row_view_start;
            precompiled_request_handles[thread][item_in_thread].output_row_view_end          = output_row_view_end;
            precompiled_request_handles[thread][item_in_thread].output_fm_view_start         = output_fm_view_start;
            precompiled_request_handles[thread][item_in_thread].output_fm_view_end           = output_fm_view_end;
            precompiled_request_handles[thread][item_in_thread].output_image_view_start      = output_image_view_start;
            precompiled_request_handles[thread][item_in_thread].output_image_view_end        = output_image_view_end;
            precompiled_request_handles[thread][item_in_thread].padding                      = padding;
            precompiled_request_handles[thread][item_in_thread].jit_callback                 = 
                layer::convolution_f32_impl::get_generator<layer::convolution_f32_impl::convolution_generator>(
                    activation.function,
                    input_width,
                    input_height,
                    input_depth,
                    input_fmap_view_end - input_fmap_view_start + 1,
                    kernel_width,
                    kernel_height,
                    kernel_ifmap,
                    output_width,
                    output_height,
                    output_depth,
                    block_size,
                    out_fm_pckg_size,
                    num_blocks_full,
                    partial_block_size,
                    stride_x,
                    stride_y)->generator_callback;


            ++item_in_pool;
        }

        precompiled_jobs[thread] = {convolution_f32_impl::unpack_convolve_callback_handle, &precompiled_request_handles[thread]};
    }*/
};

convolution_cpu_jit_batch1::~convolution_cpu_jit_batch1() 
{
};

namespace
{

struct attach
{
    attach()
    {
        auto key_fw = std::make_tuple(engine::cpu, memory::format::byxf_f32, memory::format::byxf_f32);
        auto val_fw = convolution_cpu_jit_batch1::create;

        conv_fw_implementation_map.insert( {key_fw, val_fw} );
    }

    ~attach()
    {
    }
};

#ifdef __GNUC__
    __attribute__((visibility("default")))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

}
}

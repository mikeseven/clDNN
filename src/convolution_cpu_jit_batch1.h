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

#pragma once

#include "convolution.h"

namespace neural {

    typedef void (convolution_generator_callback_t)(
        float* input_ptr, 
        float* weights_ptr, 
        float* bias_ptr, 
        float* output_ptr,
        uint64_t input_fmap_view_start,
        uint64_t input_fmap_view_end,
        uint64_t input_column_view_start,
        uint64_t input_row_view_start,
        uint64_t kernel_input_fmap_view_start,
        uint64_t kernel_out_fmap_view_start,
        uint64_t output_column_view_start,
        uint64_t output_row_view_start,
        uint64_t output_row_view_end,
        uint64_t output_fm_view_start,
        uint64_t output_fm_view_end,
        uint64_t output_image_view_start,
        uint64_t output_image_view_end);

    struct parameters_convolution_f32_precompiled_jit
    {
        float**                            input_ptr; 
        float**                            weights_ptr; 
        float**                            bias_ptr; 
        float**                            output_ptr;
        uint64_t                           input_fmap_view_start;
        uint64_t                           input_fmap_view_end;
        uint64_t                           input_column_view_start;
        uint64_t                           input_row_view_start;
        uint64_t                           kernel_input_fmap_view_start;
        uint64_t                           kernel_out_fmap_view_start;
        uint64_t                           output_column_view_start;
        uint64_t                           output_column_view_end;
        uint64_t                           output_row_view_start;
        uint64_t                           output_row_view_end;
        uint64_t                           output_fm_view_start;
        uint64_t                           output_fm_view_end;
        uint64_t                           output_image_view_start;
        uint64_t                           output_image_view_end;
        neural::padding::type              padding;
        convolution_generator_callback_t*  callback;
    };


    class convolution_cpu_jit_batch1 : is_an_implementation {
        std::vector<task> tasks;

        std::vector<parameters_convolution_f32_precompiled_jit> precompiled_request_handles;

    public:
        convolution_cpu_jit_batch1(convolution &arg);
        ~convolution_cpu_jit_batch1();

        static is_an_implementation *create(convolution &arg) { return new convolution_cpu_jit_batch1(arg); };
        task_package work() { return this->tasks; };
    };
}
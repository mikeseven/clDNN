﻿/*
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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "kernel_runner.h"
#include "kernel.h"
#include <chrono>

namespace cldnn { namespace gpu {

kernel_runner::kernel_runner(engine_impl& engine_ref, bool weights_and_bias_exist) :
    engine(&engine_ref),
    weights_and_bias_exist(weights_and_bias_exist)
{
}

void kernel_runner::prepare_kernel_args(const KernelSelector::KernelsData& kernels_data, gpu::kernel::kernel_arguments_data& args)
{
    const auto& base_params = *static_cast<KernelSelector::BaseParams*>(kernels_data[0].params.get());

    // Prepare input buffers
    if (input_buffers.empty())
    {
        for (const auto& input : base_params.inputs)
        {
            int num_of_input_elements = (int)input.PhysicalSize();
            input_buffers.push_back(engine->allocate_buffer({ from_data_type(input.GetDType()), format::bfyx, tensor(1, 1, num_of_input_elements, 1) }));
        }
    }
    for (const auto& input : input_buffers)
    {
        args.inputs.push_back(input);
    }

    // Prepare output buffer
    if (output_buffers.empty())
    {
        int num_of_output_elements = (int)base_params.output.PhysicalSize();
        output_buffers.push_back(engine->allocate_buffer({ from_data_type(base_params.output.GetDType()), format::bfyx, tensor(1, 1, num_of_output_elements, 1) }));
    }

    args.output = output_buffers[0];

    if (weights_and_bias_exist)
    {
        // Prepare weight buffer
        const auto& weights_bias_params = *static_cast<KernelSelector::WeightBiasParams*>(kernels_data[0].params.get());
        int num_of_weight_elements = (int)weights_bias_params.weights.PhysicalSize();
        if (weight_buffers.empty())
        {
            weight_buffers.push_back(engine->allocate_buffer({ from_weights_type(weights_bias_params.weights.GetDType()), format::bfyx, tensor(1, 1, num_of_weight_elements, 1) }));
        }
        while (weight_buffers[0]->get_layout().bytes_count() < weights_bias_params.weights.PhysicalSizeInBytes())
        {
            // Weights layout depends on the kernel. Multiply the buffer size by 2 until it is big enough 
            // (to avoid complex computations of the exact buffer size according to the chosen layout). 
            weight_buffers.clear();
            num_of_weight_elements *= 2;
            weight_buffers.push_back(engine->allocate_buffer({ from_weights_type(weights_bias_params.weights.GetDType()), format::bfyx, tensor(1, 1, num_of_weight_elements, 1) }));
        }
        args.weights = weight_buffers[0];

        // Prepare bias buffer
        if (!weights_bias_params.bias.empty())
        {
            if (bias_buffers.empty())
            {
                int num_of_bias_elements = (int)weights_bias_params.bias[0].PhysicalSize();
                bias_buffers.push_back(engine->allocate_buffer({ from_data_type(weights_bias_params.bias[0].GetDType()), format::bfyx, tensor(1, 1, num_of_bias_elements, 1) }));
            }
            args.bias = bias_buffers[0];
        }  
    }

    args.split = 0;
}

std::vector<uint64_t> kernel_runner::run_kernels(const KernelSelector::KernelsData& kernels_data)
{
    auto context = engine->get_context();

    std::vector<uint64_t> run_times;

    int num_of_kernels_to_run = (int)kernels_data.size();

    KernelSelector::KernelsData::const_iterator batch_start = kernels_data.begin();
    KernelSelector::KernelsData::const_iterator batch_end;

    while (num_of_kernels_to_run > 0)
    {
        int current_compilation_batch = std::min(num_of_kernels_to_run, compilation_batch_size);
        batch_end = batch_start + current_compilation_batch;

        std::vector<gpu::kernel> kernels;

        for (auto it = batch_start; it < batch_end; it++)
        {
            kernels.push_back(kernel(context, it->kernels[0].kernelString, false, true));
        }

        gpu::kernel::kernel_arguments_data args;

        prepare_kernel_args(kernels_data, args);

        int i = 0;
        for (auto it = batch_start; it < batch_end; it++)
        {
            std::vector<event_impl::ptr> events;
            uint64_t kernel_run_time = 0;
            int num_of_runs = 0;

            for (int iteration = 0; iteration < runs_per_kernel; iteration++)
            {
                event_impl::ptr event;
                try
                {
                    event = kernels[i].run(it->kernels[0], {}, args);
                }
                catch (...)
                {
                    // Could not run this kernel. Push back NULL event (will be ignored later).
                }
                events.push_back(event);
            }
                
            context->queue().finish();

            for (auto event : events)
            {
                if (event.get() != NULL)
                {
                    auto profiling_intervals = event->get_profiling_info();
                    for (auto const& profiling_interval : profiling_intervals)
                    {
                        if (strcmp(profiling_interval.name, "executing") == 0)
                        {
                            kernel_run_time += profiling_interval.nanoseconds;
                            num_of_runs++;
                            break;
                        }
                    }
                }
            }

            if (num_of_runs > 0)
            {
                run_times.push_back(kernel_run_time / num_of_runs);
            }
            else
            {
                run_times.push_back(std::numeric_limits<uint64_t>::max());
            }
            i++;
        }

        num_of_kernels_to_run -= current_compilation_batch;
        batch_start += current_compilation_batch;
    }

    return run_times;
}

}}

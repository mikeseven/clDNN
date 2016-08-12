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
#include "implementation_map.h"
#include "multidimensional_counter.h"
#include "kernel.h"

namespace neural {

    const std::string kernelName = "Mean_subtract_GPU";
    const std::string kernelCode = R"__krnl(
KERNEL(Mean_subtract_GPU)(const __global neural_memory* input_mem, __global neural_memory* output_mem, const __global neural_memory* mean_mem)
{
    const __global float* input = (const __global float*)get_data(input_mem);
    __global float* output = (__global float*)get_data(output_mem);
    const __global float* mean = (const __global float*)get_data(mean_mem);
 
    __global uint* input_size = get_raw(input_mem);

    const uint batch_num = input_size[0];

    const int global_id = get_global_id(0);
    output[global_id] = input[global_id] - mean[global_id / batch_num];
}
)__krnl";

struct mean_subtract_gpu : is_an_implementation {
    mean_subtract &outer;
    gpu::kernel _kernel;

    mean_subtract_gpu(mean_subtract &arg)
        : is_an_implementation(neural::type_id<mean_subtract_gpu>())
        , outer(arg)
        , _kernel(kernelName)
    {}

    static void implementation(const void *ptr) {
        auto me = static_cast<const mean_subtract_gpu*>(ptr);
        auto& outer = me->outer;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);
        auto& mean_mem = outer.input_memory(1);

        size_t output_bufSize = outer.output_memory(0).count();

        // calculate local workgroup size
        int lws = 16;
        while (output_bufSize % lws)
        {
            lws--;
        }

        me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>
            ({ output_bufSize, std::min(output_bufSize, static_cast<size_t>(lws)) }, input_mem, output_mem, mean_mem);
    }

    static is_an_implementation *create(mean_subtract &arg) {
        auto& mean_arg = arg.input_memory(1).argument;
        if (mean_arg.format != memory::format::yxfb_f32) throw std::runtime_error("mean_subtract mean isn't yxfb_f32 format");
        return new mean_subtract_gpu(arg);
    }

    task_group work() override { return{ { task{ implementation, this } }, schedule::single }; };
};

namespace {
    struct attach {
        attach() {
            gpu::kernel_templates::add(kernelName, kernelCode);

            auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
            auto val_fw = mean_subtract_gpu::create;

            implementation_map<mean_subtract>::add(key_fw, val_fw);
        }
        ~attach() {}
    };

#ifdef __GNUC__
        __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
        attach attach_impl;

    }
}

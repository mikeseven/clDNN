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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "api/neural.h"
#include "multidimensional_counter.h"
#include "fully_connected.h"
#include "kernel.h"

const std::string kernelName = "Fully_Connected_GPU";
const std::string kernelCode = R"__krnl(
__kernel void Fully_Connected_GPU(__global neural_memory* input_mem, __global neural_memory* weights_mem, __global neural_memory* bias_mem, __global neural_memory* dst_mem)
{
    __global uint* input_size = get_raw(input_mem);
    __global uint* weights_size = get_raw(weights_mem);
    __global float* input = (__global float*)get_data(input_mem);
    __global float* weights = (__global float*)get_data(weights_mem);
    __global float* bias = (__global float*)get_data(bias_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);

    const int x = get_global_id(0);

    pDst[x] = 0;
    uint outXIdx = x / input_size[0];
    uint inputBatchIdx = x % input_size[0];
    uint weightYIdx = outXIdx * weights_size[0];
    for (uint i = 0; i < input_size[2]; i++)
    {
        pDst[x] += input[i * input_size[0] + inputBatchIdx] * weights[weightYIdx + i];
    }
    pDst[x] += bias[outXIdx];
}
)__krnl";

namespace neural {

    struct fully_connected_gpu : is_an_implementation {
        const fully_connected &outer;
        gpu::kernel<gpu::input_mem, gpu::input_mem, gpu::input_mem, gpu::output_mem> _kernel;

        fully_connected_gpu(fully_connected &arg, const std::string& kernel_name)
            : is_an_implementation(neural::type_id<fully_connected_gpu>())
            , outer(arg), _kernel(kernel_name)
        {};
        ~fully_connected_gpu() {}

        static void implementation(const void *ptr) {
            auto me = static_cast<const fully_connected_gpu *>(ptr);
            auto this_fc = &me->outer;

            // input
            auto& input_mem = this_fc->input_memory(0);

            assert(1 == input_mem.argument.size.feature.size());
            assert(1 == input_mem.argument.size.batch.size());
            assert(1 == input_mem.argument.size.feature[0]);

            // weights
            auto& weight_mem = this_fc->input_memory(1);

            // bias
            auto& bias_mem = this_fc->input_memory(2);

            // output
            auto& output_mem = this_fc->output_memory(0);
            
            auto output_bufSize = output_mem.count();

            const_cast<fully_connected_gpu*>(me)->_kernel({ output_bufSize, output_bufSize }, input_mem, weight_mem, bias_mem, output_mem);
        }

        task_group work() override {
            return{ { task{ implementation, this } }, schedule::single };
        }

        static is_an_implementation *create(fully_connected &arg) {
            return new fully_connected_gpu(arg, kernelName);
        };
    };

namespace {
    struct attach {
        attach() {
            gpu::kernel_templates::add(kernelName, kernelCode);
            auto val_fw = fully_connected_gpu::create;
            fully_connected_fw_implementation_map::instance().insert({ std::make_tuple(engine::gpu, memory::format::xb_f32, memory::format::xb_f32), val_fw });
            fully_connected_fw_implementation_map::instance().insert({ std::make_tuple(engine::gpu, memory::format::x_f32,  memory::format::x_f32), val_fw });
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
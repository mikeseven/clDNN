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
#include "fully_connected_common_gpu.h"
#include "fully_connected_relu.h"
#include "kernel.h"

const std::string kernelName = "Fully_Connected_Relu_GPU";
const std::string kernelCode_Begin = R"__krnl(
KERNEL (Fully_Connected_Relu_GPU)(__global neural_memory* input_mem, __global neural_memory* dst_mem)
{
)__krnl";
const std::string kernelCode_Relu = R"__krnl(
pDst[x] = max(pDst[x], 0.0f) + NEGATIVE_SLOPE * min(pDst[x], 0.0f);
)__krnl";

const std::string kernelCode_End = R"__krnl(
}
)__krnl";

namespace neural {

    struct fully_connected_relu_gpu : is_an_implementation {
        const fully_connected_relu &outer;
       
        fully_connected_relu_gpu(fully_connected_relu &arg)
            : is_an_implementation(neural::type_id<fully_connected_relu_gpu>())
            , outer(arg)
        {};
        ~fully_connected_relu_gpu() {}

        static void implementation(const void *ptr) {
            auto me = static_cast<const fully_connected_relu_gpu *>(ptr);
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

            float negative_slope = this_fc->argument.negative_slope;
            gpu::jit_constants mem_consts{  
                gpu::make_jit_constant("WEIGHTS", weight_mem),
                gpu::make_jit_constant("BIASES", bias_mem),
                gpu::make_jit_constant("NEGATIVE_SLOPE", std::to_string(negative_slope))
            };

            gpu::kernel<gpu::input_mem, gpu::output_mem> _kernel(kernelName, mem_consts);
            _kernel({ output_bufSize, output_bufSize }, input_mem, output_mem);
        }

        task_group work() override {
            return{ { task{ implementation, this } }, schedule::single };
        }

        static is_an_implementation *create(fully_connected_relu &arg) {
            return new fully_connected_relu_gpu(arg);
        };
    };

namespace {
    struct attach {
        attach() {
            gpu::kernel_templates::add(kernelName, kernelCode_Begin + fully_connected_code_xb + kernelCode_Relu + kernelCode_End);
            auto val_fw = fully_connected_relu_gpu::create;
            fully_connected_relu_fw_implementation_map::instance().insert({ std::make_tuple(engine::gpu, memory::format::xb_f32, memory::format::xb_f32), val_fw });
            fully_connected_relu_fw_implementation_map::instance().insert({ std::make_tuple(engine::gpu, memory::format::x_f32,  memory::format::x_f32), val_fw });
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
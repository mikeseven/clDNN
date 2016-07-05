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

#include "relu_gpu.h"
#include "multidimensional_counter.h"
#include "kernel.h"

namespace neural {

const std::string kernelName = "Relu_GPU";
const std::string kernelCode = R"__krnl(
KERNEL(Relu_GPU)(const __global neural_memory* input_mem, __global neural_memory* output_mem)
{
    const __global float* input = (const __global float*)get_data(input_mem);
    __global float* output = (__global float*)get_data(output_mem);
    
    const int global_id = get_global_id(0);
    output[global_id] = max(0.0f, input[global_id]);
}
)__krnl";

relu_gpu::relu_gpu(relu &arg)
    : is_an_implementation(neural::type_id<relu_gpu>())
    , outer(arg)
{};

relu_gpu::~relu_gpu() {}

void relu_gpu::implementation(const void *ptr) {
    auto this_relu = static_cast<const relu *>(ptr);

    //auto& output_size   = this_relu->argument.output_size;

    //assert( 1 == output_size.feature.size() );
    //assert( 1 == output_size.batch.size()   );

    auto& input_mem = this_relu->input_memory(0);
    auto& output_mem = this_relu->output_memory(0);

    size_t dstSize = output_mem.count();

    int lws = 16;
    while (dstSize % lws)
    {
        lws--;
    }

    auto kernel = gpu::kernel<gpu::input_mem, gpu::output_mem>(kernelName);
    kernel({ dstSize, std::min(dstSize, (size_t)lws) }, input_mem, output_mem);

}

namespace {
struct attach {
    attach() {
        gpu::kernel_templates::add(kernelName, kernelCode);
        auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = relu_gpu::create;

        relu_fw_implementation_map::instance().insert( {key_fw, val_fw} ); //todo keys should be different
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

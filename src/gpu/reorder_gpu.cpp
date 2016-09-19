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
#include "multidimensional_counter.h"
#include "implementation_map.h"
#include "kernel.h"

const std::string kernelName = "reorder_GPU";
const std::string kernelCode = R"__krnl(
//uint FUNC(YXFB)(uint size[4], uint pos[4]) {
//    return pos[1] + size[1] * (pos[2] + size[2] * (pos[3] + size[3] * pos[0]));
//}
KERNEL (reorder_GPU)(__global neural_memory* input_mem, __global neural_memory* dst_mem)
{
    __global uint* input_size = get_raw(input_mem);
    __global float* input = (__global float*)get_data(input_mem);
    __global float* pDst = (__global float*)get_data(dst_mem);

    const int global_id = get_global_id(0);

    const int batch_num = input_size[0];
    const int batch_offset = global_id % batch_num;

    const int ifm_num = input_size[1];
    const int ifm_offset = (global_id / batch_num) % ifm_num;

    const int x = (global_id / batch_num) / ifm_num;

    uint pos1D = global_id;
    uint pos[DIMENSIONS]; // position in each of dimensions
    for(uint i = DIMENSIONS-1; i >= 0 ; i--)
    {
        //pos[i] = pos1D % SIZE[i];
        pos1D /= SIZE[i]; 
    }

    //uint input_pos=0;// = FUNC_CALL(YXFB)(SIZE, pos);
    //uint output_pos=0;

    //pDst[output_pos] = input[input_pos];
}
)__krnl";

namespace neural {
struct reorder_gpu : is_an_implementation {
    const reorder& outer;
        gpu::kernel _kernel;

        reorder_gpu(reorder &arg): is_an_implementation(neural::type_id<reorder_gpu>())
        , outer(arg)
        , _kernel(kernelName, get_jit_constants())
    {}

    gpu::jit_constants get_jit_constants() const {
        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        std::stringstream s;
        s << "(uint[]){ ";
        for (auto i = 0; i < input_mem.argument.size.raw.size(); i++)
        {
            s << static_cast<float>(input_mem.argument.size.raw[i]) << ", ";
        }
        s << " }";
        
        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT", input_mem.argument.size),
            gpu::make_jit_constant("OUTPUT", output_mem.argument.size),
            gpu::make_jit_constant("DIMENSIONS", std::to_string(input_mem.argument.size.raw.size())),
            gpu::make_jit_constant("SIZE", s.str())
        };

        return mem_consts;
    }
    static void implementation(const void *ptr) {
        auto me = static_cast<const reorder_gpu*>(ptr);
        auto& outer = me->outer;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        if (input_mem.argument.size.raw.size() != output_mem.argument.size.raw.size()) throw std::runtime_error("Reorder input/output number of dimension does not match.");

        size_t dstSize = output_mem.count();

        int lws = 16;
        while (dstSize % lws)
        {
            lws--;
        }

        me->_kernel.run<gpu::input_mem, gpu::output_mem>
            ({ dstSize, std::min(dstSize, static_cast<size_t>(lws)) },
                input_mem,
                output_mem);
    }


    static is_an_implementation *create(reorder &arg) {
        auto input_arg = arg.input_memory(0).argument;
        auto output_arg = arg.output_memory(0).argument;

        return new reorder_gpu(arg);
    }

    task_group work() override { return{ { task{ implementation, this } }, schedule::single }; }

};
    
    template<>
    struct implementation_key<reorder> {
        typedef neural::engine::type type;
        type operator()(reorder& primitive) { return primitive.argument.engine; }
    };

    namespace {
        struct attach {
            attach() {
                gpu::kernel_templates::add(kernelName, kernelCode);
                implementation_map<reorder>::add({
                    { engine::type::gpu, reorder_gpu::create }
                });
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
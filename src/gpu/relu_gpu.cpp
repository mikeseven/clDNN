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
#include "relu_gpu.h"
#include "cache/primitive_db.h"

namespace neural 
{

const std::string kernelName = "Relu_GPU";

struct relu_gpu : is_an_implementation {
    relu &outer;
    gpu::kernel _kernel;

    relu_gpu(relu &arg) : is_an_implementation(neural::type_id<relu_gpu>())
        , outer(arg)
        , _kernel(kernelName, get_jit_constants()) {}

	gpu::jit_constants get_jit_constants() const
	{
		gpu::jit_constants mem_consts
		{
			gpu::make_jit_constant("RELU", ""),
			gpu::make_jit_constant("NEGATIVE_SLOPE", outer.argument.negative_slope),
		};

		return mem_consts;
	}

    static void implementation(const void *ptr) 
	{
        auto me = static_cast<const relu_gpu *>(ptr);
        auto& outer = me->outer;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        size_t dstSize = output_mem.count();

        int lws = 16;
        while (dstSize % lws)
        {
            lws--;
        }

        me->_kernel.run<gpu::input_mem, gpu::output_mem>
            ({ dstSize, std::min(dstSize, static_cast<size_t>(lws)) }, input_mem, output_mem);
    }

    static is_an_implementation *create(relu &arg) { return new relu_gpu(arg); };
    task_group work() override { return{ { task{ implementation, this } }, schedule::unordered }; };
};


namespace {
struct attach {
    attach() {
		// cache implementation phase #1 that is a initial switch for using primitive database instead of string kernels
		// at later steps primitive database will be created only once per loading library but as for now it would require 
		// large refactor, so it will be done in smaller incremental steps. The same goes for picking first implementation
		// from the returned list.
		gpu::manager::primitive_db database;
        gpu::kernel_templates::add(kernelName, database.get(kernelName).at(0));
        auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = relu_gpu::create;

        implementation_map<relu>::add(key_fw, val_fw); //todo keys should be different
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

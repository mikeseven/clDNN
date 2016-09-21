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
uint FUNC(OUT_FORMAT)(uint size[4], uint pos[4]) {
    OUT_FORMAT_IMPLEMENTATION
}
KERNEL (reorder_GPU)(__global neural_memory* in_mem, __global neural_memory* out_mem)
{
    __global float* input = (__global float*)get_data(in_mem);
    __global float* output = (__global float*)get_data(out_mem);

    uint pos1D = get_global_id(0);
    uint pos[DIMENSIONS]; // position in each of dimensions
	for(uint i = 0; i < DIMENSIONS; i++)
    {
		uint order_idx = CALCULATION_ORDER[i];
		pos[order_idx] = pos1D % SIZE[order_idx];
        pos1D /= SIZE[order_idx]; 
    }

    uint output_pos = FUNC_CALL(OUT_FORMAT)(SIZE, pos);

    output[output_pos] = input[get_global_id(0)];
}
)__krnl";

const std::string kernelName_subtract = "reorder_subtract_GPU";
const std::string kernelCode_subtract = R"__krnl(
uint FUNC(OUT_FORMAT)(uint size[DIMENSIONS], uint pos[DIMENSIONS]) {
    OUT_FORMAT_IMPLEMENTATION
}
uint FUNC(SUBTRACT_FORMAT)(uint size[DIMENSIONS], uint pos[DIMENSIONS]) {

	SUBTRACT_FORMAT_IMPLEMENTATION
}
KERNEL (reorder_subtract_GPU)(__global neural_memory* in_mem, __global neural_memory* out_mem, __global neural_memory* subtract_values)
{
    __global float* input = (__global float*)get_data(in_mem);
    __global float* output = (__global float*)get_data(out_mem);
    __global float* subtract = (__global float*)get_data(subtract_values);

    uint pos1D = get_global_id(0);
    uint pos[DIMENSIONS]; // position in each of dimensions
	for(uint i = 0; i < DIMENSIONS; i++)
    {
		uint order_idx = CALCULATION_ORDER[i];
		pos[order_idx] = pos1D % SIZE[order_idx];
        pos1D /= SIZE[order_idx]; 
    }

    uint output_pos = FUNC_CALL(OUT_FORMAT)(SIZE, pos);
	// We set it to 0 because we subtract the same values from every input batch
	pos[0] = 0;
	uint subtract_pos = FUNC_CALL(SUBTRACT_FORMAT)(SIZE, pos);
    output[output_pos] = input[get_global_id(0)] - subtract[subtract_pos];
}
)__krnl";

namespace neural {
struct reorder_gpu : is_an_implementation {
    const reorder& outer;
	bool have_subtraction;
	gpu::kernel _kernel;

        reorder_gpu(reorder &arg): is_an_implementation(neural::type_id<reorder_gpu>())
        , outer(arg)
		, have_subtraction(arg.argument.input.size() > 1)
        , _kernel(select_kernel_name(), get_jit_constants())
    {}

	// We need to specify the output idx based on input position
	static std::string get_idx_calculation(memory::format::type type) {
		switch (type)
		{
		case memory::format::type::yxfb_f32:
			return "return pos[0] + size[0] * (pos[1] + size[1]*(pos[2] + size[2] * pos[3]));";
		case memory::format::type::byxf_f32:
			return "return pos[1] + size[1] * (pos[2] + size[2] * (pos[3] + size[3] * pos[0]));";
		case memory::format::type::bfyx_f32:
			return "return pos[2] + size[2] * (pos[3] + size[3] * (pos[1] + size[1] * pos[0]));";
		default:
			throw std::invalid_argument("This format is not supported in GPU reorder");
		}
	}

	// To read input memory linearly we need to specify the order of reading
	static std::string get_calculation_order(memory::format::type type)
	{
		switch (type)
		{
		case memory::format::type::byxf_f32:
			return "(uint[]){ 1, 2, 3, 0 }";
		case memory::format::type::yxfb_f32:
			return "(uint[]){ 0, 1, 2, 3 }";
		case memory::format::type::bfyx_f32:
			return "(uint[]){ 2, 3, 1, 0 }";
		default:
			throw std::invalid_argument("This format is not supported in GPU reorder");
		}
	}

	const std::string& select_kernel_name() const {
		// if we got values to subtract, then choose apropriate kernel
		if (this->have_subtraction)
			return kernelName_subtract;
		else
			return kernelName;

	}
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
            gpu::make_jit_constant("DIMENSIONS", std::to_string(input_mem.argument.size.raw.size())),
            gpu::make_jit_constant("SIZE", s.str()),
			gpu::make_jit_constant("OUT_FORMAT_IMPLEMENTATION", get_idx_calculation(output_mem.argument.format)),
			gpu::make_jit_constant("CALCULATION_ORDER", get_calculation_order(input_mem.argument.format))
        };

		if (this->have_subtraction)
		{
			auto& subtract_mem = outer.input_memory(1);
			mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_FORMAT_IMPLEMENTATION", get_idx_calculation(subtract_mem.argument.format)));
		}
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

		if (me->have_subtraction)
		{
			me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>
				({ dstSize, std::min(dstSize, static_cast<size_t>(lws)) },
					input_mem,
					output_mem,
					outer.input_memory(1));
		}
		else
		{
			me->_kernel.run<gpu::input_mem, gpu::output_mem>
				({ dstSize, std::min(dstSize, static_cast<size_t>(lws)) },
					input_mem,
					output_mem);
		}
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
				gpu::kernel_templates::add(kernelName_subtract, kernelCode_subtract);
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
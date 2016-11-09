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
#include "ocl_toolkit.h"
#include "cache/primitive_db.h"

const std::string kernelName = "reorder_GPU";
const std::string kernelName_subtract = "reorder_subtract_GPU";
const std::string kernelName_subtract_values = "reorder_subtract_values_GPU";

namespace neural {
// GPU engine information helpers.
namespace
{
struct gpu_info_helper : gpu::context_holder
{
    gpu::engine_info get_engine_info() const
    {
        return context()->get_engine_info();
    }
};
}

struct reorder_gpu : is_an_implementation {
    const reorder& outer;
    bool have_subtraction;
    gpu::kernel _kernel;
    gpu::kernel_execution_options _exec_options;

    reorder_gpu(reorder &arg): is_an_implementation(neural::type_id<reorder_gpu>())
    , outer(arg)

    , have_subtraction(arg.argument.input.size() > 1)
    , _kernel(select_kernel_name(), get_jit_constants())
    , _exec_options(get_execution_options())
    {}
    // We need to specify the output idx based on input position
    static std::string get_idx_calculation(memory::format::type type) {
        switch (type)
        {
        case memory::format::type::yxfb_f32:
        case memory::format::type::yxfb_f16:
            return "return pos[0] + size[0] * (pos[1] + size[1]*(pos[2] + size[2] * pos[3]));";
        case memory::format::type::byxf_f32:
        case memory::format::type::byxf_f16:
            return "return pos[1] + size[1] * (pos[2] + size[2] * (pos[3] + size[3] * pos[0]));";
        case memory::format::type::bfyx_f32:
        case memory::format::type::bfyx_f16:
            return "return pos[2] + size[2] * (pos[3] + size[3] * (pos[1] + size[1] * pos[0]));";
        case memory::format::type::oiyx_f32:
        case memory::format::type::oiyx_f16:
            return "return pos[3] + size[3] * (pos[4] + size[4] * (pos[2] + size[2] * pos[1]));";
        case memory::format::type::yxio_f32:
        case memory::format::type::yxio_f16:
            return "return pos[1] + size[1] * (pos[2] + size[2] * (pos[3] + size[3] * pos[4]));";
        case memory::format::type::bx_f32:
        case memory::format::type::bx_f16:
            return "return pos[2] + size[2]*pos[0];";
        case memory::format::type::xb_f32:
        case memory::format::type::xb_f16:
            return "return pos[0] + size[0]*pos[2];";

        default:
            throw std::invalid_argument("This format is not supported in GPU reorder");
        }
    }

    // To read input memory linearly we need to specify the order of reading
    static std::vector<uint32_t> get_calculation_order(memory::format::type type)
    {
        switch(type)
        {
        case memory::format::type::byxf_f32:
        case memory::format::type::byxf_f16:
            return { 1, 2, 3, 0 };
        case memory::format::type::yxfb_f32:
        case memory::format::type::yxfb_f16:
            return { 0, 1, 2, 3 };
        case memory::format::type::bfyx_f32:
        case memory::format::type::bfyx_f16:
            return { 2, 3, 1, 0 };
        case memory::format::type::oiyx_f32:
        case memory::format::type::oiyx_f16:
            return { 0, 3, 4, 2, 1 };
        case memory::format::type::yxio_f32:
        case memory::format::type::yxio_f16:
            return { 0, 1, 2, 3, 4 };
        case memory::format::type::bx_f32:
        case memory::format::type::bx_f16:
            return { 1, 2, 0 };
        case memory::format::type::xb_f32:
        case memory::format::type::xb_f16:
            return { 1, 0, 2 };
        default:
            throw std::invalid_argument("This format is not supported in GPU reorder");
        }
    }

    static std::string get_calculation_order_string(memory::format::type type)
    {
        std::ostringstream os;
        os << "(uint[]){ ";
        for(auto i : get_calculation_order(type)) {
            os << i << ", ";
        }
        os << " }";
        return os.str();
    }

    const std::string& select_kernel_name() const {
        // if we got values to subtract, then choose apropriate kernel
        if (have_subtraction)
            return kernelName_subtract;
        else if (!outer.argument.subtract_per_feature.empty())
            return kernelName_subtract_values;
        else
            return kernelName;
    }

    gpu::jit_constants get_jit_constants() const {
        gpu_info_helper gpu_info;
        auto engine_info = gpu_info.get_engine_info();

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        auto input_use_half = memory::traits(input_mem.argument.format).type->name == type_id<half_t>()->name;
        auto output_use_half = memory::traits(output_mem.argument.format).type->name == type_id<half_t>()->name;
        int input_output_type_cvt = input_use_half != output_use_half;

        if (!engine_info.supports_fp16 && (input_use_half || output_use_half))
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("DIMENSIONS", std::to_string(input_mem.argument.size.raw.size())),
            gpu::make_jit_constant("OUT_FORMAT_IMPLEMENTATION", get_idx_calculation(output_mem.argument.format)),
            gpu::make_jit_constant("CALCULATION_ORDER", get_calculation_order_string(input_mem.argument.format)),
            gpu::make_jit_constant("SRC_TYPE", input_use_half ? std::string("half") : std::string("float")),
            gpu::make_jit_constant("DEST_TYPE", output_use_half ? std::string("half") : std::string("float")),
            gpu::make_jit_constant("SRC_DEST_TYPE_CVT", input_output_type_cvt),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(engine_info.supports_fp16))
        };
        {
            std::stringstream s;
            s << "(uint[]){ ";
            for (uint32_t i = 0; i < input_mem.argument.size.raw.size(); i++)
            {
                s << static_cast<float>(input_mem.argument.size.raw[i]) << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("SIZE", s.str()));
        }
        if (have_subtraction)
        {
            auto& subtract_mem = outer.input_memory(1);

            auto subtract_use_half = memory::traits(subtract_mem.argument.format).type->name == type_id<half_t>()->name;
            int subtract_input_type_cvt = subtract_use_half != input_use_half;

            if (!engine_info.supports_fp16 && subtract_use_half)
                throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_FORMAT_IMPLEMENTATION", get_idx_calculation(subtract_mem.argument.format)));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_TYPE", subtract_use_half ? std::string("half") : std::string("float")));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_SRC_TYPE_CVT", subtract_input_type_cvt));
        }
        else if (!outer.argument.subtract_per_feature.empty())
        {
            std::stringstream s;
            s << "(float[]){ ";
            for (uint32_t i = 0; i < outer.argument.subtract_per_feature.size(); i++)
            {
                s << outer.argument.subtract_per_feature[i] << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("VALUE_TO_SUBTRACT", s.str()));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_TYPE", "float"));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_SRC_TYPE_CVT", input_use_half));
        }
        return mem_consts;
    }

    gpu::kernel_execution_options get_execution_options() const {
        auto& input_mem = outer.input_memory(0);
        auto& input_size_raw = input_mem.argument.size.raw;
        auto dimensions = input_size_raw.size();
        auto order = get_calculation_order(input_mem.argument.format);
        if (dimensions != order.size()) throw std::runtime_error("Reorder number of input dimensions != size of indices order");

        size_t gws_2 = input_size_raw[order[dimensions - 1]];
        size_t gws_1 = input_size_raw[order[dimensions - 2]];
        size_t gws_0 = 1;
        for (size_t i = 0; i < dimensions - 2; i++) {
            gws_0 *= input_size_raw[order[i]];
        }

        return { {gws_0, gws_1, gws_2} };
    }

    static void implementation(const void *ptr) {
        auto me = static_cast<const reorder_gpu*>(ptr);
        auto& outer = me->outer;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);

        if (input_mem.argument.size.raw.size() != output_mem.argument.size.raw.size()) throw std::runtime_error("Reorder input/output number of dimension does not match.");

        if (me->have_subtraction)
        {
            me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>
                (me->_exec_options,
                    input_mem,
                    output_mem,
                    outer.input_memory(1));
        }
        else
        {
            me->_kernel.run<gpu::input_mem, gpu::output_mem>
                (me->_exec_options,
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


    namespace {
        struct attach {
            attach() {
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
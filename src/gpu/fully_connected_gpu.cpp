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

#include "neural_impl.h"
#include "network_impl.h"
#include "engine_impl.h"
#include "implementation_map.h"
#include "kernel.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace neural
{
// Kernel names.
static const std::string kernel_name_xb_xb = "fully_connected_gpu_xb_xb";
static const std::string kernel_name_xb_bx = "fully_connected_gpu_xb_bx";
static const std::string kernel_name_xb_bx_b8 = "fully_connected_gpu_xb_bx_b8";
static const std::string kernel_name_xb_xb_b8_x8 = "fully_connected_gpu_xb_xb_b8_x8";
static const std::string kernel_name_xb_xb_b16 = "fully_connected_gpu_xb_xb_b16";
static const std::string kernel_name_xb_xb_b8_x8_vload = "fully_connected_gpu_xb_xb_b8_x8_vload";
static const std::string kernel_name_yxfn = "fully_connected_gpu_yxfn";
static const std::string kernel_name_xb_xb_block_fp16 = "fully_connected_gpu_xb_xb_block_fp16";
static const std::string kernel_name_bx_bx_from_fyx = "fully_connected_gpu_bx_xb_from_fyx";
static const std::string kernel_name_bx_bx_from_fyxb = "fully_connected_gpu_bx_xb_from_fyxb";

struct fully_connected_gpu : is_an_implementation
{
    fully_connected& _outer;
    std::vector<cldnn::refcounted_obj_ptr<cldnn::network_impl>> reorder;

    struct kernel_data 
    {
        size_t gws0, gws1;
        size_t lws0, lws1;
        std::string kernel_name;
        bool fp16_unit_used;
        union
        {
            struct
            {
                uint32_t unit_byte_size;
                const char* chunk_type;
                uint32_t chunk_byte_size;
                uint32_t units_per_chunk;
                uint32_t bytes_per_sg_read;
                uint32_t units_per_sg_read;
                uint32_t rg_count;
                uint32_t last_rg_size;
            } data_xb_xb_fp16;
        };
    } _kernel_data;
    gpu::kernel _kernel;
       
    fully_connected_gpu(fully_connected& arg)
      : _outer(arg),
        _kernel_data(set_kernel_data(_outer)),
        _kernel(_outer.get_network().get_engine()->get_context(), _outer.id(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data))
    {
        auto const& input_mem = _outer.input_memory(0);

        if (input_mem.argument().format == memory::format::bfyx_f32 &&
            input_mem.argument().size.batch[0] > 1)
        {
            cldnn::topology topology(
                cldnn::input_layout("input", input_mem.get_layout()),
                cldnn::reorder("reorder", "input", cldnn::layout{ cldnn::data_types::f32, input_mem.argument().size.transform(cldnn::format::yxfb, 1) }, "", { cldnn::format::yx,{ 0,0 } })
            );
            reorder.push_back({ _outer.get_network().get_engine()->build_network(api_cast(topology.get()), cldnn::build_options()), false });
        }
    }

    static kernel_data set_kernel_data(const fully_connected& outer)
    {
        const auto& input_mem  = outer.input_memory(0);   // input
        const auto& weight_mem = outer.weights_memory();  // weights
        const auto& output_mem = outer.output_memory();  // output

        kernel_data kd;

        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = output_mem.count();
        kd.gws1 = 1;

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;

        bool batch_multiple_of_8 = input_mem.argument().size.batch[0] % 8 == 0;

        auto input_mem_format = input_mem.argument().format;

        //for bfyx,b>1 there will be reorder from bfyx to yxfb before this fc (created later in ctor), so do calculations for this format rather than original bfyx
        if (input_mem_format == memory::format::bfyx_f32 && input_mem.argument().size.batch[0] > 1)
            input_mem_format = memory::format::yxfb_f32;

        switch (input_mem_format)
        {
        case memory::format::bfyx_f32:
        case memory::format::bx_f32:
        {
            switch (weight_mem.argument().format)
            {
            case memory::format::fyxb_f32:
            case memory::format::xb_f32:
            {
                if (input_mem.argument().size.batch[0] != 1)
                {
                    kd.kernel_name = kernel_name_bx_bx_from_fyxb;
                }
                else
                {
                    kd.kernel_name = kernel_name_bx_bx_from_fyx;
                }
                break;
            }
            default:
                throw std::invalid_argument("Weight memory format is not supported");
            }
            break;
        }
        case memory::format::yxfb_f32:
        case memory::format::xb_f32:
        case memory::format::x_f32:
        {
            switch (weight_mem.argument().format)
            {
            case memory::format::byxf_f32:
            case memory::format::bx_f32:
            {
                if (input_mem.argument().size.batch[0] >= 8)
                {
                    kd.gws0 = output_mem.argument().size.batch[0];
                    kd.gws1 = output_mem.argument().size.spatial[0];
                    kd.lws0 = 8;
                    kd.lws1 = 1;
                    kd.kernel_name = kernel_name_xb_bx_b8;
                }
                else
                {
                    kd.kernel_name = kernel_name_xb_bx;
                }
                break;
            }
            case memory::format::yxfb_f32:
            case memory::format::xb_f32:
            {
                if (batch_multiple_of_8 &&
                    (output_mem.count() / output_mem.argument().size.batch[0]) % 8 == 0)
                {
                    size_t groups_per_batches = get_local_groups_size(output_mem);
                    kd.gws0 = output_mem.count() / (get_neurons_per_work_item(output_mem) * get_batches_per_work_item(output_mem) * groups_per_batches);
                    kd.gws1 = groups_per_batches;
                    kd.lws0 = get_local_work_group_size(output_mem);
                    kd.lws1 = 1;
                    kd.kernel_name = kernel_name_xb_xb_b8_x8_vload;
                }
                else
                {
                    kd.kernel_name = kernel_name_xb_xb;
                }
                break;
            }
            case memory::format::bfyx_f32:
            {
                kd.kernel_name = kernel_name_yxfn;
                break;
            }
            default:
                throw std::invalid_argument("Weight memory format is not supported");
            }
            break;
        }

        case memory::format::yxfb_f16:
        case memory::format::xb_f16:
        case memory::format::x_f16:
        {
            switch (weight_mem.argument().format)
            {
            case memory::format::yxfb_f16:
            case memory::format::xb_f16:
            {
                auto batch_size = output_mem.argument().size.batch[0];
                auto response_size = weight_mem.argument().size.batch[0];
                //bool batch_size_pow_2 = batch_size > 0 && (batch_size & (batch_size - 1)) == 0;

                constexpr uint32_t unit_byte_size = sizeof(cl_half);
                const char* chunk_type = "uint";
                constexpr uint32_t chunk_byte_size = sizeof(cl_uint);
                constexpr uint32_t sub_group_size = 16;
                constexpr uint32_t units_per_chunk = chunk_byte_size / unit_byte_size;
                constexpr uint32_t units_per_sg_read = sub_group_size * units_per_chunk;

                if (/*batch_size_pow_2 ||*/ (batch_size > 0 && batch_size % units_per_sg_read == 0) &&
                    response_size > 0 && response_size * unit_byte_size % 4 == 0) // Temporary: response size must be compatible with block read.
                {
                    // Number of response groups. Each group (except last) writes units_per_sg_read responses
                    // for at least one input data set from batch.
                    auto rg_count = (response_size + units_per_sg_read - 1) / units_per_sg_read;

                    kd.lws0 = sub_group_size;
                    // Rounding up to nearest non-lower multiply of units_per_sg_read.
                    kd.gws0 = rg_count * sub_group_size;
                    kd.lws1 = 1;
                    kd.gws1 = batch_size / units_per_sg_read;

                    kd.kernel_name = kernel_name_xb_xb_block_fp16;

                    kd.data_xb_xb_fp16.unit_byte_size    = unit_byte_size;
                    kd.data_xb_xb_fp16.chunk_type        = chunk_type;
                    kd.data_xb_xb_fp16.chunk_byte_size   = chunk_byte_size;
                    kd.data_xb_xb_fp16.units_per_chunk   = units_per_chunk;
                    kd.data_xb_xb_fp16.bytes_per_sg_read = sub_group_size * chunk_byte_size;
                    kd.data_xb_xb_fp16.units_per_sg_read = units_per_sg_read;
                    kd.data_xb_xb_fp16.rg_count          = rg_count;
                    kd.data_xb_xb_fp16.last_rg_size      = rg_count * units_per_sg_read - response_size;
                    break;
                }

                kd.kernel_name = kernel_name_xb_xb;
                break;
            }

            default:
                throw std::invalid_argument("Weight memory format is not supported");
            }
            break;
        }

        default:
            throw std::invalid_argument("Input memory format is not supported");
        }
        return kd;
    }

    // how many neurons for a single batch will a single work item produce 
    static int get_neurons_per_work_item(const cldnn::memory &output_mem)
    {
        int batch_size = output_mem.argument().size.batch[0];
        auto out_elements_count_per_batch = output_mem.count() / batch_size;
        if (out_elements_count_per_batch % 16 == 0)
            return 2;
        else
            return 1;
    }

    // how many batches will a single work item compute
    static int get_batches_per_work_item(const cldnn::memory &output_mem)
    {
        int batch_size = output_mem.argument().size.batch[0];
        return std::min(batch_size, 32);
    }

    static int get_local_work_group_size(const cldnn::memory &output_mem)
    {
        int batch_size = output_mem.argument().size.batch[0];
        if (batch_size >= 16)
            return 8;
        auto out_elements_count_per_batch = output_mem.count() / batch_size;
        if (out_elements_count_per_batch % 16 == 0)
            return 16;
        else
            return 8;
    }

    static int get_local_groups_size(const cldnn::memory &output_mem)
    {
        int batch_size = output_mem.argument().size.batch[0];
        return std::max(1, batch_size / get_batches_per_work_item(output_mem));
    }

    static gpu::jit_constants get_jit_constants(const fully_connected& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_network().get_engine()->get_context()->get_engine_info();

        const auto& input_mem  = outer.input_memory(0);   // input
        const auto& weight_mem = outer.input_memory(1);   // weights
        const auto& output_mem = outer.output_memory();  // output

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",                input_mem.argument().size),
            gpu::make_jit_constant("OUTPUT",               output_mem.argument().size),
            gpu::make_jit_constant("INPUT_ELEMENTS_COUNT", input_mem.count() / input_mem.argument().size.batch[0]),
            gpu::make_jit_constant("WEIGHTS",              weight_mem.argument().size),
            gpu::make_jit_constant("FP16_SUPPORTED",       static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",       static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",            data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",        data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("RELU",                 outer.argument.with_activation),
            gpu::make_jit_constant("NEGATIVE_SLOPE",       outer.argument.activation_negative_slope),
        };

        if (data.kernel_name == kernel_name_xb_xb_block_fp16)
        {
            mem_consts.add_constant(gpu::make_jit_constant("SUB_GROUP_SIZE",       data.lws0));
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_BATCH", data.gws1));

            mem_consts.add_constant(gpu::make_jit_constant("UNIT_BYTE_SIZE",    data.data_xb_xb_fp16.unit_byte_size));
            mem_consts.add_constant(gpu::make_jit_constant("CHUNK_TYPE",        data.data_xb_xb_fp16.chunk_type));
            mem_consts.add_constant(gpu::make_jit_constant("CHUNK_BYTE_SIZE",   data.data_xb_xb_fp16.chunk_byte_size));
            mem_consts.add_constant(gpu::make_jit_constant("UNITS_PER_CHUNK",   data.data_xb_xb_fp16.units_per_chunk));
            mem_consts.add_constant(gpu::make_jit_constant("BYTES_PER_SG_READ", data.data_xb_xb_fp16.bytes_per_sg_read));
            mem_consts.add_constant(gpu::make_jit_constant("UNITS_PER_SG_READ", data.data_xb_xb_fp16.units_per_sg_read));
            mem_consts.add_constant(gpu::make_jit_constant("RG_COUNT",          data.data_xb_xb_fp16.rg_count));
            mem_consts.add_constant(gpu::make_jit_constant("LAST_RG_SIZE",      data.data_xb_xb_fp16.last_rg_size));
        }

        if (data.kernel_name == kernel_name_xb_xb_b8_x8_vload ||
            data.kernel_name == kernel_name_xb_xb_b16)
        {
            int batch_size = input_mem.argument().size.batch[0];
            const int batches_per_work_item = get_batches_per_work_item(output_mem);
            const int local_work_group_size = get_local_work_group_size(output_mem);

            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE",                         local_work_group_size));
            mem_consts.add_constant(gpu::make_jit_constant("NEURONS_PER_WORK_ITEM",                         get_neurons_per_work_item(output_mem)));                                     // how many neurons for a single batch will a single work item produce
            mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM",                         batches_per_work_item));                                                     // how many batches will a single work item compute
            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max((batch_size / batches_per_work_item) / local_work_group_size, 1))); // how many local work groups we need to compute single element for each batch
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS",        batch_size / batches_per_work_item));                                        // how many work items we need to compute single element for each batch
        }
        return mem_consts;
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        const auto& kd    = _kernel_data;

        const auto& input_mem  = _outer.input_memory(0);   // input
        const auto& weight_mem = _outer.input_memory(1);   // weights
        const auto& bias_mem   = _outer.input_memory(2);   // biases
        const auto& output_mem = _outer.output_memory();  // output

        if (reorder.empty())
            return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
                ({ { kd.gws0, kd.gws1 },{ kd.lws0, kd.lws1 } }, events, input_mem, output_mem, weight_mem, bias_mem);


        auto network = reorder[0];
        network->set_input_data("input", api_cast(input_mem.get()));
        network->execute(events);
        auto output_id = network->get_output_ids()[0];

        auto reorder_output = network->get_primitive(output_id)->output_memory();

        return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
            ({ { kd.gws0, kd.gws1 },{ kd.lws0, kd.lws1 } }, { network->get_primitive_event(output_id) }, reorder_output, output_mem, weight_mem, bias_mem);
    }

    static is_an_implementation *create(fully_connected &arg) {

        auto& input_mem = arg.input_memory(0);
        auto& input_size = input_mem.argument().size;

        // validate arguments
        if (input_size.format == cldnn::format::yxfb ||
			input_size.format == cldnn::format::bfyx)
        {
            // weights
            auto& weight_size = arg.input_memory(1).argument().size;
            if (   input_size.feature.size() != weight_size.feature.size()
                || input_size.batch.size()   != weight_size.batch.size()
                || input_size.feature[0]     != weight_size.feature[0])
                throw std::invalid_argument("Input and weights sizes do not match");
        }
        else {
            // int a,b,c; a*b*c = 1  => a=b=c=1
            if (1 != input_size.feature.size() * input_size.batch.size() * input_size.feature[0])
                throw std::invalid_argument("Wrong input size");
        }

        return new fully_connected_gpu(arg);
    };
};

namespace {
    struct attach {
        attach() {
            auto val_fw = fully_connected_gpu::create;

            implementation_map<fully_connected>::add({
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::xb_f32), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::x_f32), val_fw },

                { std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::xb_f16), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::x_f16), val_fw },

                { std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::bx_f32), val_fw },
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

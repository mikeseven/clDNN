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
#include "kd_selector.h"

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
static const std::string kernel_name_xb_xb_b8_x8_vload = "fully_connected_gpu_xb_xb_b8_x8_vload";
static const std::string kernel_name_xb_bs_xs_xsv8_bsv8_vload = "fully_connected_gpu_xb_bs_xs_xsv8_bsv8_vload";
static const std::string kernel_name_yxfn = "fully_connected_gpu_yxfn";
static const std::string kernel_name_xb_xb_block_fp16 = "fully_connected_gpu_xb_xb_block_fp16";
static const std::string kernel_name_bx_bx_from_fyx = "fully_connected_gpu_bx_xb_from_fyx";
static const std::string kernel_name_bx_bx_from_fyxb = "fully_connected_gpu_bx_xb_from_fyxb";
static const std::string kernel_name_bx_bs_x_bsv16_b1 = "fully_connected_gpu_bx_bs_x_bsv16_b1"; // Fully connected algorithm for batch 1 that supports bx or bfyx input (weights in special format: bs_x_bsv16).

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::architectures>
{
    static constexpr neural::gpu::engine_info_internal::architectures value = neural::gpu::engine_info_internal::architectures::GEN_UNKNOWN;
};

template <>
struct kd_default_value_selector<gpu::engine_info_internal::configurations>
{
    static constexpr gpu::engine_info_internal::configurations value = gpu::engine_info_internal::configurations::GT_UNKNOWN;
};

// how many batches will a single work item compute
static int get_batches_per_work_item(const cldnn::memory &output_mem)
{
    int batch_size = output_mem.argument().size.batch[0];
    return std::min(batch_size, 32);
}

static int get_local_groups_size(const cldnn::memory &output_mem)
{
    int batch_size = output_mem.argument().size.batch[0];
    return std::max(1, batch_size / get_batches_per_work_item(output_mem));
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

struct fully_connected_gpu : is_an_implementation
{
    fully_connected& _outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data 
    {
        std::vector<cldnn::refcounted_obj_ptr<cldnn::network_impl>> reorder;
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
            struct
            {
                uint32_t unit_byte_size;
                const char* chunk_type;
                uint32_t chunk_byte_size;
                uint32_t units_per_chunk;
                uint32_t bytes_per_sg_read;
                uint32_t units_per_sg_read;
                uint32_t responses_per_sg_exec;
                uint32_t in_chunk_prefetch_size;
                uint32_t filter_chunk_prefetch_size;
            } data_bx_bs_x_bsv16;
        };
    } _kernel_data;
    gpu::kernel _kernel;

    typedef kd_selector_t<kernel_data, fully_connected, neural::memory::format::type, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, gpu::engine_info_internal::configurations> ks_type;
    static ks_type ks;

    fully_connected_gpu(fully_connected& arg)
      : _outer(arg)
        , _engine_info(arg.get_network().get_engine()->get_context()->get_engine_info())
        , _kernel_data(ks.get_kernel(_outer, _outer.input_memory(0).argument().format, _outer.weights_memory().argument().format, _outer.input_memory(0).argument().size.batch[0], _engine_info.architecture, _engine_info.configuration))
        ,_kernel(_outer.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(_outer, _kernel_data), _outer.id())
    {
    }

    static kernel_data set_kernel_data(const fully_connected& outer)
    {
        const auto& input_mem  = outer.input_memory(0);   // input
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

        return kd;
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

        if (data.kernel_name == kernel_name_bx_bs_x_bsv16_b1)
        {
            mem_consts.add_constant(gpu::make_jit_constant("SUB_GROUP_SIZE",       data.lws0));
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_BATCH", data.gws1));

            mem_consts.add_constant(gpu::make_jit_constant("UNIT_BYTE_SIZE",             data.data_bx_bs_x_bsv16.unit_byte_size));
            mem_consts.add_constant(gpu::make_jit_constant("CHUNK_TYPE",                 data.data_bx_bs_x_bsv16.chunk_type));
            mem_consts.add_constant(gpu::make_jit_constant("CHUNK_BYTE_SIZE",            data.data_bx_bs_x_bsv16.chunk_byte_size));
            mem_consts.add_constant(gpu::make_jit_constant("UNITS_PER_CHUNK",            data.data_bx_bs_x_bsv16.units_per_chunk));
            mem_consts.add_constant(gpu::make_jit_constant("BYTES_PER_SG_READ",          data.data_bx_bs_x_bsv16.bytes_per_sg_read));
            mem_consts.add_constant(gpu::make_jit_constant("UNITS_PER_SG_READ",          data.data_bx_bs_x_bsv16.units_per_sg_read));
            mem_consts.add_constant(gpu::make_jit_constant("RESPONSES_PER_SG_EXEC",      data.data_bx_bs_x_bsv16.responses_per_sg_exec));
            mem_consts.add_constant(gpu::make_jit_constant("IN_CHUNK_PREFETCH_SIZE",     data.data_bx_bs_x_bsv16.in_chunk_prefetch_size));
            mem_consts.add_constant(gpu::make_jit_constant("FILTER_CHUNK_PREFETCH_SIZE", data.data_bx_bs_x_bsv16.filter_chunk_prefetch_size));
        }

        if (data.kernel_name == kernel_name_xb_xb_b8_x8_vload ||
            data.kernel_name == kernel_name_xb_bs_xs_xsv8_bsv8_vload)
        {
            const int batches_per_work_item = get_batches_per_work_item(output_mem);

            mem_consts.add_constant(gpu::make_jit_constant("NEURONS_PER_WORK_ITEM", get_neurons_per_work_item(output_mem))); // how many neurons for a single batch will a single work item produce
            mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", batches_per_work_item));                 // how many batches will a single work item compute
            mem_consts.add_constant(gpu::make_jit_constant("OUTPUT_ELEMENTS_COUNT", output_mem.count() / output_mem.argument().size.batch[0]));
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

        if (kd.reorder.empty())
            return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
                ({ { kd.gws0, kd.gws1 },{ kd.lws0, kd.lws1 } }, events, input_mem, output_mem, weight_mem, bias_mem);


        auto network = kd.reorder[0];
        network->set_input_data("input", api_cast(input_mem.get()));
        network->execute(events);
        auto output_id = network->get_output_ids()[0];

        auto reorder_output = network->get_primitive(output_id)->output_memory();

        return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
            ({ { kd.gws0, kd.gws1 },{ kd.lws0, kd.lws1 } }, { network->get_primitive_event(output_id) }, reorder_output, output_mem, weight_mem, bias_mem);
    }

    static is_an_implementation *create(fully_connected &arg) 
    {
        auto& input_mem = arg.input_memory(0);
        auto& input_size = input_mem.argument().size;
        auto& weights_mem = arg.weights_memory();
        auto& weights_size = weights_mem.argument().size;

        // validate arguments
        if (input_size.format == cldnn::format::yxfb ||
            input_size.format == cldnn::format::bfyx)
        {
            if (weights_mem.argument().format != memory::format::bs_xs_xsv8_bsv8_f32 &&
                weights_mem.argument().format != memory::format::bs_x_bsv16_f32 &&
                weights_mem.argument().format != memory::format::bs_x_bsv16_f16)
            {
                // weights
                if (input_size.feature.size() != weights_size.feature.size()
                    || input_size.batch.size() != weights_size.batch.size()
                    || input_size.feature[0] != weights_size.feature[0])
                    throw std::invalid_argument("Input and weights sizes do not match");
            }
        }
        else {
            // int a,b,c; a*b*c = 1  => a=b=c=1
            if (1 != input_size.feature.size() * input_size.batch.size() * input_size.feature[0])
                throw std::invalid_argument("Wrong input size");
        }

        return new fully_connected_gpu(arg);
    };
};

fully_connected_gpu::kernel_data default_yxfb_f32_bfyx_f32(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);
    kd.kernel_name = kernel_name_yxfn;
    return kd;
}

fully_connected_gpu::kernel_data default_yxfb_f32(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);
    bool batch_multiple_of_8 = arg.input_memory(0).argument().size.batch[0] % 8 == 0;

    if (batch_multiple_of_8 &&
        (arg.output_memory().count() / arg.output_memory().argument().size.batch[0]) % 8 == 0)
    {
        size_t groups_per_batches = get_local_groups_size(arg.output_memory());
        kd.gws0 = arg.output_memory().count() / (get_neurons_per_work_item(arg.output_memory()) * get_batches_per_work_item(arg.output_memory()) * groups_per_batches);
        kd.gws1 = groups_per_batches;
        kd.lws0 = 8;
        kd.lws1 = 1;
        kd.kernel_name = kernel_name_xb_xb_b8_x8_vload;
    }
    else
    {
        kd.kernel_name = kernel_name_xb_xb;
    }

    return kd;
}

fully_connected_gpu::kernel_data default_xb_f32_bx_f32(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);

    auto input_mem = arg.input_memory(0);
    auto output_mem = arg.output_memory();

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
    return kd;
}

fully_connected_gpu::kernel_data default_bfyx_f32(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);
    if (arg.input_memory(0).argument().size.batch[0] != 1)
    {
        auto input_mem = arg.input_memory(0);
        cldnn::topology topology(
            cldnn::input_layout("input", input_mem.get_layout()),
            cldnn::reorder("reorder", "input", cldnn::layout{ input_mem.get_layout().data_type, input_mem.argument().size.transform(cldnn::format::yxfb, 1) }, "", { cldnn::format::yx,{ 0,0 } })
        );
        kd.reorder.push_back({ arg.get_network().get_engine()->build_network(api_cast(topology.get()), cldnn::build_options()), false });
        kd.kernel_name = kernel_name_xb_xb;
    }
    else
    {
        kd.kernel_name = kernel_name_bx_bx_from_fyx;
    }
    return kd;
}

fully_connected_gpu::kernel_data default_yxfb_f32_bs_xs_xsv8_bsv8_f32(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);
    bool batch_multiple_of_8 = arg.input_memory(0).argument().size.batch[0] % 8 == 0;

    if (batch_multiple_of_8)
    {
        size_t groups_per_batches = get_local_groups_size(arg.output_memory());
        kd.gws0 = cldnn::align_to(arg.output_memory().count() / (get_neurons_per_work_item(arg.output_memory()) * get_batches_per_work_item(arg.output_memory()) * groups_per_batches), 8);
        kd.gws1 = groups_per_batches;
        kd.lws0 = 8;
        kd.lws1 = 1;
        kd.kernel_name = kernel_name_xb_bs_xs_xsv8_bsv8_vload;
    }
    else
    {
        // TODO: implement this case
        throw std::runtime_error("Not implemented bs_xs_bsv8_xsv8_f32 for batch not multiple of 8");
    }

    return kd;
}

fully_connected_gpu::kernel_data default_bfyx_f32_bs_xs_xsv8_bsv8_f32(const fully_connected& arg)
{
    auto input_mem = arg.input_memory(0);
    auto input_size = input_mem.get_layout().size;
    if (input_size.batch[0] < 8)
    {
        // TODO: implement this case
        throw std::runtime_error("default_bfyx_f32_bs_xs_xsv8_bsv8_f32 with batch < 8 not implemented!");
    }

    auto expected_mem_size = cldnn::tensor(cldnn::format::bs_xs_xsv8_bsv8,
    {
        input_size.batch[0], input_size.feature[0] * input_size.spatial[0] * input_size.spatial[1]
    });
    cldnn::layout expected_mem_layout(cldnn::data_types::f32, expected_mem_size);
    cldnn::topology topology(
        cldnn::input_layout("input", input_mem.get_layout()),
        cldnn::reorder("reorder", "input", expected_mem_layout)
    );
    
    fully_connected_gpu::kernel_data kd = default_yxfb_f32_bs_xs_xsv8_bsv8_f32(arg);
    kd.reorder.push_back({ arg.get_network().get_engine()->build_network(api_cast(topology.get()), cldnn::build_options()), false });
    return kd;
}

fully_connected_gpu::kernel_data default_xb_f32_bs_xs_xsv8_bsv8_f32(const fully_connected& arg)
{
    auto input_mem = arg.input_memory(0);
    auto input_size = input_mem.get_layout().size;
    if (input_size.batch[0] < 8)
    {
        // TODO: implement this case
        throw std::runtime_error("default_xb_f32_bs_xs_xsv8_bsv8_f32 with batch < 8 not implemented!");
    }
    cldnn::topology topology(
        cldnn::input_layout("input", input_mem.get_layout()),
        cldnn::reorder("reorder", "input", cldnn::layout{ cldnn::data_types::f32, input_mem.argument().size.transform(cldnn::format::bs_xs_xsv8_bsv8, 1) }, "")
    );

    fully_connected_gpu::kernel_data kd = default_yxfb_f32_bs_xs_xsv8_bsv8_f32(arg);
    kd.reorder.push_back({ arg.get_network().get_engine()->build_network(api_cast(topology.get()), cldnn::build_options()), false });
    return kd;
}

fully_connected_gpu::kernel_data default_bfyx_f32_fyxb_f32_b1(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);
    kd.kernel_name = kernel_name_bx_bx_from_fyx;
    return kd;
}

fully_connected_gpu::kernel_data default_yxfb_fp16(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);

    auto batch_size = arg.output_memory().argument().size.batch[0];
    auto response_size = arg.weights_memory().argument().size.batch[0];
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
        auto rg_count = cldnn::ceil_div(response_size, units_per_sg_read);

        kd.lws0 = sub_group_size;
        // Number of work items needed to process all response groups.
        kd.gws0 = rg_count * sub_group_size;
        kd.lws1 = 1;
        kd.gws1 = batch_size / units_per_sg_read;

        kd.kernel_name = kernel_name_xb_xb_block_fp16;

        kd.data_xb_xb_fp16.unit_byte_size = unit_byte_size;
        kd.data_xb_xb_fp16.chunk_type = chunk_type;
        kd.data_xb_xb_fp16.chunk_byte_size = chunk_byte_size;
        kd.data_xb_xb_fp16.units_per_chunk = units_per_chunk;
        kd.data_xb_xb_fp16.bytes_per_sg_read = sub_group_size * chunk_byte_size;
        kd.data_xb_xb_fp16.units_per_sg_read = units_per_sg_read;
        kd.data_xb_xb_fp16.rg_count = rg_count;
        kd.data_xb_xb_fp16.last_rg_size = response_size % units_per_sg_read;
    }
    else
    {
        kd.kernel_name = kernel_name_xb_xb;
    }

    return kd;
}

fully_connected_gpu::kernel_data default_bfyx_fp16(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);
    if (arg.input_memory(0).argument().size.batch[0] != 1)
    {
        kd.kernel_name = kernel_name_bx_bx_from_fyxb;
    }
    else
    {
        kd.kernel_name = kernel_name_bx_bx_from_fyx;
    }
    return kd;
}

fully_connected_gpu::kernel_data default_bfyx_bs_x_bsv16_b1(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);

    auto response_size = arg.weights_memory().argument().size.batch[0];

    // Properties of chunk and unit.
    const uint32_t unit_byte_size = kd.fp16_unit_used ? sizeof(cl_half) : sizeof(float);
    const char* chunk_type = "uint";
    constexpr uint32_t chunk_byte_size = sizeof(cl_uint);
    constexpr uint32_t sub_group_size = 16;
    const uint32_t units_per_chunk = chunk_byte_size / unit_byte_size;
    const uint32_t units_per_sg_read = sub_group_size * units_per_chunk;
    // Properties of primitive responses.
    constexpr uint32_t responses_per_sg_exec = 16; // Must match batch slice size of weights format (bs_x_bsv16).

    // Number of response groups. Each group (except last) writes responses_per_sg_exec responses
    // for at least one input data set from batch.
    auto rg_count = cldnn::ceil_div(response_size, responses_per_sg_exec);

    kd.lws0 = sub_group_size;
    // Number of work items needed to process all response groups.
    kd.gws0 = rg_count * sub_group_size;
    kd.lws1 = 1;
    kd.gws1 = 1;

    kd.kernel_name = kernel_name_bx_bs_x_bsv16_b1;

    kd.data_bx_bs_x_bsv16.unit_byte_size = unit_byte_size;
    kd.data_bx_bs_x_bsv16.chunk_type = chunk_type;
    kd.data_bx_bs_x_bsv16.chunk_byte_size = chunk_byte_size;
    kd.data_bx_bs_x_bsv16.units_per_chunk = units_per_chunk;
    kd.data_bx_bs_x_bsv16.bytes_per_sg_read = sub_group_size * chunk_byte_size;
    kd.data_bx_bs_x_bsv16.units_per_sg_read = units_per_sg_read;
    kd.data_bx_bs_x_bsv16.responses_per_sg_exec = responses_per_sg_exec;
    kd.data_bx_bs_x_bsv16.in_chunk_prefetch_size = 2;
    kd.data_bx_bs_x_bsv16.filter_chunk_prefetch_size = responses_per_sg_exec;

    return kd;
}

fully_connected_gpu::kernel_data default_bfyx_f16_yxfb_f16(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);
    kd.kernel_name = kernel_name_bx_bx_from_fyxb;

    auto input_mem = arg.input_memory(0);
    cldnn::topology topology(
        cldnn::input_layout("input", input_mem.get_layout()),
        cldnn::reorder("reorder", "input", cldnn::layout{ input_mem.get_layout().data_type, input_mem.argument().size.transform(cldnn::format::yxfb, 1) }, "", { cldnn::format::yx,{ 0,0 } })
    );
    kd.reorder.push_back({ arg.get_network().get_engine()->build_network(api_cast(topology.get()), cldnn::build_options()), false });
    return kd;
}

fully_connected_gpu::kernel_data default_bfyx_f16_fyxb_f16_b1(const fully_connected& arg)
{
    fully_connected_gpu::kernel_data kd = fully_connected_gpu::set_kernel_data(arg);
    kd.kernel_name = kernel_name_bx_bx_from_fyx;
    return kd;
}

fully_connected_gpu::ks_type fully_connected_gpu::ks = {
    { std::make_tuple(memory::format::yxfb_f32, memory::format::bfyx_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb_f32_bfyx_f32 },

    { std::make_tuple(memory::format::yxfb_f32, memory::format::yxfb_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb_f32 },
    { std::make_tuple(memory::format::xb_f32, memory::format::xb_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb_f32 },
    { std::make_tuple(memory::format::x_f32, memory::format::xb_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb_f32 },

    { std::make_tuple(memory::format::xb_f32, memory::format::bx_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_xb_f32_bx_f32 },
    { std::make_tuple(memory::format::x_f32, memory::format::bx_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_xb_f32_bx_f32 },

    { std::make_tuple(memory::format::bfyx_f32, memory::format::yxfb_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_f32 },

    { std::make_tuple(memory::format::bfyx_f32, memory::format::fyxb_f32, 1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_f32_fyxb_f32_b1 },
    { std::make_tuple(memory::format::bx_f32, memory::format::xb_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_f32 },

    { std::make_tuple(memory::format::bfyx_f32, memory::format::bs_xs_xsv8_bsv8_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_f32_bs_xs_xsv8_bsv8_f32 },
    { std::make_tuple(memory::format::yxfb_f32, memory::format::bs_xs_xsv8_bsv8_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb_f32_bs_xs_xsv8_bsv8_f32 },
    { std::make_tuple(memory::format::xb_f32, memory::format::bs_xs_xsv8_bsv8_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_xb_f32_bs_xs_xsv8_bsv8_f32 },

    { std::make_tuple(memory::format::bfyx_f32, memory::format::bs_x_bsv16_f32, 1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_bs_x_bsv16_b1 },
    { std::make_tuple(memory::format::bx_f32, memory::format::bs_x_bsv16_f32, 1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_bs_x_bsv16_b1 },
    
    { std::make_tuple(memory::format::yxfb_f16, memory::format::yxfb_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb_fp16 },
    { std::make_tuple(memory::format::xb_f16, memory::format::xb_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb_fp16 },
    { std::make_tuple(memory::format::x_f16, memory::format::xb_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxfb_fp16 },

    { std::make_tuple(memory::format::bfyx_f16, memory::format::yxfb_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_f16_yxfb_f16 },
    { std::make_tuple(memory::format::bfyx_f16, memory::format::fyxb_f16, 1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_f16_fyxb_f16_b1 },
    { std::make_tuple(memory::format::bx_f16, memory::format::xb_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_fp16 },

    { std::make_tuple(memory::format::bfyx_f16, memory::format::bs_x_bsv16_f16, 1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_bs_x_bsv16_b1 },
    { std::make_tuple(memory::format::bx_f16, memory::format::bs_x_bsv16_f16, 1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_bs_x_bsv16_b1 },
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

                { std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f16), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, memory::format::bx_f16), val_fw },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}

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

#include "neural_impl.h"
#include "implementation_map.h"
#include "kernel.h"
#include "network_impl.h"
#include "engine_impl.h"
#include "boost/functional/hash.hpp"
#include "kd_selector.h"

#include <unordered_map>
#include <initializer_list>
#include "kd_selector.h"

namespace neural 
{

static const std::string kernel_name_yxfb_oiyx = "convolution_gpu_yxfb_oiyx";
static const std::string kernel_name_yxfb_yxio = "convolution_gpu_yxfb_yxio";
static const std::string kernel_name_yxfb_yxio_b1_block = "convolution_gpu_yxfb_yxio_b1_block";
static const std::string kernel_name_yxfb_yxio_b1_block_multiple_x = "convolution_gpu_yxfb_yxio_b1_block_multiple_x";
static const std::string kernel_name_yxfb_yxio_b8 = "convolution_gpu_yxfb_yxio_b8";
static const std::string kernel_name_yxfb_yxio_b16 = "convolution_gpu_yxfb_yxio_b16";
static const std::string kernel_name_yxfb_yxio_b16_fp16 = "convolution_gpu_yxfb_yxio_b16_fp16";
static const std::string kernel_name_yxfb_yxio_fp16 = "convolution_gpu_yxfb_yxio_fp16";
static const std::string kernel_name_bfyx_os_iyx_osv16_b1_f32 = "convolution_gpu_bfyx_os_iyx_osv16_b1_f32";
static const std::string kernel_name_bfyx_os_iyx_osv16_b1_f32_stride1 = "convolution_gpu_bfyx_os_iyx_osv16_b1_f32_stride1";
static const std::string kernel_name_bfyx_os_iyx_osv16_b1_f32_stride2 = "convolution_gpu_bfyx_os_iyx_osv16_b1_f32_stride2";
static const std::string kernel_name_bfyx_os_iyx_osv16_b1_f32_kernel_size_1 = "convolution_gpu_bfyx_os_iyx_osv16_b1_f32_kernel_size_1";

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

struct convolution_gpu : is_an_implementation {
    convolution &outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        size_t ofm_per_work_item; // how many output feature maps a single work item compute
        size_t batches_per_work_item; // how many batches will a single work item compute
        size_t block_width, block_height; // used for kernels processing blocks
        std::string kernel_name;
    } _kernel_data;

    gpu::kernel _kernel;

    static kernel_data set_default(const convolution& arg)
    {
        auto& output_mem = arg.output_memory();
        auto split = arg.argument.split;
        auto batch_size = output_mem.argument().size.batch[0];

        kernel_data kd;
        kd.gws0 = (output_mem.argument().size.feature[0] * batch_size) / split;
        kd.lws0 = std::min(kd.gws0, static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0)
        {
            kd.lws0--;
        }
        kd.gws1 = output_mem.argument().size.spatial[0];
        kd.gws2 = output_mem.argument().size.spatial[1];
        kd.lws1 = 1;
        kd.lws2 = 1;
        kd.ofm_per_work_item = 1;
        kd.batches_per_work_item = 1;
        kd.block_width = 1;
        kd.block_height = 1;
        return kd;
    }

    typedef kd_selector_t<kernel_data, convolution, neural::memory::format::type, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, gpu::engine_info_internal::configurations> ks_type;
    static ks_type ks;

    convolution_gpu(convolution &arg)
        : outer(arg)
        , _engine_info(arg.get_network().get_engine()->get_context()->get_engine_info())
        , _kernel_data(ks.get_kernel(outer, outer.input_memory(0).argument().format, outer.weights_memory(0).argument().format, outer.input_memory(0).argument().size.batch[0], _engine_info.architecture, _engine_info.configuration))
        , _kernel(arg.get_network().get_engine()->get_context(), outer.id(), _kernel_data.kernel_name, get_jit_constants())
    {}

    gpu::jit_constants get_jit_constants() const {

        auto& input_mem = outer.input_memory(0);
        auto input_offset = outer.desc()->input_offset().transform(input_mem.get_layout().size.format, 0);
        auto& output_mem = outer.output_memory();
        auto output_offset = outer.desc()->output_offset().transform(output_mem.get_layout().size.format, 0);
        auto& output_size = outer.output_memory().argument().size;
        auto& filter_mem = outer.weights_memory(0);
        auto split = outer.argument.split;
        auto negative_slope = outer.argument.activation_negative_slope;

        const int batch_size = output_mem.argument().size.batch[0];

        auto input_size = outer.input().at(0)->non_padded_output_layout().size;
        cldnn::tensor stride(cldnn::format::yx, { std::min(outer.argument.stride.spatial[0], input_size.spatial[0]),
            std::min(outer.argument.stride.spatial[1], input_size.spatial[1]) });
        cldnn::padding input_padding(cldnn::format::yx, { filter_mem.argument().size.spatial[0] - 1, filter_mem.argument().size.spatial[1] - 1 });

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT", input_size),
            gpu::make_jit_constant("OUTPUT", outer.non_padded_output_layout().size),
            gpu::make_jit_constant("STRIDE", stride),
            gpu::make_jit_constant("INPUT_OFFSET", input_offset),
            gpu::make_jit_constant("OUTPUT_OFFSET", output_offset),
            gpu::make_jit_constant("OUTPUT_LIMIT", output_size),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(_engine_info.supports_fp16)),
            gpu::make_jit_constant("INPUT_PADDING", input_padding.size()),
            gpu::make_jit_constant("OUTPUT_PADDING", outer.argument.output_padding().size())
        };

        if (outer.argument.with_activation)
        {
            mem_consts.add_constant(gpu::make_jit_constant("NEGATIVE_SLOPE", negative_slope));
            mem_consts.add_constant(gpu::make_jit_constant("RELU", ""));
        }

        mem_consts.add_constant(gpu::make_jit_constant("FILTER", filter_mem.argument().size));
        mem_consts.add_constant(gpu::make_jit_constant("FILTER_ARRAY_NUM", split));

        mem_consts.add_constant(gpu::make_jit_constant("FILTER_OUTPUT_FEATURE_NUM", "FILTER_FEATURE_NUM_0"));
        mem_consts.add_constant(gpu::make_jit_constant("FILTER_INPUT_FEATURE_NUM", "FILTER_FEATURE_NUM_1"));

        if (filter_mem.argument().format == memory::format::yxio_f32 ||
            filter_mem.argument().format == memory::format::yxoi_f32 ||
            filter_mem.argument().format == memory::format::yxio_f16)
        {
            const int local_work_group_size = static_cast<int>(_kernel_data.lws0);
            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE", local_work_group_size));
            mem_consts.add_constant(gpu::make_jit_constant("OFM_PER_WORK_ITEM", _kernel_data.ofm_per_work_item)); // how many output feature maps for a single batch will a single work item produce
            mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", _kernel_data.batches_per_work_item)); // how many batches will a single work item compute
            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max((batch_size / static_cast<int>(_kernel_data.batches_per_work_item)) / local_work_group_size, 1))); // how many local work groups we need to compute single element for each batch
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS", batch_size / _kernel_data.batches_per_work_item)); // how many work items we need to compute single element for each batch

            if (_kernel_data.kernel_name == kernel_name_yxfb_yxio_b16_fp16)
            {
                if (batch_size >= 64)
                    mem_consts.add_constant(gpu::make_jit_constant("USE_BLOCK_READ_2", ""));
                else if (batch_size >= 32)
                    mem_consts.add_constant(gpu::make_jit_constant("USE_BLOCK_READ_1", ""));
            }
            // A LITTLE HACK, for convolutions with low number of input features don't use block reads, and it will speed up by 25%
            // TODO - investigate why is this happening
            else if (input_mem.argument().size.feature[0] > 4)
            {
                mem_consts.add_constant(gpu::make_jit_constant("USE_BLOCK_READ_2", ""));
            }
        }
        if (_kernel_data.kernel_name == kernel_name_yxfb_yxio_b1_block_multiple_x)
        {
            mem_consts.add_constant(gpu::make_jit_constant("USE_VECTOR", _kernel_data.ofm_per_work_item));
            if (_kernel_data.ofm_per_work_item == 8)
                mem_consts.add_constant(gpu::make_jit_constant("X_PER_WORK_ITEM", 2));
            else if (_kernel_data.ofm_per_work_item == 4)
                mem_consts.add_constant(gpu::make_jit_constant("X_PER_WORK_ITEM", 4));
            else
                mem_consts.add_constant(gpu::make_jit_constant("X_PER_WORK_ITEM", 8));
        }

        if (_kernel_data.kernel_name == kernel_name_bfyx_os_iyx_osv16_b1_f32_stride1)
        {
            mem_consts.add_constant(gpu::make_jit_constant("OUT_BLOCK_WIDTH", _kernel_data.block_width));
            mem_consts.add_constant(gpu::make_jit_constant("OUT_BLOCK_HEIGHT", _kernel_data.block_height));
        }

        return mem_consts;
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        auto me = this;

        auto split = outer.argument.split;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory();
        auto& filter_mem = outer.weights_memory(0);

        if (outer.desc()->padding_type() != cldnn::padding::zero)
            throw std::invalid_argument("Unknown padding mode in convolution.");

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        if (input_mem.get_layout().data_type != output_mem.get_layout().data_type)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of output.");
        if (input_mem.get_layout().data_type != filter_mem.get_layout().data_type)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of filter.");

        auto& kd = me->_kernel_data;

        std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>> tmp_events(events);

        // execute kernels
        for (decltype(split) i = 0; i < split; i++) {
            assert(kd.gws0 % kd.lws0 == 0);
            auto event = me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem, uint32_t>
                ({ { kd.gws0, kd.gws1, kd.gws2 },{ kd.lws0, kd.lws1, kd.lws2 } },
                    tmp_events,
                    input_mem,
                    output_mem,
                    outer.weights_memory(i), //filters
                    outer.bias_memory(i), //biases
                    i);
            tmp_events.clear();
            tmp_events.emplace_back(event);
        }
        return tmp_events.at(0);
    }

    static is_an_implementation *create(convolution &arg) {
        auto filter_arg = arg.weights_memory(0).argument(); //convolution filter

        assert(arg.output_memory().argument().size.feature[0] / arg.argument.split == filter_arg.size.feature[0]); // memory::format oixy
        
        switch (filter_arg.format)
        {
        // FP32 (float)
        case memory::format::oiyx_f32:
        case memory::format::yxio_f32:
        case memory::format::os_iyx_osv16_f32:
        // FP16 (half)
        case memory::format::oiyx_f16:
        case memory::format::yxio_f16:
            break;
        default:
            throw std::runtime_error("Convolution weights format unsupported");
        }

        return new convolution_gpu(arg);
    }
};

convolution_gpu::kernel_data default_oiyx_f32(const convolution& arg)
{
    convolution_gpu::kernel_data kd = convolution_gpu::set_default(arg);
    kd.kernel_name = kernel_name_yxfb_oiyx;
    return kd;
}

convolution_gpu::kernel_data default_yxio_f32(const convolution& arg)
{
    convolution_gpu::kernel_data kd = convolution_gpu::set_default(arg);
    kd.kernel_name = kernel_name_yxfb_yxio;
    return kd;
}

convolution_gpu::kernel_data default_yxio_f32_b1(const convolution& arg)
{
    auto& filter_mem = arg.weights_memory(0);
    auto& output_mem = arg.output_memory();
    auto split = arg.argument.split;
    auto batch_size = output_mem.argument().size.batch[0];

    convolution_gpu::kernel_data kd = convolution_gpu::set_default(arg);
    kd.lws0 = 16;
    if (filter_mem.argument().size.feature[0] * batch_size % kd.lws0 != 0)
    {
        kd = default_yxio_f32(arg);
    }
    else
    {
        int output_feature_count = filter_mem.argument().size.feature[0];
        // We cannot return 8 because we are processing 4 spatial coordinates for batch1,
        // and if we use more than 4 ofm_per_work_item we downgrade simd16 to simd8 which would break this algorithm.
        // NOTE: We could return 8 but then we must process only 2 coordinates, which is slower than processing 4 coordinates using blockread4
        // TODO: experiment with SIMD8 version of algorithm and check if it could be faster
        /*if (output_feature_count % (lws * 8) == 0)
        {
        kd.ofm_per_work_item = 8;
        kd.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(kd.gws1) / 2.0f));
        }
        else*/ if (output_feature_count % (kd.lws0 * 4) == 0)
        {
            kd.ofm_per_work_item = 4;
            // We compute multiple spatial coordinates "x" in a single workitem that's why we must divide
            kd.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(kd.gws1) / 4.0f));
        }
        else if (output_feature_count % (kd.lws0 * 2) == 0)
        {
            kd.ofm_per_work_item = 2;
            kd.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(kd.gws1) / 8.0f));
        }
        else
        {
            kd.ofm_per_work_item = 1;
            kd.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(kd.gws1) / 8.0f));
        }
        kd.kernel_name = kernel_name_yxfb_yxio_b1_block_multiple_x;

        kd.gws0 = (output_mem.argument().size.feature[0] * batch_size / (kd.ofm_per_work_item * kd.batches_per_work_item)) / split;
    }
    return kd;
}

convolution_gpu::kernel_data default_yxio_f32_b8(const convolution& arg)
{
    auto& filter_mem = arg.weights_memory(0);
    auto& output_mem = arg.output_memory();
    auto split = arg.argument.split;
    auto batch_size = output_mem.argument().size.batch[0];

    convolution_gpu::kernel_data kd = convolution_gpu::set_default(arg);
    kd.lws0 = batch_size == 8 ? 8 : 16;
    if (filter_mem.argument().size.feature[0] * batch_size % kd.lws0 != 0)
    {
        kd = default_yxio_f32(arg);
    }
    else
    {
        if (((filter_mem.argument().size.feature[0] * batch_size) / 16) % kd.lws0)
        {
            kd.ofm_per_work_item = 8;
        }
        else
        {
            kd.ofm_per_work_item = 16;
        }
        kd.kernel_name = kernel_name_yxfb_yxio_b8;
    
        kd.gws0 = (output_mem.argument().size.feature[0] * batch_size / (kd.ofm_per_work_item * kd.batches_per_work_item)) / split;
    }
    return kd;
}

convolution_gpu::kernel_data default_yxio_f32_b32(const convolution& arg)
{
    auto& filter_mem = arg.weights_memory(0);
    auto& output_mem = arg.output_memory();
    auto split = arg.argument.split;
    auto batch_size = output_mem.argument().size.batch[0];

    convolution_gpu::kernel_data kd = convolution_gpu::set_default(arg);
    kd.lws0 = 16;
    if (filter_mem.argument().size.feature[0] * batch_size % kd.lws0 != 0)
    {
        kd = default_yxio_f32(arg);
    }
    else
    {
        kd.ofm_per_work_item = 8;
        kd.batches_per_work_item = 2;
        kd.kernel_name = kernel_name_yxfb_yxio_b16;

        kd.gws0 = (output_mem.argument().size.feature[0] * batch_size / (kd.ofm_per_work_item * kd.batches_per_work_item)) / split;
    }
    return kd;
}

convolution_gpu::kernel_data default_yxio_f16(const convolution& arg)
{
    convolution_gpu::kernel_data kd = convolution_gpu::set_default(arg);
    kd.kernel_name = kernel_name_yxfb_yxio_fp16;
    return kd;
}

convolution_gpu::kernel_data default_yxio_f16_b16(const convolution& arg)
{
    auto& filter_mem = arg.weights_memory(0);
    auto& output_mem = arg.output_memory();
    auto batch_size = output_mem.argument().size.batch[0];
    auto filter_ofm_num = filter_mem.argument().size.feature[0];

    const uint32_t min_ofm_per_wi = 16;
    const uint32_t min_batches_per_wi = 1;
    const uint32_t min_lws = 16;

    convolution_gpu::kernel_data kd = convolution_gpu::set_default(arg);
    // Number of output features is positive and dividable by minimum number of output features processed inside work item.
    if (filter_ofm_num > 0 && filter_ofm_num % min_ofm_per_wi == 0 &&
        // Batch size is positive and dividable by minimum number of batches processed when smallest local work size is used.
        batch_size > 0 && batch_size % (min_batches_per_wi * min_lws) == 0)
    {
        kd.ofm_per_work_item = min_ofm_per_wi;
        if (batch_size % (4 * min_batches_per_wi * min_lws) == 0)
        {
            kd.batches_per_work_item = 4 * min_batches_per_wi; // USE_BLOCK_READ_2 + as_half4
        }
        else if (batch_size % (2 * min_batches_per_wi * min_lws) == 0)
        {
            kd.batches_per_work_item = 2 * min_batches_per_wi; // USE_BLOCK_READ_1 + as_half2
        }
        else
        {
            kd.batches_per_work_item = min_batches_per_wi;
        }
        // Assume that number of features in output correctly based on split and on number of output features in filter.
        assert(output_mem.argument().size.feature[0] == filter_ofm_num * arg.argument.split);
        kd.gws0 = filter_ofm_num * batch_size / (kd.ofm_per_work_item * kd.batches_per_work_item);
        kd.lws0 = min_lws;
        kd.kernel_name = kernel_name_yxfb_yxio_b16_fp16;
    }
    else
    {
        kd = default_yxio_f16(arg);
    }
    return kd;
}

convolution_gpu::kernel_data defauly_bfyx_yxio_b1_f32(const convolution& arg)
{
    auto& filter_mem = arg.input_memory(1);

    int block_width = 4;
    int block_height = 3;

    convolution_gpu::kernel_data kd = convolution_gpu::set_default(arg);
    if (arg.argument.stride.spatial[0] == 1 && arg.argument.stride.spatial[1] == 1)
    {
        if (filter_mem.argument().size.spatial[0] == 1 && filter_mem.argument().size.spatial[1] == 1)
        {
            kd.kernel_name = kernel_name_bfyx_os_iyx_osv16_b1_f32_kernel_size_1;
            block_width = 16;
            block_height = 1;
        }
        else
        {
            kd.kernel_name = kernel_name_bfyx_os_iyx_osv16_b1_f32_stride1;
            block_width = 6;
            block_height = 4;
        }
    }
    else if (arg.argument.stride.spatial[0] == 2 && arg.argument.stride.spatial[1] == 2)
    {
        kd.kernel_name = kernel_name_bfyx_os_iyx_osv16_b1_f32_stride2;
        block_width = 5;
        block_height = 4;
    }
    else
    {
        kd.kernel_name = kernel_name_bfyx_os_iyx_osv16_b1_f32;
    }

    auto output_size = arg.non_padded_output_layout().size;
    kd.gws0 = static_cast<size_t>(std::ceil(static_cast<float>(output_size.spatial[0]) / block_width));
    kd.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(output_size.spatial[1]) / block_height));
    kd.gws2 = filter_mem.argument().size.feature[0];
    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = 16;
    if (kd.gws2 % 16)
    {
        kd.gws2 += 16;
        kd.gws2 -= kd.gws2 % 16;
    }
    kd.block_width = block_width;
    kd.block_height = block_height;
    return kd;
}

convolution_gpu::ks_type convolution_gpu::ks = {
    { std::make_tuple(memory::format::yxfb_f32, memory::format::oiyx_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_oiyx_f32 },
    { std::make_tuple(memory::format::yxfb_f32, memory::format::yxio_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f32 },
    { std::make_tuple(memory::format::yxfb_f32, memory::format::yxio_f32, 1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f32_b1 },
    { std::make_tuple(memory::format::yxfb_f32, memory::format::yxio_f32, 8, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f32_b8 },
    { std::make_tuple(memory::format::yxfb_f32, memory::format::yxio_f32, 16, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f32_b8 },
    { std::make_tuple(memory::format::yxfb_f32, memory::format::yxio_f32, 32, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f32_b32 },
    { std::make_tuple(memory::format::yxfb_f32, memory::format::yxio_f32, 64, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f32_b32 },
    { std::make_tuple(memory::format::yxfb_f32, memory::format::yxio_f32, 128, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f32_b32 },

    { std::make_tuple(memory::format::yxfb_f16, memory::format::yxio_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f16 },
    { std::make_tuple(memory::format::yxfb_f16, memory::format::yxio_f16, 16, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f16_b16 },
    { std::make_tuple(memory::format::yxfb_f16, memory::format::yxio_f16, 32, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f16_b16 },
    { std::make_tuple(memory::format::yxfb_f16, memory::format::yxio_f16, 64, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f16_b16 },
    { std::make_tuple(memory::format::yxfb_f16, memory::format::yxio_f16, 128, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio_f16_b16 },

    { std::make_tuple(memory::format::bfyx_f32, memory::format::os_iyx_osv16_f32, 1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), defauly_bfyx_yxio_b1_f32 },
};

namespace{
    struct attach {
        attach() {
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), convolution_gpu::create);
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

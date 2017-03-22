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
#include "kd_selector.h"

#include <initializer_list>

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
static const std::string kernel_name_bfyx_os_iyx_osv16 = "convolution_gpu_bfyx_os_iyx_osv16";

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
        size_t prefetch;
        size_t input_block_array_size; ///< Number of elements in array of UNIT_TYPE that must be specified in kernel to store/cache input block.
        size_t input_block_width;      ///< Number of elements in X dimension stored/cached in input block.
        std::string kernel_name;       ///< Name of a kernel/algorithm to execute.
        bool fp16_unit_used;           ///< Value indicating that FP16 half precision floating point type will be used (instead of single precision).
        size_t leftovers;
    } _kernel_data;

    gpu::kernel _kernel;

    static kernel_data set_default(const convolution& arg)
    {
        const auto& input_mem = arg.input_memory(0);  // input
        const auto& output_mem = arg.output_memory(); // output

        auto split = arg.argument.split();
        auto batch_size = output_mem.argument().size.batch[0];

        kernel_data kd;

        kd.fp16_unit_used = input_mem.get_layout().data_type == cldnn::data_types::f16;

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
        kd.prefetch = 0;
        kd.input_block_array_size = 0;
        kd.input_block_width = 0;
        kd.leftovers = 0;
        return kd;
    }

    typedef kd_selector_t<kernel_data, convolution, neural::memory::format::type, neural::memory::format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, gpu::engine_info_internal::configurations> ks_type;
    static ks_type ks;

    convolution_gpu(convolution &arg)
        : outer(arg)
        , _engine_info(arg.get_network().get_engine()->get_context()->get_engine_info())
        , _kernel_data(ks.get_kernel(outer, outer.input_memory(0).argument().format, outer.weights_memory(0).argument().format, outer.input_memory(0).argument().size.batch[0], _engine_info.architecture, _engine_info.configuration))
        , _kernel(arg.get_network().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(), outer.id())
    {}

    gpu::jit_constants get_jit_constants() const {

        auto& input_mem = outer.input_memory(0);
        auto input_offset = outer.desc()->input_offset().transform(input_mem.get_layout().size.format, 0);
        auto& output_mem = outer.output_memory();
        auto output_offset = outer.desc()->output_offset().transform(output_mem.get_layout().size.format, 0);
        auto& output_size = outer.output_memory().argument().size;
        auto& filter_mem = outer.weights_memory(0);
        auto split = outer.argument.split();

        const int batch_size = output_mem.argument().size.batch[0];

        auto input_size = outer.input().at(0)->non_padded_output_layout().size;
        cldnn::tensor stride(cldnn::format::yx, { std::min(outer.argument.stride.spatial[1], input_size.spatial[1]),
            std::min(outer.argument.stride.spatial[0], input_size.spatial[0]) });
        cldnn::padding input_padding = outer.input().at(0)->desc()->output_padding();

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",                     input_size),
            gpu::make_jit_constant("OUTPUT",                    outer.non_padded_output_layout().size),
            gpu::make_jit_constant("STRIDE",                    stride),
            gpu::make_jit_constant("INPUT_OFFSET",              input_offset),
            gpu::make_jit_constant("OUTPUT_OFFSET",             output_offset),
            // TODO: Output limit is incorrect for following cases (1. primitive used as input for two different convolutions with different padding, 2. asymmetric padding). Need to be checked and corrected.
            gpu::make_jit_constant("OUTPUT_LIMIT",              output_size),
            gpu::make_jit_constant("INPUT_PADDING",             input_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",            outer.argument.output_padding()),
            gpu::make_jit_constant("FILTER",                    filter_mem.argument().size),
            gpu::make_jit_constant("FILTER_ARRAY_NUM",          split),
            gpu::make_jit_constant("FILTER_OUTPUT_FEATURE_NUM", "FILTER_FEATURE_NUM_0"),
            gpu::make_jit_constant("FILTER_INPUT_FEATURE_NUM",  "FILTER_FEATURE_NUM_1"),
            gpu::make_jit_constant("FP16_SUPPORTED",            static_cast<int>(_engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",            static_cast<int>(_kernel_data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",                 _kernel_data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",             _kernel_data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("RELU",                      static_cast<int>(outer.argument.with_activation)),
            gpu::make_jit_constant("NEGATIVE_SLOPE",            outer.argument.activation_negative_slope),
        };

        if (filter_mem.argument().format == memory::format::yxio_f32 ||
            filter_mem.argument().format == memory::format::yxoi_f32 ||
            filter_mem.argument().format == memory::format::yxio_f16)
        {
            const auto local_work_group_size = _kernel_data.lws0;

            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE",                         local_work_group_size));
            mem_consts.add_constant(gpu::make_jit_constant("OFM_PER_WORK_ITEM",                             _kernel_data.ofm_per_work_item)); // how many output feature maps for a single batch will a single work item produce
            mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM",                         _kernel_data.batches_per_work_item)); // how many batches will a single work item compute
            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max(batch_size / _kernel_data.batches_per_work_item / local_work_group_size, static_cast<size_t>(1)))); // how many local work groups we need to compute single element for each batch
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS",        batch_size / _kernel_data.batches_per_work_item)); // how many work items we need to compute single element for each batch

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

        if (input_mem.argument().format == memory::format::bfyx_f32 ||
            input_mem.argument().format == memory::format::bfyx_f16)
        {
            mem_consts.add_constant(gpu::make_jit_constant("SUB_GROUP_SIZE",      _kernel_data.lws2));
            mem_consts.add_constant(gpu::make_jit_constant("OUT_BLOCK_WIDTH",     _kernel_data.block_width));
            mem_consts.add_constant(gpu::make_jit_constant("OUT_BLOCK_HEIGHT",    _kernel_data.block_height));
            mem_consts.add_constant(gpu::make_jit_constant("IN_BLOCK_ARRAY_SIZE", _kernel_data.input_block_array_size));
            mem_consts.add_constant(gpu::make_jit_constant("IN_BLOCK_WIDTH",      _kernel_data.input_block_width));
            mem_consts.add_constant(gpu::make_jit_constant("PREFETCH",            _kernel_data.prefetch));
            if (_kernel_data.leftovers)
                mem_consts.add_constant(gpu::make_jit_constant("LEFTOVERS",       _kernel_data.leftovers));
        }

        return mem_consts;
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> execute(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events) override
    {
        auto me = this;

        auto split = outer.argument.split();

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory();
        auto& filter_mem = outer.weights_memory(0);

        if (outer.desc()->padding_filling_value() != 0.0f)
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

        assert(arg.output_memory().argument().size.feature[0] / arg.argument.split() == filter_arg.size.feature[0]); // memory::format oixy
        
        switch (filter_arg.format)
        {
        // FP32 (float)
        case memory::format::oiyx_f32:
        case memory::format::yxio_f32:
        case memory::format::os_iyx_osv16_f32:
        // FP16 (half)
        case memory::format::oiyx_f16:
        case memory::format::yxio_f16:
        case memory::format::os_iyx_osv16_f16:
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
    auto split = arg.argument.split();
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
    auto split = arg.argument.split();
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
    auto split = arg.argument.split();
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
        assert(output_mem.argument().size.feature[0] == filter_ofm_num * arg.argument.split());
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

/// Computes dimensions of input block required for "kernel_name_bfyx_os_iyx_osv16" algorithm.
///
/// @param output_block_width    Width of output block used in algorithm.
/// @param output_block_height   Height of output block used in algorithm.
/// @param filter_size           Tensor with spatial (X, Y) data containing filter size used in algorithm.
/// @param stride                Tensor with spatial (X, Y) data containing stride used in algorithm.
/// @param sub_group_size        Number of work items grouped in single sub-group (enforced SIMD size).
/// @param read_chunk_size       Size of smallest chunk of data that can be read by sub-group in algorithm
///                              (in number of elements).
/// @param min_read_size         Minimal number of elements of input read by sub-group when reading single
///                              row od data.
///                              (Number of elements read by sub-group for single row will meet the formula:
///                              max(@p min_read_size, n * @p read_chunk_size), where n is a positive integer value.)
///
/// @return   Pair containing:
///            - [first] Number of sub-group-sized vectors of unit type needed to store/cache input block.
///                      The number is equivalent to size of array of UNIT_TYPE in work item needed to store/cache
///                      input block. It can be declared in kernel in form similar to:
///                         UNIT_TYPE in_block[ret.first];
///            - [second] Number of elements in X dimension needed to be read from input to compute output block
///                       without re-reading input.
static std::pair<size_t, size_t> get_bfyx_req_input_block_dims(
    size_t output_block_width,
    size_t output_block_height,
    const cldnn::tensor& filter_size,
    const cldnn::tensor& stride,
    size_t sub_group_size = 16,
    size_t read_chunk_size = 8,
    size_t min_read_size = 16)
{
    assert(output_block_width > 0 && output_block_height > 0);
    assert(stride.spatial[0] > 0 && stride.spatial[1] > 0);
    assert(filter_size.spatial[0] > 0 && filter_size.spatial[1] > 0);

    // Number of elements in X dimension needed from input to compute output block without re-reading input.
    std::size_t input_block_req_width = (output_block_width - 1) * stride.spatial[0] + filter_size.spatial[0];
    // Number of elements in Y dimension needed from input to compute output block without re-reading input.
    std::size_t input_block_req_height = (output_block_height - 1) * stride.spatial[1] + filter_size.spatial[1];

    // Required number of elements in X dimension rounded to nearest >= read chunk size.
    std::size_t input_block_read_width = std::max(cldnn::round_up_to(input_block_req_width, read_chunk_size), min_read_size);
    // Number of sub-group-sized vectors of unit type needed to store input block.
    std::size_t input_block_array_size = cldnn::ceil_div(input_block_req_height * input_block_read_width, sub_group_size);

    return std::make_pair(input_block_array_size, input_block_read_width);
}

convolution_gpu::kernel_data default_bfyx_os_iyx_osv16(const convolution& arg)
{
    auto& filter_mem = arg.input_memory(1);
    auto output_size = arg.non_padded_output_layout().size;

    convolution_gpu::kernel_data kd = convolution_gpu::set_default(arg);
    kd.kernel_name = kernel_name_bfyx_os_iyx_osv16;

    // Maximum supported size (in any dimension) of filter by "kernel_name_bfyx_os_iyx_osv16" kernel.
    constexpr int max_supported_filter_size = 11;
    // Sub-group size used by "kernel_name_bfyx_os_iyx_osv16" kernel.
    constexpr int sub_group_size = 16;

    const uint32_t of_threads_per_batch = cldnn::round_up_to(filter_mem.argument().size.feature[0], sub_group_size);
    kd.leftovers = of_threads_per_batch - filter_mem.argument().size.feature[0];

    if (filter_mem.argument().size.spatial[0] > max_supported_filter_size ||
        filter_mem.argument().size.spatial[1] > max_supported_filter_size)
    {
        // TODO: Implement and use naive bfyx algorithm here.
        // TODO: Implement naive bfyx algorithm here and add fall-back condition if abs(input_offset) >= filter_size.
        throw std::runtime_error("Unsupported filter size (> 11) in bfyx convolution");
    }

    if (arg.argument.stride.spatial[0] == 1 && arg.argument.stride.spatial[1] == 1)
    {
        if (filter_mem.argument().size.spatial[0] == 1 && filter_mem.argument().size.spatial[1] == 1)
        {
            kd.block_width = 16;
            kd.block_height = 1;
            kd.prefetch = 4;
        }        
        //if less than 16 values is required to compute one single row of output
        //then each WI shall compute one sinle row to maximise reuse within SIMD subgroup (this gives very nice performance results)
        else if (output_size.spatial[0] + filter_mem.argument().size.spatial[0] - 1 < sub_group_size)
        {
            kd.block_width = output_size.spatial[0];
            kd.block_height = 1;
            kd.prefetch = 4;
        }
        else if (filter_mem.argument().size.spatial[0] < 5 && filter_mem.argument().size.spatial[1] < 5)
        {
            kd.block_width = sub_group_size - filter_mem.argument().size.spatial[0] + 1;
            kd.block_height = 2;
            kd.prefetch = 4;
        }
        else
        {
            kd.block_width = 4;
            kd.block_height = 3;
            kd.prefetch = 4;
        }
    }
    else if (arg.argument.stride.spatial[0] == 2 && arg.argument.stride.spatial[1] == 2)
    {
        kd.block_width = 5;
        kd.block_height = 4;
        kd.prefetch = 4;
    }
    else if (arg.argument.stride.spatial[0] == 4 && arg.argument.stride.spatial[1] == 4)
    {
        kd.block_width = 4;
        kd.block_height = 3;
        kd.prefetch = 5;
    }
    else
    {
        // TODO: Implement and use naive bfyx algorithm here.
        throw std::runtime_error("Unsupported stride (!= 1,2,4) in bfyx convolution");
    }


    // Non-naive algorithm is used.
    if (kd.kernel_name == kernel_name_bfyx_os_iyx_osv16)
    {
        auto input_block_dims = get_bfyx_req_input_block_dims(kd.block_width, kd.block_height,
                                                                filter_mem.argument().size,
                                                                arg.argument.stride,
                                                                sub_group_size,
                                                                kd.fp16_unit_used ? sub_group_size : sub_group_size / 2,
                                                                sub_group_size);
        kd.input_block_array_size = input_block_dims.first;
        kd.input_block_width      = input_block_dims.second;
    }
    
    kd.gws0 = cldnn::ceil_div(output_size.spatial[0], kd.block_width);
    kd.gws1 = cldnn::ceil_div(output_size.spatial[1], kd.block_height);
    kd.gws2 = of_threads_per_batch * output_size.batch[0];

    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = sub_group_size;

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

    { std::make_tuple(memory::format::bfyx_f32, memory::format::os_iyx_osv16_f32, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_os_iyx_osv16 },

    { std::make_tuple(memory::format::bfyx_f16, memory::format::os_iyx_osv16_f16, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_bfyx_os_iyx_osv16 },
};

namespace{
    struct attach {
        attach() {
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f32), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::yxfb_f16), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f32), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, memory::format::bfyx_f16), convolution_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}

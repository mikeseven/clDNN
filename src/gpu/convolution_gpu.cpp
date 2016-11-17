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
#include "cache/primitive_db.h"

namespace neural 
{

const std::string kernelName_YXFB_memory = "Convolution_GPU_YXFB_memory";
const std::string kernelName_YXFB_OYXI_memory = "Convolution_GPU_YXFB_OYXI_memory";
const std::string kernelName_YXFB_YXOI_memory = "Convolution_GPU_YXFB_YXOI_memory";
const std::string kernelName_YXFB_YXOI_B8_memory = "Convolution_GPU_YXFB_YXOI_B8_memory";
const std::string kernelName_YXFB_YXIO = "Convolution_GPU_YXFB_YXIO";
const std::string kernelName_YXFB_YXIO_B1_memory = "Convolution_GPU_YXFB_YXIO_B1_memory";
const std::string kernelName_YXFB_YXIO_B1_vload_memory = "Convolution_GPU_YXFB_YXIO_B1_vload_memory";
const std::string kernelName_YXFB_YXIO_B1_block_memory = "Convolution_GPU_YXFB_YXIO_B1_block_memory";
const std::string kernelName_YXFB_YXIO_B1_block_x2_memory = "Convolution_GPU_YXFB_YXIO_B1_block_x2_memory";
const std::string kernelName_YXFB_YXIO_B8_memory = "Convolution_GPU_YXFB_YXIO_B8_memory";
const std::string kernelName_YXFB_YXIO_B16_memory = "Convolution_GPU_YXFB_YXIO_B16_memory";
const std::string kernelName_YXFB_YXOI_B8_F8_memory = "Convolution_GPU_YXFB_YXOI_B8_F8_memory";
const std::string kernel_name_yxfb_yxio_b16_fp16_memory = "convolution_gpu_yxfb_yxio_b16_fp16_memory";
const std::string kernel_name_yxfb_yxio_fp16 = "convolution_gpu_yxfb_yxio_fp16";


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

struct convolution_gpu : is_an_implementation {
    convolution &outer;
    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0;
        std::string kernel_name;
    } _kernel_data;

    bool in_feature_multiple_of_8; // if input feature maps are multiple of 8
    bool out_feature_multiple_of_8; // if output feature maps are multiple of 8
    gpu::kernel _kernel;

    convolution_gpu(convolution &arg): is_an_implementation(neural::type_id<convolution_gpu>())
        , outer(arg)
        , _kernel_data(set_kernel_data())
        , in_feature_multiple_of_8(outer.input_memory(0).argument.size.feature[0] % 8 == 0)
        , out_feature_multiple_of_8(outer.output_memory(0).argument.size.feature[0] % 8 == 0)
        , _kernel(_kernel_data.kernel_name, get_jit_constants())
    {}

    const kernel_data set_kernel_data() const
    {
        auto& input_mem = outer.input_memory(0);
        auto& filter_mem = outer.input_memory(1);
        auto& output_mem = outer.output_memory(0);

        kernel_data kd;
        kd.gws1 = output_mem.argument.size.spatial[0];
        kd.gws2 = output_mem.argument.size.spatial[1];

        auto split = outer.argument.split;
        auto batch_size = input_mem.argument.size.batch[0];

        if (padding::zero != outer.argument.padding)
            throw std::invalid_argument("Unknown padding mode in convolution.");

        switch (input_mem.argument.format) {
        // FP32 (float)
        case memory::format::yxfb_f32:
        {
            switch (filter_mem.argument.format) {
            case memory::format::oyxi_f32:
            case memory::format::oiyx_f32:
            {
                kd.gws0 = (output_mem.argument.size.feature[0] * batch_size) / split;
                kd.lws0 = std::min(kd.gws0, static_cast<size_t>(32));
                while (kd.gws0 % kd.lws0)
                {
                    kd.lws0--;
                }
                if (filter_mem.argument.format == memory::format::oiyx_f32)
                {
                    kd.kernel_name = kernelName_YXFB_memory;
                }
                else
                {
                    kd.kernel_name = kernelName_YXFB_OYXI_memory;
                }
                break;
            }
            case memory::format::yxoi_f32:
                if (batch_size == 8 &&
                    out_feature_multiple_of_8)
                {
                    if (in_feature_multiple_of_8)
                    {
                        kd.gws0 = (output_mem.argument.size.feature[0] * 8) / split;
                        kd.kernel_name = kernelName_YXFB_YXOI_B8_F8_memory;
                    }
                    else
                    {
                        uint32_t ofm_per_workitem = get_ofm_per_work_item(batch_size, filter_mem);
                        kd.gws0 = (output_mem.argument.size.feature[0] / (ofm_per_workitem / batch_size)) / split;
                        kd.kernel_name = kernelName_YXFB_YXOI_B8_memory;
                    }
                }
                else
                {
                    kd.gws0 = output_mem.count() / split;
                    kd.kernel_name = kernelName_YXFB_YXOI_memory;
                }
                kd.lws0 = 8;
                break;
            case memory::format::yxio_f32:
                if (filter_mem.argument.size.feature[0] * batch_size % get_local_work_group_size(batch_size, filter_mem) != 0)
                {
                    kd.gws0 = (output_mem.argument.size.feature[0] * output_mem.argument.size.batch[0]) / split;
                    kd.lws0 = 32;
                    while (kd.gws0 % kd.lws0)
                    {
                        kd.lws0--;
                    }
                    kd.kernel_name = kernelName_YXFB_YXIO;
                }
                else
                {
                    uint32_t ofm_per_workitem = get_ofm_per_work_item(batch_size, filter_mem);
                    uint32_t batches_per_workitem = get_batches_per_work_item(batch_size, filter_mem);
                    kd.gws0 = (output_mem.argument.size.feature[0] * batch_size / (ofm_per_workitem * batches_per_workitem)) / split;
                    kd.lws0 = static_cast<size_t>(get_local_work_group_size(batch_size, filter_mem));
                    if (batch_size == 1)
                    {
                        int ofm_per_work_item = get_ofm_per_work_item(batch_size, filter_mem);
                        if (ofm_per_work_item == 8 ||
                            ofm_per_work_item == 4 ||
                            ofm_per_work_item == 2)
                        {
                            // We compute 2 spatial coordinates "x" in a single workitem that's why we divide it by 2
                            kd.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(kd.gws1) / 2.0f));
                            kd.kernel_name = kernelName_YXFB_YXIO_B1_block_x2_memory;
                        }
                        else
                            kd.kernel_name = kernelName_YXFB_YXIO_B1_memory;
                    }
                    else if (input_mem.argument.size.batch[0] > 16)
                    {
                        kd.kernel_name = kernelName_YXFB_YXIO_B16_memory;
                    }
                    else
                    {
                        kd.kernel_name = kernelName_YXFB_YXIO_B8_memory;
                    }
                }
                break;
            default:
                throw std::invalid_argument("Filter memory is not supported");
            }
            break;
        }

        // FP16 (half)
        case memory::format::yxfb_f16:
        {
            switch (filter_mem.argument.format) {
            case memory::format::yxio_f16:
                // Number of output features is positive and dividable by 16.
                if ((filter_mem.argument.size.feature[0] & ~0xFU) != 0 &&
                    // Batch size is positive and dividable by 16.
                    (input_mem.argument.size.batch[0] & ~0xFU) != 0)
                {
                    uint32_t ofm_per_workitem = get_ofm_per_work_item(batch_size, filter_mem);
                    uint32_t batches_per_workitem = get_batches_per_work_item(batch_size, filter_mem);
                    kd.gws0 = (output_mem.argument.size.feature[0] * batch_size / (ofm_per_workitem * batches_per_workitem)) / split;
                    kd.lws0 = static_cast<size_t>(get_local_work_group_size(batch_size, filter_mem));
                    kd.kernel_name = kernel_name_yxfb_yxio_b16_fp16_memory;
                }
                else
                {
                    kd.gws0 = (output_mem.argument.size.feature[0] * output_mem.argument.size.batch[0]) / split;
                    kd.lws0 = 32;
                    while (kd.gws0 % kd.lws0)
                    {
                        kd.lws0--;
                    }
                    kd.kernel_name = kernel_name_yxfb_yxio_fp16;
                }
                break;
            default:
                throw std::invalid_argument("Filter memory is not supported");
            }
            break;
        }

        default:
            throw std::invalid_argument("Input memory format is not supported");
        }

        return kd;
    }

    // how many output feature maps for a single batch will a single work item produce 
    static int get_ofm_per_work_item(const int batch_size, const neural::memory& filter_mem)
    {
        if (batch_size == 1)
        {
            int output_feature_count = filter_mem.argument.size.feature[0];
            int lws = get_local_work_group_size(batch_size, filter_mem);
            if (output_feature_count % (lws * 8) == 0)
            {
                return 8;
            }
            else if (output_feature_count % (lws * 4) == 0)
            {
                return 4;
            }
            else if (output_feature_count % (lws * 2) == 0)
            {
                return 2;
            }
            return 1;
        }
        if (memory::traits(filter_mem.argument.format).type->id == type_id<half_t>()->id)
            return 16;
        if (batch_size < 32)
        {
            int lws = get_local_work_group_size(batch_size, filter_mem);
            if (((filter_mem.argument.size.feature[0] * batch_size) / 16) % lws)
                return 8;
            return 16;
        }
        return 8;
    }

    // how many batches will a single work item compute
    static int get_batches_per_work_item(const int batch_size, const neural::memory& filter_mem)
    {
        if (memory::traits(filter_mem.argument.format).type->id == type_id<half_t>()->id)
        {
            if (batch_size >= 64)
                return 4; // USE_BLOCK_READ_2 + as_half4
            if (batch_size >= 32)
                return 2; // USE_BLOCK_READ_1 + as_half2
            return 1;
        }
        return batch_size > 16 ? 2 : 1;
    }

    static int get_local_work_group_size(const int batch_size, const neural::memory& filter_mem)
    {
        if (memory::traits(filter_mem.argument.format).type->id == type_id<half_t>()->id)
            return 16;
        if (batch_size == 1)
        {
            return 16;
        }
        return batch_size > 8 ? 16 : 8;
    }

    gpu::jit_constants get_jit_constants() const {
        gpu_info_helper gpu_info;
        auto engine_info = gpu_info.get_engine_info();

        auto& input_mem = outer.input_memory(0);
        auto& input_offset = outer.argument.input_offset;
        auto& output_mem = outer.output_memory(0);
        auto& output_offset = outer.argument.output_offset;
        auto& output_size = outer.argument.output_size;
        auto& filter_mem = outer.input_memory(1);
        auto split = outer.argument.split;
        auto negative_slope = outer.argument.negative_slope;

        neural::vector<uint32_t> stride(outer.argument.stride);
        stride.spatial[0] = std::min(stride.spatial[0], input_mem.argument.size.spatial[0]);
        stride.spatial[1] = std::min(stride.spatial[1], input_mem.argument.size.spatial[1]);

        // weights neural::vector is: {b}, {ofm, ifm} {spatials}
        // ofm - output feature maps
        // ifm - input feature maps
        // b = 1 always
        // (weights can't have batch so it is equal to 1)
        // Ofm and batch is cropped, ofm will be hold manually
        // Batch is included in output size

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT", input_mem.argument.size),
            gpu::make_jit_constant("OUTPUT", output_mem.argument.size),
            gpu::make_jit_constant("STRIDE", stride),
            gpu::make_jit_constant("INPUT_OFFSET", input_offset),
            gpu::make_jit_constant("OUTPUT_OFFSET", output_offset),
            gpu::make_jit_constant("OUTPUT_LIMIT", output_size),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(engine_info.supports_fp16))
        };

        if (outer.argument.use_relu)
        {
            mem_consts.add_constant(gpu::make_jit_constant("NEGATIVE_SLOPE", negative_slope));
            mem_consts.add_constant(gpu::make_jit_constant("RELU", ""));
        }

        mem_consts.add_constant(gpu::make_jit_constant("FILTER", outer.input_memory(1).argument.size));
        mem_consts.add_constant(gpu::make_jit_constant("FILTER_ARRAY_NUM", split));

        mem_consts.add_constant(gpu::make_jit_constant("FILTER_OUTPUT_FEATURE_NUM", "FILTER_FEATURE_NUM_0"));
        mem_consts.add_constant(gpu::make_jit_constant("FILTER_INPUT_FEATURE_NUM", "FILTER_FEATURE_NUM_1"));

        if (filter_mem.argument.format == memory::format::yxio_f32 ||
            filter_mem.argument.format == memory::format::yxoi_f32 ||
            filter_mem.argument.format == memory::format::yxio_f16)
        {
            const int batch_size = output_mem.argument.size.batch[0];
            const int batches_per_work_item = get_batches_per_work_item(batch_size, filter_mem);
            const int local_work_group_size = get_local_work_group_size(batch_size, filter_mem);
            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE", local_work_group_size));
            mem_consts.add_constant(gpu::make_jit_constant("OFM_PER_WORK_ITEM", get_ofm_per_work_item(batch_size, filter_mem))); // how many output feature maps for a single batch will a single work item produce
            mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", batches_per_work_item)); // how many batches will a single work item compute
            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max((batch_size / batches_per_work_item) / local_work_group_size, 1))); // how many local work groups we need to compute single element for each batch
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS", batch_size / batches_per_work_item)); // how many work items we need to compute single element for each batch
            // A LITTLE HACK, for convolutions with low number of input features don't use block reads, and it will speed up by 25%
            // TODO - investigate why is this happening
            if (memory::traits(filter_mem.argument.format).type->id == type_id<half_t>()->id)
            {
                if (batch_size >= 64)
                    mem_consts.add_constant(gpu::make_jit_constant("USE_BLOCK_READ_2", ""));
                else if (batch_size >= 32)
                    mem_consts.add_constant(gpu::make_jit_constant("USE_BLOCK_READ_1", ""));
            }
            else if (input_mem.argument.size.feature[0] > 4)
            {
                mem_consts.add_constant(gpu::make_jit_constant("USE_BLOCK_READ_2", ""));
            }
            if (batch_size == 1)
            {
                int ofm_per_work_item = get_ofm_per_work_item(batch_size, filter_mem);
                if (ofm_per_work_item == 8)
                {
                    mem_consts.add_constant(gpu::make_jit_constant("USE_VECTOR_8", ""));
                }
                else if (ofm_per_work_item == 4)
                {
                    mem_consts.add_constant(gpu::make_jit_constant("USE_VECTOR_4", ""));
                }
                else if(ofm_per_work_item == 2)
                {
                    mem_consts.add_constant(gpu::make_jit_constant("USE_VECTOR_2", ""));
                }
            }
        }
        if (_kernel_data.kernel_name == kernelName_YXFB_YXIO_B1_block_x2_memory)
        {
            mem_consts.add_constant(gpu::make_jit_constant("X_PER_WORK_ITEM", 2));
        }
        return mem_consts;
    }

    static void implementation(const void *ptr) 
	{
        auto me = static_cast<const convolution_gpu*>(ptr);
        auto& outer = me->outer;

        auto split = outer.argument.split;

        auto& input_mem = outer.input_memory(0);
        auto& output_mem = outer.output_memory(0);
        auto& filter_mem = outer.input_memory(1);
        // weights neural::vector is: {b}, {ofm, ifm} {spatials}
        // ofm - output feature maps
        // ifm - input feature maps
        // b = 1 always
        // (weights can't have batch so it is equal to 1)
        // Ofm and batch is cropped, ofm will be hold manually
        // Batch is included in output size

        if (outer.argument.padding != padding::zero)
            throw std::invalid_argument("Unknown padding mode in convolution.");

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        if (memory::traits(input_mem.argument.format).type->id != memory::traits(output_mem.argument.format).type->id)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of output.");
        if (memory::traits(input_mem.argument.format).type->id != memory::traits(filter_mem.argument.format).type->id)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of filter.");

        auto& kd = me->_kernel_data;

        // execute kernels
        for (uint32_t i = 0; i < split; i++) {
            assert(kd.gws0 % kd.lws0 == 0);
            me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem, uint32_t>
                ({ { kd.gws0, kd.gws1, kd.gws2 },{ kd.lws0, 1, 1 } },
                    input_mem,
                    output_mem,
                    outer.input_memory(i * 2 + 1), //filters
                    outer.input_memory(i * 2 + 2), //biases
                    i);
        }
    }

    static is_an_implementation *create(convolution &arg) {
        auto& filter_arg = arg.input_memory(1).argument; //convolution filter

        assert(arg.argument.output_size.feature[0] / arg.argument.split == filter_arg.size.feature[0]); // memory::format oixy
        
        switch (filter_arg.format)
        {
        // FP32 (float)
        case memory::format::oiyx_f32:
        case memory::format::yxoi_f32:
        case memory::format::oyxi_f32:
        case memory::format::yxio_f32:
        // FP16 (half)
        case memory::format::oiyx_f16:
        case memory::format::yxoi_f16:
        case memory::format::oyxi_f16:
        case memory::format::yxio_f16:
            break;
        default:
            throw std::runtime_error("Convolution weights format unsupported");
        }

        return new convolution_gpu(arg);
    }

    task_group work() override {
        return{ { task{ implementation, this } }, schedule::single };
    }
};


namespace{
    struct attach {
        attach() {
            auto val_fw = convolution_gpu::create;

            auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
            implementation_map<convolution>::add(key_fw, val_fw);
            key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f16, memory::format::yxfb_f16);
            implementation_map<convolution>::add(key_fw, val_fw);
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

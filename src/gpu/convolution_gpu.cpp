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
#include "relu_gpu.h"
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
const std::string kernelName_YXFB_YXIO_B8_memory = "Convolution_GPU_YXFB_YXIO_B8_memory";
const std::string kernelName_YXFB_YXIO_B16_memory = "Convolution_GPU_YXFB_YXIO_B16_memory";
const std::string kernelName_YXFB_YXOI_B8_F8_memory = "Convolution_GPU_YXFB_YXOI_B8_F8_memory";

struct convolution_gpu : is_an_implementation {
    convolution &outer;
    bool in_feature_multiple_of_8; // if input feature maps are multiple of 8
    bool out_feature_multiple_of_8; // if output feature maps are multiple of 8
    gpu::kernel _kernel;

    convolution_gpu(convolution &arg): is_an_implementation(neural::type_id<convolution_gpu>())
        , outer(arg)
        , in_feature_multiple_of_8(outer.input_memory(0).argument.size.feature[0] % 8 == 0)
        , out_feature_multiple_of_8(outer.output_memory(0).argument.size.feature[0] % 8 == 0)
        , _kernel(select_kernel_name(), get_jit_constants())
    {}

    const std::string& select_kernel_name() const {
        // input
        auto& input_mem = outer.input_memory(0);
        auto& filter_mem = outer.input_memory(1);
        auto batch_size = input_mem.argument.size.batch[0];

        if (padding::zero != outer.argument.padding)
            throw std::invalid_argument("Unknown padding mode in convolution.");

        switch (input_mem.argument.format) {
        case memory::format::yxfb_f32:
            switch (filter_mem.argument.format) {
            case memory::format::oiyx_f32:
                return kernelName_YXFB_memory;
            case memory::format::yxoi_f32:
                if (batch_size == 8 &&
                    out_feature_multiple_of_8)
                {
                    if (in_feature_multiple_of_8)
                        return kernelName_YXFB_YXOI_B8_F8_memory;
                    else
                        return kernelName_YXFB_YXOI_B8_memory;
                }
                else
                    return kernelName_YXFB_YXOI_memory;
            case memory::format::yxio_f32:
                if (batch_size == 1)
                {
                    int ofm_per_work_item = get_ofm_per_work_item(batch_size, filter_mem);
                    if (ofm_per_work_item == 8 ||
                        ofm_per_work_item == 4 ||
                        ofm_per_work_item == 2)
                        return kernelName_YXFB_YXIO_B1_block_memory;
                    else
                        return kernelName_YXFB_YXIO_B1_memory;
                }
                if (input_mem.argument.size.batch[0] > 16)
                    return kernelName_YXFB_YXIO_B16_memory;
                else
                    return kernelName_YXFB_YXIO_B8_memory;
            case memory::format::oyxi_f32:
                return kernelName_YXFB_OYXI_memory;
            default:
                throw std::invalid_argument("Filter memory is not supported");
            }
        default:
            throw std::invalid_argument("Input memory format is not supported");
        }
    }

    // how many output feature maps for a single batch will a single work item produce 
    static int get_ofm_per_work_item(const int batch_size, const neural::memory& filter_mem)
    {
        if (batch_size == 1)
        {
            int output_feature_count = filter_mem.argument.size.feature[0];
            int lws = get_local_work_group_size(batch_size);
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
        return batch_size > 16 ? 8 : 16;
    }

    // how many batches will a single work item compute
    static int get_batches_per_work_item(const int batch_size)
    {
        return batch_size > 16 ? 2 : 1;
    }

    static int get_local_work_group_size(const int batch_size)
    {
        if (batch_size == 1)
        {
            return 16;
        }
        return batch_size > 8 ? 16 : 8;
    }

    gpu::jit_constants get_jit_constants() const {

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
            filter_mem.argument.format == memory::format::yxoi_f32)
        {
            const int batch_size = output_mem.argument.size.batch[0];
            const int batches_per_work_item = get_batches_per_work_item(batch_size);
            const int local_work_group_size = get_local_work_group_size(batch_size);
            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE", local_work_group_size));
            mem_consts.add_constant(gpu::make_jit_constant("OFM_PER_WORK_ITEM", get_ofm_per_work_item(batch_size, filter_mem))); // how many output feature maps for a single batch will a single work item produce
            mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", batches_per_work_item)); // how many batches will a single work item compute
            mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max((batch_size / batches_per_work_item) / local_work_group_size, 1))); // how many local work groups we need to compute single element for each batch
            mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS", batch_size / batches_per_work_item)); // how many work items we need to compute single element for each batch
            // A LITTLE HACK, for convolutions with low number of input features don't use block reads, and it will speed up by 25%
            // TODO - investigate why is this happening
            if (input_mem.argument.size.feature[0] > 4)
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

        auto dstSize = output_mem.count();

        size_t gws0;
        size_t lws0;

        uint32_t batch_size = output_mem.argument.size.batch[0];

        // compute global and local work sizes for kernels
        switch (input_mem.argument.format) {
        case memory::format::yxfb_f32:
            switch (filter_mem.argument.format)
            {
            case memory::format::yxoi_f32:
            {
                if (batch_size == 8 &&
                    me->out_feature_multiple_of_8)
                {
                    if (me->in_feature_multiple_of_8)
                    {
                        gws0 = (output_mem.argument.size.feature[0] * 8) / split;
                    }
                    else
                    {
                        uint32_t ofm_per_workitem = get_ofm_per_work_item(batch_size, filter_mem);
                        gws0 = (output_mem.argument.size.feature[0] / (ofm_per_workitem / batch_size)) / split;
                    }
                }
                else
                {
                    gws0 = dstSize / split;
                }
                lws0 = 8;
                break;
            }
            case memory::format::yxio_f32:
            {
                uint32_t ofm_per_workitem = get_ofm_per_work_item(batch_size, filter_mem);
                uint32_t batches_per_workitem = get_batches_per_work_item(batch_size);
                gws0 = (output_mem.argument.size.feature[0] * batch_size / (ofm_per_workitem * batches_per_workitem)) / split;
                lws0 = static_cast<size_t>(get_local_work_group_size(batch_size));
                break;
            }
            case memory::format::oiyx_f32:
            case memory::format::oyxi_f32:
            {
                gws0 = (output_mem.argument.size.feature[0] * batch_size) / split;
                lws0 = std::min(gws0, static_cast<size_t>(32));
                while (gws0%lws0)
                {
                    lws0 /= 2;
                }
                break;
            }
            default:
                throw std::invalid_argument("Filter memory format is not supported");
            }
            break;
        default:
            throw std::invalid_argument("Input memory format is not supported");
        }

        // execute kernels
        for (uint32_t i = 0; i < split; i++) {
            me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem, uint32_t>
                ({ { gws0, output_mem.argument.size.spatial[0], output_mem.argument.size.spatial[1] },{ lws0, 1, 1 } },
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
        case memory::format::oiyx_f32:
        case memory::format::yxoi_f32:
        case memory::format::oyxi_f32:
        case memory::format::yxio_f32:
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
			// cache implementation phase #1 that is a initial switch for using primitive database instead of string kernels
			// at later steps primitive database will be created only once per loading library but as for now it would require 
			// large refactor, so it will be done in smaller incremental steps. The same goes for picking first implementation
			// from the returned list.
			gpu::manager::primitive_db database; 
            gpu::kernel_templates::add(kernelName_YXFB_memory, database.get(kernelName_YXFB_memory).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_YXOI_memory, database.get(kernelName_YXFB_YXOI_memory).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_OYXI_memory, database.get(kernelName_YXFB_OYXI_memory).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_YXOI_B8_memory, database.get(kernelName_YXFB_YXOI_B8_memory).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_YXIO, database.get(kernelName_YXFB_YXIO).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_YXIO_B1_memory, database.get(kernelName_YXFB_YXIO_B1_memory).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_YXIO_B1_vload_memory, database.get(kernelName_YXFB_YXIO_B1_vload_memory).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_YXIO_B1_block_memory, database.get(kernelName_YXFB_YXIO_B1_block_memory).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_YXIO_B8_memory, database.get(kernelName_YXFB_YXIO_B8_memory).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_YXIO_B16_memory, database.get(kernelName_YXFB_YXIO_B16_memory).at(0));
            gpu::kernel_templates::add(kernelName_YXFB_YXOI_B8_F8_memory, database.get(kernelName_YXFB_YXOI_B8_F8_memory).at(0));
            auto val_fw = convolution_gpu::create;

            auto key_fw = std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
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

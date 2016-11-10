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
#include "api/neural.h"
#include "relu_gpu.h"
#include "implementation_map.h"
#include "kernel.h"
#include "cache/primitive_db.h"

const std::string kernelName_xb_xb_memory = "Fully_Connected_GPU_xb_xb_memory";
const std::string kernelName_xb_bx_memory = "Fully_Connected_GPU_xb_bx_memory";
const std::string kernelName_xb_bx_b8_memory = "Fully_Connected_GPU_xb_bx_b8_memory";
const std::string kernelName_xb_xb_b8_x8_memory = "Fully_Connected_GPU_xb_xb_b8_x8_memory";
const std::string kernelName_xb_xb_b16_memory = "Fully_Connected_GPU_xb_xb_b16_memory";
const std::string KernelName_xb_xb_b8_x8_memory_vload = "Fully_Connected_GPU_xb_xb_b8_x8_memory_vload";
const std::string kernelName_yxfn_memory = "Fully_Connected_GPU_yxfn_memory";

namespace neural {

    struct fully_connected_gpu : is_an_implementation {
        fully_connected &outer;
        struct kernel_data 
        {
            size_t gws0, gws1;
            size_t lws0, lws1;
            std::string kernel_name;
        } _kernel_data;
        gpu::kernel _kernel;
       
        fully_connected_gpu(fully_connected &arg)
            : is_an_implementation(neural::type_id<fully_connected_gpu>())
            , outer(arg)
            , _kernel_data(set_kernel_data())
            , _kernel(_kernel_data.kernel_name, get_jit_constants())
        {}

        const kernel_data set_kernel_data() const
        {
            kernel_data kd;
            kd.gws0 = outer.output_memory(0).count();
            kd.gws1 = 1;
            kd.lws0 = 32;
            kd.lws1 = 1;

            auto& input_mem = outer.input_memory(0);
            auto& weight_mem = outer.input_memory(1);
            auto& output_mem = outer.output_memory(0);

            auto output_bufSize = output_mem.count();

            bool batch_multiple_of_8 = input_mem.argument.size.batch[0] % 8 == 0;

            // calculate local workgroup size
            while (output_bufSize % kd.lws0) {
                kd.lws0--;
            }

            switch (input_mem.argument.format)
            {
            case memory::format::yxfb_f32:
            case memory::format::xb_f32:
            case memory::format::x_f32:
            {
                switch (weight_mem.argument.format)
                {
                case memory::format::byxf_f32:
                case memory::format::bx_f32:
                {
                    if (batch_multiple_of_8)
                    {
                        kd.gws0 = output_mem.argument.size.batch[0];
                        kd.gws1 = output_mem.argument.size.spatial[0];
                        kd.lws0 = 8;
                        kd.lws1 = 1;
                        kd.kernel_name = kernelName_xb_bx_b8_memory;
                    }
                    else
                    {
                        kd.kernel_name = kernelName_xb_bx_memory;
                    }
                    break;
                }
                case memory::format::yxfb_f32:
                case memory::format::xb_f32:
                {
                    if (batch_multiple_of_8 &&
                        (output_mem.count() / output_mem.argument.size.batch[0]) % 8 == 0)
                    {
                        size_t groups_per_batches = get_local_groups_size(output_mem);
                        kd.gws0 = output_mem.count() / (get_neurons_per_work_item(output_mem) * get_batches_per_work_item(output_mem) * groups_per_batches);
                        kd.gws1 = groups_per_batches;
                        kd.lws0 = get_local_work_group_size(output_mem);
                        kd.lws1 = 1;
                        kd.kernel_name = KernelName_xb_xb_b8_x8_memory_vload;
                    }
                    else
                    {
                        kd.kernel_name = kernelName_xb_xb_memory;
                    }
                    break;
                }
                case memory::format::bfyx_f32:
                {
                    kd.kernel_name = kernelName_yxfn_memory;
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
        static int get_neurons_per_work_item(const neural::memory &output_mem)
        {
            int batch_size = output_mem.argument.size.batch[0];
            auto out_elements_count_per_batch = output_mem.count() / batch_size;
            if (out_elements_count_per_batch % 16 == 0)
                return 2;
            else
                return 1;
        }

        // how many batches will a single work item compute
        static int get_batches_per_work_item(const neural::memory &output_mem)
        {
            int batch_size = output_mem.argument.size.batch[0];
            return std::min(batch_size, 32);
        }

        static int get_local_work_group_size(const neural::memory &output_mem)
        {
            int batch_size = output_mem.argument.size.batch[0];
            if (batch_size >= 16)
                return 8;
            auto out_elements_count_per_batch = output_mem.count() / batch_size;
            if (out_elements_count_per_batch % 16 == 0)
                return 16;
            else
                return 8;
        }

        static int get_local_groups_size(const neural::memory &output_mem)
        {
            int batch_size = output_mem.argument.size.batch[0];
            return std::max(1, batch_size / get_batches_per_work_item(output_mem));
        }

        gpu::jit_constants get_jit_constants() const {
            auto& input_mem = outer.input_memory(0);
            auto& output_mem = outer.output_memory(0);
            auto& weight_mem = outer.input_memory(1);

            gpu::jit_constants mem_consts{
                gpu::make_jit_constant("INPUT", input_mem.argument.size),
                gpu::make_jit_constant("OUTPUT", output_mem.argument.size),
                gpu::make_jit_constant("INPUT_ELEMENTS_COUNT", input_mem.count() / input_mem.argument.size.batch[0]),
                gpu::make_jit_constant("WEIGHTS", weight_mem.argument.size)
            };

            if (outer.argument.use_relu)
            {
                mem_consts.add_constant(gpu::make_jit_constant("RELU", ""));
                mem_consts.add_constant(gpu::make_jit_constant("NEGATIVE_SLOPE", outer.argument.negative_slope));
            }

            if (_kernel_data.kernel_name == KernelName_xb_xb_b8_x8_memory_vload ||
                _kernel_data.kernel_name == kernelName_xb_xb_b16_memory)
            {
                int batch_size = input_mem.argument.size.batch[0];
                const int batches_per_work_item = get_batches_per_work_item(output_mem);
                const int local_work_group_size = get_local_work_group_size(output_mem);
                mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE", local_work_group_size));
                mem_consts.add_constant(gpu::make_jit_constant("NEURONS_PER_WORK_ITEM", get_neurons_per_work_item(output_mem))); // how many neurons for a single batch will a single work item produce
                mem_consts.add_constant(gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", batches_per_work_item)); // how many batches will a single work item compute
                mem_consts.add_constant(gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max((batch_size / batches_per_work_item) / local_work_group_size, 1))); // how many local work groups we need to compute single element for each batch
                mem_consts.add_constant(gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS", batch_size / batches_per_work_item)); // how many work items we need to compute single element for each batch
            }
            return mem_consts;
        }

        static void implementation(const void *ptr) {
            auto me = static_cast<const fully_connected_gpu *>(ptr);
            auto& outer = me->outer;

            auto& input_mem = outer.input_memory(0);
            auto& weight_mem = outer.input_memory(1);
            auto& bias_mem = outer.input_memory(2);
            auto& output_mem = outer.output_memory(0);

            auto& kd = me->_kernel_data;

            me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>
                ({ {kd.gws0, kd.gws1 }, { kd.lws0, kd.lws1 } }, input_mem, output_mem, weight_mem, bias_mem);

        }

        task_group work() override {
            return{ { task{ implementation, this } }, schedule::single };
        }

        static is_an_implementation *create(fully_connected &arg) {

            auto& input_mem = arg.input_memory(0);
            auto& input_size = input_mem.argument.size;

            // validate arguments
            if (input_mem.argument.format == memory::format::yxfb_f32) {
                // weights
                auto& weight_size = arg.input_memory(1).argument.size;
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
                { std::make_tuple(engine::gpu, memory::format::yxfb_f32, memory::format::xb_f32), val_fw },
                { std::make_tuple(engine::gpu, memory::format::xb_f32, memory::format::xb_f32), val_fw },
                { std::make_tuple(engine::gpu, memory::format::x_f32,  memory::format::x_f32), val_fw }
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
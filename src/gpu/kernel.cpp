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

#include <iterator>
#include "kernel.h"
#include "memory_gpu.h"

namespace neural { namespace gpu {

    namespace
    {
        class memory_arg {
            cldnn::memory _mem;

        protected:
            memory_arg(const cldnn::memory& mem) : _mem(mem) {}

        public:
            const cl::Buffer& get_buffer() const { return static_cast<const gpu_buffer*>(api_cast(_mem.get()))->get_buffer(); }
        };

        class input_mem : public memory_arg {
        public:
            input_mem(const cldnn::memory& mem) :memory_arg(mem) {}
        };

        class output_mem : public memory_arg {
        public:
            output_mem(const cldnn::memory& mem) :memory_arg(mem) {}
        };

        template<typename T, class Enable = void>
        struct kernel_arg_handler;

        template<typename T>
        struct kernel_arg_handler<T, typename std::enable_if<!std::is_base_of<memory_arg, T>::value>::type> {
            static const T& get(const T& arg) { return arg; }
        };

        template<typename T>
        struct kernel_arg_handler<T, typename std::enable_if<std::is_base_of<memory_arg, T>::value>::type> {
            static const cl::Buffer& get(const T& arg) { return arg.get_buffer(); }
        };

        inline cl::NDRange toNDRange(const std::vector<size_t>& v)
        {
            switch (v.size())
            {
            case 1:
                return cl::NDRange(v[0]);
            case 2:
                return cl::NDRange(v[0], v[1]);
            case 3:
                return cl::NDRange(v[0], v[1], v[2]);
            default:
                return cl::NullRange;
            }
        }

        void set_arguments(
            cl::Kernel& kernel,
            const ArgumentDescpirtor& args_desc,
            const kernel::kernel_arguments_desc& args)
        {
            size_t input_index = 0;

            const auto& data = args_desc.data;

            for (uint32_t i = 0; i < static_cast<uint32_t>(data.size()); i++)
            {
                cl_int status = CL_INVALID_ARG_VALUE;

                switch (data[i].t)
                {
                case ArgumentDescpirtor::Types::INPUT:
                    if (input_index < args.inputs.size() && args.inputs[input_index])
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.inputs[input_index]));
                        input_index++;
                    }
                    break;
                case ArgumentDescpirtor::Types::OUTPUT:
                    if (args.output)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::output_mem>::get(*args.output));
                    }
                    break;
                case ArgumentDescpirtor::Types::WEIGHTS:
                    if (args.weights)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.weights));
                    }
                    break;
                case ArgumentDescpirtor::Types::BIAS:
                    if (args.bias)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.bias));
                    }
                    break;
                case ArgumentDescpirtor::Types::LOOKUP_TABLE:
                    if (args.lookup_table)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.lookup_table));
                    }
                    break;
                case ArgumentDescpirtor::Types::SCALE_TABLE:
                    if (args.scale_table)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.scale_table));
                    }
                    break;
                case ArgumentDescpirtor::Types::SLOPE:
                    if (args.slope)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.slope));
                    }
                    break;
                case ArgumentDescpirtor::Types::SPLIT:
                    status = kernel.setArg(i, args.split);
                    break;
                case ArgumentDescpirtor::Types::UINT8:
                    status = kernel.setArg(i, data[i].v.u8);
                    break;
                case ArgumentDescpirtor::Types::UINT16:
                    status = kernel.setArg(i, data[i].v.u16);
                    break;
                case ArgumentDescpirtor::Types::UINT32:
                    status = kernel.setArg(i, data[i].v.u32);
                    break;
                case ArgumentDescpirtor::Types::UINT64:
                    status = kernel.setArg(i, data[i].v.u64);
                    break;
                case ArgumentDescpirtor::Types::INT8:
                    status = kernel.setArg(i, data[i].v.s8);
                    break;
                case ArgumentDescpirtor::Types::INT16:
                    status = kernel.setArg(i, data[i].v.s16);
                    break;
                case ArgumentDescpirtor::Types::INT32:
                    status = kernel.setArg(i, data[i].v.s32);
                    break;
                case ArgumentDescpirtor::Types::INT64:
                    status = kernel.setArg(i, data[i].v.s64);
                    break;
                case ArgumentDescpirtor::Types::FLOAT32:
                    status = kernel.setArg(i, data[i].v.f32);
                    break;
                case ArgumentDescpirtor::Types::FLOAT64:
                    status = kernel.setArg(i, data[i].v.f64);
                    break;
                default:
                    break;
                }

                if (status != CL_SUCCESS)
                {
                    throw std::runtime_error("Error set args\n");
                }
            }
        }
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> kernel::run(
        const KernelSelector::clKernelData& kernel_data,
        const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& dependencies,
        const kernel_arguments_desc& args) const
    {
        cl::Event end_event;
        std::vector<cl::Event> events;

        bool run_this_layer = false;
        if (context()->enabled_single_kernel())
        {
            std::string proper_layer_name = kernel_data.layerID;
            if (proper_layer_name.compare(context()->single_kernel_name()) == 0)
            {
                run_this_layer = true;
            }
        }
        else
        {
            run_this_layer = true;

            events.reserve(dependencies.size());
            for (auto& dependency : dependencies)
            {
                events.emplace_back(dependency->get());
            }
        }

        if (run_this_layer)
        {
            auto clkernel = context()->get_kernels_cache().get_kernel(_kernel_id);

            set_arguments(clkernel, kernel_data.argsDesc, args);

            context()->queue().enqueueNDRangeKernel(
                clkernel,
                cl::NullRange,
                toNDRange(kernel_data.workGroups.global),
                toNDRange(kernel_data.workGroups.local),
                &events,
                &end_event);
        }

        return{ new cldnn::event_impl(end_event), false };
    }

} }
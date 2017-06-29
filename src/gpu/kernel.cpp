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

void kernel_execution_options::set_local_sizes()
{
    const size_t optimal_lws_values[] = { 256, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1 };
    auto total_lws = std::accumulate(std::begin(_local), std::end(_local), size_t(1), std::multiplies<size_t>());
    assert(total_lws != 0 && total_lws <= 256);

    for (auto i = _local.size(); i < _global.size(); ++i)
    {
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = 0;
        while(rest_lws < optimal_lws_values[lws_idx]) lws_idx++;

        while (_global[i] % optimal_lws_values[lws_idx]) lws_idx++;

        _local.push_back(optimal_lws_values[lws_idx]);
        total_lws *= optimal_lws_values[lws_idx];
    }
}

std::vector<uint32_t> get_tensor_array(cldnn::format fmt, const cldnn::tensor& t)
{
    std::vector<uint32_t> ret;
    
    auto&& sizes = t.sizes(fmt);
    ret.reserve(sizes.size());

    for (auto itr = sizes.rbegin(); itr != sizes.rend(); ++itr)
        ret.push_back(*itr);

    return ret;
}

std::vector<uint32_t> get_accumulated_tensor_array(cldnn::format fmt, const cldnn::tensor& t)
{
    std::vector<uint32_t> ret;

    auto&& sizes = t.sizes(fmt);
    ret.reserve(sizes.size());

    uint32_t acc = 1;
    for (auto itr = sizes.rbegin(); itr != sizes.rend(); ++itr)
    {
        ret.push_back(acc);
        acc *= *itr;
    }

    return ret;

}

std::vector<uint32_t> get_sizes_array(cldnn::layout const& layout)
{
    return get_tensor_array(layout.format, layout.size);
}

std::vector<uint32_t> get_buffer_sizes_array(cldnn::layout const& layout)
{
    return get_tensor_array(layout.format, layout.get_buffer_size());
}

std::vector<uint32_t> get_accumulated_sizes_array(cldnn::layout const& layout)
{
    return get_accumulated_tensor_array(layout.format, layout.size);
}

std::vector<uint32_t> get_accumulated_buffer_sizes_array(cldnn::layout const& layout)
{
    return get_accumulated_tensor_array(layout.format, layout.get_buffer_size());
}

std::string get_offsets_string(size_t dimensions, const cldnn::tensor &sizes)
{
    std::stringstream os;
    os << "(uint[]){ ";
    for (size_t i = 0; i < dimensions; i++)
    {
        os << static_cast<uint32_t>(sizes.raw[i]) << ", ";
    }
    os << " }";
    return os.str();
}

namespace 
{
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
        const kernel_selector::argument_descpirtor& args_desc,
        const kernel::kernel_arguments_desc& args)
    {
        int8_t input_index = 0;

        const auto& data = args_desc.data;

        for (uint32_t i = 0; i < static_cast<uint32_t>(data.size()); i++)
        {
            cl_int status = CL_INVALID_ARG_VALUE;

            switch (data[i].t)
            {
            case kernel_selector::argument_descpirtor_types::INPUT:
                {
                    int8_t current_index;
                    if (args.use_input_index_as_order)
                    {
                        current_index = data[i].v.s8;
                    }
                    else
                    {
                        current_index = input_index;
                        input_index++;
                    }

                    if (current_index >= 0 &&
                        current_index < (int8_t)args.inputs.size() && 
                        args.inputs[current_index])
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.inputs[current_index]));
                    }
                    else
                    {
                        status = kernel.setArg(i, cl::Buffer());
                    }
                }
                break;
            case kernel_selector::argument_descpirtor_types::OUTPUT:
                if (args.output)
                {
                    status = kernel.setArg(i, kernel_arg_handler<gpu::output_mem>::get(*args.output));
                }
                break;
            case kernel_selector::argument_descpirtor_types::WEIGHTS:
                if (args.weights)
                {
                    status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.weights));
                }
                break;
            case kernel_selector::argument_descpirtor_types::BIAS:
                if (args.bias)
                {
                    status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.bias));
                }
                break;
            case kernel_selector::argument_descpirtor_types::LOOKUP_TABLE:
                if (args.lookup_table)
                {
                    status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.lookup_table));
                }
                break;
            case kernel_selector::argument_descpirtor_types::SCALE_TABLE:
                if (args.scale_table)
                {
                    status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.scale_table));
                }
                break;
            case kernel_selector::argument_descpirtor_types::SLOPE:
                if (args.slope)
                {
                    status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*args.slope));
                }
                break;
            case kernel_selector::argument_descpirtor_types::SPLIT:
                status = kernel.setArg(i, args.split);
                break;
            case kernel_selector::argument_descpirtor_types::UINT8:
                status = kernel.setArg(i, data[i].v.u8);
                break;
            case kernel_selector::argument_descpirtor_types::UINT16:
                status = kernel.setArg(i, data[i].v.u16);
                break;
            case kernel_selector::argument_descpirtor_types::UINT32:
                status = kernel.setArg(i, data[i].v.u32);
                break;
            case kernel_selector::argument_descpirtor_types::UINT64:
                status = kernel.setArg(i, data[i].v.u64);
                break;
            case kernel_selector::argument_descpirtor_types::INT8:
                status = kernel.setArg(i, data[i].v.s8);
                break;
            case kernel_selector::argument_descpirtor_types::INT16:
                status = kernel.setArg(i, data[i].v.s16);
                break;
            case kernel_selector::argument_descpirtor_types::INT32:
                status = kernel.setArg(i, data[i].v.s32);
                break;
            case kernel_selector::argument_descpirtor_types::INT64:
                status = kernel.setArg(i, data[i].v.s64);
                break;
            case kernel_selector::argument_descpirtor_types::FLOAT32:
                status = kernel.setArg(i, data[i].v.f32);
                break;
            case kernel_selector::argument_descpirtor_types::FLOAT64:
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

cldnn::refcounted_obj_ptr<cldnn::event_impl> kernel::run_custom_kernel(
        const kernel_selector::cl_kernel_data& kernel_data,
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
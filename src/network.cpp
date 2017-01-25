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
#include "api/topology.hpp"
#include "api/primitives/input_layout.hpp"
#include "network_impl.h"
#include "engine_impl.h"
#include "event_impl.h"
#include "network_builder.h"
#include "primitive_type.h"
#include "input_layout_arg.h"
#include <algorithm>
#include "gpu/kernel.h"

namespace cldnn
{
const char warmup_kernel_name[] = "warm_up_gpu";

network_impl::network_impl(refcounted_obj_ptr<engine_impl> engine, refcounted_obj_ptr<topology_impl> topology, const std::vector<primitive_id>& outputs)
    : _engine(engine)
    , _topology(topology)
    , _output_ids(outputs)
{
    for (auto& output : _output_ids)
    {
        auto p = get_primitive(output);
        assert(p);
    }

    for (auto& p : _primitives)
    {
        if (p.second->type() == input_layout::type_id())
        {
            _input_names.insert({ p.second->id(), false });
        }
    }

    //pre-compile program and warm-up
    auto context = _engine->get_context();
    neural::gpu::kernel warmup_kernel(context, "", warmup_kernel_name);
    cl::Buffer out(context->context(), CL_MEM_WRITE_ONLY, 1);
    warmup_kernel.run<cl_int, cl_int, cl_int, cl::Buffer>({ 1024, 8 }, {}, 0, 111, 7, out);
    context->queue().finish();
}

void network_impl::reset_execution(bool wait)
{
    if (wait && _events.size() > 0)
    {
        std::vector<cl::Event> events;
        events.reserve(_events.size());
        for (auto& pair : _events)
        {
            auto clevent = pair.second->get();
            auto event_status = clevent.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
            if (event_status != CL_COMPLETE)
            {
                events.emplace_back(clevent);
            }
        }
        if (events.size() > 0)
        {
            cl::WaitForEvents(events);
        }
    }
    _events.clear();
}

void network_impl::set_input_data(const primitive_id& id, memory_impl* data)
{
    auto& primitive = _primitives.at(id);
    if (primitive->type() != input_layout::type_id()) throw std::invalid_argument("primitive " + id + " is not an input");

    auto input_c = std::static_pointer_cast<const input_layout_arg>(primitive);
    auto input = std::const_pointer_cast<input_layout_arg>(input_c);

    //Wait for previous execution completion
    reset_execution(true);
    input->set_data(data);
    _input_names[input->id()] = true;
}

void network_impl::execute(const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    auto all_inputs_are_set = std::all_of(
                                    std::begin(_input_names),
                                    std::end(_input_names),
                                    [](decltype(*std::begin(_input_names)) p)
                                    {
                                        return p.second;
                                    });
    if (!all_inputs_are_set) throw std::runtime_error("not all inputs are set");

    //Wait for previous execution completion
    reset_execution(false);

    for(auto& output_id : _output_ids)
    {
        auto primitive = get_primitive(output_id);
        auto output_event = execute_primitive(primitive, events);
    }
}

std::shared_ptr<const primitive_arg> network_impl::get_primitive(const primitive_id& id)
{
    auto it = _primitives.find(id);
    if (it != _primitives.end())
    {
        return it->second;
    }

    auto& desc = _topology->at(id)->primitive_desc;
    auto primitive = desc->type()->create_arg(*this, desc);
    return _primitives.insert({ id, primitive }).first->second;
}

std::vector<std::shared_ptr<const primitive_arg>> network_impl::get_primitives(const std::vector<primitive_id>& ids)
{
    std::vector<std::shared_ptr<const primitive_arg>> result(ids.size());
    std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) { return get_primitive(id); });
    return result;
}

refcounted_obj_ptr<event_impl> network_impl::execute_primitive(const std::shared_ptr<const primitive_arg>& primitive, const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    auto id = primitive->id();
    auto it = _events.find(id);
    if(it != _events.end())
    {
        return it->second;
    }
    return _events.insert({ id, primitive->execute(events) }).first->second;
}

}

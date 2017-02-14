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
#include "api/cldnn.h"
#include "api_impl.h"
#include "engine_impl.h"
#include "topology_impl.h"
#include "primitive_type.h"
#include "network_impl.h"
#include "memory_impl.h"
#include "primitive_arg.h"


extern "C"
{

cldnn_topology cldnn_create_topology(cldnn_status* status)
{
    return exception_handler<cldnn_topology>(CLDNN_ERROR, status, nullptr, [&]()
    {
        return api_cast(new cldnn::topology_impl());
    });
}

void cldnn_add_primitive(cldnn_topology topology, const CLDNN_PRIMITIVE_DESC(primitive)* dto, cldnn_status* status)
{
    return exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(topology)->add(dto->type->from_dto(dto));
    });
}

void cldnn_retain_topology(cldnn_topology topology, cldnn_status* status)
{
    return exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(topology)->add_ref();
    });
}
void cldnn_release_topology(cldnn_topology topology, cldnn_status* status)
{
    return exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(topology)->release();
    });
}

uint32_t cldnn_get_engine_count(/*cldnn_engine_type*/ int32_t type, cldnn_status* status)
{
    if (type == cldnn_engine_type::cldnn_engine_ocl)
    {
        if (status) *status = CLDNN_SUCCESS;
        return 1;
    }
    else
    {
        if (status) *status = CLDNN_UNSUPPORTED;
        return 0;
    }
}

cldnn_engine cldnn_create_engine(/*cldnn_engine_type*/ int32_t type, uint32_t engine_num, const cldnn_engine_configuration* configuration, cldnn_status* status)
{
    if (engine_num > 0 || (type != cldnn_engine_type::cldnn_engine_ocl))
    {
        if (status)
            *status = CLDNN_UNSUPPORTED;
        return nullptr;
    }

    return exception_handler<cldnn_engine>(CLDNN_ERROR, status, nullptr, [&]()
    {
        return api_cast(new cldnn::engine_impl(configuration ? cldnn::engine_configuration(*configuration) : cldnn::engine_configuration()));
    });
}

void cldnn_retain_engine(cldnn_engine engine, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]() { api_cast(engine)->add_ref(); });
}

void cldnn_release_engine(cldnn_engine engine, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]() { api_cast(engine)->release(); });
}

cldnn_engine_info cldnn_get_engine_info(cldnn_engine engine, cldnn_status* status)
{
    return exception_handler<cldnn_engine_info>(CLDNN_ERROR, status, { 0, 0, 0, 0, 0, 0 }, [&]() -> cldnn_engine_info
    {
        auto info = api_cast(engine)->get_engine_info();
        return
        {
            info.cores_count,
            info.core_frequency,
            info.max_work_group_size,
            info.max_local_mem_size,
            info.supports_fp16,
            info.supports_fp16_denorms
        };
    });
}

/*cldnn_engine_type*/ int32_t cldnn_get_engine_type(cldnn_engine engine, cldnn_status* status)
{
    return exception_handler<int32_t>(CLDNN_ERROR, status, cldnn_engine_ocl, [&]()
    {
        return static_cast<int32_t>(api_cast(engine)->type());
    });
}

cldnn_event cldnn_create_user_event(cldnn_engine engine, cldnn_status* status)
{
    return exception_handler<cldnn_event>(CLDNN_ERROR, status, nullptr, [&]()
    {
        return api_cast(api_cast(engine)->create_user_event());
    });
}

void cldnn_retain_event(cldnn_event event, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]() { api_cast(event)->add_ref(); });
}

void cldnn_release_event(cldnn_event event, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]() { api_cast(event)->release(); });
}

void cldnn_wait_for_event(cldnn_event event, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(event)->wait();
    });
}

void cldnn_set_event(cldnn_event event, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(event)->set();
    });
}

void cldnn_add_event_handler(cldnn_event event, cldnn_event_handler handler, void* param, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(event)->add_event_handler(handler, param);
    });
}

void cldnn_get_event_profiling_info(cldnn_event event, cldnn_profiling_interval* profiling, size_t size, size_t* size_ret, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        auto& profiling_info = api_cast(event)->get_profiling_info();
        if (size_ret)
            *size_ret = profiling_info.size();
        if(size < profiling_info.size())
        {
            if(status) *status = CLDNN_INVALID_ARG;
            return;
        }
        for(decltype(profiling_info.size()) i = 0; i < profiling_info.size(); i++)
        {
            profiling[i].name = profiling_info[i].name;
            profiling[i].nanoseconds = profiling_info[i].nanoseconds;
        }
    });
}

cldnn_network cldnn_build_network(cldnn_engine engine, cldnn_topology topology, cldnn_build_option* options, size_t options_num, cldnn_status* status)
{
    return exception_handler<cldnn_network>(CLDNN_ERROR, status, nullptr, [&]()
    {
        cldnn::build_options options_obj(cldnn::array_ref<cldnn_build_option>(options, options_num));
        return api_cast(api_cast(engine)->build_network(api_cast(topology), options_obj));
    });
}

void cldnn_retain_network(cldnn_network network, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(network)->add_ref();
    });
}

void cldnn_release_network(cldnn_network network, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(network)->release();
    });
}

void cldnn_set_network_input(cldnn_network network, cldnn_primitive_id id, cldnn_memory mem, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(network)->set_input_data(id, api_cast(mem));
    });
}

cldnn_engine cldnn_get_network_engine(cldnn_network network, cldnn_status* status)
{
    return exception_handler<cldnn_engine>(CLDNN_ERROR, status, nullptr, [&]()
    {
        auto engine_ptr = api_cast(network)->get_engine();
        if (!engine_ptr) throw std::logic_error("no assigned engine");
        return api_cast(engine_ptr.detach());
    });
}

cldnn_topology cldnn_get_network_topology(cldnn_network network, cldnn_status* status)
{
    return exception_handler<cldnn_topology>(CLDNN_ERROR, status, nullptr, [&]()
    {
        auto topology_ptr = api_cast(network)->get_topology();
        if(!topology_ptr) throw std::logic_error("no assigned topology");
        return api_cast(topology_ptr.detach());
    });
}

void cldnn_get_network_output_names(cldnn_network network, char* names, size_t size, size_t* size_ret, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        auto& output_ids = api_cast(network)->get_output_ids();
        *size_ret = std::accumulate(
            std::begin(output_ids),
            std::end(output_ids),
            size_t(1), // final zero symbol
            [](size_t acc, const cldnn::primitive_id& id)
            {
                return acc + id.size() + 1; // plus zero symbol
            });

        if(size < *size_ret)
        {
            if (status) *status = CLDNN_INVALID_ARG;
            return;
        }

        size_t i = 0;
        for(auto& id: output_ids)
        {
// workaround for Microsoft VC++
#if defined _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
            i += id.copy(names + i, size - i - 2);
#if defined _MSC_VER
#pragma warning(pop)
#endif
            names[i++] = 0; // plus zero symbol
            assert(i < size);
        }
        names[i] = 0; // final zero symbol
    });
}

void cldnn_execute_network(cldnn_network network, cldnn_event* dependencies, size_t deps_num, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>> deps;
        deps.reserve(deps_num);
        for(size_t i = 0; i < deps_num; i++)
        {
            deps.emplace_back(api_cast(dependencies[i]));
        }

        api_cast(network)->execute(deps);
    });
}

cldnn_network_output cldnn_get_network_output(cldnn_network network, const char* name, cldnn_status* status)
{
    cldnn_network_output error_result = { nullptr, nullptr };
    return exception_handler<cldnn_network_output>(CLDNN_ERROR, status, error_result, [&]() -> cldnn_network_output
    {
        cldnn::primitive_id id(name);
        auto event = api_cast(network)->get_primitive_event(id);

        auto mem_result = api_cast(network)->get_primitive(id)->output_memory().get();
        api_cast(mem_result)->add_ref();
        return { api_cast(event.detach()), mem_result };
    });
}

cldnn_memory cldnn_allocate_memory(cldnn_engine engine, cldnn_layout layout, cldnn_status* status)
{
    return exception_handler<cldnn_memory>(CLDNN_ERROR, status, nullptr, [&]()
    {
        return api_cast(api_cast(engine)->allocate_buffer(layout));
    });
}

cldnn_memory cldnn_attach_memory(cldnn_layout layout, void* pointer, size_t size, cldnn_status* status)
{
    return exception_handler<cldnn_memory>(CLDNN_ERROR, status, nullptr, [&]()
    {
        cldnn::layout layout_obj(layout);
        if (layout_obj.data_size() > size) std::invalid_argument("buffer size does not match layout size");
        return api_cast(new cldnn::simple_attached_memory(layout_obj, pointer));
    });
}

void cldnn_retain_memory(cldnn_memory memory, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(memory)->add_ref();
    });
}

void cldnn_release_memory(cldnn_memory memory, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(memory)->release();
    });
}

void* cldnn_lock_memory(cldnn_memory memory, cldnn_status* status)
{
    return exception_handler<void*>(CLDNN_ERROR, status, nullptr, [&]()
    {
        return api_cast(memory)->lock();
    });
}

void cldnn_unlock_memory(cldnn_memory memory, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        api_cast(memory)->unlock();
    });
}

cldnn_layout cldnn_get_memory_layout(cldnn_memory memory, cldnn_status* status)
{
    cldnn_layout error_result = cldnn::layout(cldnn::data_types::f32, { cldnn::format::x, {0} });

    return exception_handler<cldnn_layout>(CLDNN_ERROR, status, error_result, [&]()
    {
        return api_cast(memory)->get_layout();
    });
}

cldnn_engine cldnn_get_memory_engine(cldnn_memory memory, cldnn_status* status)
{
    return exception_handler<cldnn_engine>(CLDNN_ERROR, status, nullptr, [&]()
    {
        auto engine_ptr = api_cast(memory)->get_engine();
        if (!engine_ptr) throw std::logic_error("no assigned engine");
        return api_cast(engine_ptr.detach());
    });
}

} /* extern "C" */

#define PRIMITIVE_TYPE_ID_CALL_IMPL(PType) \
namespace cldnn { primitive_type_id PType##_type_id(); }\
extern "C" CLDNN_API cldnn_primitive_type_id cldnn_##PType##_type_id(cldnn_status* status)\
{\
    return exception_handler<cldnn_primitive_type_id>(CLDNN_ERROR, status, nullptr, []()\
    {\
        return cldnn::PType##_type_id();\
    });\
}

PRIMITIVE_TYPE_ID_CALL_IMPL(activation)
PRIMITIVE_TYPE_ID_CALL_IMPL(batch_norm)
PRIMITIVE_TYPE_ID_CALL_IMPL(convolution)
PRIMITIVE_TYPE_ID_CALL_IMPL(crop)
PRIMITIVE_TYPE_ID_CALL_IMPL(data)
PRIMITIVE_TYPE_ID_CALL_IMPL(depth_concatenate)
PRIMITIVE_TYPE_ID_CALL_IMPL(eltwise)
PRIMITIVE_TYPE_ID_CALL_IMPL(fully_connected)
PRIMITIVE_TYPE_ID_CALL_IMPL(input_layout)
PRIMITIVE_TYPE_ID_CALL_IMPL(mean_substract)
PRIMITIVE_TYPE_ID_CALL_IMPL(normalization)
PRIMITIVE_TYPE_ID_CALL_IMPL(pooling)
PRIMITIVE_TYPE_ID_CALL_IMPL(reorder)
PRIMITIVE_TYPE_ID_CALL_IMPL(scale)
PRIMITIVE_TYPE_ID_CALL_IMPL(softmax)
PRIMITIVE_TYPE_ID_CALL_IMPL(simpler_nms)
PRIMITIVE_TYPE_ID_CALL_IMPL(roi_pooling)
PRIMITIVE_TYPE_ID_CALL_IMPL(prior_box)
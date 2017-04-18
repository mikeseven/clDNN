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
#include "program_impl.h"
#include "primitive_type.h"
#include "network_impl.h"
#include "memory_impl.h"
#include "primitive_inst.h"

namespace cldnn {
    last_err& last_err::instance() {
        thread_local static last_err _instance;
        return _instance;
    }
}

#define SHOULD_NOT_BE_NULL(arg, msg_prefix) \
    if(arg == nullptr) \
        throw std::invalid_argument( std::string(msg_prefix)  + " should not be null.");
#define SHOULD_NOT_EQUAL_0(arg, msg_prefix) \
    if(arg == 0) \
        throw std::invalid_argument( std::string(msg_prefix)  + " should not equals 0.");
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
        SHOULD_NOT_BE_NULL(topology,           "Topology");
        SHOULD_NOT_BE_NULL(dto,                "Primitive");
        SHOULD_NOT_BE_NULL(dto->id,            "Primitive id");
        SHOULD_NOT_BE_NULL(dto->type,          "Primitive type");
        api_cast(topology)->add(dto->type->from_dto(dto));
    });
}

void cldnn_retain_topology(cldnn_topology topology, cldnn_status* status)
{
    return exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(topology, "Topology");
        api_cast(topology)->add_ref();
    });
}
void cldnn_release_topology(cldnn_topology topology, cldnn_status* status)
{
    return exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(topology, "Topology");
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
        if (status) *status = CLDNN_DEVICE_ERROR;
        return 0;
    }
}

cldnn_engine cldnn_create_engine(/*cldnn_engine_type*/ int32_t type, uint32_t engine_num, const cldnn_engine_configuration* configuration, cldnn_status* status)
{
    if (engine_num > 0 || (type != cldnn_engine_type::cldnn_engine_ocl))
    {
        if (status)
            *status = CLDNN_DEVICE_ERROR;
        return nullptr;
    }

    return exception_handler<cldnn_engine>(CLDNN_ERROR, status, nullptr, [&]()
    {
        return api_cast(new cldnn::engine_impl(configuration ? cldnn::engine_configuration(*configuration) : cldnn::engine_configuration()));
    });
}

void cldnn_retain_engine(cldnn_engine engine, cldnn_status* status)
{
        exception_handler(CLDNN_ERROR, status, [&]() 
        { 
            SHOULD_NOT_BE_NULL(engine, "Engine");
            api_cast(engine)->add_ref(); 
        });
}

void cldnn_release_engine(cldnn_engine engine, cldnn_status* status)
{
        exception_handler(CLDNN_ERROR, status, [&]() 
        { 
            SHOULD_NOT_BE_NULL(engine, "Engine");
            api_cast(engine)->release(); 
        });
}

cldnn_engine_info cldnn_get_engine_info(cldnn_engine engine, cldnn_status* status)
{
    return exception_handler<cldnn_engine_info>(CLDNN_ERROR, status, { 0, 0, 0, 0, 0, 0 }, [&]() -> cldnn_engine_info
    {
        SHOULD_NOT_BE_NULL(engine, "Engine");
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
        SHOULD_NOT_BE_NULL(engine, "Engine");
        return static_cast<int32_t>(api_cast(engine)->type());
    });
}

cldnn_event cldnn_create_user_event(cldnn_engine engine, cldnn_status* status)
{
    return exception_handler<cldnn_event>(CLDNN_ERROR, status, nullptr, [&]()
    {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        return api_cast(api_cast(engine)->create_user_event());
    });
}

void cldnn_retain_event(cldnn_event event, cldnn_status* status)
{

    exception_handler(CLDNN_ERROR, status, [&]() 
    { 
        SHOULD_NOT_BE_NULL(event, "Event");
        api_cast(event)->add_ref(); 
    });
}

void cldnn_release_event(cldnn_event event, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]() 
    { 
        SHOULD_NOT_BE_NULL(event, "Event");
        api_cast(event)->release(); 
    });
}

void cldnn_wait_for_event(cldnn_event event, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(event, "Event");
        api_cast(event)->wait();
    });
}

void cldnn_set_event(cldnn_event event, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(event, "Event");
        api_cast(event)->set();
    });
}

void cldnn_add_event_handler(cldnn_event event, cldnn_event_handler handler, void* param, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(handler, "Handler");
        SHOULD_NOT_BE_NULL(event,   "Event");
        api_cast(event)->add_event_handler(handler, param);
    });
}

void cldnn_get_event_profiling_info(cldnn_event event, cldnn_profiling_interval* profiling, size_t size, size_t* size_ret, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(event, "Event");
        auto& profiling_info = api_cast(event)->get_profiling_info();
        SHOULD_NOT_EQUAL_0(profiling_info.size(), "Profiling info of event");
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

cldnn_program cldnn_build_program(cldnn_engine engine, cldnn_topology topology, cldnn_build_option* options, size_t options_num, cldnn_status* status)
{
    return exception_handler<cldnn_program>(CLDNN_ERROR, status, nullptr, [&]()
    {
        SHOULD_NOT_BE_NULL(engine,   "Engine");
        SHOULD_NOT_BE_NULL(topology, "Topology");
        cldnn::build_options options_obj(cldnn::array_ref<cldnn_build_option>(options, options_num));
        return api_cast(api_cast(engine)->build_program(*api_cast(topology), options_obj));
    });
}

void cldnn_retain_program(cldnn_program program, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(program, "Program");
        api_cast(program)->add_ref();
    });
}

void cldnn_release_program(cldnn_program program, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(program, "Program");
        api_cast(program)->release();
    });
}

cldnn_network cldnn_allocate_network(cldnn_program program, cldnn_status* status)
{
    return exception_handler<cldnn_network>(CLDNN_ERROR, status, nullptr, [&]()
    {
        SHOULD_NOT_BE_NULL(program, "Program");
        return api_cast(api_cast(program)->get_engine()->allocate_network(api_cast(program)));
    });
}

cldnn_network cldnn_build_network(cldnn_engine engine, cldnn_topology topology, cldnn_build_option* options, size_t options_num, cldnn_status* status)
{
    cldnn_program program = cldnn_build_program(engine, topology, options, options_num, status);
    if (!program)
        return nullptr;

    cldnn_network network = cldnn_allocate_network(program, status);
    cldnn_release_program(program, nullptr);
    return network;
}
void cldnn_retain_network(cldnn_network network, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(network, "Network");
        api_cast(network)->add_ref();
    });
}

void cldnn_release_network(cldnn_network network, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(network, "Network");
        api_cast(network)->release();
    });
}

void cldnn_set_network_input(cldnn_network network, cldnn_primitive_id id, cldnn_memory mem, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        auto mem_size = api_cast(mem)->size();
        SHOULD_NOT_BE_NULL(network,     "Network");
        SHOULD_NOT_BE_NULL(id,          "Id");
        SHOULD_NOT_BE_NULL(mem,         "Mem");
        SHOULD_NOT_EQUAL_0(mem_size,    "Memory size");
        api_cast(network)->set_input_data(id, api_cast(mem));
    });
}

cldnn_engine cldnn_get_network_engine(cldnn_network network, cldnn_status* status)
{
    return exception_handler<cldnn_engine>(CLDNN_ERROR, status, nullptr, [&]()
    {
        SHOULD_NOT_BE_NULL(network, "Network");
        auto engine_ptr = api_cast(network)->get_engine();
        if (!engine_ptr) throw std::logic_error("no assigned engine");
        return api_cast(engine_ptr.detach());
    });
}

cldnn_program cldnn_get_network_program(cldnn_network network, cldnn_status* status)
{
    return exception_handler<cldnn_program>(CLDNN_ERROR, status, nullptr, [&]()
    {   
        SHOULD_NOT_BE_NULL(network, "Network");
        auto program_ptr = api_cast(network)->get_program();
        if(!program_ptr) throw std::logic_error("no assigned program");
        return api_cast(const_cast<cldnn::program_impl*>(program_ptr.detach()));
    });
}

void cldnn_get_network_output_names(cldnn_network network, char* names, size_t size, size_t* size_ret, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        auto output_size = api_cast(network)->get_output_ids().size();
        SHOULD_NOT_BE_NULL(network,        "Network");
        SHOULD_NOT_EQUAL_0(output_size, "Output size");
        auto&& output_ids = api_cast(network)->get_output_ids();
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
        SHOULD_NOT_BE_NULL(network, "Network");
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
        SHOULD_NOT_BE_NULL(network, "Network");
        SHOULD_NOT_BE_NULL(name,    "ID of primitive");
        cldnn::primitive_id id(name);
        auto event = api_cast(network)->get_primitive_event(id);
        auto mem_result = api_cast(network)->get_primitive(id)->output_memory().get();
        api_cast(mem_result)->add_ref();
        return{ api_cast(event.detach()), mem_result };
    });
}

cldnn_memory cldnn_allocate_memory(cldnn_engine engine, cldnn_layout layout, cldnn_status* status)
{
    return exception_handler<cldnn_memory>(CLDNN_ERROR, status, nullptr, [&]()
    {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        if (layout.size.format < cldnn_format_any || layout.size.format >= cldnn_format_format_num)
            throw std::invalid_argument("Unknown format of layout.");
        if (layout.data_type != cldnn_data_type::cldnn_f16 &&
            layout.data_type != cldnn_data_type::cldnn_f32 &&
            layout.data_type != cldnn_data_type::cldnn_f64 &&
            layout.data_type != cldnn_data_type::cldnn_i8  &&
            layout.data_type != cldnn_data_type::cldnn_i16 &&
            layout.data_type != cldnn_data_type::cldnn_i32 &&
            layout.data_type != cldnn_data_type::cldnn_i64)
            throw std::invalid_argument("Unknown data_type of layout.");
        return api_cast(api_cast(engine)->allocate_buffer(layout));
    });
}

cldnn_memory cldnn_attach_memory(cldnn_layout layout, void* pointer, size_t size, cldnn_status* status)
{
    return exception_handler<cldnn_memory>(CLDNN_ERROR, status, nullptr, [&]()
    {
        cldnn::layout layout_obj(layout);
        if (layout_obj.data_size() > size) 
            std::invalid_argument("buffer size does not match layout size");
        return api_cast(new cldnn::simple_attached_memory(layout_obj, pointer));
    });
}

void cldnn_retain_memory(cldnn_memory memory, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        api_cast(memory)->add_ref();
    });
}

void cldnn_release_memory(cldnn_memory memory, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        api_cast(memory)->release();
    });
}

void* cldnn_lock_memory(cldnn_memory memory, cldnn_status* status)
{
    return exception_handler<void*>(CLDNN_ERROR, status, nullptr, [&]()
    {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        return api_cast(memory)->lock();
    });
}

void cldnn_unlock_memory(cldnn_memory memory, cldnn_status* status)
{
    exception_handler(CLDNN_ERROR, status, [&]()
    {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        api_cast(memory)->unlock();
    });
}

cldnn_layout cldnn_get_memory_layout(cldnn_memory memory, cldnn_status* status)
{
    cldnn_layout error_result = cldnn::layout(cldnn::data_types::f32, { cldnn::format::x, {0} });

    return exception_handler<cldnn_layout>(CLDNN_ERROR, status, error_result, [&]()
    {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        auto memory_size = api_cast(memory)->size();
        SHOULD_NOT_EQUAL_0(memory_size, "Memory size");
        return api_cast(memory)->get_layout();
    });
}

cldnn_engine cldnn_get_memory_engine(cldnn_memory memory, cldnn_status* status)
{
    return exception_handler<cldnn_engine>(CLDNN_ERROR, status, nullptr, [&]()
    {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        auto engine_ptr = api_cast(memory)->get_engine();
        if (!engine_ptr) throw std::logic_error("no assigned engine");
        return api_cast(engine_ptr.detach());
    });
}

const char* cldnn_get_last_error_message()
{
    try {
        return cldnn::last_err::instance().get_last_error_message().c_str();
    }
    catch (...)
    {
        return "Reading error message failed.";
    }
}


CLDNN_API uint16_t cldnn_float_to_half(float value, cldnn_status* status)
{
    return exception_handler<uint16_t>(CLDNN_ERROR, status, 0, [&]()
    {
        return float_to_half(value);
    });
}

CLDNN_API float cldnn_half_to_float(uint16_t value, cldnn_status* status)
{
    return exception_handler<float>(CLDNN_ERROR, status, 0.0f, [&]()
    {
        return half_to_float(value);
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
PRIMITIVE_TYPE_ID_CALL_IMPL(deconvolution)
PRIMITIVE_TYPE_ID_CALL_IMPL(depth_concatenate)
PRIMITIVE_TYPE_ID_CALL_IMPL(eltwise)
PRIMITIVE_TYPE_ID_CALL_IMPL(fully_connected)
PRIMITIVE_TYPE_ID_CALL_IMPL(input_layout)
PRIMITIVE_TYPE_ID_CALL_IMPL(normalization)
PRIMITIVE_TYPE_ID_CALL_IMPL(pooling)
PRIMITIVE_TYPE_ID_CALL_IMPL(reorder)
PRIMITIVE_TYPE_ID_CALL_IMPL(scale)
PRIMITIVE_TYPE_ID_CALL_IMPL(softmax)
PRIMITIVE_TYPE_ID_CALL_IMPL(simpler_nms)
PRIMITIVE_TYPE_ID_CALL_IMPL(roi_pooling)
PRIMITIVE_TYPE_ID_CALL_IMPL(prior_box)
PRIMITIVE_TYPE_ID_CALL_IMPL(detection_output)
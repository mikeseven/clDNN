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
#include "event_impl.h"
#include "engine_impl.h"

namespace cldnn
{

//class simple_user_event : public event_impl
//{
//public:
//    simple_user_event() :
//        _is_set(false)
//    {}
//
//    void wait() override
//    {
//        std::unique_lock<std::mutex> lock(_mutex);
//        if (_is_set) return;
//        _cv.wait(lock, [&] {return _is_set; });
//    }
//
//    void set() override
//    {
//        {
//            std::lock_guard<std::mutex> lock(_mutex);
//            _is_set = true;
//        }
//        _cv.notify_all();
//        for (auto& pair : _handlers)
//        {
//            pair.first(pair.second);
//        }
//    }
//
//    void add_event_handler(event::event_handler handler, void* data) override
//    {
//        if (handler == nullptr) throw std::invalid_argument("event handler");
//        _handlers.push_back({ handler, data });
//    }
//
//private:
//    bool _is_set;
//    std::mutex _mutex;
//    std::condition_variable _cv;
//    std::vector<std::pair<event::event_handler, void*>> _handlers;
//};


event::event(const event& other):_impl(other._impl)
{
    _impl->add_ref();
}

event& event::operator=(const event& other)
{
    if (_impl == other._impl) return *this;
    _impl->release();
    _impl = other._impl;
    _impl->add_ref();
    return *this;
}
event::~event()
{
    _impl->release();
}


event_impl* event::create_user_event_impl(const engine& engine, status_t* status) noexcept
{
    try
    {
        if (status)
            *status = CLDNN_SUCCESS;
        return engine.get()->create_user_event();
    }
    catch (...)
    {
        if (status)
            *status = CLDNN_ERROR;
        return nullptr;
    }
}

status_t event::add_event_handler_impl(event_handler handler, void* param) noexcept
{
    try
    {
        _impl->add_event_handler(handler, param);
        return CLDNN_SUCCESS;
    }
    catch(...)
    {
        return CLDNN_ERROR;
    }
}
status_t event::set_impl() noexcept
{
    try
    {
        _impl->set();
        return CLDNN_SUCCESS;
    }
    catch (...)
    {
        return CLDNN_ERROR;
    }
}
status_t event::wait_impl() const noexcept
{
    try
    {
        _impl->wait();
        return CLDNN_SUCCESS;
    }
    catch (...)
    {
        return CLDNN_ERROR;
    }
}
}

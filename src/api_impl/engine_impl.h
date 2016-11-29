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
#pragma once
#include <map>
#include "api/cldnn.hpp"
#include "refcounted_obj.h"

namespace cldnn
{
class primitive;
class engine_impl : public refcounted_obj<engine_impl>
{
public:
    engine_impl(const context& ctx, const engine_configuration& conf):_context(ctx), _configuration(conf)
    {
        
    }

    const context& get_context() const { return _context; }

    buffer* allocate_buffer(layout layout);
    const engine_configuration& configuration() const { return _configuration; }

private:
    context _context;
    engine_configuration _configuration;
};
}
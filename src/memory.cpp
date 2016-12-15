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

#include <functional>
#include <numeric>
#include <atomic>

//todo(usarel) : collapse memory/memory_gpu
#include "gpu/memory_gpu.h"

namespace neural {

memory::arguments::arguments(memory::format::type aformat, const vector<uint32_t>& asize)
    : format(aformat)
    , size(asize)
    
{}

size_t memory::count() const 
{
    return _buffer->size() / traits(argument.format).type->size;
}

size_t memory::size_of(arguments arg) 
{
    return std::accumulate(
        arg.size.raw.begin(),
        arg.size.raw.end(),
        memory::traits(arg.format).type->size,
        std::multiplies<size_t>()
    );
}

std::shared_ptr<memory::buffer> create_buffer(memory::arguments arg)
{
	return std::make_shared<gpu::gpu_buffer>(arg);
}

primitive memory::describe(memory::arguments arg)
{
    auto buffer = create_buffer(arg);
    return new memory(arg, buffer);
}

primitive memory::allocate(memory::arguments arg)
{
    auto buffer = create_buffer(arg);
    return new memory(arg, buffer);
}

} // namespace neural
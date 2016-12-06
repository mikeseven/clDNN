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

memory::arguments::arguments(memory::format::type aformat, const vector<uint32_t>& asize, const vector<uint32_t>& apadding)
    : format(aformat)
    , size(asize)
    
{
    padding = size;
    // convert padding to be the same dimensions as size
    if (size.batch[0] == 1)
    {
        for (uint32_t i = 0; i < size.batch.size(); i++)
        {
            padding.batch[i] = apadding.batch[i];
        }
        for (uint32_t i = 0; i < size.feature.size(); i++)
        {
            padding.feature[i] = apadding.feature[i];
        }
        for (uint32_t i = 0; i < size.spatial.size(); i++)
        {
            padding.spatial[i] = apadding.spatial[i];
        }
    }
    // IMPORTANT!!! batch is not 1 so zero the padding, because we support padding only when batch size == 1
    else
    {
        for (uint32_t i = 0; i < padding.raw.size(); i++)
        {
            padding.raw[i] = 0;
        }
    }
}

size_t memory::count() const 
{
    return _elements_count;
}

size_t memory::size_of(arguments arg) 
{
    neural::vector<uint32_t> padded_size = arg.size;
    for (uint32_t i = 0; i < padded_size.batch.size(); i++)
    {
        padded_size.batch[i] += 2 * arg.padding.batch[i];
    }
    for (uint32_t i = 0; i < padded_size.feature.size(); i++)
    {
        padded_size.feature[i] += 2 * arg.padding.feature[i];
    }
    for (uint32_t i = 0; i < padded_size.spatial.size(); i++)
    {
        padded_size.spatial[i] += 2 * arg.padding.spatial[i];
    }

    return std::accumulate(
        padded_size.raw.begin(),
        padded_size.raw.end(),
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
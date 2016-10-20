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

#include "weights_optimizer.h"

using namespace neural;

Weights_optimizer::Weights_optimizer(bool enabled) :
    enabled(enabled) 
{}

bool Weights_optimizer::needs_optimization(const primitive & prim, file::weights_type type)
{
    const memory& mem = get_memory_primitive(prim);
    if (type == file::weights_type::bias)
    {
        return false;
    }
    else if (type == file::weights_type::mean)
    {
        return false;
    }
    else if (type == file::weights_type::convolution)
    {
        // TODO!!! put better logic here.
        if (mem.argument.format != memory::format::yxio_f32)
        {
            auto reordered_prim = neural::reorder::create(
            {
                memory::format::yxio_f32,
                mem.argument.size,
                prim,
            });
            primitives.push_back(reordered_prim);
            return true;
        }
        return false;
    }
    else if (type == file::weights_type::fully_connected)
    {
        // TODO!!! put better logic here.
        if (memory::traits(mem.argument.format).dimension == 4)
        {
            if (mem.argument.format != memory::format::yxfb_f32)
            {
                auto reordered_prim = neural::reorder::create(
                {
                    memory::format::yxfb_f32,
                    mem.argument.size,
                    prim,
                });
                primitives.push_back(reordered_prim);
                return true;
            }
        }
        else if (memory::traits(mem.argument.format).dimension == 2)
        {
            if (mem.argument.format != memory::format::xb_f32)
            {
                auto reordered_prim = neural::reorder::create(
                {
                    memory::format::xb_f32,
                    mem.argument.size,
                    prim,
                });
                primitives.push_back(reordered_prim);
                return true;
            }
        }
    }
    return false;
}

neural::primitive Weights_optimizer::create_weights_from_file(const std::string& path, neural::file::weights_type type)
{
    neural::primitive prim = file::create({ path, type });
    if (enabled)
    {
        if (needs_optimization(prim, type))
        {
            return primitives.back();
        }
    }
    return prim;
}

void Weights_optimizer::optimize(const neural::worker& worker)
{
    if (enabled)
    {
        for (auto& p : primitives)
        {
            worker.execute(p.work());
        }

        //OCL buffers mapping blocks until all primitives are completed
        primitives.back().output[0].as<const neural::memory&>().pointer<float>();
    }
}

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

weights_optimizer::weights_optimizer(int batch_size, bool enabled, bool use_half) :
    _enabled(enabled), _use_half(use_half), _batch_size(batch_size)
{}

bool weights_optimizer::_needs_optimization(const primitive & prim, file::weights_type type, bool use_half)
{
    const memory& mem = get_memory_primitive(prim);
    if (type == file::weights_type::bias)
    {
        // TODO!!! put better logic here.
        auto expected_mem_format = use_half ? memory::format::x_f16 : memory::format::x_f32;
        if (mem.argument.format != expected_mem_format)
        {
            auto reordered_prim = neural::reorder::create(
            {
                expected_mem_format,
                mem.argument.size,
                prim
            });
            _primitives.push_back(reordered_prim);
            return true;
        }
        return false;
    }
    else if (type == file::weights_type::mean)
    {
        // TODO!!! put better logic here.
        // NOTE: For reorder there is no need to reorder mean again. For mean_subtract the reorder is needed
        //       when data is not in yxfb_f32 or bfyx_f32 formats (yxfb_f16 or bfyx_f16 if use_half is true).
        //
        //       The problem is that we are unable to detect for which primitive these weights are optimized.
        //       Currently mean will not be optimized in any way (mean_subtract is not used in any topology).
        return false;
    }
    else if (type == file::weights_type::convolution)
    {
        // TODO!!! put better logic here.
        auto expected_mem_format = _batch_size == 1 ? ( use_half ? memory::format::yxio_f16 : memory::format::os_iyx_osv16_f32 ) : ( use_half ? memory::format::yxio_f16 : memory::format::yxio_f32);
        if (mem.argument.format != expected_mem_format)
        {
            auto reordered_prim = neural::reorder::create(
            {
                expected_mem_format,
                mem.argument.size,
                prim
            });
            _primitives.push_back(reordered_prim);
            return true;
        }
        return false;
    }
    else if (type == file::weights_type::fully_connected)
    {
        // TODO!!! put better logic here.
        if (memory::traits(mem.argument.format).dimension == 4)
        {
            auto expected_mem_format = use_half ? memory::format::yxfb_f16 : memory::format::yxfb_f32;
            if (mem.argument.format != expected_mem_format)
            {
                auto reordered_prim = neural::reorder::create(
                {
                    expected_mem_format,
                    mem.argument.size,
                    prim
                });
                _primitives.push_back(reordered_prim);
                return true;
            }
        }
        else if (memory::traits(mem.argument.format).dimension == 2)
        {
            auto expected_mem_format = use_half ? memory::format::xb_f16 : memory::format::xb_f32;
            if (mem.argument.format != expected_mem_format)
            {
                auto reordered_prim = neural::reorder::create(
                {
                    expected_mem_format,
                    mem.argument.size,
                    prim
                });
                _primitives.push_back(reordered_prim);
                return true;
            }
        }
    }
    return false;
}

neural::primitive weights_optimizer::create_weights_from_file(
    const std::string& path, neural::file::weights_type type, const boost::optional<bool>& use_half)
{
    neural::primitive prim = file::create({ path, type });
    if (_enabled)
    {
        if (_needs_optimization(prim, type, use_half.value_or(_use_half)))
        {
            return _primitives.back();
        }
    }
    return prim;
}

void weights_optimizer::optimize(const neural::worker& worker)
{
    if (_enabled && !_primitives.empty())
    {
        for (auto& p : _primitives)
        {
            worker.execute(p.work());
        }

        //OCL buffers mapping blocks until all primitives are completed
        _primitives.back().output[0].as<const neural::memory&>().pointer<char>();
    }
}

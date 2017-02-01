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
#include <api/primitives/data.hpp>
#include <api/primitives/reorder.hpp>
#include <boost/filesystem.hpp>

weights_optimizer::weights_optimizer(const cldnn::engine& eng, int batch_size, bool enabled, bool use_half, bool use_bfyx) :
    _enabled(enabled), _use_half(use_half), _use_bfyx(use_bfyx), _batch_size(batch_size), _engine(eng)
{}

cldnn::primitive_id weights_optimizer::_needs_optimization(const cldnn::memory& mem, const cldnn::primitive_id& mem_id, file::weights_type type, bool use_half)
{
    auto reorder_id = std::string("reorder_") + mem_id;
    auto data_type = use_half ? cldnn::data_types::f16 : cldnn::data_types::f32;
    auto input_size = mem.get_layout().size;
    auto expected_mem_size = input_size;

    if (type == file::weights_type::bias)
    {
        // TODO!!! put better logic here.
        expected_mem_size = cldnn::tensor(cldnn::format::x, { static_cast<cldnn::tensor::value_type>(mem.get_layout().count()) });
    }
    else if (type == file::weights_type::mean)
    {
        // TODO!!! put better logic here.
        // NOTE: For reorder there is no need to reorder mean again. For mean_subtract the reorder is needed
        //       when data is not in yxfb_f32 or bfyx_f32 formats (yxfb_f16 or bfyx_f16 if use_half is true).
        //
        //       The problem is that we are unable to detect for which primitive these weights are optimized.
        //       Currently mean will not be optimized in any way (mean_subtract is not used in any topology).
        return mem_id;
    }
    else if (type == file::weights_type::convolution)
    {
        // TODO!!! put better logic here.
        expected_mem_size = _use_bfyx
            ? cldnn::tensor(cldnn::format::os_iyx_osv16,
                {
                    input_size.feature[0], input_size.feature[1], input_size.spatial[0], input_size.spatial[1] // order: "oiyx"
                })
            : cldnn::tensor(cldnn::format::yxio,
                {
                    input_size.spatial[0], input_size.spatial[1], input_size.feature[1], input_size.feature[0]  // order: "yxio"
                });
    }
    else if (type == file::weights_type::fully_connected)
    {
        // TODO!!! put better logic here.
        if (cldnn::neural_memory::traits(mem.get_layout()).dimension == 4)
        {
            if (_use_bfyx && _batch_size >= 8 && !use_half)
            {
                expected_mem_size = cldnn::tensor(cldnn::format::bs_xs_xsv8_bsv8,
                {
                    input_size.batch[0], input_size.feature[0] * input_size.spatial[0] * input_size.spatial[1]
                });
            }
            else
            {
                expected_mem_size = _use_bfyx && _batch_size == 1 && !use_half
                    ? cldnn::tensor(cldnn::format::fyxb,
                    {
                        input_size.feature[0], input_size.spatial[0], input_size.spatial[1], input_size.batch[0] // order: "fyxb"
                    })
                    : cldnn::tensor(cldnn::format::yxfb,
                    {
                        input_size.spatial[0], input_size.spatial[1], input_size.feature[0], input_size.batch[0]  // order: "yxfb"
                    });
            }
        }
        else if (cldnn::neural_memory::traits(mem.get_layout()).dimension == 2)
        {
            expected_mem_size = _use_bfyx && _batch_size >= 8 && !use_half 
                ? cldnn::tensor(cldnn::format::bs_xs_xsv8_bsv8,
                {
                    input_size.batch[0], input_size.spatial[0]  // order: "bs_xs_bsv8_xsv8"
                })
                : cldnn::tensor(cldnn::format::xb,
            {
                input_size.spatial[0], input_size.batch[0]  // order: "xb"
            });
        }
    }
    cldnn::layout expected_mem_layout(data_type, expected_mem_size);

    if (mem.get_layout() != expected_mem_layout)
    {
        _topology.add(cldnn::reorder(reorder_id, mem_id, expected_mem_layout));
        return reorder_id;
    }
    return mem_id;
}

cldnn::primitive_id weights_optimizer::create_weights_from_file(
    const std::string& path, file::weights_type type, const boost::optional<bool>& use_half)
{
    auto mem = file::create({ _engine, path, type });
    auto data_id = boost::filesystem::path(path).filename().string();
    _topology.add(cldnn::data(data_id, mem));
    return _enabled
        ? _needs_optimization(mem, data_id, type, use_half.value_or(_use_half))
        : data_id;
}

auto weights_optimizer::optimize() const -> decltype(cldnn::network(_engine, _topology).execute())
{
    return cldnn::network(_engine, _topology).execute();
}

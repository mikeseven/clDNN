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

using namespace cldnn;

weights_optimizer::weights_optimizer(refcounted_obj_ptr<engine_impl> eng, bool enabled)
    : _enabled(enabled), _topology(new topology_impl(), false), _engine(eng), _outputs()
{
}

cldnn::primitive_id weights_optimizer::_try_optimize(const cldnn::memory& mem, const cldnn::primitive_id& mem_id, unsigned int batch_size)
{
    auto reorder_id = std::string("reorder_") + mem_id;
    auto data_type = mem.get_layout().data_type; //currently weights optimizer shouldn't change data type
    auto input_size = mem.get_layout().size;
    auto expected_mem_size = input_size;

    auto input_format = input_size.format;


    if (mem_id == "imagenet_mean.nnd")
    {
        // TODO!!! put better logic here.
        // NOTE: For reorder there is no need to reorder mean again. For mean_subtract the reorder is needed
        //       when data is not in yxfb_f32 or bfyx_f32 formats (yxfb_f16 or bfyx_f16 if use_half is true).
        //
        //       The problem is that we are unable to detect for which primitive these weights are optimized.
        //       Currently mean will not be optimized in any way (mean_subtract is not used in any topology).
        return mem_id;
    }

    if (input_format == cldnn::format::x) //bias
    {
        // TODO!!! put better logic here.
        expected_mem_size = cldnn::tensor(cldnn::format::x, { static_cast<cldnn::tensor::value_type>(mem.get_layout().count()) });
    }
    else if (input_format == cldnn::format::oiyx || input_format == cldnn::format::yxio) //conv
    {
        // TODO!!! put better logic here.
        expected_mem_size = cldnn::tensor(cldnn::format::os_iyx_osv16,
        {
            input_size.feature[0], input_size.feature[1], input_size.spatial[0], input_size.spatial[1] // order: "oiyx"
        });
    }
    else if (input_format == cldnn::format::bfyx || input_format == cldnn::format::yxfb || input_format == cldnn::format::bx || input_format == cldnn::format::xb) //fc
    {
        // TODO!!! put better logic here.
        if (cldnn::neural_memory::traits(mem.get_layout()).dimension == 4)
        {
            if (batch_size > 1 && data_type != data_types::f16)
            {
                expected_mem_size = cldnn::tensor(cldnn::format::bs_xs_xsv8_bsv8,
                {
                    input_size.batch[0], input_size.feature[0] * input_size.spatial[0] * input_size.spatial[1]
                });
            }
            else
            {
                expected_mem_size = batch_size == 1
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
            expected_mem_size = batch_size >= 8 && data_type != data_types::f16
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
        _topology->add(std::make_shared<cldnn::reorder>(reorder_id, mem_id, expected_mem_layout));
        _outputs.push_back(reorder_id);
        return reorder_id;
    }
    return mem_id;
}

cldnn::primitive_id cldnn::weights_optimizer::add_weights(const std::shared_ptr<const data> data_prim, unsigned int batch_size)
{
    _topology->add(data_prim);
    return _enabled
        ? _try_optimize(data_prim->mem, data_prim->id(), batch_size)
        : data_prim->id();
}

auto weights_optimizer::optimize() const -> deduce_ret_type_t<decltype(&network_impl::get_primitives)>
{
    network_impl net(_engine, _topology, _outputs);
    net.execute(std::vector<refcounted_obj_ptr<event_impl>>());
    return net.get_primitives(net.get_output_ids());
}

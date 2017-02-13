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

layout weights_optimizer::get_expected_layout(const cldnn::memory& mem, weights_type type, std::shared_ptr<const convolution> prim, layout const& output_layout)
{
    auto current_layout = mem.get_layout();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = output_layout.data_type;

    auto batch = output_layout.size.batch[0];

    switch (type)
    {
    case weights_type::bias: //convolution bias
        expected_tensor = cldnn::tensor(cldnn::format::x, { static_cast<tensor::value_type>(current_layout.count()) });
        break;

    case weights_type::weights: //convolution weights
        if (batch == 1 || expected_data_type != data_types::f16)
            expected_tensor = current_layout.size.transform(format::os_iyx_osv16, 1);
        else
            expected_tensor = current_layout.size.transform(format::yxio, 1);

        break;

    default:
        throw std::runtime_error("Unsupported weights type in weights_optimizer::get_expected_layout for convolution primitive");
    }

    return layout(expected_data_type, expected_tensor);
}

layout weights_optimizer::get_expected_layout(const cldnn::memory& mem, weights_type type, std::shared_ptr<const fully_connected> prim, layout const& output_layout)
{
    auto current_layout = mem.get_layout();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = output_layout.data_type;

    auto batch = output_layout.size.batch[0];

    switch (type)
    {
    case weights_type::bias: //fc bias
        expected_tensor = cldnn::tensor(cldnn::format::x, { static_cast<tensor::value_type>(current_layout.count()) });
        break;

    case weights_type::weights: //fc weights
    {
        auto dimensions = neural_memory::traits(mem.get_layout()).dimension;
        if (dimensions == 4)
        {
            if (batch > 1 && expected_data_type != data_types::f16)
            {
                expected_tensor = cldnn::tensor(cldnn::format::bs_xs_xsv8_bsv8,
                {
                    current_layout.size.batch[0], current_layout.size.feature[0] * current_layout.size.spatial[0] * current_layout.size.spatial[1]
                });
            }
            else if (batch == 1)
            {
                expected_tensor = cldnn::tensor(cldnn::format::bs_x_bsv16,
                {
                    current_layout.size.batch[0], current_layout.size.feature[0] * current_layout.size.spatial[0] * current_layout.size.spatial[1]
                });
                // TODO: Check is there is no preformance regression for FP32 and, if not, remove these comments.
                //expected_tensor = current_layout.size.transform(format::fyxb, 1);
            }
            else
                expected_tensor = current_layout.size.transform(format::yxfb, 1);
        }
        else if (dimensions == 2)
        {
            if (batch >= 8 && expected_data_type != data_types::f16)
                expected_tensor = current_layout.size.transform(format::bs_xs_xsv8_bsv8, 1);
            else if (batch == 1)
                expected_tensor = current_layout.size.transform(format::bs_x_bsv16, 1);
            else
                expected_tensor = current_layout.size.transform(format::xb, 1);
        }

        break;
    }

    default:
        throw std::runtime_error("Unsupported weights type in weights_optimizer::get_expected_layout for fully-connected primitive");
    }

    return layout(expected_data_type, expected_tensor);
}

bool weights_optimizer::add_reorder_if_needed(const cldnn::memory& mem, const cldnn::primitive_id& memid, layout const& expected_layout)
{
    if (mem.get_layout() != expected_layout)
    {
        auto reorder_id = "reorder_" + memid;
        _topology->add(std::make_shared<cldnn::reorder>(reorder_id, memid, expected_layout));
        _outputs.push_back(reorder_id);
        return true;
    }

    return false;
}

auto weights_optimizer::optimize() const -> meta::deduce_ret_type_t<decltype(&network_impl::get_primitives)>
{
    if (!_enabled)
        return meta::deduce_ret_type_t<decltype(&network_impl::get_primitives)>();

    network_impl net(_engine, _topology, _outputs);
    net.execute(std::vector<refcounted_obj_ptr<event_impl>>());
    return net.get_primitives(net.get_output_ids());
}

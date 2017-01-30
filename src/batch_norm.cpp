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

#include "batch_norm_arg.h"
#include "primitive_type_base.h"
#include "network_impl.h"

namespace cldnn
{
    primitive_type_id batch_norm_type_id()
    {
        static primitive_type_base<batch_norm, batch_norm_arg> instance;
        return &instance;
    }

    layout batch_norm_arg::calc_output_layout(const topology_map& topology_map, std::shared_ptr<const batch_norm> desc)
    {
        auto input_desc = topology_map.at(desc->input()[0])->primitive_desc;
        auto result = input_desc->type()->calc_output_layout(topology_map, input_desc);
        return result;
    }

    batch_norm_arg::batch_norm_arg(network_impl& network, std::shared_ptr<const batch_norm> desc)
        :primitive_arg_base(network, desc, calc_output_layout(network.get_topology()->get_primitives(), desc))
    {
        auto input_format = input_memory(0).get_layout().size.format;
        auto output_format = output_memory().get_layout().size.format;
        auto mean_format = mean_memory().get_layout().size.format;
        auto variance_format = variance_memory().get_layout().size.format;

        if (input_format != format::yxfb)
        {
            throw std::runtime_error("Batch norm input is not in yxfb format!");
        }
        if (output_format != format::yxfb)
        {
            throw std::runtime_error("Batch norm output is not in yxfb format!");
        }
        if (mean_format != format::yxfb &&
            mean_format != format::bfyx)
        {
            throw std::runtime_error("Mean is not in yxfb or bfyx format!");
        }
        if (variance_format != format::yxfb &&
            variance_format != format::bfyx)
        {
            throw std::runtime_error("Variance is not in yxfb or bfyx format!");
        }
    }

    const memory& batch_norm_arg::mean_memory() const
    {
        return _network.get_primitive(argument.mean)->output_memory();
    }

    const memory& batch_norm_arg::variance_memory() const
    {
        return _network.get_primitive(argument.variance)->output_memory();
    }

    const bool& batch_norm_arg::use_global_stats() const
    {
        return argument.use_global_stats;
    }
}
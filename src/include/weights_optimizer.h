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

#pragma once


#include "network_impl.h"
#include "memory_impl.h"

#include <boost/optional.hpp>

#include <vector>

namespace cldnn
{
// TODO!!! - optimize weights based on HW
class weights_optimizer
{
public:
    enum class weights_type
    {
        mean,
        convolution,
        bias,
        fully_connected
    };

    bool _enabled;
    refcounted_obj_ptr<topology_impl> _topology;
    refcounted_obj_ptr<engine_impl> _engine;
    std::vector<primitive_id> _outputs;

    //this function returns either mem_id directly, if mem does not need optimization, or id of a reorder which tooks mem as input and returns it in optimizied format
    cldnn::primitive_id _try_optimize(const cldnn::memory& mem, const cldnn::primitive_id& mem_id, weights_type type, unsigned int batch_size);

public:
    explicit weights_optimizer(refcounted_obj_ptr<engine_impl> eng, bool enabled = true);

    cldnn::primitive_id add_weights(const std::shared_ptr<const data> data_prim, weights_type type, unsigned int batch_size);

    auto optimize() const -> decltype(network_impl(_engine, _topology, _outputs).execute(std::vector<refcounted_obj_ptr<event_impl>>()));
    auto get_engine() { return _engine; }
};
}

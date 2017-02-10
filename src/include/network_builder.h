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
#include "engine_impl.h"
#include "network_impl.h"
#include "topology_impl.h"
#include "weights_optimizer.h"

namespace cldnn
{

class network_builder
{
public:
    network_builder(refcounted_obj_ptr<engine_impl> eng, const build_options& options);
    network_impl* build_network(refcounted_obj_ptr<topology_impl> tpl);
    const refcounted_obj_ptr<engine_impl>& get_engine() const { return _engine; }

private:
    enum class input_type
    {
        image,          //input_layout is used for network input, i.e. image to classify
        weights         //input_layout is used for weights, i.e. weights or bias for convolution/fc
    };

    struct input_info
    {
        input_type type;
        std::vector<primitive_id> users;
    };

    std::map<primitive_id, input_info> inputs;

    void add_input(primitive_id const& id, input_type type, primitive_id const& user);

private:
    const refcounted_obj_ptr<engine_impl> _engine;
    build_options _options;
    topology_map _topology_map;

    layout_optimizer _lo;

    void optimize_topology();

    // Prepares output padding for primitives
    // TODO: case when input primitive is used by multiple primitives
    void prepare_padding();

    void reorder_inputs();
    void optimize_weights();

    void add_if_new(std::pair<std::shared_ptr<const reorder>, bool> const& reorder_from_optimizer);
};
}

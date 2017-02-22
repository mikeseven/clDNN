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
#include "layout_optimizer.h"

namespace cldnn
{

class network_builder
{
public:
    network_builder(refcounted_obj_ptr<engine_impl> eng, const build_options& options);
    network_impl* build_network(refcounted_obj_ptr<topology_impl> tpl);
    const refcounted_obj_ptr<engine_impl>& get_engine() const { return _engine; }

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

    void add_if_new(std::pair<std::shared_ptr<reorder>, bool> const& reorder_from_optimizer);
};
}

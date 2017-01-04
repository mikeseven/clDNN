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


#include "api/network.hpp"
#include "api/memory.hpp"
#include "file.h"
#include <boost/optional.hpp>


// TODO!!! - optimize weights based on HW
class weights_optimizer
{
    bool _enabled;
    bool _use_half;
    int _batch_size;
    cldnn::topology _topology;
    cldnn::engine _engine;

    cldnn::primitive_id _needs_optimization(const cldnn::memory& mem, const cldnn::primitive_id& mem_id,
                             file::weights_type type,
                             bool use_half);

public:
    explicit weights_optimizer(const cldnn::engine& eng, int batch_size, bool enabled = true,
                               bool use_half = false);

    cldnn::primitive_id create_weights_from_file(const std::string& path,
                                               file::weights_type type,
                                               const boost::optional<bool>& use_half = boost::none);

    auto optimize() const -> decltype(cldnn::network(_engine, _topology).execute());
    cldnn::engine get_engine() { return _engine; }
};
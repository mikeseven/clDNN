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
#include "api/primitives/batch_norm.hpp"
#include "primitive_arg.h"
#include <memory>

namespace cldnn
{
    class batch_norm_arg : public primitive_arg_base<batch_norm>
    {
    public:
        batch_norm_arg(network_impl& network, std::shared_ptr<const batch_norm> desc);

        static layout calc_output_layout(network_impl& network, std::shared_ptr<const batch_norm> desc);

        const memory& mean_memory() const;
        const memory& variance_memory() const;
        const bool& use_global_stats() const;
    };
}

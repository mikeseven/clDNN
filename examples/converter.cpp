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

#include "api/CPP/memory.hpp"
#include "api/CPP/network.hpp"
#include "file.h"
#include "common/common_tools.h"
#include <api/CPP/data.hpp>
#include <api/CPP/reorder.hpp>

// memory->memory convolution
void convert_weights(cldnn::data_types dt, cldnn::format::type format, std::string convertion_path)
{
    using namespace cldnn;
    if (format >= format::format_num) throw std::runtime_error("format is out of range");
    std::vector<std::string> weights = get_directory_weights("weights");
    cldnn::engine engine;
    for (const auto& w : weights)
    {
        if (w.find(convertion_path) != std::string::npos)
        {
            auto mem = file::read({ engine, w.c_str() });
            layout output_layout{
                dt,
                mem.get_layout().format,
                mem.get_layout().size
            };

            topology topology(data("input", mem), reorder("reorder", "input", output_layout));

            auto output = network(engine, topology).execute().at("reorder").get_memory();

            file::serialize(output, w);
        }
    }

}

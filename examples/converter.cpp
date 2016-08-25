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

#include "api/neural.h"
#include "common/common_tools.h"
#include "memory_utils.h"

// memory->memory convolution
void convert_weights(neural::memory::format::type format, std::string convertion_path)
{
    using namespace neural;
    if (format >= memory::format::format_num) throw std::runtime_error("format is out of range");
    std::vector<std::string> weights = get_directory_weights("weights");
    for (const auto& w : weights)
    {
        if (w.find(convertion_path) != std::string::npos)
        {
            auto mem = file::create({ engine::cpu, w.c_str() });
            auto reordered_mem = reorder::create(
            {
                engine::cpu,
                (format),
                mem.as<const memory&>().argument.size, // do not resize
                mem
            });
            execute({ reordered_mem }).wait();
            file::serialize(reordered_mem.output[0], w);
        }
    }

}

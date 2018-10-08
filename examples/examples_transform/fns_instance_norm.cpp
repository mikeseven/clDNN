// Copyright (c) 2018 Intel Corporation
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


#include "topologies.h"

#include <api/CPP/input_layout.hpp>
#include <api/CPP/reorder.hpp>
#include <api/CPP/permute.hpp>

#include <string>

#include "file_system_utils.h"
#include "file.h"


using namespace cldnn;
using namespace cldnn::utils::examples;


topology build_fns_instance_norm(const std::string& weights_dir, const engine& engine, layout& input_layout,
                                 const bool mean_subtract)
{
    // Set default spatial size if it is not provided before.
    if (input_layout.size.spatial[0] <= 0 || input_layout.size.spatial[1] <= 0)
    {
        input_layout.size.spatial[0] = 224;
        input_layout.size.spatial[1] = 224;
    }

    const auto input = cldnn::input_layout("input", input_layout);
    topology topology_inst{input};

    primitive_id corrected_input = input;
    if (mean_subtract)
    {
        // Subtract mean values if necessary.
        const auto reorder_mean    = file::create({engine, join_path(weights_dir, "fns_instance_norm_mean.nnd")});
        const auto reordered_input = reorder(
            "img_reorder",
            input,
            {input_layout.data_type, format::bfyx, input_layout.size},
            reorder_mean);
        topology_inst.add(reorder_mean, reordered_input);
        corrected_input = reordered_input;
    }
    else
    {
        const auto reordered_input = reorder("img_reorder", input, format::bfyx, input_layout.data_type);
        topology_inst.add(reordered_input);
        corrected_input = reordered_input;
    }

    const auto img_transpose = permute("img_transpose", corrected_input, {0, 1, 3, 2});
    const auto img_output = reorder("output", img_transpose, format::byxf, input_layout.data_type);

    topology_inst.add(img_transpose, img_output);
    return topology_inst;
}

// Copyright (c) 2016, 2018 Intel Corporation
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


#include <api/CPP/data.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/reorder.hpp>

#include "common_tools.h"
#include "file.h"


/// @brief Converts weights to new data type and layout. Weights are assumed to be
///        in "weights" directory.
///
/// @details Converter does not provide quantization support for integral data types.
///          Weights are converted in-place.
///
/// @param data_type        Data type to which weights will be converted.
/// @param format_type      Layout format to which weights will be converted.
/// @param conv_path_filter Weights path include filter (the string must match part of
///                         weights file path to be considered for conversion).
///                         If empty string is specified, all weights are converted.
/// @param validate_magic   Indicates that magic identifier of .nnd file should be checked
///                         for validity.
void convert_weights(cldnn::data_types data_type, cldnn::format::type format_type,
                     const std::string& conv_path_filter = "", bool validate_magic = false)
{
    using namespace cldnn;

    if (format_type >= format::format_num)
        throw std::runtime_error("Specified layout format is not supported in converter.");

    engine engine;
    std::vector<std::string> weights = get_directory_weights("weights");
    for (const auto& w : weights)
    {
        if (w.find(conv_path_filter) != std::string::npos)
        {
            auto mem = file::read({engine, w}, validate_magic);
            layout output_layout{
                data_type,
                mem.get_layout().format,
                mem.get_layout().size
            };

            topology topology(data("input", mem), reorder("reorder", "input", output_layout));

            auto output = network(engine, topology).execute().at("reorder").get_memory();

            file::serialize(output, w);
        }
    }
}

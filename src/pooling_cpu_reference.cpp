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

#include "pooling_cpu_reference.h"
#include "multidimensional_counter.h"
#include "implementation_map.h"

namespace neural {

    pooling_cpu_reference::pooling_cpu_reference(pooling &arg)
        : is_an_implementation(neural::type_id<pooling_cpu_reference>())
        , outer(arg) {};
    pooling_cpu_reference::~pooling_cpu_reference() {};
    void pooling_cpu_reference::implementation(const void *ptr) {
            auto this_pooling = static_cast<const pooling *>(ptr);
            auto input = this_pooling->argument.input[0].primitive.as<const memory&>().pointer<float>();
            auto output = this_pooling->argument.output[0].as<const memory&>().pointer<float>();

            auto& input_arg = this_pooling->argument.input[0].primitive.as<const memory&>().argument;

            auto& input_buffer_size = input_arg.size;
            auto& input_offset = this_pooling->argument.input_offset;

            auto& output_arg = this_pooling->argument.output[0].as<const memory&>().argument;
            auto& output_buffer_size = output_arg.size;
            auto& output_offset = this_pooling->argument.output_offset;
            auto& output_size = this_pooling->argument.output_size;

            auto& stride  = this_pooling->argument.stride;
            auto& window  = this_pooling->argument.size;
            auto& padding = this_pooling->argument.padding;

            if (padding::zero != padding)                                      throw std::runtime_error("Pooling support only zero padding.");
            if (input_arg.format != memory::format::yxfb_f32)                  throw std::runtime_error("Pooling reference uses yxfb_f32 format."); //todo, only this format?
            if (input_buffer_size.raw.size() != output_buffer_size.raw.size()) throw std::runtime_error("Pooling input/output number of dimension does not match.");
            if (stride.raw.size() != output_buffer_size.raw.size())            throw std::runtime_error("Pooling stride/output number of dimension does not match.");
            if (window.raw.size() != output_buffer_size.raw.size())            throw std::runtime_error("Pooling window_size/output number of dimension does not match.");
            if (input_arg.format != output_arg.format)                         throw std::runtime_error("Pooling input/output data format does not match.");

            // general formula: output size = (input size - window size) / step + 1
            for (size_t i = 0; i < input_offset.raw.size(); ++i) {
                if (output_buffer_size.raw[i] < output_size.raw[i] + output_offset.raw[i])
                    throw std::runtime_error("Pooling output buffer size is to small.");
            }

            namespace nd = ndimensional;
            nd::value<uint32_t> range(output_size);
            nd::value<uint32_t> window_range(window);
            auto calc_in_idx = nd::choose_calculate_idx(input_arg.format);
            auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);
            switch (this_pooling->argument.mode) {
            case pooling::mode::max:
                for (auto pos : range) {
                    auto out_idx = calc_out_idx(output_arg.size.raw, pos + output_offset);

                    float acc = -std::numeric_limits<float>::max();
                    for (auto win_pos : window_range) {
                        const std::vector<int32_t> arg_in_idx = nd::value<int32_t>(input_offset) + pos*stride + win_pos;

                        if (nd::is_out_of_range(input_arg.size, arg_in_idx))
                        {
                            // Pad with zero.
                            acc = std::max(acc, 0.0f);
                            continue;
                        }

                        auto in_idx = calc_in_idx(input_arg.size.raw, { arg_in_idx.begin(), arg_in_idx.end() });
                        acc = std::max(acc, input[in_idx]);
                    }
                    output[out_idx] = acc;
                }
                break;
            case pooling::mode::average:
            {
                auto window_elements = std::accumulate(window.raw.cbegin(), window.raw.cend(), 1, std::multiplies<uint32_t>());
                for (auto pos : range) {
                    auto out_idx = calc_out_idx(output_arg.size.raw, pos + output_offset);

                    float acc = 0.0f;
                    for (auto win_pos : window_range) {
                        const std::vector<int32_t> arg_in_idx = nd::value<int32_t>(input_offset) + pos*stride + win_pos;

                        if (nd::is_out_of_range(input_arg.size, arg_in_idx))
                            continue;

                        auto in_idx = calc_in_idx(input_arg.size.raw, { arg_in_idx.begin(), arg_in_idx.end() });
                        acc += input[in_idx];
                    }
                    output[out_idx] = acc / window_elements;
                }
            }
            break;
            default:
                throw std::runtime_error("Unknown pooling mode.");
            }
    };


namespace
{

    struct attach
    {
        attach()
        {
            auto key_fw = std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32);
            auto val_fw = pooling_cpu_reference::create;
            implementation_map<pooling>::add(key_fw, val_fw);
        }

        ~attach()
        {
        }
    };

#ifdef __GNUC__
    __attribute__((visibility("default")))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}
}

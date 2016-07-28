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
#include "api/neural.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"
#include "fully_connected_relu.h"

namespace neural {

    struct fully_connected_relu_reference : is_an_implementation {
        const fully_connected_relu &outer;
        fully_connected_relu_reference(fully_connected_relu &arg)
            : is_an_implementation(neural::type_id<fully_connected_relu_reference>())
            , outer(arg)
        {};
        ~fully_connected_relu_reference() {}

        static void implementation(const void *ptr) {
            auto this_fc = static_cast<const fully_connected_relu *>(ptr);
            auto input = this_fc->input_memory(0).pointer<float>();
            auto output = this_fc->output_memory(0).pointer<float>();
            auto weight = this_fc->input_memory(1).pointer<float>();
            auto& weight_buffer_size = this_fc->input_memory(1).argument.size;
            auto bias = this_fc->argument.input[2].primitive.as<const memory&>().pointer<float>();

            auto& input_arg = this_fc->input_memory(0).argument;
            auto& input_buffer_size = input_arg.size;

            auto& output_arg = this_fc->output_memory(0).argument;
            auto& output_buffer_size = output_arg.size;

            auto& weight_arg = this_fc->input_memory(1).argument;

            assert(1 == input_buffer_size.feature.size());
            assert(1 == input_buffer_size.batch.size());
            assert(1 == input_buffer_size.feature[0]);

            namespace nd = ndimensional;
            fill(this_fc->output_memory(0), 0.0f);

            const int DATA_INDEX = 2;
            const int BATCH_INDEX = 0;

            nd::value<uint32_t> range_output(output_buffer_size);
            range_output[BATCH_INDEX] = 1; //in every iteration whole batch is computed at once, so it has to be removed from the range
            nd::value<uint32_t> range_input(input_buffer_size);
            nd::value<uint32_t> range_weight(weight_buffer_size);

            auto calc_in_idx = nd::choose_calculate_idx(input_arg.format);
            auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);
            auto calc_w_idx = nd::choose_calculate_idx(weight_arg.format);

            std::vector<uint32_t> arg_weight_idx(3);
            for (auto pos_out : range_output) {
                auto out_idx = calc_out_idx(output_arg.size.raw, pos_out);

                for (auto pos_in : range_input) {
                    auto in_idx = calc_in_idx(input_arg.size.raw, pos_in);

                    arg_weight_idx[DATA_INDEX] = pos_out[DATA_INDEX];
                    arg_weight_idx[BATCH_INDEX] = pos_in[DATA_INDEX];
                    auto w_idx = calc_w_idx(weight_arg.size.raw, arg_weight_idx);
                    output[out_idx + pos_in[BATCH_INDEX]] += input[in_idx] * weight[w_idx];
                }
                for (auto b = 0u; b < range_input[BATCH_INDEX]; b++) {
                    output[out_idx + b] += bias[pos_out[DATA_INDEX]];
                    output[out_idx + b] = std::max(output[out_idx + b], 0.0f) +
                        this_fc->argument.negative_slope*std::min(0.0f, output[out_idx + b]);
                }
            }
        }

        task_group work() override {
            return{ { task{ implementation, &outer } }, schedule::single };
        }

        static is_an_implementation *create(fully_connected_relu &arg) { return new fully_connected_relu_reference(arg); };
    };

namespace {
    struct attach {
        attach() {
            auto val_fw = fully_connected_relu_reference::create;
            fully_connected_relu_fw_implementation_map::instance().insert({ std::make_tuple(engine::reference, memory::format::xb_f32, memory::format::xb_f32), val_fw });
            fully_connected_relu_fw_implementation_map::instance().insert({ std::make_tuple(engine::reference, memory::format::x_f32,  memory::format::x_f32), val_fw });
            fully_connected_relu_fw_implementation_map::instance().insert({ std::make_tuple(engine::reference, memory::format::yxfb_f32,  memory::format::xb_f32), val_fw });
        }
        ~attach() {}
    };

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
    attach attach_impl;

}
}
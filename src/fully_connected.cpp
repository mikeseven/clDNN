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

namespace neural {

struct fully_connected_reference : is_an_implementation {
    const fully_connected &outer;
    fully_connected_reference(fully_connected &arg)
        : is_an_implementation(neural::type_id<fully_connected_reference>())
        , outer(arg)
    {};
    ~fully_connected_reference() {}

    static void implementation(const void *ptr) {
        auto this_fc = static_cast<const fully_connected *>(ptr);

        auto& input_mem = this_fc->input_memory(0);
        auto& output_mem = this_fc->output_memory(0);
        auto& weight_mem = this_fc->input_memory(1);
        auto& bias_mem = this_fc->input_memory(2);
        
        float *input, *output, *weight, *bias;
        auto& weight_buffer_size = this_fc->input_memory(1).argument.size;

        auto& input_arg = this_fc->input_memory(0).argument;
        auto& input_buffer_size = input_arg.size;

        auto& output_arg = this_fc->output_memory(0).argument;
        auto& output_buffer_size = output_arg.size;

        assert( 1 == input_buffer_size.feature.size());
        assert( 1 == input_buffer_size.batch.size()  );
        assert( 1 == input_buffer_size.feature[0]    );

        namespace nd = ndimensional;
        fill(this_fc->output_memory(0), 0.0f);

        const int DATA_INDEX = 2;
        const int BATCH_INDEX = 0;

        nd::value<uint32_t> range_output(output_buffer_size);
        range_output[BATCH_INDEX] = 1; //in every iteration whole batch is computed at once, so it has to be removed from the range
        nd::value<uint32_t> range_input(input_buffer_size);
        nd::value<uint32_t> range_weight(weight_buffer_size);

        auto calc_in_ptr = nd::choose_calculate_ptr(input_mem);
        auto calc_out_ptr = nd::choose_calculate_ptr(output_mem);
        auto calc_w_ptr = nd::choose_calculate_ptr(weight_mem);
        auto calc_bias_ptr = nd::choose_calculate_ptr(bias_mem);

        std::vector<uint32_t> arg_weight_idx(3);
        nd::value<uint32_t> batch_offset(range_output.size());

        for (auto pos_out : range_output) {

            for (auto pos_in : range_input) {
                input = static_cast<float*>(calc_in_ptr(input_mem, pos_in));
                batch_offset[BATCH_INDEX] = pos_in[BATCH_INDEX];

                arg_weight_idx[DATA_INDEX] = pos_out[DATA_INDEX];
                arg_weight_idx[BATCH_INDEX] = pos_in[DATA_INDEX];

                weight = static_cast<float*>(calc_w_ptr(weight_mem, arg_weight_idx));
                output = static_cast<float*>(calc_out_ptr(output_mem, pos_out + batch_offset));

                *output += *input * *weight;
            }
            for (auto b = 0u; b < range_input[BATCH_INDEX]; b++) {
                batch_offset[BATCH_INDEX] = b;
                output = static_cast<float*>(calc_out_ptr(output_mem, pos_out + batch_offset));
                bias = static_cast<float*>(calc_bias_ptr(bias_mem,{0, 0, pos_out[DATA_INDEX]}));

                *output += *bias;
            }
        }
    }

    task_group work() {
        return {{task{ implementation, &outer}}, schedule::single};
    }

    static is_an_implementation *create(fully_connected &arg) { return new fully_connected_reference(arg); };
};

//                                    engine                output                        input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(fully_connected &)>> implementation_map = {
    { std::make_tuple(engine::reference, memory::format::xb_f32, memory::format::xb_f32), fully_connected_reference::create },
    { std::make_tuple(engine::reference, memory::format::x_f32,  memory::format::x_f32),  fully_connected_reference::create }
};

fully_connected::arguments::arguments( neural::engine::type eng,
                                       primitive            out,
                                       primitive            in,
                                       primitive            weights,
                                       primitive            bias)
: engine(eng)
, output({out})
, input({in, weights, bias})
{
};

// creates primitive with fully_connected implementation that supports provided arguments
primitive fully_connected::create(fully_connected::arguments arg) {
    auto& input_arg = arg.input[0].primitive.as<const memory&>().argument;
    auto& output_arg = arg.output[0].as<const memory&>().argument;
    auto& weight_arg = arg.input[1].primitive.as<const memory&>().argument;

    if (input_arg.size.raw.size() != output_arg.size.raw.size())    throw std::runtime_error("Fully connected input/output number of dimension does not match.");
    if (weight_arg.format != memory::format::xb_f32 &&
        weight_arg.format != memory::format::x_f32)                 throw std::runtime_error("Fully connected weight format is not xb_f32 or x_f32.");

    // wrap relu into RAII wrapper
    std::unique_ptr<fully_connected> result(new fully_connected(arg));

    // create implementation for non-lazy evaluation
    if(0 == (arg.engine & engine::lazy)) {
        // lookup in database; throw if not found
        auto key = std::make_tuple(arg.engine, result->input_memory(0).argument.format, result->output_memory(0).argument.format);
        auto it = implementation_map.find(key);
        if (it == std::end(implementation_map)) throw std::runtime_error("not yet implemented");

        // create implementation & attach it to result
        auto implementation = it->second(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }

    // release RAII wrapper, return naked pointer
    return result.release();
}

}
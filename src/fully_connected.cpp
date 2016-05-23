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
#include "fully_connected.h"
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
        auto input = static_cast<float*>(this_fc->input_memory(0).pointer);
        auto output = static_cast<float*>(this_fc->output_memory(0).pointer);
        auto weight = static_cast<float*>(this_fc->input_memory(1).pointer);
        auto& weight_buffer_size = this_fc->input_memory(1).argument.size;
        auto bias   = static_cast<float*>(this_fc->argument.input[2].primitive.as<const memory&>().pointer);


        auto& input_arg = this_fc->input_memory(0).argument;
        auto& input_buffer_size = input_arg.size;

        auto& output_arg = this_fc->output_memory(0).argument;
        auto& output_buffer_size = output_arg.size;

        auto& weight_arg = this_fc->input_memory(1).argument;

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

        auto calc_in_idx  = nd::choose_calculate_idx(input_arg.format);
        auto calc_out_idx = nd::choose_calculate_idx(output_arg.format);
        auto calc_w_idx   = nd::choose_calculate_idx(weight_arg.format);

        std::vector<uint32_t> arg_weight_idx(3);
        for (auto pos_out : range_output){
                auto out_idx = calc_out_idx(output_arg.size.raw, pos_out);

                for (auto pos_in : range_input){
                    auto in_idx = calc_in_idx(input_arg.size.raw, pos_in);

                    arg_weight_idx[DATA_INDEX]  = pos_out[DATA_INDEX];
                    arg_weight_idx[BATCH_INDEX] = pos_in [DATA_INDEX];
                    auto w_idx = calc_w_idx(weight_arg.size.raw, arg_weight_idx);
                    output[out_idx + pos_in[BATCH_INDEX]] += input[in_idx] * weight[w_idx];
                }
                for (auto  b=0u; b < range_input[BATCH_INDEX]; b++)
                    output[out_idx + b] += bias[pos_out[DATA_INDEX]];
        }
    }

    std::vector<task> work() {
        return{ task{ implementation, &outer } };
    }

    static is_an_implementation *create(fully_connected &arg) { return new fully_connected_reference(arg); };
};

//                                    engine                output                        input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;


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
        auto it = fully_con_implementation_map::instance().find(key);
        if (it == std::end(fully_con_implementation_map::instance())) throw std::runtime_error("not yet implemented");

        // create implementation & attach it to result
        auto implementation = it->second(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }

    // release RAII wrapper, return naked pointer
    return result.release();
}

namespace {
	struct attach {
		attach() {
			fully_con_implementation_map::instance().insert({ std::make_tuple(engine::reference, memory::format::xb_f32, memory::format::xb_f32), fully_connected_reference::create });
			fully_con_implementation_map::instance().insert({ std::make_tuple(engine::reference, memory::format::x_f32,  memory::format::x_f32),  fully_connected_reference::create });
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
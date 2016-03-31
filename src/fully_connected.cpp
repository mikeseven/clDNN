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
#include<iostream>//todo: remove
namespace neural {

struct fully_connected_reference : is_an_implementation {
    const fully_connected &outer;
    fully_connected_reference(fully_connected &arg)
        : is_an_implementation(neural::type_id<fully_connected_reference>())
        , outer(arg)
    {};
    ~fully_connected_reference() {}

    static void implementation(const void *ptr) {
        auto this_ful_con = static_cast<const fully_connected *>(ptr);
        auto input = static_cast<float*>(this_ful_con->input_memory(0).pointer);
        auto output = static_cast<float*>(this_ful_con->output_memory(0).pointer);
        auto weight = static_cast<float*>(this_ful_con->argument.weight.as<const memory&>().pointer);
        auto weight_size = this_ful_con->argument.weight.as<const memory&>().argument.size;

        auto input_memory_arg = this_ful_con->input_memory(0).argument;
        auto input_buffer_size = input_memory_arg.size;
        auto input_size = this_ful_con->input_memory(0).argument.size;

        auto output_memory_arg = this_ful_con->output_memory(0).argument;
        auto output_buffer_size = output_memory_arg.size;
        auto output_size = this_ful_con->argument.output_size;

        if (input_buffer_size.size() != output_buffer_size.size())throw std::runtime_error("Fully connected input/output number of dimension does not match.");
        if (input_memory_arg.format != output_memory_arg.format)  throw std::runtime_error("Fully connected input/output data format does not match.");
        if (this_ful_con->argument.weight.as<const memory&>().argument.format != memory::format::xb_f32 && 
            this_ful_con->argument.weight.as<const memory&>().argument.format != memory::format::x_f32) throw std::runtime_error("Fully connected weight format is not xb_f32 or x_f32.");

        assert(this_ful_con->input_memory(0).argument.size.size()==1 || this_ful_con->input_memory(0).argument.size.size() == 2);

        // up-casts data format form 1D to 2D if necessary; DOES not copy memory, just redescribes 1D input buffer as 2D (x+batch) with batch=1
        auto mem_arg_in = this_ful_con->input_memory(0).argument;
        if(mem_arg_in.size.size()==1) {
            mem_arg_in.size.emplace_back(1);
            mem_arg_in.format = memory::format::xb_f32;
        }
        mem_arg_in.owns_memory = false;
        auto in_wrapper = memory::create(mem_arg_in);
        in_wrapper(input);

        auto mem_arg_out = this_ful_con->output_memory(0).argument;
        if (mem_arg_out.size.size() == 1) {
            mem_arg_out.size.emplace_back(1);
            mem_arg_out.format = memory::format::xb_f32;
        }
        mem_arg_out.owns_memory = false;
        auto out_wrapper = memory::create(mem_arg_out);
        out_wrapper(output);

        namespace nd = ndimensional;

        int batch_size = 1;
        nd::value<uint32_t> range_output(0);

        if (this_ful_con->argument.weight.as<const memory&>().argument.format == memory::format::x_f32) {
            batch_size = 1;
            range_output = nd::value<uint32_t> (output_size); //there is no batch, so nothing has to be removed
        } 
        else if (this_ful_con->argument.weight.as<const memory&>().argument.format == memory::format::xb_f32) {
            batch_size = (int)input_buffer_size[1];
            range_output = nd::value<uint32_t>({ begin(output_size), end(output_size) - 1 }); //in every iteration whole batch is computed at once, so it has to be removed from the range
        }

        nd::value<uint32_t> range_weight(weight_size);
        nd::value<uint32_t> range_input(input_size);
        nd::calculate_idx<uint32_t> calc_in_idx(input_buffer_size);
        nd::calculate_idx<uint32_t> calc_out_idx(output_buffer_size);
        nd::calculate_idx<uint32_t> calc_w_idx(weight_size);

        int data_index = 0; //todo type traits
        int batch_index = 1;

        int batch_counter = 0;
        float *acc = new float[batch_size];
        for (auto i = 0; i < batch_size; ++i) {
            acc[i] = 0;
        }

        for (auto pos_out : range_output){

                auto out_idx = calc_out_idx(pos_out);

                for (auto pos_in : range_input){
                    auto in_idx = calc_in_idx(pos_in);
                    auto w_idx = calc_w_idx(std::vector<uint32_t>{ pos_out[data_index], pos_in[data_index] });
                    if (this_ful_con->argument.weight.as<const memory&>().argument.format == memory::format::xb_f32) {
                        batch_counter = pos_in[1]; // in format x_f32 there is nothing in pos_in[1]
                    }
                    acc[batch_counter] += input[in_idx] * weight[w_idx];
                }
                for (auto i = 0; i < batch_size; ++i) {
                    output[out_idx + i] = acc[i];
                    acc[i] = 0;
                }
        }
    }


    std::vector<task> work() {
        return{ task{ implementation, &outer } };
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

fully_connected::arguments::arguments( neural::engine::type  eng,
                                       primitive             out,
                                       primitive             in,
                                       primitive             weights)
: engine(eng)
, output({out})
, output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
, input({in})
, weight(weights){};

// creates primitive with fully_connected implementation that supports provided arguments
primitive fully_connected::create(fully_connected::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<fully_connected> result(new fully_connected(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result->input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = implementation_map.find(key);
    if (it == std::end(implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

}
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
        auto this_fc = static_cast<const fully_connected *>(ptr);
        //auto input = static_cast<float*>(this_fc->input_memory(0).pointer);
        //auto output = static_cast<float*>(this_fc->output_memory(0).pointer);
        //auto weight = static_cast<float*>(this_fc->input_memory(1).pointer);
        //auto weight_buffer_size = this_fc->input_memory(1).argument.size;
        auto input  = static_cast<float*>(this_fc->argument.input[0].primitive.as<const memory&>().pointer);  //todo tmp solution
        auto weight = static_cast<float*>(this_fc->argument.input[1].primitive.as<const memory&>().pointer);
        auto output = static_cast<float*>(this_fc->argument.output[0].as<const memory&>().pointer);
        auto& weight_buffer_size = this_fc->argument.input[1].primitive.as<const memory&>().argument.size;

        auto& input_memory_arg  = this_fc->argument.input[0].primitive.as<const memory&>().argument; //todo tmp solution
        //auto& input_memory_arg = this_fc->input_memory(0).argument;
        auto& input_buffer_size = input_memory_arg.size;

        auto& output_memory_arg = this_fc->argument.output[0].as<const memory&>().argument;
        //auto& output_memory_arg = this_fc->output_memory(0).argument;
        auto& output_buffer_size = output_memory_arg.size;

        auto& weight_memory_arg = this_fc->argument.input[0].primitive.as<const memory&>().argument.format;
        //auto weight_memory_arg = this_fc->input_memory(1).argument.format;

        if (input_buffer_size.raw.size() != output_buffer_size.raw.size()) throw std::runtime_error("Fully connected input/output number of dimension does not match.");
        if (input_memory_arg.format      != output_memory_arg.format)      throw std::runtime_error("Fully connected input/output data format does not match.");
        if (weight_memory_arg            != memory::format::xb_f32 &&
            weight_memory_arg            != memory::format::x_f32)
            throw std::runtime_error("Fully connected weight format is not xb_f32 or x_f32.");

        assert( 1 == input_buffer_size.feature.size());
        assert( 1 == input_buffer_size.batch.size()  );
        assert( 1 == input_buffer_size.feature[0]    );

        // up-casts data format form 1D to 2D if necessary; DOES not copy memory, just redescribes 1D input buffer as 2D (x+batch) with batch=1
        //auto& mem_arg_in = this_fc->argument.input[0].primitive.as<const memory&>();
        //if(mem_arg_in.size.size()==1) {
        //    mem_arg_in.size.emplace_back(1);
        //    mem_arg_in.format = memory::format::xb_f32;
        //}
        //mem_arg_in.owns_memory = false;

        auto mem_arg_in = this_fc->argument.input[0].primitive.as<const memory&>().argument;
        mem_arg_in.format = memory::format::xb_f32;
        mem_arg_in.owns_memory = false;
        auto in_wrapper = memory::create(mem_arg_in);
        in_wrapper(input);

        //auto mem_arg_out = this_fc->output_memory(0).argument;
        //if (mem_arg_out.size.size() == 1) {
        //    mem_arg_out.size.emplace_back(1);
        //    mem_arg_out.format = memory::format::xb_f32;
        //}
        //mem_arg_out.owns_memory = false;
        //auto out_wrapper = memory::create(mem_arg_out);
        //out_wrapper(output);
        auto mem_arg_out = this_fc->argument.output[0].as<const memory&>().argument;
        mem_arg_out.format = memory::format::xb_f32;
        mem_arg_out.owns_memory = false;
        auto out_wrapper = memory::create(mem_arg_out);
        out_wrapper(output);


        namespace nd = ndimensional;

        //if (weight_memory_arg == memory::format::x_f32) {
        //    range_output = nd::value<uint32_t> (output_buffer_size); //there is no batch, so nothing has to be removed
        //}
        //else if (weight_memory_arg == memory::format::xb_f32) {
        //    range_output = nd::value<uint32_t>({ begin(output_buffer_size), end(output_buffer_size) - 1 }); //in every iteration whole batch is computed at once, so it has to be removed from the range
        //}

        //this_fc->output_memory(0).fill(0.0f);
        this_fc->argument.output[0].as<const memory&>().fill(0.0f);

        int data_index = 2;
        int batch_index = 0;
        //nd::value<uint32_t> range_output({{output_buffer_size.spatial[0]}}); //in every iteration whole batch is computed at once, so it has to be removed from the range
        nd::value<uint32_t> range_output(output_buffer_size); //in every iteration whole batch is computed at once, so it has to be removed from the range
        range_output[batch_index] = 1;
        nd::value<uint32_t> range_input(input_buffer_size);
        nd::value<uint32_t> range_weight(weight_buffer_size);
        nd::calculate_idx<uint32_t, memory::format::xb_f32> calc_in_idx(input_buffer_size);
        nd::calculate_idx<uint32_t, memory::format::xb_f32> calc_out_idx(output_buffer_size);
        nd::calculate_idx<uint32_t, memory::format::xb_f32> calc_w_idx(weight_buffer_size);

        std::vector<uint32_t> arg_weight_idx(3);

        for (auto pos_out : range_output){
                auto out_idx = calc_out_idx(pos_out);

                for (auto pos_in : range_input){
                    auto in_idx = calc_in_idx(pos_in);
                    arg_weight_idx[data_index]  = pos_out[data_index];
                    arg_weight_idx[batch_index] = pos_in [data_index];
                    auto w_idx = calc_w_idx(arg_weight_idx);
                    output[out_idx + pos_in[batch_index]] += input[in_idx] * weight[w_idx];
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

fully_connected::arguments::arguments( neural::engine::type   eng,
                                       primitive              out,
                                       primitive              in,
                                       primitive              weights)
: engine(eng)
, output({out})
, output_size(out.as<const memory&>().argument.size)
, input({in, weights})
{
};

// creates primitive with fully_connected implementation that supports provided arguments
primitive fully_connected::create(fully_connected::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<fully_connected> result(new fully_connected(arg));

    // lookup in database; throw if not found

        //todo tmp solution
    auto& infmt = result->argument.input[0].primitive.as<const memory&>().argument.format;
    auto& outfmt= result->argument.output[0].as<const memory&>().argument.format;
    auto key = std::make_tuple(arg.engine, infmt, outfmt);

   // auto key = std::make_tuple(arg.engine, result->input_memory(0).argument.format, result->output_memory(0).argument.format);
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
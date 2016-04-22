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
#include "multidimensional_counter.h"
#include <climits>
#include <cmath>

namespace neural {
namespace normalization {

namespace {
struct softmax_reference : is_an_implementation {
    const softmax &outer;
    softmax_reference(softmax &arg)
        : is_an_implementation(neural::type_id<softmax_reference>())
        , outer(arg)
    {};
    ~softmax_reference() {}

    static void implementation(const void *ptr) {
        auto this_softmax = static_cast<const softmax *>(ptr);
        //auto input        = static_cast<float*>(this_softmax->input_memory(0).pointer);
        //auto output       = static_cast<float*>(this_softmax->output_memory(0).pointer);
        auto input  = static_cast<float*>(this_softmax->argument.input[0].primitive.as<const memory&>().pointer);  //todo tmp solution
        auto output = static_cast<float*>(this_softmax->argument.output[0].as<const memory&>().pointer);

        auto input_offset  = this_softmax->argument.input_offset;
        auto output_offset = this_softmax->argument.output_offset;
        auto output_size   = this_softmax->argument.output_size;

        //auto input_arg  = this_relu->input_memory(0).argument;
        //auto output_arg = this_relu->output_memory(0).argument;
        auto input_arg  = this_softmax->argument.input[0].primitive.as<const memory&>().argument; //todo tmp solution
        auto output_arg = this_softmax->argument.output[0].as<const memory&>().argument;

        if(input_arg.format          != memory::format::xb_f32    ) throw std::runtime_error("Softmax reference uses xb_f32 format."); // todo should be format independent
        if(input_arg.format          != output_arg.format         ) throw std::runtime_error("Softmax input/output data format does not match.");
        if(input_arg.size.raw.size() != output_arg.size.raw.size()) throw std::runtime_error("Softmax input/output number of dimension does not match.");

        for(size_t i = 0; i < input_arg.size.raw.size(); ++i){
            if( input_arg.size.raw[i] < output_size.raw[i] +  input_offset.raw[i]) throw std::runtime_error("Softmax input/output size does not match.");
            if(output_arg.size.raw[i] < output_size.raw[i] + output_offset.raw[i]) throw std::runtime_error("Softmax sizes to small.");
        }

        assert( 1 == output_size.feature.size() );
        assert( 1 == output_size.batch.size()   );
        int batch_index = 1;
        std::vector<float> v_max( output_size.batch[0], -std::numeric_limits<float>::max() );
        std::vector<float> v_acc( output_size.batch[0] );

        namespace nd = ndimensional;
        nd::value<uint32_t> range (output_size);
        nd::calculate_idx<uint32_t, static_cast<int>(memory::format::xb_f32)> calc_in_idx  (input_arg.size);
        nd::calculate_idx<uint32_t, static_cast<int>(memory::format::xb_f32)> calc_out_idx (output_arg.size);

        // find max val per batch
        for(auto pos : range) {
            auto in_idx  = calc_in_idx (pos + input_offset );
            v_max[ pos[batch_index] ] = std::max( v_max[pos[batch_index]], input[in_idx]);
        }
        for(auto pos : range) {
            auto in_idx  = calc_in_idx (pos + input_offset );
            auto out_idx = calc_out_idx(pos + output_offset);

            output[out_idx] = input[in_idx] - v_max[ pos[batch_index] ]; // subtracte max val from every data point per batch
            output[out_idx] = std::exp(output[out_idx]);  // exp
            v_acc[ pos[batch_index] ] += output[out_idx]; // sum eveything per batch
        }
        for(auto pos : range) {
            auto out_idx = calc_out_idx(pos + output_offset);
            output[out_idx] /= v_acc[ pos[batch_index] ]; // compute softmax
        }
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(softmax &arg) { return new softmax_reference(arg); };
};

} // namespace {

softmax::arguments::arguments( neural::engine::type eng, primitive out, vector<uint32_t> out_off, vector<uint32_t> out_siz, primitive in, vector<int32_t> in_off)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    {}

softmax::arguments::arguments( neural::engine::type eng, primitive out, primitive in )
    : engine(eng)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input({in})
    , input_offset(in.as<const memory&>().argument.size.batch.size(), in.as<const memory&>().argument.size.spatial.size(), in.as<const memory&>().argument.size.feature.size())
    {}

//                                    engine                output                        input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(softmax &)>> forward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::xb_f32, memory::format::xb_f32), softmax_reference::create}
};
// creates primitive with softmax implementation that supports provided arguments
primitive softmax::create(softmax::arguments arg) {
    // wrap softmax into RAII wrapper
    std::unique_ptr<softmax> result(new softmax(arg));

    // lookup in database; throw if not found
            //todo tmp solution
    auto& infmt = result->argument.input[0].primitive.as<const memory&>().argument.format;
    auto& outfmt= result->argument.output[0].as<const memory&>().argument.format;
    auto key = std::make_tuple(arg.engine, infmt, outfmt);
   // auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = forward_implementation_map.find(key);
    if(it==std::end(forward_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

} // namespace normalization
} // namespace neural

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
#include <algorithm>
#include <functional>
#include <numeric>
#include <map>
#include <tuple>
#include <limits>

namespace neural {

struct pooling_reference : is_an_implementation {
    const pooling &outer;
    pooling_reference(pooling &arg)
        : is_an_implementation(neural::type_id<pooling_reference>())
        , outer(arg)
    {};
    ~pooling_reference() {}

    static void implementation(const void *ptr) {
        auto this_pooling = static_cast<const pooling *>(ptr);
        auto input        = static_cast<float*>(this_pooling->input_memory(0).pointer);
        auto output       = static_cast<float*>(this_pooling->output_memory(0).pointer);

        auto input_memory_arg  = this_pooling->input_memory(0).argument;
        auto input_buffer_size = input_memory_arg.size;
        auto input_offset      = this_pooling->argument.input_offset;

        auto output_memory_arg = this_pooling->output_memory(0).argument;
        auto output_buffer_size= output_memory_arg.size;
        auto output_offset     = this_pooling->argument.output_offset;
        auto output_size       = this_pooling->argument.output_size;

        auto stride            = this_pooling->argument.stride;
        auto window            = this_pooling->argument.size;
        auto padding           = this_pooling->argument.padding; //todo, padding is not supported yet

        if(padding::zero            != padding)                   throw std::runtime_error("Padding is not supported.");
        if(input_memory_arg.format  != memory::format::yxfb_f32)  throw std::runtime_error("Pooling reference uses yxfb_f32 format."); //todo, only this format?
        if(input_buffer_size.size() != output_buffer_size.size()) throw std::runtime_error("Pooling input/output number of dimension does not match.");
        if(stride.size()            != output_buffer_size.size()) throw std::runtime_error("Pooling stride/output number of dimension does not match.");
        if(window.size()            != output_buffer_size.size()) throw std::runtime_error("Pooling window_size/output number of dimension does not match.");
        if(input_memory_arg.format  != output_memory_arg.format)  throw std::runtime_error("Pooling input/output data format does not match.");

        // general formula: output size = (input size - window size) / step + 1
        for(size_t i = 0; i < input_offset.size(); ++i){
            if(output_size[i] < (static_cast<int32_t>(input_buffer_size[i]) - input_offset[i]) / (stride[i] + 1) )
                throw std::runtime_error("Output size of pooling is to small.");

            if(output_buffer_size[i] < output_size[i] + output_offset[i])
                throw std::runtime_error("Pooling output buffer size is to small.");
        }

        namespace nd = ndimensional;
        nd::value<uint32_t> range(output_size);
        nd::value<uint32_t> window_range(window);
        nd::calculate_idx<uint32_t> calc_in_idx(input_buffer_size);
        nd::calculate_idx<uint32_t> calc_out_idx(output_buffer_size);
        switch( this_pooling->argument.mode ){
            case pooling::mode::max:
                for(auto pos : range) {
                    auto out_idx = calc_out_idx(pos + output_offset);

                    float acc = -std::numeric_limits<float>::max();
                    for(auto win_pos : window_range){
                        const std::vector<int32_t> arg_in_idx = nd::value<int32_t>(input_offset) + pos*stride + win_pos;

                        if( calc_in_idx.is_out_of_range(arg_in_idx) )
                        {
                            // Pad with zero.
                            acc = std::max(acc, 0.0f);
                            continue;
                        }    

                        auto in_idx = calc_in_idx(arg_in_idx);
                        acc = std::max(acc, input[in_idx]);
                    }
                    output[out_idx] = acc;
                }
                break;
            case pooling::mode::average:
            {
                auto window_elements = std::accumulate(window.cbegin(), window.cend(), 1, std::multiplies<uint32_t>());
                for(auto pos : range) {
                    auto out_idx = calc_out_idx(pos + output_offset);

                    float acc = 0.0f;
                    for(auto win_pos : window_range){
                        const std::vector<int32_t> arg_in_idx = nd::value<int32_t>(input_offset) + pos*stride + win_pos;

                        if( calc_in_idx.is_out_of_range(arg_in_idx) )
                            continue;

                        auto in_idx = calc_in_idx(arg_in_idx);
                        acc += input[in_idx];
                    }
                    output[out_idx] = acc/window_elements;
                }
            }
                break;
            default:
                throw std::runtime_error("Unknown pooling mode.");
        }
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(pooling &arg) { return new pooling_reference(arg); };
};

pooling::arguments::arguments( neural::engine::type  eng,
                               pooling::mode::type   mode,
                               memory::format::type  o_frmt,
                               std::vector<uint32_t> out_off,
                               std::vector<uint32_t> out_siz,
                               primitive             in,
                               std::vector<int32_t>  in_off,
                               std::vector<uint32_t> strd,
                               std::vector<uint32_t> siz,
                               neural::padding::type padd)
    : engine(eng)
    , mode(mode)
    , output( {memory::create({eng, o_frmt, out_siz, true})} )
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , size(siz)
    , padding(padd) {};

pooling::arguments::arguments( neural::engine::type  eng,
                               pooling::mode::type   mode,
                               primitive             out,
                               std::vector<uint32_t> out_off,
                               std::vector<uint32_t> out_siz,
                               primitive             in,
                               std::vector<int32_t>  in_off,
                               std::vector<uint32_t> strd,
                               std::vector<uint32_t> siz,
                               neural::padding::type padd)
    : engine(eng)
    , mode(mode)
    , output({out})
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , size(siz)
    , padding(padd) {};

pooling::arguments::arguments( neural::engine::type eng,
                              pooling::mode::type   mode,
                              primitive             out,
                              primitive             in,
                              std::vector<uint32_t> strd,
                              std::vector<uint32_t> siz,
                              neural::padding::type padd)
    : engine(eng)
    , mode(mode)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.size())
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset(in.as<const memory&>().argument.size.size())
    , stride({strd})
    , size({siz})
    , padding(padd) {};

pooling::arguments::arguments( neural::engine::type eng,
                              pooling::mode::type   mode,
                              primitive             out,
                              primitive             in,
                              std::vector<int32_t>  in_off,
                              std::vector<uint32_t> strd,
                              std::vector<uint32_t> siz,
                              neural::padding::type padd)
    : engine(eng)
    , mode(mode)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.size())
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset({in_off})
    , stride({strd})
    , size({siz})
    , padding(padd) {};

//                                    engine          output                  input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(pooling &)>> implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), pooling_reference::create}
};

// creates primitive with pooling implementation that supports provided arguments
primitive pooling::create(pooling::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<pooling> result(new pooling(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = implementation_map.find(key);
    if(it==std::end(implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

}
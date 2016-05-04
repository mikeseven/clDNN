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

#include "multidimensional_counter.h"
#include "pooling.h"

namespace neural {

pooling::arguments::arguments( neural::engine::type     eng,
                               pooling::mode::type      p_mode,
                               memory::format::type     o_frmt,
                               neural::vector<uint32_t> out_off,
                               neural::vector<uint32_t> out_siz,
                               primitive                in,
                               neural::vector<int32_t>  in_off,
                               neural::vector<uint32_t> strd,
                               neural::vector<uint32_t> siz,
                               neural::padding::type    padd)
    : engine(eng)
    , mode(p_mode)
    , output( {memory::create({eng, o_frmt, out_siz, true})} )
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , size(siz)
    , padding(padd) {};

pooling::arguments::arguments( neural::engine::type     eng,
                               pooling::mode::type      p_mode,
                               primitive                out,
                               neural::vector<uint32_t> out_off,
                               neural::vector<uint32_t> out_siz,
                               primitive                in,
                               neural::vector<int32_t>  in_off,
                               neural::vector<uint32_t> strd,
                               neural::vector<uint32_t> siz,
                               neural::padding::type    padd)
    : engine(eng)
    , mode(p_mode)
    , output({out})
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , size(siz)
    , padding(padd) {};

pooling::arguments::arguments( neural::engine::type     eng,
                               pooling::mode::type      p_mode,
                               primitive                out,
                               primitive                in,
                               neural::vector<uint32_t> strd,
                               neural::vector<uint32_t> siz,
                               neural::padding::type    padd)
    : engine(eng)
    , mode(p_mode)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input({in})
    , input_offset(in.as<const memory&>().argument.size.batch.size(), in.as<const memory&>().argument.size.spatial.size(), in.as<const memory&>().argument.size.feature.size())
    , stride(strd)
    , size(siz)
    , padding(padd) {};

pooling::arguments::arguments( neural::engine::type     eng,
                               pooling::mode::type      p_mode,
                               primitive                out,
                               primitive                in,
                               neural::vector<int32_t>  in_off,
                               neural::vector<uint32_t> strd,
                               neural::vector<uint32_t> siz,
                               neural::padding::type    padd)
    : engine(eng)
    , mode(p_mode)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , size(siz)
    , padding(padd) {};

//                                    engine          output                  input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(pooling &)>> implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), pooling_cpu_reference::create}
};

// creates primitive with pooling implementation that supports provided arguments
primitive pooling::create(pooling::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<pooling> result(new pooling(arg));

    // lookup in database; throw if not found
            //todo tmp solution
    auto& infmt = result->argument.input[0].primitive.as<const memory&>().argument.format;
    auto& outfmt= result->argument.output[0].as<const memory&>().argument.format;
    auto key = std::make_tuple(arg.engine, infmt, outfmt);
//    auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
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
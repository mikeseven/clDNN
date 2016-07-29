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
#include "implementation_map.h"

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
    , output( {memory::allocate({eng, o_frmt, out_siz})} )
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , size(siz)
    , padding(padd) {};

pooling::arguments::arguments( neural::engine::type     eng,
                               pooling::mode::type      p_mode,
                               memory::format::type     o_frmt,
                               primitive                in,
                               neural::vector<uint32_t> strd,
                               neural::vector<uint32_t> siz,
                               neural::padding::type    padd)
    : engine(eng)
    , mode(p_mode)
    , input({in})
    , stride(strd)
    , size(siz)
    , padding(padd) 
    , output_offset({ 0,0 })
{
    // verify if primitive has one output.
    if (in.output.size() != 1) throw std::runtime_error("more than one output in primitive isn't supported yet");
    auto output_memory = in.output[0].as<const memory&>().argument;
    // compute size of output after pooling (downsampling)
    auto spatial_x = (output_memory.size.spatial[0]) / strd.spatial[0];
    auto spatial_y = (output_memory.size.spatial[1]) / strd.spatial[1];

    output_size = { 
        output_memory.size.batch[0],
        {
            spatial_x,
            spatial_y
        },
        output_memory.size.feature[0]
    };
    output = { memory::allocate({eng, o_frmt, output_size }) };
};


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
    , stride(strd)
    , size(siz)
    , padding(padd) 
{
    if (in.id() == type_id<const memory>()->id)
    {
        input = { in };
    }
    else
    {
        input = { in.output[0] };
    }

    input_offset =
    {
        input[0].primitive.as<const memory&>().argument.size.batch.size(),
        input[0].primitive.as<const memory&>().argument.size.spatial.size(),
        input[0].primitive.as<const memory&>().argument.size.feature.size(),
    };
};

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

// creates primitive with pooling implementation that supports provided arguments
primitive pooling::create(pooling::arguments arg) {
    return is_a_primitive::create<pooling>(arg);
}

}
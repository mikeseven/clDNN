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
#include "relu.h"

namespace neural {

relu::arguments::arguments(neural::engine::type engine, primitive out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, float slp)
    : engine(engine)
    , output({ out })
    , output_offset({ out_off })
    , output_size({ out_siz })
    , input({ in })
    , input_offset({ in_off })
    , negative_slope(slp) {}

relu::arguments::arguments(neural::engine::type engine, primitive out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off)
    : engine(engine)
    , output({ out })
    , output_offset({ out_off })
    , output_size({ out_siz })
    , input({ in })
    , input_offset({ in_off })
    , negative_slope(0.0f) {}

relu::arguments::arguments(neural::engine::type engine, primitive out, primitive in, float slp)
    : engine(engine)
    , output({ out })
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input({ in })
    , input_offset(in.as<const memory&>().argument.size.batch.size(), in.as<const memory&>().argument.size.spatial.size(), in.as<const memory&>().argument.size.feature.size())
    , negative_slope(slp) {}

relu::arguments::arguments(neural::engine::type engine, primitive out, primitive in)
    : engine(engine)
    , output({ out })
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input({ in })
    , input_offset(in.as<const memory&>().argument.size.batch.size(), in.as<const memory&>().argument.size.spatial.size(), in.as<const memory&>().argument.size.feature.size())
    , negative_slope(0.0f) {}

relu_backward::arguments::arguments(neural::engine::type engine, primitive out, neural::vector<uint32_t> out_offset, neural::vector<uint32_t> out_size, std::vector<primitive_at> in, std::vector<neural::vector<uint32_t>> in_offsets, float neg_slope)
    : engine(engine)
    , output({ out })
    , output_offset(out_offset)
    , output_size(out_size)
    , input(in)
    , input_offset(in_offsets)
    , negative_slope(neg_slope) {}

relu_backward::arguments::arguments(neural::engine::type engine, primitive out, std::vector<primitive_at> in, float neg_slope)
    : engine(engine)
    , output({ out })
    , output_offset(out.as<const memory&>().argument.size.batch.size(), out.as<const memory&>().argument.size.spatial.size(), out.as<const memory&>().argument.size.feature.size())
    , output_size(out.as<const memory&>().argument.size)
    , input(in)
    , input_offset({ in.size(),{ in[0].primitive.as<const memory&>().argument.size.batch.size(), in[0].primitive.as<const memory&>().argument.size.spatial.size(), in[0].primitive.as<const memory&>().argument.size.feature.size() } })
    , negative_slope(neg_slope) {}

// creates primitive with relu implementation that supports provided arguments
primitive relu::create(relu::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<relu> result(new relu(arg));

    // lookup in database; throw if not found
    //todo tmp solution
    auto& infmt = result->argument.input[0].primitive.as<const memory&>().argument.format;
    auto& outfmt = result->argument.output[0].as<const memory&>().argument.format;
    auto key = std::make_tuple(arg.engine, infmt, outfmt);
    //auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = relu_fw_implementation_map.find(key);
    if (it == std::end(relu_fw_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}


primitive relu_backward::create(relu_backward::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<relu_backward> result(new relu_backward(arg));

    // lookup in database; throw if not found
    //todo tmp solution
    auto& infmt = result->argument.input[0].primitive.as<const memory&>().argument.format;
    auto& outfmt = result->argument.output[0].as<const memory&>().argument.format;
    auto key = std::make_tuple(arg.engine, infmt, outfmt);
    //    auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = relu_bw_implementation_map.find(key);
    if (it == std::end(relu_bw_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}
}
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
#include "lrn.h"

namespace neural {

normalization::response::arguments::arguments(
                 neural::engine::type aengine,
                 primitive aoutput,
                 primitive ainput,
                 uint32_t  asize,
                 neural::padding::type apadding,
                 float ak,
                 float aalpha,
                 float abeta)
    : engine(aengine)
    , output({ aoutput })
    , output_offset(aoutput.as<const memory&>().argument.size.batch.size(), aoutput.as<const memory&>().argument.size.spatial.size(), aoutput.as<const memory&>().argument.size.feature.size())
    , output_size(aoutput.as<const memory&>().argument.size)
    , input({ ainput })
    , input_offset(ainput.as<const memory&>().argument.size.batch.size(), ainput.as<const memory&>().argument.size.spatial.size(), ainput.as<const memory&>().argument.size.feature.size())
    , size(asize)
    , padding(apadding)
    , k(ak)
    , alpha(aalpha)
    , beta(abeta)
{ };

normalization::response::arguments::arguments(
                 neural::engine::type aengine,
                 memory::format::type aoutput_fmt,
                 primitive ainput,
                 uint32_t  asize,
                 neural::padding::type apadding,
                 float ak,
                 float aalpha,
                 float abeta)
    : engine(aengine)
    , input({ ainput })
    , size(asize)
    , padding(apadding)
    , k(ak)
    , alpha(aalpha)
    , beta(abeta)
    , output({ memory::allocate({aengine, aoutput_fmt,output_size}) })
{ 
    if (ainput.output.size() != 1) throw std::runtime_error("should have one output");
    auto input_mem = ainput.output[0].as<const memory&>().argument;
    input_offset =
    {
        input_mem.size.batch.size(),
        input_mem.size.spatial.size(),
        input_mem.size.feature.size()
    };
    output_offset =
    {
        input_mem.size.batch.size(),
        input_mem.size.spatial.size(),
        input_mem.size.feature.size()
    };
    output_size = input_mem.size;
    output = {
        memory::allocate(
        {
            aengine,
            aoutput_fmt,
            output_size
        }) };
};

normalization::response::arguments::arguments(
                 neural::engine::type aengine,
                 primitive aoutput,
                 vector<uint32_t> aoutput_offset,
                 vector<uint32_t> aoutput_size,
                 primitive ainput,
                 vector<int32_t> ainput_offset,
                 uint32_t asize,
                 neural::padding::type apadding,
                 float ak,
                 float aalpha,
                 float abeta)
    : engine (aengine)
    , output({ aoutput })
    , output_offset (aoutput_offset)
    , output_size (aoutput_size)
    , input({ ainput })
    , input_offset (ainput_offset)
    , size (asize)
    , padding (apadding)
    , k (ak)
    , alpha (aalpha)
    , beta (abeta) { };


// creates primitive with convolution implementation that supports provided arguments
primitive normalization::response::create(response::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<response> result(new response(arg));

    // create implementation for non-lazy evaluation
    if(0 == (arg.engine & engine::lazy)) {
        // lookup in database; throw if not found
        lrn_fw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
        auto it = lrn_fw_implementation_map::instance().find(key);
        if(it==std::end(lrn_fw_implementation_map::instance())) throw std::runtime_error("Not yet implemented.");

        // create implementation & attach it to result
        auto implementation = it->second(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }

    // release RAII wrapper, return naked pointer
    return result.release();
}
/*
primitive normalization::response_backward::create(normalization::response_backward::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<normalization::response_backward> result(new normalization::response_backward(arg));

    // create implementation for non-lazy evaluation
    if(0 == (arg.engine & engine::lazy)) {
        // lookup in database; throw if not found
        lrn_bw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
        auto it = lrn_bw_implementation_map.find(key);
        if(it==std::end(lrn_bw_implementation_map)) throw std::runtime_error("Not yet implemented.");

        // create implementation & attach it to result
        auto implementation = it->second(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }

    // release RAII wrapper, return naked pointer
    return result.release();
}
*/
}

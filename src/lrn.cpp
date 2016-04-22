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

singleton_map<lrn_fw_key, std::function<is_an_implementation *(normalization::response &)>>         & lrn_fw_implementation_map = singleton_map<lrn_fw_key, std::function<is_an_implementation *(normalization::response &)>>         ::instance();
//singletion_map<lrn_bw_key, std::function<is_an_implementation *(normalization::response_backward &)>>& lrn_bw_implementation_map = singletion_map<lrn_bw_key, std::function<is_an_implementation *(normalization::response_backward &)>>::instance();

normalization::response::arguments::arguments(
                 neural::engine::type aengine,
                 primitive aoutput,
                 std::vector<uint32_t> aoutput_offset,
                 std::vector<uint32_t> aoutput_size,
                 primitive ainput,
                 std::vector<int32_t> ainput_offset,
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

    // lookup in database; throw if not found
    lrn_fw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = lrn_fw_implementation_map.find(key);
    if(it==std::end(lrn_fw_implementation_map)) throw std::runtime_error("Not yet implemented.");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}
/*
primitive normalization::response_backward::create(normalization::response_backward::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<normalization::response_backward> result(new normalization::response_backward(arg));

    // lookup in database; throw if not found
    lrn_bw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = lrn_bw_implementation_map.find(key);
    if(it==std::end(lrn_bw_implementation_map)) throw std::runtime_error("Not yet implemented.");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}
*/
}

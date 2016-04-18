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

/*
convolution::arguments::arguments( neural::engine::type  eng,
                                   primitive             out,
                                   std::vector<uint32_t> out_off,
                                   std::vector<uint32_t> out_siz,
                                   primitive             in,
                                   std::vector<int32_t>  in_off,
                                   std::vector<uint32_t> strd,
                                   primitive             weights,
                                   primitive             biases,
                                   neural::padding::type padd)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , weight(weights)
    , bias(biases)
    , padding(padd) {};

convolution::arguments::arguments( neural::engine::type  eng,
                                   primitive             out,
                                   primitive             in,
                                   std::vector<uint32_t> strd,
                                   primitive             weights,
                                   primitive             biases,
                                   neural::padding::type padd)
    : engine(eng)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.size())
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset(in.as<const memory&>().argument.size.size())
    , stride(strd)
    , weight(weights)
    , bias(biases)
    , padding(padd) {};

convolution_backward::arguments::arguments( neural::engine::type   eng,
                                            std::vector<primitive> out,
                                            std::vector<uint32_t>  out_off,
                                            std::vector<uint32_t>  in_siz,
                                            std::vector<primitive> in,
                                            std::vector<int32_t>   in_off,
                                            std::vector<uint32_t>  strd,
                                            neural::padding::type  padd)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , input_size(in_siz)
    , input(in.cbegin(), in.cend())
    , input_offset(in_off)
    , stride(strd)
    , padding(padd) {};

convolution_backward::arguments::arguments( neural::engine::type   eng,
                                            std::vector<primitive> out,
                                            std::vector<primitive> in,
                                            std::vector<uint32_t>  strd,
                                            neural::padding::type  padd)
    : engine(eng)
    , output({out})
    , output_offset(out[0].as<const memory&>().argument.size.size())
    , input_size(in[0].as<const memory&>().argument.size)
    , input(in.cbegin(), in.cend())
    , input_offset(in[0].as<const memory&>().argument.size.size())
    , stride(strd)
    , padding(padd) {};
*/
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

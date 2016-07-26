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
#include "convolution_relu.h"
#include "convolution_common.h"
#include <sstream>

namespace neural {

convolution_relu::arguments::arguments( neural::engine::type     eng,
                                   primitive                out,
                                   neural::vector<uint32_t> out_off,
                                   neural::vector<uint32_t> out_siz,
                                   std::vector<primitive_at>in,
                                   neural::vector<int32_t>  in_off,
                                   neural::vector<uint32_t> strd,
                                   neural::padding::type    padd,
                                   float nslp,
                                   size_t splt)
    : convolution_common::arguments(eng, out, out_off, out_siz, in, in_off, strd, padd, splt), negative_slope(nslp) {};

convolution_relu::arguments::arguments( neural::engine::type     eng,
                                   primitive                out,
                                   std::vector<primitive_at>in,
                                   neural::vector<uint32_t> strd,
                                   neural::padding::type    padd,
                                   float nslp,
                                   size_t splt)
    : convolution_common::arguments(eng, out, in, strd, padd, splt), negative_slope(nslp) {};

convolution_relu::arguments::arguments(neural::engine::type     eng,
    memory::format::type    out_fmt,
    std::vector<primitive_at>in,
    neural::vector<int32_t>  in_off,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    float nslp,
    size_t splt)
    : convolution_common::arguments(eng, out_fmt, in, in_off, strd, padd, splt), negative_slope(nslp) {};

convolution_relu::arguments::arguments(neural::engine::type     eng,
    memory::format::type    out_fmt,
    std::vector<primitive_at>in,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    float nslp,
    size_t splt)
    : convolution_common::arguments(eng, out_fmt, in, strd, padd, splt), negative_slope(nslp) {}

convolution_relu::arguments::arguments( neural::engine::type     eng,
                                   primitive                out,
                                   std::vector<primitive_at>in,
                                   neural::padding::type    padd,
                                   float nslp,
                                   size_t splt)
    : convolution_common::arguments(eng, out, in, padd, splt), negative_slope(nslp) {};

// creates primitive with convolution_relu implementation that supports provided arguments
primitive convolution_relu::create(convolution_relu::arguments arg) {
    try {
        validate_convolution_common_params(arg);
    }
    catch (std::runtime_error err) {
        throw std::runtime_error(std::string("Convolution_Relu ") + err.what());
    }

    // wrap relu into RAII wrapper
    std::unique_ptr<convolution_relu> result(new convolution_relu(arg));

    // create implementation for non-lazy evaluation
    if(0 == (arg.engine & engine::lazy)) {
        // lookup in database; throw if not found
        conv_relu_fw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
        auto it = conv_relu_fw_implementation_map.find(key);
        if(it==std::end(conv_relu_fw_implementation_map)) throw std::runtime_error("Not yet implemented.");

        // create implementation & attach it to result
        auto implementation = it->second(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }

    // release RAII wrapper, return naked pointer
    return result.release();
}
}

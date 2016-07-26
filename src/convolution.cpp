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
#include "convolution.h"
#include "convolution_common.h"
#include <sstream>

namespace neural {

convolution::arguments::arguments(neural::engine::type     eng,
    primitive                out,
    neural::vector<uint32_t> out_off,
    neural::vector<uint32_t> out_siz,
    std::vector<primitive_at>in,
    neural::vector<int32_t>  in_off,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    size_t                   splt)
    : convolution_common::arguments(eng, out, out_off, out_siz, in, in_off, strd, padd, splt) {};

convolution::arguments::arguments(neural::engine::type     eng,
    primitive                out,
    std::vector<primitive_at>in,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    size_t                   splt)
    : convolution_common::arguments(eng, out, in, strd, padd, splt) {};

convolution::arguments::arguments(neural::engine::type     eng,
    memory::format::type     out_fmt,
    std::vector<primitive_at>in,
    neural::vector<int32_t>  in_off,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    size_t                   splt)
    : convolution_common::arguments(eng, out_fmt, in, in_off, strd, padd, splt) {};

convolution::arguments::arguments(neural::engine::type     eng,
    memory::format::type     out_fmt,
    std::vector<primitive_at>in,
    neural::vector<uint32_t> strd,
    neural::padding::type    padd,
    size_t                   splt)
    : convolution_common::arguments(eng, out_fmt, in, strd, padd, splt) {}

convolution::arguments::arguments(neural::engine::type     eng,
    primitive                out,
    std::vector<primitive_at>in,
    neural::padding::type    padd,
    size_t                   splt)
    : convolution_common::arguments(eng, out, in, padd, splt) {};

convolution_backward::arguments::arguments( neural::engine::type     eng,
                                            std::vector<primitive>   out,
                                            neural::vector<uint32_t> out_off,
                                            neural::vector<uint32_t> in_siz,
                                            std::vector<primitive>   in,
                                            neural::vector<int32_t>  in_off,
                                            neural::vector<uint32_t> strd,
                                            neural::padding::type    padd)
    : engine(eng)
    , output(out)
    , output_offset(out_off)
    , input_size(in_siz)
    , input(in.cbegin(), in.cend())
    , input_offset(in_off)
    , stride(strd)
    , padding(padd) {};

convolution_backward::arguments::arguments( neural::engine::type     eng,
                                            std::vector<primitive>   out,
                                            std::vector<primitive>   in,
                                            neural::vector<uint32_t> strd,
                                            neural::padding::type    padd)
    : engine(eng)
    , output(out)
    , output_offset(out[0].as<const memory&>().argument.size.batch.size(), out[0].as<const memory&>().argument.size.spatial.size(), out[0].as<const memory&>().argument.size.feature.size())
    , input_size(in[0].as<const memory&>().argument.size)
    , input(in.cbegin(), in.cend())
    , input_offset(in[0].as<const memory&>().argument.size.batch.size(), in[0].as<const memory&>().argument.size.spatial.size(), in[0].as<const memory&>().argument.size.feature.size())
    , stride(strd)
    , padding(padd) {};

// creates primitive with convolution implementation that supports provided arguments
primitive convolution::create(convolution::arguments arg) {
    try {
        validate_convolution_common_params(arg);
    }
    catch (std::runtime_error err) {
        throw std::runtime_error(std::string("Convolution ") + err.what());
    }

    // wrap relu into RAII wrapper
    std::unique_ptr<convolution> result(new convolution(arg));

    // create implementation for non-lazy evaluation
    if(0 == (arg.engine & engine::lazy)) {
        // lookup in database; throw if not found
        conv_fw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
        auto it = conv_fw_implementation_map.find(key);
        if(it==std::end(conv_fw_implementation_map)) throw std::runtime_error("Not yet implemented.");

        // create implementation & attach it to result
        auto implementation = it->second(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }

    // release RAII wrapper, return naked pointer
    return result.release();
}
primitive convolution_backward::create(convolution_backward::arguments arg) {
    auto& bw_input_size = arg.input_size;  // todo output or input?
    auto& bw_input_offset = arg.input_offset;

    assert(1 == bw_input_size.feature.size());
    assert(1 == bw_input_size.batch.size());

    auto& bw_input_arg = arg.input[0].primitive.as<const memory&>().argument;
    auto& fw_input_arg = arg.input[1].primitive.as<const memory&>().argument;
    auto& filter_arg = arg.input[2].primitive.as<const memory&>().argument;
    auto& bias_arg = arg.input[3].primitive.as<const memory&>().argument;

    auto& bw_output_arg = arg.output[0].as<const memory&>().argument;
    auto& filter_diff_arg = arg.output[1].as<const memory&>().argument;
    auto& bias_diff_arg = arg.output[2].as<const memory&>().argument;

    auto& stride = arg.stride;

    if (bw_input_offset.raw.size() != bw_output_arg.size.raw.size())    throw std::runtime_error("Backward convolution bw_input_offset/bw_output number of dimension does not match.");
    if (bw_input_size.raw.size() != bw_output_arg.size.raw.size())      throw std::runtime_error("Backward convolution bw_input/bw_output number of dimension does not match.");
    if (stride.raw.size() != bw_output_arg.size.raw.size())             throw std::runtime_error("Backward convolution stride/bw_output number of dimension does not match.");
    if (bw_input_size.raw.size() != fw_input_arg.size.raw.size())       throw std::runtime_error("Backward convolution bw_input/fw_output number of dimension does not match.");
    if (filter_arg.size.raw.size() != bw_output_arg.size.raw.size())    throw std::runtime_error("Backward convolution filter size/bw_output number of dimension does not match.");
    if (filter_arg.size.raw.size() != filter_diff_arg.size.raw.size())  throw std::runtime_error("Backward convolution weights/weights_diff number of dimension does not match.");
    if (bw_input_arg.format != bw_output_arg.format)                    throw std::runtime_error("Backward convolution bw_input/bw_output data format does not match.");
    if (bw_input_arg.format != filter_arg.format)                       throw std::runtime_error("Backward convolution bw_input/weights data format does not match.");
    if (bw_input_arg.format != fw_input_arg.format)                     throw std::runtime_error("Backward convolution bw_input/fw_output data format does not match.");
    if (bias_arg.size.raw.size() != 3 &&
        bias_arg.size.batch[0] != 1 &&
        bias_arg.size.feature[0] != 1)                                  throw std::runtime_error("Backward convolution biases isn't 1D vector.");
    if (bias_arg.size.raw.size() != bias_diff_arg.size.raw.size())      throw std::runtime_error("Backward convolution bias/bias_diff number dimensions doesn't match.");
    if (bias_arg.size.spatial[0] != bw_input_arg.size.feature[0])       throw std::runtime_error("Backward convolution biases/bw_input dimensions does not match.");
    if (bias_arg.size != bias_diff_arg.size)                            throw std::runtime_error("Backward convolution bias/bias_diff size doesn't match.");

    auto& bw_output_offset = arg.output_offset;

    for (size_t i = 0; i < bw_output_offset.raw.size(); ++i) {
        if (bw_input_arg.size.raw[i] < bw_input_size.raw[i] + bw_output_offset.raw[i])
            throw std::runtime_error("Backward convolution bw_input buffer size is too small.");

        if (bw_output_arg.size.raw[i] != fw_input_arg.size.raw[i])
            throw std::runtime_error("Sizes of BW output and FW input buffers in convolution bw must be equal.");
    }

    // wrap relu into RAII wrapper
    std::unique_ptr<convolution_backward> result(new convolution_backward(arg));

    // create implementation for non-lazy evaluation
    if(0 == (arg.engine & engine::lazy)) {
        // lookup in database; throw if not found
        conv_bw_key key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
        auto it = conv_bw_implementation_map.find(key);
        if(it==std::end(conv_bw_implementation_map)) throw std::runtime_error("Not yet implemented.");

        // create implementation & attach it to result
        auto implementation = it->second(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }

    // release RAII wrapper, return naked pointer
    return result.release();
}
}

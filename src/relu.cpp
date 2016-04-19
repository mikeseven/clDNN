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
#include <tuple>
#include <map>
#include <functional>

namespace neural {

namespace {
struct relu_reference : is_an_implementation {
    const relu &outer;
    relu_reference(relu &arg)
        : is_an_implementation(neural::type_id<relu_reference>())
        , outer(arg)
    {};
    ~relu_reference() {}

    static void implementation(const void *ptr) {
        auto this_relu = static_cast<const relu *>(ptr);

        auto input_offset  = this_relu->argument.input_offset;
        auto output_offset = this_relu->argument.output_offset;
        auto output_size   = this_relu->argument.output_size;

        auto input_arg  = this_relu->input_memory(0).argument;
        auto output_arg = this_relu->output_memory(0).argument;

        if(input_arg.format != memory::format::yxfb_f32)  throw std::runtime_error("ReLU reference uses yxfb_f32 format.");
        if(input_arg.size.size() != output_arg.size.size())throw std::runtime_error("ReLU input/output number of dimension does not match.");
        if(input_arg.format != output_arg.format)  throw std::runtime_error("ReLU input/output data format does not match.");
        for(auto &x : input_offset.raw)  if(x < 0)                   throw std::runtime_error("ReLU negative input offset.");

        // todo: another data order in std::vector and neural::vector
        //for(size_t i = 0; i < input_arg.size.size(); ++i){
        //    if(input_arg.size[i]  < output_size.raw[i] +  input_offset.raw[i]) throw std::runtime_error("ReLU input/output size does not match.");
        //    if(output_arg.size[i] < output_size.raw[i] + output_offset.raw[i]) throw std::runtime_error("ReLU sizes to small.");
        //}

        auto input  = static_cast<float*>(this_relu->input_memory(0).pointer);
        auto output = static_cast<float*>(this_relu->output_memory(0).pointer);

        namespace nd = ndimensional;
        nd::value<uint32_t> range ( {output_size.raw[2], output_size.raw[3], output_size.raw[1], output_size.raw[0] });
        nd::calculate_idx<uint32_t> calc_in_idx  (input_arg.size);
        nd::calculate_idx<uint32_t> calc_out_idx (output_arg.size);

        std::vector<int32_t>  input_offset_reordered  { input_offset.raw[2],  input_offset.raw[3],  input_offset.raw[1],  input_offset.raw[0]  };
        std::vector<uint32_t> output_offset_reordered {output_offset.raw[2], output_offset.raw[3], output_offset.raw[1], output_offset.raw[0] };
        for(auto pos : range) {
            auto in_idx  = calc_in_idx (pos + input_offset_reordered );
            auto out_idx = calc_out_idx(pos + output_offset_reordered);

            output[out_idx] = std::max( input[in_idx], 0.0f) + this_relu->argument.negative_slope * std::min( input[in_idx], 0.0f);
        }
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(relu &arg) { return new relu_reference(arg); };
};

struct relu_backward_reference : is_an_implementation {
    const relu_backward &outer;
    relu_backward_reference(relu_backward &arg)
        : is_an_implementation(neural::type_id<relu_backward_reference>())
        , outer(arg)
    {};
    ~relu_backward_reference() {}

    static void implementation(const void *ptr)
    {
        auto this_relu = static_cast<const relu_backward *>(ptr);

        if(this_relu->input().size() != 2)
            throw std::runtime_error("ReLU backward: number of inputs is incorrect.");

        if(this_relu->output().size() != 1)
            throw std::runtime_error("ReLU backward: number of outputs is incorrect.");

        auto forward_output_grad = static_cast<float*>(this_relu->input_memory(0).pointer);
        auto forward_input       = static_cast<float*>(this_relu->input_memory(1).pointer);
        auto forward_input_grad  = static_cast<float*>(this_relu->output_memory(0).pointer);

        auto forward_output_grad_arg    = this_relu->input_memory(0).argument;
        auto forward_output_grad_sizes   = forward_output_grad_arg.size;
        auto forward_output_grad_offset = this_relu->argument.input_offset[0];

        auto forward_input_arg    = this_relu->input_memory(1).argument;
        auto forward_input_sizes   = forward_input_arg.size;
        auto forward_input_offset = this_relu->argument.input_offset[1];

        auto forward_input_grad_arg    = this_relu->output_memory(0).argument;
        auto forward_input_grad_sizes   = forward_input_grad_arg.size;
        auto forward_input_grad_offset = this_relu->argument.output_offset;

        auto processed_window_sizes = this_relu->argument.output_size;

        if(forward_output_grad_sizes.size() != forward_input_sizes.size() || forward_input_sizes.size() != forward_input_grad_sizes.size())
            throw std::runtime_error("ReLU backward: number of IO dimension does not match.");

        if(forward_output_grad_arg.format != forward_input_arg.format || forward_input_arg.format != forward_input_grad_arg.format)
            throw std::runtime_error("ReLU backward: IO data format does not match.");

        for(size_t i = 0; i < forward_output_grad_sizes.size(); ++i){
            if(forward_output_grad_sizes[i] < processed_window_sizes[i] + forward_output_grad_offset[i]) throw std::runtime_error("ReLU backward: forward_output_grad size does not match the offset.");
            if(forward_input_sizes[i] < processed_window_sizes[i] + forward_input_offset[i])             throw std::runtime_error("ReLU backward: forward_input size does not match the offset.");
            if(forward_input_grad_sizes[i] < processed_window_sizes[i] + forward_input_grad_offset[i])   throw std::runtime_error("ReLU backward: forward_input_grad size does not match the offset.");
        }

        namespace nd = ndimensional;
        nd::value<uint32_t> range (processed_window_sizes);
        nd::calculate_idx<uint32_t> calc_forward_input_idx(forward_input_sizes);
        nd::calculate_idx<uint32_t> calc_forward_output_grad_idx(forward_output_grad_sizes);
        nd::calculate_idx<uint32_t> calc_forward_input_grad_idx(forward_input_grad_sizes);
        for(auto pos : range) {
            auto forward_input_idx  = calc_forward_input_idx (pos + forward_input_offset);
            auto forward_output_grad_idx = calc_forward_output_grad_idx(pos + forward_output_grad_offset);
            auto forward_input_grad_idx = calc_forward_input_grad_idx(pos + forward_input_grad_offset);

            forward_input_grad[forward_input_grad_idx] = (forward_input[forward_input_idx] <= 0.0f ? 0.0f : 1.0f) * forward_output_grad[forward_output_grad_idx];
        }
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(relu_backward &arg) { return new relu_backward_reference(arg); };
};

} // namespace {

//todo discuss, output size is always needed or can be uninitialized?
relu::arguments::arguments( neural::engine::type engine, primitive out, vector<uint32_t> out_off, vector<uint32_t> out_siz, primitive in, vector<int32_t> in_off, float slp)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}

relu::arguments::arguments( neural::engine::type engine, primitive out, vector<uint32_t> out_off, vector<uint32_t> out_siz, primitive in, vector<int32_t> in_off)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope() {}

relu::arguments::arguments( neural::engine::type engine, primitive out, primitive in, float slp )
    : engine(engine)
    , output({out})
    , output_offset(1, static_cast<uint32_t>(out.as<const memory&>().argument.size.size())-2, 1)
    , output_size({out.as<const memory&>().argument.size[3], {out.as<const memory&>().argument.size[0], out.as<const memory&>().argument.size[1]}, out.as<const memory&>().argument.size[2]})
    , input({in})
    , input_offset(1, static_cast<int32_t>(in.as<const memory&>().argument.size.size())-2, 1)
    , negative_slope(slp) {}

relu::arguments::arguments( neural::engine::type engine, primitive out, primitive in )
    : engine(engine)
    , output({out})
    , output_offset(1, static_cast<uint32_t>(out.as<const memory&>().argument.size.size())-2, 1)
    , output_size({out.as<const memory&>().argument.size[3], {out.as<const memory&>().argument.size[0], out.as<const memory&>().argument.size[1]}, out.as<const memory&>().argument.size[2]})
    , input({in})
    , input_offset(1, static_cast<int32_t>(in.as<const memory&>().argument.size.size())-2, 1)
    , negative_slope(0.0f) {}

relu_backward::arguments::arguments(neural::engine::type engine, std::vector<primitive> out, std::vector<uint32_t> out_offset, std::vector<uint32_t> out_size, std::vector<primitive_at> in, std::vector<std::vector<uint32_t>> in_offsets, float neg_slope)
    : engine(engine)
    , output(out)
    , output_offset(out_offset)
    , output_size(out_size)
    , input(in)
    , input_offset(in_offsets)
    , negative_slope(neg_slope) {}

relu_backward::arguments::arguments(neural::engine::type engine, std::vector<primitive> out, std::vector<primitive_at> in, float neg_slope)
    : engine(engine)
    , output(out)
    , output_offset(static_cast<uint32_t>(out[0].as<const memory&>().argument.size.size()))
    , output_size(out[0].as<const memory&>().argument.size.begin(), out[0].as<const memory&>().argument.size.end())
    , input(in)
    , input_offset(in.size(), std::vector<uint32_t>(in[0].primitive.as<const memory&>().argument.size.size(), 0))
    , negative_slope(neg_slope) {}

//                                    engine                output                        input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(relu &)>> forward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), relu_reference::create}
};
// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(relu_backward &)>> backward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), relu_backward_reference::create}
};
// creates primitive with relu implementation that supports provided arguments
primitive relu::create(relu::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<relu> result(new relu(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = forward_implementation_map.find(key);
    if(it==std::end(forward_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}
// creates primitive with relu implementation that supports provided arguments
primitive relu_backward::create(relu_backward::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<relu_backward> result(new relu_backward(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = backward_implementation_map.find(key);
    if(it==std::end(backward_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

}

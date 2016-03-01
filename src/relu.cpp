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
        auto input     = static_cast<float*>(this_relu->input_memory(0).pointer);
        auto output    = static_cast<float*>(this_relu->output_memory(0).pointer);

        auto input_memory_arg  = this_relu->input_memory(0).argument;
        auto input_buffer_size = input_memory_arg.size;
        auto input_offset      = this_relu->argument.input_offset;

        auto output_memory_arg = this_relu->output_memory(0).argument;
        auto output_buffer_size= output_memory_arg.size;
        auto output_offset     = this_relu->argument.output_offset;
        auto output_size       = this_relu->argument.output_size;

        if(input_memory_arg.format != memory::format::yxfb_f32)  throw std::runtime_error("ReLU reference uses yxfb_f32 format.");
        if(input_buffer_size.size() != output_buffer_size.size())throw std::runtime_error("ReLU input/output number of dimension does not match.");
        if(input_memory_arg.format != output_memory_arg.format)  throw std::runtime_error("ReLU input/output data format does not match.");
        for(auto &x : input_offset)  if(x < 0)                   throw std::runtime_error("ReLU negative input offset.");

        for(size_t i = 0; i < input_buffer_size.size(); ++i){
            if(input_buffer_size[i]  < output_size[i] + input_offset[i])  throw std::runtime_error("ReLU input/output size does not match.");
            if(output_buffer_size[i] < output_size[i] + output_offset[i]) throw std::runtime_error("ReLU sizes to small.");
        }

        namespace nd = ndimensional;
        nd::value<uint32_t> range (output_size);
        nd::calculate_idx<uint32_t> calc_in_idx  (input_buffer_size);
        nd::calculate_idx<uint32_t> calc_out_idx (output_buffer_size);
        for(auto pos : range) {
            auto in_idx  = calc_in_idx (pos + input_offset );
            auto out_idx = calc_out_idx(pos + output_offset);

            output[out_idx] = std::max( input[in_idx], 0.0f) + this_relu->argument.negative_slope * std::min( input[in_idx], 0.0f);
        }
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(relu &arg) { return new relu_reference(arg); };
};

//                                    engine                output                        input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(relu &)>> forward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), relu_reference::create}
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
        auto forward_output_grad_size   = forward_output_grad_arg.size;
        auto forward_output_grad_offset = this_relu->argument.input_offset[0];

        auto forward_input_arg    = this_relu->input_memory(1).argument;
        auto forward_input_size   = forward_input_arg.size;
        auto forward_input_offset = this_relu->argument.input_offset[1];

        auto forward_input_grad_arg    = this_relu->output_memory(0).argument;
        auto forward_input_grad_size   = forward_input_grad_arg.size;
        auto forward_input_grad_offset = this_relu->argument.output_offset;

        auto window_size = this_relu->argument.output_size;

        if(forward_output_grad_size.size() != forward_input_size.size() || forward_input_size.size() != forward_input_grad_size.size()) 
            throw std::runtime_error("ReLU backward: number of IO dimension does not match.");

        if(forward_output_grad_arg.format != forward_input_arg.format || forward_input_arg.format != forward_input_grad_arg.format)
            throw std::runtime_error("ReLU backward: IO data format does not match.");

        for(size_t i = 0; i < forward_output_grad_size.size(); ++i){
            if(forward_output_grad_size[i] < window_size[i] + forward_output_grad_offset[i]) throw std::runtime_error("ReLU backward: forward_output_grad size does not match the offset.");
            if(forward_input_size[i] < window_size[i] + forward_input_offset[i])             throw std::runtime_error("ReLU backward: forward_input size does not match the offset.");
            if(forward_input_grad_size[i] < window_size[i] + forward_input_grad_offset[i])   throw std::runtime_error("ReLU backward: forward_input_grad size does not match the offset.");
        }

        std::vector<uint32_t> counter(window_size.size() - 1, 0);

        std::vector<uint32_t> acc(forward_input_offset.size());

        while(!counter_finished(window_size, counter))
        {
            // calculate offset without most frequently changing dimension to reduce function calls
            // most changing dimension has linear layout in memory
            std::transform(counter.begin(), counter.end(), forward_input_offset.begin(), acc.begin(), std::plus<uint32_t>());
            auto forward_input_ptr = forward_input + calculate_offset(forward_input_size, acc) + forward_input_offset.back();

            std::transform(counter.begin(), counter.end(), forward_output_grad_offset.begin(), acc.begin(), std::plus<uint32_t>());
            auto forward_output_grad_ptr = forward_output_grad + calculate_offset(forward_output_grad_size, acc) + forward_output_grad_offset.back();

            std::transform(counter.begin(), counter.end(), forward_input_grad_offset.begin(), acc.begin(), std::plus<uint32_t>());
            auto forward_input_grad_ptr = forward_input_grad + calculate_offset(forward_input_grad_size, acc) + forward_input_grad_offset.back();

            // relu backprop on linear buffer
            for (uint32_t i = 0; i < window_size.back() ; ++i) 
                forward_input_grad_ptr[i] = (forward_input_ptr[i] <= 0.0f ? 0.0f : 1.0f) * forward_output_grad_ptr[i];

            counter_increase(window_size, counter);
        }
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(relu_backward &arg) { return new relu_backward_reference(arg); };
};

//                                    engine                output                        input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(relu_backward &)>> backward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), relu_backward_reference::create}
};

} // namespace {

//todo discuss, output size is always needed or can be uninitialized?
relu::arguments::arguments( neural::engine::type engine, primitive out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, float slp)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}

relu::arguments::arguments( neural::engine::type engine, primitive out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off)
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
    , output_offset({out.as<const memory&>().argument.size.size()})
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset({in.as<const memory&>().argument.size.size()})
    , negative_slope(slp) {}

relu::arguments::arguments( neural::engine::type engine, primitive out, primitive in )
    : engine(engine)
    , output({out})
    , output_offset({out.as<const memory&>().argument.size.size()})
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset({in.as<const memory&>().argument.size.size()})
    , negative_slope(0.0f) {}

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

relu_backward::arguments::arguments(neural::engine::type engine, std::vector<primitive> out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, std::vector<primitive_at> in, std::vector<std::vector<uint32_t>> in_off, float slp)
    : engine(engine)
    , output(out)
    , output_offset(out_off)
    , output_size(out_siz)
    , input(in)
    , input_offset(in_off)
    , negative_slope(slp) {}

relu_backward::arguments::arguments(neural::engine::type engine, std::vector<primitive> out, std::vector<primitive_at> in, float slp)
    : engine(engine)
    , output(out)
    , output_offset({out[0].as<const memory&>().argument.size.size()})
    , output_size(out[0].as<const memory&>().argument.size.begin(), out[0].as<const memory&>().argument.size.end())
    , input(in)
    , input_offset(in.size(), std::vector<uint32_t>(in[0].primitive.as<const memory&>().argument.size.size(), 0))
    , negative_slope(slp) {}

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
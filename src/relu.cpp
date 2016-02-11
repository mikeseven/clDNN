#include "neural.h"
#include <algorithm>

namespace neural {

namespace {

struct relu_reference : is_a_unknown {
    const relu &outer;
    relu_reference(relu &arg)
        : is_a_unknown(neural::type_id<relu_reference>()) 
        , outer(arg) 
    {};
    ~relu_reference() {}

    static void implementation(const void *ptr) {
        auto this_relu = static_cast<const relu *>(ptr);
        auto input     = static_cast<float*>(this_relu->input_memory(0).pointer);
        auto output    = static_cast<float*>(this_relu->output_memory(0).pointer);

        auto input_memory_arg  = this_relu->input_memory(0).argument;
        auto output_memory_arg = this_relu->output_memory(0).argument;
        auto input_offset      = this_relu->argument.input_offset;
        auto output_offset     = this_relu->argument.output_offset;
        auto output_size       = this_relu->argument.output_size;

        if(input_memory_arg.size.size() != output_memory_arg.size.size()) throw std::runtime_error("ReLU input/output number of dimension does not match.");
        if(input_memory_arg.format      != output_memory_arg.format)      throw std::runtime_error("ReLU input/output data format does not match.");

        for(unsigned i = 0; i < input_memory_arg.size.size(); ++i)
            if(input_memory_arg.size[i] - input_offset[i] != output_size[i]) throw std::runtime_error("ReLU input/output size does not match.");
        
        std::vector<size_t> counter( output_size.size(), 0 );



        size_t count_src = 1;
        for (size_t i = 0; i < count_src; ++i)
        output[i] = std::max(input[i], 0.0f) + this_relu->argument.negative_slope * std::min(input[i], 0.0f);
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }
};

} // namespace {
//todo discuss, output size is always needed or can be uninitialized?
relu::arguments::arguments(neural::engine engine, primitive out, std::vector<uint32_t>out_off, std::vector<uint32_t>out_siz, primitive in, std::vector<int32_t>in_off, float slp)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}

relu::arguments::arguments(neural::engine engine, primitive out, std::vector<uint32_t>out_off, std::vector<uint32_t>out_siz, primitive in, std::vector<int32_t>in_off)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope() {}

relu::arguments::arguments( neural::engine engine, primitive out, std::vector<uint32_t>out_off, primitive in, std::vector<int32_t>in_off, float slp )
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}// todo output_size initialization

relu::arguments::arguments( neural::engine engine, primitive out, std::vector<uint32_t>out_off, primitive in, std::vector<int32_t>in_off )
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , input({in})
    , input_offset({in_off})
    , negative_slope() {} // todo output_size initialization

relu::arguments::arguments( neural::engine arg_engine, primitive arg_output, primitive arg_input, float arg_neg_slope )
    : engine(arg_engine)
    , output({arg_output})
    , input({arg_input})
    , negative_slope(arg_neg_slope) {} //todo offsets zero initialization, output_size initialization

relu::arguments::arguments( neural::engine arg_engine, primitive arg_output, primitive arg_input )
    : engine(arg_engine)
    , output({arg_output})
    , input({arg_input})
    , negative_slope(0.0f) {} //todo offsets zero initialization, output_size initialization

primitive relu::create(relu::arguments arg) {
    relu *result = new relu(arg);
    if(    arg.engine==engine::reference
        && memory::format::yxfb_f32==result-> input_memory(0).argument.format
        && memory::format::yxfb_f32==result->output_memory(0).argument.format)
    {
        auto implementation = new relu_reference(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }
    arg.engine;

    return result;
}

}
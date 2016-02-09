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
        
        size_t count_src = 1;
        size_t count_dst = 1;

        for(unsigned i = 0; i < this_relu->argument.input[0].primitive.as<const neural::memory &>().argument.size.size(); ++i)
            count_src *= this_relu->argument.input[0].primitive.as<const neural::memory &>().argument.size[i];
        for(unsigned i = 0; i < this_relu->argument.output[0].as<const neural::memory &>().argument.size.size(); ++i)
            count_dst *= this_relu->argument.output[0].as<const neural::memory &>().argument.size[i];

        if( count_dst != count_src )
            throw std::runtime_error("ReLU input/output size does not match.");

        for (size_t i = 0; i < count_src; ++i) {
        output[i] = std::max(input[i], .0f)
            + this_relu->argument.negative_slope * std::min(input[i], .0f);
        }
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }
};

} // namespace {

relu::arguments::arguments( neural::engine arg_engine, neural::primitive arg_output, neural::primitive arg_input )
    : engine(arg_engine)
    , output({arg_output})
    , input({arg_input})
    , negative_slope(0.0f) {}

relu::arguments::arguments( neural::engine arg_engine, neural::primitive arg_output, neural::primitive arg_input, float arg_neg_slope )
    : engine(arg_engine)
    , output({arg_output})
    , input({arg_input})
    , negative_slope(arg_neg_slope) {}

primitive relu::create(relu::arguments arg) {
    relu *result = new relu(arg);
    //if(    arg.engine==engine::reference
    //    && memory::format::yxfb_f32==result-> input_memory(0).argument.format
    //    && memory::format::yxfb_f32==result->output_memory(0).argument.format)
    if(    arg.engine==engine::reference)
    if( memory::format::yxfb_f32==result-> input_memory(0).argument.format)
    if( memory::format::yxfb_f32==result->output_memory(0).argument.format)    {
        auto implementation = new relu_reference(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }
    arg.engine;

    return result;
}

}
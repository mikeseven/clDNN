#include "neural.h"
#include <algorithm>

namespace neural {

namespace {

struct relu_reference : is_a_unknown
{
    relu_reference() : is_a_unknown(neural::type_id<relu_reference>()) {}
    ~relu_reference() {}

    static void implementation(void *const ptr) {
        auto argument = static_cast<relu::arguments *>(ptr);

        size_t count_src = 1;
        size_t count_dst = 1;

        for(unsigned i = 0; i < argument->input[0].primitive.as<neural::memory *>()->argument.size.size(); ++i)
            count_src *= argument->input[0].primitive.as<neural::memory *>()->argument.size[i];
        for(unsigned i = 0; i < argument->output[0].as<neural::memory *>()->argument.size.size(); ++i)
            count_dst *= argument->output[0].as<neural::memory *>()->argument.size[i];

        if( count_dst != count_src )
            throw std::runtime_error("ReLU input/output size does not match.");

        auto input  = static_cast<float *>(argument->input[0].primitive.output[0].as<const neural::memory *>()->pointer);
        auto output = static_cast<float *>(argument->output[0].output[0].as<const neural::memory *>()->pointer);

        for (size_t i = 0; i < count_src; ++i) {
        output[i] = std::max(input[i], .0f)
            + argument->negative_slope * std::min(input[i], .0f);
        }
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

    auto tmp = new relu_reference();

    if(arg.engine==engine::reference
       && memory::format::xyzb==arg.input[0].primitive.output[0].as<neural::memory *>()->argument.format)
    {
        result->_private.reset(new relu_reference());
        result->_work.push_back({relu_reference::implementation, (void *const)&result->argument});
    }
    arg.engine;
    
    // [WIP]

    return result;
}

}
#include "neural.h"

namespace neural {

namespace {

struct relu_reference : is_a_unknown
{
    relu_reference() : is_a_unknown(neural::type_id<relu_reference>()) {}
    ~relu_reference() {}

    static void implementation(void *const ptr) {
        auto argument = reinterpret_cast<relu::arguments *>(ptr);

        auto count = argument->input[0].primitive.as<neural::memory *>()->argument.size[0];

        if( count != argument->output[0].as<neural::memory *>()->argument.size[0] )
            throw std::runtime_error("ReLU input/output size does not match.");

        //auto top_data = argument->input[0].primitive.input

        //for (int i = 0; i < count; ++i) {
        //top_data[i] = std::max(bottom_data[i], Dtype(0))
        //    + negative_slope * std::min(bottom_data[i], Dtype(0));
        //}
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
        //result->_private.reset(new relu_reference());
        //result->_work.push_back({relu_reference::implementation, (void *const)&result->argument});
    }
    //arg.engine;
    

    // [WIP]

    return result;
}

}
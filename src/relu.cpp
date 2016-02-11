#include "neural.h"
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
        auto input_vec  =  this_relu->input_memory(0).argument.size;
        auto output_vec =  this_relu->output_memory(0).argument.size;

        size_t count_src = 1;
        size_t count_dst = 1;
        for(auto x : input_vec ) count_src *= x;
        for(auto x : output_vec) count_dst *= x;

        if( count_dst != count_src )
            throw std::runtime_error("ReLU input/output size does not match.");

        for (size_t i = 0; i < count_src; ++i)
            output[i] = std::max(input[i], 0.0f) + this_relu->argument.negative_slope * std::min(input[i], 0.0f);
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(relu &arg) { return new relu_reference(arg); };
};

//                                    engine          output                  input
using implementation_key = std::tuple<neural::engine, neural::memory::format, neural::memory::format>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(relu &)>> implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), relu_reference::create}
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




// creates primitive with relu implementation that supports provided arguments
primitive relu::create(relu::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<relu> result(new relu(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = implementation_map.find(key);
    if(it==std::end(implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

}
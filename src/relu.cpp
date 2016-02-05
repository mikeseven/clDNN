#include "neural.h"

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
        auto that = static_cast<const relu *>(ptr);
        auto input  = that->input_memory(0).pointer;
        auto output = that->output_memory(0).pointer;
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }
};

} // namespace {


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
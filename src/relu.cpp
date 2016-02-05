#include "neural.h"

namespace neural {

namespace {

struct relu_reference : is_a_unknown {
    const relu::arguments &argument;
    relu_reference(relu::arguments &arg)
        : is_a_unknown(neural::type_id<relu_reference>()) 
        , argument(arg) 
    {};
    ~relu_reference() {}

    static void implementation(const void *ptr) {
        const relu::arguments &argument = *static_cast<const relu::arguments *>(ptr);
        auto input  = argument.input[0].primitive.output[0].as<const neural::memory *>()->pointer;
        auto output = argument.output[0].as<const neural::memory *>()->pointer;
    }

    std::vector<task> work() {
        return {task{implementation, &argument}};
    }
};

} // namespace {


primitive relu::create(relu::arguments arg) {
    relu *result = new relu(arg);

    if(    arg.engine==engine::reference
        && memory::format::yxfb_f32==arg.input[0].primitive.output[0].as<const neural::memory *>()->argument.format
        && memory::format::yxfb_f32==arg.output[0].as<const neural::memory *>()->argument.format)
    {
        auto implementation = new relu_reference(arg);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }
    arg.engine;

    return result;
}

}
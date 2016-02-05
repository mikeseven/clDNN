#include "neural.h"

namespace neural {

namespace {

struct relu_reference : is_a_unknown
{
    relu_reference() : is_a_unknown(neural::type_id<relu_reference>()) {};
    ~relu_reference() {}

    static void implementation(void *const ptr) {
        auto argument = reinterpret_cast<relu::arguments *>(ptr);
    }
};

} // namespace {


primitive relu::create(relu::arguments arg) {
    relu *result = new relu(arg);

    //if(arg.engine==engine::reference
    //    && memory::format::xyzb==arg.input[0].primitive.output[0].as<neural::memory *>()->argument.format)
    //{
    //    result->_private.reset(new relu_reference());
    //    result->_work.push_back({relu_reference::implementation, (void *const)&result->argument});
    //}
    //arg.engine;
    

    // [WIP]

    return result;
}

}
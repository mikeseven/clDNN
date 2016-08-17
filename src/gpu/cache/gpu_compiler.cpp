#include "gpu_compiler.h"
// TODO add forwarding to cl2_wrapper.h
// include cl2_wrapper.h

namespace neural { namespace gpu { namespace cache {

namespace {
    
code inject_jit(const jit& compile_options, const code& code)
{
    return compile_options + code; //TODO temporary untill we merge proper mechanism
}

}

binary_data gpu_compiler::compile(context* context, const jit& compile_options, const code& code)
{
    static_cast<void>(context);
    return binary_data(inject_jit(compile_options, code)); //TODO temporary untill we merge proper mechanism
}

} } }

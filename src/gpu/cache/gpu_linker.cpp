#include "gpu_linker.h"

namespace neural { namespace gpu { namespace manager {

gpu_program gpu_linker::link(context * context, const std::vector<cache::binary_data>& kernels)
{
    static_cast<void>(context);
    static_cast<void>(kernels);
    return gpu_program(); //TODO return something valid
}

} } }

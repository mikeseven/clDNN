#include "kernel_manager.h"

namespace neural { namespace gpu { namespace manager {

gpu_program manager::kernel_manager::get(context* context, const std::vector<std::pair<jit, primitive_id>>& primitives)
{
    std::vector<cache::binary_data> kernels;
    for (const auto& p : primitives) { kernels.push_back(selector.get(context, p.first, p.second)); }
    return gpu_linker::link(context, kernels);
}

} } }

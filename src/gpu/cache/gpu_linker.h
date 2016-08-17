#pragma once

#include "cache_types.h"
#include "manager_types.h"

#include <vector>

namespace neural { namespace gpu { namespace manager {

/// \brief Class wrapping compile feature of kernel device compiler
///
struct gpu_linker
{
    static gpu_program link(context *context, const std::vector<cache::binary_data>& kernels);

};

} } }
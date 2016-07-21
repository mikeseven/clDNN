#pragma once

#include "primitive_selector.h"
#include "gpu_linker.h"

namespace neural { namespace gpu { namespace manager {

/// \brief Class building gpu programs out of best available 
/// kernels for a list of primitives accompaniated by jit
///
struct kernel_manager
{
	kernel_manager( ) = default;
	gpu_program get(context* context, const std::vector<std::pair<jit, primitive_id>>& primitives);

private:
	primitive_selector selector;
};

} } }

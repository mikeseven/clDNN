#pragma once

#include "cache_types.h"

namespace neural { namespace gpu { namespace cache {

/// \brief Class wrapping compile feature of kernel device compiler
/// 
struct gpu_compiler
{
	static binary_data compile(context* context, const jit& compile_options, const code& code);
};

} } }

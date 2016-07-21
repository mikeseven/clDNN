#pragma once

#include "cache_types.h"

namespace neural { namespace gpu { namespace cache {

/// \brief Class providing a cost for kernel that satisfies properties for 
/// strict ordering allowing choosing the "best" kernel binary
///
struct cost_model
{
	struct cost
	{
		cost(size_t value);
		cost() = default;
		bool operator<(const cost& rhs) const;
	private:
		size_t value; //TODO might need more elaborated state
	};

	static cost rate(const binary_data& kernel_binary);
};

} } }
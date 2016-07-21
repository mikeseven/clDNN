#pragma once

#include <string>
#include <utility>
#include <unordered_map>
#include "serialization.h"
#include "persistent_cache.h"
#include "gpu_compiler.h"
#include "cost_model.h"

namespace neural { namespace gpu { namespace cache {

/// \brief Class that provides transparent cache/compiler interface for collecting compilation results 
///
class cache
{
public:
	cache();
	~cache();

	using type = std::pair<binary_data, cost_model::cost>;

	type get(context* context, kernel kernel);

private:
	binary_cache kernel_binaries;
	persistent_cache file_cache;
};

} } }
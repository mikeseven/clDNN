#pragma once

#include "cache_types.h"

namespace neural { namespace cache {

/// \brief Utility class providing serialize and deserialize mechanisms for cache state
///
struct serialization
{
	static binary_data serialize(const binary_cache& prog);
	static binary_cache deserialize(const binary_data& data);
};

} }
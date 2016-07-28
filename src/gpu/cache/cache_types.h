#pragma once

#include <string>
#include <unordered_map>
#include "common_types.h"

namespace neural { namespace gpu { namespace cache {

using binary_data = std::string;
static_assert(sizeof(binary_data::value_type) == 1, "Binary data has to represent byte array");

using binary_cache = std::unordered_map<size_t, binary_data>;
using kernel = std::pair<jit, code>;

} } }
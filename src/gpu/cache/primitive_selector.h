#pragma once

#include "primitive_db.h"
#include "cache.h"

namespace neural { namespace manager {

/// \brief Class that selects a best binary using ordering provided by cost model
///
struct primitive_selector
{
	primitive_selector( );

	cache::binary_data get(context* context, const jit& jit, const primitive_id& id);

private:
	primitive_db db;
	cache::cache binary_cache;
};

} }
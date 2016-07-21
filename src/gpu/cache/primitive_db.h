#pragma once

#include <unordered_map>
#include <vector>
#include "common_types.h"
#include "manager_types.h"

/// \brief Class providing interface to retrieve a list of primitive implementations per primitive id
///
namespace neural { namespace gpu { namespace manager {

struct primitive_db
{
	primitive_db( );

	std::vector<code> get(const primitive_id& id);

private:
	std::unordered_multimap<primitive_id, code> primitives;
};

} } }
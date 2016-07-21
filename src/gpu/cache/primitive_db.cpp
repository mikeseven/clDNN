#include "primitive_db.h"
#include <algorithm>

namespace neural { namespace gpu { namespace manager {

primitive_db::primitive_db() : primitives({
	#include "primitive_db.inc"
}) { }

std::vector<code> primitive_db::get(const primitive_id & id)
{
	auto codes = primitives.equal_range(id);
	std::vector<code> temp;
	std::for_each(codes.first, codes.second, [&](auto c){ temp.push_back(c.second); });
	return temp;
}

} } }

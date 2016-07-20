#include "primitive_selector.h"
#include <algorithm>

namespace neural { namespace manager {

primitive_selector::primitive_selector() : binary_cache(), db() { }

cache::binary_data primitive_selector::get(context* context, const jit & jit, const primitive_id & id)
{
	auto codes = db.get(id);
	std::pair<cache::binary_data, cache::cost_model::cost> best;
	for (const auto& c : codes) { best = std::max(best, binary_cache.get(context, std::make_pair(jit, c)),
												  [](auto l, auto r){ return l.second < r.second; }); }
	return best.first;
}

} }
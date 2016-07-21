#include "cache.h"

namespace neural { namespace gpu { namespace cache {

static const char* cache_file_name = ""; //TODO building name

cache::cache() : file_cache(cache_file_name), kernel_binaries(serialization::deserialize(file_cache.get())) { }

cache::~cache() { file_cache.set(serialization::serialize(kernel_binaries)); }

cache::type cache::get(context* context, kernel kernel)
{
	std::hash<std::string> h;
	size_t hash = h(kernel.first + kernel.second);
	auto it = kernel_binaries.find(hash);
	auto binary = (it == kernel_binaries.end()) ?
		(*kernel_binaries.insert(std::make_pair(hash, gpu_compiler::compile(context, kernel.first, kernel.second))).first).second :
		(*it).second;
	return std::make_pair(binary, cost_model::rate(binary));
}

} } }
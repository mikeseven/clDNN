#include "cache.h"

namespace neural { namespace gpu { namespace cache {

static const char* cache_file_name = "cl_dnn_cache.intel"; //TODO building name

cache::cache() : file_cache(cache_file_name), kernel_binaries(serialization::deserialize(file_cache.get())) { }

cache::~cache()
{
    if (dirty)
    {
        file_cache.set(serialization::serialize(kernel_binaries));
    }
}

cache::type cache::get(context* context, kernel kernel)
{
    std::hash<std::string> h;
    size_t hash = h(kernel.first + kernel.second);
    auto it = kernel_binaries.find(hash);

    binary_data binary;

    if (it == kernel_binaries.end())
    {
        dirty = true;
        binary = gpu_compiler::compile(context, kernel.first, kernel.second);
        kernel_binaries[hash] = binary;

    }
    else
    {
        binary = it->second;
    }

    return std::make_pair(binary, cost_model::rate(binary));
}

} } }
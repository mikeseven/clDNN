#pragma once

#include "cache_types.h"

namespace neural { namespace gpu { namespace cache {

/// \brief Class providing persistent cache (in file) functionality for our kernel binary base
///
class persistent_cache
{
public:
	persistent_cache(const char * cache_file_name);
	~persistent_cache() = default;

	binary_data get();
	void set(binary_data);

private:
	struct cache_file
	{
		cache_file(const char* file_name);
		~cache_file() = default;
		binary_data read();
		void write(const binary_data&);
	private:
		const char* cache_file_name;
	} file;
};

} } }
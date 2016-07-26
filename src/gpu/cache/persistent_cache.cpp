#include "persistent_cache.h"
#include <fstream>
#include <sstream>

namespace neural { namespace gpu { namespace cache {

persistent_cache::persistent_cache(const char* cache_file_name) : file(cache_file_name) { }

binary_data persistent_cache::get() { return file.read(); }

void persistent_cache::set(binary_data data) { file.write(data); }

persistent_cache::cache_file::cache_file(const char* file_name) : cache_file_name(file_name) { }


binary_data persistent_cache::cache_file::read()
{
	std::ifstream c_file(cache_file_name, std::ios::binary);
	if (c_file.is_open())
	{
		std::stringstream data;
		data << c_file.rdbuf();
		c_file.close();
		return data.str();
	}
	throw(errno);
}

void persistent_cache::cache_file::write(const binary_data& data)
{
	std::ofstream c_file(cache_file_name, std::ios::binary);
	if (c_file.is_open())
	{
		c_file << data;
		c_file.close();
		return;
	}
	throw (errno);
}

} } }
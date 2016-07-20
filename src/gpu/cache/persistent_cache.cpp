#include "persistent_cache.h"
#include <fstream>
#include <sstream>

namespace neural { namespace cache {

persistent_cache::persistent_cache(const char* cache_file_name) : file(cache_file_name) { }

binary_data persistent_cache::get() { return file.read(); }

void persistent_cache::set(binary_data data) { file.write(data); }

persistent_cache::cache_file::cache_file(const char* file_name) : cache_file_name(file_name) { }


binary_data persistent_cache::cache_file::read()
{
	std::ifstream file(cache_file_name, std::ios::binary);
	if (file.is_open())
	{
		std::stringstream data;
		data << file.rdbuf();
		file.close();
		return data.str();
	}
	throw(errno);
}

void persistent_cache::cache_file::write(const binary_data& data)
{
	std::ofstream file(cache_file_name, std::ios::binary);
	if (file.is_open())
	{
		file << data;
		file.close();
		return;
	}
	throw (errno);
}

} }
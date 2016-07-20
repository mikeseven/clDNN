#include "serialization.h"

namespace neural { namespace cache {

namespace {

template <class T>
static void push_back_as_binary(binary_data& binary, const T& value)
{
	// we convert value to an array of binary data elements and insert that at the end of the binary vector
	binary.insert(binary.end(), reinterpret_cast<const binary_data::value_type *>(&value),
		reinterpret_cast<const binary_data::value_type *>((&value) + 1));
}

template <class It>
static void push_back_as_binary(binary_data& binary, It begin, It end)
{
	// we convert value to an array of binary data elements and insert that at the end of the binary vector
	binary.insert(binary.end(), reinterpret_cast<const binary_data::value_type *>(begin),
		reinterpret_cast<const binary_data::value_type *>(end));
}

}

binary_data serialization::serialize(const binary_cache& kernels)
{
	binary_data result;
	auto count = kernels.size();
	// we store data in file in the following way: kernels count, (hash, binary_size, binary), (...)...
	push_back_as_binary(result, count);
	for (const auto& hk : kernels)
	{
		push_back_as_binary(result, hk.first);
		auto size = hk.second.size();
		push_back_as_binary(result, size);
		push_back_as_binary(result, &hk.second[0], &hk.second[0] + size);
	}
	return result;
}

namespace {

template <class T>
static T read_as_binary(const binary_data& data, size_t& offset)
{
	static_assert(sizeof(T) % sizeof(binary_data::value_type) == 0, "Size of T has to be divisible by size of binary data element");
	if (data.size() < (offset += sizeof(T) / sizeof(binary_data::value_type))) throw std::out_of_range("binary data ended unexpectedly");
	return *reinterpret_cast<const T*>(&data[offset - sizeof(T) / sizeof(binary_data::value_type)]);
}

static binary_data read_as_binary(const binary_data& data, size_t& offset, size_t count)
{
	if (data.size() < offset + count) throw std::out_of_range("binary data ended unexpectedly");
	offset += count;
	return binary_data(data.begin() + offset - count, data.begin() + offset);
}

}

// we store data in file in the following way: kernels count, (hash, binary_size, binary), (...)...
binary_cache serialization::deserialize(const binary_data& data)
{
	binary_cache ret;
	size_t offset = 0;
	size_t count = read_as_binary<size_t>(data, offset);
	for (size_t i = 0; i < count; ++i)
	{
		size_t hash = read_as_binary<size_t>(data, offset);
		size_t size = read_as_binary<size_t>(data, offset);
		ret[hash] = read_as_binary(data, offset, size);
	}
	return ret;
}

} }
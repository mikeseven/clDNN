#pragma once
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include "api\neural.h"

namespace ndimensional {

template<typename T>
class value_fmt : public std::vector<T> {
    template<typename U> struct change_signedness;
    template<>           struct change_signedness<uint32_t> { using type =  int32_t; };
    template<>           struct change_signedness< int32_t> { using type = uint32_t; };
    template<>           struct change_signedness<uint64_t> { using type =  int64_t; };
    template<>           struct change_signedness< int64_t> { using type = uint64_t; };
    using negT = typename change_signedness<T>::type;

public:
	neural::memory::format::type format;
// Iterator represents number in number system in which maximum value_fmt of each digit at index 'i'
// [denoted _current.at(i)] is limited by corresponding value.at(i).
// When during incrementation _current(i)==value_fmt(i) digit at position 'i' it overflows with carry over to the left.
// It means that digit at 'i' is zeroed and digit at 'i-1' is incremented.
// The least significant digit is on the last(max index) position of the vector.
    class iterator {
        value_fmt<T> _current;
        const value_fmt &_ref;
        iterator(const value_fmt &arg, bool is_begin)
            : _current(is_begin ? value_fmt(arg.format,arg.size()) : arg)
            , _ref(arg) {};
        friend class value_fmt<T>;
    public:
        iterator(const iterator &) = default;
        iterator(iterator &&) = default;
        ~iterator() = default;
        iterator& operator=(const iterator &) = default;
        iterator& operator=(iterator &&) = default;
        iterator& operator++() {
            for(size_t at=_current.size(); at--;) {
                if(++_current[at] == _ref[at])
                    _current[at]=0;
                else
                    return *this;
            }
            _current = _ref;
            return *this;
        }
        const value_fmt<T> &operator*() const { return _current; }
        iterator operator++(int) { iterator result(*this); ++(*this); return result; }
        bool operator==(const iterator &rhs) const { return &_ref==&rhs._ref && _current==rhs._current; }
        bool operator!=(const iterator &rhs) const { return !(*this==rhs); }
    };

    iterator begin() { return iterator(*this, true); }
    iterator end()   { return iterator(*this, false); }

    value_fmt(size_t size = 0 ) : std::vector<T>(size, T(0)) {};
	value_fmt(neural::memory::format::type _fmt, size_t size) : format(_fmt), std::vector<T>(size, T(0)) {};
    value_fmt(neural::memory::format::type _fmt, std::vector<T> arg) : format(_fmt), std::vector<T>(arg) {};
    value_fmt(neural::memory::format::type _fmt, std::initializer_list<T> il) : format(_fmt), std::vector<T>(il) {};

    //todo range check asserts
    value_fmt &operator+=(std::vector<   T> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::plus<T>());       return *this; }
    value_fmt &operator+=(std::vector<negT> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::plus<T>());       return *this; }
    value_fmt &operator*=(std::vector<   T> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::multiplies<T>()); return *this; }
    value_fmt &operator*=(std::vector<negT> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::multiplies<T>()); return *this; }
    value_fmt  operator+ (std::vector<   T> &arg) { value_fmt result=*this; return result+=arg; }
    value_fmt  operator+ (std::vector<negT> &arg) { value_fmt result=*this; return result+=arg; }
    value_fmt  operator* (std::vector<   T> &arg) { value_fmt result=*this; return result*=arg; }
    value_fmt  operator* (std::vector<negT> &arg) { value_fmt result=*this; return result*=arg; }

    template<typename T> friend std::ostream &operator<<(std::ostream &, value_fmt &);
};

template<typename T>
std::ostream &operator<<(std::ostream &out, value_fmt<T> &val) {
    for(size_t at = 0; at < val.size(); ++at)
        out << val[at] << (at + 1 == val.size() ? "" : ",");
    return out;
}

template<typename T>
class calculate_idx_fmt {
	std::vector<T> size;
	std::vector<T> stride;

public:
	neural::memory::format::type format;
	calculate_idx_fmt( neural::memory::format::type _fmt, const std::vector<T>& v_size )
	: format(_fmt)
	,size(v_size)
	, stride(v_size) {

    stride.emplace_back(1); //this element is used in operator()
    for(size_t i = stride.size() - 1; i > 0; --i)
        stride[i-1] *= stride[i];
};
	size_t operator() ( const value_fmt<T>& pos );
};

template<typename T>
size_t calculate_idx_fmt<T>::operator()( const value_fmt<T>& _position ){
	size_t result_idx = 0;
	
	value_fmt<T> position(0);
	for(int i =0; i< _position.size();++i){
		auto o = neural::memory::traits(this->format).order[i];
		for(int j =0; j <  _position.size();++j)
		{
			auto p = neural::memory::traits(_position.format).order[j];
			if(o == p)
				position.push_back(_position[j]);
		}
	}

	assert(
		[&]() -> bool {
		if(position.size() <= 0)
			return false;
		for(size_t i = 0; i < position.size(); ++i)
			if(size[i] <= position[i]) return false;

			return true;
		}() == true );

	// Number of iterations depends on length of position vector.
	// 'position' can be shorter than 'size' because last numbers (with the highest indexes) coressponds data with linear memory layout.
	// If 'position' is shorter than 'size' than function returns offset to some block of data
	for(size_t i = 0; i != position.size(); ++i){
		auto idx = position.size() - 1 - i;
		result_idx += stride[idx+1] * position[idx];
	};

	return result_idx;
}
};



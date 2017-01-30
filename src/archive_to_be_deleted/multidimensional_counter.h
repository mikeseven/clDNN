/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <string>
#include "neural.h"

namespace ndimensional {

template<typename U> struct change_signedness;
template<> struct change_signedness< int32_t> { using type = uint32_t; };
template<> struct change_signedness<uint32_t> { using type =  int32_t; };
template<> struct change_signedness<uint64_t> { using type =  int64_t; };
template<> struct change_signedness< int64_t> { using type = uint64_t; };

template<typename T>
class value : public std::vector<T> {
    using negT = typename change_signedness<T>::type;
public:
// Iterator represents number in number system in which maximum value of each digit at index 'i'
// [denoted _current.at(i)] is limited by corresponding value.at(i).
// When during incrementation _current(i)==value(i) digit at position 'i' it overflows with carry over to the left.
// It means that digit at 'i' is zeroed and digit at 'i-1' is incremented.
// The least significant digit is on the last(max index) position of the vector.
    class iterator {
        value<T> _current;
        const value &_ref;
        iterator(const value &arg, bool is_begin)
            : _current(is_begin ? value(arg.size()) : arg)
            , _ref(arg) {};
        friend class value<T>;
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
        const value<T> &operator*() const { return _current; }
        iterator operator++(int) { iterator result(*this); ++(*this); return result; }
        bool operator==(const iterator &rhs) const { return &_ref==&rhs._ref && _current==rhs._current; }
        bool operator!=(const iterator &rhs) const { return !(*this==rhs); }
    };

    iterator begin() { return iterator(*this, true); }
    iterator end()   { return iterator(*this, false); }

    value(size_t size) : std::vector<T>(size, T(0)) {};
    value(std::vector<T> arg) : std::vector<T>(arg) {};
    value(neural::vector<T> arg) : std::vector<T>(arg.raw) {};
    value(std::initializer_list<T> il) : std::vector<T>(il) {};

    value &operator+=(const std::vector<   T> &arg) { std::transform(arg.crbegin(), arg.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::plus <T>());      return *this; }
    value &operator+=(const std::vector<negT> &arg) { std::transform(arg.crbegin(), arg.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::plus <T>());      return *this; }
    value &operator-=(const std::vector<   T> &arg) { std::transform(arg.crbegin(), arg.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::minus<T>());      return *this; }
    value &operator-=(const std::vector<negT> &arg) { std::transform(arg.crbegin(), arg.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::minus<T>());      return *this; }
    value &operator*=(const std::vector<   T> &arg) { std::transform(arg.crbegin(), arg.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::multiplies<T>()); return *this; }
    value &operator*=(const std::vector<negT> &arg) { std::transform(arg.crbegin(), arg.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::multiplies<T>()); return *this; }
    value  operator+ (const std::vector<   T> &arg) { value result=*this; return result+=arg; }
    value  operator+ (const std::vector<negT> &arg) { value result=*this; return result+=arg; }
    value  operator- (const std::vector<   T> &arg) { value result=*this; return result-=arg; }
    value  operator- (const std::vector<negT> &arg) { value result=*this; return result-=arg; }
    value  operator* (const std::vector<   T> &arg) { value result=*this; return result*=arg; }
    value  operator* (const std::vector<negT> &arg) { value result=*this; return result*=arg; }

    value &operator+=(const neural::vector<   T> &arg) { std::transform(arg.raw.crbegin(), arg.raw.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::plus <T>());      return *this; }
    value &operator+=(const neural::vector<negT> &arg) { std::transform(arg.raw.crbegin(), arg.raw.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::plus <T>());      return *this; }
    value &operator-=(const neural::vector<   T> &arg) { std::transform(arg.raw.crbegin(), arg.raw.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::minus<T>());      return *this; }
    value &operator-=(const neural::vector<negT> &arg) { std::transform(arg.raw.crbegin(), arg.raw.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::minus<T>());      return *this; }
    value &operator*=(const neural::vector<   T> &arg) { std::transform(arg.raw.crbegin(), arg.raw.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::multiplies<T>()); return *this; }
    value &operator*=(const neural::vector<negT> &arg) { std::transform(arg.raw.crbegin(), arg.raw.crend(), std::vector<T>::rbegin(), std::vector<T>::rbegin(), std::multiplies<T>()); return *this; }
    value  operator+ (const neural::vector<   T> &arg) { value result = *this; return result += arg; }
    value  operator+ (const neural::vector<negT> &arg) { value result = *this; return result += arg; }
    value  operator- (const neural::vector<   T> &arg) { value result = *this; return result -= arg; }
    value  operator- (const neural::vector<negT> &arg) { value result = *this; return result -= arg; }
    value  operator* (const neural::vector<   T> &arg) { value result = *this; return result *= arg; }
    value  operator* (const neural::vector<negT> &arg) { value result = *this; return result *= arg; }

    template<typename U> friend std::ostream &operator<<(std::ostream &, ndimensional::value<U> &);
};

template<typename U>
std::ostream &operator<<(std::ostream &out, value<U> &val) {
    for(size_t at = 0; at < val.size(); ++at)
        out << val[at] << (at + 1 == val.size() ? "" : ",");
    return out;
}

// todo rename or remove fptr
typedef size_t (*fptr)(std::vector<uint32_t> size, std::vector<uint32_t> pos);
DLL_SYM fptr choose_calculate_idx(neural::memory::format::type arg);

inline bool is_out_of_range(const std::vector<uint32_t> size, const std::vector<uint32_t> pos){
    assert( pos.size() == size.size() );

    for(size_t i = 0; i < pos.size(); ++i)
        if(pos[i] >= size[i])
            return true;

    return false;
}
inline bool is_out_of_range(const std::vector<uint32_t> size, const std::vector<int32_t> pos){
    for(size_t i = 0; i < pos.size(); ++i)
        if(pos[i] < 0)
            return true;

    std::vector<uint32_t> pos_u {pos.begin(), pos.end()};
    return is_out_of_range(size, pos_u);
}
inline bool is_out_of_range(const neural::vector<uint32_t> size, const std::vector<uint32_t> pos){
    return is_out_of_range(size.raw, pos);
}
inline bool is_out_of_range(const neural::vector<uint32_t> size, const std::vector<int32_t> pos){
    return is_out_of_range(size.raw, pos);
}

}

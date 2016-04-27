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

    value &operator+=(const std::vector<   T> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::plus<T>());       return *this; }
    value &operator+=(const std::vector<negT> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::plus<T>());       return *this; }
    value &operator*=(const std::vector<   T> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::multiplies<T>()); return *this; }
    value &operator*=(const std::vector<negT> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::multiplies<T>()); return *this; }
    value  operator+ (const std::vector<   T> &arg) { value result=*this; return result+=arg; }
    value  operator+ (const std::vector<negT> &arg) { value result=*this; return result+=arg; }
    value  operator* (const std::vector<   T> &arg) { value result=*this; return result*=arg; }
    value  operator* (const std::vector<negT> &arg) { value result=*this; return result*=arg; }

    value &operator+=(const neural::vector<   T> &arg) { std::transform(arg.raw.cbegin(), arg.raw.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::plus<T>());       return *this; }
    value &operator+=(const neural::vector<negT> &arg) { std::transform(arg.raw.cbegin(), arg.raw.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::plus<T>());       return *this; }
    value &operator*=(const neural::vector<   T> &arg) { std::transform(arg.raw.cbegin(), arg.raw.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::multiplies<T>()); return *this; }
    value &operator*=(const neural::vector<negT> &arg) { std::transform(arg.raw.cbegin(), arg.raw.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::multiplies<T>()); return *this; }
    value  operator+ (const neural::vector<   T> &arg) { value result=*this; return result+=arg.raw; }
    value  operator+ (const neural::vector<negT> &arg) { value result=*this; return result+=arg.raw; }
    value  operator* (const neural::vector<   T> &arg) { value result=*this; return result*=arg.raw; }
    value  operator* (const neural::vector<negT> &arg) { value result=*this; return result*=arg.raw; }

    template<typename U> friend std::ostream &operator<<(std::ostream &, ndimensional::value<U> &);
};

template<typename U>
std::ostream &operator<<(std::ostream &out, value<U> &val) {
    for(size_t at = 0; at < val.size(); ++at)
        out << val[at] << (at + 1 == val.size() ? "" : ",");
    return out;
}

////////////////////////////////////////////////////////////////
template<typename T>
class calculate_idx_interface{
protected:
    using negT = typename change_signedness<T>::type;

    std::vector<T> size;
    std::vector<T> stride;

    calculate_idx_interface( const std::vector<T>& v_size ) : size(v_size), stride(v_size.size()) {
        static_assert(std::is_unsigned<T>::value, "calculate_idx_interface<T> constructor accepts only unsigned types");
    };
    virtual ~calculate_idx_interface() = 0;

public:
    size_t operator() ( const std::vector<   T>& pos );
    size_t operator() ( const std::vector<negT>& pos );
    size_t operator() ( const neural::vector<   T>& pos );
    size_t operator() ( const neural::vector<negT>& pos );

    bool is_out_of_range( const std::vector<   T>& pos );
    bool is_out_of_range( const std::vector<negT>& pos );
    bool is_out_of_range( const neural::vector<   T>& pos );
    bool is_out_of_range( const neural::vector<negT>& pos );
};
template<typename T>
inline calculate_idx_interface<T>::~calculate_idx_interface(){}
template<typename T>
inline size_t calculate_idx_interface<T>::operator()( const std::vector<T>& position ){
    size_t result_idx = 0;

    assert(
        [&]() -> bool {
        for(size_t i = 0; i < position.size(); ++i)
            if(size[i] <= position[i]) return false;

          return true;
        }() == true );
    assert(position.size() == size.size());

    for(size_t i = 0; i != position.size(); ++i){
        result_idx += stride[i] * position[i];
    };

    return result_idx;
}
template<typename T>
inline size_t calculate_idx_interface<T>::operator()( const std::vector<negT>& position ){
    size_t result_idx = 0;

    assert(
        [&]() -> bool {
        for(size_t i = 0; i < position.size(); ++i)
            if(size[i] <= static_cast<T>(position[i])) return false;

          return true;
        }() == true );
    assert(position.size() == size.size());

    for(size_t i = 0; i != position.size(); ++i){
        result_idx += stride[i] * position[i];
    };

    return result_idx;
}
template<typename T>
inline bool calculate_idx_interface<T>::is_out_of_range( const std::vector<negT>& pos ){
    assert( pos.size() == size.size() );

    for(uint32_t i = 0; i < pos.size(); ++i)
        if(static_cast<T>(pos[i]) >= size[i])
            return true;

    return false;
}
template<typename T>
inline bool calculate_idx_interface<T>::is_out_of_range( const std::vector<T>& pos ){
    assert( pos.size() == size.size() );

    for(uint32_t i = 0; i < pos.size(); ++i)
        if(pos[i] >= size[i])
            return true;

    return false;
}


// todo tmp solution, should we just pass pointer to stride table? It can be done in runtime
template<typename T, neural::memory::format::type FORMAT>
class calculate_idx : public calculate_idx_interface<T>  {
public:
    calculate_idx( const std::vector<T>& v_size ) {
        throw std::runtime_error("Template specialization of calculate_idx is not implemented for (int)memory::format: " + std::to_string(FORMAT) );
    }
};

template<typename T>
class calculate_idx<T, neural::memory::format::yxfb_f32> : public calculate_idx_interface<T>{
public:
    calculate_idx( const neural::vector<T>& v_size ) : calculate_idx( v_size.raw ) {};
    calculate_idx( const std::vector<T>& v_size )
    : calculate_idx_interface<T>(v_size) {

        assert( 4 == v_size.size() );
        assert( 0 != std::accumulate(v_size.cbegin(), v_size.cend(), 1, std::multiplies<T>()));

        // strides for yxfb format
        // vectors v_size and stride use format: b, f, spatials(y,x...)
        this->stride[0] = 1;
        this->stride[1] = v_size[0];
        this->stride[2] = v_size[0] * v_size[1] * v_size[3];
        this->stride[3] = v_size[0] * v_size[1];
    };
};
template<typename T>
class calculate_idx<T, neural::memory::format::xb_f32> : public calculate_idx_interface<T>{
public:
    calculate_idx( const neural::vector<T>& v_size ) : calculate_idx( v_size.raw ) {};
    calculate_idx( const std::vector<T>& v_size )
    : calculate_idx_interface<T>(v_size) {

        assert( 3 == v_size.size() ); // b, f=1, spatial(x)
        assert( 1 == v_size[1] );     // 1 feature map, just for compatibility
        assert( 0 != std::accumulate(v_size.cbegin(), v_size.cend(), 1, std::multiplies<T>()));

        // strides for xb format
        // vectors v_size and stride use format: b, f, spatial(x)
        this->stride[0] = 1;
        this->stride[2] = v_size[0];
    };
};
template<typename T>
class calculate_idx<T, neural::memory::format::x_f32> : public calculate_idx_interface<T>{
public:
    calculate_idx( const neural::vector<T>& v_size ) : calculate_idx( v_size.raw ) {};
    calculate_idx( const std::vector<T>& v_size )
    : calculate_idx_interface<T>(v_size) {

        assert( 3 == v_size.size() ); // b, f=1, spatial(x)
        assert( 1 == v_size[0] );
        assert( 1 == v_size[1] );     // 1 feature map, just for compatibility
        assert( 0 != v_size[2] );

        // strides for xb format
        // vectors v_size and stride use format: b, f, spatial(x)
        this->stride[2] = 1;
    };
};

/////////////////////////
template<neural::memory::format::type FORMAT>
size_t index(std::vector<uint32_t> size, std::vector<uint32_t> pos);

template<> DLL_SYM
size_t index<neural::memory::format::yxfb_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos);

//template<neural::memory::format::type FORMAT>
//size_t index(std::vector<uint32_t> size, std::vector<int32_t> pos){
//    assert(
//    [&]() -> bool {
//    for(size_t i = 0; i < pos.size(); ++i)
//        if(pos[i] < 0) return false;
//
//    return true;
//    }() == true );
//
//    return index<FORMAT>(size, std::vector<uint32_t>{pos.begin(), pos.end()});
//};

// todo pos vector int or uint?
// todo rename or remove fptr
typedef size_t (*fptr)(std::vector<uint32_t> size, std::vector<uint32_t> pos);
DLL_SYM fptr choose_calucalte_idx(neural::memory::format::type arg);

/////////////////////////
template<typename T>
class calculate_idx_obselote{
    using negT = typename change_signedness<T>::type;

    std::vector<T> size;
    std::vector<T> stride;
public:
    calculate_idx_obselote( const std::vector<T>& v_size )
    : size(v_size)
    , stride(v_size) {

        static_assert(std::is_unsigned<T>::value, "calculate_idx_obselote<T> constructor accepts only unsigned types");  //this template should be used only with unsigned types

        stride.emplace_back(1); //this element is used in operator()
        for(size_t i = stride.size() - 1; i > 0; --i)
            stride[i-1] *= stride[i];
    };

    size_t operator() ( const std::vector<   T>& pos );
    size_t operator() ( const std::vector<negT>& pos );

    bool is_out_of_range( const std::vector<   T>& pos );
    bool is_out_of_range( const std::vector<negT>& pos );
};

template<typename T>
inline size_t calculate_idx_obselote<T>::operator()( const std::vector<T>& position ){
    size_t result_idx = 0;

    assert(
        [&]() -> bool {
        for(size_t i = 0; i < position.size(); ++i)
            if(size[i] <= position[i]) return false;

          return true;
        }() == true );

    // Number of iterations depends on length of position vector.
    // 'position' can be shorter than 'size' because last numbers (with the highest indexes) coressponds data with linear memory_obselote layout.
    // If 'position' is shorter than 'size' than function returns offset to some block of data
    for(size_t i = 0; i != position.size(); ++i){
        auto idx = position.size() - 1 - i;
        result_idx += stride[idx+1] * position[idx];
    };

    return result_idx;
}

template<typename T>
inline size_t calculate_idx_obselote<T>::operator()( const std::vector<negT>& position ){
    size_t result_idx = 0;

    assert(
        [&]() -> bool {
        for(size_t i = 0; i < position.size(); ++i)
            if(size[i] <= static_cast<T>(position[i])) return false;

          return true;
        }() == true );

    // Number of iterations depends on length of position vector.
    // 'position' can be shorter than 'size' because last numbers (with the highest indexes) coressponds data with linear memory_obselote layout.
    // If 'position' is shorter than 'size' than function returns offset to some block of data
    for(size_t i = 0; i != position.size(); ++i){
        auto idx = position.size() - 1 - i;
        result_idx += stride[idx+1] * position[idx];
    };

    return result_idx;
}

template<typename T>
inline bool calculate_idx_obselote<T>::is_out_of_range( const std::vector<negT>& pos ){
    assert( pos.size() <= size.size() );

    for(uint32_t i = 0; i < pos.size(); ++i)
        if(static_cast<T>(pos[i]) >= size[i])
            return true;

    return false;
}

template<typename T>
inline bool calculate_idx_obselote<T>::is_out_of_range( const std::vector<T>& pos ){
    assert( pos.size() <= size.size() );

    for(uint32_t i = 0; i < pos.size(); ++i)
        if(pos[i] >= size[i])
            return true;

    return false;
}
}

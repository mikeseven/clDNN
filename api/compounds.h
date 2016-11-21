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
#include <cassert>

namespace cldnn {

template<typename T>
class array_ref
{
public:
    typedef const T* iterator;
    typedef const T* const_iterator;
    typedef size_t size_type;

private:
    const T* _data;
    size_t _size;
public:
    array_ref() :_data(nullptr), _size(0) {}
    array_ref(const T& val) :_data(&val), _size(1) {}
    array_ref(const T* data, size_t size) :_data(data), _size(size) {}

    template<typename A>
    array_ref(const std::vector<T, A>& vec) : _data(vec.data()), _size(vec.size()) {}

    template<size_t N>
    array_ref(const T(&arr)[N]) : _data(arr), _size(N) {}

    array_ref(const array_ref& other) : _data(other._data), _size(other._size) {}

    array_ref& operator=(const array_ref& other)
    {
        if (this == &other)
            return *this;
        _data = other._data;
        _size = other._size;
        return *this;
    }

    const T* data() const { return _data; }
    size_t size() const { return _size; }
    bool empty() const { return _size == 0; }

    iterator begin() const { return _data; }
    iterator end() const { return _data + _size; }
    const_iterator cbegin() const { return _data; }
    const_iterator cend() const { return  _data + _size; }

    const T& operator[](size_t idx) const
    {
        assert(idx < _size);
        return _data[idx];
    }

    std::vector<T> vector() const { return std::vector<T>(_data, _data + _size); }
};

template<typename Char>
size_t basic_strlen(const Char* str) = delete;

template<>
inline size_t basic_strlen<char>(const char* str) { return std::strlen(str); }

template<>
inline size_t basic_strlen<wchar_t>(const wchar_t* str) { return std::wcslen(str); }

template<typename Char>
class basic_string_ref
{
public:
    typedef const Char* iterator;
    typedef const Char* const_iterator;
    typedef size_t size_type;

private:
    const Char* _data;
    size_t _size;
public:
    basic_string_ref() :_data(nullptr), _size(0) {}
    basic_string_ref(const Char* str) : _data(str), _size(basic_strlen(str)) {}

    template<typename T, typename A>
    basic_string_ref(const std::basic_string<Char, T, A>& str) : _data(str.c_str()), _size(str.size()) {}

    basic_string_ref(const basic_string_ref& other) : _data(other._data), _size(other._size) {}

    basic_string_ref& operator=(const basic_string_ref& other)
    {
        if (this == &other)
            return *this;
        _data = other._data;
        _size = other._size;
        return *this;
    }

    const Char* data() const { return _data; }
    const Char* c_str() const { return _data; }
    size_t size() const { return _size; }
    size_t length() const { return _size; }
    bool empty() const { return _size == 0; }

    iterator begin() const { return _data; }
    iterator end() const { return _data + _size; }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    const Char& operator[](size_t idx)
    {
        assert(idx < _size);
        return _data[idx];
    }

    std::basic_string<Char> str() const { return std::basic_string<Char>(_data, _size); }
    operator std::basic_string<Char>() const { return str(); }
};

typedef basic_string_ref<char> string_ref;
typedef basic_string_ref<wchar_t> wstring_ref;

template<typename F, typename S>
struct pair
{
    F first;
    S second;
};

//template<typename T, size_t N>
//class small_vec
//{
//public:
//    typedef const T* iterator;
//    typedef const T* const_iterator;
//    typedef size_t size_type;
//
//private:
//    T      _data[N];
//    size_t _size;
//public:
//    small_vec() : _size(0) {}
//    small_vec(const T& val) :_data{ val }, _size(1)
//    {
//        static_assert(N > 0, "out of range");
//    }
//    small_vec(const T* data, size_t size) : _size(size)
//    {
//        assert(size <= N);
//        for(size_t i = 0; i < size; i++)
//        {
//            _data[i] = data[i];
//        }
//    }
//
//    template<typename A>
//    small_vec(const std::vector<T, A>& vec) : _size(vec.size())
//    {
//        assert(_size < N);
//        std::copy(std::begin(vec), std::end(vec), begin());
//    }
//
//    template<size_t M, typename Dummy = std::enable_if< M<=N >>
//    small_vec(const T(& arr)[M]) : _data(arr), _size(M) {}
//
//    small_vec(const small_vec& other) : _data(other._data), _size(other._size) {}
//
//    small_vec& operator=(const small_vec& other)
//    {
//        if (this == &other)
//            return *this;
//        _data = other._data;
//        _size = other._size;
//        return *this;
//    }
//
//    const T* data() const { return _data; }
//    size_t size() const { return _size; }
//    bool empty() const { return _size == 0; }
//
//    iterator begin() const { return _data; }
//    iterator end() const { return _data + _size; }
//    const_iterator cbegin() const { return _data; }
//    const_iterator cend() const { return  _data + _size; }
//
//    const T& operator[](size_t idx) const
//    {
//        assert(idx < _size);
//        return _data[idx];
//    }
//
//    std::vector<T> vector() const { return std::vector<T>(_data, _data + _size); }
//};


}

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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "cldnn_defs.h"
#include "compounds.h"
#include "tensor.hpp"

#include <algorithm>
#include <string>
#include <vector>


namespace cldnn
{
struct padding
{
    float filling_value() const { return _filling_value; }

    /// Gets lower padding sizes. For spatials, it means size of left (X) and top (Y) padding.
    ///
    /// @return Tensor with padding for top/left/lower bounds of data.
    tensor lower_size() const { return _lower_size; }
    /// Gets upper padding sizes. For spatials, it means size of right (X) and bottom (Y) padding.
    ///
    /// @return Tensor with padding for bottom/right/upper bounds of data.
    tensor upper_size() const { return _upper_size; }
    /// Gets format of tensors used in padding.
    cldnn::format format() const { return _lower_size.format; }

    padding(cldnn::format format, const std::vector<tensor::value_type>& lower_sizes, const std::vector<tensor::value_type>& upper_sizes, float filling_value = 0.0f)
        : _lower_size(format, 0, to_abs(lower_sizes)), _upper_size(format, 0, to_abs(upper_sizes)), _filling_value(filling_value)
    {}

    padding(cldnn::format format, const std::vector<tensor::value_type>& sizes, float filling_value = 0.0f)
        : padding(format, sizes, sizes, filling_value)
    {}

    padding(): padding(format::x, {0}) {}

    padding(const cldnn_padding& other)
        : _lower_size(other.lower_size), _upper_size(other.upper_size), _filling_value(other.filling_value)
    {}

    operator cldnn_padding() const
    {
        return { static_cast<cldnn_tensor>(_lower_size),
                 static_cast<cldnn_tensor>(_upper_size),
                 _filling_value };
    }

    // returns true if padding size is not zero
    explicit operator bool() const
    {
        return std::any_of(_lower_size.raw.begin(), _lower_size.raw.end(), [](const tensor::value_type& el) { return el != 0; }) ||
               std::any_of(_upper_size.raw.begin(), _upper_size.raw.end(), [](const tensor::value_type& el) { return el != 0; });
    }

    padding max(const padding& padd) const
    {
        padding result = *this;
        result._size = _size.max(padd._size);
        return result;
    }

private:
    tensor _lower_size;    ///< Lower padding sizes. For spatials, it means size of left (X) and top (Y) padding.
    tensor _upper_size;    ///< Upper padding sizes. For spatials, it means size of right (X) and bottom (Y) padding.
    // TODO: Add support for non-zero filling value (if necessary) or remove variable (if not necessary).
    float  _filling_value; ///< Filling value for an element of padding. If data type of elements is different than float it is converted
                           ///< to it using round-towards-nearest-even (for floating-point data types) or round-towards-zero (for integral
                           ///< data types).

    static std::vector<tensor::value_type> to_abs(const std::vector<tensor::value_type>& sizes)
    {
        std::vector<tensor::value_type> result;
        result.reserve(sizes.size());
        std::transform(sizes.cbegin(), sizes.cend(), std::back_inserter(result), [](const tensor::value_type& el) { return abs(el); });
        return result; // NRVO
    }
};

CLDNN_API_CLASS(padding)

// TODO check this:
using primitive_type_id = cldnn_primitive_type_id;
using primitive_id_ref = cldnn_primitive_id;
using primitive_id = std::string;

namespace details
{
class primitive_id_arr
{
private:
    template <class CppStr, class CStr>
    struct primitive_id_arr_ref
    {
        primitive_id_arr_ref(CppStr& cppstr, CStr& cstr) : myself(cppstr, cstr)
        {}

        std::add_const_t<CppStr>& operator *() const { return *myself.first; }

        template <class... Dummy, class Guard = std::enable_if_t<!std::is_const<CppStr>::value>>
        CStr& operator *() { return *myself.first; }

        template <class... Dummy, class Guard = std::enable_if_t<!std::is_const<CppStr>::value>>
        primitive_id_arr_ref& operator=(primitive_id const& id)
        {
            myself.first = id;
            myself.second = myself.first.c_str();
            return *this;
        }

        template <class... Dummy, class Guard = std::enable_if_t<!std::is_const<CppStr>::value>>
        primitive_id_arr_ref& operator=(primitive_id&& id)
        {
            myself.first = std::move(id);
            myself.second = myself.first.c_str();
            return *this;
        }

        operator std::add_const_t<CppStr>& () const { return myself.first; }

        //for inter-compatibility with std::reference_wrapper
        // U foo;
        // U bar;
        // T<U> t{ bar };
        // t.get() = foo;
        //where T can be either primitive_id_arr_ref of std::reference_wrapper
        //should in both cases modify wrapped 'bar' object
        decltype(auto) get() { return *this; }
        decltype(auto) get() const { return *this; }

    private:
        std::pair<CppStr&, CStr&> myself;
    };

    template <class CppStr, class CStr>
    struct iterator_base
    {
        using reference = primitive_id_arr_ref<CppStr, CStr>;
        using const_reference = primitive_id_arr_ref<std::add_const_t<CppStr>, std::add_const_t<CStr>>;
        using pointer = CppStr*;
        using const_pointer = std::add_const_t<CppStr>*;
        using value_type = CppStr;
        using difference_type = ptrdiff_t;

        iterator_base(CppStr* cppstr, CStr* cstr) : myself(cppstr, cstr)
        {}

        iterator_base& operator ++() { ++myself.first; ++myself.second; return *this; }
        iterator_base operator ++(int) { auto copy = *this; ++myself.first; ++myself.second; return copy; }

        bool operator ==(iterator_base const& itr) { return myself.first == itr.myself.first; }
        bool operator !=(iterator_base const& itr) { return myself.first != itr.myself.first; }

        const_reference operator *() const { return const_reference{ *myself.first, *myself.second }; }

        template <class... Dummy, class Guard = std::enable_if_t<!std::is_const<CppStr>::value>>
        reference operator *() { return reference{ *myself.first, *myself.second }; }

    private:
        std::pair<CppStr*, CStr*> myself;
    };

public:
    using iterator = iterator_base<primitive_id, cldnn_primitive_id>;
    using const_iterator = iterator_base<const primitive_id, const cldnn_primitive_id>;

public:
    primitive_id_arr(const std::vector<primitive_id>& arr)
        : _data_store(arr)
    {
        update_ref();
    }

    primitive_id_arr(std::vector<primitive_id>&& arr)
        : _data_store(std::move(arr))
    {
        update_ref();
    }

    primitive_id_arr(const cldnn_primitive_id_arr& arr)
        : _data_store(create_store(arr))
    {
        update_ref();
    }

    primitive_id_arr(primitive_id_arr&& other)
        : _data_store(std::move(other)), _ref_store(std::move(other._ref_store))
    {}

    primitive_id_arr(const primitive_id_arr& other)
        : _data_store(other)
    {
        update_ref();
    }

    //allow only const functions
    auto size() const { return _data_store.size(); }
    decltype(auto) back() const { return _data_store.back(); }
    decltype(auto) front() const { return _data_store.front(); }
    decltype(auto) at(size_t idx) const { return _data_store.at(idx); }
    decltype(auto) operator[](size_t idx) const { return _data_store.operator[](idx); }
    auto data() const { return _data_store.data(); }
    auto begin() { return iterator{ _data_store.data(), _ref_store.data() }; }
    auto begin() const { return const_iterator{ _data_store.data(), _ref_store.data() }; }
    auto end() { return iterator{ _data_store.data() + size(), _ref_store.data() + size() }; }
    auto end() const { return const_iterator{ _data_store.data() + size(), _ref_store.data() + size() }; }
    auto cbegin() const { return const_iterator{ _data_store.data(), _ref_store.data() }; }
    auto cend() const { return const_iterator{ _data_store.data() + size(), _ref_store.data() + size() }; }

    //allow modifications through following functions (keeps synchronisation with C API storage)
    void update(size_t pos, primitive_id const& pid)
    {
        assert(pos < size() && "Array element out of range");
        if (pos >= size())
            return;
        _data_store[pos] = pid;
        _ref_store[pos] = at(pos).c_str();
    }

    void update(size_t pos, primitive_id&& pid)
    {
        assert(pos < size() && "Array element out of range");
        if (pos >= size())
            return;
        _data_store[pos] = std::move(pid);
        _ref_store[pos] = at(pos).c_str();
    }

    // explicit conversion
    cldnn_primitive_id_arr ref() const
    {
        return { _ref_store.data(), _ref_store.size() };
    }

    auto const& store() const
    {
        return _data_store;
    }

    operator std::vector<primitive_id> const&() const
    {
        return _data_store;
    }

private:
    std::vector<primitive_id> _data_store;
    std::vector<cldnn_primitive_id> _ref_store;

    void update_ref()
    {
        _ref_store.resize(size());
        for (size_t i = 0; i < size(); ++i)
            _ref_store[i] = at(i).c_str();
    }

    static std::vector<primitive_id> create_store(const cldnn_primitive_id_arr& arr)
    {
        //Fill _data_store by copies of strings referenced in arr
        std::vector<primitive_id> result;
        result.reserve(arr.size);
        for (size_t i = 0; i < arr.size; i++)
        {
            result.push_back(arr.data[i]);
        }
        return std::move(result);
    }
};
} // namespace detail

template<class PType>
typename PType::dto* as_dto(CLDNN_PRIMITIVE_DESC(primitive)* dto)
{
    if (dto->type != PType::type_id()) throw std::invalid_argument("type");
    return reinterpret_cast<typename PType::dto*>(dto);
}
template<class PType>
const typename PType::dto* as_dto(const CLDNN_PRIMITIVE_DESC(primitive)* dto)
{
    if (dto->type != PType::type_id()) throw std::invalid_argument("type");
    return reinterpret_cast<const typename PType::dto*>(dto);
}

struct primitive
{
    primitive(
        const primitive_type_id& type,
        const primitive_id& id,
        const std::vector<primitive_id>& input,
        const padding& input_padding = padding(),
        const padding& output_padding = padding()
    )
        :_type(type), _id(id), _input(input), _input_padding(input_padding), _output_padding(output_padding)
    {}

    primitive(const CLDNN_PRIMITIVE_DESC(primitive)* dto)
        :_type(dto->type), _id(dto->id), _input(dto->input), _input_padding(dto->input_padding), _output_padding(dto->output_padding)
    {}

    virtual ~primitive() = default;

    const primitive_type_id& type() const { return _type; }
    const primitive_id& id() const { return _id; }

    auto& input() { return _input; }
    const auto& input() const { return _input; }
    padding& input_padding() { return _input_padding; }
    const padding& input_padding() const { return _input_padding; }
    padding& output_padding() { return _output_padding; }
    const padding& output_padding() const { return _output_padding; }

    virtual const CLDNN_PRIMITIVE_DESC(primitive)* get_dto() const = 0;

    std::vector<primitive_id> dependecies() const
    {
        auto result = input().store();
        auto deps = get_dependencies();
        result.insert(result.end(), deps.begin(), deps.end());
        return result;
    }

    operator primitive_id() const { return id(); }

    //TODO remove backward compatibility
    tensor input_offset() const { return input_padding().lower_size().negate(); }
    tensor output_offset() const { return output_padding().lower_size(); }
    float padding_filling_value() const { return input_padding().filling_value(); }

protected:
    const primitive_type_id _type;
    const primitive_id _id;

    details::primitive_id_arr _input;
    padding _input_padding;
    padding _output_padding;

    virtual std::vector<primitive_id> get_dependencies() const { return{}; }
};

template<class PType, class DTO>
class primitive_base : public primitive
{
public:
    const CLDNN_PRIMITIVE_DESC(primitive)* get_dto() const override
    {
        update_dto(_dto);
        return reinterpret_cast<const CLDNN_PRIMITIVE_DESC(primitive)*>(&_dto);
    }

protected:
    explicit primitive_base(
        const primitive_id& id,
        const std::vector<primitive_id>& input,
        const padding& input_padding = padding(),
        const padding& output_padding = padding())
        : primitive(PType::type_id(), id, input, input_padding, output_padding)
    {}

    primitive_base(const DTO* dto)
        : primitive(reinterpret_cast<const CLDNN_PRIMITIVE_DESC(primitive)*>(dto))
    {
        if (dto->type != PType::type_id()) 
            throw std::invalid_argument("DTO type mismatch");
    }

    virtual void update_dto(DTO& dto) const
    {
        dto.id = _id.c_str();
        dto.type = _type;
        dto.input = _input.ref();
        dto.input_padding = _input_padding;
        dto.output_padding = _output_padding;
    }
   
private:
    mutable DTO _dto;
};

#define CLDNN_DEFINE_TYPE_ID(PType) static primitive_type_id type_id()\
    {\
        return check_status<primitive_type_id>( #PType " type id failed", [](status_t* status)\
        {\
            return cldnn_##PType##_type_id(status);\
        });\
    }

#define CLDNN_DECLATE_PRIMITIVE(PType) typedef CLDNN_PRIMITIVE_DESC(PType) dto;\
    CLDNN_DEFINE_TYPE_ID(PType)

}

namespace std
{
template <>
struct iterator_traits<cldnn::details::primitive_id_arr::const_iterator>
{
    using iterator_category = input_iterator_tag;

    using reference = cldnn::details::primitive_id_arr::const_iterator::reference;
    using pointer = cldnn::details::primitive_id_arr::const_iterator::pointer;
    using value_type = cldnn::details::primitive_id_arr::const_iterator::value_type;
    using difference_type = cldnn::details::primitive_id_arr::const_iterator::difference_type;
};

template <>
struct iterator_traits<cldnn::details::primitive_id_arr::iterator>
{
    using iterator_category = input_iterator_tag;

    using reference = cldnn::details::primitive_id_arr::const_iterator::reference;
    using pointer = cldnn::details::primitive_id_arr::const_iterator::pointer;
    using value_type = cldnn::details::primitive_id_arr::const_iterator::value_type;
    using difference_type = cldnn::details::primitive_id_arr::const_iterator::difference_type;
};
}

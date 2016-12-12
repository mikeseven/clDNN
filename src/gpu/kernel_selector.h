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

#pragma once

#include "boost/functional/hash.hpp"

#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <unordered_map>


namespace neural
{
/// Marker type that separates required selectors from optional ones in kernel selector signature.
struct optional_selector_t {};


namespace mputils
{
template <typename ... Tys> struct type_tuple;

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy, typename ElemTy> struct count_tt;

template <typename Ty, typename ... Tys, typename ElemTy>
struct count_tt<type_tuple<Ty, Tys ...>, ElemTy>
    : std::integral_constant<std::size_t, count_tt<type_tuple<Tys ...>, ElemTy>::value + static_cast<size_t>(std::is_same<Ty, ElemTy>::value)>
{};

template <typename ElemTy>
struct count_tt<type_tuple<>, ElemTy>
    : std::integral_constant<std::size_t, 0>
{};

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy> struct size_tt;

template <typename ... Tys>
struct size_tt<type_tuple<Tys ...>> : std::integral_constant<size_t, sizeof...(Tys)>
{};

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy, typename ElemTy> struct split_tt;

namespace detail
{
template <typename TypeTupleTy, typename ElemTy, typename FirstTupleTy> struct split_tt_helper1;

template <typename Ty, typename ... Tys, typename ElemTy, typename ... FirstTys>
struct split_tt_helper1<type_tuple<Ty, Tys ...>, ElemTy, type_tuple<FirstTys ...>>
    : split_tt_helper1<type_tuple<Tys ...>, ElemTy, type_tuple<FirstTys ..., Ty>>
{};

template <typename Ty, typename ... Tys, typename ... FirstTys>
struct split_tt_helper1<type_tuple<Ty, Tys ...>, Ty, type_tuple<FirstTys ...>>
{
    using first_type  = type_tuple<FirstTys ...>;
    using second_type = type_tuple<Tys ...>;
};

template <typename ElemTy, typename ... FirstTys>
struct split_tt_helper1<type_tuple<>, ElemTy, type_tuple<FirstTys ...>>
{
    using first_type  = type_tuple<>;
    using second_type = type_tuple<FirstTys ...>;
};
} // namespace detail

template <typename ... Tys, typename ElemTy>
struct split_tt<type_tuple<Tys ...>, ElemTy>
    : detail::split_tt_helper1<type_tuple<Tys ...>, ElemTy, type_tuple<>>
{};

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy, typename ElemTy> struct remove_tt;

namespace detail
{
template <typename TypeTupleTy, typename ElemTy, typename ResultTupleTy> struct remove_tt_helper1;

template <typename Ty, typename ... Tys, typename ElemTy, typename ... ResultTys>
struct remove_tt_helper1<type_tuple<Ty, Tys ...>, ElemTy, type_tuple<ResultTys ...>>
    : remove_tt_helper1<type_tuple<Tys ...>, ElemTy, type_tuple<ResultTys ..., Ty>>
{};

template <typename Ty, typename ... Tys, typename ... ResultTys>
struct remove_tt_helper1<type_tuple<Ty, Tys ...>, Ty, type_tuple<ResultTys ...>>
    : remove_tt_helper1<type_tuple<Tys ...>, Ty, type_tuple<ResultTys ...>>
{};

template <typename ElemTy, typename ... ResultTys>
struct remove_tt_helper1<type_tuple<>, ElemTy, type_tuple<ResultTys ...>>
{
    using type = type_tuple<ResultTys ...>;
};
} // namespace detail

template <typename ... Tys, typename ElemTy>
struct remove_tt<type_tuple<Tys ...>, ElemTy>
    : detail::remove_tt_helper1<type_tuple<Tys ...>, ElemTy, type_tuple<>>
{};

template <typename TypeTupleTy, typename ElemTy>
using remove_tt_t = typename remove_tt<TypeTupleTy, ElemTy>::type;

// -----------------------------------------------------------------------------------------------------------------------

template <template <typename ...> class VariadicTTy, typename TypeTupleTy> struct make_vttype_tt;

template <template <typename ...> class VariadicTTy, typename ... Tys>
struct make_vttype_tt<VariadicTTy, type_tuple<Tys ...>>
{
    using type = VariadicTTy<Tys ...>;
};

template <template <typename ...> class VariadicTTy, typename TypeTupleTy>
using make_vttype_tt_t = typename make_vttype_tt<VariadicTTy, TypeTupleTy>::type;

// -----------------------------------------------------------------------------------------------------------------------

} // namespace mputils


template<typename KernelDataTy, typename OuterTy, typename ... SelectorTys>
class kernel_selector
{
    using _selector_types = mputils::type_tuple<SelectorTys ...>;
    static_assert(mputils::count_tt<_selector_types, optional_selector_t>::value <= 1,
                  "Only one optional selectors separator can be specified. Remove redundant optional_selector_t.");
    using _filtered_selector_types = mputils::remove_tt_t<_selector_types, optional_selector_t>;
    static_assert(mputils::size_tt<_filtered_selector_types>::value > 0,
                  "At least one selector type must be specified (except separators).");

public:
    using key_type = mputils::make_vttype_tt_t<std::tuple, _filtered_selector_types>;
    using hash_type = boost::hash<key_type>;
    using mapped_type = KernelDataTy (*)(const OuterTy&);
    using map_type = std::unordered_map<key_type, mapped_type, hash_type>;
    using value_type = typename map_type::value_type;


private:
    map_type _kernel_map;


public:
    kernel_selector(const std::initializer_list<value_type>& l)
        : _kernel_map(l)
    {}

    /*
    KernelDataTy get_kernel(const OuterTy& arg, SelectorTys ... selectors)
    {
        auto value = _kernel_map.find(std::make_tuple(selectors ...));
        if (value == _kernel_map.end())
        {
            return get_kernel(arg, selectors);
        }

        KernelDataTy kd = value->second(arg);
        return kd;
    }

    KernelDataTy get_kernel(const OuterTy&)
    {
        throw std::runtime_error("ERROR: no default element in map for convolution kernels!");
    }
    */
};

kernel_selector<int, long, int, optional_selector_t, float>::key_type a = 0;

} // namespace neural

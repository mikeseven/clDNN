/*
// Copyright (c) 2017 Intel Corporation
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

#include <type_traits>

namespace cldnn
{
namespace meta
{

//helper struct to tell wheter type T is any of given types U...
//termination case when U... is empty -> return std::false_type
template <class T, class... U>
struct is_any_of : public std::false_type {};

//helper struct to tell whether type is any of given types (U, Rest...)
//recurrence case when at least one type U is present -> returns std::true_type if std::same<T, U>::value is true, otherwise call is_any_of<T, Rest...> recurrently
template <class T, class U, class... Rest>
struct is_any_of<T, U, Rest...> : public std::conditional_t<std::is_same<T, U>::value, std::true_type, is_any_of<T, Rest...>> {};

template <class T, class... U>
constexpr bool is_any_of_v = is_any_of<T, U...>::value;

//helper type for deducing return type from member function pointer
//doesn't require passing arguments like std::result_of
template <class T>
struct deduce_ret_type;

template <class Ret, class C, class... Args>
struct deduce_ret_type<Ret(C::*)(Args...)>
{
    using type = Ret;
};

template <class T>
using deduce_ret_type_t = typename deduce_ret_type<T>::type;

}
}
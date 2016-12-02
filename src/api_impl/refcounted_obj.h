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
#include <atomic>

namespace cldnn
{

/**
 * \brief Base class for all reference counted pointers aka PIMPL implementations
 */
// TODO refine this code for multithreading support
template<class T>
class refcounted_obj
{
public:
    refcounted_obj()
        : _ref_count(1)
    {}

    virtual ~refcounted_obj() = default;

    void add_ref()
    {
        ++_ref_count;
    }

    void release()
    {
        if ((--_ref_count) == 0) delete static_cast<T*>(this);
    }

private:
    std::atomic_int _ref_count;
};
}

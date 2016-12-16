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
#include "cldnn_defs.h"
#include <chrono>
#include <memory>
#include <vector>

namespace cldnn {
namespace instrumentation
{
template<class ClockTy = std::chrono::steady_clock>
class timer {
    typename ClockTy::time_point start_point;
public:
    typedef typename ClockTy::duration val_type;

    timer() :start_point(ClockTy::now()) {}
    val_type uptime() const { return ClockTy::now() - start_point; }
};

struct profiling_period
{
    virtual std::chrono::nanoseconds value() const = 0;
    virtual ~profiling_period() = default;
};

struct profiling_period_basic : profiling_period
{
    template<class _Rep, class _Period>
    profiling_period_basic(const std::chrono::duration<_Rep, _Period>& val) :
        _value(std::chrono::duration_cast<std::chrono::nanoseconds>(val)) {}

    std::chrono::nanoseconds value() const override { return _value; }
private:
    std::chrono::nanoseconds _value;
};

struct profiling_interval {
    std::string name;
    std::shared_ptr<profiling_period> value;
};

struct profiling_info {
    std::string name;
    std::vector<profiling_interval> intervals;
};
}}

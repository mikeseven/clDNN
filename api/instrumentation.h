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
#include "neural.h"
#include <chrono>
#include <sstream>
#include <iomanip>

namespace neural {
namespace gpu {
    struct configuration {
        enum device_types { default_device = 0, cpu, gpu, accelerator };

        DLL_SYM static configuration& get();

        bool enable_profiling;
        device_types device_type;
        uint32_t device_vendor;
        std::string compiler_options;
    private:
        configuration();
    };
}

namespace instrumentation {

    template<class ClockTy = std::chrono::steady_clock>
    class timer {
        typename ClockTy::time_point start_point;
    public:
        typedef typename ClockTy::duration val_type;

        timer() :start_point(ClockTy::now()) {}
        val_type uptime() const { return ClockTy::now() - start_point; }
    };

    template<class Rep, class Period>
    std::string to_string(const std::chrono::duration<Rep, Period> val) {
        namespace  ch = std::chrono;
        const ch::microseconds us(1);
        const ch::milliseconds ms(1);
        const ch::seconds s(1);
        const std::chrono::duration<Rep, Period> abs_val(std::abs(val.count()));

        std::ostringstream os;
        os << std::setprecision(3) << std::fixed;
        if (abs_val > s)       os << ch::duration_cast<ch::duration<double, ch::seconds::period>>(val).count() << " s";
        else if (abs_val > ms) os << ch::duration_cast<ch::duration<double, ch::milliseconds::period>>(val).count() << " ms";
        else if (abs_val > us) os << ch::duration_cast<ch::duration<double, ch::microseconds::period>>(val).count() << " us";
        else               os << ch::duration_cast<ch::nanoseconds>(val).count() << " ns";
        return os.str();
    }

    struct logger
    {
        DLL_SYM static void log_memory_to_file(const primitive&, std::string prefix = "");
    private:
        static const std::string dump_dir;
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
            _value(std::chrono::duration_cast<std::chrono::nanoseconds>(val)){}

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
} }

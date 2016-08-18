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
#include "neural_base.h"
#include <chrono>
#include <sstream>
#include <iomanip>

namespace neural { namespace instrumentation {

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

        std::ostringstream os;
        os << std::setprecision(3) << std::fixed;
        if (val > s)       os << std::chrono::duration_cast<ch::duration<double, ch::seconds::period>>(val).count() << " s";
        else if (val > ms) os << std::chrono::duration_cast<ch::duration<double, ch::milliseconds::period>>(val).count() << " ms";
        else if (val > us) os << std::chrono::duration_cast<ch::duration<double, ch::microseconds::period>>(val).count() << " us";
        else               os << std::chrono::duration_cast<ch::nanoseconds>(val).count() << " ns";
        return os.str();
    }

    struct logger
    {
        DLL_SYM static void log_memory_to_file(const primitive&, std::string prefix = "");
    private:
        static const std::string dump_dir;
    };

} }

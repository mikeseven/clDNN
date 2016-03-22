#pragma once

#include "convolution.h"

namespace neural {
    struct convolution_cpu_jit : is_an_implementation {
        convolution_cpu_jit(convolution &arg);
        ~convolution_cpu_jit();
        static void implementation(const void *ptr);

        static is_an_implementation *create(convolution &arg) { return new convolution_cpu_jit(arg); };
        std::vector<task> work() { return {task{implementation, &outer}}; };

        const convolution &outer;
    };
}

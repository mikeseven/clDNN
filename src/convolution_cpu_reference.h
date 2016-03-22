#pragma once

#include "convolution.h"

namespace neural {
    struct convolution_cpu_reference : is_an_implementation {
        convolution_cpu_reference(convolution &arg);
        ~convolution_cpu_reference();
        static void implementation(const void *ptr);

        static is_an_implementation *create(convolution &arg) { return new convolution_cpu_reference(arg); };
        std::vector<task> work() { return {task{implementation, &outer}}; };

        const convolution &outer;
    };
    struct convolution_backward_cpu_reference : is_an_implementation {
        convolution_backward_cpu_reference(convolution_backward &arg);
        ~convolution_backward_cpu_reference();
        static void implementation(const void *ptr);

        static is_an_implementation *create(convolution_backward &arg) { return new convolution_backward_cpu_reference(arg); };
        std::vector<task> work() { return {task{implementation, &outer}}; };

        const convolution_backward &outer;
    };
}
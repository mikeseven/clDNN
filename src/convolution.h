#pragma once

#include "api/neural.h"
#include "implementation_map.h"
#include "convolution_cpu_jit.h"
#include "convolution_cpu_reference.h"

namespace neural{
    //                                           engine                          output                  input
    using conv_fw_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;
    using conv_bw_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>; //todo

    static auto& conv_fw_implementation_map = singletion_map<conv_fw_key, std::function<is_an_implementation *(convolution &)>>::instance();
    static auto& conv_bw_implementation_map = singletion_map<conv_bw_key, std::function<is_an_implementation *(convolution_backward &)>>::instance();

    struct attach_conv_cpu_ref {
        attach_conv_cpu_ref();
    };
    struct attach_conv_cpu_jit {
        attach_conv_cpu_jit();
    };

    static void attach_conv(){
        static attach_conv_cpu_ref c0;
        static attach_conv_cpu_jit c1;
    }
}
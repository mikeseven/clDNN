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

#include "convolution_cpu_jit.h"
#include <functional>
#include <cstddef>

namespace{
#ifdef __linux__
#define nn_jit_param_reg rdi
#define nn_jit_dangerous_reg1 rcx
#define nn_jit_dangerous_reg2 rbx
#define nn_jit_dangerous_reg3 rsi
#define nn_jit_dangerous_reg4 rbp
#define nn_jit_dangerous_reg5 rsp
#else
#define nn_jit_param_reg rcx
#define nn_jit_dangerous_reg1 rdi
#define nn_jit_dangerous_reg2 rbx
#define nn_jit_dangerous_reg3 rsi
#define nn_jit_dangerous_reg4 rbp
#define nn_jit_dangerous_reg5 rsp
#endif

template<typename T> class reverse_t
{
    T& ref;
public:
    reverse_t(T& arg): ref(arg) {}
    auto begin() const -> decltype(ref.rbegin()) { return ref.rbegin (); }
    auto end()   const -> decltype(ref.rend())   { return ref.rend (); }
};
template<typename T> reverse_t<const T> reverse(const T& x) { return reverse_t<const T>(x); }
template<typename T> reverse_t<T> reverse(T& x) { return reverse_t<T>(x); }

struct jit_convolution_zxyn
{
#pragma pack(push, 1)
    struct op_data_t
    {
        float *output;
        uint32_t input_offset;
        float *filter;
        float *bias;
        float *input;
    };
#pragma pack(pop)

    std::vector<neural::task> tasks; //todo remove or replace
    static const uint32_t output_features_per_iteration = 16;
    static const uint32_t register_width_in_float       = 8;
    static const uint32_t register_width                = register_width_in_float * sizeof(float);

    class jit_code : public Xbyak::CodeGenerator
    {
        jit_code &operator=(const jit_code&) = delete;

        void preamble()
        {
            for(auto &reg : scalar_registers_to_preserve)
                push(reg);
            for(auto &reg : vector_registers_to_preserve) {
                auto offset = (&reg - &vector_registers_to_preserve[0]+1)*16;
                vmovdqu(ptr [rsp-offset], reg);
            }
        }

        void postamble()
        {
            for(auto &reg : reverse(vector_registers_to_preserve)) {
                auto offset = (&reg - &vector_registers_to_preserve[0]+1)*16;
                vmovdqu(reg, ptr [rsp-offset]);
            }
            for(auto &reg : reverse(scalar_registers_to_preserve))
                pop(reg);
            ret();
        }

        std::vector<Xbyak::Reg64> scalar_registers_to_preserve;
        std::vector<Xbyak::Xmm> vector_registers_to_preserve;

    public:

        jit_code(uint32_t output_width,
                 uint32_t output_height,
                 uint32_t output_feats,
                 uint32_t input_width,
                 uint32_t input_feats,
                 uint32_t filter_height,
                 uint32_t filter_width,
                 uint32_t stride_x,
                 uint32_t stride_y,
                 bool apply_relu,
                 void* code_ptr = nullptr,
                 size_t code_size = 40 * Xbyak::DEFAULT_MAX_CODE_SIZE)
            : Xbyak::CodeGenerator(code_size, code_ptr)
            , scalar_registers_to_preserve({nn_jit_dangerous_reg1, nn_jit_dangerous_reg2, r11, r12, r13, r14, r15})
            , vector_registers_to_preserve({xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15})
        {
            using Xbyak::Ymm;

            preamble();

            auto output                   = r10;
            auto input                    = r11;
            auto filter                   = r12;
            auto bias                     = r13;

            auto aux_input                = nn_jit_dangerous_reg1;
            auto aux_input_outer          = nn_jit_dangerous_reg2;
            auto aux_output               = r9;
            auto aux_filter               = r15;

            auto left_out_rows            = r8;
            auto left_out_blocks          = r14;
            auto left_kern_rows           = rdx;

            auto generate_for_single_output_block = [&](std::string tag,
                                                        int output_blocks,
                                                        bool vertical) {

                    const auto accumulators_count   = output_blocks * 2;

                    align(4);
                    mov(aux_input, aux_input_outer);
                    mov(aux_filter, filter);
                    for (int n = 0; n < accumulators_count; ++n)
                        vxorps(Ymm(n), Ymm(n), Ymm(n));

                    mov(left_kern_rows, filter_height);
                    L(tag + "_kern_row");
                    {
                        for (uint32_t kern_col = 0u; kern_col < filter_width; ++kern_col)
                        {
                            for (uint32_t i = 0u; i < input_feats; ++i)
                            {
                                auto filter_offset =
                                    (kern_col * input_feats * output_features_per_iteration
                                    + i * output_features_per_iteration)
                                        * sizeof(float);
                                vmovaps(ymm13, ptr [aux_filter + filter_offset]);
                                vmovaps(ymm14, ptr [aux_filter + filter_offset + register_width]);

                                for (int j = 0; j < output_blocks; ++j)
                                {
                                    auto block_offset = j * stride_x;
                                    if (vertical) block_offset = j * stride_y * input_width;
                                    auto input_offset =
                                        (block_offset * input_feats
                                        + kern_col * input_feats
                                        + i) * sizeof(float);
                                    vbroadcastss(ymm15, ptr [aux_input + input_offset]);
                                    vfmadd231ps(Ymm(2 * j + 0), ymm13, ymm15);
                                    vfmadd231ps(Ymm(2 * j + 1), ymm14, ymm15);
                                }
                            }
                        }
                        add(aux_input , static_cast<int>(input_width * input_feats * sizeof(float)));
                        add(aux_filter, static_cast<int>(filter_width * input_feats * output_features_per_iteration * sizeof(float)));
                    }
                    dec(left_kern_rows);
                    jnz(tag + "_kern_row");

                    for (int n = 0; n < accumulators_count; ++n)
                        vaddps(Ymm(n), ptr [bias + (n % 2) * register_width]);
                    if (apply_relu)
                    {
                        vxorps(ymm15, ymm15, ymm15);
                        for (int n = 0; n < accumulators_count; ++n)
                            vmaxps(Ymm(n), Ymm(n), ymm15);
                    }
                    for (int o = 0; o < output_blocks; ++o)
                    {
                        auto output_offset = o * output_feats * sizeof(float);
                        if (vertical) output_offset *= output_width;
                        vmovaps(ptr[aux_output + output_offset], Ymm(2 * o));
                        vmovaps(ptr[aux_output + output_offset + register_width], Ymm(2 * o + 1));
                    }
                };

            auto prepare_input_ptr = [&]{
                mov(input,  ptr [nn_jit_param_reg + offsetof(op_data_t, input)]);
                add(input,  ptr [nn_jit_param_reg + offsetof(op_data_t, input_offset)]);
            };

            mov(output, ptr [nn_jit_param_reg + offsetof(op_data_t, output)]);
            mov(filter, ptr [nn_jit_param_reg + offsetof(op_data_t, filter)]);
            mov(bias,   ptr [nn_jit_param_reg + offsetof(op_data_t, bias)]);
            prepare_input_ptr();

            mov(left_out_rows, output_height);
            L("output_row");
            {
                mov(aux_input_outer, input);
                mov(aux_output, output);

                mov(left_out_blocks, output_width / 6);
                L("output_block");
                {
                    generate_for_single_output_block("horizontal_full", 6, false);

                    add(aux_output, 6 * output_feats * sizeof(float));
                    add(aux_input_outer, 6 * input_feats * stride_x * sizeof(float));
                }
                dec(left_out_blocks);
                jnz("output_block");

                add(input, input_feats * input_width * stride_y * sizeof(float));
                add(output, output_feats * output_width * sizeof(float));

            }
            dec(left_out_rows);
            jnz("output_row");

            mov(output, ptr [nn_jit_param_reg + offsetof(op_data_t, output)]);
            prepare_input_ptr();
            for (auto i = 0u; i < output_height / 6; ++i)
            {
                mov(aux_input_outer, input);
                mov(aux_output, output);
                add(aux_output, (i * output_width + output_width / 6) * 6 * output_feats * sizeof(float));
                add(aux_input_outer, (i * input_width * stride_y + output_width / 6 * stride_x) * 6 * input_feats * sizeof(float));
                for (uint32_t j = 0u; j < output_height % 6; ++j)
                {
                    generate_for_single_output_block("vertical_full_" + std::to_string(i) + "_" + std::to_string(j), 6, true);
                    add(aux_output, output_feats * sizeof(float));
                }
            }
            for (uint32_t i = output_height / 6 * 6; i < output_height; ++i)
            {
                mov(aux_input_outer, input);
                mov(aux_output, output);
                add(aux_output, (i * output_width + output_width / 6 * 6) * output_feats * sizeof(float));
                add(aux_input_outer, (i * input_width * stride_y + output_width / 6 * stride_x * 6) * input_feats * sizeof(float));
                generate_for_single_output_block("horizontal_partial_" + std::to_string(i), output_width % 6, false);
            }

            postamble();
        }
    };

    std::vector<op_data_t> op_data;
    jit_code code;

    jit_convolution_zxyn(
        uint32_t batch,
        bool apply_relu,

        float*   output,
        uint32_t output_width,
        uint32_t output_height,
        uint32_t output_feature_maps,

        float*   input,
        uint32_t input_width,
        uint32_t input_height,
        uint32_t input_feature_maps,

        uint32_t stride_width,
        uint32_t stride_height,

        float *  filter,
        uint32_t filter_width,
        uint32_t filter_height,

        float*   bias,

        void* code_ptr = nullptr)
            : code(output_width, output_height, output_feature_maps,
                   input_width, input_feature_maps,
                   filter_height, filter_width,
                   stride_width, stride_height,
                   apply_relu,
                   code_ptr)
    {
        assert(output_feature_maps % output_features_per_iteration == 0);

        auto z_out_blocks = output_feature_maps / output_features_per_iteration;
        const auto batch_block_size = 24;
        for (auto bblock = 0u; bblock < batch / batch_block_size; ++bblock)
        {
            for (auto z_block = 0u; z_block < z_out_blocks; ++z_block)
            {
                for (auto b = bblock * batch_block_size; b < (bblock + 1) * batch_block_size; ++b)
                {
                    auto output_shifted = output
                        + b * output_width * output_height * output_feature_maps
                        + z_block * output_features_per_iteration;
                    auto bias_shifted = bias + z_block * output_features_per_iteration;
                    auto filter_shifted = filter
                        + z_block * output_features_per_iteration * input_feature_maps * filter_width * filter_height;
                    auto input_shifted = //input +
                        b * input_width * input_height * input_feature_maps;

                    op_data.push_back({ output_shifted,
                        input_shifted * sizeof(float),
                        filter_shifted,
                        bias_shifted,
                        input });
                }
            }
        }
        for (auto z_block = 0u; z_block < z_out_blocks; ++z_block)
        {
            for (auto b = batch / batch_block_size * batch_block_size; b < batch; ++b)
            {
                auto output_shifted = output
                    + b * output_width * output_height * output_feature_maps
                    + z_block * output_features_per_iteration;
                auto bias_shifted = bias + z_block * output_features_per_iteration;
                auto filter_shifted = filter
                    + z_block * output_features_per_iteration * input_feature_maps * filter_width * filter_height;
                auto input_shifted = b * input_width * input_height * input_feature_maps;

                op_data.push_back({ output_shifted,
                    input_shifted * sizeof(float),
                    filter_shifted,
                    bias_shifted,
                    input });
            }
        }

        tasks.resize(op_data.size());
        for (auto i = 0u; i < tasks.size(); ++i)
            tasks[i] = {reinterpret_cast<void(*)(const void*)>(code.getCode()), &op_data[i]};
    }
};
}

namespace neural {

convolution_cpu_jit::convolution_cpu_jit(convolution &arg)
    : is_an_implementation(neural::type_id<convolution_cpu_jit>())
    , outer(arg) {};
convolution_cpu_jit:: ~convolution_cpu_jit() {}
/*static*/ void convolution_cpu_jit::implementation(const void *ptr) {
    auto this_conv = static_cast<const convolution *>(ptr);

    auto& input_offset  = this_conv->argument.input_offset;
    auto& output_offset = this_conv->argument.output_offset;
    auto& output_size   = this_conv->argument.output_size;
    auto& padding       = this_conv->argument.padding;
    auto& stride        = this_conv->argument.stride;

    auto& input_arg  = this_conv->input_memory(0).argument;
    auto& output_arg = this_conv->output_memory(0).argument;

    auto& filter_arg = this_conv->argument.weight.as<const memory&>().argument; //convolution filter
    auto& bias_arg   = this_conv->argument.bias.as<const memory&>().argument;

    if(input_arg.size.size()  != output_arg.size.size())   throw std::runtime_error("Convolution input/output number of dimension does not match.");
    if(stride.size()          != output_arg.size.size())   throw std::runtime_error("Convolution stride/output number of dimension does not match.");
    if(input_arg.format       != memory::format::yxfb_f32) throw std::runtime_error("Convolution reference uses yxfb_f32 format.");             // only yxfb_f32 format is supported
    if(input_arg.format       != output_arg.format)        throw std::runtime_error("Convolution input/output data format does not match.");    // only yxfb_f32 format is supported
    if(input_arg.format       != filter_arg.format)        throw std::runtime_error("Convolution input/weights data format does not match.");   // only yxfb_f32 format is supported
    if(filter_arg.size.size() != output_arg.size.size())   throw std::runtime_error("Convolution window_size/output number of dimension does not match.");
    if(bias_arg.size.size()   != 1)                        throw std::runtime_error("Convolution biases isn't 1D vector.");
    if(bias_arg.size[0]       != output_size[2])           throw std::runtime_error("Convolution biases/output feature maps number does not match."); // todo need type traits for index of 'z' dimension
                                                                                                                                                            // than this implementation will be format independent
    auto input  = static_cast<float*>(this_conv->input_memory(0).pointer);
    auto output = static_cast<float*>(this_conv->output_memory(0).pointer);
    auto filter = static_cast<float*>(this_conv->argument.weight.as<const memory&>().pointer);
    auto bias   = static_cast<float*>(this_conv->argument.bias.as<const memory&>().pointer);

    // general formula: output size = (input size - filter size) / step + 1
    for(size_t i = 0; i < input_offset.size(); ++i){
        if(output_arg.size[i] < output_size[i] + output_offset[i])
            throw std::runtime_error("Convolution output buffer size is to small.");
    }

    int b_pos = 3; // todo typetraits
    int f_pos = 2;
    int x_pos = 1;
    int y_pos = 0;

    switch(padding){
        case padding::zero:
        {
            // todo jit conv works in xyzb format?
            // todo conv jit
            // todo how to handle offsets?
            jit_convolution_zxyn conv(
                output_size[ b_pos ], // batch??
                false, //activaction function: relu
                output,
                output_size[ x_pos ],
                output_size[ y_pos ],
                output_size[ f_pos ],
                input,
                input_arg.size[ x_pos ],
                input_arg.size[ y_pos ],
                input_arg.size[ f_pos ],
                stride[ x_pos ],
                stride[ y_pos ],
                filter,
                filter_arg.size[ x_pos ],
                filter_arg.size[ y_pos ],
                bias
                );

            for(auto& task : conv.tasks){
                task.callback(task.data);
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown padding mode in convolution.");
    }
}

namespace{

struct attach{
    attach(){
        auto key = std::make_tuple(engine::cpu, memory::format::yxfb_f32, memory::format::yxfb_f32);
        auto val_fw = convolution_cpu_jit::create;

        // todo not working
        conv_fw_implementation_map.insert( {key, val_fw} ); //todo keys should be different
    }
    ~attach(){}
};

#ifdef __GNUC__
    __attribute__((visibility("default"))) //todo meybe dll_sym?
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

}
}
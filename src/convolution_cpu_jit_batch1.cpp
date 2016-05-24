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
#define XBYAK_NO_OP_NAMES
#define XBYAK_USE_MMAP_ALLOCATOR

#include "convolution_cpu_jit_batch1.h"
#include "multidimensional_counter.h"
#include "memory_utils.h"
#include "xbyak/xbyak_util.h"

#include <thread>
#include <mutex>

const int C_simd_width = sizeof(__m256)/sizeof(float);
const int C_slice_size = C_simd_width*2;

#define UINT64GUARD(func, constarg, vararg)            \
    if((vararg) > UINT32_MAX) {                         \
        mov(rax, vararg);                               \
        func(constarg, rax);                            \
    } else {                                            \
        func(constarg, static_cast<uint32_t>(vararg));  \
    }

namespace neural {

    class convolution_generator : public ::Xbyak::CodeGenerator
    {
    public:
        convolution_generator(    
            uint64_t input_width,
            uint64_t input_height,
            uint64_t input_depth,
            uint64_t inner_loop_iterations,
            uint64_t kernel_width,
            uint64_t kernel_height,
            uint64_t kernel_ifmap,
            uint64_t output_width,
            uint64_t output_height,
            uint64_t output_depth,
            uint64_t block_size,
            uint64_t out_fm_pckg_size,
            uint64_t num_blocks_full,
            uint64_t partial_block_size,
            uint64_t stride_x,
            uint64_t stride_y);

        ~convolution_generator() {}

        convolution_generator_callback_t* generator_callback = nullptr;
    };

    convolution_generator::convolution_generator(
        uint64_t input_width,
        uint64_t input_height,
        uint64_t input_depth,
        uint64_t inner_loop_iterations,
        uint64_t kernel_width,
        uint64_t kernel_height,
        uint64_t kernel_ifmap,
        uint64_t output_width,
        uint64_t output_height,
        uint64_t output_depth,
        uint64_t block_size,
        uint64_t out_fm_pckg_size,
        uint64_t num_blocks_full,
        uint64_t partial_block_size,
        uint64_t stride_x,
        uint64_t stride_y)
    {
        using namespace ::Xbyak;
        util::Cpu Current_cpu;

        if(Current_cpu.has(util::Cpu::tAVX2))
        {
            // Precomputed constants.
            // size_t kernel_depth_size = input_fmap_view_length * C_slice_size;
            uint64_t input_row_size = input_width * input_depth;
            uint64_t output_row_size = output_width * output_depth;

            uint64_t input_image_size = input_row_size*input_height;
            uint64_t output_image_size = output_row_size*output_height;

            uint64_t weight_offset = kernel_ifmap * kernel_width * kernel_height * C_slice_size;

            uint64_t innermost_unroll_factor = 1;
            if(inner_loop_iterations % 4 == 0)
            {
                innermost_unroll_factor = 4;
            }
            else if(inner_loop_iterations % 3 == 0)
            {
                innermost_unroll_factor = 3;
            }
            else if(inner_loop_iterations % 2 == 0)
            {
                innermost_unroll_factor = 2;
            }

            // Helpers for assembly variables.
            // After stackframe setup we'll have following stack layout:

#ifdef _WIN32
            // [RBP + 48]: stack param 1, etc..
            // [RBP + 48]: stack param 0
            // [RBP + 40]: shadowspace entry 3
            // [RBP + 32]: shadowspace entry 2
            // [RBP + 24]: shadowspace entry 1
            // [RBP + 16]: shadowspace entry 0
            // [RBP +  8]: return address
            // [RBP     ]: caller RBP
            // [RBP -  8]: local variable 0
            // [RBP - 16]: local variable 1, etc..
            uint64_t stack_arg_qwords = 5;
            const Reg64&   regarg_input_ptr                      = rcx;
            const Reg64&   regarg_weights_ptr                    = rdx;
            const Reg64&   regarg_bias_ptr                       = r8;
            const Reg64&   regarg_output_ptr                     = r9;
            const Address& stackarg_input_fmap_view_start        = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_input_fmap_view_end          = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_input_column_view_start      = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_input_row_view_start         = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_kernel_input_fmap_view_start = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_kernel_out_fmap_view_start   = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_column_view_start     = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_row_view_start        = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_row_view_end          = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_fm_view_start         = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_fm_view_end           = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_image_view_start      = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_image_view_end        = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
#else
            // [RBP + 24]: stack param 1, etc..
            // [RBP + 16]: stack param 0
            // [RBP +  8]: return address
            // [RBP     ]: caller RBP
            // [RBP -  8]: local variable 0
            // [RBP - 16]: local variable 1, etc..
            // We are ignoring red zone here, our code is exception-proof :P
            uint64_t stack_arg_qwords = 1;
            const Reg64&   regarg_input_ptr                      = rdi;
            const Reg64&   regarg_weights_ptr                    = rsi;
            const Reg64&   regarg_bias_ptr                       = rdx;
            const Reg64&   regarg_output_ptr                     = rcx;
            const Reg64&   regarg_input_fmap_view_start          = r8;
            const Reg64&   regarg_input_fmap_view_end            = r9;
            const Address& stackarg_input_column_view_start      = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_input_row_view_start         = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_kernel_input_fmap_view_start = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_kernel_out_fmap_view_start   = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_column_view_start     = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_row_view_start        = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_row_view_end          = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_fm_view_start         = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_fm_view_end           = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_image_view_start      = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
            const Address& stackarg_output_image_view_end        = qword[rbp + (++stack_arg_qwords * sizeof(size_t))];
#endif

            uint64_t stack_local_qwords = 0;
            const Address& stack_input_ptr             = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_weights_ptr           = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_bias_ptr              = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_output_ptr            = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_input_fmap_view_start = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_input_fmap_view_end   = qword[rbp - (++stack_local_qwords * sizeof(size_t))];

            const Address& stack_current_image       = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_current_output_fm   = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_current_output_row  = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_current_input_row   = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_current_kernel_fm   = qword[rbp - (++stack_local_qwords * sizeof(size_t))];

            const Address& stack_current_block       = qword[rbp - (++stack_local_qwords * sizeof(size_t))];

            const Address& stack_current_kernel_row  = qword[rbp - (++stack_local_qwords * sizeof(size_t))];

            const Address& stack_input_image_offset  = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_output_image_offset = qword[rbp - (++stack_local_qwords * sizeof(size_t))];

            const Address& stack_input_offset_base   = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_output_offset_base  = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_kernel_offset_base  = qword[rbp - (++stack_local_qwords * sizeof(size_t))];

            const Address& stack_input_ptr_base      = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_output_ptr_base     = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_kernel_ptr_base     = qword[rbp - (++stack_local_qwords * sizeof(size_t))];
            const Address& stack_bias_ptr_base       = qword[rbp - (++stack_local_qwords * sizeof(size_t))];

            // ASSEMBLY STARTS HERE.

            // Prologue.
            push(rbp);
            mov(rbp, rsp);
            UINT64GUARD(sub, rsp, stack_local_qwords * sizeof(size_t));
            push(rbx);
            push(r12); 
            push(r13); 
            push(r14); 
            push(r15);

            mov(stack_input_ptr, regarg_input_ptr);
            mov(stack_weights_ptr, regarg_weights_ptr);
            mov(stack_bias_ptr, regarg_bias_ptr);
            mov(stack_output_ptr, regarg_output_ptr);

#ifdef _WIN32
            mov(rax, stackarg_input_fmap_view_start);
            mov(stack_input_fmap_view_start, rax);

            mov(rax, stackarg_input_fmap_view_end);
            mov(stack_input_fmap_view_end, rax);
#else
            mov(stack_input_fmap_view_start, regarg_input_fmap_view_start);
            mov(stack_input_fmap_view_end, regarg_input_fmap_view_end);
#endif

            // Main code.
            mov(r15, inner_loop_iterations);

            mov(rax, stackarg_output_image_view_start);
            mov(stack_current_image, rax);
            L("batch_loop_start");
            mov(rax, stackarg_output_image_view_end);
            cmp(stack_current_image, rax);
            ja("batch_loop_end", T_NEAR);

            mov(rax, input_image_size);
            mul(stack_current_image);
            mov(stack_input_image_offset, rax);

            mov(rax, output_image_size);
            mul(stack_current_image);
            mov(stack_output_image_offset, rax);

            mov(rax, stackarg_output_fm_view_start);
            mov(stack_current_output_fm, rax);
            mov(rax, stackarg_kernel_out_fmap_view_start);
            mov(stack_current_kernel_fm, rax);
            L("outfm_loop_start");
            mov(rax, stackarg_output_fm_view_end);
            cmp(stack_current_output_fm, rax);
            ja("outfm_loop_end", T_NEAR);

            mov(rax, stackarg_output_row_view_start);
            mov(stack_current_output_row, rax);
            mov(stack_current_input_row, 0);
            L("output_row_start");
            mov(rax, stackarg_output_row_view_end);
            cmp(stack_current_output_row, rax);
            ja("output_row_end", T_NEAR);

            // inp_h = input_row * kernel_stride_y + input_row_view_start
            mov(rax, stride_y);             
            mul(stack_current_input_row);
            add(rax, stackarg_input_row_view_start);

            // inp_offset1 = inp_h * input_row_size
            mov(rdx, rax);
            mov(rax, input_row_size);
            mul(rdx);

            // save inp_offset1
            mov(rcx, rax);

            // inp_offset_base = inp_offset1 + input_column_view_start * num_input_feature_maps + input_image_offset + input_fmap_view_start
            mov(rax, input_depth);
            mul(stackarg_input_column_view_start);
            add(rax, stack_input_image_offset);
            add(rax, rcx);
            mov(rcx, stack_input_fmap_view_start);
            add(rax, rcx);
            mov(stack_input_offset_base, rax);

            // out_offset1  = output_row * output_row_size
            mov(rax, output_row_size);
            mul(stack_current_output_row);

            // save out_offset1
            mov(rcx, rax);

            // out_offset = out_offset1 + output_column_view_start * num_output_feature_maps + output_image_offset + current_output_fm
            mov(rax, output_depth);
            mul(stackarg_output_column_view_start);
            add(rax, stack_output_image_offset);
            add(rax, rcx);
            add(rax, stack_current_output_fm);
            mov(stack_output_offset_base, rax);

            // kernel_offset = (kernel_feature_map / C_slice_size) * weight_offset + kernel_input_fmap_view_start * C_slice_size
            mov(rdx, stack_current_kernel_fm);
            shr(rdx, static_cast<uint32_t>(log2(C_slice_size)));
            mov(rax, weight_offset);
            mul(rdx);
            mov(rdx, stackarg_kernel_input_fmap_view_start);
            shl(rdx, static_cast<uint32_t>(log2(C_slice_size)));
            add(rax, rdx);
            mov(stack_kernel_offset_base, rax);

            // Main column block loop.
            auto inner_macro = [&](size_t block_size)
            {
                auto make_unique = [&](std::string string) {return string + std::to_string(block_size);};

                mov(rax, stack_kernel_offset_base);
                shl(rax, 2);
                add(rax, stack_weights_ptr);
                mov(stack_kernel_ptr_base, rax);

                mov(rax, stack_input_offset_base);
                shl(rax, 2);
                add(rax, stack_input_ptr);
                mov(stack_input_ptr_base, rax);

                mov(rax, stack_output_offset_base);
                shl(rax, 2);
                add(rax, stack_output_ptr);
                mov(stack_output_ptr_base, rax);

                mov(rax, stack_current_kernel_fm);
                shl(rax, 2);
                add(rax, stack_bias_ptr);
                mov(stack_bias_ptr_base, rax);

                for(uint32_t acc = 0; acc < block_size; ++acc)
                {
                    if(out_fm_pckg_size == 16)
                    {
                        vxorps(Ymm(acc * 2 + 0), Ymm(acc * 2 + 0));
                        vxorps(Ymm(acc * 2 + 1), Ymm(acc * 2 + 1));
                    }
                    else if(out_fm_pckg_size == 8)
                    {
                        vxorps(Ymm(acc), Ymm(acc));
                    }
                    else if(out_fm_pckg_size == 3)
                    {
                        vxorps(Ymm(acc * 3 + 0), Ymm(acc * 3 + 0));
                        vxorps(Ymm(acc * 3 + 1), Ymm(acc * 3 + 1));
                        vxorps(Ymm(acc * 3 + 2), Ymm(acc * 3 + 2));
                    }
                }

                // convolution inner loops
                mov(stack_current_kernel_row, 0);
                L(make_unique("kernel_height_start"));
                mov(rax, kernel_height);
                cmp(stack_current_kernel_row, rax);
                jae(make_unique("kernel_height_end"), T_NEAR);

                mov(r13, stack_input_ptr_base);
                mov(r14, stack_kernel_ptr_base);

                mov(r12, 0);
                mov(r8, kernel_width);
                align(16);        L(make_unique("kernel_width_start"));
                cmp(r12, r8);
                jae(make_unique("kernel_width_end"), T_NEAR);

                mov(r10, r13);
                mov(r11, r14);

                if(inner_loop_iterations/innermost_unroll_factor > 1)
                {
                    mov(r9, 0);
                    align(16);            L(make_unique("kernel_idepth_start"));
                    cmp(r9, r15);
                    jae(make_unique("kernel_idepth_end"), T_NEAR);
                }
                for(uint32_t inner_unroll = 0; inner_unroll < innermost_unroll_factor; ++inner_unroll)
                {
                    if(out_fm_pckg_size == 16)
                    {
                        vmovups(Ymm(14), byte[r11 + inner_unroll * C_slice_size * sizeof(float)]);
                        vmovups(Ymm(15), byte[r11 + inner_unroll * C_slice_size * sizeof(float) + C_simd_width*sizeof(float)]);

                        for(uint32_t acc = 0; acc < block_size; ++acc)
                        {
                            vbroadcastss(Ymm(13), byte[r10 + inner_unroll * sizeof(float) + acc * input_depth * stride_x * sizeof(float)]);
                            vfmadd231ps(Ymm(acc * 2 + 0), Ymm(13), Ymm(14));
                            vfmadd231ps(Ymm(acc * 2 + 1), Ymm(13), Ymm(15));
                        }
                    }
                    else if(out_fm_pckg_size == 8)
                    {
                        vmovups(Ymm(15), byte[r11 + inner_unroll * C_slice_size * sizeof(float)]);

                        for(uint32_t acc = 0; acc < block_size; ++acc)
                        {
                            vbroadcastss(Ymm(14), byte[r10 + inner_unroll * sizeof(float) + acc * input_depth * stride_x * sizeof(float)]);
                            vfmadd231ps(Ymm(acc), Ymm(14), Ymm(15));
                        }
                    }
                    else if(out_fm_pckg_size == 3)
                    {
                        vmovss(Xmm(13), byte[r11 + inner_unroll * C_slice_size * sizeof(float)]);
                        vmovss(Xmm(14), byte[r11 + inner_unroll * C_slice_size * sizeof(float) + 1 * sizeof(float)]);
                        vmovss(Xmm(15), byte[r11 + inner_unroll * C_slice_size * sizeof(float) + 2 * sizeof(float)]);

                        for(uint32_t acc = 0; acc < block_size; ++acc)
                        {
                            vbroadcastss(Ymm(12), byte[r10 + inner_unroll * sizeof(float) + acc * input_depth * stride_x * sizeof(float)]);
                            vfmadd231ps(Ymm(acc * 3 + 0), Ymm(12), Ymm(13));
                            vfmadd231ps(Ymm(acc * 3 + 1), Ymm(12), Ymm(14));
                            vfmadd231ps(Ymm(acc * 3 + 2), Ymm(12), Ymm(15));
                        }
                    }
                }

                if(inner_loop_iterations/innermost_unroll_factor > 1)
                {
                    UINT64GUARD(add, r10, innermost_unroll_factor * sizeof(float));
                    UINT64GUARD(add, r11, innermost_unroll_factor * C_slice_size * sizeof(float));
                    UINT64GUARD(add, r9, innermost_unroll_factor);

                    jmp(make_unique("kernel_idepth_start"), T_NEAR);
                    L(make_unique("kernel_idepth_end"));
                }

                UINT64GUARD(add, r13, input_depth * sizeof(float));
                UINT64GUARD(add, r14, kernel_ifmap * C_slice_size * sizeof(float));

                inc(r12);
                jmp(make_unique("kernel_width_start"), T_NEAR);
                L(make_unique("kernel_width_end"));

                UINT64GUARD(add, stack_input_ptr_base, input_row_size * sizeof(float));
                UINT64GUARD(add, stack_kernel_ptr_base, kernel_ifmap * C_slice_size * kernel_width * sizeof(float));

                inc(stack_current_kernel_row);
                jmp(make_unique("kernel_height_start"), T_NEAR);
                L(make_unique("kernel_height_end"));

                mov(rax, stack_output_ptr_base);
                mov(rcx, stack_bias_ptr_base);

                if(out_fm_pckg_size == 16)
                {
                    vxorps(Ymm(13), Ymm(13));
                    vmovups(Ymm(14), byte[rcx                               ]);
                    vmovups(Ymm(15), byte[rcx + C_simd_width * sizeof(float)]);
                    for(uint32_t acc = 0; acc < block_size; ++acc)
                    {
                        vaddps(Ymm(acc * 2 + 0), Ymm(14));
                        vaddps(Ymm(acc * 2 + 1), Ymm(15));

                        /*if(activation == NN_ACTIVATION_FUNCTION_RELU)
                        {
                            vmaxps(Ymm(acc * 2 + 0), Ymm(13));
                            vmaxps(Ymm(acc * 2 + 1), Ymm(13));
                        }*/

                        vmovups(byte[rax + (acc * output_depth               ) * sizeof(float)], Ymm(acc * 2 + 0));
                        vmovups(byte[rax + (acc * output_depth + C_simd_width) * sizeof(float)], Ymm(acc * 2 + 1));
                    }
                }
                else if(out_fm_pckg_size == 8)
                {
                    vxorps(Ymm(14), Ymm(14));
                    vmovups(Ymm(15), byte[rcx]);
                    for(uint32_t acc = 0; acc < block_size; ++acc)
                    {
                        vaddps(Ymm(acc), Ymm(15));

                        /*if(activation == NN_ACTIVATION_FUNCTION_RELU)
                        {
                            vmaxps(Ymm(acc), Ymm(14));
                        }*/

                        vmovups(byte[rax + acc * output_depth * sizeof(float)], Ymm(acc));
                    }
                }
                else if(out_fm_pckg_size == 3)
                {
                    vxorps(Ymm(12), Ymm(12));
                    vmovss(Xmm(13), byte[rcx + 0 * sizeof(float)]);
                    vmovss(Xmm(14), byte[rcx + 1 * sizeof(float)]);
                    vmovss(Xmm(15), byte[rcx + 2 * sizeof(float)]);
                    for(uint32_t acc = 0; acc < block_size; ++acc)
                    {
                        vaddss(Xmm(acc * 3 + 0), Xmm(13));
                        vaddss(Xmm(acc * 3 + 1), Xmm(14));
                        vaddss(Xmm(acc * 3 + 2), Xmm(15));

                        /*if(activation == NN_ACTIVATION_FUNCTION_RELU)
                        {
                            vmaxss(Xmm(acc * 3 + 0), Xmm(12));
                            vmaxss(Xmm(acc * 3 + 1), Xmm(12));
                            vmaxss(Xmm(acc * 3 + 2), Xmm(12));
                        }*/

                        vmovss(byte[rax + (acc * output_depth + 0) * sizeof(float)], Xmm(acc * 3 + 0));
                        vmovss(byte[rax + (acc * output_depth + 1) * sizeof(float)], Xmm(acc * 3 + 1));
                        vmovss(byte[rax + (acc * output_depth + 2) * sizeof(float)], Xmm(acc * 3 + 2));
                    }
                }
                else
                    throw std::runtime_error("unknown package size");
            };

            if(num_blocks_full != 0) {
                mov(stack_current_block, 0);
                L("block_start");
                UINT64GUARD(cmp, stack_current_block, num_blocks_full);
                jae("block_end", T_NEAR);

                inner_macro(block_size);

                UINT64GUARD(add, stack_input_offset_base, block_size * input_depth * stride_x);
                UINT64GUARD(add, stack_output_offset_base, block_size * output_depth);

                inc(stack_current_block);
                jmp("block_start", T_NEAR);
                L("block_end");
            }

            if(partial_block_size != 0) {
                inner_macro(partial_block_size);
            }

            inc(stack_current_output_row);
            inc(stack_current_input_row);
            jmp("output_row_start", T_NEAR);
            L("output_row_end");

            add(stack_current_output_fm, C_slice_size);
            add(stack_current_kernel_fm, C_slice_size);
            jmp("outfm_loop_start", T_NEAR);
            L("outfm_loop_end");

            inc(stack_current_image);
            jmp("batch_loop_start", T_NEAR);
            L("batch_loop_end");

            // Epilogue.
            pop(r15);
            pop(r14);
            pop(r13);
            pop(r12);
            pop(rbx);
            mov(rsp, rbp);
            pop(rbp);
            ret();

            // ASSEMBLY ENDS HERE.

            generator_callback = getCode<convolution_generator_callback_t*>();
        }
        else
            throw std::runtime_error("AVX2 not supported by this machine.");
    }

    template<class gen_type, class ... types>
    gen_type* get_generator(types ... args)
    {
        static std::map<std::tuple<types...>, gen_type*> generator_map;
        static std::mutex map_mutex;

        auto arg_tuple = std::tuple<types...>(args...);

        auto generator = generator_map.find(arg_tuple);

        if(generator != generator_map.end())
        {
            return generator->second;
        }
        else
        {
            auto new_generator = new gen_type(args...);
            generator_map.insert(std::pair<std::tuple<types...>, gen_type*>(arg_tuple, new_generator));
            return new_generator;
        }
    }


    void unpack_convolve_callback_handle(
        void* void_handle)
    {
        parameters_convolution_f32_precompiled_jit& handle = *reinterpret_cast<parameters_convolution_f32_precompiled_jit*>(void_handle);

        auto output_fm_view_length = handle.output_fm_view_end - handle.output_fm_view_start + 1;

        if(   output_fm_view_length % C_slice_size != 0 
           && output_fm_view_length != C_simd_width 
           && output_fm_view_length != 3) 
            throw std::runtime_error("unknown ofm view mode - convolution");

        handle.callback(
            *handle.input_ptr, 
            *handle.weights_ptr, 
            *handle.bias_ptr, 
            *handle.output_ptr,
            handle.input_fmap_view_start,
            handle.input_fmap_view_end,
            handle.input_column_view_start,
            handle.input_row_view_start,
            handle.kernel_input_fmap_view_start,
            handle.kernel_out_fmap_view_start,
            handle.output_column_view_start,
            handle.output_row_view_start,
            handle.output_row_view_end,
            handle.output_fm_view_start,
            handle.output_fm_view_end,
            handle.output_image_view_start,
            handle.output_image_view_end);
    }


convolution_cpu_jit_batch1::convolution_cpu_jit_batch1(convolution &arg)
        : is_an_implementation(neural::type_id<convolution_cpu_jit_batch1>())
{
    // Get pointers for pointers to buffers!
    auto input_ptr = reinterpret_cast<float**>(&arg.input_memory(0).pointer);
    auto output_ptr = reinterpret_cast<float**>(&arg.output_memory(0).pointer);
    auto weights_ptr = reinterpret_cast<float**>(&arg.argument.input[1].primitive.as<const memory&>().pointer);
    auto bias_ptr = reinterpret_cast<float**>(&arg.argument.input[2].primitive.as<const memory&>().pointer);

    // Get input buffer sizes.
    uint64_t input_width = arg.input_memory(0).argument.size.spatial[0];
    uint64_t input_height = arg.input_memory(0).argument.size.spatial[1];
    uint64_t input_depth = arg.input_memory(0).argument.size.feature[0];

    // Get input offsets.
    uint64_t inpfmap_begin = arg.argument.input_offset.feature[0];
    uint64_t inpfmap_end = input_depth - 1;
    uint64_t inpfmap_length = inpfmap_end - inpfmap_begin + 1;

    // Get output offsets.
    uint64_t outbatch_begin = arg.argument.output_offset.batch[0];
    uint64_t outbatch_end = outbatch_begin + arg.argument.output_size.batch[0] - 1;

    // Get output buffer sizes.
    uint64_t output_width = arg.output_memory(0).argument.size.spatial[0];
    uint64_t output_height = arg.output_memory(0).argument.size.spatial[1];
    uint64_t output_depth = arg.output_memory(0).argument.size.feature[0];

    // Get 'views' sizes for output.
    uint64_t outpcol_length = arg.argument.output_size.spatial[0];
    uint64_t outprow_length = arg.argument.output_size.spatial[1];
    uint64_t outpfmap_length = arg.argument.output_size.feature[0];

    uint64_t kernel_width = arg.argument.input[1].primitive.as<const memory &>().argument.size.spatial[0];
    uint64_t kernel_height = arg.argument.input[1].primitive.as<const memory &>().argument.size.spatial[1];
    uint64_t kernel_ifmap = arg.argument.input[1].primitive.as<const memory &>().argument.size.feature[1];
    uint64_t stride_col = arg.argument.stride.raw[2];
    uint64_t stride_row = arg.argument.stride.raw[3];

    // Make these variable basing on number of threads - this naive case won't be optimal.
    uint64_t output_row_package_size = outprow_length;
    uint64_t output_fmap_package_size = 16;
    uint64_t output_col_package_size = outpcol_length;

    uint64_t num_output_fm_items = outpfmap_length / output_fmap_package_size;
    uint64_t num_output_fm_items_remainder = outpfmap_length % output_fmap_package_size;
    uint64_t num_col_items = outpcol_length / output_col_package_size;
    uint64_t num_row_items = outprow_length / output_row_package_size;

    precompiled_request_handles.resize(num_output_fm_items * num_col_items * num_row_items);

    // Fill slave work items.
    for (uint64_t output_fm_item = 0u; output_fm_item < num_output_fm_items; ++output_fm_item)
    {
        for (uint64_t row_item = 0u; row_item < num_row_items; ++row_item)
        {
            for (uint64_t col_item = 0u; col_item < num_col_items; ++col_item)
            { 
                uint64_t item_in_pool = col_item * num_output_fm_items * num_row_items + row_item * num_output_fm_items + output_fm_item;

                uint64_t input_view_begin_col = arg.argument.input_offset.spatial[0] + col_item * output_col_package_size * stride_col;
                uint64_t input_view_begin_row = arg.argument.input_offset.spatial[1] + row_item * output_row_package_size * stride_row;

                uint64_t output_view_begin_col = arg.argument.output_offset.spatial[0] + col_item * output_col_package_size;
                uint64_t output_view_begin_row = arg.argument.output_offset.spatial[1] + row_item * output_row_package_size;
                uint64_t output_view_begin_map = arg.argument.output_offset.feature[0] + output_fm_item * output_fmap_package_size;

                uint64_t output_view_end_col = arg.argument.output_offset.spatial[0] + (col_item+1) * output_col_package_size - 1;
                uint64_t output_view_end_row = arg.argument.output_offset.spatial[1] + (row_item+1) * output_row_package_size - 1;
                uint64_t output_view_end_map = arg.argument.output_offset.feature[0] + (output_fm_item+1) * output_fmap_package_size - 1;

                if(output_fm_item+1 == num_output_fm_items && num_output_fm_items_remainder)
                {
                    // Case where we need to process only remaining FMaps.
                    output_view_end_map = output_view_begin_map + num_output_fm_items_remainder - 1;
                }

                uint64_t output_fmaps_view_length = output_view_end_map - output_view_begin_map + 1;
                uint64_t block_size = (output_fmaps_view_length == 3) 
                    ? 4 
                    : ((output_fmaps_view_length == 8)
                       ? 14
                       : 6);

                uint64_t out_fm_pckg_size = (output_fmaps_view_length == 3) 
                    ? 3 
                    : ((output_fmaps_view_length == 8)
                       ? 8
                       : 16);

                uint64_t output_column_view_length = output_view_end_col - output_view_begin_col + 1;
                uint64_t num_blocks_full      = output_column_view_length / block_size;
                uint64_t partial_block_size   = output_column_view_length % block_size;

                precompiled_request_handles[item_in_pool].input_ptr                    = input_ptr;                 
                precompiled_request_handles[item_in_pool].weights_ptr                  = weights_ptr;
                precompiled_request_handles[item_in_pool].bias_ptr                     = bias_ptr;
                precompiled_request_handles[item_in_pool].output_ptr                   = output_ptr;
                precompiled_request_handles[item_in_pool].input_fmap_view_start        = inpfmap_begin;
                precompiled_request_handles[item_in_pool].input_fmap_view_end          = inpfmap_end;
                precompiled_request_handles[item_in_pool].input_column_view_start      = input_view_begin_col;
                precompiled_request_handles[item_in_pool].input_row_view_start         = input_view_begin_row;
                precompiled_request_handles[item_in_pool].kernel_input_fmap_view_start = 0;
                precompiled_request_handles[item_in_pool].kernel_out_fmap_view_start   = output_fm_item * C_slice_size;
                precompiled_request_handles[item_in_pool].output_column_view_start     = output_view_begin_col;
                precompiled_request_handles[item_in_pool].output_row_view_start        = output_view_begin_row;
                precompiled_request_handles[item_in_pool].output_row_view_end          = output_view_end_row;
                precompiled_request_handles[item_in_pool].output_fm_view_start         = output_view_begin_map;
                precompiled_request_handles[item_in_pool].output_fm_view_end           = output_view_end_map;
                precompiled_request_handles[item_in_pool].output_image_view_start      = outbatch_begin;
                precompiled_request_handles[item_in_pool].output_image_view_end        = outbatch_end;
                precompiled_request_handles[item_in_pool].padding                      = arg.argument.padding;
                precompiled_request_handles[item_in_pool].callback                     = 
                    get_generator<convolution_generator>(
                    input_width,
                    input_height,
                    input_depth,
                    inpfmap_length,
                    kernel_width,
                    kernel_height,
                    kernel_ifmap,
                    output_width,
                    output_height,
                    output_depth,
                    block_size,
                    out_fm_pckg_size,
                    num_blocks_full,
                    partial_block_size,
                    stride_col,
                    stride_row)->generator_callback;

                tasks.push_back(
                {
                    reinterpret_cast<void(*)(const void*)>(unpack_convolve_callback_handle),
                    &precompiled_request_handles[item_in_pool]
                });
            }
        }
    }
};

convolution_cpu_jit_batch1::~convolution_cpu_jit_batch1() 
{
};

namespace
{

struct attach
{
    attach()
    {
        auto key_fw = std::make_tuple(engine::cpu, memory::format::byxf_f32, memory::format::byxf_f32);
        auto val_fw = convolution_cpu_jit_batch1::create;

        conv_fw_implementation_map.insert( {key_fw, val_fw} );
    }

    ~attach()
    {
    }
};

#ifdef __GNUC__
    __attribute__((visibility("default")))
#elif _MSC_VER
#   pragma section(".nn_init$m", read, write)
#endif
attach attach_impl;

}
}

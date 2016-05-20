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

#include "convolution_cpu_jit_generic.h"
#include "memory_utils.h"
#include "xbyak/xbyak_util.h"
#include <cstddef>
#include <set>

namespace{
    template<typename T> class reverse_t {
        T& ref;
    public:
        reverse_t(T& arg): ref(arg) {}
        auto begin() const -> decltype(ref.rbegin()) { return ref.rbegin (); }
        auto end()   const -> decltype(ref.rend())   { return ref.rend (); }
    };
    template<typename T> reverse_t<T> reverse(T& x) { return reverse_t<T>(x); }

    struct jit_convolution_generic : public neural::is_an_implementation {
#pragma pack(push, 1)
    struct op_data_t {
        float *output;
        float *input;
        float *filter;
        int8_t type; // 0:init, 1:normal, 2:finalize
    };

    struct op_array_t {
        uint64_t count;
        op_data_t *array;
        uint64_t output_feature_block_stride;
        uint64_t output_feature_blocks;
        uint64_t filter_feature_blocks;
    };
#pragma pack(pop)

    std::vector<neural::task> tasks;
    static const uint64_t input_features_per_iteration  = 8;
    static const uint64_t output_features_per_iteration = 4;
    static const uint64_t batch_size                    = 24;
    static const uint64_t register_width_in_float       = 8;

    // code generator
    class jit_code : public Xbyak::CodeGenerator {
        jit_code &operator=(const jit_code&) = delete;
    public:
        jit_code(uint64_t input_feature_maps, void *code_ptr=nullptr, size_t code_size = Xbyak::DEFAULT_MAX_CODE_SIZE)
            : Xbyak::CodeGenerator(code_size, code_ptr)
        {
            using Xbyak::Ymm;

            const auto accumulators_count   = output_features_per_iteration*batch_size/register_width_in_float;
            const auto input_feature_blocks = input_feature_maps/input_features_per_iteration;

            // lambda generating code for one operation:
            // for each block of 4 output feature maps
            //     execute 'prologue'
            //     for each block of input
            //         process one internal block
            auto code_op_type = [&](uint8_t type, std::function<void()> prologue, std::function<void()> epilogue) {

                auto code_op_type_block = [&](int n) {
                    vmovaps(ymm12, ptr [rbx+n*96]   );
                    vmovaps(ymm13, ptr [rbx+n*96+32]);
                    vmovaps(ymm14, ptr [rbx+n*96+64]);
                    vbroadcastss(ymm15, ptr [rcx+(n*output_features_per_iteration+0)*sizeof(float)]); vfmadd231ps(ymm0, ymm12, ymm15); vfmadd231ps(ymm1,  ymm13, ymm15); vfmadd231ps(ymm2,  ymm14, ymm15);
                    vbroadcastss(ymm15, ptr [rcx+(n*output_features_per_iteration+1)*sizeof(float)]); vfmadd231ps(ymm3, ymm12, ymm15); vfmadd231ps(ymm4,  ymm13, ymm15); vfmadd231ps(ymm5,  ymm14, ymm15);
                    vbroadcastss(ymm15, ptr [rcx+(n*output_features_per_iteration+2)*sizeof(float)]); vfmadd231ps(ymm6, ymm12, ymm15); vfmadd231ps(ymm7,  ymm13, ymm15); vfmadd231ps(ymm8,  ymm14, ymm15);
                    vbroadcastss(ymm15, ptr [rcx+(n*output_features_per_iteration+3)*sizeof(float)]); vfmadd231ps(ymm9, ymm12, ymm15); vfmadd231ps(ymm10, ymm13, ymm15); vfmadd231ps(ymm11, ymm14, ymm15);
                };

                std::string op_type_            = std::string("op_type_")          + std::to_string(type);
                std::string op_internal_loop_   = std::string("op_internal_loop_") + std::to_string(type);

                align(4);
                L(op_type_);
                    mov(rbx, r11);
                    mov(rcx, r12);
                    mov(rdx, input_feature_blocks);

                    prologue();

                    align(4);
                    L(op_internal_loop_);
                        for(uint64_t n=0; n<input_features_per_iteration; ++n) code_op_type_block(static_cast<int>(n));
                        add(rbx,    batch_size*output_features_per_iteration*input_features_per_iteration);
                        add(rcx, sizeof(float)*output_features_per_iteration*input_features_per_iteration);
                        dec(rdx);
                    jnz(op_internal_loop_);

                    epilogue();

                    add(rax, r9);
                    add(r12, r10);
                    dec(r8);
                jnz(op_type_);
                add(r15, sizeof(op_data_t));
                cmp(r15, r14);
                jb("op_loop");
            };

            auto code_prologue_load         = [&]() { for(uint64_t n=0; n<accumulators_count; ++n) vmovaps(Ymm(static_cast<int>(n)),  ptr [rax+32*n] );};
            auto code_prologue_zero         = [&]() { for(uint64_t n=0; n<accumulators_count; ++n) vxorps(Ymm(static_cast<int>(n)), Ymm(static_cast<int>(n)), Ymm(static_cast<int>(n)));};
            auto code_epilogue_store        = [&]() { for(uint64_t n=0; n<accumulators_count; ++n) vmovaps(ptr [rax+32*n], Ymm(static_cast<int>(n)));};
            auto code_epilogue_relu_store   = [&]() {
                vxorps(ymm15, ymm15, ymm15);
                for(uint64_t n=0; n<accumulators_count; ++n) vmaxps(Ymm(static_cast<int>(n)), Ymm(static_cast<int>(n)), ymm15);
                for(uint64_t n=0; n<accumulators_count; ++n) vmovaps(ptr [rax+32*n], Ymm(static_cast<int>(n)));
            };

            // <- code starts here

            // lists of registers to preserve on call
#ifdef _WIN32
            auto preserve_registers_scalar  = std::vector<Xbyak::Reg64>{rbx, rdi, r11, r12, r13, r14, r15};
#else
            auto preserve_registers_scalar  = std::vector<Xbyak::Reg64>{rbp, rbx, r11, r12, r13, r14, r15};
#endif

            auto preserve_registers_xmm     = std::vector<Xbyak::Xmm>{xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15};

            // preserve registers on stack
            for(auto &reg : preserve_registers_scalar) push(reg);
            for(auto &reg : preserve_registers_xmm)    {auto offset = (&reg - &(preserve_registers_xmm[0])+1)*16; vmovdqu(ptr [rsp-offset], reg);}

#ifdef __unix__
            mov(rcx, rdi);
#endif
            mov(r15, rcx);  // really a op_array_t *
            mov(r9,  ptr [r15+offsetof(op_array_t, output_feature_block_stride)]);
            mov(r10, ptr [r15+offsetof(op_array_t, filter_feature_blocks)]);
            mov(r13, ptr [r15+offsetof(op_array_t, output_feature_blocks)]);

            // "init" ops: initial zeroing of accumulators (instead of loading them)
            imul(r14, ptr [r15+offsetof(op_array_t,count)], sizeof(op_data_t));
            mov(r15, ptr [r15+offsetof(op_array_t,array)]);
            add(r14, r15);

            // loop over all operations
            align(4);
            L("op_loop");
                mov(r8, r13);
                mov(rax, ptr [r15+offsetof(op_data_t,output)]);
                mov(r11, ptr [r15+offsetof(op_data_t,input)]);
                mov(r12, ptr [r15+offsetof(op_data_t,filter)]);
                cmp(byte [r15+offsetof(op_data_t,type)], 1);
                jne("op_type_02", T_NEAR);

                // normal = load -> calculation -> store
                code_op_type(1, code_prologue_load, code_epilogue_store);
                jmp("end", T_NEAR);

                // tasks 0 & 2
                align(4);
                L("op_type_02");
                cmp(byte [r15+offsetof(op_data_t,type)], 2);
                je("op_type_2", T_NEAR);

                // initial = zeroing -> calculation -> store
                code_op_type(0, code_prologue_zero, code_epilogue_store);

                // initial = load -> calculation -> relu -> store
                code_op_type(2, code_prologue_load, code_epilogue_relu_store);

            L("end");

            // restore registers
            for(auto &reg : reverse(preserve_registers_xmm)) {
                auto offset = (&reg - &preserve_registers_xmm[0]+1)*16;
                vmovdqu(reg, ptr [rsp-offset]);
            }
            for(auto &reg : reverse(preserve_registers_scalar)) pop(reg);

            ret();
        };
    };

    std::vector<std::vector<op_data_t>> op_data;
    std::vector<op_array_t> op_array;

    /*
    float fma_clocks() { return fma_clock_count; }
    */
    const float fma_clock_count = 0.0f;

    jit_convolution_generic(
          float *output, uint64_t output_width, uint64_t output_height, uint64_t output_feature_maps
        , float * input, uint64_t  input_width, uint64_t  input_height, uint64_t  input_feature_maps,uint64_t stride_width, uint64_t stride_height
        , float *filter, uint64_t filter_size
        , uint64_t block_width, uint64_t block_height
    ) : is_an_implementation(neural::type_id<jit_convolution_generic>())
    {
        // create tasks list
        assert(input_feature_maps  % input_features_per_iteration ==0 && "input feature map count is not a multiple of features-per-iteration");
        assert(output_feature_maps % output_features_per_iteration==0 && "output feature map count is not a multiple of features-per-iteration");

        // allocating buffers
        const auto output_feature_blocks = output_feature_maps/output_features_per_iteration;
        const auto input_feature_blocks  = input_feature_maps/input_features_per_iteration;

        const uint64_t fma_per_iteration           = batch_size/register_width_in_float*output_features_per_iteration*input_features_per_iteration;
        const uint64_t fma_per_all_output_features = fma_per_iteration*input_feature_blocks*output_feature_blocks;
        const uint64_t fma_per_image_filtering     = fma_per_all_output_features*output_height*output_width*filter_size*filter_size;
        *const_cast<float *>(&fma_clock_count)     = 0.5f*fma_per_image_filtering;

        // creating tasks
        const auto blocks_in_column = (output_height+block_height-1)/block_height;
        const auto blocks_in_row    = (output_width +block_width -1)/block_width;
        const auto job_count        = blocks_in_column*blocks_in_row;

        static jit_code code(input_feature_maps);

        tasks.resize(job_count);
        op_data.resize(job_count);
        op_array.resize(job_count);

        const auto filter_radius = (filter_size-1)/2;
        for(uint64_t by=0; by<blocks_in_column; ++by)
            for(uint64_t bx=0; bx<blocks_in_row; ++bx) {

                auto output_block_stride = output_features_per_iteration*batch_size;
                auto filter_feature_blocks = output_features_per_iteration*input_feature_maps;

                auto at = by*blocks_in_row + bx;
                std::set<std::tuple<int64_t, int64_t>> exists;
                std::map<std::tuple<int64_t,int64_t,int64_t,int64_t>, op_data_t> sorted;
                auto at_pos = 0;
                for(uint64_t byi=0; byi<block_height; ++byi)
                    for(uint64_t bxi=0; bxi<block_width; ++bxi) {
                        auto y=by*block_height+byi;
                        auto x=bx*block_width +bxi;
                        if(y>=output_height || x>=output_width) continue;
                        for(uint64_t ky=0; ky<filter_size; ++ky) {
                            int64_t kyr=ky-filter_radius;
                            int64_t sy =y*stride_height+kyr;
                            if(sy<0 || static_cast<uint64_t>(sy)>=input_height) continue;
                            for(uint64_t kx=0; kx<filter_size; ++kx) {
                                int64_t kxr=kx-filter_radius;
                                int64_t sx=x*stride_width+kxr;
                                if(sx<0 || static_cast<uint64_t>(sx)>=input_width) continue;
                                //auto at_tuple = std::make_tuple(x,y); todo why unused?

                                auto output_block = output_feature_blocks*(x + output_width*y);
                                auto filter_block = output_feature_blocks*(kx + filter_size*ky);

                                sorted.insert({
                                    std::make_tuple(at_pos, at_pos, x,y),
                                    {
                                        output + output_block_stride*output_block
                                        , input + batch_size*input_feature_maps*((x*stride_width+kx-filter_radius) + input_width*(y*stride_height+ky-filter_radius))
                                        , filter + filter_feature_blocks*filter_block
                                        , 1
                                    }
                                });
                                ++at_pos;
                            }
                        }
                    }
                std::set<std::tuple<int64_t, int64_t>> first;
                for(auto it = sorted.begin(); it!=sorted.end(); ++it) {
                    auto local_at = std::make_tuple(std::get<2>(it->first), std::get<3>(it->first));
                    if(first.find(local_at)==first.end()) {
                        first.insert(local_at);
                        it->second.type = 0;
                    }
                }

                std::set<std::tuple<int64_t, int64_t>> last;
                for(auto it = sorted.rbegin(); it!=sorted.rend(); ++it) {
                    auto local_at = std::make_tuple(std::get<2>(it->first), std::get<3>(it->first));
                    if(last.find(local_at)==last.end()) {
                        last.insert(local_at);
                        it->second.type = 2;
                    }
                }

                for(auto &data : sorted)
                    op_data[at].push_back(std::get<1>(data));

                op_array[at] = {
                    op_data[at].size()
                    , &op_data[at][0]
                    , output_block_stride*sizeof(float)
                    , output_feature_maps/output_features_per_iteration
                    , filter_feature_blocks*sizeof(float)
                };

                tasks[at].callback = reinterpret_cast<void (*)(const void *)>(code.getCode());
                tasks[at].data     = &op_array[at];
            }
    }

    std::vector<neural::task> work() {
        return this->tasks;
    };
    ~jit_convolution_generic(){};
};
}

namespace neural {

convolution_cpu_jit_generic::convolution_cpu_jit_generic(convolution &arg)
    : is_an_implementation(neural::type_id<convolution_cpu_jit_generic>())
    , outer(arg)
{
    auto output_ptr  = reinterpret_cast<float*>(&arg.output_memory(0).pointer);
    auto input_ptr   = reinterpret_cast<float*>(&arg.input_memory(0).pointer);
    auto weights_ptr = reinterpret_cast<float*>(&arg.input_memory(1).pointer);
    //auto bias_ptr    = reinterpret_cast<float*>(&arg.input_memory(2).pointer); //todo add support for biases

    auto& stride      = outer.argument.stride;
    auto& output_size = outer.argument.output_size;
    auto& input_arg   = outer.input_memory(0).argument;
    auto& weights_arg = outer.input_memory(1).argument; //convolution filter

    //const int b_pos = 0;
    const int f_pos = 1;
    const int y_pos = 2;
    const int x_pos = 3;

    assert( 2 == output_size.spatial.size() );
    assert( weights_arg.size.raw[ x_pos ] == weights_arg.size.raw[ y_pos ] ); //todo what is weights format?
    assert( input_arg.size.feature % jit_convolution_generic::input_features_per_iteration ==0 && "input feature count is not multiple of input_features_per_iteration");
    assert( output_size.feature    % jit_convolution_generic::output_features_per_iteration==0 && "output feature count is not multiple of output_features_per_iteration");

    switch(arg.argument.padding){
        case padding::zero:
        {
            // todo jit conv format?
            // todo how to handle offsets?
            jit_convolution_generic* jit_convolution_generic_ptr = new jit_convolution_generic(
                output_ptr,
                output_size.raw[ x_pos ],
                output_size.raw[ y_pos ],
                output_size.raw[ f_pos ],
                input_ptr,
                input_arg.size.raw[ x_pos ],
                input_arg.size.raw[ y_pos ],
                input_arg.size.raw[ f_pos ],
                stride.raw[ x_pos ],
                stride.raw[ y_pos ],
                weights_ptr,
                weights_arg.size.raw[ x_pos+1 ], // filter is square x == y
                1, // magic number, block w
                1  // magic number, block h
                );

            jit_conv_ptr.reset(jit_convolution_generic_ptr);
            break;
        }
        default:
            throw std::runtime_error("Unknown padding mode in convolution.");
    }
};

convolution_cpu_jit_generic::~convolution_cpu_jit_generic()
{
};

namespace
{

struct attach
{
    attach()
    {
        auto key_fw = std::make_tuple(engine::cpu, memory::format::bfxy_f32, memory::format::bfxy_f32); //todo check
        auto val_fw = convolution_cpu_jit_generic::create;

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

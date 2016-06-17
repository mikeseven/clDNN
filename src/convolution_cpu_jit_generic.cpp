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

namespace {
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
    struct op_data_t { //todo offsets can be 32bit, does it impact performance?
        uint64_t output_offset;
        uint64_t input_offset;
        uint64_t filter_offset;
        uint64_t bias_offset;
        int8_t type; // 0:init, 1:normal, 2:finalize
    };

    struct op_array_t {
        uint64_t count;
        op_data_t *array;
        uint64_t output_feature_block_stride;
        uint64_t output_feature_blocks;
        uint64_t filter_feature_blocks;
        float **output;
        float **input;
        float **filter;
        float **bias;
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
                    vmovups(ymm12, ptr [rbx+n*96]   );
                    vmovups(ymm13, ptr [rbx+n*96+32]);
                    vmovups(ymm14, ptr [rbx+n*96+64]);
                    vbroadcastss(ymm15, ptr [rcx+(n*output_features_per_iteration+0)*sizeof(float)]); vfmadd231ps(ymm0, ymm12, ymm15); vfmadd231ps(ymm1,  ymm13, ymm15); vfmadd231ps(ymm2,  ymm14, ymm15);
                    vbroadcastss(ymm15, ptr [rcx+(n*output_features_per_iteration+1)*sizeof(float)]); vfmadd231ps(ymm3, ymm12, ymm15); vfmadd231ps(ymm4,  ymm13, ymm15); vfmadd231ps(ymm5,  ymm14, ymm15);
                    vbroadcastss(ymm15, ptr [rcx+(n*output_features_per_iteration+2)*sizeof(float)]); vfmadd231ps(ymm6, ymm12, ymm15); vfmadd231ps(ymm7,  ymm13, ymm15); vfmadd231ps(ymm8,  ymm14, ymm15);
                    vbroadcastss(ymm15, ptr [rcx+(n*output_features_per_iteration+3)*sizeof(float)]); vfmadd231ps(ymm9, ymm12, ymm15); vfmadd231ps(ymm10, ymm13, ymm15); vfmadd231ps(ymm11, ymm14, ymm15);
                };

                std::string op_type_            = std::string("op_type_")          + std::to_string(type);
                std::string op_internal_loop_   = std::string("op_internal_loop_") + std::to_string(type);

                align(4);
                L(op_type_);
                    mov(rbx, r11);  // input
                    mov(rcx, r12);  // weights
                    mov(rdx, input_feature_blocks);

                    prologue();

                    align(4);
                    L(op_internal_loop_);
                        for(uint64_t n=0; n<input_features_per_iteration; ++n)
                            code_op_type_block(static_cast<int>(n));

                        add(rbx,    batch_size*output_features_per_iteration*input_features_per_iteration);
                        add(rcx, sizeof(float)*output_features_per_iteration*input_features_per_iteration);
                        dec(rdx);
                    jnz(op_internal_loop_);

                    epilogue();

                    add(rax, r9);
                    add(r12, r10);  //r10 filter feature blocks
                    dec(r8);
                jnz(op_type_);
                add(r15, sizeof(op_data_t));
                cmp(r15, r14);
                jb("op_loop");
            };

            auto code_prologue_load         = [&]() { for(uint64_t n=0; n<accumulators_count; ++n) vmovups(Ymm(static_cast<int>(n)),  ptr [rax+32*n] );};
            auto code_prologue_zero         = [&]() {
                for(uint64_t n=0; n<output_features_per_iteration; ++n) {
                    vbroadcastss(Ymm(static_cast<int>(n*3)), ptr [rdi]);
                    vmovaps(Ymm(static_cast<int>(n*3+1)), Ymm(static_cast<int>(n*3)));
                    vmovaps(Ymm(static_cast<int>(n*3+2)), Ymm(static_cast<int>(n*3)));
                    add(rdi, sizeof(float));
                }
            };
            auto code_epilogue_store        = [&]() { for(uint64_t n=0; n<accumulators_count; ++n) vmovups(ptr [rax+32*n], Ymm(static_cast<int>(n)));};

            // <- code starts here

            // lists of registers to preserve on call
#ifdef __unix__
            auto preserve_registers_scalar  = std::vector<Xbyak::Reg64>{rbp, rbx, r11, r12, r13, r14, r15};
#elif _WIN32
            auto preserve_registers_scalar  = std::vector<Xbyak::Reg64>{rbx, rsi, rdi, r11, r12, r13, r14, r15};
#endif

            auto preserve_registers_xmm     = std::vector<Xbyak::Xmm>{xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15};

            // preserve registers on stack
            for(auto &reg : preserve_registers_scalar) push(reg);
            for(auto &reg : preserve_registers_xmm)    {auto offset = (&reg - &(preserve_registers_xmm[0])+1)*16; vmovdqu(ptr [rsp-offset], reg);}

#ifdef __unix__
            mov(r15, rdi);
#elif _WIN32
            mov(r15, rcx);  // really a op_array_t *
#endif
            mov(rsi, r15);

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
                mov(rax, ptr [rsi+offsetof(op_array_t,output)]);
                mov(rax, ptr [rax]);
                add(rax, ptr [r15+offsetof(op_data_t,output_offset)]);
                mov(r11, ptr [rsi+offsetof(op_array_t,input)]);
                mov(r11, ptr [r11]);
                add(r11, ptr [r15+offsetof(op_data_t,input_offset)]);
                mov(r12, ptr [rsi+offsetof(op_array_t,filter)]);
                mov(r12, ptr [r12]);
                add(r12, ptr [r15+offsetof(op_data_t,filter_offset)]);
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
                mov(rdi, ptr [rsi+offsetof(op_array_t,bias)]);
                mov(rdi, ptr [rdi]);
                add(rdi, ptr [r15+offsetof(op_data_t,bias_offset)]);
                code_op_type(0, code_prologue_zero, code_epilogue_store);

                // initial = load -> calculation -> relu -> store
                code_op_type(2, code_prologue_load, code_epilogue_store);

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

    jit_code code;
    std::vector<std::vector<op_data_t>> op_data;
    std::vector<op_array_t> op_array;

    /*
    float fma_clocks() { return fma_clock_count; }
    */
    const float fma_clock_count = 0.0f;

    jit_convolution_generic(
          float **output, uint64_t output_width, uint64_t output_height, uint64_t output_feature_maps
        , float ** input, uint64_t  input_width, uint64_t  input_height, uint64_t  input_feature_maps,uint64_t stride_x, uint64_t stride_y
        , float **filter, uint64_t filter_size
        , float **  bias
        , uint64_t group_count
        , uint64_t batch
    ) : is_an_implementation(neural::type_id<jit_convolution_generic>())
      , code(input_feature_maps/group_count)
    {
        // create tasks list
        assert(input_feature_maps  % input_features_per_iteration ==0 && "input feature map count is not a multiple of features-per-iteration");
        assert(output_feature_maps % output_features_per_iteration==0 && "output feature map count is not a multiple of features-per-iteration");

        const auto input_feature_maps_group  = input_feature_maps  / group_count;
        const auto output_feature_maps_group = output_feature_maps / group_count;
        assert( input_feature_maps_group % input_features_per_iteration ==0 && "input feature map count is not a multiple of features-per-iteration");
        assert(output_feature_maps_group % output_features_per_iteration==0 && "output feature map count is not a multiple of features-per-iteration");

        // allocating buffers
        const auto output_feature_blocks = output_feature_maps/output_features_per_iteration;
        //const auto  input_feature_blocks = input_feature_maps/input_features_per_iteration;
        const auto output_feature_blocks_group = output_feature_maps_group/output_features_per_iteration;
        //const auto  input_feature_blocks_group = input_feature_maps_group /input_features_per_iteration;

        // creating tasks
        const auto batch_blocks = batch/batch_size;
        const auto job_count = output_height*output_width*group_count*batch_blocks;

        tasks.resize(job_count);
        op_data.resize(job_count);
        op_array.resize(job_count);

        auto output_block_stride         = output_features_per_iteration*batch_size;
        //auto filter_feature_blocks       = output_features_per_iteration*input_feature_maps;
        auto filter_feature_blocks_group = output_features_per_iteration*input_feature_maps_group;
        const auto image_spatial_area    = output_height*output_width;
        const auto filter_radius = (filter_size-1)/2;

        for(uint64_t bblock = 0; bblock < batch_blocks; ++bblock)
        for(uint64_t y=0; y<output_height; ++y)
            for(uint64_t x=0; x<output_width; ++x)
            {
                for(uint64_t group=0; group<group_count; ++group) //todo iterate through group_count
                {
                    auto at = x+output_width*(y + output_height*(group + group_count*bblock));
                    std::map<std::tuple<int64_t,int64_t,int64_t,int64_t>, op_data_t> sorted;
                    auto at_pos = 0;
                    if(y>=output_height || x>=output_width) continue;
                    for(uint64_t ky=0; ky<filter_size; ++ky) {
                        int64_t kyr=ky-filter_radius;
                        int64_t sy =y*stride_y+kyr;
                        if(sy<0 || static_cast<uint64_t>(sy)>=input_height) continue;
                        for(uint64_t kx=0; kx<filter_size; ++kx) {
                            int64_t kxr=kx-filter_radius;
                            int64_t sx=x*stride_x+kxr;
                            if(sx<0 || static_cast<uint64_t>(sx)>=input_width) continue;

                            auto output_index = output_feature_blocks*(x + output_width*y) ;
                            auto filter_index = output_feature_blocks*(kx + filter_size*ky);

                            sorted.insert({
                                std::make_tuple(at_pos, 0, x,y),
                                {
                                      sizeof(float)*(output_block_stride*(output_index+group*output_feature_blocks_group+ bblock*image_spatial_area*output_feature_blocks))
                                    , sizeof(float)*(batch_size*(input_feature_maps*(sx + input_width*(sy + bblock* input_height)) + group*input_feature_maps_group ))
                                    //, sizeof(float)*filter_feature_blocks_group*(filter_index+group)
                                    , sizeof(float)*(group*input_feature_maps_group*output_feature_maps_group + filter_index*filter_feature_blocks_group)
                                    , sizeof(float)*output_feature_maps_group*group
                                    , 1
                                }
                            });
                            ++at_pos;
                        }
                    }

                    std::set<std::tuple<int64_t, int64_t, int64_t>> first;
                    for(auto it = sorted.begin(); it!=sorted.end(); ++it) {
                        auto local_at = std::make_tuple(std::get<1>(it->first), std::get<2>(it->first), std::get<3>(it->first));
                        if(first.find(local_at)==first.end()) {
                            first.insert(local_at);
                            it->second.type = 0;
                        }
                    }

                    std::set<std::tuple<int64_t, int64_t, int64_t>> last;
                    for(auto it = sorted.rbegin(); it!=sorted.rend(); ++it) {
                        auto local_at = std::make_tuple(std::get<1>(it->first), std::get<2>(it->first), std::get<3>(it->first));
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
                        , output_feature_blocks/group_count
                        , filter_feature_blocks_group*sizeof(float)
                        , output
                        , input
                        , filter
                        , bias
                    };

                    tasks[at].callback = reinterpret_cast<void (*)(const void *)>(code.getCode());
                    tasks[at].data     = &op_array[at];
                }
             }
    }

    neural::task_group work() {
        return {this->tasks, neural::schedule::split}; // todo check
    };
    ~jit_convolution_generic(){};
};
}

namespace neural {

convolution_cpu_jit_generic::convolution_cpu_jit_generic(convolution &arg)
    : is_an_implementation(neural::type_id<convolution_cpu_jit_generic>())
    , outer(arg)
{
    auto& stride      = outer.argument.stride;
    auto& output_size = outer.argument.output_size;
    auto& input_arg   = outer.input_memory(0).argument;
    auto& weights_arg = outer.input_memory(1).argument; //convolution filter

    //const int b_pos = 0;
    const int f_pos = 1;
    const int x_pos = 2;
    const int y_pos = 3;

    assert( 2 == output_size.spatial.size() );
    assert( weights_arg.size.spatial[0] == weights_arg.size.spatial[1] );
    assert( input_arg.size.feature[0] % jit_convolution_generic::input_features_per_iteration ==0 && "input feature count is not multiple of input_features_per_iteration");
    assert( output_size.feature[0]    % jit_convolution_generic::output_features_per_iteration==0 && "output feature count is not multiple of output_features_per_iteration");

    switch(arg.argument.padding){
        case padding::zero:
        {
            // todo how to handle offsets?
            jit_convolution_generic* jit_convolution_generic_ptr = new jit_convolution_generic(
                reinterpret_cast<float**>(&arg.output_memory(0).pointer),
                output_size.raw[ x_pos ],
                output_size.raw[ y_pos ],
                output_size.raw[ f_pos ],
                reinterpret_cast<float**>(&arg.input_memory(0).pointer),
                input_arg.size.raw[ x_pos ],
                input_arg.size.raw[ y_pos ],
                input_arg.size.raw[ f_pos ],
                stride.raw[ x_pos ],
                stride.raw[ y_pos ],
                reinterpret_cast<float**>(&arg.input_memory(1).pointer),
                weights_arg.size.raw[ x_pos+1 ], // filter is square x == y
                reinterpret_cast<float**>(&arg.input_memory(2).pointer),
                input_arg.size.raw[f_pos]/weights_arg.size.raw[f_pos+1],
                input_arg.size.batch[0]
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
        auto key_fw = std::make_tuple(engine::cpu, memory::format::byxf_b24_f32, memory::format::byxf_b24_f32); //todo check
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

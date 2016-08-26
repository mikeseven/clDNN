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
#include <algorithm>
#include <chrono>
#include <string>

namespace neural {

// neural::memory
//
// Primitive that describes data in memory in known format.
// Used to describe both user-allocated data buffers and internal ones.
// Format defines both layout in memory and reresentation of single value.
// Format is identified by enumeration.
// For each format there one can:
//  - determine number of dimensions & value format using traits
//  - convert coordinates to a memory offset
//
//
// Examples:
//
//   Describe memory avaialble to 'cpu' engine, with memory format yxfb_f32, 3 feature maps, resolution 224x224 and batch 24.
//     auto input  = memory::describe({engine::cpu, memory::format::yxfb_f32, {3,  {224, 224}, 24}});
//
//   Allocate memory avaialble to 'cpu' engine, with memory format yxfb_f32, 3 feature maps, resolution 224x224 and batch 24.
//     auto input  = memory::allocate({engine::cpu, memory::format::yxfb_f32, {3,  {224, 224}, 24}});
struct memory : is_a_primitive {

    struct format_traits {
        const uint8_t       dimension;
        const type_traits  *type;
    };

    class format { format(); public: enum type {
        x_f32,
        xb_f32,     // 1D+batch, float32
        bx_f32,     // 1D+batch, float32
        yxfn_f32,   // 3D + number of neurons - used in fully connected weights
        yxfb_f32,   // 3D+batch, float32
        byxf_f32,   // for convolution_cpu_jit_batch1
        bfyx_f32,   // used in Caffe
        fyxb_f32,   // used in Caffe
        oiyx_f32,   // format used only for weights: o - output feature maps, i - input feature maps
        yxoi_f32,   // format used only for weights: o - output feature maps, i - input feature maps
        oyxi_f32,   // format used only for weights: o - output feature maps, i - input feature maps
        byxf_b24_f32,        // for convolution_cpu_generic
        yxoi_o4_f32,       // for convolution_cpu_generic
        os_yxi_sv16_f32,   // format used only for weights: os - output slice, i - input feature maps, sv16 - 16 values of single slice
        bs_yxf_bv24_f32,
        format_num,
        any=static_cast<uint32_t>(-1)
    }; };

    static format_traits traits(format::type fmt) {
        switch(fmt) {
        case format::   x_f32: return {1, type_id<float>()};
        case format::  bx_f32:
        case format::  xb_f32: return {2, type_id<float>()};
        case format::yxfn_f32:
        case format::yxfb_f32:
        case format::byxf_f32:
        case format::bfyx_f32:
        case format::oiyx_f32:
        case format::yxoi_f32:
        case format::oyxi_f32:
        case format::fyxb_f32:
        case format::bs_yxf_bv24_f32:
        case format::byxf_b24_f32:
        case format::yxoi_o4_f32:
        case format::os_yxi_sv16_f32: return {4, type_id<float>()};
        default: throw std::runtime_error("unknown memory::format");
        }
    }

    struct arguments {
        neural::engine::type            engine;
        neural::memory::format::type    format;
        neural::vector<uint32_t>        size;

        DLL_SYM arguments(neural::engine::type aengine, memory::format::type aformat, neural::vector<uint32_t> asize);
    };

    struct buffer
    {
        virtual void* lock() = 0;
        virtual void release() = 0;
        virtual void reset(void*) = 0;
        virtual size_t size() = 0;
        virtual ~buffer() = default;
    };

    template<typename T>
    class ptr
    {
        std::shared_ptr<buffer> _buffer;
        T* data;
        friend struct memory;
    public:
        ptr(std::shared_ptr<buffer> buffer) : _buffer(buffer), data(static_cast<T*>(buffer->lock())) { }
        ptr(const ptr& rhs) : _buffer(rhs._buffer), data(static_cast<T*>(_buffer->lock())) { }
        ptr& operator=(const ptr& rhs) {
            _buffer->release();
            _buffer = rhs._buffer;
            data = static_cast<T*>(_buffer->lock());
            return *this;
        }

        ~ptr() { _buffer->release(); }

        size_t size() const { return _buffer->size()/sizeof(T); }

#if defined(_SECURE_SCL) && (_SECURE_SCL > 0)
        typedef stdext::checked_array_iterator<T*> iterator;
        stdext::checked_array_iterator<T*> begin() const& {
            return stdext::make_checked_array_iterator(data, size());
        }
        stdext::checked_array_iterator<T*> end() const& {
            return stdext::make_checked_array_iterator(data, size(), size());
        }
#else
        typedef T* iterator;
        T* begin() const& { return data; }
        T* end() const& { return data + size(); }
#endif

        T& operator[](size_t idx) const& {
            assert(idx < size());
            return data[idx];
        }

        friend bool operator==(const ptr& lhs, const ptr& rhs) {
            return lhs.data == rhs.data;
        }

        friend bool operator!=(const ptr& lhs, const ptr& rhs) {
            return !(lhs == rhs);
        }

        // do not use this class as temporary object
        // ReSharper disable CppMemberFunctionMayBeStatic, CppMemberFunctionMayBeConst
        void begin() && {}
        void end() && {}
        void operator[](size_t idx) && {}
        // ReSharper restore CppMemberFunctionMayBeConst, CppMemberFunctionMayBeStatic
    };

    const arguments argument;

    std::shared_ptr<buffer> get_buffer() const { return _buffer; }

    template<typename T>
    ptr<T> pointer() const { return ptr<T>(get_buffer()); }

    DLL_SYM static primitive describe(arguments);
    DLL_SYM static primitive allocate(arguments);
    DLL_SYM static size_t size_of(arguments);

    void execute_argument(void *arg) const override
    {
        get_buffer()->reset(arg);
    }
    DLL_SYM size_t count() const;

private:
    std::shared_ptr<buffer> _buffer;
    memory(arguments arg, std::shared_ptr<buffer> buffer) : is_a_primitive(type_id<const memory>()), argument(arg), _buffer(buffer) {};
    friend class is_a_primitive;
};




// neural::file
//
// File that is loaded and becomes a data.
//
//
// Examples:
//
//   Load data from file into memory available to CPU.
//     auto weight = file::create({engine::cpu, "weight.nnb"});
//
//   Load data from file into memory available to CPU & validate that format is yxfb_f32, 96 feature maps, 3D data [224x224x3].
//     auto weight = file::create({engine::cpu, "weight.nnb"}, memory::format::yxfb_f32, {96, {224, 224, 3}}});
struct file : is_a_primitive {
    enum weights_type
    {
        convolution,
        fully_connected
    };
    struct arguments {
        neural::engine::type         engine;
        std::string                  name;
        weights_type                 weight_type;
        std::vector<primitive>       output;

        DLL_SYM arguments(neural::engine::type aengine, std::string aname, memory::format::type aformat, std::vector<uint32_t> &asize);
        DLL_SYM arguments(neural::engine::type aengine, std::string aname, primitive aoutput);
        DLL_SYM arguments(neural::engine::type aengine, std::string aname, weights_type = weights_type::convolution);
    };
    const arguments argument;

    DLL_SYM static primitive create(arguments);
    DLL_SYM static void serialize(const primitive&, const std::string&);
    file &operator()(void *);
private:
    file(arguments arg) : is_a_primitive(type_id<const file>()), argument(arg) {};
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// neural::reorder
//
// Changes how data is ordered in memory. Value type is not changed & all information is preserved.
// Corresponding values are bitwise equal before/after reorder.
//
//
// Examples:
//
//   Reorder yxfb_f32 to byxf_f32 on user-specified buffers on reference engine.
//     neural::primitive input   = memory::describe({engine::reference, memory::format::yxfb_f32, {16, {4, 8}, 1}});
//     neural::primitive output  = memory::describe({engine::reference, memory::format::byxf_f32, {16, {4, 8}, 1}});
//     neural::primitive reorder = reorder::create(reorder::arguments{engine::reference,input,output});
struct reorder : is_a_primitive {
    struct arguments {
        neural::engine::type        engine;
        std::vector<primitive>      output;
        std::vector<primitive_at>   input;  // 1: {input}

        DLL_SYM arguments(neural::engine::type engine, primitive_at input, primitive output);
        DLL_SYM arguments(neural::engine::type engine, neural::memory::format::type out_fmt, neural::vector<uint32_t> out_sizes, primitive_at input);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { reorder::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    reorder(arguments arg) : is_a_primitive(type_id<const reorder>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// neural::mean_subtract
//
// Subtract mean from input
//
// Example:
//
//  auto input = memory::describe({engine::refetence, memory::format::yxfb_f32, { 16, {4, 8}, 3 }));
//  auto output = memory::describe({engine::refetence, memory::format::yxfb_f32, { 16, {4, 8}, 3 }));
//  auto mean = file::create({engine::cpu, "mean.nnb"});
//  neural::primitive mean_subtract = mean_subtract::create({engine::reference, output, input, mean});
//
struct mean_subtract : is_a_primitive {
    struct arguments {
        neural::engine::type      engine;
        std::vector<primitive>    output;
        std::vector<primitive_at> input; // 2: input, mean

        DLL_SYM arguments(neural::engine::type, primitive out, primitive in, primitive mean);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt, primitive in, primitive mean);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { mean_subtract::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    mean_subtract(arguments arg) : is_a_primitive(type_id<const mean_subtract>()), argument(arg) {};
    const std::vector<primitive_at> &input() const { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// struct common for convolution and convolution_relu to reduce code duplication
struct convolution_common {
    struct arguments {
        neural::engine::type      engine;
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;            // 3 : {input, weight, bias}
        neural::vector<int32_t>   input_offset;
        neural::vector<uint32_t>  stride;
        neural::padding::type     padding;
        size_t                    split; // on how many cards split the computation to

        arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                 neural::vector<uint32_t> stride, neural::padding::type, size_t split = 1);
        arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                 uint32_t                 stride, neural::padding::type, size_t split = 1);
        arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type, size_t split = 1);
        arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                                                  neural::padding::type, size_t split = 1);
        arguments(neural::engine::type, primitive                    out,                                                                         std::vector<primitive_at> in,                                 neural::vector<uint32_t> stride, neural::padding::type, size_t split = 1);
        arguments(neural::engine::type, primitive                    out,                                                                         std::vector<primitive_at> in,                                                                  neural::padding::type, size_t split = 1);
        arguments(neural::engine::type, primitive                    out,     neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, std::vector<primitive_at> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type, size_t split = 1);
    };

    static void validate_params(const arguments &arg);
};




// neural::convolution
//
// Performs forward spatial convolution with weight sharing.
// Parameters are defined in context of "direct" convolution, but actual algorithm is not implied.
// Look into docs/size_offset_stride_padding.html for description how size, offsets, stride & padding parameter work.
//
//
// Example:
//
//   In batch 24 convolve 224x224 3-feature-map user-specified inputs into 96-feature-map user-specified outputs.
//     auto input  = memory::describe({engine::cpu, memory::format::yxfb_f32, {3,  {224, 224}, 24}});
//     auto output = memory::describe({engine::cpu, memory::format::yxfb_f32, {96, {224, 224}, 24}});
//     auto weight = file::create({engine::cpu, "weight.nnb"});
//     auto bias   = file::create({engine::cpu, "bias.nnb"});
//     auto conv   = convolution::create({engine::cpu, output, input, weight, bias, padding::zero});
//
//   As above, but convolution allocated it's output buffer.
//     auto input  = memory::describe({engine::cpu, memory::format::yxfb_f32, {3,  {224, 224}, 24}});
//     auto weight = file::create({engine::cpu, "weight.nnb"});
//     auto bias   = file::create({engine::cpu, "bias.nnb"});
//     auto conv   = convolution::create({engine::cpu, memory::format::yxfb_f32, input, weight, bias, padding::zero});
struct convolution : is_a_primitive {
    struct arguments : convolution_common::arguments{
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, size_t split=1);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                 neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, size_t split=1);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                 uint32_t                 stride, neural::padding::type = neural::padding::type::zero, size_t split=1);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                                                  neural::padding::type = neural::padding::type::zero, size_t split=1);
        DLL_SYM arguments(neural::engine::type, primitive                    out,                                                                         std::vector<primitive_at> in,                                 neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, size_t split=1);
        DLL_SYM arguments(neural::engine::type, primitive                    out,                                                                         std::vector<primitive_at> in,                                                                  neural::padding::type = neural::padding::type::zero, size_t split=1);
        DLL_SYM arguments(neural::engine::type, primitive                    out,     neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, std::vector<primitive_at> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, size_t split=1);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { convolution::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    convolution(arguments arg) : is_a_primitive(type_id<const convolution>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// neural::convolution_relu
//
// Fused layer: convolution fused with relu.
//
//
// Example:
//   In batch 24 convolve & relu-activate 224x224 3-feature-map user-specified inputs into 96-feature-map user-specified outputs.
//     auto input  = memory::describe({engine::cpu, memory::format::yxfb_f32, {3,  {224, 224}, 24}});
//     auto output = memory::describe({engine::cpu, memory::format::yxfb_f32, {96, {224, 224}, 24}});
//     auto weight = file::create({engine::cpu, "weight.nnb"});
//     auto bias   = file::create({engine::cpu, "bias.nnb"});
//     auto conv   = convolution_relu::create({engine::cpu, output, input, weight, bias, padding::zero, 0.0f});
struct convolution_relu : is_a_primitive {
    struct arguments : convolution_common::arguments {
        float                     negative_slope;

        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, float negative_slope=0, size_t split=1);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                 neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, float negative_slope=0, size_t split=1);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                 uint32_t                 stride, neural::padding::type = neural::padding::type::zero, float negative_slope=0, size_t split=1);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                                                  neural::padding::type = neural::padding::type::zero, float negative_slope=0, size_t split=1);
        DLL_SYM arguments(neural::engine::type, primitive                        out,                                                                     std::vector<primitive_at> in,                                 neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, float negative_slope=0, size_t split=1);
        DLL_SYM arguments(neural::engine::type, primitive                        out,                                                                     std::vector<primitive_at> in,                                                                  neural::padding::type = neural::padding::type::zero, float negative_slope=0, size_t split=1);
        DLL_SYM arguments(neural::engine::type, primitive                        out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, std::vector<primitive_at> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, float negative_slope=0, size_t split=1);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { convolution_relu::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    convolution_relu(arguments arg) : is_a_primitive(type_id<const convolution_relu>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

    friend class is_a_primitive;
};




// neural::convolution_backward
//
// Performs backward spatial convolution with weight sharing.
// Parameters are defined in context of "direct" convolution, but actual algorithm is not implied.
//
//
// Examples:
//
//   Backward pass:
//     auto bw_output    = memory::describe({eng, memory::format::yxfb_f32, {1, {2, 2}, 1}});
//     auto bw_input     = memory::describe({eng, memory::format::yxfb_f32, {1, {3, 3}, 1}});
//     auto fw_input     = memory::describe({eng, memory::format::yxfb_f32, {1, {2, 2}, 1}});
//     auto weights      = memory::describe({eng, memory::format::yxfb_f32, {1, {2, 2}, 1}});
//     auto weights_diff = memory::describe({eng, memory::format::yxfb_f32, {1, {2, 2}, 1}});
//     auto biases       = memory::describe({eng, memory::format::x_f32,    {1, {{1}} , 1}});
//     auto biases_diff  = memory::describe({eng, memory::format::x_f32,    {1, {{1}} , 1}});
//     auto conv_bw = convolution_backward::create({engine::reference,
//         std::vector<primitive>{bw_output, weights_diff, biases_diff},
//         {bw_input, fw_input, weights, biases}, {1, {1, 1}, 1}, padding::zero});
struct convolution_backward : is_a_primitive {
    struct arguments {
        neural::engine::type      engine;
        std::vector<primitive>    output;         // 3: {backward output, weight diff, bias diff}
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  input_size;
        std::vector<primitive_at> input;          // 4: {backward input, forward input, filter, bias}
        neural::vector<int32_t>   input_offset;
        neural::vector<uint32_t>  stride;
        neural::padding::type     padding;

        DLL_SYM arguments(neural::engine::type, std::vector<neural::memory::format::type> out_fmt, neural::vector<uint32_t> out_off, neural::vector<uint32_t> in_siz, std::vector<primitive> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, std::vector<neural::memory::format::type> out_fmt,                                                                    std::vector<primitive> in,                                 neural::vector<uint32_t> stride, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, std::vector<neural::memory::format::type> out_fmt,                                                                    std::vector<primitive> in,                                 uint32_t                 stride, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, std::vector<primitive>                    out,                                                                        std::vector<primitive> in,                                 neural::vector<uint32_t> stride, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, std::vector<primitive>                    out,     neural::vector<uint32_t> out_off, neural::vector<uint32_t> in_siz, std::vector<primitive> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { convolution::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    convolution_backward(arguments arg) : is_a_primitive(type_id<const convolution_backward>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// neural::fully_connected
//
// Forward pass of fully connected layer.
//
//
// Example:
//    6 input neurons 7 output neurons.
//     auto input   = memory::describe({engine::reference, memory::format::xb_f32, { 1, {{6}},  1} });
//     auto output  = memory::describe({engine::reference, memory::format::xb_f32, { 1, {{7}},  1} });
//     auto weights = memory::describe({engine::reference, memory::format::xy_f32, { 1, {6, 7}, 1} });
//     auto biases  = memory::describe({engine::reference, memory::format::x_f32,  { 1, {{7}},  1} });
//     auto act = fully_connected::create({engine::reference, output, input, weights, biases});
struct fully_connected : is_a_primitive {
    struct arguments {
        neural::engine::type        engine;
        std::vector<primitive>      output;
        std::vector<primitive_at>   input;  // 3: {input, weights, bias}

        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt, primitive in, primitive weights, primitive bias);
        DLL_SYM arguments(neural::engine::type, primitive                    out,     primitive in, primitive weights, primitive bias);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { fully_connected::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    fully_connected(arguments arg) : is_a_primitive(type_id<const fully_connected>()), argument(arg) {};
    const std::vector<primitive_at>  &input()  const { return argument.input;  };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// neural::relu
//
// Activiation using rectified linear unit, forward pass.
//
//
// Example:
//   Perform max(x,0) activation on user specified buffers.
//     auto input  = memory::describe({engine::cpu, memory::format::yxfb_f32, {224, 224, 3, 24}});
//     auto output = memory::describe({engine::cpu, memory::format::yxfb_f32, {224, 224, 3, 24}});
//     auto act    = relu::create({engine::cpu, output, input});
struct relu : is_a_primitive {
    struct arguments {
        neural::engine::type      engine;
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;            // 1: {input}
        neural::vector<int32_t>   input_offset;
        float                     negative_slope;

        DLL_SYM arguments(neural::engine::type, memory::format::type out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, float slp);
        DLL_SYM arguments(neural::engine::type, memory::format::type out,                                                                     primitive in,                                 float slp);
        DLL_SYM arguments(neural::engine::type, memory::format::type out,                                                                     primitive in);
        DLL_SYM arguments(neural::engine::type, primitive            out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, float slp);
        DLL_SYM arguments(neural::engine::type, primitive            out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off);
        DLL_SYM arguments(neural::engine::type, primitive            out,                                                                     primitive in,                                  float slp);
        DLL_SYM arguments(neural::engine::type, primitive            out,                                                                     primitive in);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { relu::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    relu(arguments arg) : is_a_primitive(type_id<const relu>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// neural::relu_backward
//
// Activiation using rectified linear unit, backward pass.
//
//
// Example:
//   Backward pass:
//     auto forward_input       = memory::describe({engine::reference, memory::format::yxfb_f32, {8, {8, 8}, 3}});
//     auto forward_output_grad = memory::describe({engine::reference, memory::format::yxfb_f32, {8, {8, 8}, 3}});
//     auto forward_input_grad  = memory::describe({engine::reference, memory::format::yxfb_f32, {8, {8, 8}, 3}});
//     auto act = relu_backward::create({engine::reference, {forward_input_grad}, {forward_output_grad, forward_input}});
struct relu_backward : is_a_primitive {
    struct arguments {
        neural::engine::type                   engine;
        std::vector<primitive>                 output;         // 1: {forward_input_grad}
        neural::vector<uint32_t>               output_offset;
        neural::vector<uint32_t>               output_size;
        std::vector<primitive_at>              input;          // 2: {forward_output_grad, forward_input}
        std::vector<neural::vector<uint32_t>>  input_offset;
        float                                  negative_slope;

        DLL_SYM arguments(neural::engine::type, primitive out, neural::vector<uint32_t> out_offset, neural::vector<uint32_t> out_size, std::vector<primitive_at> in, std::vector<neural::vector<uint32_t>> in_offsets, float neg_slope = 0.0f);
        DLL_SYM arguments(neural::engine::type, primitive out,                                                                          std::vector<primitive_at> in,                                                  float neg_slope = 0.0f);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { relu_backward::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

private:
    relu_backward(arguments arg) : is_a_primitive(type_id<const relu_backward>()), argument(arg) {};
    friend class is_a_primitive;
};




// neural::pooling
//
// Pooling, forward.
//
//
// Example:
//   2x2 max pooling with stride 1 and offset.
//     auto input  = memory::describe({engine::reference, memory::format::yxfb_f32, { 2, {6, 6}, 2}});
//     auto output = memory::describe({engine::reference, memory::format::yxfb_f32, { 3, {7, 7}, 3}});
//     auto pool  = pooling::create({engine::reference, pooling::mode::max,
//                                     output, {0,{1,2},1}, {2,{5,5},2},    // output primitive, offset, size
//                                     input, {0,{0,0},0},                  // input primitive, offset
//                                     {1,{1,1},1},                         // stride
//                                     {1,{2, 2},1},                        // pooling size
//                                     padding::zero});                     // padding mode
struct pooling : is_a_primitive {
    class mode { mode(); public: enum type { max, average }; };

    struct arguments {
        neural::engine::type      engine;
        pooling::mode::type       mode;
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;          // 1: {input}
        neural::vector<int32_t>   input_offset;
        neural::vector<uint32_t>  stride;
        neural::vector<uint32_t>  size;
        neural::padding::type     padding;

        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, neural::memory::format::type o_frmt, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, neural::memory::format::type o_frmt,                                                                     primitive in,                                 neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, neural::memory::format::type o_frmt,                                                                     primitive in,                                 uint32_t                 strd, uint32_t                 siz, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,                                                                        primitive in,                                 uint32_t                 strd, uint32_t                 siz, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,                                                                        primitive in,                                 neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,                                                                        primitive in, neural::vector<int32_t> in_off, neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,                                                                        primitive in, neural::vector<int32_t> in_off, uint32_t                 strd, uint32_t                 siz, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,    neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,                                                                        primitive in,                                 neural::vector<uint32_t> strd,                               neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,                                                                        primitive in,                                 uint32_t                 strd,                               neural::padding::type);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { pooling::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    pooling(arguments arg) : is_a_primitive(type_id<const pooling>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};





namespace normalization { /////////////////////////////////////////////////////////////////////////////////////////////


// neural::normalization::response
//
// Forward local response normalization as described in chapter 3.3 of "ImageNet Classification with Deep Convolutional
// Neural Networks" by Khrizevsky, Sutskever, Hinton. see: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
// Alogrithm:
//   b(i,x,y) = a(i,x,y) / (k+alpha*sum(min(N-1, i+n/2); j=max(0,i-n/2); a(j,x,y)^2))
// Where:
//   b(i,x,y) : value at x, y from i-th feature map after normalization
//   b(i,x,y) : value at x, y from i-th feature map before normalization
//   N : number of feature maps
//   n : size of normalization
//   k, alpha, beta : hyper parameters (equal to 2, 10e-4, 0.75 in paper)
//
//
// Example:
//
//   LRN on ragion of 3 with [k,alpha,beta] = [1,1,0.75]
//     auto  input = memory::describe({ engine::reference, memory::format::yxfb_f32, {1, {2, 2}, 7}});
//     auto output = memory::describe({ engine::reference, memory::format::yxfb_f32, {1, {2, 2}, 7}});
//     auto lrn = normalization::response::create({engine::reference, output, input, 3, padding::zero, 1.0f, 1.0f, 0.75f});
struct /*normalization*/response : is_a_primitive {
    struct arguments {
        neural::engine::type        engine;
        std::vector<primitive>      output;
        vector<uint32_t>            output_offset;
        vector<uint32_t>            output_size;
        std::vector<primitive_at>   input;          // 1: input
        vector<int32_t>             input_offset;
        uint32_t                    size;
        neural::padding::type       padding;
        float                       k;
        float                       alpha;
        float                       beta;

        DLL_SYM arguments(neural::engine::type _engine, primitive _output,          primitive _input, uint32_t  _size, neural::padding::type _padding, float _k, float _alpha, float _beta);
        DLL_SYM arguments(neural::engine::type _engine, memory::format::type _output_fmt, primitive _input, uint32_t  _size, neural::padding::type _padding, float _k, float _alpha, float _beta);
        DLL_SYM arguments(neural::engine::type _engine, primitive _output,          vector<uint32_t> _output_offset, vector<uint32_t> _output_size, primitive _input, vector<int32_t> _input_offset, uint32_t _input_size, neural::padding::type _padding, float _k, float _alfa, float _beta);

    };
    const arguments argument;

    struct query_entry : is_a_query_entry { response::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    response(arguments arg) : is_a_primitive(type_id<const response>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// neural::normalization::softmax
//
// Normalizes results so they sum to 1.
// Algorithm:
//   b = e^a/sum(N-1; j=0; e^j)
// Where:
//   N : number of values to normalize
//   b : value after normalization
//   a : value before normalization
struct /*normalization*/softmax : is_a_primitive {
    struct arguments {
        neural::engine::type      engine;
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;          // 1: input
        neural::vector<int32_t>   input_offset;

        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt, primitive in);
        DLL_SYM arguments(neural::engine::type, primitive                             out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off);
        DLL_SYM arguments(neural::engine::type, primitive                             out, primitive in);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { softmax::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
private:
    softmax(arguments arg) : is_a_primitive(type_id<const softmax>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// neural::normalization::batch_training_forward
//
// Performs batch normalization as discribed in "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
// by Ioffe, Szegedy, see: http://arxiv.org/abs/1502.03167
// This is forward pass ran during training.
//
//
// Example:
//   Normalize 320x240, 3 feature maps images in batch 16.
//     auto input               = memory::describe({ engine::reference, memory::format::yxfb_f32, {16, {240, 320}, 3}});
//     auto bias                = memory::describe({ engine::reference, memory::format::yxfb_f32, { 1, {  1,   1}, 3}});
//     auto scale               = memory::describe({ engine::reference, memory::format::yxfb_f32, { 1, {  1,   1}, 3}});
//     auto output              = memory::describe({ engine::reference, memory::format::yxfb_f32, {16, {240, 320}, 3}});
//     auto current_inv_std_dev = memory::describe({ engine::reference, memory::format::yxfb_f32, { 1, {  1,   1}, 3}});
//     auto moving_average      = memory::describe({ engine::reference, memory::format::yxfb_f32, { 1, {  1,   1}, 3}});
//     auto moving_inv_std_dev  = memory::describe({ engine::reference, memory::format::yxfb_f32, { 1, {  1,   1}, 3}});
//     auto current_average     = memory::describe({ engine::reference, memory::format::yxfb_f32, { 1, {  1,   1}, 3}});
//     auto bn = normalization::batch_training_forward::create({engine::reference, {output, current_average, current_inv_std_dev, moving_average, moving_inv_std_dev}, {input, scale, bias}, 1.0, std::numeric_limits<float>::epsilon()});

struct /*normalization*/batch_training_forward : is_a_primitive {
    struct arguments {
        neural::engine::type        engine;
        std::vector<primitive>      output;         // 3-5: {output, current_mean, current_inv_std_dev, [moving_mean, moving_inv_std_dev]}
        std::vector<primitive_at>   input;          // 3: {input, scale, bias}
        bool                        spatial;
        double                      exp_avg_factor;
        double                      epsilon;

        DLL_SYM arguments(neural::engine::type, std::vector<primitive>, std::vector<primitive_at>, bool, double, double);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { batch_training_forward::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

private:
    batch_training_forward(arguments arg) : is_a_primitive(type_id<const batch_training_forward>()), argument(arg) {};
    friend class is_a_primitive;
};




// neural::normalization::batch_training_backward
//
// Backward pass fo batch normalization.
//
//
// Example:
//   Backward pass of batch normalization on 16x32 images with 64 feature maps and batch 128.
//     auto forward_input       = memory::describe({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
//     auto forward_scale       = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto forward_bias        = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto output_grad         = memory::describe({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
//     auto current_mean        = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto current_inv_std_dev = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto input_grad = memory::describe({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
//     auto scale_grad = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto bias_grad  = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto bn = normalization::batch_training_backward::create({engine::reference, {input_grad, scale_grad, bias_grad}, {forward_input, forward_scale, forward_bias, output_grad, current_mean, current_inv_std_dev}});
struct /*normalization*/batch_training_backward : is_a_primitive {
    struct arguments {
        neural::engine::type        engine;
        std::vector<primitive>      output;         // 3: {input_grad, scale_grad, bias_grad}
        std::vector<primitive_at>   input;          // 6: {forward_input, forward_scale, forward_bias, output_grad, current_mean, current_inv_std_dev}
        bool                        spatial;

        DLL_SYM arguments(neural::engine::type, std::vector<primitive>, std::vector<primitive_at>, bool);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { batch_training_backward::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

private:
    batch_training_backward(arguments arg) : is_a_primitive(type_id<const batch_training_backward>()), argument(arg) {};
    friend class is_a_primitive;
};




// neural::normalization::batch_inference
//
// Batch normalization, forward pass for inference.
//
//
// Example:
//   Inference pass of batch normalization on 16x32 images with 64 feature maps and batch 128.
//     auto input       = memory::describe({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
//     auto scale       = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto bias        = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto average     = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto inv_std_dev = memory::describe({engine::reference, memory::format::yxfb_f32, {1, {1, 1}, 64}});
//     auto output      = memory::describe({engine::reference, memory::format::yxfb_f32, {128, {16, 32}, 64}});
//     auto bn = normalization::batch_inference::create({engine::reference, {output}, {input, scale, bias, average, inv_std_dev}, true});

struct /*normalization*/batch_inference : is_a_primitive
{
    struct arguments
    {
        neural::engine::type        engine;
        std::vector<primitive>      output;         // 1: {output}
        std::vector<primitive_at>   input;          // 5: {input, scale, bias, precomputed_mean, precomputed_inv_std_dev}
        bool                        spatial;

        DLL_SYM arguments(neural::engine::type, std::vector<primitive>, std::vector<primitive_at>, bool);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { batch_inference::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

private:
    batch_inference(arguments arg) : is_a_primitive(type_id<const batch_inference>()), argument(arg) {};
    friend class is_a_primitive;
};//normalization /////////////////////////////////////////////////////////////////////////////////////////////////////
}



// neural::fully_connected_relu
//
// Fused layer: fully connected fused with relu.
//
//
// Example:
//    6 input neurons 7 output neurons with relu activation.
//     auto input   = memory::describe({engine::reference, memory::format::xb_f32, { 1, {{6}},  1} });
//     auto output  = memory::describe({engine::reference, memory::format::xb_f32, { 1, {{7}},  1} });
//     auto weights = memory::describe({engine::reference, memory::format::xy_f32, { 1, {6, 7}, 1} });
//     auto biases  = memory::describe({engine::reference, memory::format::x_f32,  { 1, {{7}},  1} });
//     auto act = fully_connected_relu::create({engine::reference, output, input, weights, biases, 0.0f});

struct fully_connected_relu : is_a_primitive {
    struct arguments {
        neural::engine::type        engine;
        std::vector<primitive>      output;
        std::vector<primitive_at>   input;  // 3: {input, weights, bias}
        float                       negative_slope;

        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt, primitive in, primitive weights, primitive bias, float negative_slope);
        DLL_SYM arguments(neural::engine::type, primitive                        out, primitive in, primitive weights, primitive bias, float negative_slope);
    };
    const neural::fully_connected_relu::arguments argument;

    struct query_entry : is_a_query_entry { neural::fully_connected_relu::arguments arguments; };
    static std::vector<query_entry> query(neural::fully_connected_relu::arguments);
    DLL_SYM static primitive create(neural::fully_connected_relu::arguments);
private:
    fully_connected_relu(fully_connected_relu::arguments arg) : is_a_primitive(type_id<const fully_connected_relu>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};




// neural::worker_cpu
//
// Worker for executing primitives for engine::cpu.
// Internally implemented as thread pool.
class nn_thread_worker_pool;
struct worker_cpu : is_a_worker {
    struct arguments {
        uint32_t thread_pool_size;

        DLL_SYM arguments(uint32_t arg_threadpool_size);
        DLL_SYM arguments(                            );
    };
    arguments argument;

    const bool owns_pool;
    const std::unique_ptr<nn_thread_worker_pool> thread_pool;

    DLL_SYM static worker create(arguments);
    DLL_SYM static worker create(arguments, nn_thread_worker_pool &);
    DLL_SYM void execute(const neural::task_group& requests) const;
    DLL_SYM neural::engine::type engine() const {return neural::engine::cpu;}

    ~worker_cpu();

private:
    worker_cpu(arguments arg);
    worker_cpu(arguments arg, nn_thread_worker_pool &);
};

class program_builder;
struct worker_gpu : is_a_worker {
    struct arguments {
        bool profiling_enabled;

        DLL_SYM arguments();
        DLL_SYM arguments(bool profiling);
    };

    DLL_SYM static worker create(arguments arg = arguments());
    DLL_SYM void execute(const neural::task_group& requests) const override;
    neural::engine::type engine() const override { return neural::engine::gpu; }
    DLL_SYM std::vector<std::pair<std::string, std::chrono::nanoseconds>> get_profiling_info() const;
private:
    std::shared_ptr<program_builder> builder;
    worker_gpu(arguments);
};

} // namespace neural

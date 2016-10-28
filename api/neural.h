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

namespace neural 
{
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
//   Describe memory available to gpu, with memory format yxfb_f32, 3 feature maps, resolution 224x224 and batch 24.
//     auto input  = memory::describe({memory::format::yxfb_f32, {3,  {224, 224}, 24}});
//
//   Allocate memory available to gpu, with memory format yxfb_f32, 3 feature maps, resolution 224x224 and batch 24.
//     auto input  = memory::allocate({memory::format::yxfb_f32, {3,  {224, 224}, 24}});
struct memory : is_a_primitive 
{
    struct format_traits
    {
        const uint8_t       dimension;
        const type_traits  *type;
    };

    class format 
    { 
        format();

    public:
        enum type : uint8_t 
        {
            // FP32 (single precision float)
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
            yxio_f32,   // format used only for weights: o - output feature maps, i - input feature maps
            byxf_b24_f32,        // for convolution_cpu_generic
            yxoi_o4_f32,       // for convolution_cpu_generic
            os_yxi_sv16_f32,   // format used only for weights: os - output slice, i - input feature maps, sv16 - 16 values of single slice
            bs_yxf_bv24_f32,

            // FP16 (half precision float)
            x_f16,
            xb_f16,            // 1D+batch, FP16 (half precision float)
            bx_f16,            // 1D+batch, FP16 (half precision float)
            yxfn_f16,          // 3D + number of neurons - used in fully connected weights
            yxfb_f16,          // 3D+batch, FP16 (half precision float)
            byxf_f16,          // for convolution_cpu_jit_batch1
            bfyx_f16,          // used in Caffe
            fyxb_f16,          // used in Caffe
            oiyx_f16,          // format used only for weights: o - output feature maps, i - input feature maps
            yxoi_f16,          // format used only for weights: o - output feature maps, i - input feature maps
            oyxi_f16,          // format used only for weights: o - output feature maps, i - input feature maps
            yxio_f16,          // format used only for weights: o - output feature maps, i - input feature maps
            byxf_b24_f16,      // for convolution_cpu_generic
            yxoi_o4_f16,       // for convolution_cpu_generic
            os_yxi_sv16_f16,   // format used only for weights: os - output slice, i - input feature maps, sv16 - 16 values of single slice
            bs_yxf_bv24_f16,

            format_num,
            any = static_cast<uint8_t>(-1)
        }; 
    };

    static format_traits traits(format::type fmt) 
    {
        switch(fmt) 
        {
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
        case format::yxio_f32:
        case format::fyxb_f32:
        case format::bs_yxf_bv24_f32:
        case format::byxf_b24_f32:
        case format::yxoi_o4_f32:
        case format::os_yxi_sv16_f32: return {4, type_id<float>()};

        case format::   x_f16: return {1, type_id<half_t>()};
        case format::  bx_f16:
        case format::  xb_f16: return {2, type_id<half_t>()};
        case format::yxfn_f16:
        case format::yxfb_f16:
        case format::byxf_f16:
        case format::bfyx_f16:
        case format::oiyx_f16:
        case format::yxoi_f16:
        case format::oyxi_f16:
        case format::yxio_f16:
        case format::fyxb_f16:
        case format::bs_yxf_bv24_f16:
        case format::byxf_b24_f16:
        case format::yxoi_o4_f16:
        case format::os_yxi_sv16_f16: return {4, type_id<half_t>()};
        default: throw std::runtime_error("unknown memory::format");
        }
    }

    struct arguments 
	{
        neural::memory::format::type    format;
        neural::vector<uint32_t>        size;

        DLL_SYM arguments(memory::format::type aformat, neural::vector<uint32_t> asize);
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
        stdext::checked_array_iterator<T*> begin() const& 
		{
            return stdext::make_checked_array_iterator(data, size());
        }

        stdext::checked_array_iterator<T*> end() const& 
		{
            return stdext::make_checked_array_iterator(data, size(), size());
        }
#else
        typedef T* iterator;
        T* begin() const& { return data; }
        T* end() const& { return data + size(); }
#endif

        T& operator[](size_t idx) const& 
		{
            assert(idx < size());
            return data[idx];
        }

        friend bool operator==(const ptr& lhs, const ptr& rhs) 
		{
            return lhs.data == rhs.data;
        }

        friend bool operator!=(const ptr& lhs, const ptr& rhs) 
		{
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
//   Load data from file into memory available to GPU.
//     auto weight = file::create({"weight.nnb"});
//
//   Load data from file into memory available to GPU & validate that format is yxfb_f32, 96 feature maps, 3D data [224x224x3].
//     auto weight = file::create({"weight.nnb"}, memory::format::yxfb_f32, {96, {224, 224, 3}}});
struct file : is_a_primitive 
{
    enum weights_type
    {
        bias,
        convolution,
        fully_connected,
        mean
    };
    struct arguments 
	{
        std::string                  name;
        weights_type                 weight_type;
        std::vector<primitive>       output;

        DLL_SYM arguments(std::string aname, memory::format::type aformat, std::vector<uint32_t> &asize);
        DLL_SYM arguments(std::string aname, primitive aoutput);
        DLL_SYM arguments(std::string aname, weights_type = weights_type::convolution);
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
// Also merged with subtraction layer, which can subtract values while doing reordering.
// NOTE THAT THIS WILL SUBTRACT THE SAME VALUES FROM EACH BATCH.
//
// Examples:
//
//   Reorder yxfb_f32 to byxf_f32 on user-specified buffers .
//     neural::primitive input   = memory::describe({memory::format::yxfb_f32, {16, {4, 8}, 1}});
//     neural::primitive output  = memory::describe({memory::format::byxf_f32, {16, {4, 8}, 1}});
//     neural::primitive reorder = reorder::create(reorder::arguments{input,output});
struct reorder : is_a_primitive 
{
    struct arguments 
	{
        std::vector<primitive>      output;
        std::vector<primitive_at>   input;  // 1/2: {input} / {input, subtract_values}
        std::vector<float>          subtract_per_feature; // values to subtract from feature/channel, only one value per feature/channel.
        bool dummy; // TODO!!! - dummy parameter needed because of primitive conversion to anything, so primitive can convert to primitive_at or std::vector<float>... this is because of bad design, need to change it in future!!!!!!

        DLL_SYM arguments(primitive_at input, primitive output);
        DLL_SYM arguments(primitive output, primitive input, primitive values_to_subtract);
        DLL_SYM arguments(primitive output, primitive input, const std::vector<float>& value_to_subtract, bool dummy);
        DLL_SYM arguments(neural::memory::format::type out_fmt, neural::vector<uint32_t> out_sizes, primitive_at input);
        DLL_SYM arguments(neural::memory::format::type out_fmt, neural::vector<uint32_t> out_sizes, primitive_at input, primitive_at values_to_subtract);
        DLL_SYM arguments(neural::memory::format::type out_fmt, neural::vector<uint32_t> out_sizes, primitive_at input, const std::vector<float>& value_to_subtract, bool dummy);

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



// neural::depth_concatenate
//
// Concatenates two or more input data into one output data.
// TODO!!!! Write some good documentation here!
//
// Examples:
//
//   Concatenate two yxfb_f32 inputs into one yxfb_f32 output.
//     neural::primitive input1  = memory::describe({memory::format::yxfb_f32, {16, {4, 8}, 2}});
//     neural::primitive input2  = memory::describe({memory::format::yxfb_f32, {16, {4, 8}, 2}});
//     neural::primitive output  = memory::describe({memory::format::yxfb_f32, {16, {4, 8}, 4}});
//     neural::primitive depth_concatenate = depth_concatenate::create( reorder::arguments{ { input1, input2 }, output } );
struct depth_concatenate : is_a_primitive
{
    struct arguments
    {
        std::vector<primitive>      output;
        std::vector<primitive_at>   input;

        DLL_SYM arguments(std::vector<primitive_at> input, primitive output);
        DLL_SYM arguments(neural::memory::format::type out_fmt, std::vector<primitive_at> input);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { depth_concatenate::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);

private:
    depth_concatenate(arguments arg) : is_a_primitive(type_id<const depth_concatenate>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
    friend class is_a_primitive;
};


// neural::mean_subtract
//
// Subtract mean from input
//
// Example:
//
//  auto input = memory::describe({memory::format::yxfb_f32, { 16, {4, 8}, 3 }));
//  auto output = memory::describe({ememory::format::yxfb_f32, { 16, {4, 8}, 3 }));
//  auto mean = file::create({"mean.nnb"});
//  neural::primitive mean_subtract = mean_subtract::create({output, input, mean});
//
struct mean_subtract : is_a_primitive 
{
    struct arguments 
	{
        std::vector<primitive>    output;
        std::vector<primitive_at> input; // 2: input, mean

        DLL_SYM arguments(primitive out, primitive in, primitive mean);
        DLL_SYM arguments(neural::memory::format::type out_fmt, primitive in, primitive mean);
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


// neural::convolution
//
// Performs forward spatial convolution with weight sharing. Also supports built-in Relu available by setting it in arguments.
// Parameters are defined in context of "direct" convolution, but actual algorithm is not implied.
// Look into docs/size_offset_stride_padding.html for description how size, offsets, stride & padding parameter work.
//
//
// Example:
//
//   In batch 24 convolve 224x224 3-feature-map user-specified inputs into 96-feature-map user-specified outputs.
//     auto input  = memory::describe({memory::format::yxfb_f32, {3,  {224, 224}, 24}});
//     auto output = memory::describe({memory::format::yxfb_f32, {96, {224, 224}, 24}});
//     auto weight = file::create({"weight.nnb"});
//     auto bias   = file::create({"bias.nnb"});
//     auto conv   = convolution::create({output, input, weight, bias, padding::zero});
//
//   As above, but convolution allocated it's output buffer.
//     auto input  = memory::describe({memory::format::yxfb_f32, {3,  {224, 224}, 24}});
//     auto weight = file::create({"weight.nnb"});
//     auto bias   = file::create({"bias.nnb"});
//     auto conv   = convolution::create({memory::format::yxfb_f32, input, weight, bias, padding::zero});
//
// Example:
//   In batch 24 convolve & relu-activate 224x224 3-feature-map user-specified inputs into 96-feature-map user-specified outputs.
//     auto input  = memory::describe({memory::format::yxfb_f32, {3,  {224, 224}, 24}});
//     auto output = memory::describe({memory::format::yxfb_f32, {96, {224, 224}, 24}});
//     auto weight = file::create({"weight.nnb"});
//     auto bias   = file::create({"bias.nnb"});
//     auto conv   = convolution::create({output, input, weight, bias, padding::zero, 0.0f, 1, true});
struct convolution : is_a_primitive 
{
    struct arguments 
	{
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;            // 3 : {input, weight, bias}
        neural::vector<int32_t>   input_offset;
        neural::vector<uint32_t>  stride;
        neural::padding::type     padding;
        size_t                    split; // on how many cards split the computation to
        bool                      use_relu;
        float                     negative_slope;

        DLL_SYM arguments(neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, size_t split=1, bool use_relu=false, float negative_slope=0.0f);
        DLL_SYM arguments(neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                 neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, size_t split=1, bool use_relu=false, float negative_slope=0.0f);
        DLL_SYM arguments(neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                 uint32_t                 stride, neural::padding::type = neural::padding::type::zero, size_t split=1, bool use_relu=false, float negative_slope=0.0f);
        DLL_SYM arguments(neural::memory::format::type out_fmt,                                                                     std::vector<primitive_at> in,                                                                  neural::padding::type = neural::padding::type::zero, size_t split=1, bool use_relu=false, float negative_slope=0.0f);
        DLL_SYM arguments(primitive                    out,                                                                         std::vector<primitive_at> in,                                 neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, size_t split=1, bool use_relu=false, float negative_slope=0.0f);
        DLL_SYM arguments(primitive                    out,                                                                         std::vector<primitive_at> in,                                                                  neural::padding::type = neural::padding::type::zero, size_t split=1, bool use_relu=false, float negative_slope=0.0f);
        DLL_SYM arguments(primitive                    out,     neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, std::vector<primitive_at> in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, neural::padding::type = neural::padding::type::zero, size_t split=1, bool use_relu=false, float negative_slope=0.0f);
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


// neural::fully_connected
//
// Forward pass of fully connected layer. Also supports built-in Relu available by setting it in arguments.
//
//
// Example:
//    6 input neurons 7 output neurons.
//     auto input   = memory::describe({memory::format::xb_f32, { 1, {{6}},  1} });
//     auto output  = memory::describe({memory::format::xb_f32, { 1, {{7}},  1} });
//     auto weights = memory::describe({memory::format::xy_f32, { 1, {6, 7}, 1} });
//     auto biases  = memory::describe({memory::format::x_f32,  { 1, {{7}},  1} });
//     auto act = fully_connected::create({output, input, weights, biases});
//
// Example:
//    6 input neurons 7 output neurons with relu activation.
//     auto input   = memory::describe({memory::format::xb_f32, { 1, {{6}},  1} });
//     auto output  = memory::describe({memory::format::xb_f32, { 1, {{7}},  1} });
//     auto weights = memory::describe({memory::format::xy_f32, { 1, {6, 7}, 1} });
//     auto biases  = memory::describe({memory::format::x_f32,  { 1, {{7}},  1} });
//     auto act = fully_connected::create({output, input, weights, biases, true, 0.0f});
struct fully_connected : is_a_primitive 
{
    struct arguments
	{
        std::vector<primitive>      output;
        std::vector<primitive_at>   input;  // 3: {input, weights, bias}
        bool                        use_relu;
        float                       negative_slope;

        DLL_SYM arguments(neural::memory::format::type out_fmt, primitive in, primitive weights, primitive bias, bool use_relu=false, float negative_slope=0.0f);
        DLL_SYM arguments(primitive                    out,     primitive in, primitive weights, primitive bias, bool use_relu=false, float negative_slope=0.0f);
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
//     auto input  = memory::describe({memory::format::yxfb_f32, {224, 224, 3, 24}});
//     auto output = memory::describe({memory::format::yxfb_f32, {224, 224, 3, 24}});
//     auto act    = relu::create({output, input});
struct relu : is_a_primitive 
{
    struct arguments 
	{
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;            // 1: {input}
        neural::vector<int32_t>   input_offset;
        float                     negative_slope;

        DLL_SYM arguments(memory::format::type out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, float slp);
        DLL_SYM arguments(memory::format::type out,                                                                     primitive in,                                 float slp);
        DLL_SYM arguments(memory::format::type out,                                                                     primitive in);
        DLL_SYM arguments(primitive            out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, float slp);
        DLL_SYM arguments(primitive            out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off);
        DLL_SYM arguments(primitive            out,                                                                     primitive in,                                  float slp);
        DLL_SYM arguments(primitive            out,                                                                     primitive in);
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


// neural::pooling
//
// Pooling, forward.
//
//
// Example:
//   2x2 max pooling with stride 1 and offset.
//     auto input  = memory::describe({memory::format::yxfb_f32, { 2, {6, 6}, 2}});
//     auto output = memory::describe({memory::format::yxfb_f32, { 3, {7, 7}, 3}});
//     auto pool  = pooling::create({pooling::mode::max,
//                                     output, {0,{1,2},1}, {2,{5,5},2},    // output primitive, offset, size
//                                     input, {0,{0,0},0},                  // input primitive, offset
//                                     {1,{1,1},1},                         // stride
//                                     {1,{2, 2},1},                        // pooling size
//                                     padding::zero});                     // padding mode
struct pooling : is_a_primitive 
{
    class mode { mode(); public: enum type { max, average }; };

    struct arguments 
	{
        pooling::mode::type       mode;
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;          // 1: {input}
        neural::vector<int32_t>   input_offset;
        neural::vector<uint32_t>  stride;
        neural::vector<uint32_t>  size;
        neural::padding::type     padding;

        DLL_SYM arguments(neural::pooling::mode::type, neural::memory::format::type o_frmt, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::pooling::mode::type, neural::memory::format::type o_frmt,                                                                     primitive in,                                 neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::pooling::mode::type, neural::memory::format::type o_frmt,                                                                     primitive in,                                 uint32_t                 strd, uint32_t                 siz, neural::padding::type);
        DLL_SYM arguments(neural::pooling::mode::type, primitive                    out,                                                                        primitive in,                                 uint32_t                 strd, uint32_t                 siz, neural::padding::type);
        DLL_SYM arguments(neural::pooling::mode::type, primitive                    out,                                                                        primitive in,                                 neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::pooling::mode::type, primitive                    out,                                                                        primitive in, neural::vector<int32_t> in_off, neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::pooling::mode::type, primitive                    out,                                                                        primitive in, neural::vector<int32_t> in_off, uint32_t                 strd, uint32_t                 siz, neural::padding::type);
        DLL_SYM arguments(neural::pooling::mode::type, primitive                    out,    neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, neural::vector<uint32_t> strd, neural::vector<uint32_t> siz, neural::padding::type);
        DLL_SYM arguments(neural::pooling::mode::type, primitive                    out,                                                                        primitive in,                                 neural::vector<uint32_t> strd,                               neural::padding::type);
        DLL_SYM arguments(neural::pooling::mode::type, primitive                    out,                                                                        primitive in,                                 uint32_t                 strd,                               neural::padding::type);
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


namespace normalization 
{ 
/////////////////////////////////////////////////////////////////////////////////////////////
// neural::normalization::response
//
// Forward local response normalization as described in chapter 3.3 of "ImageNet Classification with Deep Convolutional
// Neural Networks" by Khrizevsky, Sutskever, Hinton. see: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
// Algorithm:
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
//   LRN on region of 3 with [k,alpha,beta] = [1,1,0.75]
//     auto  input = memory::describe({ memory::format::yxfb_f32, {1, {2, 2}, 7}});
//     auto output = memory::describe({ memory::format::yxfb_f32, {1, {2, 2}, 7}});
//     auto lrn = normalization::response::create({output, input, 3, padding::zero, 1.0f, 1.0f, 0.75f});
struct /*normalization*/response : is_a_primitive 
{
    struct arguments 
	{
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

        DLL_SYM arguments(primitive _output,          primitive _input, uint32_t  _size, neural::padding::type _padding, float _k, float _alpha, float _beta);
        DLL_SYM arguments(memory::format::type _output_fmt, primitive _input, uint32_t  _size, neural::padding::type _padding, float _k, float _alpha, float _beta);
        DLL_SYM arguments(primitive _output,          vector<uint32_t> _output_offset, vector<uint32_t> _output_size, primitive _input, vector<int32_t> _input_offset, uint32_t _input_size, neural::padding::type _padding, float _k, float _alfa, float _beta);
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
struct /*normalization*/softmax : is_a_primitive 
{
    struct arguments 
	{
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;          // 1: input
        neural::vector<int32_t>   input_offset;

        DLL_SYM arguments(neural::memory::format::type out_fmt, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off);
        DLL_SYM arguments(neural::memory::format::type out_fmt, primitive in);
        DLL_SYM arguments(primitive                             out, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off);
        DLL_SYM arguments(primitive                             out, primitive in);
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

}//normalization /////////////////////////////////////////////////////////////////////////////////////////////////////


class program_builder;
namespace instrumentation {
    struct profiling_info;
}
struct worker_gpu : is_a_worker 
{
    DLL_SYM static worker create();
    DLL_SYM void execute(const neural::task_group& requests) const override;
    DLL_SYM const std::vector<instrumentation::profiling_info>& get_profiling_info() const;

private:
    std::shared_ptr<program_builder> builder;
    worker_gpu();
};

} // namespace neural

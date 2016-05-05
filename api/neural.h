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

namespace neural {

// data in memory in known format; format = {order, type} of values
struct memory : is_a_primitive {

    struct format_traits {
        const uint8_t       dimension;
        const type_traits  *type;
    };

    class format { format(); public: enum type {
        scalar_f32, // single scalar, float32
        x_f32,
        xb_f32,     // 1D+batch, float32
        yxfb_f32,   // 3D+batch, float32
        fyxb_f32,
        xyfb_f32,
        fxyb_f32,
        byxf_f32,
        bfyx_f32,
        bxyf_f32,
        bfxy_f32,
        scalar_f64, // single scalar, float64
        x_f64,
        yxfb_f64,   // 3D+batch, float64
        fyxb_f64,
        xyfb_f64,
        fxyb_f64,
        byxf_f64,
        bfyx_f64,
        bxyf_f64,
        bfxy_f64,
        any=static_cast<uint32_t>(-1)
    }; };

    static const format_traits traits(format::type fmt) {
        switch(fmt) {
        case format::scalar_f32:
        case format::   x_f32: return {1, type_id<float>()};
		case format::  xb_f32: return {2, type_id<float>()};
        case format::yxfb_f32:
        case format::fyxb_f32:
        case format::xyfb_f32:
        case format::fxyb_f32:
        case format::byxf_f32:
        case format::bfyx_f32:
        case format::bxyf_f32:
		case format::bfxy_f32: return {4, type_id<float>()};
        case format::scalar_f64:
        case format::   x_f64: return {1, type_id<double>()};
        case format::yxfb_f64:
        case format::fyxb_f64:
        case format::xyfb_f64:
        case format::fxyb_f64:
        case format::byxf_f64:
        case format::bfyx_f64:
        case format::bxyf_f64:
        case format::bfxy_f64: return {4, type_id<double>()};
        default: throw std::runtime_error("unknown memory::format");
        }
    }

    struct arguments {
        neural::engine::type            engine;
        neural::memory::format::type    format;
        neural::vector<uint32_t>        size;
        bool                            owns_memory;

        DLL_SYM arguments(neural::engine::type aengine, memory::format::type aformat, neural::vector<uint32_t> asize);
        DLL_SYM arguments(neural::engine::type aengine, memory::format::type aformat, neural::vector<uint32_t> asize, bool aowns_memory);
    };
    const arguments argument;
    mutable void *pointer;

    DLL_SYM static primitive create(arguments);
    memory &operator()(void *ptr) { pointer = ptr; return *this; };
    primitive clone() const { return create(argument); }
    void execute_argument(void *arg) const {
        if(argument.owns_memory) throw std::runtime_error("memory::execute_argument: this a container with its own memory; cannot set new pointer");
        else pointer = arg;
    }
    DLL_SYM size_t count() const;

    ~memory();
private:

    memory(arguments arg) : is_a_primitive(type_id<const memory>()), argument(arg), pointer(0) {};
};

// file that is loaded and becomes a data
struct file : is_a_primitive {
    struct arguments {
        neural::engine::type    engine;
        std::string             name;
        std::vector<primitive>  output;

        DLL_SYM arguments(neural::engine::type aengine, std::string aname, memory::format::type aformat, std::vector<uint32_t> &asize);
        DLL_SYM arguments(neural::engine::type aengine, std::string aname, memory::format::type aformat);
        DLL_SYM arguments(neural::engine::type aengine, std::string aname, primitive aoutput);
        DLL_SYM arguments(neural::engine::type aengine, std::string aname);
    };
    const arguments argument;

    DLL_SYM static primitive create(arguments);
    file &operator()(void *);
    primitive clone() const { return create(argument); }
private:
    file(arguments arg) : is_a_primitive(type_id<const file>()), argument(arg) {};
    const std::vector<primitive>     &output() const { return argument.output; };
};
// reorder data, type is not changed
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
    primitive clone() const { return create(argument); }
private:
    reorder(arguments arg) : is_a_primitive(type_id<const reorder>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
};
// direct convolution
struct convolution : is_a_primitive {
    struct arguments {
        neural::engine::type      engine;
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;
        neural::vector<int32_t>   input_offset;
        neural::vector<uint32_t>  stride;
        primitive                 weight;
        primitive                 bias;
        neural::padding::type     padding;

        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt, neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, primitive weights, primitive biases, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     primitive in,                                 neural::vector<uint32_t> stride, primitive weights, primitive biases, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                                                     primitive in,                                 uint32_t                 stride, primitive weights, primitive biases, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, primitive                    out,                                                                         primitive in,                                 neural::vector<uint32_t> stride, primitive weights, primitive biases, neural::padding::type);
        DLL_SYM arguments(neural::engine::type, primitive                    out,     neural::vector<uint32_t> out_off, neural::vector<uint32_t> out_siz, primitive in, neural::vector<int32_t> in_off, neural::vector<uint32_t> stride, primitive weights, primitive biases, neural::padding::type);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { convolution::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    convolution(arguments arg) : is_a_primitive(type_id<const convolution>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

    std::unique_ptr<is_an_implementation> _private;
};
struct convolution_backward : is_a_primitive {
    struct arguments {
        neural::engine::type      engine;
        std::vector<primitive>    output;         // 3: {bw_output, weight_diff, bias diff}
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  input_size;
        std::vector<primitive_at> input;          // 4: {bw_input, fw_input, filter, bias}
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
    primitive clone() const { return create(argument); }
private:
    convolution_backward(arguments arg) : is_a_primitive(type_id<const convolution_backward>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

    std::unique_ptr<is_an_implementation> _private;
};
// fully connected
struct fully_connected : is_a_primitive {
    struct arguments {
        neural::engine::type        engine;
        std::vector<primitive>      output;
        neural::vector<uint32_t>    output_size;
        std::vector<primitive_at>   input;

        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt, neural::vector<uint32_t> out_siz, primitive in, primitive weights);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type out_fmt,                                   primitive in, primitive weights);
        DLL_SYM arguments(neural::engine::type, primitive                    out,                                       primitive in, primitive weights);
        DLL_SYM arguments(neural::engine::type, primitive                    out,     neural::vector<uint32_t> out_siz, primitive in, primitive weights);

    };
    const arguments argument;

    struct query_entry : is_a_query_entry { fully_connected::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    fully_connected(arguments arg) : is_a_primitive(type_id<const fully_connected>()), argument(arg) {};
    const std::vector<primitive_at>  &input()  const { return argument.input;  };
    const std::vector<primitive>     &output() const { return argument.output; };

    std::unique_ptr<is_an_implementation> _private;
};
// relu activation
// [TODO] need to activation base class ?
// [TODO] "any" on slope ?
struct relu : is_a_primitive {
    struct arguments {
        neural::engine::type      engine;
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;          // 1: input
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
    primitive clone() const { return create(argument); }
private:
    relu(arguments arg) : is_a_primitive(type_id<const relu>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

    std::unique_ptr<is_an_implementation> _private;
};
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
    primitive clone() const { return create(argument); }

private:
    relu_backward(arguments arg) : is_a_primitive(type_id<const relu_backward>()), argument(arg) {};

    std::unique_ptr<is_an_implementation> _private;
};
// pooling
struct pooling : is_a_primitive {
    class mode { mode(); public: enum type { max, average }; };

    struct arguments {
        neural::engine::type      engine;
        pooling::mode::type       mode;
        std::vector<primitive>    output;
        neural::vector<uint32_t>  output_offset;
        neural::vector<uint32_t>  output_size;
        std::vector<primitive_at> input;          // 1: input
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
    primitive clone() const { return create(argument); }
private:
    pooling(arguments arg) : is_a_primitive(type_id<const pooling>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

    std::unique_ptr<is_an_implementation> _private;
};


namespace normalization { /////////////////////////////////////////////////////////////////////////////////////////////
// normalization of response
//todo remove memory_obsolete
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

            DLL_SYM arguments(neural::engine::type _engine, primitive _output, primitive _input, uint32_t  _size, neural::padding::type _padding, float _k, float _alpha, float _beta);
            DLL_SYM arguments(neural::engine::type _engine, primitive _output, vector<uint32_t> _output_offset, vector<uint32_t> _output_size, primitive _input, vector<int32_t> _input_offset, uint32_t _input_size, neural::padding::type _padding, float _k, float _alfa, float _beta);

        };
        const arguments argument;

        struct query_entry : is_a_query_entry { response::arguments arguments; };
        static std::vector<query_entry> query(arguments);
        DLL_SYM static primitive create(arguments);
        primitive clone() const { return create(argument); }
    private:
        response(arguments arg) : is_a_primitive(type_id<const response>()), argument(arg) {};
        const std::vector<primitive_at>  &input() const { return argument.input; };
        const std::vector<primitive>     &output() const { return argument.output; };

        std::unique_ptr<is_an_implementation> _private;
    };


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
        primitive clone() const { return create(argument); }
    private:
        softmax(arguments arg) : is_a_primitive(type_id<const softmax>()), argument(arg) {};
        const std::vector<primitive_at>  &input() const { return argument.input; };
        const std::vector<primitive>     &output() const { return argument.output; };

        std::unique_ptr<is_an_implementation> _private;
    };

// batch normalization training - forward
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
    primitive clone() const { return create(argument); }
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

private:
    batch_training_forward(arguments arg) : is_a_primitive(type_id<const batch_training_forward>()), argument(arg) {};
};

// batch normalization training - backward
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
    primitive clone() const { return create(argument); }
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

private:
    batch_training_backward(arguments arg) : is_a_primitive(type_id<const batch_training_backward>()), argument(arg) {};
};

// batch normalization inference
struct /*normalization*/batch_inference : is_a_primitive {
    struct arguments {
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
    primitive clone() const { return create(argument); }
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

private:
    batch_inference(arguments arg) : is_a_primitive(type_id<const batch_inference>()), argument(arg) {};
};

};//normalization /////////////////////////////////////////////////////////////////////////////////////////////////////

// direct convolution+relu
struct convolution_relu : is_a_primitive {
    struct arguments {
        neural::engine::type        engine;
        std::vector<primitive>      output;
        vector<uint32_t>            output_offset;
        vector<uint32_t>            output_size;
        std::vector<primitive_at>   input;          // 3: input, filter, bias
        vector<int32_t>             input_offset;
        vector<uint32_t>            input_stride;
        neural::padding::type       padding;
        float                       negative_slope;

        DLL_SYM arguments(neural::engine::type, neural::memory::format::type, vector<uint32_t>, vector<uint32_t>, primitive, vector<int32_t>, vector<uint32_t>, primitive, primitive, neural::padding::type, float);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type,                                     primitive,                  vector<uint32_t>, primitive, primitive, neural::padding::type, float);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type,                                     primitive,                  uint32_t,         primitive, primitive, neural::padding::type, float);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type,                                     primitive,                                    primitive, primitive, neural::padding::type, float);
        DLL_SYM arguments(neural::engine::type, primitive,                                                        primitive,                                    primitive, primitive, neural::padding::type, float);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { convolution_relu::arguments arguments; };
    static std::vector<query_entry> query(arguments);
    DLL_SYM static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    convolution_relu(arguments arg) : is_a_primitive(type_id<const convolution_relu>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

    std::unique_ptr<is_an_implementation> _private;
};

// fully connected + relu
struct fully_connected_relu : is_a_primitive {
    struct arguments {
        neural::engine::type        engine;
        std::vector<primitive>      output;
        vector<uint32_t>            output_offset;
        vector<uint32_t>            output_size;
        std::vector<primitive_at>   input;          // 3: input, filter, bias
        vector<int32_t>             input_offset;
        vector<uint32_t>            input_stride;
        float                       negative_slope;

        DLL_SYM arguments(neural::engine::type, neural::memory::format::type, vector<uint32_t>, vector<uint32_t>, primitive, vector<int32_t>, vector<uint32_t>, primitive, primitive, float);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type,                                     primitive,                  vector<uint32_t>, primitive, primitive, float);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type,                                     primitive,                  uint32_t,         primitive, primitive, float);
        DLL_SYM arguments(neural::engine::type, neural::memory::format::type,                                     primitive,                                    primitive, primitive, float);
        DLL_SYM arguments(neural::engine::type, primitive,                                                        primitive,                                    primitive, primitive, float);
    };
    const neural::fully_connected_relu::arguments argument;

    struct query_entry : is_a_query_entry { neural::fully_connected_relu::arguments arguments; };
    static std::vector<query_entry> query(neural::fully_connected_relu::arguments);
    DLL_SYM static primitive create(neural::fully_connected_relu::arguments);
    primitive clone() const { return create(neural::fully_connected_relu::argument); }
private:
    fully_connected_relu(fully_connected_relu::arguments arg) : is_a_primitive(type_id<const fully_connected_relu>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
};

}

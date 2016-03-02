#pragma once

#include "neural_base.h"

namespace neural {

// data in memory in known format; format = {order, type} of values
struct memory : is_a_primitive {
	enum axis {x,y,f,b};
    struct format_traits {
        const uint8_t       dimension;
		const std::vector<axis>	order;
        const type_traits  *type;
    };

    class format { format(); public: enum type {
        xb_f32,     // 1D+batch, float32
        yxfb_f32,   // 3D+batch, float32
        fyxb_f32,
        xyfb_f32,
        fxyb_f32,
        byxf_f32,
        bfyx_f32,
        bxyf_f32,
        bfxy_f32,
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
		case format::  xb_f32: return {2, {x,b} , type_id<float>()};
        case format::yxfb_f32: return {4, {y,x,f,b} , type_id<float>()};
        case format::fyxb_f32: return {4, {f,y,x,b} , type_id<float>()};
        case format::xyfb_f32: return {4, {x,y,f,b} , type_id<float>()};
        case format::fxyb_f32: return {4, {f,x,y,b} , type_id<float>()};
        case format::byxf_f32: return {4, {b,y,x,f} , type_id<float>()};
        case format::bfyx_f32: return {4, {b,f,y,x} , type_id<float>()};
        case format::bxyf_f32: return {4, {b,x,y,f} , type_id<float>()};
		case format::bfxy_f32: return {4, {b,f,x,y}, type_id<float>()};
        case format::yxfb_f64: return {4, {y,x,f,b}, type_id<float>()};
        case format::fyxb_f64: return {4, {f,y,x,b}, type_id<float>()};
        case format::xyfb_f64: return {4, {x,y,f,b}, type_id<float>()};
        case format::fxyb_f64: return {4, {f,x,y,b}, type_id<float>()};
        case format::byxf_f64: return {4, {b,y,x,f}, type_id<float>()};
        case format::bfyx_f64: return {4, {b,f,y,x}, type_id<float>()};
        case format::bxyf_f64: return {4, {b,x,y,f}, type_id<float>()};
        case format::bfxy_f64: return {4, {b,f,x,y}, type_id<double>()};
        default: throw std::runtime_error("unknown memory::format");
        }
    }

    struct arguments {
        engine::type            engine;
        format::type            format;
        std::vector<uint32_t>   size;
        bool                    owns_memory;

        arguments(neural::engine::type aengine, memory::format::type aformat, std::vector<uint32_t> asize);
        arguments(neural::engine::type aengine, memory::format::type aformat, std::vector<uint32_t> asize, bool aowns_memory);
    };
    const arguments argument;
    mutable void *pointer;

    static primitive create(arguments);
    memory &operator()(void *ptr) { pointer = ptr; return *this; };
    primitive clone() const { return create(argument); }
    void execute_argument(void *arg) const {
        if(argument.owns_memory) throw std::runtime_error("memory::execute_argument: this a container with its own memory; cannot set new pointer");
        else pointer = arg;
    }
    size_t count() const;

    ~memory();
private:
    memory(arguments arg) : is_a_primitive(type_id<const memory>()), argument(arg), pointer(0) {};
};



// file that is loaded and becomes a data
struct file : is_a_primitive {
    struct arguments {
        engine::type            engine;
        std::string             name;
        std::vector<primitive>  output;

        arguments(neural::engine::type aengine, std::string aname, memory::format::type aformat, std::vector<uint32_t> &asize);
        arguments(neural::engine::type aengine, std::string aname, memory::format::type aformat);
        arguments(neural::engine::type aengine, std::string aname, primitive aoutput);
        arguments(neural::engine::type aengine, std::string aname);
    };
    const arguments argument;

    static primitive create(arguments);
    file &operator()(void *);
    primitive clone() const { return create(argument); }
private:
    file(arguments arg) : is_a_primitive(type_id<const file>()), argument(arg) {};
    const std::vector<primitive>     &output() const { return argument.output; };
};



// reorder data, type is not changed
struct reorder : is_a_primitive {
    struct arguments {
        engine::type                engine;
        std::vector<primitive>      output;
        std::vector<primitive_at>   input;  // 1: {input}

        arguments(neural::engine::type, neural::memory::format::type,std::vector<uint32_t> out_sizes, primitive_at);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments;};
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    reorder(arguments arg) : is_a_primitive(type_id<const reorder>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
};

// direct convolution
struct convolution : is_a_primitive {
    struct arguments {
        engine::type                engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 3: {input, filter, bias}
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       input_stride;
        padding::type               padding;

        arguments(neural::engine::type, neural::memory::format::type, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, std::vector<uint32_t>, primitive, primitive, neural::padding::type);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                       std::vector<uint32_t>, primitive, primitive, neural::padding::type);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                       uint32_t,              primitive, primitive, neural::padding::type);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                                              primitive, primitive, neural::padding::type);
        arguments(neural::engine::type, primitive,                                                                  primitive,                                              primitive, primitive, neural::padding::type);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    convolution(arguments arg) : is_a_primitive(type_id<const convolution>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
};



// fully connected
struct fully_connected : is_a_primitive {
    struct arguments {
        engine::type                engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 3: input, filter, bias
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       input_stride;

        arguments(neural::engine::type, neural::memory::format::type, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, std::vector<uint32_t>, primitive, primitive);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                       std::vector<uint32_t>, primitive, primitive);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                       uint32_t,              primitive, primitive);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                                              primitive, primitive);
        arguments(neural::engine::type, primitive,                                                                  primitive,                                              primitive, primitive);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    fully_connected(arguments arg) : is_a_primitive(type_id<const fully_connected>()), argument(arg) {};
    const std::vector<primitive_at>  &input()  const { return argument.input;  };
    const std::vector<primitive>     &output() const { return argument.output; };
};



// relu activation
// [TODO] need to activation base class ?
// [TODO] "any" on slope ?
struct relu : is_a_primitive {
    struct arguments {
        engine::type              engine;
        std::vector<primitive>    output;
        std::vector<uint32_t>     output_offset;
        std::vector<uint32_t>     output_size;
        std::vector<primitive_at> input;          // 1: input
        std::vector<int32_t>      input_offset;
        float                     negative_slope;

        arguments(neural::engine::type, memory::format::type out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, float);
        arguments(neural::engine::type, memory::format::type out,                                                               primitive in,                              float);
        arguments(neural::engine::type, memory::format::type out,                                                               primitive in);
        arguments(neural::engine::type, primitive            out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, float slp);
        arguments(neural::engine::type, primitive            out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off);
        arguments(neural::engine::type, primitive            out,                                                               primitive in,                              float slp);
        arguments(neural::engine::type, primitive            out,                                                               primitive in);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    relu(arguments arg) : is_a_primitive(type_id<const relu>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

    std::unique_ptr<is_an_implementation> _private;
};



// pooling
struct pooling : is_a_primitive {
    class mode { mode(); public: enum type { max, average }; };

    struct arguments {
        engine::type                engine;
        pooling::mode::type         mode;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 1: input
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       stride;
        std::vector<uint32_t>       size;
        padding::type               padding;

        arguments(neural::engine::type, neural::pooling::mode::type, neural::memory::format::type o_frmt, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, std::vector<uint32_t> strd, std::vector<uint32_t> siz, neural::padding::type);
        arguments(neural::engine::type, neural::pooling::mode::type, neural::memory::format::type o_frmt,                                                               primitive in,                              std::vector<uint32_t> strd, std::vector<uint32_t> siz, neural::padding::type);
        arguments(neural::engine::type, neural::pooling::mode::type, neural::memory::format::type o_frmt,                                                               primitive in,                              uint32_t              strd, uint32_t              siz, neural::padding::type);
        arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,    std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, std::vector<uint32_t> strd, std::vector<uint32_t> siz, neural::padding::type);
        arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,                                                                  primitive in,                              std::vector<uint32_t> strd,                            neural::padding::type);
        arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,                                                                  primitive in,                              uint32_t              strd,                            neural::padding::type);
        arguments(neural::engine::type, neural::pooling::mode::type, primitive                    out,                                                                  primitive in,                              uint32_t              strd);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    pooling(arguments arg) : is_a_primitive(type_id<const pooling>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

    std::unique_ptr<is_an_implementation> _private;
};



namespace normalization { /////////////////////////////////////////////////////////////////////////////////////////////
// normalization of response
struct /*normalization*/response : is_a_primitive {
    struct arguments {
        engine::type                engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 1: input
        std::vector<int32_t>        input_offset;
        uint32_t                    size;
        padding::type               padding;
        float                       bias;
        float                       alpha;
        float                       beta;

        arguments(neural::engine::type, neural::memory::format::type, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, uint32_t, neural::padding::type, float, float, float);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                       uint32_t, neural::padding::type, float, float, float);
        arguments(neural::engine::type, primitive,                                                                  primitive,                       uint32_t, neural::padding::type, float, float, float);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    response(arguments arg) : is_a_primitive(type_id<const response>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
};



struct /*normalization*/softmax : is_a_primitive {
    struct arguments {
        engine::type                engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 1: input
        std::vector<int32_t>        input_offset;

        arguments(neural::engine::type, neural::memory::format::type, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive);
        arguments(neural::engine::type, primitive,                                                                  primitive);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    softmax(arguments arg) : is_a_primitive(type_id<const softmax>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
};

// batch normalization training - forward
struct /*normalization*/batch_training_forward : is_a_primitive {
    struct arguments {
        engine::type                engine;
        std::vector<primitive>      output;         // 3-5: {output, current_mean, current_inv_std_dev, [moving_mean, moving_inv_std_dev]}
        std::vector<primitive_at>   input;          // 3: {input, scale, bias}
        bool                        spatial;
        double                      exp_avg_factor;
        double                      epsilon;

        arguments(neural::engine::type, std::vector<primitive>, std::vector<primitive_at>, bool, double, double);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

private:
    batch_training_forward(arguments arg) : is_a_primitive(type_id<const batch_training_forward>()), argument(arg) {};
};

// batch normalization training - backward
struct /*normalization*/batch_training_backward : is_a_primitive {
    struct arguments {
        engine::type                engine;
        std::vector<primitive>      output;         // 3: {input_grad, scale_grad, bias_grad}
        std::vector<primitive_at>   input;          // 6: {forward_input, forward_scale, forward_bias, output_grad, current_mean, current_inv_std_dev}
        bool                        spatial;

        arguments(neural::engine::type, std::vector<primitive>, std::vector<primitive_at>, bool);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };

private:
    batch_training_backward(arguments arg) : is_a_primitive(type_id<const batch_training_backward>()), argument(arg) {};
};

// batch normalization inference
struct /*normalization*/batch_inference : is_a_primitive {
    struct arguments {
        engine::type                engine;
        std::vector<primitive>      output;         // 1: {output}
        std::vector<primitive_at>   input;          // 5: {input, scale, bias, precomputed_mean, precomputed_inv_std_dev}
        bool                        spatial;

        arguments(neural::engine::type, std::vector<primitive>, std::vector<primitive_at>, bool);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
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
        engine::type                engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 3: input, filter, bias
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       input_stride;
        neural::padding::type       padding;
        float                       negative_slope;

        arguments(neural::engine::type, neural::memory::format::type, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, std::vector<uint32_t>, primitive, primitive, neural::padding::type, float);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                       std::vector<uint32_t>, primitive, primitive, neural::padding::type, float);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                       uint32_t,              primitive, primitive, neural::padding::type, float);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                                              primitive, primitive, neural::padding::type, float);
        arguments(neural::engine::type, primitive,                                                                  primitive,                                              primitive, primitive, neural::padding::type, float);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    convolution_relu(arguments arg) : is_a_primitive(type_id<const convolution_relu>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
};



// fully connected + relu
struct fully_connected_relu : is_a_primitive {
    struct arguments {
        engine::type                engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 3: input, filter, bias
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       input_stride;
        float                       negative_slope;

        arguments(neural::engine::type, neural::memory::format::type, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, std::vector<uint32_t>, primitive, primitive, float);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                       std::vector<uint32_t>, primitive, primitive, float);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                       uint32_t,              primitive, primitive, float);
        arguments(neural::engine::type, neural::memory::format::type,                                               primitive,                                              primitive, primitive, float);
        arguments(neural::engine::type, primitive,                                                                  primitive,                                              primitive, primitive, float);
    };
    const arguments argument;

    struct query_entry : is_a_query_entry { arguments arguments; };
    static std::vector<query_entry> query(arguments);
    static primitive create(arguments);
    primitive clone() const { return create(argument); }
private:
    fully_connected_relu(arguments arg) : is_a_primitive(type_id<const fully_connected_relu>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  { return argument.input; };
    const std::vector<primitive>     &output() const { return argument.output; };
};

}
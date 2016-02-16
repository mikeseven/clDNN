#pragma once

#include "neural_base.h"

namespace neural {


// data in memory in known format; format consists of memory {order, type} of values
struct memory : is_a_primitive {
    enum class format : size_t { 
        xb_f32,     // 1D+batch, float32
        yxfb_f32,   // 3D+batch, float32
        any=static_cast<size_t>(-1) };

    struct arguments {
        engine              engine;
        format              format;
        std::vector<size_t> size;
    };
    const arguments argument;
    mutable void *pointer;

    static primitive create(arguments);
    memory &operator()(void *ptr) { pointer = ptr; return *this; };
    primitive clone() const { return create(argument); }
    void execute_argument(void *arg) const { pointer = arg; }
private:
    memory(arguments arg) : is_a_primitive(type_id<const memory>()), argument(arg), pointer(0) {};
    const std::vector<primitive_at>  &input()  const {throw std::runtime_error("No inputs in memory descritiption"); };
    const std::vector<primitive>     &output() const {throw std::runtime_error("No outputs in memory descritiption"); };
};



// [TODO] should it have querries ?
struct file : is_a_primitive {
    enum class format : size_t { nndata, any=static_cast<size_t>(-1) };

    struct arguments {
        engine      engine;
        std::string name;
        format      format;

        arguments(neural::engine aengine, std::string aname, file::format aformat)  : engine(aengine), name(aname), format(aformat) {};
        arguments(neural::engine aengine, std::string aname)                        : engine(aengine), name(aname), format(file::format::any) {};
    };
    const arguments argument;

    static primitive create(arguments);
    file &operator()(void *);
    primitive clone() const { return create(argument); }
private:
    file(arguments arg) : is_a_primitive(type_id<const file>()), argument(arg) {};
    const std::vector<primitive_at>  &input() const  {throw std::runtime_error("no inputs in file reader"); };
    const std::vector<primitive>     &output() const {throw std::runtime_error("no outputs in file reader"); };
};



// reorder data, type is not changed
struct reorder : is_a_primitive {
    struct arguments {
        engine                      engine;
        std::vector<primitive>      output;
        std::vector<primitive_at>   input;  // 1: input

        arguments(neural::engine, neural::memory::format, primitive_at);
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
        engine                      engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 3: input, filter, bias
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       input_stride;
        neural::padding             padding;

        arguments(neural::engine, neural::memory::format, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, std::vector<uint32_t>, primitive, primitive, neural::padding);
        arguments(neural::engine, neural::memory::format,                                               primitive,                       std::vector<uint32_t>, primitive, primitive, neural::padding);
        arguments(neural::engine, neural::memory::format,                                               primitive,                       uint32_t,              primitive, primitive, neural::padding);
        arguments(neural::engine, neural::memory::format,                                               primitive,                                              primitive, primitive, neural::padding);
        arguments(neural::engine, primitive,                                                            primitive,                                              primitive, primitive, neural::padding);
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
        engine                      engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 3: input, filter, bias
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       input_stride;

        arguments(neural::engine, neural::memory::format, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, std::vector<uint32_t>, primitive, primitive);
        arguments(neural::engine, neural::memory::format,                                               primitive,                       std::vector<uint32_t>, primitive, primitive);
        arguments(neural::engine, neural::memory::format,                                               primitive,                       uint32_t,              primitive, primitive);
        arguments(neural::engine, neural::memory::format,                                               primitive,                                              primitive, primitive);
        arguments(neural::engine, primitive,                                                            primitive,                                              primitive, primitive);
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
        engine                    engine;
        std::vector<primitive>    output;
        std::vector<uint32_t>     output_offset;
        std::vector<uint32_t>     output_size;
        std::vector<primitive_at> input;          // 1: input
        std::vector<int32_t>      input_offset;
        float                     negative_slope;

        arguments(neural::engine, memory::format out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, float);
        arguments(neural::engine, memory::format out,                                                               primitive in,                              float);
        arguments(neural::engine, memory::format out,                                                               primitive in);                     
        arguments(neural::engine, primitive      out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, float slp);
        arguments(neural::engine, primitive      out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off);
        arguments(neural::engine, primitive      out,                                                               primitive in,                              float slp);
        arguments(neural::engine, primitive      out,                                                               primitive in);
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
    enum class mode : size_t { max, average };

    struct arguments {
        engine                      engine;
        pooling::mode               mode;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 1: input
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       stride;
        std::vector<uint32_t>       size;
        padding                     padding;

        arguments(neural::engine, neural::pooling::mode, neural::memory::format, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, std::vector<uint32_t>, std::vector<uint32_t>, neural::padding);
        arguments(neural::engine, neural::pooling::mode, neural::memory::format,                                               primitive,                       std::vector<uint32_t>, std::vector<uint32_t>, neural::padding);
        arguments(neural::engine, neural::pooling::mode, neural::memory::format,                                               primitive,                       uint32_t,              uint32_t,              neural::padding);
        arguments(neural::engine, neural::pooling::mode, primitive,                                                            primitive,                       std::vector<uint32_t>,                        neural::padding);
        arguments(neural::engine, neural::pooling::mode, primitive,                                                            primitive,                       uint32_t,                                     neural::padding);
        arguments(neural::engine, neural::pooling::mode, primitive,                                                            primitive,                       uint32_t);
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
};



namespace normalization {
// normalization of response
struct /*normalization*/response : is_a_primitive {
    struct arguments {
        engine                      engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 1: input
        std::vector<int32_t>        input_offset;
        uint32_t                    size;
        padding                     padding;
        float                       bias;
        float                       alpha;
        float                       beta;

        arguments(neural::engine, neural::memory::format, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, uint32_t, neural::padding, float, float, float);
        arguments(neural::engine, neural::memory::format,                                               primitive,                       uint32_t, neural::padding, float, float, float);
        arguments(neural::engine, primitive,                                                            primitive,                       uint32_t, neural::padding, float, float, float);
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
        engine                      engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 1: input
        std::vector<int32_t>        input_offset;

        arguments(neural::engine, neural::memory::format, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>);
        arguments(neural::engine, neural::memory::format,                                               primitive);
        arguments(neural::engine, primitive,                                                            primitive);
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
};//normalization



// direct convolution+relu
struct convolution_relu : is_a_primitive {
    struct arguments {
        engine                      engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 3: input, filter, bias
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       input_stride;
        neural::padding             padding;
        float                       negative_slope;

        arguments(neural::engine, neural::memory::format, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, std::vector<uint32_t>, primitive, primitive, neural::padding, float);
        arguments(neural::engine, neural::memory::format,                                               primitive,                       std::vector<uint32_t>, primitive, primitive, neural::padding, float);
        arguments(neural::engine, neural::memory::format,                                               primitive,                       uint32_t,              primitive, primitive, neural::padding, float);
        arguments(neural::engine, neural::memory::format,                                               primitive,                                              primitive, primitive, neural::padding, float);
        arguments(neural::engine, primitive,                                                            primitive,                                              primitive, primitive, neural::padding, float);
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
        engine                      engine;
        std::vector<primitive>      output;
        std::vector<uint32_t>       output_offset;
        std::vector<uint32_t>       output_size;
        std::vector<primitive_at>   input;          // 3: input, filter, bias
        std::vector<int32_t>        input_offset;
        std::vector<uint32_t>       input_stride;
        float                       negative_slope;

        arguments(neural::engine, neural::memory::format, std::vector<uint32_t>, std::vector<uint32_t>, primitive, std::vector<int32_t>, std::vector<uint32_t>, primitive, primitive, float);
        arguments(neural::engine, neural::memory::format,                                               primitive,                       std::vector<uint32_t>, primitive, primitive, float);
        arguments(neural::engine, neural::memory::format,                                               primitive,                       uint32_t,              primitive, primitive, float);
        arguments(neural::engine, neural::memory::format,                                               primitive,                                              primitive, primitive, float);
        arguments(neural::engine, primitive,                                                            primitive,                                              primitive, primitive, float);
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
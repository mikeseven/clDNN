#include "api/neural.h"
#include "multidimensional_counter.h"

namespace neural {

struct convolution_reference : is_an_implementation {
    const convolution &outer;
    convolution_reference(convolution &arg)
        : is_an_implementation(neural::type_id<convolution_reference>())
        , outer(arg)
    {};
    ~convolution_reference() {}

    static void implementation(const void *ptr) {
        auto this_conv = static_cast<const convolution *>(ptr);
        auto input     = static_cast<float*>(this_conv->input_memory(0).pointer);
        auto output    = static_cast<float*>(this_conv->output_memory(0).pointer);
        auto window    = static_cast<float*>(this_conv->argument.weight.as<const memory&>().pointer);
        auto bias      = static_cast<float*>(this_conv->argument.bias.as<const memory&>().pointer);

        auto input_memory_arg  = this_conv->input_memory(0).argument;
        auto input_buffer_size = input_memory_arg.size;
        auto input_offset      = this_conv->argument.input_offset;

        auto output_memory_arg = this_conv->output_memory(0).argument;
        auto output_buffer_size= output_memory_arg.size;
        auto output_offset     = this_conv->argument.output_offset;
        auto output_size       = this_conv->argument.output_size;

        auto window_memory_arg = this_conv->argument.weight.as<const memory&>().argument; //convolution filter
        auto window_buffer_size= window_memory_arg.size;

        auto bias_memory_arg   = this_conv->argument.bias.as<const memory&>().argument;
        auto bias_buffer_size  = bias_memory_arg.size;

        auto stride  = this_conv->argument.stride;
        auto padding = this_conv->argument.padding;

        if(input_buffer_size.size()  != output_buffer_size.size()) throw std::runtime_error("Convolution input/output number of dimension does not match.");
        if(stride.size()             != output_buffer_size.size()) throw std::runtime_error("Convolution stride/output number of dimension does not match.");
        if(input_memory_arg.format   != memory::format::yxfb_f32)  throw std::runtime_error("Convolution reference uses yxfb_f32 format.");             // only yxfb_f32 format is supported
        if(input_memory_arg.format   != output_memory_arg.format)  throw std::runtime_error("Convolution input/output data format does not match.");    // only yxfb_f32 format is supported
        if(input_memory_arg.format   != window_memory_arg.format)  throw std::runtime_error("Convolution input/weights data format does not match.");   // only yxfb_f32 format is supported
        if(window_buffer_size.size() != output_buffer_size.size()) throw std::runtime_error("Convolution window_size/output number of dimension does not match.");
        if(bias_buffer_size.size()   != 1)                         throw std::runtime_error("Convolution biases isn't 1D vector.");
        if(bias_buffer_size[0]       != output_size[2])            throw std::runtime_error("Convolution biases/output feature maps number does not match."); // todo need type traits for index of 'z' dimension
                                                                                                                                                              // than this implementation will be format independent
        // general formula: output size = (input size - window size) / step + 1
        for(size_t i = 0; i < input_offset.size(); ++i){
            if(output_size[i] < (static_cast<int32_t>(input_buffer_size[i]) - input_offset[i]) / (stride[i] + 1) )
                throw std::runtime_error("Output size of convolution is to small.");

            if(output_buffer_size[i] < output_size[i] + output_offset[i])
                throw std::runtime_error("Convolution output buffer size is to small.");
        }

        namespace nd = ndimensional;
        nd::value<uint32_t> range (output_size);
        nd::value<uint32_t> window_range (window_buffer_size);
        nd::calculate_idx<uint32_t> calc_in_idx  (input_buffer_size);
        nd::calculate_idx<uint32_t> calc_out_idx (output_buffer_size);
        nd::calculate_idx<uint32_t> calc_win_idx (window_buffer_size);
        switch(padding){
            case padding::zero:
                for(auto pos : range) {
                    float acc = 0;
                    auto out_idx = calc_out_idx(pos + output_offset);

                    for(auto win_pos : window_range){
                        const std::vector<int32_t> arg_in_idx = nd::value<int32_t>(input_offset) + pos*stride + win_pos;

                        if( calc_in_idx.is_out_of_range(arg_in_idx) )
                            continue;

                        auto in_idx  = calc_in_idx (arg_in_idx);
                        auto win_idx = calc_win_idx(win_pos);
                        acc += input[in_idx] * window[win_idx];
                    }
                    output[out_idx] = acc + bias[ pos[2] ]; // todo need type traits for index of 'z' dimension
                }
                break;
            default:
                throw std::runtime_error("Unknown padding mode in convolution.");
        }
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(convolution &arg) { return new convolution_reference(arg); };
};

convolution::arguments::arguments( neural::engine::type  eng,
                                   primitive             out,
                                   std::vector<uint32_t> out_off,
                                   std::vector<uint32_t> out_siz,
                                   primitive             in,
                                   std::vector<int32_t>  in_off,
                                   std::vector<uint32_t> strd,
                                   primitive             weights,
                                   primitive             biases,
                                   neural::padding::type padd)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , weight(weights)
    , bias(biases)
    , padding(padd) {};

convolution::arguments::arguments( neural::engine::type  eng,
                                   primitive             out,
                                   primitive             in,
                                   std::vector<uint32_t> strd,
                                   primitive             weights,
                                   primitive             biases,
                                   neural::padding::type padd)
    : engine(eng)
    , output({out})
    , output_offset(out.as<const memory&>().argument.size.size())
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset(in.as<const memory&>().argument.size.size())
    , stride(strd)
    , weight(weights)
    , bias(biases)
    , padding(padd) {};

struct convolution_backward_reference : is_an_implementation {
    const convolution_backward &outer;
    convolution_backward_reference(convolution_backward &arg)
        : is_an_implementation(neural::type_id<convolution_backward_reference>())
        , outer(arg)
    {};
    ~convolution_backward_reference() {}

    static void implementation(const void *ptr) {
        auto this_bw_conv = static_cast<const convolution_backward *>(ptr);

        auto bw_input     = static_cast<float*>(this_bw_conv->input_memory(0).pointer);
        auto fw_input     = static_cast<float*>(this_bw_conv->input_memory(1).pointer);
        auto window       = static_cast<float*>(this_bw_conv->input_memory(2).pointer);
        auto bias         = static_cast<float*>(this_bw_conv->input_memory(3).pointer);

        auto bw_output    = static_cast<float*>(this_bw_conv->output_memory(0).pointer);
        auto window_diff  = static_cast<float*>(this_bw_conv->output_memory(1).pointer);

        auto bw_input_memory_arg  = this_bw_conv->input_memory(0).argument;
        auto bw_input_offset      = this_bw_conv->argument.input_offset;
        auto bw_input_size        = this_bw_conv->argument.input_size;  // todo output or input?

        auto fw_input_memory_arg  = this_bw_conv->input_memory(1).argument;
        auto fw_input_size        = bw_input_memory_arg.size;
        auto fw_input_offset      = this_bw_conv->argument.input_offset;

        auto window_memory_arg    = this_bw_conv->input_memory(2).argument;
        auto window_size          = bw_input_memory_arg.size;

        auto bias_memory_arg      = this_bw_conv->input_memory(3).argument;
        auto bias_size            = bw_input_memory_arg.size;

        auto bw_output_memory_arg = this_bw_conv->output_memory(0).argument;
        auto bw_output_size       = bw_output_memory_arg.size;
        auto bw_output_offset     = this_bw_conv->argument.output_offset;

        auto window_diff_memory_arg  = this_bw_conv->output_memory(1).argument;
        auto window_diff_size        = bw_input_memory_arg.size;

        auto stride  = this_bw_conv->argument.stride;
        auto padding = this_bw_conv->argument.padding;

        if(bw_input_size.size()       != bw_output_size.size())      throw std::runtime_error("Backward convolution bw_input/bw_output number of dimension does not match.");
        if(stride.size()              != bw_output_size.size())      throw std::runtime_error("Backward convolution stride/bw_output number of dimension does not match.");
        if(bw_input_memory_arg.format != memory::format::yxfb_f32)   throw std::runtime_error("Backward convolution reference uses yxfb_f32 format.");                // only yxfb_f32 format is supported
        if(bw_input_memory_arg.format != bw_output_memory_arg.format)throw std::runtime_error("Backward convolution bw_input/bw_output data format does not match."); // only yxfb_f32 format is supported
        if(bw_input_memory_arg.format != window_memory_arg.format)   throw std::runtime_error("Backward convolution bw_input/weights data format does not match.");   // only yxfb_f32 format is supported
        if(bw_input_size.size()       != fw_input_size.size())       throw std::runtime_error("Backward convolution bw_input/fw_output number of dimension does not match.");
        if(bw_input_memory_arg.format != fw_input_memory_arg.format) throw std::runtime_error("Backward convolution bw_input/fw_output data format does not match."); // only yxfb_f32 format is supported
        if(window_size.size()         != bw_output_size.size())      throw std::runtime_error("Backward convolution window_size/bw_output number of dimension does not match.");
        if(window_size.size()         != window_diff_size.size())    throw std::runtime_error("Backward convolution weights/weights_diff number of dimension does not match.");
        if(bias_size.size()           != 1)                          throw std::runtime_error("Backward convolution biases isn't 1D vector.");
        if(bias_size[0]               != bw_output_size[2])          throw std::runtime_error("Backward convolution biases/bw_output feature maps number does not match."); // todo need type traits for index of 'z' dimension

        // general formula: output size = (input size - window size) / step + 1
        for(size_t i = 0; i < fw_input_offset.size(); ++i){
            if(bw_input_size[i] < (static_cast<int32_t>(fw_input_size[i]) - bw_output_offset[i]) / (stride[i] + 1) )
                throw std::runtime_error("Input size of bw convolution is to small.");

            if(bw_input_size[i] < bw_output_size[i] + bw_output_offset[i])
                throw std::runtime_error("Backward convolution bw_input buffer size is to small.");
        }

        namespace nd = ndimensional;
        nd::value<uint32_t> range (bw_input_size); //todo in/out size?
        nd::value<uint32_t> window_range (window_size);
        nd::calculate_idx<uint32_t> calc_in_idx  (bw_input_size);
        nd::calculate_idx<uint32_t> calc_out_idx (bw_output_size);
        nd::calculate_idx<uint32_t> calc_win_idx (window_size);

        switch(padding){
            case padding::zero:
                for(auto pos : range) {
                    float acc = 0;
                    auto in_idx = calc_in_idx(pos + bw_input_offset);

                    for(auto win_pos : window_range){
                        const std::vector<uint32_t> arg_out_idx = nd::value<uint32_t>(bw_output_offset) + pos*stride + win_pos;

                        if( calc_in_idx.is_out_of_range(arg_out_idx) )
                            continue;

                        auto out_idx = calc_out_idx(arg_out_idx);
                        auto win_idx = calc_win_idx(win_pos);
                        bw_output[out_idx] += bw_input[in_idx] * window[win_idx];

                        // fw_input and bw_output is the same memory (place) in NN (here, represented by 2 separate buffers) so I can use
                        // index from one buffer to index second. fw_input stores nother data than bw_output, because of direction of workload computations
                        window_diff[win_idx] += bw_input[in_idx] * fw_input[out_idx];
                    }
                  //  output[out_idx] = acc + bias[ pos[2] ]; // todo need type traits for index of 'z' dimension
                }
                break;
            default:
                throw std::runtime_error("Unknown padding mode in backward convolution.");
        }

    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(convolution_backward &arg) { return new convolution_backward_reference(arg); };
};

convolution_backward::arguments::arguments( neural::engine::type   eng,
                                            std::vector<primitive> out,
                                            std::vector<uint32_t>  out_off,
                                            std::vector<uint32_t>  out_siz,
                                            std::vector<primitive> in,
                                            std::vector<int32_t>   in_off,
                                            std::vector<uint32_t>  strd,
                                            neural::padding::type  padd)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , input_size(out_siz)
    , input(in.cbegin(), in.cend())
    , input_offset(in_off)
    , stride(strd)
    , padding(padd) {};

convolution_backward::arguments::arguments( neural::engine::type   eng,
                                            std::vector<primitive> out,
                                            std::vector<primitive> in,
                                            std::vector<uint32_t>  strd,
                                            neural::padding::type  padd)
    : engine(eng)
    , output({out})
    , output_offset(out[0].as<const memory&>().argument.size.size())
    , input_size(out[0].as<const memory&>().argument.size.begin(), out[0].as<const memory&>().argument.size.end())
    , input(in.cbegin(), in.cend())
    , input_offset(in[0].as<const memory&>().argument.size.size())
    , stride(strd)
    , padding(padd) {};


//                                    engine          output                  input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;
// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(convolution &)>> forward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), convolution_reference::create},
};

static std::map<implementation_key, std::function<is_an_implementation *(convolution_backward &)>> backward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), convolution_backward_reference::create},
};
// creates primitive with convolution implementation that supports provided arguments
primitive convolution::create(convolution::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<convolution> result(new convolution(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = forward_implementation_map.find(key);
    if(it==std::end(forward_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}
primitive convolution_backward::create(convolution_backward::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<convolution_backward> result(new convolution_backward(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = backward_implementation_map.find(key);
    if(it==std::end(backward_implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}
}
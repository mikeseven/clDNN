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

        auto& input_offset  = this_conv->argument.input_offset;
        auto& output_offset = this_conv->argument.output_offset;
        auto& output_size   = this_conv->argument.output_size;
        auto& padding       = this_conv->argument.padding;
        auto& stride        = this_conv->argument.stride;

        auto& input_arg  = this_conv->input_memory(0).argument;
        auto& output_arg = this_conv->output_memory(0).argument;

        auto& filter_arg = this_conv->argument.weight.as<const memory&>().argument; //convolution filter
        auto& bias_arg   = this_conv->argument.bias.as<const memory&>().argument;

        int f_pos = 2; //todo need type traits

        if(input_arg.size.size()  != output_arg.size.size())   throw std::runtime_error("Convolution input/output number of dimension does not match.");
        if(stride.size()          != output_arg.size.size())   throw std::runtime_error("Convolution stride/output number of dimension does not match.");
        if(input_arg.format       != memory::format::yxfb_f32) throw std::runtime_error("Convolution reference uses yxfb_f32 format.");             // only yxfb_f32 format is supported
        if(input_arg.format       != output_arg.format)        throw std::runtime_error("Convolution input/output data format does not match.");    // only yxfb_f32 format is supported
        if(input_arg.format       != filter_arg.format)        throw std::runtime_error("Convolution input/weights data format does not match.");   // only yxfb_f32 format is supported
        if(filter_arg.size.size() != output_arg.size.size())   throw std::runtime_error("Convolution window_size/output number of dimension does not match.");
        if(bias_arg.size.size()   != 1)                        throw std::runtime_error("Convolution biases isn't 1D vector.");
        if(bias_arg.size[0]       != output_size[f_pos])       throw std::runtime_error("Convolution biases/output feature maps number does not match."); // todo need type traits for index of 'z' dimension
                                                                                                                                                              // than this implementation will be format independent
        auto input  = static_cast<float*>(this_conv->input_memory(0).pointer);
        auto output = static_cast<float*>(this_conv->output_memory(0).pointer);
        auto filter = static_cast<float*>(this_conv->argument.weight.as<const memory&>().pointer);
        auto bias   = static_cast<float*>(this_conv->argument.bias.as<const memory&>().pointer);

        for(size_t i = 0; i < input_offset.size(); ++i){
            // general formula: output size = (input size - filter size) / step + 1
            if(output_size[i] <
               std::abs(static_cast<int32_t>(input_arg.size[i] - input_offset[i] - filter_arg.size[i])) / stride[i] + 1) //todo is it safe?
                throw std::runtime_error("Output size of convolution is to small.");

            if(output_arg.size[i] < output_size[i] + output_offset[i])
                throw std::runtime_error("Convolution output buffer size is to small.");
        }

        namespace nd = ndimensional;
        nd::value<uint32_t> range (output_size);
        nd::value<uint32_t> window_range (filter_arg.size);
        nd::calculate_idx<uint32_t> calc_in_idx  (input_arg.size);
        nd::calculate_idx<uint32_t> calc_out_idx (output_arg.size);
        nd::calculate_idx<uint32_t> calc_win_idx (filter_arg.size);
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
                        acc += input[in_idx] * filter[win_idx];
                    }
                    output[out_idx] = acc + bias[ pos[f_pos] ]; // todo need type traits for index of 'f' dimension
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

    static void implementation(const void *ptr) { //todo tests
        auto this_bw_conv = static_cast<const convolution_backward *>(ptr);

        auto& bw_input_size    = this_bw_conv->argument.input_size;  // todo output or input?
        auto& bw_input_offset  = this_bw_conv->argument.input_offset;
        auto& bw_output_offset = this_bw_conv->argument.output_offset;
        auto& stride           = this_bw_conv->argument.stride;
        auto& padding          = this_bw_conv->argument.padding;

        auto& bw_input_arg     = this_bw_conv->input_memory(0).argument;
        auto& fw_input_arg     = this_bw_conv->input_memory(1).argument;
        auto& filter_arg       = this_bw_conv->input_memory(2).argument;
        auto& bias_arg         = this_bw_conv->input_memory(3).argument;

        auto& bw_output_arg    = this_bw_conv->output_memory(0).argument;
        auto& filter_diff_arg  = this_bw_conv->output_memory(1).argument;
        auto& bias_diff_arg    = this_bw_conv->output_memory(2).argument;

        if(bw_input_size.size()   != bw_output_arg.size.size())   throw std::runtime_error("Backward convolution bw_input/bw_output number of dimension does not match.");
        if(stride.size()          != bw_output_arg.size.size())   throw std::runtime_error("Backward convolution stride/bw_output number of dimension does not match.");
        if(bw_input_size.size()   != fw_input_arg.size.size())    throw std::runtime_error("Backward convolution bw_input/fw_output number of dimension does not match.");
        if(filter_arg.size.size() != bw_output_arg.size.size())   throw std::runtime_error("Backward convolution filter size/bw_output number of dimension does not match.");
        if(filter_arg.size.size() != filter_diff_arg.size.size()) throw std::runtime_error("Backward convolution weights/weights_diff number of dimension does not match.");
        if(bw_input_arg.format    != bw_output_arg.format)        throw std::runtime_error("Backward convolution bw_input/bw_output data format does not match.");
        if(bw_input_arg.format    != filter_arg.format)           throw std::runtime_error("Backward convolution bw_input/weights data format does not match.");
        if(bw_input_arg.format    != fw_input_arg.format)         throw std::runtime_error("Backward convolution bw_input/fw_output data format does not match.");
        if(bias_arg.size.size()   != 1)                           throw std::runtime_error("Backward convolution biases isn't 1D vector.");
        if(bias_arg.size.size()   != bias_diff_arg.size.size())   throw std::runtime_error("Backward convolution bias/bias_diff number dimensions doesn't match.");
        if(bias_arg.size[0]       != bw_output_arg.size[2])       throw std::runtime_error("Backward convolution biases/bw_output feature maps number does not match."); // todo need type traits for index of 'z' dimension
        if(bias_arg.size[0]       != bias_diff_arg.size[0])       throw std::runtime_error("Backward convolution bias/bias_diff size doesn't match.");

        auto bw_input     = static_cast<float*>(this_bw_conv->input_memory(0).pointer);
        auto fw_input     = static_cast<float*>(this_bw_conv->input_memory(1).pointer);
        auto weights      = static_cast<float*>(this_bw_conv->input_memory(2).pointer);
        auto bias         = static_cast<float*>(this_bw_conv->input_memory(3).pointer);

        auto bw_output    = static_cast<float*>(this_bw_conv->output_memory(0).pointer);
        auto weights_diff = static_cast<float*>(this_bw_conv->output_memory(1).pointer);
        auto bias_diff    = static_cast<float*>(this_bw_conv->output_memory(2).pointer);

        for(size_t i = 0; i < bw_output_offset.size(); ++i){
            // general formula for forward: output size = (input size - filter size) / step + 1
            if(bw_input_size[i] <
               std::abs(static_cast<int32_t>(bw_output_arg.size[i] - bw_output_offset[i] - filter_arg.size[i])) / stride[i] + 1) //todo is it safe?
                throw std::runtime_error("Output size of bw convolution is to small.");

            if(bw_input_arg.size[i] < bw_input_size[i] + bw_output_offset[i])
                throw std::runtime_error("Backward convolution bw_input buffer size is to small.");
        }

        // initializie gradients with 0
        this_bw_conv->output_memory(0).fill(0.0f);
        this_bw_conv->output_memory(1).fill(0.0f);
        this_bw_conv->output_memory(2).fill(0.0f);

        int f_pos = 2; //todo need type traits

        namespace nd = ndimensional;
        nd::value<uint32_t> bias_range (bias_arg.size);
        nd::value<uint32_t> range (bw_input_size); //todo in/out size?
        nd::value<uint32_t> window_range (filter_arg.size);
        nd::calculate_idx<uint32_t> calc_bias_idx(bias_arg.size);
        nd::calculate_idx<uint32_t> calc_in_idx  (bw_input_size);
        nd::calculate_idx<uint32_t> calc_out_idx (bw_output_arg.size);
        nd::calculate_idx<uint32_t> calc_win_idx (filter_arg.size);

        switch(padding){
            case padding::zero:
            {
                for(auto pos : range) {
                    auto in_idx = calc_in_idx(pos + bw_input_offset);

                    for(auto win_pos : window_range){
                        const std::vector<uint32_t> arg_out_idx = nd::value<uint32_t>(bw_output_offset) + pos*stride + win_pos;

                        if( calc_out_idx.is_out_of_range(arg_out_idx) )
                            continue;

                        auto out_idx = calc_out_idx(arg_out_idx);
                        auto win_idx = calc_win_idx(win_pos);
                        bw_output[out_idx] += bw_input[in_idx] * weights[win_idx];
                        weights_diff[win_idx] += fw_input[out_idx] * bw_output[out_idx];
                    }
                    bias_diff[ pos[f_pos] ] += bw_input[in_idx];
                }
                break;
            }
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
                                            std::vector<uint32_t>  in_siz,
                                            std::vector<primitive> in,
                                            std::vector<int32_t>   in_off,
                                            std::vector<uint32_t>  strd,
                                            neural::padding::type  padd)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , input_size(in_siz)
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
    , input_size(in[0].as<const memory&>().argument.size)
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
#include "api/neural.h"
#include <functional>

namespace neural {

struct convolution_jit : is_an_implementation {
    const convolution &outer;
    convolution_jit(convolution &arg)
        : is_an_implementation(neural::type_id<convolution_jit>())
        , outer(arg)
    {};
    ~convolution_jit() {}

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

        if(input_arg.size.size()  != output_arg.size.size())   throw std::runtime_error("Convolution input/output number of dimension does not match.");
        if(stride.size()          != output_arg.size.size())   throw std::runtime_error("Convolution stride/output number of dimension does not match.");
        if(input_arg.format       != memory::format::yxfb_f32) throw std::runtime_error("Convolution reference uses yxfb_f32 format.");             // only yxfb_f32 format is supported
        if(input_arg.format       != output_arg.format)        throw std::runtime_error("Convolution input/output data format does not match.");    // only yxfb_f32 format is supported
        if(input_arg.format       != filter_arg.format)        throw std::runtime_error("Convolution input/weights data format does not match.");   // only yxfb_f32 format is supported
        if(filter_arg.size.size() != output_arg.size.size())   throw std::runtime_error("Convolution window_size/output number of dimension does not match.");
        if(bias_arg.size.size()   != 1)                        throw std::runtime_error("Convolution biases isn't 1D vector.");
        if(bias_arg.size[0]       != output_size[2])           throw std::runtime_error("Convolution biases/output feature maps number does not match."); // todo need type traits for index of 'z' dimension
                                                                                                                                                              // than this implementation will be format independent
        auto input  = static_cast<float*>(this_conv->input_memory(0).pointer);
        auto output = static_cast<float*>(this_conv->output_memory(0).pointer);
        auto filter = static_cast<float*>(this_conv->argument.weight.as<const memory&>().pointer);
        auto bias   = static_cast<float*>(this_conv->argument.bias.as<const memory&>().pointer);

        // general formula: output size = (input size - filter size) / step + 1
        for(size_t i = 0; i < input_offset.size(); ++i){
            if(output_size[i] < (static_cast<int32_t>(input_arg.size[i]) - input_offset[i]) / (stride[i] + 1) )
                throw std::runtime_error("Output size of convolution is to small.");

            if(output_arg.size[i] < output_size[i] + output_offset[i])
                throw std::runtime_error("Convolution output buffer size is to small.");
        }

        switch(padding){
            case padding::zero:
                // todo conv jit
                break;
            default:
                throw std::runtime_error("Unknown padding mode in convolution.");
        }
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(convolution &arg) { return new convolution_jit(arg); };
};

//                                    engine          output                  input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;
// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(convolution &)>> forward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), convolution_jit::create},
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
}
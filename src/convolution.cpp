#include "api/neural.h"
#include "multidimensional_counter.h"
//#include <algorithm>
//#include <functional>
//#include <numeric>
//#include <map>
//#include <tuple>

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

        auto input_memory_arg  = this_conv->input_memory(0).argument;
        auto input_buffer_size = input_memory_arg.size;
        auto input_offset      = this_conv->argument.input_offset;

        auto output_memory_arg = this_conv->output_memory(0).argument;
        auto output_buffer_size= output_memory_arg.size;
        auto output_offset     = this_conv->argument.output_offset;
        auto output_size       = this_conv->argument.output_size;

        auto stride            = this_conv->argument.stride;
        auto window            = (this_conv->argument.weight).as<const memory*>();
        auto biases            = this_conv->argument.bias.as<const memory*>();
        auto padding           = this_conv->argument.padding; //todo, padding is not supported yet

        if(padding::zero                != padding)                   throw std::runtime_error("Padding is not supported.");
        if(input_buffer_size.size()     != output_buffer_size.size()) throw std::runtime_error("Convolution input/output number of dimension does not match.");
        if(stride.size()                != output_buffer_size.size()) throw std::runtime_error("Convolution stride/output number of dimension does not match.");
        if(window->argument.size.size() != output_buffer_size.size()) throw std::runtime_error("Convolution window_size/output number of dimension does not match.");
        if(input_memory_arg.format      != output_memory_arg.format)  throw std::runtime_error("Convolution input/output data format does not match.");
        if(input_memory_arg.format      != memory::format::yxfb_f32)  throw std::runtime_error("Convolution reference uses yxfb_f32 format.");
        if(biases->argument.size.size() != output_size[2])            throw std::runtime_error("Convolution biases/output feature maps number does not match.");

        // general formula: output size = (input size - window size) / step + 1
        //for(size_t i = 0; i < input_offset.size(); ++i){
        //    if(output_size[i] < (static_cast<int32_t>(input_buffer_size[i]) - input_offset[i]) / (stride[i] + 1) )
        //        throw std::runtime_error("Output size of convolution is to small.");

        //    if(output_buffer_size[i] < output_size[i] + output_offset[i])
        //        throw std::runtime_error("convolution output buffer size is to small.");
        //}

        namespace nd = ndimensional;
        nd::value<uint32_t> range (output_size);
        nd::calculate_idx<uint32_t> calc_in_idx  (input_buffer_size);
        nd::calculate_idx<uint32_t> calc_out_idx (output_buffer_size);
        //for(auto pos : range) {
        //    auto out_idx = calc_out_idx(pos + output_offset);

        //    nd::value<uint32_t> window_range (window);
        //    for(auto win_pos : window_range){
        //        auto in_idx = calc_in_idx(pos*stride + input_offset + win_pos);

        //        output[out_idx] = std::max(output[out_idx], input[in_idx]);
        //    }
        //}
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(convolution &arg) { return new convolution_reference(arg); };
};

//                                    engine          output                  input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(convolution &)>> implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), convolution_reference::create}
};

convolution::arguments::arguments(neural::engine::type   eng,
                                   primitive             out,
                                   std::vector<uint32_t> out_off,
                                   std::vector<uint32_t> out_siz,
                                   primitive             in,
                                   std::vector<int32_t>  in_off,
                                   std::vector<uint32_t> strd,
                                   primitive             weights,
                                   primitive             biases,
                                   padding::type         padd)
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

convolution::arguments::arguments(neural::engine::type   eng,
                                   primitive             out,
                                   primitive             in,
                                   std::vector<uint32_t> strd,
                                   primitive             weights,
                                   primitive             biases,
                                   padding::type         padd)
    : engine(eng)
    , output({out})
    , output_offset({out.as<const memory&>().argument.size.size()})
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset({in.as<const memory&>().argument.size.size()})
    , stride(strd)
    , weight(weights)
    , bias(biases)
    , padding(padd) {};

// creates primitive with convolution implementation that supports provided arguments
primitive convolution::create(convolution::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<convolution> result(new convolution(arg));

    // lookup in database; throw if not found
    auto key = std::make_tuple(arg.engine, result-> input_memory(0).argument.format, result->output_memory(0).argument.format);
    auto it = implementation_map.find(key);
    if(it==std::end(implementation_map)) throw std::runtime_error("not yet implemented");

    // create implementation & attach it to result
    auto implementation = it->second(*result);
    result->_private.reset(implementation);
    result->_work = implementation->work();

    // release RAII wrapper, return naked pointer
    return result.release();
}

}
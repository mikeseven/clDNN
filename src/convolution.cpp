#include "api/neural.h"
#include "multidimensional_counter.h"
//#include <algorithm>
//#include <functional>
//#include <numeric>
//#include <map>
//#include <tuple>

#include <fstream> //todo remove
template<typename T> //todo remove
void save_4d_data( std::vector<uint32_t> size, std::string filename, T* data, neural::memory::format::type type ) {
    std::vector<uint32_t> xyzn_pos(4);

    switch(type){
        case neural::memory::format::yxfb_f32 : xyzn_pos = {1, 0, 2, 3}; break;
        default                               : throw std::runtime_error("unknown data format");
    }

    std::ofstream file;
    file.open( filename + ".txt" );

    ndimensional::calculate_idx<uint32_t> calculate_idx( size );

    std::vector<uint32_t> current_pos(4);
    for( uint32_t batch = 0; batch < size[ xyzn_pos[3] ]; ++batch ){
        current_pos[ xyzn_pos[3] ] = batch;
        for( uint32_t z = 0; z < size[ xyzn_pos[2] ]; ++z ){
            current_pos[ xyzn_pos[2] ] = z;
            file << "n\\z: " << batch << "\\" << z << std::endl;
            for( uint32_t x = 0; x < size[ xyzn_pos[0] ]; ++x ) {
                current_pos[ xyzn_pos[0] ] = x;
                for( uint32_t y = 0; y < size[ xyzn_pos[1] ]; ++y ) {
                    current_pos[ xyzn_pos[1] ] = y;

                    file << data[ calculate_idx(current_pos) ] << "\t";
                }
                file << std::endl;
            }
        }
    }
    file.close();
}

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

        if(padding::zero             != padding)                   throw std::runtime_error("Padding is not supported.");
        if(input_buffer_size.size()  != output_buffer_size.size()) throw std::runtime_error("Convolution input/output number of dimension does not match.");
        if(stride.size()             != output_buffer_size.size()) throw std::runtime_error("Convolution stride/output number of dimension does not match.");
        if(input_memory_arg.format   != output_memory_arg.format)  throw std::runtime_error("Convolution input/output data format does not match.");
        if(window_buffer_size.size() != output_buffer_size.size()) throw std::runtime_error("Convolution window_size/output number of dimension does not match.");
        if(input_memory_arg.format   != memory::format::yxfb_f32)  throw std::runtime_error("Convolution reference uses yxfb_f32 format.");
        if(bias_buffer_size.size()   != 1)                         throw std::runtime_error("Convolution biases isn't 1D vector.");
        if(bias_buffer_size[0]       != output_size[2])            throw std::runtime_error("Convolution biases/output feature maps number does not match.");

        // general formula: output size = (input size - window size) / step + 1
        for(size_t i = 0; i < input_offset.size(); ++i){
            if(output_size[i] < (static_cast<int32_t>(input_buffer_size[i]) - input_offset[i]) / (stride[i] + 1) )
                throw std::runtime_error("Output size of convolution is to small.");

            if(output_buffer_size[i] < output_size[i] + output_offset[i])
                throw std::runtime_error("Convolution output buffer size is to small.");
        }

        save_4d_data(input_buffer_size, "in_b", input, input_memory_arg.format); //todo remove
        save_4d_data(output_buffer_size, "out_b", output, output_memory_arg.format); //todo remove
        save_4d_data(window_buffer_size, "window", window, window_memory_arg.format); //todo remove

        namespace nd = ndimensional;
        nd::value<uint32_t> range (output_size);
        nd::value<uint32_t> window_range (window_buffer_size);
        nd::calculate_idx<uint32_t> calc_in_idx  (input_buffer_size);
        nd::calculate_idx<uint32_t> calc_out_idx (output_buffer_size);
        nd::calculate_idx<uint32_t> calc_win_idx (window_buffer_size);
        for(auto pos : range) {
            float acc = 0;
            auto out_idx = calc_out_idx(pos + output_offset);

            for(auto win_pos : window_range){
                auto in_idx  = calc_in_idx (pos*stride + input_offset + win_pos);
                auto win_idx = calc_win_idx(win_pos);

                acc += input[in_idx] * window[win_idx] + bias[ pos[2] ];
            }
            output[out_idx] = acc;
        }
    save_4d_data(output_buffer_size, "out_a", output, output_memory_arg.format); //todo remove
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
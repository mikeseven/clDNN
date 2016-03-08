#include "api/neural.h"
#include "multidimensional_counter.h"
#include <climits>
#include <cmath>

namespace neural {
namespace normalization {

namespace {
struct softmax_reference : is_an_implementation {
    const softmax &outer;
    softmax_reference(softmax &arg)
        : is_an_implementation(neural::type_id<softmax_reference>())
        , outer(arg)
    {};
    ~softmax_reference() {}

    static void implementation(const void *ptr) {
        auto this_softmax = static_cast<const softmax *>(ptr);
        auto input        = static_cast<float*>(this_softmax->input_memory(0).pointer);
        auto output       = static_cast<float*>(this_softmax->output_memory(0).pointer);

        auto input_memory_arg  = this_softmax->input_memory(0).argument;
        auto input_buffer_size = input_memory_arg.size;
        auto input_offset      = this_softmax->argument.input_offset;

        auto output_memory_arg = this_softmax->output_memory(0).argument;
        auto output_buffer_size= output_memory_arg.size;
        auto output_offset     = this_softmax->argument.output_offset;
        auto output_size       = this_softmax->argument.output_size;

        if(input_memory_arg.format  != memory::format::xb_f32)    throw std::runtime_error("Softmax reference uses xb_f32 format."); // todo should be format independent
        if(input_memory_arg.format  != output_memory_arg.format)  throw std::runtime_error("Softmax input/output data format does not match.");
        if(input_buffer_size.size() != output_buffer_size.size()) throw std::runtime_error("Softmax input/output number of dimension does not match.");

        for(size_t i = 0; i < input_buffer_size.size(); ++i){
            if(input_buffer_size[i]  < output_size[i] + input_offset[i])  throw std::runtime_error("Softmax input/output size does not match.");
            if(output_buffer_size[i] < output_size[i] + output_offset[i]) throw std::runtime_error("Softmax sizes to small.");
        }

        int data_index = 0; //todo type traits
        int batch_index = 1;

        std::vector<float> v_max( output_size[batch_index], -std::numeric_limits<float>::max() );
        std::vector<float> v_acc( output_size[batch_index] );

        namespace nd = ndimensional;
        nd::value<uint32_t> range (output_size);
        nd::calculate_idx<uint32_t> calc_in_idx  (input_buffer_size);
        nd::calculate_idx<uint32_t> calc_out_idx (output_buffer_size);

        // find max val per batch
        for(auto pos : range) {
            auto in_idx  = calc_in_idx (pos + input_offset );
            v_max[ pos[batch_index] ] = std::max( v_max[pos[batch_index]], input[in_idx]);
        }
        for(auto pos : range) {
            auto in_idx  = calc_in_idx (pos + input_offset );
            auto out_idx = calc_out_idx(pos + output_offset);

            output[out_idx] = input[in_idx] - v_max[ pos[batch_index] ]; // subtracte max val from every data point per batch
            output[out_idx] = std::exp(output[out_idx]);  // exp
            v_acc[ pos[batch_index] ] += output[out_idx]; // sum eveything per batch
        }
        for(auto pos : range) {
            auto out_idx = calc_out_idx(pos + output_offset);
            output[out_idx] /= v_acc[ pos[batch_index] ]; // compute softmax
        }

    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(softmax &arg) { return new softmax_reference(arg); };
};

//                                    engine                output                        input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(softmax &)>> forward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::xb_f32, memory::format::xb_f32), softmax_reference::create}
};

} // namespace {

softmax::arguments::arguments( neural::engine::type eng, primitive out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off)
    : engine(eng)
    , output({out})
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off) {}

softmax::arguments::arguments( neural::engine::type eng, primitive out, primitive in )
    : engine(eng)
    , output({out})
    , output_offset(static_cast<uint32_t>(out.as<const memory&>().argument.size.size()))
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset(static_cast<uint32_t>(in.as<const memory&>().argument.size.size())) {}

softmax::arguments::arguments( neural::engine::type eng, memory::format::type out_fmt, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off)
    : engine(eng)
    , output({memory::create({eng, out_fmt, out_siz, true})})
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off) {}

// creates primitive with softmax implementation that supports provided arguments
primitive softmax::create(softmax::arguments arg) {
    // wrap softmax into RAII wrapper
    std::unique_ptr<softmax> result(new softmax(arg));

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

} // namespace normalization
} // namespace neural

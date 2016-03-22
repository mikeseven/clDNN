#include "api/neural.h"
#include "multidimensional_counter.h"
#include "convolution.h"

namespace neural {

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
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), convolution_cpu_reference::create},
    //{std::make_tuple(engine::cpu, memory::format::yxfb_f32, memory::format::yxfb_f32), convolution_cpu_jit::create} //todo singleton map, and add definitions in implementation files
};

static std::map<implementation_key, std::function<is_an_implementation *(convolution_backward &)>> backward_implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), convolution_backward_cpu_reference::create},
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
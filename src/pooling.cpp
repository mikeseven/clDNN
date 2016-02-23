#include "neural.h"
#include "multidimensional_counter.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <map>
#include <tuple>

namespace neural {

struct pooling_reference : is_an_implementation {
    const pooling &outer;
    pooling_reference(pooling &arg)
        : is_an_implementation(neural::type_id<pooling_reference>())
        , outer(arg)
    {};
    ~pooling_reference() {}

    static void implementation(const void *ptr) {
        auto this_pooling = static_cast<const pooling *>(ptr);
        auto input        = static_cast<float*>(this_pooling->input_memory(0).pointer);
        auto output       = static_cast<float*>(this_pooling->output_memory(0).pointer);

        auto input_memory_arg  = this_pooling->input_memory(0).argument;
        auto input_buffer_size = input_memory_arg.size;
        auto input_offset      = this_pooling->argument.input_offset;

        auto output_memory_arg = this_pooling->output_memory(0).argument;
        auto output_buffer_size= output_memory_arg.size;
        auto output_offset     = this_pooling->argument.output_offset;
        auto output_size       = this_pooling->argument.output_size;

        auto stride_size       = this_pooling->argument.stride;
        auto window_size       = this_pooling->argument.size;
        auto padding           = this_pooling->argument.padding; //todo

        if(input_memory_arg.format  != memory::format::yxfb_f32)  throw std::runtime_error("Pooling reference uses yxfb_f32 format."); //todo ?
        if(input_buffer_size.size() != output_buffer_size.size()) throw std::runtime_error("Pooling input/output number of dimension does not match.");
        if(stride_size.size()       != output_buffer_size.size()) throw std::runtime_error("Pooling stride/output number of dimension does not match.");
        if(window_size.size()       != output_buffer_size.size()) throw std::runtime_error("Pooling window_size/output number of dimension does not match.");
        if(input_memory_arg.format  != output_memory_arg.format)  throw std::runtime_error("Pooling input/output data format does not match.");
        //for(auto &x : input_offset)  if(x < 0)                    throw std::runtime_error("Pooling negative input offset."); //todo
        //for(auto &x : output_offset) if(x < 0)                    throw std::runtime_error("Pooling negative output offset."); //todo
        for(auto &x : stride_size)   if(x < 0)                    throw std::runtime_error("Pooling negative stride.");

        // general formula: output size = (input size - window size) / step + 1
        for(size_t i = 0; i < input_offset.size(); ++i){ //todo
            if(output_size[i] < (static_cast<int32_t>(input_buffer_size[i]) - input_offset[i]) / (stride_size[i] + 1) )
                throw std::runtime_error("Output size of pooling is to small.");

            if(output_buffer_size[i] < output_size[i] + output_offset[i])
                throw std::runtime_error("Pooling output buffer size is to small.");
        }

        multidimensional_counter<uint32_t> counter( output_size,
                                                    output_size.size(),
                                                    input_buffer_size,
                                                    {input_offset.begin(), input_offset.end()}, //todo will fail if negative
                                                    stride_size,
                                                    output_buffer_size,
                                                    output_offset
                                                   );



        if( pooling::mode::max == this_pooling->argument.mode ){
            //todo

            while( !counter.counter_finished() ){
                auto out_idx = counter.calculate_out_idx();

                std::vector<uint32_t> acc(input_offset.begin(), input_offset.end()); //todo will fail if negative
                std::transform(acc.begin(), acc.end(), counter.get_counter().begin(), acc.begin(), std::plus<uint32_t>());

                multidimensional_counter<uint32_t> window_counter( window_size,
                                                                   output_size.size(), //size of in/out-put, window, offsets etc. are equal
                                                                   input_buffer_size,
                                                                   acc,
                                                                   stride_size,
                                                                   output_buffer_size,
                                                                   output_offset
                                                                 );

                while( !window_counter.counter_finished() ){
                    auto in_idx = window_counter.calculate_in_idx();

                    output[out_idx] = std::max(output[out_idx], input[in_idx]);

                    window_counter.counter_increase();
                }
                counter.counter_increase();
            }

            //while( !counter_finished(output_size, general_counter) ){
            //    auto out_offset = calculate_offset(output_size, input_acc);

            //    while( !counter_finished(window_size, window_counter) ){
            //        auto in_offset  = calculate_offset(output_size, output_acc);
            //        output[out_offset] = std::max( input[in_offset], output[out_offset] );
            //        counter_increase(window_size, window_counter);
            //    }

            //counter_increase(output_size, general_counter);
            //}
        } else {// avg
            //todo
        }

/*

        auto uint_input_offset = std::vector<uint32_t>(input_offset.begin(), input_offset.end());  //pooling has always non negative offset

        std::vector<uint32_t> acc(uint_input_offset.size());
        while( !is_end() ){
            // calculate offset without most frequently changing dimension to reduce function calls
            // most changing dimension has linear layout in memory
            std::transform( counter.begin(), counter.end(), uint_input_offset.begin(), acc.begin(), std::plus<uint32_t>());
            auto in_offset  = calculate_offset(input_whole_size , acc ) + input_offset.back();

            std::transform( counter.begin(), counter.end(), output_offset.begin(), acc.begin(), std::plus<uint32_t>());
            auto out_offset = calculate_offset(output_whole_size, acc) + output_offset.back();

            // pooling on linear buffer
            for (uint32_t i = 0; i < output_size.back() ; ++i) {
                output[out_offset + i] = std::max( input[in_offset + i], 0.0f)
                                                 + this_pooling->argument.negative_slope * std::min( input[in_offset + i], 0.0f);
            }
            increse_counter();
        }*/
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(pooling &arg) { return new pooling_reference(arg); };
};

//                                    engine          output                  input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(pooling &)>> implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), pooling_reference::create}
};

pooling::arguments::arguments( neural::engine::type  eng,
                               pooling::mode::type   mode,
                               memory::format::type  o_frmt,
                               std::vector<uint32_t> out_off,
                               std::vector<uint32_t> out_siz,
                               primitive             in,
                               std::vector<int32_t>  in_off,
                               std::vector<uint32_t> strd,
                               std::vector<uint32_t> siz,
                               neural::padding::type padd)
    : engine(eng)
    , mode(mode)
    , output( {memory::create({eng, o_frmt, out_siz})} )
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(strd)
    , size(siz)
    , padding(padd) {};

// creates primitive with pooling implementation that supports provided arguments
primitive pooling::create(pooling::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<pooling> result(new pooling(arg));

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
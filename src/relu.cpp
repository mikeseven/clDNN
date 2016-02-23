#include "neural.h"
#include "multidimensional_counter.h"
#include <algorithm>
#include <tuple>
#include <map>
#include <functional>

#include <fstream>//todo remove
size_t calculate_idx(const std::vector<uint32_t> &size, const std::vector<uint32_t> &position){ //todo remove
    size_t offset = 0;

    for(size_t i = 0; i < position.size(); ++i)
        if(size[i] <= position[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );

    for(size_t i = 0; i != position.size(); ++i){    // number of iterations
        auto idx = position.size() - 1 - i;
        offset += std::accumulate(size.begin() + idx + 1, size.end(), 1, std::multiplies<uint32_t>() ) * position[idx];
    };

    return offset;
};
template<typename T> //todo remove
void save_4d_data_yxzb( std::vector<uint32_t> size, std::string filename, T* data ) {
    std::ofstream file;
    file.open( filename + ".txt" );

    for( uint32_t batch = 0; batch < size[3]; ++batch )
    for( uint32_t z = 0; z < size[2]; ++z ) {
        file << "n\\z: " << batch << "\\" << z << std::endl;
        for( uint32_t y = 0; y < size[1]; ++y ) {
            for( uint32_t x = 0; x < size[0]; ++x ) {
                file << *(data + calculate_idx(size, {y, x, z}) + batch) << "\t";
            }
            file << std::endl;
        }
    }
    file.close();
}

namespace neural {

namespace {

struct relu_reference : is_an_implementation {
    const relu &outer;
    relu_reference(relu &arg)
        : is_an_implementation(neural::type_id<relu_reference>()) 
        , outer(arg) 
    {};
    ~relu_reference() {}

    static void implementation(const void *ptr) {
        auto this_relu = static_cast<const relu *>(ptr);
        auto input     = static_cast<float*>(this_relu->input_memory(0).pointer);
        auto output    = static_cast<float*>(this_relu->output_memory(0).pointer);

        auto input_memory_arg  = this_relu->input_memory(0).argument;
        auto input_whole_size  = input_memory_arg.size;
        auto input_offset      = this_relu->argument.input_offset;

        auto output_memory_arg = this_relu->output_memory(0).argument;
        auto output_whole_size = output_memory_arg.size;
        auto output_offset     = this_relu->argument.output_offset;
        auto output_size       = this_relu->argument.output_size;

        if(input_memory_arg.format != memory::format::yxfb_f32) throw std::runtime_error("ReLU reference uses yxfb_f32 format.");
        if(input_whole_size.size() != output_whole_size.size()) throw std::runtime_error("ReLU input/output number of dimension does not match.");
        if(input_memory_arg.format != output_memory_arg.format) throw std::runtime_error("ReLU input/output data format does not match.");
        for(auto &x : input_offset)  if(x < 0)                  throw std::runtime_error("ReLU negative input offset.");

        save_4d_data_yxzb<float>(input_whole_size, "in_before", input); //todo remove
        save_4d_data_yxzb<float>(output_whole_size, "out_before", output); //todo remove

        for(size_t i = 0; i < input_whole_size.size(); ++i){
            if(input_whole_size[i]  < output_size[i] + input_offset[i] ) throw std::runtime_error("ReLU input/output size does not match.");
            if(output_whole_size[i] < output_size[i] + output_offset[i]) throw std::runtime_error("ReLU sizes to small.");
        }


        //std::vector<uint32_t> counter( output_size.size() - 1, 0 );

        auto uint_input_offset = std::vector<uint32_t>(input_offset.begin(), input_offset.end());  //relu has always non negative offset

        multidimensional_counter<uint32_t> counter( output_size,                                                   
                                                    output_size.size() - 1,
                                                    input_whole_size,
                                                    uint_input_offset,
                                                    output_whole_size,
                                                    output_offset );

        std::vector<uint32_t> acc(uint_input_offset.size());
        while( !counter.counter_finished() ){
            // calculate offset without most frequently changing dimension to reduce function calls
            // most changing dimension has linear layout in memory
            //std::transform( counter.begin(), counter.end(), uint_input_offset.begin(), acc.begin(), std::plus<uint32_t>());
            //auto in_offset  = calculate_idx(input_whole_size , acc ) + input_offset.back();

            //std::transform( counter.begin(), counter.end(), output_offset.begin(), acc.begin(), std::plus<uint32_t>());
            //auto out_offset = calculate_idx(output_whole_size, acc) + output_offset.back();

            //   std::transform( counter.begin(), counter.end(), output_offset.begin(), acc.begin(), std::plus<uint32_t>());
            auto in_offset  = counter.calculate_in_idx()  + input_offset.back();
            auto out_offset = counter.calculate_out_idx() + output_offset.back();

            // relu on linear buffer
            for (uint32_t i = 0; i < output_size.back() ; ++i) {
                output[out_offset + i] = std::max( input[in_offset + i], 0.0f) 
                                                 + this_relu->argument.negative_slope * std::min( input[in_offset + i], 0.0f);
            }
            counter.counter_increase();
        }
        save_4d_data_yxzb<float>(output_whole_size, "out_after", output); //todo remove
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(relu &arg) { return new relu_reference(arg); };
};

//                                    engine          output                  input
using implementation_key = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(relu &)>> implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), relu_reference::create}
};

} // namespace {
//todo discuss, output size is always needed or can be uninitialized?
relu::arguments::arguments( neural::engine::type engine, primitive out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, float slp)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}

relu::arguments::arguments( neural::engine::type engine, primitive out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope() {}

relu::arguments::arguments( neural::engine::type engine, primitive out, primitive in, float slp )
    : engine(engine)
    , output({out})
    , output_offset({out.as<const memory&>().argument.size.size()})
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset({in.as<const memory&>().argument.size.size()})
    , negative_slope(slp) {}

relu::arguments::arguments( neural::engine::type engine, primitive out, primitive in )
    : engine(engine)
    , output({out})
    , output_offset({out.as<const memory&>().argument.size.size()})
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset({in.as<const memory&>().argument.size.size()})
    , negative_slope(0.0f) {}




// creates primitive with relu implementation that supports provided arguments
primitive relu::create(relu::arguments arg) {
    // wrap relu into RAII wrapper
    std::unique_ptr<relu> result(new relu(arg));

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
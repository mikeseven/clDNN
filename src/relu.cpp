#include "neural.h"
#include <algorithm>
#include <numeric>
#include <tuple>
#include <map>
#include <functional>
#include <fstream>//todo remove
namespace neural {

namespace {

template<typename T> //todo remove
void save_4d_data_yxzb( std::vector<size_t> size, std::string filename, T* data ) {
    std::ofstream file;
    file.open( filename + ".txt" );

    for( size_t batch = 0; batch < size[3]; ++batch )
    for( size_t z = 0; z < size[2]; ++z ) {
        file << "n\\z: " << batch << "\\" << z << std::endl;
        for( size_t y = 0; y < size[1]; ++y ) {
            for( size_t x = 0; x < size[0]; ++x ) {
                file << *(data + calculate_offset(size, {y, x, z}) + batch) << "\t";
            }
            file << std::endl;
        }
    }
    file.close();
}

auto calculate_offset = [](const std::vector<size_t> &size, const std::vector<size_t> &position){    //todo normal function?
    size_t offset = 0;

    for(size_t i = 0; i != position.size(); ++i){    // number of iterations
        auto idx = position.size() - 1 - i;            // have to count starting with most frequently changing variable in counter
        offset += std::accumulate(size.begin() + idx + 1, size.end(), 1, std::multiplies<size_t>() ) * position[idx];
    };

    return offset;
}; 

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

        //todo validate data is float

        auto input_memory_arg  = this_relu->input_memory(0).argument;
        auto input_whole_size  = input_memory_arg.size;
        auto input_offset      = this_relu->argument.input_offset;

        auto output_memory_arg = this_relu->output_memory(0).argument;
        auto output_whole_size = output_memory_arg.size;
        auto output_offset     = this_relu->argument.output_offset;
        auto output_size       = this_relu->argument.output_size;

        if(input_whole_size.size() != output_whole_size.size()) throw std::runtime_error("ReLU input/output number of dimension does not match.");
        if(input_memory_arg.format != output_memory_arg.format) throw std::runtime_error("ReLU input/output data format does not match.");
        for(auto &x : input_offset)  if(x < 0)                  throw std::runtime_error("ReLU negative input offset.");
        for(auto &x : output_offset) if(x < 0)                  throw std::runtime_error("ReLU negative output offset.");

        save_4d_data_yxzb<float>(input_whole_size, "in_before", input); //todo remove
        save_4d_data_yxzb<float>(output_whole_size, "out_before", output); //todo remove

        for(size_t i = 0; i < input_whole_size.size(); ++i){
            if(input_whole_size[i]  < output_size[i] + input_offset[i] ) throw std::runtime_error("ReLU input/output size does not match.");
            if(output_whole_size[i] < output_size[i] + output_offset[i]) throw std::runtime_error("ReLU sizes to small.");
        }
//        std::vector<uint32_t> counter( output_size.size() - 1, 0 ); // last position indicates linear memory layout, it's not needed in counter
        std::vector<uint32_t> counter( output_size.size(), 0 ); // last position indicates linear memory layout, it's not needed in counter

        auto is_end = [&output_size, &counter](){
            for(auto it1  = counter.begin(), it2 = output_size.begin(); it1 != counter.end(); ++it1, ++it2)
                if(*it1 != *it2) return false;

            return true;
        };

        auto increse_counter = [&counter, &output_size](){
            // Counter is vector representing number in which each digit counter(i) maximum value is limited by output_size(i)
            // when counter(i)==output_size(i) it overflows with carry to the left
            // The least significant digit is on the last(max) position of the vector
            ++counter.back();

            for(auto i = counter.size() - 1; i > 0; --i)
                if( counter[i] == output_size[i] ){
                    counter[i] = 0;
                    ++counter[i-1];
                }

            // After all counter(i) equal output_size(i) counter is zeroed through overflow
            // thus after this case we write output_size to counter
            if( counter[0] == output_size[0] )
                for(auto i = counter.size() - 1; i > 0; --i)
                    counter[i] = output_size[i];
        };

        auto uint_input_offset = std::vector<size_t>(input_offset.begin(), input_offset.end());  //relu has always non negative offset

        std::vector<size_t>   acc(uint_input_offset.size());
        while( !is_end() ){
            // relu on linear buffer

            for (uint32_t i = 0; i < output_size.back() ; ++i) {  //todo offsets can be calculated without inner most dimension
                std::transform( counter.begin(), counter.end(), uint_input_offset.begin(), acc.begin(), std::plus<size_t>());
                auto in_offset  = calculate_offset(input_whole_size , acc );
            
                std::transform( counter.begin(), counter.end(), output_offset.begin(), acc.begin(), std::plus<size_t>());
                auto out_offset = calculate_offset(output_whole_size, acc);
                // calculate idx
             //   output[out_offset + i] = std::max( input[in_offset + i], 0.0f) 
            //                                     + this_relu->argument.negative_slope * std::min( input[in_offset + i], 0.0f);
                output[out_offset] = std::max( input[in_offset], 0.0f) 
                                                + this_relu->argument.negative_slope * std::min( input[in_offset], 0.0f);
            }
            increse_counter();
        }
             save_4d_data_yxzb<float>(output_whole_size, "out_after", output); //todo remove
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }

    static is_an_implementation *create(relu &arg) { return new relu_reference(arg); };
};

//                                    engine          output                  input
using implementation_key = std::tuple<neural::engine, neural::memory::format, neural::memory::format>;

// map of available implementations
static std::map<implementation_key, std::function<is_an_implementation *(relu &)>> implementation_map = {
    {std::make_tuple(engine::reference, memory::format::yxfb_f32, memory::format::yxfb_f32), relu_reference::create}
};

} // namespace {
//todo discuss, output size is always needed or can be uninitialized?
relu::arguments::arguments( neural::engine engine, primitive out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, float slp)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}
                                                                                      
relu::arguments::arguments( neural::engine engine, primitive out, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope() {}
                                                                                      
relu::arguments::arguments( neural::engine engine, primitive out, primitive in, float slp )
    : engine(engine)
    , output({out})
    , output_offset({out.as<const memory&>().argument.size.size()})
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset({in.as<const memory&>().argument.size.size()})
    , negative_slope(slp) {}

relu::arguments::arguments( neural::engine engine, primitive out, primitive in )
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
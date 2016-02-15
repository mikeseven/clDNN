#include "neural.h"
#include <algorithm>
#include <tuple>
#include <map>
#include <functional>

namespace neural {

namespace {
auto calculate_offset = [](const std::vector<size_t> &size, const std::vector<size_t> &position){    //todo normal function?
    auto calucalet_offset_by_variable = [&size, &position](size_t idx){    //todo change name?
        size_t offset = 1;

        for(size_t i = size.size() - 1; i > idx; --i) 
            offset *= size[i];

        offset *= position[idx];

        return offset;
    };

    size_t offset = 0;

    for(unsigned i = 0; i != position.size(); ++i){    // number of iterations
        auto idx = position.size() - 1 - i;            // have to count starting with most frequently changing variable in counter
        offset += calucalet_offset_by_variable(idx);
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

        auto input_memory_arg  = this_relu->input_memory(0).argument;
        auto input_whole_size  = input_memory_arg.size;
        auto input_offset      = this_relu->argument.input_offset;

        auto output_memory_arg = this_relu->output_memory(0).argument;
        auto output_whole_size = output_memory_arg.size;
        auto output_offset     = this_relu->argument.output_offset;
        auto output_size       = this_relu->argument.output_size;

        if(input_whole_size.size() != output_whole_size.size()) throw std::runtime_error("ReLU input/output number of dimension does not match.");
        if(input_memory_arg.format != output_memory_arg.format) throw std::runtime_error("ReLU input/output data format does not match.");

        for(unsigned i = 0; i < input_whole_size.size(); ++i){
            if(input_whole_size[i]  < output_size[i] + input_offset[i] ) throw std::runtime_error("ReLU input/output size does not match.");
            if(output_whole_size[i] < output_size[i] + output_offset[i]) throw std::runtime_error("ReLU sizes to small.");
        }    
        std::vector<size_t> counter( output_size.size(), 0 ); // last position indicates linear memory layout

        auto is_end = [&output_size, &counter](){
            for(auto it1  = counter.begin(), it2 = output_size.begin(); it1 != counter.end(); ++it1, ++it2)
                if(*it1 != *it2) return false;
            
            return true;
        };
        auto increse_counter = [&counter, &output_size](){
            ++counter.back();

            for(auto i = counter.size() - 1; i > 0; --i)
                if( counter[i] == output_size[i] ){
                    counter[i] = 0; 
                    ++counter[i-1];
                }

            if( counter[0] == output_size[0] )
                for(auto i = counter.size() - 1; i > 0; --i)
                    counter[i] = output_size[i];
        };

        auto uint_input_offset = std::vector<size_t>(input_offset.begin(), input_offset.end());  //todo relu has always non negative offset?

        std::vector<size_t> acc(uint_input_offset.size());
        while( !is_end() ){
            // relu on linear buffer
            for (size_t i = 0; i < output_size.back() ; ++i) {  //todo offsets can be calculated without inner most dimension
                // calculate idx
                std::transform( uint_input_offset.begin(), uint_input_offset.end(), counter.begin(), acc.begin(), std::plus<size_t>());
                auto in_offset  = calculate_offset(input_whole_size , acc );

                std::transform( output_offset.begin(), output_offset.end(), counter.begin(), acc.begin(), std::plus<size_t>());
                auto out_offset = calculate_offset(output_whole_size, acc);

                *(output + out_offset) = std::max( *(input + in_offset), 0.0f) 
                                                + this_relu->argument.negative_slope * std::min( *(input + in_offset), 0.0f);

                increse_counter();
            }
        }
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
relu::arguments::arguments( neural::engine engine, primitive out, std::vector<size_t>out_off, std::vector<size_t>out_siz, primitive in, std::vector<int32_t>in_off, float slp)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}

relu::arguments::arguments( neural::engine engine, primitive out, std::vector<size_t>out_off, std::vector<size_t>out_siz, primitive in, std::vector<int32_t>in_off)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope() {}

relu::arguments::arguments( neural::engine engine, primitive out, std::vector<size_t>out_off, primitive in, std::vector<int32_t>in_off, float slp )
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size(out.as<const memory&>().argument.size.size() )
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {

     std::transform( out.as<const memory&>().argument.size.begin(), 
                     out.as<const memory&>().argument.size.end(),
                     out_off.begin(),
                     output_size.begin(),
                     std::plus<size_t>() );
}

relu::arguments::arguments( neural::engine engine, primitive out, std::vector<size_t>out_off, primitive in, std::vector<int32_t>in_off )
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size(out.as<const memory&>().argument.size.size())
    , input({in})
    , input_offset({in_off})
    , negative_slope() {

    std::transform( out.as<const memory&>().argument.size.begin(), 
                    out.as<const memory&>().argument.size.end(),
                    out_off.begin(),
                    output_size.begin(),
                    std::plus<size_t>() );
}

relu::arguments::arguments( neural::engine engine, primitive out, primitive in, float slp )
    : engine(engine)
    , output({out})
    , output_offset({out.as<const memory&>().argument.size.size()})
    , output_size(out.as<const memory&>().argument.size)
    , input({in})
    , input_offset({in.as<const memory&>().argument.size.size()})
    , negative_slope(slp) {}

relu::arguments::arguments( neural::engine engine, primitive out, primitive in )
    : engine(engine)
    , output({out})
    , output_offset({out.as<const memory&>().argument.size.size()})
    , output_size(out.as<const memory&>().argument.size)
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
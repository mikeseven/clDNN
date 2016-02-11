#include "neural.h"
#include <algorithm>

namespace neural {

namespace {

struct relu_reference : is_a_unknown {
    const relu &outer;
    relu_reference(relu &arg)
        : is_a_unknown(neural::type_id<relu_reference>()) 
        , outer(arg) 
    {};
    ~relu_reference() {}

    static void implementation(const void *ptr) {
        auto this_relu = static_cast<const relu *>(ptr);
        auto input     = static_cast<float*>(this_relu->input_memory(0).pointer);
        auto output    = static_cast<float*>(this_relu->output_memory(0).pointer);

        auto input_memory_arg  = this_relu->input_memory(0).argument;
        auto output_memory_arg = this_relu->output_memory(0).argument;
        auto input_offset      = this_relu->argument.input_offset;
        auto output_offset     = this_relu->argument.output_offset;
        auto output_size       = this_relu->argument.output_size;

        if(input_memory_arg.size.size() != output_memory_arg.size.size()) throw std::runtime_error("ReLU input/output number of dimension does not match.");
        if(input_memory_arg.format      != output_memory_arg.format)      throw std::runtime_error("ReLU input/output data format does not match.");

#define TEST
#ifdef TEST
        for(unsigned i = 0; i < input_memory_arg.size.size(); ++i)
            if(input_memory_arg.size[i] - input_offset[i] != output_size[i]) throw std::runtime_error("ReLU input/output size does not match.");
      
        auto calculate_offset = [](const std::vector<uint32_t> &size, const std::vector<uint32_t> &counter){
            auto calucalet_offset_by_variable = [&size, &counter](size_t idx){    //todo change name?
                size_t offset = 1;
                
                for(size_t i = size.size() - 1; i > idx; --i) 
                    offset *= size[i];
                
                offset *= counter[idx];

                return offset;
            };

            size_t offset = 0;            

            for(unsigned i = 0; i != counter.size(); ++i){    // number of iterations
                auto idx = counter.size() - 1 - i;            // have to count starting with most frequently changing variable in counter (not size)
                offset += calucalet_offset_by_variable(idx);
            };

            return offset;
        };   
        
        std::vector<uint32_t> counter( output_size.size() - 1, 0 ); // last position indicates linear memory layout, it's not needed in counter
       
        auto is_end = [&output_size, &counter](){ //do not compare last digit, counter is N-1 long, while size is N long
            for(auto it1  = counter.begin(), it2 = output_size.begin(); it1 != counter.end(); ++it1, ++it2)
                if(*it1 >= *it2) return false;
            
            return true;
        };
        auto increse_counter = [&counter, &output_size](){
            ++counter.back();

            for(auto i = counter.size() - 1; i > 0; --i)
                if( counter[i] == output_size[i] ){
                    counter[i] = 0; 
                    ++counter[i-1];
                }

        };

        while( !is_end() ){
            // todo n dimensional relu
    
            // calculate idx
            auto offset = calculate_offset(output_size, counter);
            // relu on linear buffer
            for (size_t i = 0; i < output_size.back() ; ++i)
                *(output + offset + i) = std::max( *(input + offset + i), 0.0f) 
                                         + this_relu->argument.negative_slope * std::min( *(input + offset + i), 0.0f);
                //output[i] = std::max(input[i], 0.0f) + this_relu->argument.negative_slope * std::min(input[i], 0.0f);
        
            increse_counter();
        }
#else
        auto input_vec  =  this_relu->input_memory(0).argument.size;
        auto output_vec =  this_relu->output_memory(0).argument.size;

        size_t count_src = 1;
        size_t count_dst = 1;
        for(auto x : input_vec ) count_src *= x;
        for(auto x : output_vec) count_dst *= x;

        if( count_dst != count_src )
            throw std::runtime_error("ReLU input/output size does not match.");

        for (size_t i = 0; i < count_src; ++i)
          output[i] = std::max(input[i], 0.0f) + this_relu->argument.negative_slope * std::min(input[i], 0.0f);
#endif
    }

    std::vector<task> work() {
        return {task{implementation, &outer}};
    }
};

} // namespace {
//todo discuss, output size is always needed or can be uninitialized?
relu::arguments::arguments(neural::engine engine, primitive out, std::vector<uint32_t>out_off, std::vector<uint32_t>out_siz, primitive in, std::vector<int32_t>in_off, float slp)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}

relu::arguments::arguments(neural::engine engine, primitive out, std::vector<uint32_t>out_off, std::vector<uint32_t>out_siz, primitive in, std::vector<int32_t>in_off)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope() {}

relu::arguments::arguments( neural::engine engine, primitive out, std::vector<uint32_t>out_off, primitive in, std::vector<int32_t>in_off, float slp )
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}// todo output_size initialization

relu::arguments::arguments( neural::engine engine, primitive out, std::vector<uint32_t>out_off, primitive in, std::vector<int32_t>in_off )
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , input({in})
    , input_offset({in_off})
    , negative_slope() {} // todo output_size initialization

relu::arguments::arguments( neural::engine arg_engine, primitive arg_output, primitive arg_input, float arg_neg_slope )
    : engine(arg_engine)
    , output({arg_output})
    , input({arg_input})
    , negative_slope(arg_neg_slope) {} //todo offsets zero initialization, output_size initialization

relu::arguments::arguments( neural::engine arg_engine, primitive arg_output, primitive arg_input )
    : engine(arg_engine)
    , output({arg_output})
    , input({arg_input})
    , negative_slope(0.0f) {} //todo offsets zero initialization, output_size initialization

primitive relu::create(relu::arguments arg) {
    relu *result = new relu(arg);
    if(    arg.engine==engine::reference
        && memory::format::yxfb_f32==result-> input_memory(0).argument.format
        && memory::format::yxfb_f32==result->output_memory(0).argument.format)
    {
        auto implementation = new relu_reference(*result);
        result->_private.reset(implementation);
        result->_work = implementation->work();
    }
    arg.engine;

    return result;
}

}
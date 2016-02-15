#include "neural.h"
#include <algorithm>

#include<iostream>
#include<fstream>
static unsigned tmp_counter=0; //todo remove

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
        auto idx = position.size() - 1 - i;            // have to count starting with most frequently changing variable in counter (not size)
        offset += calucalet_offset_by_variable(idx);
    };

    return offset;
}; 

template<typename T>
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
        auto input_whole_size  = input_memory_arg.size;
        auto input_offset      = this_relu->argument.input_offset;

        auto output_memory_arg = this_relu->output_memory(0).argument;
        auto output_whole_size = output_memory_arg.size;
        auto output_offset     = this_relu->argument.output_offset;
        auto output_size       = this_relu->argument.output_size;

        if(input_whole_size.size() != output_whole_size.size()) throw std::runtime_error("ReLU input/output number of dimension does not match.");
        if(input_memory_arg.format != output_memory_arg.format) throw std::runtime_error("ReLU input/output data format does not match.");

        save_4d_data_yxzb<float>(input_whole_size, "in_before", input); //todo remove
        save_4d_data_yxzb<float>(output_whole_size, "out_before", output); //todo remove

#define TEST  //todo remove
#ifdef TEST
        for(unsigned i = 0; i < input_whole_size.size(); ++i){
            //if(input_whole_size[i]  - input_offset[i]  > output_size[i]) throw std::runtime_error("ReLU input/output size does not match."); //todo addition instead of multiplication?
            //if(output_whole_size[i] - output_offset[i] < output_size[i]) throw std::runtime_error("ReLU sizes to small."); //todo output_size indicates blob to copy, whole_size shouldnt be the same
            if(input_whole_size[i]  < output_size[i] + input_offset[i] ) throw std::runtime_error("ReLU input/output size does not match."); //todo addition instead of multiplication?
            if(output_whole_size[i] < output_size[i] + output_offset[i]) throw std::runtime_error("ReLU sizes to small."); //todo output_size indicates blob to copy, whole_size shouldnt be the same
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
            // todo n dimensional relu
            

            // relu on linear buffer
            for (size_t i = 0; i < output_size.back() ; ++i,             tmp_counter++) { //todo remove)
                // calculate idx
                std::transform( uint_input_offset.begin(), uint_input_offset.end(), counter.begin(), acc.begin(), std::plus<size_t>());
                auto in_offset  = calculate_offset(input_whole_size , acc );
                auto next_in  = *(input + in_offset);

                std::transform( output_offset.begin(), output_offset.end(), counter.begin(), acc.begin(), std::plus<size_t>());
                auto out_offset = calculate_offset(output_whole_size, acc);
                auto next_out = *(output + out_offset);

                *(output + out_offset) = std::max( *(input + in_offset), 0.0f) 
                                                + this_relu->argument.negative_slope * std::min( *(input + in_offset), 0.0f);

                //output[i] = std::max(input[i], 0.0f) + this_relu->argument.negative_slope * std::min(input[i], 0.0f);
        
                increse_counter();
            }
        }

#else //todo remove
        auto input_vec  =  this_relu->input_memory(0).argument.size;
        auto output_vec =  this_relu->output_memory(0).argument.size;

        size_t count_src = 1;
        size_t count_dst = 1;
        for(auto x : input_vec ) count_src *= x;
        for(auto x : output_vec) count_dst *= x;

        if( count_dst != count_src )
            throw std::runtime_error("ReLU input/output size does not match.");

        for (size_t i = 0; i < count_src; ++i           , tmp_counter++) //todo remove)
          output[i] = std::max(input[i], 0.0f) + this_relu->argument.negative_slope * std::min(input[i], 0.0f);
#endif
     save_4d_data_yxzb<float>(output_whole_size, "out_after", output); //todo remove
     if(1)
        int x = 1;
    }


    std::vector<task> work() {
        return {task{implementation, &outer}};
    }
};

} // namespace {
//todo discuss, output size is always needed or can be uninitialized?
relu::arguments::arguments(neural::engine engine, primitive out, std::vector<size_t>out_off, std::vector<size_t>out_siz, primitive in, std::vector<int32_t>in_off, float slp)
    : engine(engine)
    , output({out})
    , output_offset({out_off})
    , output_size({out_siz})
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}

relu::arguments::arguments(neural::engine engine, primitive out, std::vector<size_t>out_off, std::vector<size_t>out_siz, primitive in, std::vector<int32_t>in_off)
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
    , input({in})
    , input_offset({in_off})
    , negative_slope(slp) {}// todo output_size initialization

relu::arguments::arguments( neural::engine engine, primitive out, std::vector<size_t>out_off, primitive in, std::vector<int32_t>in_off )
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
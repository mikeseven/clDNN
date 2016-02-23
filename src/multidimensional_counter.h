#pragma once
#include <vector>
#include <numeric>
#include <string>

    //size_t calculate_idx(const std::vector<int32_t>  &size, const std::vector<int32_t>  &position);
    //bool counter_finished(const std::vector<uint32_t> &size, const std::vector<uint32_t> &counter);
    //bool counter_finished(const std::vector<int32_t>  &size, const std::vector<int32_t>  &counter);
    //void counter_increase(const std::vector<uint32_t> &size, std::vector<uint32_t> &counter);
    //void counter_increase(const std::vector<int32_t>  &size, std::vector<int32_t>  &counter);
    //void counter_increase(const std::vector<int32_t>  &size, std::vector<int32_t>  &counter, std::vector<int32_t> &val);
namespace neural{

template<typename T>
class multidimensional_counter {
    std::vector<T> size;            //todo consts
    std::vector<T> counter;

    std::vector<T> in_buffer_size;
    std::vector<T> in_offset;
    std::vector<T> in_position;
    std::vector<T> out_buffer_size;
    std::vector<T> out_offset;
    std::vector<T> out_position;

public:
    multidimensional_counter(const std::vector<T> &size,
                             uint32_t             counter_length,
                             const std::vector<T> &in_buf_size,
                             const std::vector<T> &in_offset,
                             const std::vector<T> &out_buf_size,
                             const std::vector<T> &out_offset);

    multidimensional_counter(const std::vector<T> &size,
                             uint32_t             counter_length,
                             const std::vector<T> &in_buf_size,
                             const std::vector<T> &out_buf_size);

    size_t calculate_in_idx(const std::vector<T> &position);
    size_t calculate_in_idx();
    size_t calculate_out_idx(const std::vector<T> &position);
    size_t calculate_out_idx();
    bool counter_finished();
    void counter_increase();

    std::vector<T> get_size()        { return size;        };      //todo consts  
    std::vector<T> get_counter()     { return counter;     };
    std::vector<T> get_in_position() { return  in_position;};
    std::vector<T> get_out_position(){ return out_position;};
};

template<typename T>
inline multidimensional_counter<T>::multidimensional_counter( const std::vector<T> &v_size,
                                                              uint32_t             counter_length,
                                                              const std::vector<T> &v_in_buf_size,
                                                              const std::vector<T> &v_in_offset,
                                                              const std::vector<T> &v_out_buf_size,
                                                              const std::vector<T> &v_out_offset)
    : size({v_size})
    , counter(counter_length, 0)
    , in_buffer_size({v_in_buf_size})
    , in_offset({v_in_offset})
    , in_position({v_in_offset})
    , out_buffer_size({v_out_buf_size})
    , out_offset({v_out_offset})
    , out_position({out_offset}) 
    {}

template<typename T>
inline multidimensional_counter<T>::multidimensional_counter( const std::vector<T> &v_size,
                                                              uint32_t             counter_length,
                                                              const std::vector<T> &v_in_buf_size,
                                                              const std::vector<T> &v_out_buf_size)
    : size({v_size})
    , counter(counter_length, 0)
    , in_buffer_size({v_in_buf_size})
    , in_offset(counter_length, 0)
    , in_position(counter_length, 0)
    , out_buffer_size({v_out_buf_size})
    , out_offset(counter_length, 0)
    , out_position(counter_length, 0) 
    {}

template<typename T>
size_t multidimensional_counter<T>::calculate_out_idx( const std::vector<T>& position ){
    size_t idx = 0;

    for(size_t i = 0; i < position.size(); ++i)
        if(size[i] <= position[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );

    for(size_t i = 0; i != position.size(); ++i){    // number of iterations
        auto idx = position.size() - 1 - i;
        idx += std::accumulate(out_buffer_size.begin() + idx + 1, out_buffer_size.end(), 1, std::multiplies<uint32_t>() ) * position[idx];
    };

    return idx;
}

template<typename T>
size_t multidimensional_counter<T>::calculate_in_idx( const std::vector<T>& position ){
    size_t idx = 0;

    for(size_t i = 0; i < position.size(); ++i)
        if(size[i] <= position[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );

    for(size_t i = 0; i != position.size(); ++i){    // number of iterations
        auto idx = position.size() - 1 - i;
        idx += std::accumulate(in_buffer_size.begin() + idx + 1, in_buffer_size.end(), 1, std::multiplies<uint32_t>() ) * position[idx];
    };

    return idx;
}

template<typename T>
size_t multidimensional_counter<T>::calculate_out_idx(){
    size_t result_idx = 0;

    for(size_t i = 0; i < counter.size(); ++i)
        if(size[i] <= counter[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );
    
    for(size_t i = 0; i != out_position.size(); ++i){    // number of iterations
        auto idx = out_position.size() - 1 - i;
        result_idx += std::accumulate(out_buffer_size.begin() + idx + 1, out_buffer_size.end(), 1, std::multiplies<uint32_t>() ) * out_position[idx];
    };

    return result_idx;
}

template<typename T>
size_t multidimensional_counter<T>::calculate_in_idx(){
    size_t result_idx = 0;

    for(size_t i = 0; i < counter.size(); ++i)
        if(size[i] <= counter[i]) throw std::in_of_range("Position is greater or equall to size at index: " + std::to_string(i) );
    
    for(size_t i = 0; i != in_position.size(); ++i){    // number of iterations
        auto idx = in_position.size() - 1 - i;
        result_idx += std::accumulate(in_buffer_size.begin() + idx + 1, in_buffer_size.end(), 1, std::multiplies<uint32_t>() ) * in_position[idx];
    };

    return result_idx;
}

template<typename T>
bool multidimensional_counter<T>::counter_finished(){
    for(auto it1  = counter.begin(), it2 = size.begin(); it1 != counter.cend(); ++it1, ++it2)
        if(*it1 != *it2) return false;

    return true;
}

template<typename T>
void multidimensional_counter<T>::counter_increase(){
    // Counter is vector representing number in number system in which maximum value of each digit at index 'i'
    // [denoted counter(i)] is limited by corresponding output_size(i). 
    // Counter can be shorter than size by any number of least significant digits.
    // When during incrementation counter(i)==output_size(i) digit at position 'i' it overflows with carry over to the left.
    // It means that digit at 'i' is zeroed and digit at 'i-1' is incremented.
    // The least significant digit is on the last(max index) position of the vector.
    ++counter.back();

    for(auto i = counter.size() - 1; i > 0; --i)
        if( counter[i] == size[i] ){
            counter[i] = 0;
            ++counter[i-1];
            
            out_position[i] = out_offset[i];
            ++out_position[i-1];

            in_position[i] = in_offset[i];
            ++in_position[i-1];
        } else 
            break;

    // After all counter(i) equal output_size(i) counter is zeroed through overflow
    // thus after this case we write output_size to counter
    if( counter[0] == size[0] )
        for(auto i = counter.size() - 1; i > 0; --i)
            counter[i] = size[i];
}

} //namespace neural

/*
void counter_increase(const std::vector<int32_t> &size, std::vector<int32_t> &counter, std::vector<int32_t> &val){
    for(auto i = val.size() - 1; i > 0; --i){
        counter[i] += val[i];

        for(auto x = i; x > 0; --x)
            if( counter[x] >= size[x] ){
                counter[x-1] += counter[x]/size[x];
                counter[x] = 0;
            }
    }

    // todo, isn't it to dangerous?
    // After all counter(i) equal output_size(i) counter is zeroed through overflow
    // thus after this case we write output_size to counter
    if( counter[0] >= size[0] )
        for(auto i = counter.size() - 1; i > 0; --i)
            counter[i] = size[i];
};
*/
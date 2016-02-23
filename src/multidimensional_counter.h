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
    std::vector<T> size;        //todo const
    std::vector<T> offset;      //todo const
    std::vector<T> counter;
    std::vector<T> position;

public:
    multidimensional_counter(const std::vector<T> &size, uint32_t counter_length);
    multidimensional_counter(const std::vector<T> &size, uint32_t counter_length, const std::vector<T> &offset);

    size_t calculate_idx(const std::vector<T> &position);
    size_t calculate_idx();
    bool counter_finished();
    void counter_increase();

    std::vector<T> get_size()   { return size;    };      //todo const
    std::vector<T> get_offset() { return offset;  }      //todo const
    std::vector<T> get_counter(){ return counter; };
};

template<typename T>
inline multidimensional_counter<T>::multidimensional_counter( const std::vector<T>& v_size, uint32_t counter_length )
    : size({v_size})
    , counter(counter_length, 0)
    , offset(counter_length, 0)
    , position(counter_length, 0){}

template<typename T>
inline multidimensional_counter<T>::multidimensional_counter( const std::vector<T>& v_size, uint32_t counter_length, const std::vector<T>& v_offset )
    : size({v_size})
    , counter(counter_length, 0)
    , offset({v_offset})
    , position({v_offset}){}

template<typename T>
size_t multidimensional_counter<T>::calculate_idx( const std::vector<T>& position ){
    size_t idx = 0;

    for(size_t i = 0; i < position.size(); ++i)
        if(size[i] <= position[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );

    for(size_t i = 0; i != position.size(); ++i){    // number of iterations
        auto idx = position.size() - 1 - i;
        idx += std::accumulate(size.begin() + idx + 1, size.end(), 1, std::multiplies<uint32_t>() ) * position[idx];
    };

    return idx;
}

template<typename T>
size_t multidimensional_counter<T>::calculate_idx(){
    size_t idx = 0;

    //for(size_t i = 0; i < counter.size(); ++i)
    //    if(size[i] <= counter[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );
    //
    //for(size_t i = 0; i != counter.size(); ++i){    // number of iterations
    //    auto idx = counter.size() - 1 - i;
    //    idx += std::accumulate(size.begin() + idx + 1, size.end(), 1, std::multiplies<uint32_t>() ) * counter[idx];
    //};

    for(size_t i = 0; i < counter.size(); ++i)
        if(size[i] <= counter[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );
    
    for(size_t i = 0; i != counter.size(); ++i){    // number of iterations
        auto idx = counter.size() - 1 - i;
        idx += std::accumulate(size.begin() + idx + 1, size.end(), 1, std::multiplies<uint32_t>() ) * position[idx];
    };

    return idx;
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
            
            position[i] = offset[i];
            ++position[i-1];
        }

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
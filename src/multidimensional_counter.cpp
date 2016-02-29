#include "multidimensional_counter.h"
#include <numeric>
#include <string>

size_t calculate_offset(const std::vector<uint32_t> &size, const std::vector<uint32_t> &position){
    size_t offset = 0;

    for(size_t i = 0; i < position.size(); ++i)
        if(size[i] <= position[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );

    for(size_t i = 0; i != position.size(); ++i){    // number of iterations
        auto idx = position.size() - 1 - i;
        offset += std::accumulate(size.begin() + idx + 1, size.end(), 1, std::multiplies<uint32_t>() ) * position[idx];
    };

    return offset;
};
bool counter_finished(const std::vector<uint32_t> &size, const std::vector<uint32_t> &counter){
    for(auto it1  = counter.begin(), it2 = size.begin(); it1 != counter.end(); ++it1, ++it2)
        if(*it1 != *it2) return false;

    return true;
};
void counter_increase(const std::vector<uint32_t> &size, std::vector<uint32_t> &counter){
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
        }

    // After all counter(i) equal output_size(i) counter is zeroed through overflow
    // thus after this case we write output_size to counter
    if( counter[0] == size[0] )
        for(auto i = counter.size() - 1; i > 0; --i)
            counter[i] = size[i];
};
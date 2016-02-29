#pragma once
#include <vector>
#include <functional>
#include <algorithm>

#include <vector>//todo remove
#include <numeric>//todo remove
#include <string>//todo remove

namespace ndimensional {

template<typename T>
class value : public std::vector<T> {
    template<typename U> struct change_signedness;
    template<>           struct change_signedness<uint32_t> { using type =  int32_t; };
    template<>           struct change_signedness< int32_t> { using type = uint32_t; };
    template<>           struct change_signedness<uint64_t> { using type =  int64_t; };
    template<>           struct change_signedness< int64_t> { using type = uint64_t; };
    using negT = typename change_signedness<T>::type;
public:
    class iterator {
        value<T> _current;
        const value &_ref;
        iterator(const value &arg, bool is_begin)
            : _current(is_begin ? value(arg.size()) : arg)
            , _ref(arg) {};
        friend class value<T>;
    public:
        iterator(const iterator &) = default;
        ~iterator() = default;
        iterator& operator=(const iterator &) = default;
        iterator& operator++() {
            for(size_t at=_current.size(); at--;) {
                if(++_current[at] == _ref[at])
                    _current[at]=0;
                else
                    return *this;
            }
            _current = _ref;
            return *this;
        }
        const value<T> &operator*() const { return _current; }
        iterator operator++(int) { iterator result(*this); ++(*this); return result; }
        bool operator==(const iterator &rhs) const { return &_ref==&rhs._ref && _current==rhs._current; }
        bool operator!=(const iterator &rhs) const { return !(*this==rhs); }
    };

    iterator begin() { return iterator(*this, true); }
    iterator end()   { return iterator(*this, false); }

    value(size_t size) : std::vector<T>(size, T(0)) {};
    value(std::vector<T> arg) : std::vector<T>(arg) {};
    value(std::initializer_list<T> il) : std::vector<T>(il) {};

    //todo range check asserts
    value &operator+=(std::vector<   T> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::plus<T>());       return *this; }
    value &operator+=(std::vector<negT> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::plus<T>());       return *this; }
    value &operator*=(std::vector<   T> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::multiplies<T>()); return *this; }
    value &operator*=(std::vector<negT> &arg) { std::transform(arg.cbegin(), arg.cend(), std::vector<T>::begin(), std::vector<T>::begin(), std::multiplies<T>()); return *this; }
    value  operator+ (std::vector<   T> &arg) { value result=*this; return result+=arg; }
    value  operator+ (std::vector<negT> &arg) { value result=*this; return result+=arg; }
    value  operator* (std::vector<   T> &arg) { value result=*this; return result*=arg; }
    value  operator* (std::vector<negT> &arg) { value result=*this; return result*=arg; }

    template<typename T> friend std::ostream &operator<<(std::ostream &, value &);
};

template<typename T>
std::ostream &operator<<(std::ostream &out, value<T> &val) {
    for(size_t at=0; at < val.size(); ++at)
        out << val[at] << (at + 1 == val.size() ? "" : ",");
    return out;
}

template<typename T>
size_t calculate_idx( const std::vector<T>& size, const std::vector<T>& position ){
    size_t result_idx = 0;

    for(size_t i = 0; i < position.size(); ++i)
        if(size[i] <= position[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );

    for(size_t i = 0; i != position.size(); ++i){    // number of iterations
        auto idx = position.size() - 1 - i;
        result_idx += std::accumulate(size.cbegin() + idx + 1, size.cend(), 1, std::multiplies<uint32_t>() ) * position[idx];
    };

    return result_idx;
}

}
namespace neural{

template<typename T>
class multidimensional_counter {
    std::vector<T> size;            //todo consts
    std::vector<T> counter;

    std::vector<T> in_buffer_size;
    std::vector<T> in_offset; //todo negative will fail
    std::vector<T> in_position;
    std::vector<T> in_stride;
    std::vector<T> out_buffer_size;
    std::vector<T> out_offset;
    std::vector<T> out_position;

public:
    multidimensional_counter(const std::vector<T> &size,
                             uint32_t             counter_length,
                             const std::vector<T> &in_buf_size,
                             const std::vector<T> &in_offset,
                             const std::vector<T> &in_stride,
                             const std::vector<T> &out_buf_size,
                             const std::vector<T> &out_offset);

    multidimensional_counter(const std::vector<T> &size,
                             uint32_t             counter_length,
                             const std::vector<T> &in_buf_size,
                             const std::vector<T> &in_stride,
                             const std::vector<T> &out_buf_size);

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

    std::vector<T> get_size()        {return size;        };      //todo consts
    std::vector<T> get_counter()     {return counter;     };
    std::vector<T> get_in_position() {return in_position; };
    std::vector<T> get_out_position(){return out_position;};
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
    , in_offset(v_in_offset.begin(), v_in_offset.begin() + counter_length)
    , in_position({in_offset})
    , out_buffer_size({v_out_buf_size})
    , out_offset(v_out_offset.begin(), v_out_offset.begin() + counter_length)
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
inline multidimensional_counter<T>::multidimensional_counter( const std::vector<T> &v_size,
                                                              uint32_t             counter_length,
                                                              const std::vector<T> &v_in_buf_size,
                                                              const std::vector<T> &v_in_offset,
                                                              const std::vector<T> &v_in_stride,
                                                              const std::vector<T> &v_out_buf_size,
                                                              const std::vector<T> &v_out_offset)
    : size({v_size})
    , counter(counter_length, 0)
    , in_buffer_size({v_in_buf_size})
    , in_offset( v_in_offset.begin(), v_in_offset.begin() + counter_length )
    , in_stride( v_in_stride.begin(), v_in_stride.begin() + counter_length )
    , in_position({in_offset})
    , out_buffer_size({v_out_buf_size})
    , out_offset(v_out_offset.begin(), v_out_offset.begin() + counter_length)
    , out_position({out_offset})
    {}

template<typename T>
inline multidimensional_counter<T>::multidimensional_counter( const std::vector<T> &v_size,
                                                              uint32_t             counter_length,
                                                              const std::vector<T> &v_in_buf_size,
                                                              const std::vector<T> &v_in_stride,
                                                              const std::vector<T> &v_out_buf_size)
    : size({v_size})
    , counter(counter_length, 0)
    , in_buffer_size({v_in_buf_size})
    , in_offset(counter_length, 0)
    , in_position(counter_length, 0)
    , in_stride( v_in_stride.begin(), v_in_stride.begin() + counter_length )
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
        if(size[i] <= counter[i]) throw std::out_of_range("Position is greater or equall to size at index: " + std::to_string(i) );

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
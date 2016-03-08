#pragma once
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>

namespace ndimensional {

template<typename U> struct change_signedness;
template<> struct change_signedness< int32_t> { using type = uint32_t; };
template<> struct change_signedness<uint32_t> { using type =  int32_t; };
template<> struct change_signedness<uint64_t> { using type =  int64_t; };
template<> struct change_signedness< int64_t> { using type = uint64_t; };

template<typename T>
class value : public std::vector<T> {
    using negT = typename change_signedness<T>::type;
public:
// Iterator represents number in number system in which maximum value of each digit at index 'i'
// [denoted _current.at(i)] is limited by corresponding value.at(i).
// When during incrementation _current(i)==value(i) digit at position 'i' it overflows with carry over to the left.
// It means that digit at 'i' is zeroed and digit at 'i-1' is incremented.
// The least significant digit is on the last(max index) position of the vector.
    class iterator {
        value<T> _current;
        const value &_ref;
        iterator(const value &arg, bool is_begin)
            : _current(is_begin ? value(arg.size()) : arg)
            , _ref(arg) {};
        friend class value<T>;
    public:
        iterator(const iterator &) = default;
        iterator(iterator &&) = default;
        ~iterator() = default;
        iterator& operator=(const iterator &) = default;
        iterator& operator=(iterator &&) = default;
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

    template<typename U> friend std::ostream &operator<<(std::ostream &, ndimensional::value<U> &);
};

template<typename U>
std::ostream &operator<<(std::ostream &out, value<U> &val) {
    for(size_t at = 0; at < val.size(); ++at)
        out << val[at] << (at + 1 == val.size() ? "" : ",");
    return out;
}

template<typename T>
class calculate_idx{
    using negT = typename change_signedness<T>::type;

    std::vector<T> size;
    std::vector<T> stride;
public:
    calculate_idx( const std::vector<T>& v_size )
    : size(v_size)
    , stride(v_size) {

        assert( std::is_unsigned<T>() == true );  //this template should be used only with unsigned types

        stride.emplace_back(1); //this element is used in operator()
        for(size_t i = stride.size() - 1; i > 0; --i)
            stride[i-1] *= stride[i];
    };

    size_t operator() ( const std::vector<   T>& pos );
    size_t operator() ( const std::vector<negT>& pos );

    bool is_out_of_range( const std::vector<   T>& pos );
    bool is_out_of_range( const std::vector<negT>& pos );
};

template<typename T>
inline size_t calculate_idx<T>::operator()( const std::vector<T>& position ){
    size_t result_idx = 0;

    assert(
        [&]() -> bool {
        for(size_t i = 0; i < position.size(); ++i)
            if(size[i] <= position[i]) return false;

          return true;
        }() == true );

    // Number of iterations depends on length of position vector.
    // 'position' can be shorter than 'size' because last numbers (with the highest indexes) coressponds data with linear memory layout.
    // If 'position' is shorter than 'size' than function returns offset to some block of data
    for(size_t i = 0; i != position.size(); ++i){
        auto idx = position.size() - 1 - i;
        result_idx += stride[idx+1] * position[idx];
    };

    return result_idx;
}

template<typename T>
inline size_t calculate_idx<T>::operator()( const std::vector<negT>& position ){
    size_t result_idx = 0;

    assert(
        [&]() -> bool {
        for(size_t i = 0; i < position.size(); ++i)
            if(size[i] <= position[i]) return false;

          return true;
        }() == true );

    // Number of iterations depends on length of position vector.
    // 'position' can be shorter than 'size' because last numbers (with the highest indexes) coressponds data with linear memory layout.
    // If 'position' is shorter than 'size' than function returns offset to some block of data
    for(size_t i = 0; i != position.size(); ++i){
        auto idx = position.size() - 1 - i;
        result_idx += stride[idx+1] * position[idx];
    };

    return result_idx;
}

template<typename T>
inline bool calculate_idx<T>::is_out_of_range( const std::vector<negT>& pos ){
    bool out_of_range = false;

    assert( pos.size() < size.size() );

    for(uint32_t i = 0; i < pos.size() && !out_of_range; ++i)
        out_of_range |= (pos[i] < 0) | (static_cast<negT>(pos[i]) > size[i]);

    return out_of_range;
}

template<typename T>
inline bool calculate_idx<T>::is_out_of_range( const std::vector<T>& pos ){
    bool out_of_range = false;

    assert( pos.size() < size.size() );

    for(uint32_t i = 0; i < pos.size() && !skip; ++i)
        skip |= (pos[i] > size[i]);

    return out_of_range;
}
}

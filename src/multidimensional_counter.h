#pragma once
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>

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
    for(size_t at = 0; at < val.size(); ++at)
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

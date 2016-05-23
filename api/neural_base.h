/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <map>
#include <exception>
#include <cassert>



// exporting symbols form dynamic library
#ifdef EXPORT_NEURAL_SYMBOLS
#   if defined(_MSC_VER)
       //  Microsoft
#      define DLL_SYM __declspec(dllexport)
#   elif defined(__GNUC__)
       //  GCC
#      define DLL_SYM __attribute__((visibility("default")))
#   else
#      define DLL_SYM
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#else //import dll
#   if defined(_MSC_VER)
       //  Microsoft
#      define DLL_SYM __declspec(dllimport)
#   elif defined(__GNUC__)
       //  GCC
#      define DLL_SYM
#   else
#      define DLL_SYM
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#endif


namespace neural {

// task to be performed in form of callback & data for it
struct task {
    void (*callback)(const void *);
    const void *data;
};

#if defined(_MSC_VER)
namespace {
// (de)initializing global constructors within MKL-DNN
extern "C" DLL_SYM void _cdecl nn_init();
extern "C" DLL_SYM void _cdecl nn_exit();

template<typename T = void> class lib_init_t {
    lib_init_t() { nn_init(); }
    ~lib_init_t() { nn_exit(); }
    void operator=(lib_init_t const&) = delete;
public:
    DLL_SYM static lib_init_t &instance() {
        static lib_init_t instance_;
        return instance_;
    }
};

//...by singleton injected into application
template class lib_init_t<void>;

} //namespace {
#endif // _MSC_VER


// neural::vector
// ...is a container for storing data size of or position within multi-dimensional data.
// It is split into 3 parts: vector:batch, vector:spatial, vector:features.
// In most use cases batch and features are scalars. Features is a vector eg. for weights in convolution.
template<typename T> struct vector {
    std::vector<T> raw;
    class ref_vector {
        std::vector<T> &raw_;
        size_t begin_;
        size_t end_;
        ref_vector(std::vector<T> &raw, size_t begin, size_t end) : raw_(raw), begin_(begin), end_(end) {};
        friend struct vector<T>;
    public:
        typename std::vector<T>::iterator begin() { return raw_.begin()+begin_; }
        typename std::vector<T>::iterator end()   { return raw_.begin()+end_; }
        typename std::vector<T>::const_iterator cbegin() const { return raw_.cbegin() + begin_; }
        typename std::vector<T>::const_iterator cend() const { return raw_.cbegin() + end_; }
        size_t size() const { return end_-begin_; }
        operator T() const { return raw_[0]; }
        T& operator[](size_t at) { assert(at<end_ - begin_); return raw_[begin_ + at]; }
        T operator[](size_t at) const { assert(at<end_ - begin_); return raw_[begin_ + at]; }
    } batch, feature, spatial;
    bool operator==(const vector &rhs) const { return rhs.spatial==spatial && rhs.feature==feature && rhs.batch==batch; }
    bool operator!=(const vector &rhs) const { return !(*this==rhs); }
    vector(const vector &arg)
        : raw(arg.raw)
        , batch  (raw, arg.batch.begin_,   arg.batch.end_)
        , feature(raw, arg.feature.begin_, arg.feature.end_)
        , spatial(raw, arg.spatial.begin_, arg.spatial.end_)
    {}
    vector &operator=(const vector &arg)  {
        raw = arg.raw;
        spatial.raw_ = arg.spatial.raw_; // todo what is it?
    }
    vector() : raw(0), batch(raw,0,0), feature(raw,0,0), spatial(raw,0,0) {}
    vector(size_t size): raw(2+size), batch(raw,0,1), feature(raw,1,2), spatial(raw,2, 2+size) {}
    vector(const T arg_batch, const std::vector<T> arg_spatial, const T arg_feature)
        : batch(raw,0,1)
        , feature(raw,1,2)
        , spatial(raw,2, 2+arg_spatial.size())
    {
        raw.push_back(arg_batch);
        raw.push_back(arg_feature);
        raw.insert(raw.end(), arg_spatial.begin(), arg_spatial.end());
    };
    vector(const T arg_batch, const std::vector<T> &arg_spatial, const std::vector<T> &arg_feature)
        : batch  (raw, 0, 1)
        , feature(raw, 1, 1+arg_feature.size())
        , spatial(raw, 1+arg_feature.size(), 1+arg_feature.size()+arg_spatial.size())
    {
        raw.push_back(arg_batch);
        raw.insert(raw.end(), arg_feature.begin(), arg_feature.end());
        raw.insert(raw.end(), arg_spatial.begin(), arg_spatial.end());
    };
    vector(const size_t len_batch, const size_t len_spatial, const size_t len_feature)
        : batch  (raw , 0                    , len_batch)
        , feature(raw , len_batch            , len_batch+len_feature)
        , spatial(raw , len_batch+len_feature, len_batch+len_feature+len_spatial)
    {
        raw.resize(len_batch + len_feature + len_spatial);
    };
    vector(const std::vector<T> arg_spatial, const T arg_feature)
        : batch(raw,0,1)
        , feature(raw,1,2)
        , spatial(raw,2,2+arg_spatial.size())
    {
        raw.push_back(1);
        raw.push_back(arg_feature);
        raw.insert(raw.end(), arg_spatial.begin(), arg_spatial.end());
    };
    vector(const std::vector<T> &arg_spatial, const std::vector<T> arg_feature)
        : batch  (raw, 0, 1)
        , feature(raw, 1, 1+arg_feature.size())
        , spatial(raw, 1+arg_feature.size(), 1+arg_feature.size()+arg_spatial.size())
    {
        raw.push_back(1);
        raw.insert(raw.end(), arg_feature.begin(), arg_feature.end());
        raw.insert(raw.end(), arg_spatial.begin(), arg_spatial.end());
    };
    vector(const std::vector<T> arg_spatial)
        : batch(raw,0,1)
        , feature(raw,1,2)
        , spatial(raw,2,2+arg_spatial.size())
    {
        raw.push_back(1);
        raw.push_back(1);
        raw.insert(raw.end(), arg_spatial.begin(), arg_spatial.end());
    };
    vector(const T arg_batch, const std::vector<T> arg_spatial)
        : batch(raw,0,1)
        , feature(raw,1,2)
        , spatial(raw,2, 2+arg_spatial.size())
    {
        raw.push_back(arg_batch);
        raw.push_back(1);
        raw.insert(raw.end(), arg_spatial.begin(), arg_spatial.end());
    };
};

// type_traits
// ...contains unique id, size, traits & name of particular type
// part of minimal RTTI-independent runtime type traits
struct type_traits {
    const size_t          id;
    const size_t          size;
    const bool            is_floating_point;
    const char *          const name;
    type_traits(size_t _id, size_t _size, bool _ifp, const char *_name) : id(_id), size(_size), is_floating_point(_ifp), name(_name) {};
private:
    type_traits(const type_traits &);
    type_traits &operator=(const type_traits &);
};

// is_floating_point<T>
// ...compile-time detection if type is floatinig-point [for non-C++11 compilers]
template<typename T> struct is_floating_point        { static const bool value = false; };
template<>           struct is_floating_point<float> { static const bool value = true; };
template<>           struct is_floating_point<double>{ static const bool value = true; };
#if defined HALF_HALF_HPP
template<>           struct is_floating_point<half>  { static const bool value = true; };
#endif

DLL_SYM type_traits* typeid_register(size_t size, bool is_float, const std::string& str);

// type_id()
// ...returns pointer to type-traits
#if defined _MSC_VER
template<typename T_type> __declspec(noinline)      type_traits *type_id() {
    static std::string signature = __FUNCSIG__;
    static std::string type_name = signature.substr(signature.find('<', 0)+1, signature.find('>', 0)-signature.find('<', 0)-1);
#else
template<typename T_type> __attribute__((noinline)) type_traits *type_id() {
    static std::string signature =__PRETTY_FUNCTION__;
    static std::string type_name = signature.substr(signature.find('=', 0)+2, signature.find(']', 0)-signature.find('=', 0)-2);
#endif
    static type_traits *ti = typeid_register(sizeof(T_type), is_floating_point<T_type>::value, type_name);
    return ti;
}

class engine  { engine();  public: enum type { 
    // engines
      reference                     // naive & easy to debug implementation for validation
    , cpu                           // optimized CPU implementation
    , gpu                           // GPU implementation
    , any=static_cast<uint32_t>(-1) // 'any' engine for querries

    // attributies
    , lazy = 0x80000000             // lazy evaluation
}; };
inline engine::type operator|(engine::type a, engine::type b) { return static_cast<engine::type>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b)); };

class padding { padding(); public: enum type { zero }; };

// value in any format
class any_value {
    std::tuple<size_t, void *> _value;
public:
    template<typename T> any_value &operator=(const T &arg) {
        assert(type_id<T>()->id==std::get<0>(_value));
        std::get<1>(_value) = arg;
        return *this;
    }
    template<typename T> operator T() const{
        assert(type_id<T>()->id==std::get<0>(_value));
        return std::get<1>(_value);
    }
    template<typename T> const T& as() const {
        assert(type_id<T>()->id==std::get<0>(_value));
        return *reinterpret_cast<T *>(std::get<1>(_value));
    }
};

// lookup wrapper, attaches type to key string and does lookup
class any_value_type_lookup {
    const std::map<std::string, any_value> &_map;
    std::string                             _key;
public:
    any_value_type_lookup(const std::map<std::string, any_value> &map, std::string key) : _map(map), _key(key) {};
    any_value_type_lookup &operator=(const any_value_type_lookup &);

    friend class is_a_primitive;
public:
    template<typename T> T as() const {
        std::string key = _key + ":" + type_id<T>()->name;
        auto it = _map.find(key);
        if(it!=_map.end()) return it->second.as<T>();
        else throw std::runtime_error("[TBD]");
    }
    template<typename T> operator T() const{ return as<T>(); }
    std::string s()   const { return as<std::string>(); }
#if defined HALF_HALF_HPP
    float       f16() const { return as<half>(); }
#endif
    float       f32() const { return as<float>(); }
    double      f64() const { return as<double>(); }
    uint8_t     u8()  const { return as<uint8_t>(); }
    uint16_t    u16() const { return as<uint16_t>(); }
    uint32_t    u32() const { return as<uint32_t>(); }
    uint64_t    u64() const { return as<uint64_t>(); }
    int8_t      i8()  const { return as<int8_t>(); }
    int16_t     i16() const { return as<int16_t>(); }
    int32_t     i32() const { return as<int32_t>(); }
    int64_t     i64() const { return as<int64_t>(); }
};

// each query entry is another possible compilation with its own set of attributes
class is_a_query_entry {
    std::map<std::string, any_value> _map;
public:
    any_value_type_lookup operator[](std::string key) const { return any_value_type_lookup(_map, key); }
};

class is_a_worker {
    is_a_worker(const is_a_worker &);
    is_a_worker &operator=(const is_a_worker &);

protected:
    type_traits                     *_type_traits;
    std::map<std::string, any_value> _map;
    is_a_worker(type_traits *traits) : _type_traits(traits) {}
public:
    virtual ~is_a_worker() {};
    virtual any_value_type_lookup operator[](std::string &key) const { return any_value_type_lookup(_map, key); }

    virtual void execute(const std::vector<task>& requests) const = 0;
    virtual neural::engine::type engine() const = 0;
};

class worker {
    std::shared_ptr<is_a_worker> _pointer;

public:
    worker(is_a_worker *raw) : _pointer(raw) {};
    worker(const worker &other) : _pointer(other._pointer) {};

    neural::engine::type engine() { return _pointer->engine(); }
    void execute(const std::vector<task>& requests) const { _pointer->execute(requests);}
};

// cheap to copy handle with reference counting
class is_a_primitive;
struct primitive_at;
class primitive {
    std::shared_ptr<const is_a_primitive> _pointer;

    // [C++1x] replace with std:: versions
    template<typename T> struct is_reference         { static const bool value = false; };
    template<typename T> struct is_reference<T&>     { static const bool value = true; };
    template<typename T> struct remove_reference     {typedef T type;};
    template<typename T> struct remove_reference<T&> {typedef T type;};

public:
    primitive(const is_a_primitive *raw) : _pointer(raw) {};
    primitive(const primitive &other) : _pointer(other._pointer) {};
    any_value_type_lookup operator[] (const std::string &arg) const;
    const primitive operator()(void *argument) const;
#if defined __GNUC__
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
    class input {
        const primitive *get_base() const {
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(this);
            ptr -= (size_t)&reinterpret_cast<const volatile char&>((((primitive *)0)->input));
            return reinterpret_cast<const primitive *>(ptr);
        }
    public:
        inline const primitive_at operator[](uint32_t) const;
        inline const primitive_at operator[](std::string) const;
        inline size_t size() const;
    } input;
    class output {
        const primitive *get_base() const {
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(this);
            ptr -= (size_t)&reinterpret_cast<const volatile char&>((((primitive *)0)->output));
            return reinterpret_cast<const primitive *>(ptr);
        }
    public:
        inline const primitive operator[](uint32_t) const;
        inline const primitive operator[](std::string) const;
        inline size_t size() const;
    } output;
#if defined __GNUC__
#   pragma GCC diagnostic pop
#endif
    template<typename T> T as() const;
    template<typename T> operator T() { return as<T>(); }
    const std::vector<task> &work() const;
    size_t id() const;
    bool operator==(const primitive &other) const { return _pointer==other._pointer; }
    bool operator!=(const primitive &other) const { return !(*this==other); }
};


struct primitive_at {
    const neural::primitive primitive;
    const uint32_t          at;
    primitive_at(const neural::primitive aprimitive) : primitive(aprimitive), at(0) {}
    primitive_at(const neural::primitive aprimitive, const uint32_t pos) : primitive(aprimitive), at(pos) {}
};

struct memory;

// is_a_primitive is a base class for all primitives exposing common interface; primiary user is a primitive wrapper
class is_a_primitive {
    is_a_primitive(const is_a_primitive &);
    is_a_primitive &operator=(const is_a_primitive &);
protected:
    type_traits                     *_type_traits;
    std::map<std::string, any_value> _map;
    std::vector<task>                _work;
    is_a_primitive(type_traits *traits) : _type_traits(traits) {}
public:
    virtual ~is_a_primitive() {};
    virtual any_value_type_lookup operator[](std::string &key) const { return any_value_type_lookup(_map, key); }
    virtual const std::vector<primitive_at>  &input()  const { throw std::runtime_error(std::string("no inputs in ")+_type_traits->name); };
    virtual const std::vector<primitive>     &output() const { throw std::runtime_error(std::string("no outputs in ")+_type_traits->name); };
    const memory &input_memory(uint32_t at) const {
        auto prim = input()[at].primitive;
        return (prim.id()==type_id<const memory>()->id ? prim : prim.output[input()[at].at]).as<const memory &>();
    }
    const memory &output_memory(uint32_t at) const  { return output()[at].as<const memory &>(); };
    virtual void execute_argument(void *) const { throw std::runtime_error(std::string("execute-time argument not supported in")+_type_traits->name); }
    friend class primitive;

    // to be removed when new thread queue will be done
    friend struct nn_semaphore;
    friend struct nn_thread_worker;
    friend struct nn_thread_worker_pool;
};

// implementations of inline functions from primitive
inline const primitive_at           primitive::input::operator[] (uint32_t at) const { return get_base()->_pointer->input()[at]; }
inline size_t                       primitive::input::size() const { return get_base()->_pointer->input().size(); }
inline const primitive              primitive::operator()(void *argument) const { _pointer->execute_argument(argument); return *this; }
inline const std::vector<task> &    primitive::work() const { return _pointer->_work; }
inline size_t                       primitive::id() const { return _pointer->_type_traits->id; }
inline const primitive              primitive::output::operator[](uint32_t at) const { return get_base()->_pointer.get()->output()[at]; }
inline size_t                       primitive::output::size() const { return get_base()->_pointer.get()->output().size(); }
inline any_value_type_lookup        primitive::operator[](const std::string &key) const { return any_value_type_lookup(_pointer->_map, key); }


template<typename T> T primitive::as() const {
    // [C++1x] replace with static_assert
    assert(is_reference<T>::value == true);
    assert(type_id<typename remove_reference<T>::type>()->id == _pointer->_type_traits->id);
    return *reinterpret_cast<typename remove_reference<T>::type *>(_pointer.get());
}
// unkown structure with type info for cast validation
class is_an_implementation {
    const type_traits *const _type_traits;
protected:
    is_an_implementation(const type_traits *arg) : _type_traits(arg) {};
public:
    virtual std::vector<task> work() = 0;
    virtual ~is_an_implementation() {};
};

// asynchronous result
class async_result {
    std::shared_ptr<volatile uint32_t> _tasks_left;
    async_result(std::shared_ptr<volatile uint32_t> arg) : _tasks_left(arg) {}

    // execution of sequence of primitives
    friend DLL_SYM async_result execute(std::vector<primitive>, worker);
    friend DLL_SYM async_result execute(std::vector<primitive>);
public:
    uint32_t tasks_left() { return *_tasks_left; };
    void sync() { while(tasks_left()); }
};
DLL_SYM async_result execute(std::vector<primitive>, worker);
DLL_SYM async_result execute(std::vector<primitive>);

}

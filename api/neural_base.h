#pragma once

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <map>
#include <exception>
#include <cassert>

// [TODO]
//      rename is_a_... to something gramatically correct
//      compiler-agnostic compile-time assertions for C++98


namespace neural {

// minimal RTTI-independent type traits
struct type_traits {
    const size_t          id;
    const size_t          size;
    const char *    const name;
    type_traits(size_t _id, size_t _size, const char *_name) : id(_id), size(_size), name(_name) {};
    type_traits &operator=(const type_traits &);
};

template<typename T_type> __declspec(noinline) auto type_id() -> type_traits * {
#if defined _MSC_VER
    static std::string signature = __FUNCSIG__;
    static std::string type_name = signature.substr(signature.find('<', 0)+1, signature.find('>', 0)-signature.find('<', 0)-1);
#else
    static std::string signature =__PRETTY_FUNCTION__;
    static std::string type_name = signature.substr(signature.find('=', 0)+2, signature.find(']', 0)-signature.find('=', 0)-2);
#endif

    static type_traits ti{{reinterpret_cast<size_t>(&ti)}, sizeof(T_type), type_name.c_str()};
    return &ti;
}


enum class engine : size_t { reference, cpu, any=static_cast<size_t>(-1) };
enum class padding  : size_t { zero, data };

// value in any format
class any_value {
    std::tuple<size_t, void *> _value;
public:
    template<typename T> any_value &operator=(const T &arg) {
        //assert(type_id(T)==std::get<0>(_value));
        std::get<1>(_value) = arg;
        return *this;
    }
    template<typename T> operator T() const{
        //assert(type_id(T)==std::get<0>(_value));
        return std::get<1>(_value);
    }
    template<typename T> const T& as() const {
        //assert(type_id(T())->id==std::get<0>(_value)->id);
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

// task to be performed in form of callback & data for it
struct task {
    void (*callback)(const void *);
    const void *data;
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
    any_value_type_lookup operator[] (const std::string &key) const;
    const primitive operator()(void *argument) const;
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

    template<typename T> T as() const {
        // [C++1x] replace with static_assert
        assert(is_reference<T>::value==true);
        assert(type_id<remove_reference<T>::type>()->id ==_pointer->_type_traits->id);
        return *reinterpret_cast<remove_reference<T>::type *>(_pointer.get());
    }
    template<typename T> operator T() { return as<T>(); }
    const std::vector<task> &work();
    size_t id() const;
};


struct primitive_at {
    const primitive   primitive;
    const uint32_t    at;
    primitive_at(const neural::primitive aprimitive) : primitive(aprimitive), at(0) {}
};

struct memory;

// is_a_primitive is a base class for all primitives exposing common interface; primiary user is a primitive wrapper
class is_a_primitive {
protected:
    type_traits                     *_type_traits;
    std::map<std::string, any_value> _map;
    std::vector<task>                _work;
    is_a_primitive(type_traits *traits) : _type_traits(traits) {}
public:
    virtual ~is_a_primitive() {};
    virtual primitive clone() const = 0;
    virtual any_value_type_lookup operator[](std::string &key) const { return any_value_type_lookup(_map, key); }
    virtual const std::vector<primitive_at>  &input() const = 0;
    virtual const std::vector<primitive>     &output() const = 0;
    const memory &input_memory(uint32_t at) const { 
        auto prim = input()[at].primitive;
        return (prim.id()==type_id<memory>()->id ? prim : prim.output[input()[at].at]).as<const memory &>();
    }
    const memory &output_memory(uint32_t at) const  { return output()[at].as<const memory &>(); };
    virtual void execute_argument(void *argument) const { throw std::runtime_error("this primitive does not need execute-time argument"); }
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
inline const std::vector<task> &    primitive::work() { return _pointer->_work; }
inline size_t                       primitive::id() const { return _pointer->_type_traits->id; }

inline const primitive              primitive::output::operator[](uint32_t at) const { return get_base()->_pointer.get()->output()[at]; }
inline size_t                       primitive::output::size() const { return get_base()->_pointer.get()->output().size(); }
inline const size_t                 primitive::id() { return _pointer.get()->_type_traits->id; }

// unkown structure with type info for cast validation
class is_a_unknown {
    const type_traits *const _type_traits;
protected:
    is_a_unknown(const type_traits *arg) : _type_traits(arg) {};
public:
    virtual ~is_a_unknown(){};
};

// execution of sequence of primitives
void execute(std::vector<primitive>);
}
#include "api/neural.h"
#include <map>
#include <functional>

template<typename T, typename U>
class singleton_map : public std::map<T, U> {
    singleton_map() : std::map<T, U>() {};
    singleton_map(singleton_map const&) = delete;
    void operator=(singleton_map const&) = delete;

    public:
    static singleton_map &instance() {
        static singleton_map instance_;
        return instance_;
    }
};

namespace neural {

using default_key_type = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;

template<typename T>
default_key_type default_key_builder(T* impl) {
    return std::make_tuple(impl->argument.engine, impl->input_memory(0).argument.format, impl->output_memory(0).argument.format);
}

template<typename Impl,
    typename key_type = default_key_type,
    typename key_builder = default_key_builder<Impl>>
    class implementation_map {
    public:
        //using key_type = std::tuple<neural::engine::type, neural::memory::format::type, neural::memory::format::type>;
        using factory_type = std::function<is_an_implementation *(Impl &)>;
        using map_type = singleton_map<key_type, factory_type>;

        template<typename Arg>
        static Impl* create(Arg arg) {
            // wrap into RAII wrapper 
            std::unique_ptr<Impl> result(new Impl(arg));

            // create implementation for non-lazy evaluation 
            if (0 == (arg.engine & engine::lazy)) {
                // lookup in database; throw if not found 
                auto key = key_builder(result.get());
                auto it = map_type::instance().find(key);
                if (it == std::end(map_type::instance())) throw std::runtime_error("not yet implemented");

                // create implementation & attach it to result 
                auto implementation = it->second(*result);

                result->_private.reset(implementation);
                result->_work = implementation->work();
            }

            // release RAII wrapper, return naked pointer 
            return result.release();
        }

        static void add(key_type key, factory_type factory) {
            map_type::instance().insert(key, factory);
        }
};

}

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

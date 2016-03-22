#include "api/neural.h"
#include <map>
#include <functional>

template<typename T, typename U>
class singletion_map : public std::map<T, U> {
    singletion_map() : std::map<T, U>() {};
    singletion_map(singletion_map const&) = delete;
    void operator=(singletion_map const&) = delete;

    public:
    static singletion_map &instance() {
        static singletion_map instance_;
        return instance_;
    }
};

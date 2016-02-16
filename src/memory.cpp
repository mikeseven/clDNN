#include "neural.h"

#include <functional>
#include <numeric>

namespace neural {

memory::arguments::arguments(neural::engine aengine, memory::format aformat, std::vector<uint32_t> asize)
    : engine(aengine)
    , format(aformat)
    , size(asize)
    , owns_memory(false) {}

memory::arguments::arguments(neural::engine aengine, memory::format aformat, std::vector<uint32_t> asize, bool aowns_memory)
    : engine(aengine)
    , format(aformat)
    , size(asize)
    , owns_memory(aowns_memory) {}


size_t memory::count() const {
    return std::accumulate(argument.size.begin(), argument.size.end(), size_t(1), std::multiplies<size_t>());
}

memory::~memory() {
    if(argument.owns_memory) delete[] pointer;
}

primitive memory::create(memory::arguments arg){
    auto result = std::unique_ptr<memory>(new memory(arg));
    if(arg.owns_memory) {
        result->pointer = new char[result->count()*memory::traits(arg.format).type->size];
    }
    return result.release();
}

}
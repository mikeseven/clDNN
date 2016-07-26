#include "cost_model.h"

namespace neural { namespace gpu { namespace cache {

cost_model::cost::cost(size_t value) : value{ value } { }

bool cost_model::cost::operator<(const cost & rhs) const
{
    return value < rhs.value;
}

cost_model::cost cost_model::rate(const binary_data & kernel_binary)
{
    return cost{ kernel_binary.length() };
}

} } }
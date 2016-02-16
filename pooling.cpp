#include "neural.h"

namespace neural {

pooling::arguments::arguments(neural::engine, pooling::mode, memory::format o_frmt, std::vector<uint32_t> out_off, std::vector<uint32_t> out_siz, primitive in, std::vector<int32_t> in_off, std::vector<uint32_t> stride, std::vector<uint32_t> size, neural::padding);

    : engine(eng)
    , mode(mode)
    , output({out})
    , output_offset({out.as<const memory&>().argument.size.size()})
    , output_size(out.as<const memory&>().argument.size.begin(), out.as<const memory&>().argument.size.end())
    , input({in})
    , input_offset({in.as<const memory&>().argument.size.size()})
    , stride(stride)
    , size({0}) //todo ?
    , padding(padd)
{}

}
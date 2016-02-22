#include "neural.h"

namespace neural {

pooling::arguments::arguments( neural::engine eng,
                               pooling::mode mode,
                               memory::format o_frmt,
                               std::vector<uint32_t> out_off,
                               std::vector<uint32_t> out_siz,
                               primitive in,
                               std::vector<int32_t> in_off,
                               std::vector<uint32_t> arg_stride,
                               std::vector<uint32_t> arg_size,
                               neural::padding apadd)
    : engine(eng)
    , mode(mode)
    , output( {memory::create({eng, o_frmt, out_siz})} )
    , output_offset(out_off)
    , output_size(out_siz)
    , input({in})
    , input_offset(in_off)
    , stride(arg_stride)
    , size(arg_size)
    , padding(apadd)
{}

}
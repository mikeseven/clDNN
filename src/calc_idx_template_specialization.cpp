#include "multidimensional_counter.h"

namespace ndimensional{

template<>
size_t index<neural::memory::format::yxfb_f32>(std::vector<uint32_t> size, std::vector<uint32_t> pos){
    assert(
    [&]() -> bool {
    for(size_t i = 0; i < pos.size(); ++i)
        if(size[i] <= pos[i]) return false;

        return true;
    }() == true );
    assert(pos.size() == size.size());

    // strides for yxfb format
    // vectors v_size and stride use format: b, f, spatials(y,x...)
    return pos[0] + size[0] * (pos[1] + size[1]*(pos[3] + size[3] * pos[2]));
};

fptr choose_calucalte_idx(neural::memory::format::type arg){
    fptr ptr;
    switch (arg){
        case neural::memory::format::type::yxfb_f32:
            ptr = index<neural::memory::format::type::yxfb_f32>;
            break;
        default:
            throw std::runtime_error("choose_calucalte_idx has no case for memory::format " + std::to_string(arg));
    }
    return ptr;
};

}

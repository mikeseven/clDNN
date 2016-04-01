#ifdef _MSC_VER

#include <new>
#include <iostream>

#include "api/dll.h"

using void_fun_ptr = void (__cdecl *)();

#pragma warning(disable : 4075)

#pragma section(".nn_init$a", read, write)
__declspec(allocate(".nn_init$a")) void_fun_ptr nn_cl_seg_start = reinterpret_cast<void_fun_ptr>(1);
#pragma section(".nn_init$b", read, write)
__declspec(allocate(".nn_init$b")) void_fun_ptr nn_cl_seg_zero = nullptr;
#pragma section(".nn_init$z", read, write)
__declspec(allocate(".nn_init$z")) void_fun_ptr nn_cl_seg_end = reinterpret_cast<void_fun_ptr>(1);

int nn_init_seg_add(void_fun_ptr fp) {
    static auto ptr = &nn_cl_seg_start;
    ptr[0] = fp;
    ptr[1] = nullptr;
    ++ptr;
    return 0;
}

#pragma init_seg(".nn_init$m", nn_init_seg_add)

extern "C" DLL_SYM void _cdecl nn_init() {
    for (auto fp=(&nn_cl_seg_start)+1; fp<&nn_cl_seg_end; ++fp) if (*fp) (*fp)();
}
extern "C" DLL_SYM void _cdecl nn_exit() {
    for (auto fp=&nn_cl_seg_start; *fp; ++fp) (*fp)();
}

#endif // _MSC_VER

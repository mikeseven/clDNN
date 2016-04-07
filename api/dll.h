#pragma once

#ifdef EXPORT_NIA_SYMBOLS
#   if defined(_MSC_VER)
       //  Microsoft
#      define DLL_SYM __declspec(dllexport)
#   elif defined(__GNUC__)
       //  GCC
#      define DLL_SYM __attribute__((visibility("default")))
#   else
#      define DLL_SYM
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#else //import dll
#   if defined(_MSC_VER)
       //  Microsoft
#      define DLL_SYM __declspec(dllimport)
#   elif defined(__GNUC__)
       //  GCC
#      define DLL_SYM
#   else
#      define DLL_SYM
#      pragma warning Unknown dynamic link import/export semantics.
#   endif
#endif

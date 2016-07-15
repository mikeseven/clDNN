/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
#ifdef WIN32

#ifndef NOMINMAX
#   define NOMINMAX
#endif

#include <windows.h>
#include <iomanip>
#include <chrono>
#include <sstream>
#include "os_windows.h"

const std::string show_HTML_command("START ");
const std::string dynamic_library_extension(".dll");

void *dlopen(const char *filename, int) {
    return LoadLibraryA(filename);
}

int dlclose(void *handle) {
    return FreeLibrary(static_cast<HINSTANCE>(handle)) == 0 ? 1 : 0;
}

void *dlsym(void *handle, const char *symbol) {
    return GetProcAddress(static_cast<HINSTANCE>(handle), symbol);
}

char *dlerror(void) {
    DWORD errCode = GetLastError();
    char *err;
    if (!FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,
        errCode,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // default language
        (LPTSTR)&err,
        0,
        NULL))
        return nullptr;
    static char buffer[1024];
    _snprintf_s(buffer, sizeof(buffer), "%s", err);
    LocalFree(err);
    return buffer;
}

DIR *opendir(const char *name)
{
    if (!name || !*name) return nullptr;
    else {
        DIR *dir = new DIR;
        dir->name = name;
        dir->name += dir->name.back() == '/' || dir->name.back() == '\\' ? "*" : "/*";
        if (-1 == (dir->handle = _findfirst(dir->name.c_str(), &dir->fileinfo))) {
            delete dir;
            dir = nullptr;
        }
        else dir->result.d_name = nullptr;
        return dir;
    }
}

int closedir(DIR *dir) {
    if (!dir) return -1;
    else {
        intptr_t handle = dir->handle;
        delete dir;
        return -1 == handle ? -1 : _findclose(handle);
    }
}

dirent *readdir(DIR *dir) {
    if (!dir || -1 == dir->handle) return nullptr;
    else {
        if (dir->result.d_name && -1 == _findnext(dir->handle, &dir->fileinfo)) return nullptr;
        else {
            dirent *result = &dir->result;
            result->d_name = dir->fileinfo.name;
            result->d_type = dir->fileinfo.attrib & (_A_ARCH | _A_NORMAL | _A_RDONLY) ? DT_REG : DT_UNKNOWN;
            return result;
        }
    }
}

bool is_regular_file(struct dirent* folder_entry) {
    return folder_entry->d_type == DT_REG;
}

std::string get_timestamp() {
    std::stringstream timestamp;
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    struct tm _tm;
    localtime_s(&_tm, &in_time_t);
    timestamp << std::put_time(&_tm, "%Y%m%d%H%M%S");
    return timestamp.str();
}

#endif // windows
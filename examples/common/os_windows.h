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
#pragma once

#ifdef _WIN32

#include <string>
#include <io.h>
#include <regex>

// dlopen, dlclose, dlsym, dlerror
const int RTLD_LAZY = 0;

// opendir, closedir, readdir
const int DT_UNKNOWN = 0;
const int DT_REG = 8;

extern const std::string show_HTML_command;
extern const std::string dynamic_library_extension;

void *dlopen(const char *filename, int);
int dlclose(void *handle);
void *dlsym(void *handle, const char *symbol);
char *dlerror(void);

struct dirent {
    char           *d_name;
    unsigned char   d_type;
    std::string     str_d_name;
};

struct DIR {
    intptr_t        handle;
    _finddata_t     fileinfo;
    dirent          result;
    std::string     name;
};

DIR *opendir(const char *name);
int closedir(DIR *dir);
dirent *readdir(DIR *dir);
bool is_regular_file(struct dirent* folder_entry);
std::string get_timestamp();

#endif //windows


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

#include "os_windows.h"
#include <string>
// returns list of files (path+filename) from specified directory
std::vector<std::string> get_directory_images(std::string images_path) {
    std::vector<std::string> result;
    if (DIR *folder = opendir(images_path.c_str())) {
        dirent *folder_entry = readdir(folder);
        const auto image_file = std::regex(".*\\.(jpe?g|png|bmp|gif|j2k|jp2|tiff|JPEG|JPG|PNG)$");
        while (folder_entry != nullptr) {
            if (std::regex_match(folder_entry->d_name, image_file) && is_regular_file(folder_entry))
                result.push_back(images_path + "/" + folder_entry->d_name);
            folder_entry = readdir(folder);
        }
        closedir(folder);
    }
    return result;
}
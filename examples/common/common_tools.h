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

#include "api/neural.h"
#include "api/instrumentation.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


/// Information about executable.
class executable_info
{
    std::string _path;
    std::string _file_name_wo_ext;
    std::string _dir;


public:
    /// Gets absolute path to executable.
    const std::string& path() const
    {
        return _path;
    }

    /// Gets executable file name without extension (stem name).
    const std::string& file_name_wo_ext() const
    {
        return _file_name_wo_ext;
    }

    /// Gets aboulte path to executable directory.
    const std::string& dir() const
    {
        return _dir;
    }

    /// Creates new instance of information about current executable.
    ///
    /// @tparam StrTy1_ Type of first string argument (std::string constructible; use for forwarding).
    /// @tparam StrTy2_ Type of second string argument (std::string constructible; use for forwarding).
    /// @tparam StrTy3_ Type of third string argument (std::string constructible; use for forwarding).
    ///
    /// @param path             Absolute path to executable.
    /// @param file_name_wo_ext Executable file name without extension (stem name).
    /// @param dir              Absolute path to executable directory.
    template <typename StrTy1_, typename StrTy2_, typename StrTy3_,
              typename = std::enable_if_t<std::is_constructible<std::string, StrTy1_>::value &&
                                          std::is_constructible<std::string, StrTy2_>::value &&
                                          std::is_constructible<std::string, StrTy3_>::value, void>>
    executable_info(StrTy1_&& path, StrTy2_&& file_name_wo_ext, StrTy3_&& dir)
        : _path(std::forward<StrTy1_>(path)),
          _file_name_wo_ext(std::forward<StrTy2_>(file_name_wo_ext)),
          _dir(std::forward<StrTy3_>(dir))
    {}
};


/// Sets information about executable based on "main"'s command-line arguments.
///
/// It works only once (if successful). Next calls to this function will not modify
/// global executable's information object.
///
/// @param argc Main function arguments count.
/// @param argv Main function argument values.
///
/// @exception std::runtime_error Main function arguments do not contain executable name.
/// @exception boost::filesystem::filesystem_error Cannot compute absolute path to executable.
void set_executable_info(int argc, const char* const argv[]);

/// Gets information about executable.
///
/// Information is fetched only if information was set using set_executable_info() and not yet
/// destroyed (during global destruction). Otherwise, exception is thrown.
///
/// @return Shared pointer pointing to valid executable information.
///
/// @exception std::runtime_error Executable information was not set or it is no longer valid.
std::shared_ptr<const executable_info> get_executable_info();


/// Joins path using native path/directory separator.
///
/// @param parent Parent path.
/// @param child  Child part of path.
///
/// @return Joined path.
std::string join_path(const std::string& parent, const std::string& child);


std::vector<std::string> get_directory_images(const std::string& images_path);
std::vector<std::string> get_directory_weights(const std::string& images_path);
void load_images_from_file_list(const std::vector<std::string>& images_list, neural::primitive& memory); 

void print_profiling_table(std::ostream& os, const std::vector<neural::instrumentation::profiling_info>& profiling_info);
neural::worker create_worker();
uint32_t get_next_nearest_power_of_two(int number);
uint32_t get_gpu_batch_size(int number);
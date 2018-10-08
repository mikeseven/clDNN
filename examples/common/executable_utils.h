// Copyright (c) 2016-2018 Intel Corporation
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

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>


// --------------------------------------------------------------------------------------------------------------------
// Executable utilities.
// --------------------------------------------------------------------------------------------------------------------

namespace cldnn
{
namespace utils
{
namespace examples
{

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

    /// Gets absolute path to executable directory.
    const std::string& dir() const
    {
        return _dir;
    }

    /// Creates new instance of information about current executable.
    ///
    /// @tparam Str1Ty Type of first string argument (std::string constructible; use for forwarding).
    /// @tparam Str2Ty Type of second string argument (std::string constructible; use for forwarding).
    /// @tparam Str3Ty Type of third string argument (std::string constructible; use for forwarding).
    ///
    /// @param path             Absolute path to executable.
    /// @param file_name_wo_ext Executable file name without extension (stem name).
    /// @param dir              Absolute path to executable directory.
    template <typename Str1Ty, typename Str2Ty, typename Str3Ty,
              typename = std::enable_if_t<std::is_constructible<std::string, Str1Ty>::value &&
                                          std::is_constructible<std::string, Str2Ty>::value &&
                                          std::is_constructible<std::string, Str3Ty>::value, void>>
    executable_info(Str1Ty&& path, Str2Ty&& file_name_wo_ext, Str3Ty&& dir)
        : _path(std::forward<Str1Ty>(path)),
          _file_name_wo_ext(std::forward<Str2Ty>(file_name_wo_ext)),
          _dir(std::forward<Str3Ty>(dir))
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

} // namespace examples
} // namespace utils
} // namespace cldnn

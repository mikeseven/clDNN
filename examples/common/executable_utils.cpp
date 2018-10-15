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

#include "executable_utils.h"

#include <boost/filesystem.hpp>

using namespace boost::filesystem;
using namespace cldnn::utils::examples;

namespace cldnn
{
namespace utils
{
namespace examples
{

/// Global weak pointer to executable information.
///
/// Used to detect misuses:
///  * Using get_executable_info() before set_executable_info().
///  * Using get_executable_info() after destruction of global info object (during global destruction).
static std::weak_ptr<const executable_info> exec_info_ptr;


void set_executable_info(const int argc, const char* const argv[])
{
    if (argc <= 0)
        throw std::runtime_error("Arguments of \"main\" function do not contain executable name.");

    const std::string exec_name_arg = argv[0];
    if (exec_name_arg.empty())
        throw std::runtime_error("Arguments of \"main\" function do not contain executable name.");

    auto exec_abs_path = system_complete(exec_name_arg);

    // Safe (guarded call-once) creation of information object.
    static auto info = std::make_shared<executable_info>(
        exec_abs_path.string(), exec_abs_path.stem().string(), exec_abs_path.parent_path().string());
    exec_info_ptr = info;
}

std::shared_ptr<const executable_info> get_executable_info()
{
    auto exec_info = exec_info_ptr.lock();
    if (exec_info == nullptr)
        throw std::runtime_error("Executable information was not set or it is already destroyed.");

    return exec_info; // NRVO
}

} // namespace examples
} // namespace utils
} // namespace cldnn

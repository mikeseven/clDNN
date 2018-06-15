// Copyright (c) 2016, 2018 Intel Corporation
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


#include "api/CPP/memory.hpp"
#include "api/CPP/data.hpp"

#include <string>


namespace file
{
    /// @brief File arguments.
    struct arguments
    {
        /// @brief Creates file arguments.
        ///
        /// @param eng       Engine on which data primitive / memory objects will be created.
        /// @param file_name Name of the file to open / serialize.
        arguments(const cldnn::engine& eng, std::string file_name);

        cldnn::engine                engine;
        std::string                  name;
    };

    /// @brief Reads .nnd file pointed by passed arguments into newly created clDNN memory object.
    ///
    /// @param arg            Arguments describing file location and engine on which returned 
    ///                       memory object will be located.
    /// @param validate_magic Indicates that .nnd file header magic field should be checked.
    /// @return               Memory object with contents of the .nnd file.
    cldnn::memory read(arguments arg, bool validate_magic = false);

    /// @brief Reads .nnd file pointed by passed arguments into newly created clDNN data primitive.
    ///
    /// @param arg            Arguments describing file location and engine on which returned 
    ///                       data primitive will be located.
    /// @param validate_magic Indicates that .nnd file header magic field should be checked.
    /// @return               Data primitive with attached memory object with
    ///                       contents of the .nnd file.
    cldnn::data create(arguments arg, bool validate_magic = false);


    /// @brief Serializes memory into file.
    ///
    /// @param data            Data to write to .nnd file.
    /// @param file_name       Name of .nnd file.
    /// @param old_layout_mode Indicates that old mode of handling .nnd layout format should be used.
    ///                        If @c true, it will return old layout format (shifted properly based on data type
    ///                        used in .nnd).
    void serialize(const cldnn::memory& data, const std::string& file_name, bool old_layout_mode = false);
}

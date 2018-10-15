// Copyright (c) 2018 Intel Corporation
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

#include "common_tools.h"

#include <cstddef>
#include <vector>
#include <fstream>


namespace cldnn
{
namespace utils
{
namespace examples
{
namespace transform
{

/// @brief Output printer for output from transformation-like neural networks.
class output_printer
{
    // Flags.
    /// @brief Indicates that comparison calculation is enabled.
    bool _is_diff_enabled;
    /// @brief Indicates that comparison calculation will also incorporate per-channel comparisons.
    bool _is_diff_per_ch_enabled;
    /// @brief Indicates that comparison calculation will also incorporate histograms.
    bool _is_hist_enabled;

    /// @brief Indicates that output should be written in loss-less format (currently PNG).
    ///
    /// @details Output files names will composed from full original output file name (with extension) with
    ///          additional PNG extension attached.
    bool _lossless_out;

    // Directories needed by output printer.
    /// @brief Directory where report file will be saved.
    std::string _report_dir;
    /// @brief Directory where output image files will be saved.
    std::string _outputs_dir;
    /// @brief Directory where diff image files will be saved (if output_printer::_is_diff_enabled is turned on).
    std::string _diffs_dir;
    /// @brief Directory where red channel diff image files will be saved (if output_printer::_is_diff_enabled
    ///        and output_printer::_is_diff_per_ch_enabled are turned on).
    std::string _diffs_red_dir;
    /// @brief Directory where green channel diff image files will be saved (if output_printer::_is_diff_enabled
    ///        and output_printer::_is_diff_per_ch_enabled are turned on).
    std::string _diffs_green_dir;
    /// @brief Directory where blue channel diff image files will be saved (if output_printer::_is_diff_enabled
    ///        and output_printer::_is_diff_per_ch_enabled are turned on).
    std::string _diffs_blue_dir;
    /// @brief Directory where diff histograms (from black) will be saved (if output_printer::_is_diff_enabled
    ///        and output_printer::_is_hist_enabled are turned on).
    std::string _hists_dir;
    /// @brief Directory where red channel diff histograms will be saved (if output_printer::_is_diff_enabled,
    ///        output_printer::_is_diff_per_ch_enabled and output_printer::_is_hist_enabled are turned on).
    std::string _hists_red_dir;
    /// @brief Directory where green channel diff histograms will be saved (if output_printer::_is_diff_enabled,
    ///        output_printer::_is_diff_per_ch_enabled and output_printer::_is_hist_enabled are turned on).
    std::string _hists_green_dir;
    /// @brief Directory where blue channel diff histograms will be saved (if output_printer::_is_diff_enabled,
    ///        output_printer::_is_diff_per_ch_enabled and output_printer::_is_hist_enabled are turned on).
    std::string _hists_blue_dir;

    /// @brief Report file (output stream).
    std::ofstream _report;

    /// @brief Number of batch group (calculated in a way that makes it start at one).
    std::size_t _batch_group_idx;

public:
    /// @brief Creates new instance of output printer (transformation-like neural networks).
    ///
    /// @param report_file_name_wo_ext Report file stem name (file name without extension).
    /// @param report_title            Part of presented report title (usually used model name).
    /// @param enable_diffs            Indicates that image comparison should be enabled.
    /// @param enable_per_ch_diffs     Indicates that image comparison should also incorporate
    ///                                per-channel image comparison.
    /// @n                             Ignored if @p enable_diffs is @c false.
    /// @param enable_hists            Indicates that image comparison should also incorporate
    ///                                diff histogram calculations.
    /// @n                             Ignored if @p enable_diffs is @c false.
    /// @param output_lossless         Indicates that written output (image) files should be
    ///                                written in loss-less format (currently PNG).
    /// @param report_dir              Optional report / output / diffs / histograms directory.
    /// @n                             If not specified, the default dump directory will be used.
    output_printer(const std::string& report_file_name_wo_ext, const std::string& report_title,
                   bool enable_diffs = true, bool enable_per_ch_diffs = true, bool enable_hists = true,
                   bool output_lossless = true, const std::string& report_dir = "");
    /// @brief Create batch entry in report.
    ///
    /// @param input_file_paths   Input images / files that are part of currently reported batch.
    /// @param img_memory         Memory object containing output images / data (BYXF expected).
    /// @param input_root_dir     Root directory of input images / files (used to calc relative paths).
    /// @param out_print_type     Print / Verbosity type used to print current entry (does not affect
    ///                           report, but affects standard output and returned value).
    /// @param compare_root_dir   Root directory where reference images for comparision are located.
    ///                           If not specified, diff is disabled.
    /// @param diff_threshold     Threshold of pixel difference below which image is treated
    ///                           as matching (for diff pixel frequency).
    /// @param diff_1pc_threshold Threshold of pixel difference below which image is treated
    ///                           as matching (for over 1-percentile diff pixel frequency).
    /// @param diff_5pc_threshold Threshold of pixel difference below which image is treated
    ///                           as matching (for over 5-percentile diff pixel frequency).
    /// @param min_size           Indicates that saved image should be resized to specified
    ///                           minimum size (width or height whichever is lower is changed
    ///                           to this size; aspect ratio is preserved).
    /// @n                        If not specified, the resizing of image is omitted.
    /// @return                   If comparision is turned on, the function returns @c true,
    ///                           if all images have been compared and all returned match.
    /// @n                        Otherwise, @c false is returned.
    bool batch(const std::vector<std::string>& input_file_paths, const memory& img_memory,
               const std::string& input_root_dir,
               print_type out_print_type           = print_type::verbose,
               const std::string& compare_root_dir = "",
               double diff_threshold               = 0.25,
               double diff_1pc_threshold           = 0.05,
               double diff_5pc_threshold           = 0.01,
               unsigned min_size                   = 0);

    // Special functions (disallow copy, allow move).
    output_printer(const output_printer& other) = delete;
    output_printer(output_printer&& other) = default;
    output_printer& operator=(const output_printer& other) = delete;
    output_printer& operator=(output_printer&& other) = default;

    /// @brief Destroys output printer (flushes and closes report file).
    ~output_printer();
};

} // namespace transform
} // namespace examples
} // namespace utils
} // namespace cldnn

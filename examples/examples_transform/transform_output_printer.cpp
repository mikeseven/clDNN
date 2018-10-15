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

#include "transform_output_printer.h"

#include "instrumentation.h"
#include "image_toolkit.h"

#include <api/CPP/memory.hpp>

#include <boost/filesystem.hpp>

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <cerrno>
#include <stdexcept>
#include <system_error>


namespace cldnn
{
namespace utils
{
namespace examples
{
namespace transform
{
// Aliases.
namespace bfs = boost::filesystem;


output_printer::output_printer(const std::string& report_file_name_wo_ext, const std::string& report_title,
                               const bool enable_diffs, const bool enable_per_ch_diffs, const bool enable_hists,
                               const bool output_lossless, const std::string& report_dir)
    : _is_diff_enabled(enable_diffs), _is_diff_per_ch_enabled(enable_diffs && enable_per_ch_diffs),
      _is_hist_enabled(enable_diffs && enable_hists), _lossless_out(output_lossless), _batch_group_idx(1)
{
    _report_dir = report_dir.empty() 
        ? ::instrumentation::logger::get_dumps_dir()
        : report_dir;

    _outputs_dir     = join_path(_report_dir, "outputs");
    _diffs_dir       = join_path(_report_dir, "diffs");
    _diffs_red_dir   = join_path(_report_dir, "diffs_red");
    _diffs_green_dir = join_path(_report_dir, "diffs_green");
    _diffs_blue_dir  = join_path(_report_dir, "diffs_blue");
    _hists_dir       = join_path(_report_dir, "hists");
    _hists_red_dir   = join_path(_report_dir, "hists_red");
    _hists_green_dir = join_path(_report_dir, "hists_green");
    _hists_blue_dir  = join_path(_report_dir, "hists_blue");

    bfs::create_directories(_report_dir);
    bfs::create_directories(_outputs_dir);
    if (_is_diff_enabled)
        bfs::create_directories(_diffs_dir);
    if (_is_diff_per_ch_enabled)
    {
        bfs::create_directories(_diffs_red_dir);
        bfs::create_directories(_diffs_blue_dir);
        bfs::create_directories(_diffs_green_dir);
    }
    if (_is_hist_enabled)
    {
        bfs::create_directories(_hists_dir);
        if (_is_diff_per_ch_enabled)
        {
            bfs::create_directories(_hists_red_dir);
            bfs::create_directories(_hists_blue_dir);
            bfs::create_directories(_hists_green_dir);
        }
    }

    _report.open(join_path(_report_dir, report_file_name_wo_ext + ".html"), std::ios::out | std::ios::trunc);
    if (!_report.is_open())
        throw std::system_error(errno, std::system_category());

    // begin HTML file
    _report << R"###(<?xml version="1.0" encoding="UTF-8"?>

    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN"
    "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
    <html xml:lang="en" xmlns="http://www.w3.org/1999/xhtml"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://www.w3.org/1999/xhtml
                              http://www.w3.org/MarkUp/SCHEMA/xhtml11.xsd">
    <head>
      <meta charset="utf-8"/>
      <title>")###" << report_title << R"###(" results</title>
      <style>
        body {
          font-family: sans-serif;
          background-color: whitesmoke;
        }

        table.recognitions { padding: 0.5em }

        .heading {
          font-family: monospace;
          font-weight: bold;
          font-size: 24pt;
          color: dimgray;
          padding: 3pt 0 1pt 12pt;
          vertical-align: middle;
        }

        .heading-sep {
          border: none;
          height: 2pt;
          color: darkgray;
          background-color: darkgray;
          -moz-box-shadow: 0 1pt 6pt 1pt darkgray;
          -webkit-box-shadow: 0 1pt 6pt 1pt darkgray;
          box-shadow: 0 1pt 6pt 1pt darkgray;
        }

        .trans-items td { padding: 6pt 0 }

        .trans-items-sep {
          border-top: 2pt dashed darkgray;
          border-bottom: 2pt dashed darkgray;
          padding: 3pt;
          color: darkgray;
          text-align: center;
          font-family: monospace;
          font-size: 12pt;
          font-weight: bold;
        }

        .trans-item {
          font-family: monospace;
          vertical-align: top;
          -moz-box-shadow: 5pt 5pt 9pt 1pt black;
          -moz-box-shadow: 5pt 5pt 9pt 1pt rgba(0, 0, 0, 0.79);
          -webkit-box-shadow: 5pt 5pt 9pt 1pt black;
          -webkit-box-shadow: 5pt 5pt 9pt 1pt rgba(0, 0, 0, 0.79);
          box-shadow: 5pt 5pt 9pt 1pt black;
          box-shadow: 5pt 5pt 9pt 1pt rgba(0, 0, 0, 0.79);
        }

        .trans-item td {
          border-collapse: collapse;
          vertical-align: top;
          min-width: 300px;
          width: 33%;
        }

        .trans-item div {
          font-size: 8pt;
          padding: 6pt 12pt;
        }

        .trans-item-head {
          font-size: 12pt;
          font-weight: bold;
          padding: 3pt 0 3pt 6pt;
        }

        .trans-item-info {
          font-size: 10pt;
          vertical-align: top;
          padding: 3pt 6pt;
        }

        .trans-item-info span {
          font-size: 12pt;
          font-weight: bold;
        }

        .trans-item-vsep { border-left: 1pt dashed gray }

        .trans-item-hsep { border-top: 1pt dashed gray }
      </style>
    </head>
    <body>
      <h2 class="heading">Results from ")###" << report_title << R"###(" run</h2>
      <hr class="heading-sep"/>
      <table class="trans-items">
      )###";
}

bool output_printer::batch(const std::vector<std::string>& input_file_paths, const memory& img_memory,
                           const std::string& input_root_dir, const print_type out_print_type,
                           const std::string& compare_root_dir, const double diff_threshold,
                           const double diff_1pc_threshold, const double diff_5pc_threshold, const unsigned min_size)
{
    const std::string lossless_ext = ".png"; // TODO: [FUTURE] Make constexpr when C++17 support will be added.


    // Presentation / comparison flags.
    const auto is_diff_enabled        = _is_diff_enabled && !compare_root_dir.empty();
    const auto is_diff_per_ch_enabled = is_diff_enabled && _is_diff_per_ch_enabled;
    const auto is_hist_enabled        = is_diff_enabled && _is_hist_enabled;

    auto is_batch_matched = is_diff_enabled;

    // Calculate load/save locations.
    const auto ref_file_ld_paths         = rebase_file_paths(input_file_paths, input_root_dir, compare_root_dir);
    auto out_file_sv_paths               = rebase_file_paths(input_file_paths, input_root_dir, _outputs_dir);

    if (_lossless_out)
    {
        for (auto& out_file_sv_path : out_file_sv_paths)
            out_file_sv_path.append(lossless_ext);
    }

    const auto diffs_file_sv_paths       = rebase_file_paths(out_file_sv_paths, _outputs_dir, _diffs_dir);
    const auto diffs_red_file_sv_paths   = rebase_file_paths(out_file_sv_paths, _outputs_dir, _diffs_red_dir);
    const auto diffs_green_file_sv_paths = rebase_file_paths(out_file_sv_paths, _outputs_dir, _diffs_green_dir);
    const auto diffs_blue_file_sv_paths  = rebase_file_paths(out_file_sv_paths, _outputs_dir, _diffs_blue_dir);
    const auto hists_file_sv_paths       = rebase_file_paths(out_file_sv_paths, _outputs_dir, _hists_dir);
    const auto hists_red_file_sv_paths   = rebase_file_paths(out_file_sv_paths, _outputs_dir, _hists_red_dir);
    const auto hists_green_file_sv_paths = rebase_file_paths(out_file_sv_paths, _outputs_dir, _hists_green_dir);
    const auto hists_blue_file_sv_paths  = rebase_file_paths(out_file_sv_paths, _outputs_dir, _hists_blue_dir);

    // Save output images.
    auto out_images = save_image_files(out_file_sv_paths, img_memory, min_size);

    // Printing batch group info.
    _report << R"###(  <tr>
          <td class="trans-items-sep">BATCH #)###" << _batch_group_idx << R"###(</td>
        </tr>
      )###";
    ++_batch_group_idx;

    // Printing batch info.
    for (std::size_t batch_elem_idx = 0; batch_elem_idx < input_file_paths.size(); ++batch_elem_idx)
    {
        // Indicates that reference file exists and diff results are valid.
        auto diff_successful = false;

        // Paths to (image) files: relative to report file.
        const auto in_file_path  = relative(bfs::absolute(input_file_paths[batch_elem_idx]), _report_dir).string();
        const auto out_file_path = relative(bfs::absolute(out_file_sv_paths[batch_elem_idx]), _report_dir).string();
        std::string ref_file_path;
        std::string diffs_file_path;
        std::string diffs_red_file_path;
        std::string diffs_green_file_path;
        std::string diffs_blue_file_path;
        std::string hists_file_path;
        std::string hists_red_file_path;
        std::string hists_green_file_path;
        std::string hists_blue_file_path;

        // File names of (image) files: currently (due to rebasing names are the same for all image types).
        const auto in_file_name           = bfs::path(in_file_path).filename().string();
        const auto out_file_name          = bfs::path(out_file_path).filename().string();
        auto ref_file_name                = out_file_name;
        const auto& diffs_file_name       = out_file_name;
        const auto& diffs_red_file_name   = out_file_name;
        const auto& diffs_green_file_name = out_file_name;
        const auto& diffs_blue_file_name  = out_file_name;
        const auto& hists_file_name       = out_file_name;
        const auto& hists_red_file_name   = out_file_name;
        const auto& hists_green_file_name = out_file_name;
        const auto& hists_blue_file_name  = out_file_name;

        // Input debased paths.
        const auto in_file_dbs_path = relative(bfs::absolute(input_file_paths[batch_elem_idx]),
                                                             input_root_dir).string();

        const auto& out_image = out_images[batch_elem_idx];

        // Calculating diffs / histograms.
        itk::diff_results results;
        if (is_diff_enabled)
        {
            auto ref_file_ld_path  = bfs::path(ref_file_ld_paths[batch_elem_idx]);

            // Possible loss-less file (try to load/diff first).
            auto ref_file_ll_ld_path = ref_file_ld_path;
            ref_file_ll_ld_path.replace_extension(ref_file_ll_ld_path.extension().string() + lossless_ext);

            if (exists(ref_file_ll_ld_path) && is_regular_file(ref_file_ll_ld_path))
            {
                ref_file_path = relative(absolute(ref_file_ll_ld_path), _report_dir).string();
                ref_file_name = bfs::path(ref_file_path).filename().string();

                try
                {
                    auto ref_image = itk::load(ref_file_ll_ld_path.string());
                    results        = itk::diff(out_image, ref_image, _is_diff_per_ch_enabled, _is_hist_enabled);
                }
                catch (const std::runtime_error&) {}

                diff_successful = true;
            }
            if (!diff_successful && exists(ref_file_ld_path) && is_regular_file(ref_file_ld_path))
            {
                ref_file_path = relative(absolute(ref_file_ld_path), _report_dir).string();
                ref_file_name = bfs::path(ref_file_path).filename().string();

                auto ref_image = itk::load(ref_file_ld_path.string());
                results        = itk::diff(out_image, ref_image, _is_diff_per_ch_enabled, _is_hist_enabled);

                diff_successful = true;
            }

            if (results.image)
            {
                auto diffs_file_sv_path = bfs::path(diffs_file_sv_paths[batch_elem_idx]);
                itk::save(results.image, diffs_file_sv_path.string());
                diffs_file_path = relative(absolute(diffs_file_sv_path), _report_dir).string();
            }
            if (results.image_red)
            {
                auto diffs_red_file_sv_path = bfs::path(diffs_red_file_sv_paths[batch_elem_idx]);
                itk::save(results.image_red, diffs_red_file_sv_path.string());
                diffs_red_file_path = relative(absolute(diffs_red_file_sv_path), _report_dir).string();
            }
            if (results.image_green)
            {
                auto diffs_green_file_sv_path = bfs::path(diffs_green_file_sv_paths[batch_elem_idx]);
                itk::save(results.image_green, diffs_green_file_sv_path.string());
                diffs_green_file_path = relative(absolute(diffs_green_file_sv_path), _report_dir).string();
            }
            if (results.image_blue)
            {
                auto diffs_blue_file_sv_path = bfs::path(diffs_blue_file_sv_paths[batch_elem_idx]);
                itk::save(results.image_blue, diffs_blue_file_sv_path.string());
                diffs_blue_file_path = relative(absolute(diffs_blue_file_sv_path), _report_dir).string();
            }

            if (results.histogram)
            {
                auto hists_file_sv_path = bfs::path(hists_file_sv_paths[batch_elem_idx]);
                itk::save(itk::create_histogram(results.histogram), hists_file_sv_path.string());
                hists_file_path = relative(absolute(hists_file_sv_path), _report_dir).string();
            }
            if (results.histogram_red)
            {
                auto hists_red_file_sv_path = bfs::path(hists_red_file_sv_paths[batch_elem_idx]);
                itk::save(itk::create_histogram(results.histogram_red), hists_red_file_sv_path.string());
                hists_red_file_path = relative(absolute(hists_red_file_sv_path), _report_dir).string();
            }
            if (results.histogram_green)
            {
                auto hists_green_file_sv_path = bfs::path(hists_green_file_sv_paths[batch_elem_idx]);
                itk::save(itk::create_histogram(results.histogram_green), hists_green_file_sv_path.string());
                hists_green_file_path = relative(absolute(hists_green_file_sv_path), _report_dir).string();
            }
            if (results.histogram_blue)
            {
                auto hists_blue_file_sv_path = bfs::path(hists_blue_file_sv_paths[batch_elem_idx]);
                itk::save(itk::create_histogram(results.histogram_blue), hists_blue_file_sv_path.string());
                hists_blue_file_path = relative(absolute(hists_blue_file_sv_path), _report_dir).string();
            }
        }

        // Check image for match.
        auto is_image_matched = false;
        if (diff_successful)
        {
            is_image_matched = results.diff_pixel_freq < diff_threshold &&
                results.diff_1pc_pixel_freq < diff_1pc_threshold &&
                results.diff_5pc_pixel_freq < diff_5pc_threshold;
            if (!is_image_matched)
                is_batch_matched = false;
        }
        else
            is_batch_matched = false; // If image cannot be compared, batch is considered as failed to matched.

        switch (out_print_type)
        {
        case print_type::verbose:
            {
                std::cout << " -- \"" << in_file_dbs_path << "\"";
                if (diff_successful)
                {
                    std::cout << std::fixed << std::right << std::setprecision(3)
                        << " (diff: " << std::setw(7) << (results.diff_pixel_freq * 100) << "%,"
                        << " >1%-diff: " << std::setw(7) << (results.diff_1pc_pixel_freq * 100) << "%,"
                        << " >5%-diff: " << std::setw(7) << (results.diff_5pc_pixel_freq * 100) << "%): "
                        << (is_image_matched ? "CORRECT" : "WRONG");
                }
                std::cout << std::endl;
            }
            break;

        case print_type::performance:
            // This printer does not output performance reports.
            break;

        case print_type::extended_testing:
            {
                if (!diff_successful)
                {
                    std::cerr << "WARNING: Extended testing was enabled, but reference images to compare "
                                 "were not provided."
                                 "\n -- Reporting image comparison result with status: ERROR!!!" << std::endl;
                }

                std::cout << " -- \"" << in_file_dbs_path << "\"";
                if (diff_successful)
                {
                    std::cout << std::fixed << std::right << std::setprecision(3)
                        << " (diff: " << std::setw(7) << (results.diff_pixel_freq * 100) << "%,"
                        << " >1%-diff: " << std::setw(7) << (results.diff_1pc_pixel_freq * 100) << "%,"
                        << " >5%-diff: " << std::setw(7) << (results.diff_5pc_pixel_freq * 100) << "%): "
                        << (is_image_matched ? "CORRECT" : "WRONG");
                }
                else
                    std::cout << " (diff:      N/A, >1%-diff:      N/A, >5%-diff:      N/A): ERROR";
                std::cout << std::endl;
            }
            break;

        default:
            throw std::invalid_argument("Unsupported print type.");
        }


        // Writing batch report.
        _report << R"###(  <tr>
          <td>
            <table class="trans-item">
              <tr>
                <td class="trans-item-head" colspan="3">)###" << in_file_dbs_path << R"###(</td>
              </tr>
              <tr>
                <td>
                  <div>
                    input:
                    <br />
                    <a href=")###" << in_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << in_file_name << R"###(" src=")###" << in_file_path << R"###(" />
                    </a>
                  </div>
                </td>
                <td class="trans-item-vsep">
                  <div>
                    output:
                    <br />
                    <a href=")###" << out_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << out_file_name << R"###(" src=")###" << out_file_path << R"###(" />
                    </a>
                  </div>
                </td>
                )###";

        if (is_diff_enabled && diff_successful)
        {
            _report << R"###(<td class="trans-item-vsep">
                  <div>
                    reference:
                    <br />
                    <a href=")###" << ref_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << ref_file_name << R"###(" src=")###" << ref_file_path << R"###(" />
                    </a>
                  </div>
                </td>
              )###";
        }
        else
        {
            _report << R"###(<td/>
              )###";
        }

        _report << R"###(</tr>
              )###";

        if (is_diff_enabled && diff_successful)
        {
            _report
                << std::fixed << R"###(<tr>
                <td class="trans-item-hsep trans-item-info" rowspan="4">
                  <p>
                    Image difference:
                    <ul>
                      <li><a title="sum of absolute difference">SAD:</a> )###"
                << std::setprecision(0) << results.sad << R"###( (<a title="per-pixel mean value of SAD">mean:</a> )###"
                << std::setprecision(3) << results.sad_mean << R"###()</li>
                      <li><a title="sum of squared difference">SSD:</a> )###"
                << std::setprecision(0) << results.ssd << R"###( (<a title="per-pixel mean value of SSD">mean:</a> )###"
                << std::setprecision(3) << results.ssd_mean << R"###()</li>
                      <li><a title="percentage of different pixels">% of diff. pixels:</a>&nbsp;&nbsp;&nbsp;&nbsp; )###"
                << std::setprecision(3) << (results.diff_pixel_freq * 100) << R"###(%</li>
                      <li><a title="percentage of pixels different more than 1% of max difference )###"
                << R"###((above 1-percentile)">% of >1% diff. pixels:</a> )###"
                << std::setprecision(3) << (results.diff_1pc_pixel_freq * 100) << R"###(%</li>
                      <li><a title="percentage of pixels different more than 5% of max difference )###"
                << R"###((above 5-percentile)">% of >5% diff. pixels:</a> )###"
                << std::setprecision(3) << (results.diff_5pc_pixel_freq * 100) << R"###(%</li>
                    </ul>
                    <span>Match: )###" << (is_image_matched ? R"###(<span style="color: green">CORRECT</span>)###"
                                                            : R"###(<span style="color: red">WRONG</span>)###") 
                << R"###(</span>
                  </p>
                </td>
                <td class="trans-item-vsep trans-item-hsep">
                  <div>
                    difference:
                    <br />
                    <a href=")###" << diffs_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << diffs_file_name << R"###(" src=")###"
                << diffs_file_path << R"###(" />
                    </a>
                  </div>
                </td>
                <td class="trans-item-hsep">)###";

            if (is_hist_enabled && diff_successful)
            {
                _report << R"###(
                  <div>
                    histogram:
                    <br />
                    <a href=")###" << hists_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << hists_file_name << R"###(" src=")###"
                    << hists_file_path << R"###(" />
                    </a>
                  </div>)###";
            }

            _report << R"###(
                </td>
              </tr>
              )###";

            if (is_diff_per_ch_enabled && diff_successful)
            {
                _report << R"###(<tr>
                <td class="trans-item-vsep trans-item-hsep">
                  <div>
                    difference (<span style="color: red">red</span>):
                    <br />
                    <a href=")###" << diffs_red_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << diffs_red_file_name << R"###(" src=")###"
                    << diffs_red_file_path << R"###(" />
                    </a>
                  </div>
                </td>
                <td class="trans-item-hsep">)###";

                if (is_hist_enabled && diff_successful)
                {
                    _report << R"###(
                  <div>
                    histogram (<span style="color: red">red</span>):
                    <br />
                    <a href=")###" << hists_red_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << hists_red_file_name << R"###(" src=")###"
                        << hists_red_file_path << R"###(" />
                    </a>
                  </div>)###";
                }

                _report << R"###(
                </td>
              </tr>
              <tr>
                <td class="trans-item-vsep trans-item-hsep">
                  <div>
                    difference (<span style="color: green">green</span>):
                    <br />
                    <a href=")###" << diffs_green_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << diffs_green_file_name << R"###(" src=")###"
                    << diffs_green_file_path << R"###(" />
                    </a>
                  </div>
                </td>
                <td class="trans-item-hsep">)###";

                if (is_hist_enabled && diff_successful)
                {
                    _report << R"###(
                  <div>
                    histogram (<span style="color: green">green</span>):
                    <br />
                    <a href=")###" << hists_green_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << hists_green_file_name << R"###(" src=")###"
                        << hists_green_file_path << R"###(" />
                    </a>
                  </div>)###";
                }

                _report << R"###(
                </td>
              </tr>
              <tr>
                <td class="trans-item-vsep trans-item-hsep">
                  <div>
                    difference (<span style="color: blue">blue</span>):
                    <br />
                    <a href=")###" << diffs_blue_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << diffs_blue_file_name << R"###(" src=")###"
                    << diffs_blue_file_path << R"###(" />
                    </a>
                  </div>
                </td>
                <td class="trans-item-hsep">)###";

                if (is_hist_enabled && diff_successful)
                {
                    _report << R"###(
                  <div>
                    histogram (<span style="color: blue">blue</span>):
                    <br />
                    <a href=")###" << hists_blue_file_path << R"###(" target="_blank">
                      <img height="180" alt=")###" << hists_blue_file_name << R"###(" src=")###"
                        << hists_blue_file_path << R"###(" />
                    </a>
                  </div>)###";
                }

                _report << R"###(
                </td>
              </tr>
            )###";

            }
            else
            {
                _report << R"###(<tr>
                <td/>
                <td/>
              </tr>
              <tr>
                <td/>
                <td/>
              </tr>
              <tr>
                <td/>
                <td/>
              </tr>
            )###";

            }
        }

        _report << R"###(</table>
          </td>
        </tr>
      )###";
    }

    return is_batch_matched;
}

output_printer::~output_printer()
{
    _report << R"###(</table>
    </body>
    </html>)###";

    _report.flush();
    _report.close();
}

} // namespace transform
} // namespace examples
} // namespace utils
} // namespace cldnn

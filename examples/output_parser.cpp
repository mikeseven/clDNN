#include "output_parser.h"
#include "api/CPP/network.hpp"

#include "common/instrumentation.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cerrno>
#include <stdexcept>
#include <system_error>
#include <chrono>
#include <api/CPP/data.hpp>
#include <api/CPP/reorder.hpp>

// returns a vector with size equal to number of images in batch and each subvector contains sorted pairs of 
// values specifying match percentage and category number
std::vector<std::vector<std::pair<float, size_t>>> read_output(const cldnn::memory& input_mem)
{
    // Check format for types that need to be converted to float.
    auto mem_layout = input_mem.get_layout();
    bool needs_data_type_conversion = mem_layout.data_type != cldnn::data_types::f32;

    bool needs_format_conversion = true;
    switch (mem_layout.format)
    {
    case cldnn::format::bfyx:
        needs_format_conversion = true;
        break;
    case cldnn::format::yxfb:
        needs_format_conversion = false;
        break;
    default:
        throw std::invalid_argument("Unsupported format for result parser");
    }

    auto mem = input_mem;
    // Convert format if necessary.
    if(needs_format_conversion || needs_data_type_conversion)
    {
        cldnn::topology topology;
        cldnn::data input_data("input_mem", input_mem);
        topology.add(input_data);
        cldnn::primitive_id curr_id = input_data.id;

        if(needs_data_type_conversion)
        {
            mem_layout.data_type = cldnn::data_types::f32;
            cldnn::reorder data_type_reorder("data_type_reorder", curr_id, mem_layout);
            topology.add(data_type_reorder);
            curr_id = data_type_reorder.id;
        }

        if (needs_format_conversion)
        {
            mem_layout.format = cldnn::format::yxfb;
            cldnn::reorder format_reorder("format_reorder", curr_id, mem_layout);
            topology.add(format_reorder);
            curr_id = format_reorder.id;
        }

        mem = cldnn::network(cldnn::engine(), topology, {cldnn::build_option::outputs({curr_id})}).execute().begin()->second.get_memory();
    }

    auto ptr = mem.pointer<float>();
    size_t image_count = mem.get_layout().size.batch[0];
    size_t category_count = mem.get_layout().size.spatial[0];
    std::vector<std::vector<std::pair<float, size_t>>> ret;
    size_t offset = 0;
    ret.resize(image_count);
    for (auto& v : ret) { v.reserve(category_count); }
    for (size_t c = 0; c < category_count; ++c)
    {
        for (size_t i = 0; i < image_count; ++i)
        {
            ret[i].push_back(std::make_pair(ptr[offset++], c));
        }
    }
    for (auto& v : ret)
    {
        std::sort(
            v.begin(),
            v.end(),
            [](const std::pair<float, size_t>& l, const std::pair<float, size_t>& r)
                { return l.first > r.first; });
    }
    return ret;
}

std::vector<std::string> load_category_names(const std::string & file_name)
{
    std::ifstream file(file_name, std::ios::binary);
    if (file.is_open())
    {
        std::stringstream data;
        data << file.rdbuf();
        file.close();
        std::string tmp;
        std::vector<std::string> ret;
        while (std::getline(data, tmp)) { ret.push_back(tmp); }
        return ret;
    }
    throw std::system_error(errno, std::system_category( ));
}

html::html(const std::string & file_name, const std::string & title)
{
    html_file.open(instrumentation::logger::get_dumps_dir() + "/" + file_name + ".html", std::ios::out | std::ios::trunc);
    if(html_file.is_open( ))
    {
        // begin HTML file
        html_file <<
            R"(<!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8" />
                <title>)"<< title << R"(</title>
                <style>
                    body {font-family: sans-serif;}
                    p, h1, h2, th, td, li {padding: 0.5em;}
                    img {margin: 0.5em 0px 0px 0.5em;}
                    table.recognitions {padding:0.5em;}
                    .thright {text-align:right;}
                    .recognition {font-family:monospace;
                    vertical-align:top;width:300px;
                    -webkit-box-shadow: 5px 5px 9px 1px rgba(0,0,0,0.79);
                    -moz-box-shadow: 5px 5px 9px 1px rgba(0, 0, 0, 0.79);
                    box-shadow: 5px 5px 9px 1px rgba(0, 0, 0, 0.79);}
                    .goal {font-weight: bold; color: red;}
                </style>
            </head>
            <body>)";
        // HTML file content
        html_file << "<h2>" << title << "</h2>"<< std::endl << "<ul> " << "</ul><hr><hr>" << std::endl;
    }
    else
    {
        throw std::system_error(errno, std::system_category( ));
    }
}

void html::batch(const cldnn::memory& mem_primitive, const std::string& categories_file, const std::vector<std::string>& image_names, PrintType printType)
{
    auto batch = read_output(mem_primitive);
    auto categories = load_category_names(categories_file);
    html_file << "<table class=\"recognitions\" width=\"" << 301 * batch.size( ) << "px\"><tr>" << std::endl;
    // Per image
    for (size_t img_idx = 0; img_idx < batch.size( ); img_idx++)
    {
        if (img_idx >= image_names.size())
            break;

        auto& img_name = image_names[img_idx];
        auto idx = img_name.find_last_of("/\\");
    
        bool not_found = idx == std::string::npos;

        auto img_file = img_name.substr(not_found ? 0 : idx + 1);
        auto without_file = img_name.substr(0, not_found ? 0 : idx);

        auto idx1 = without_file.find_last_of("/\\");
        bool not_found1 = idx1 == std::string::npos;
        auto img_dir = without_file.substr(not_found1 ? 0 : idx1 + 1);

        html_file << "<td class=\"recognition\"><img height=\"150\" alt=\""
                  << img_name << "\" src=\"" << img_name << "\">" 
                  << "</p><p>" << img_file << "</p>" << std::endl 
                  << "</b><p>Recognitions:</p><ol>" << std::endl;
        // Top 5 categories
        for (size_t i = 0; i < std::min((size_t)5, batch[img_idx].size( )); i++)
        {
            auto& category = categories[batch[img_idx][i].second];
            std::ostringstream rounded_float;
            rounded_float.str("");
            rounded_float << std::setprecision(1) << std::fixed << batch[img_idx][i].first * 100 <<"% ";
            html_file << "<li>" << rounded_float.str( )    << category << "</li>" << std::endl;
        }

        // for testing enviroment we also output data to std out
        // this should be done on some global config flag set by
        // runtime arguments, so user could have choice between
        // html and textlog
        {
            const auto& category = categories[batch[img_idx][0].second];

            switch(printType)
            {
            case PrintType::ExtendedTesting:
            {
                bool correct = img_dir.compare(category) == 0;
                std::cout //<< "    " 
                          /*<< img_file << " "
                          << std::setprecision(2) << std::fixed << batch[img_idx][0].first * 100 << "%" << " "
                          << category
                          << " and we got it - " << (correct ? "CORRECT" : "wrong")*/
                    << (correct ? "CORRECT" : "wrong")
                    << std::endl;
            }
            break;
            case PrintType::Verbose:
            {
                std::cout << "    " << img_file << " ";
                std::cout << std::setprecision(2) << std::fixed << batch[img_idx][0].first * 100 << "%"<<" ";
                std::cout << category << std::endl;                           
            }
            break;
            default:
                throw std::invalid_argument("Unsupported print type.");
            }
        }
        // table cell end
        html_file << "</ol>" << std::endl << "    </td>";
    }
    // table end
    html_file << "</tr></table><hr>" << std::endl;
}

html::~html()
{
    html_file << std::endl << " </body>" << std::endl << "</html>" << std::endl;
    html_file.close();
}

#pragma once

#include "api/CPP/memory.hpp"
#include <vector>
#include <fstream>
#include "common_tools.h"

struct html
{
    html(const std::string& file_name, const std::string& title);
    void batch(const cldnn::memory& mem_primitive,
               const std::string& categories_file,
               const std::vector<std::string>& image_names,
               cldnn::utils::examples::print_type printType = cldnn::utils::examples::print_type::verbose);
    ~html( );

private:
    std::fstream html_file;
};

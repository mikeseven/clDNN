#pragma once

#include "api/CPP/memory.hpp"
#include <vector>
#include <fstream>
#include "common/common_tools.h"

std::vector<std::vector<std::pair<float, size_t>>> read_output(const cldnn::memory& input_mem);
std::vector<std::string> load_category_names(const std::string & file_name);

struct html
{
    html(const std::string& file_name, const std::string& title);
    void batch(const cldnn::memory& mem_primitive,
               const std::string& categories_file,
               const std::vector<std::string>& image_names,
               PrintType printType = PrintType::Verbose);
    ~html( );

private:
    std::fstream html_file;
};

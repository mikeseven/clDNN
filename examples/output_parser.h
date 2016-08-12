#pragma once

#include "api/neural.h"
#include <vector>
#include <fstream>

namespace neural
{

struct html
{
	html(const std::string& file_name, const std::string& title);
	void batch(const neural::memory & mem, 
			   const std::string& categories_file,
			   const std::vector<std::string>& image_names);
	~html( );
private:
	std::fstream html_file;
};

}

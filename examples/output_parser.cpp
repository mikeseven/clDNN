#include "output_parser.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cerrno>
#include <stdexcept>
#include <system_error>
#include <chrono>

namespace neural
{

// returns a vector with size equal to number of images in batch and each subvector contains sorted pairs of 
// values specifying match percentage and category number
std::vector<std::vector<std::pair<float, size_t>>> read_output(const neural::memory & mem)
{
	auto ptr = mem.pointer<float>();
	size_t image_count = mem.argument.size.batch[0];
	size_t category_count = 0;
	auto format = mem.argument.format; 
	std::vector<std::vector<std::pair<float, size_t>>> ret;
	size_t offset = 0;
	switch (format)
	{
	case neural::memory::format::type::xb_f32:
		category_count = mem.argument.size.spatial[0];
		ret.resize(image_count);
		for (auto& v : ret) { v.reserve(category_count); }
		for (size_t c = 0; c < category_count; ++c)
		{
			for (size_t i = 0; i < image_count; ++i)
			{
				ret[i].push_back(std::make_pair(ptr[offset++], c));
			}
		}
		for (auto& v : ret) { std::sort(v.begin( ), v.end( ), 
										[](const std::pair<float, size_t>& l, const std::pair<float, size_t>& r)
											{ return l.first > r.first; }); }
		break;
	default:
		throw std::invalid_argument("Unsupported format for result parser");
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
	auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now( ));
	html_file.open(file_name + "_" + std::to_string(t) + ".html", std::ios::out | std::ios::trunc);
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

void html::batch(const neural::memory & mem, const std::string& categories_file, const std::vector<std::string>& image_names)
{
	auto batch = read_output(mem);
	auto categories = load_category_names(categories_file);
	html_file << "<table class=\"recognitions\" width=\"" << 301 * batch.size( ) << "px\"><tr>" << std::endl;
	// Per image
	for (size_t img_idx = 0; img_idx < batch.size( ); img_idx++)
	{
		auto& img_name = image_names[img_idx];
		auto idx = img_name.find_last_of("/\\");
		auto img_file = img_name.substr(idx == std::string::npos ? 0 : idx + 1);
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
			html_file << "<li>" << rounded_float.str( )	<< category << "</li>" << std::endl;
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

}

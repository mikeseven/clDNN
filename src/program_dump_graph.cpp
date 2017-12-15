/*
// Copyright (c) 2016 Intel Corporation
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_dump_graph.h"
#include "to_string_utils.h"
#include <algorithm>
#include <vector>

namespace cldnn
{
    namespace
    {
        static const std::vector<std::string> colors =
        {
            "chartreuse",
            "aquamarine",
            "gold",
            "green",
            "blue",
            "cyan",
            "azure",
            "beige",
            "bisque",
            "blanchedalmond",
            "blueviolet",
            "brown",
            "burlywood",
            "cadetblue",
            "chocolate",
            "coral",
            "cornflowerblue",
            "cornsilk",
            "crimson",
            "aliceblue",
            "antiquewhite",
            "deeppink",
            "deepskyblue",
            "dimgray",
            "dimgrey",
            "dodgerblue",
            "firebrick",
            "floralwhite",
            "forestgreen",
            "fuchsia",
            "gainsboro",
            "ghostwhite",
            "goldenrod",
            "gray",
            "greenyellow",
            "honeydew",
            "hotpink",
            "indianred",
            "indigo",
            "ivory",
            "khaki",
            "lavender",
            "lavenderblush",
            "lawngreen",
            "lemonchiffon",
            "lightblue",
            "lightcoral",
            "lightcyan",
            "lightgoldenrodyellow",
            "lightgray",
            "lightgreen",
            "lightgrey",
            "lightpink",
            "lightsalmon",
            "lightseagreen",
            "lightskyblue",
            "lightslategray",
            "lightslategrey",
            "lightsteelblue",
            "lightyellow",
            "lime",
            "limegreen",
            "linen",
            "magenta",
            "maroon",
            "mediumaquamarine",
            "mediumblue",
            "mediumorchid",
            "mediumpurple",
            "mediumseagreen",
            "mediumslateblue",
            "mediumspringgreen",
            "mediumturquoise",
            "mediumvioletred",
            "midnightblue",
            "mintcream",
            "mistyrose",
            "moccasin",
            "navajowhite",
            "navy",
            "oldlace",
            "olive",
            "olivedrab",
            "orange",
            "orangered",
            "orchid",
            "palegoldenrod",
            "palegreen",
            "paleturquoise",
            "palevioletred",
            "papayawhip",
            "peachpuff",
            "peru",
            "pink",
            "plum",
            "powderblue",
            "purple",
            "red",
            "rosybrown",
            "royalblue",
            "saddlebrown",
            "salmon",
            "sandybrown",
            "seagreen",
            "seashell",
            "sienna",
            "silver",
            "skyblue",
            "slateblue",
            "slategray",
            "slategrey",
            "snow",
            "springgreen",
            "steelblue",
            "tan",
            "teal",
            "thistle",
            "tomato",
            "turquoise",
            "violet",
            "wheat",
            "white",
            "whitesmoke",
            "yellow",
            "yellowgreen",
            "darkblue",
            "darkcyan",
            "darkgoldenrod",
            "darkgray",
            "darkgreen",
            "darkgrey",
            "darkkhaki",
            "darkmagenta",
            "darkolivegreen",
            "darkorange",
            "darkorchid",
            "darkred",
            "darksalmon",
            "darkseagreen",
            "darkslateblue",
            "darkslategray",
            "darkslategrey",
            "darkturquoise",
            "darkviolet",
        };


    void close_stream(std::ofstream& graph)
    {
        graph.close();
    }

    std::string get_node_id(program_node* ptr)
    {
        return "node_" + std::to_string(reinterpret_cast<uintptr_t>(ptr));
    }

    void dump_full_node(std::ofstream& out, program_node* node)
    {
        out << node->type()->to_string(*node);
    }
}

    std::string get_dir_path(build_options opts)
    {
        auto path = opts.get<build_option_type::graph_dumps_dir>()->directory_path;
        if (path.empty())
        {
            return{};
        }

        if (path.back() != '/' && path.back() != '\\')
        {
            path += "/";
        }
        return path;
    }

    void dump_graph_init(std::ofstream& graph, const program_impl& program, std::function<bool(program_node const&)> const& filter)
    {
        const auto extr_oformat = [](program_node* ptr)
        {
            std::string out = "";
            switch (ptr->get_output_layout().format)
            {
            case format::yxfb: out = "yxfb"; break;
            case format::byxf: out = "byxf"; break;
            case format::bfyx: out = "bfyx"; break;
            case format::fyxb: out = "fyxb"; break;
            case format::os_iyx_osv16: out = "os_iyx_osv16"; break;
            case format::bs_xs_xsv8_bsv8: out = "bs_xs_xsv8_bsv8"; break;
            case format::bs_xs_xsv8_bsv16: out = "bs_xs_xsv8_bsv16"; break;
            case format::bs_x_bsv16: out = "bs_x_bsv16"; break;
            case format::bf8_xy16: out = "bf8_xy16"; break;
            case format::any: out = "any"; break;
            default:
                out = "unk format";
                break;
            }

            if (!ptr->is_valid_output_layout())
                out += " (invalid)";

            return out;
        };

        graph << "digraph cldnn_program {\n";
        for (auto& node : program.get_nodes())
        {
            if (filter && !filter(*node))
            {
                continue;
            }
            #ifdef __clang__
                #pragma clang diagnostic push
                #pragma clang diagnostic ignored "-Wpotentially-evaluated-expression"
            #endif
            graph << "    " << get_node_id(node.get()) << "[label=\"" << node->id() << ":\n" << get_extr_type(typeid(*node).name()) << "\n out format: " + extr_oformat(node.get())
                << "\\nprocessing number: "<< node->get_processing_num() << (node->can_be_optimized() ? "\\n optimized out" : "") << "\"";
            #ifdef __clang__
                #pragma clang diagnostic pop
            #endif

            if (node->is_type<data>() || node->is_constant())
                graph << ", shape=box";
            if (node->is_type<internal_primitive>())
                graph << ", color=blue";
            if (node->is_in_data_flow())
                graph << ", group=data_flow";
            if (node->is_reusing_memory())
            {
                graph << ", fillcolor=\"" << colors[node->get_reused_memory_color() % colors.size()] << "\" ";
                graph << " style=filled ";
            }
            graph << "];\n";

            for (auto& user : node->get_users())
            {
                if (filter && !filter(*user))
                {
                    continue;
                }
                bool doubled = true;
                if (std::find(user->get_dependencies().begin(), user->get_dependencies().end(), node.get()) == user->get_dependencies().end())
                    doubled = false;

                graph << "    " << get_node_id(node.get()) << " -> " << get_node_id(user);

                bool data_flow = node->is_in_data_flow() && user->is_in_data_flow();
                if (data_flow)
                {
                    if (doubled)
                        graph << " [color=red]";
                    else
                        graph << " [color=red, style=dashed, label=\"usr\"]";
                }
                else
                {
                    if (!doubled)
                        graph << " [style=dashed, label=\"usr\"]";
                }
                graph << ";\n";
            }

            for (auto& dep : node->get_dependencies())
            {
                if (filter && !filter(*dep))
                {
                    continue;
                }

                if (std::find(dep->get_users().begin(), dep->get_users().end(), node.get()) != dep->get_users().end())
                {
                    continue;
                }

                graph << "   " << get_node_id(node.get()) << " -> " << get_node_id(dep) << " [style=dashed, label=\"dep\", constraint=false];\n";
            }

            if (node->get_dominator() && (!filter || filter(*node->get_dominator())))
                graph << "    " << get_node_id(node.get()) << " -> " << get_node_id(node->get_dominator()) << " [style=dotted, label=\"dom\", constraint=false];\n";
            if (node->get_joint() && (!filter || filter(*node->get_joint())))
                graph << "    " << get_node_id(node.get()) << " -> " << get_node_id(node->get_joint()) << " [style=dotted, label=\"p-dom\", constraint=false];\n";
        }
        graph << "}\n";
        close_stream(graph);
    }


    void dump_graph_processing_order(std::ofstream& graph, const program_impl& program)
    { 
        for (auto node : program.get_processing_order())
            graph << reinterpret_cast<uintptr_t>(node) << " (" << node->id() << ")\n";
        graph << '\n';
        close_stream(graph);
    }

    void dump_graph_optimized(std::ofstream& graph, const program_impl& program)
    {
        for (auto& prim_id : program.get_optimized_out())
            graph << prim_id << "\n";
        graph << '\n';
        close_stream(graph);
    }

    void dump_graph_info(std::ofstream& graph, const program_impl& program, std::function<bool(program_node const&)> const& filter)
    {
        for (auto& node : program.get_nodes())
        {
            if (filter && !filter(*node))
                continue;

            dump_full_node(graph, node.get());
            graph << std::endl << std::endl;
        }
        close_stream(graph);
    }
}

 
/*
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
*/

#include "helpers.h"

void helper::mini_batch::pad(size_t size)
{
    while (_words.size() < size)
    {
        _words.push_back(_unk.second);
        _values.push_back(_pad.first);
    }
}

void helper::mini_batch::append_to_beam(const cldnn::memory & attention_output, const cldnn::memory & arg_max_output, const cldnn::memory & best_scores)
{
    _beam.add_results(attention_output, arg_max_output, best_scores);
    search_for_eos_token();
    if (_is_eos_top)
    {
        calc_prediction();
    }
}

void helper::mini_batch::set_values(const std::vector<std::string>& vocab)
{
    for (auto const& word : _words)
    {
        auto it = std::find(
            vocab.begin(),
            vocab.end(),
            word
        );
        if (it != vocab.end())
        {
            _values.push_back(static_cast<uint32_t>(std::distance(
                vocab.begin(), it)));
        }
        else
        {
            _values.push_back(_unk.first);
        }

    }
}

void helper::mini_batch::search_for_eos_token()
{
    auto last_dec_output = _beam.get_next_ys().back();
    for (size_t i = 0; i < last_dec_output.size(); i++)
    {
        if (last_dec_output.at(i) == 3) // this "3" stand for eos token
        {
            _finished.push_back({ _beam.get_scores().back().at(i),{ _beam.get_next_ys().size() - 1, i } });
        }
    }
    if (last_dec_output.at(0) == 3) //eos if first. finish 
    {
        _is_eos_top = true;
        return;
    }
    return;
}

void helper::mini_batch::calc_prediction()
{
    auto replace_special_char = [](const std::string& src)
    {
        if (src.find(";"))
        {
            if (src == "&quot;")
            {
                return std::string("\"");
            }
            auto suffix = src.substr(src.find(";") + 1);
            std::string prefix = "";
            if (src.find("&apos;") != std::string::npos)
            {
                prefix = "'";
            }
            else if (src.find("&amp;") != std::string::npos)
            {
                prefix = "&";
            }
            return prefix + suffix;
        }
        return src;
    };
    from_beam();

    std::string separator = "";
    for (size_t i = 0; i <_predictions.size(); i++)
    {
        auto const pred = _predictions.at(i);
        std::string word = "";
        if (pred == 3) // this "3" stand for eos token
        {
            continue;
        }
        if (pred == 0) // this "0" stands for unk 
        {
            auto max_index = std::distance(_attn.at(i).begin(), std::max_element(_attn.at(i).begin(), _attn.at(i).end()));
            word = _words.at(max_index);
        }
        else
        {
            word = replace_special_char(_tgt_vocab.at(pred));
        }

        _prediction_str += separator + word;
        separator = " ";
    }
}

void helper::mini_batch::from_beam()
{
    auto const& next_ys = _beam.get_next_ys();
    auto const& prev_ys = _beam.get_prev_ys();
    auto const& attns = _beam.get_attns();
    auto const& times = _finished.at(0).second.first;
    auto const& seq_len_of_attn = attns.at(0).size() / _beam.get_beam().size();

    auto k = _finished.at(0).second.second;
    for (size_t i = times; i-- > 0;)
    {
        _predictions.push_back(next_ys.at(i + 1).at(k));
        _attn.push_back({ attns.at(i).begin() + k * 5, attns.at(i).begin() + k * 5 + seq_len_of_attn });
        k = prev_ys.at(i).at(k);
    }
    std::reverse(_predictions.begin(), _predictions.end());
    std::reverse(_attn.begin(), _attn.end());
}

std::vector<float> helper::beam_search::memory_to_vector(const cldnn::memory & mem)
{
    auto ptr = mem.pointer<float>();

    std::vector<float> values;
    for (size_t i = 0; i < mem.count(); i++)
    {
        values.push_back(ptr[i]);
    }
    return values;
}

void helper::beam_search::add_prev_ys_next_ys(const cldnn::memory & mem)
{
    auto indexes = memory_to_vector(mem);
    std::vector<uint32_t> prev_y(mem.count());
    for (size_t i = 0; i < mem.count(); i++)
    {
        prev_y.at(i) = static_cast<uint32_t>(indexes.at(i)) / 24725;
    }
    _prev_ys.push_back(prev_y);

    std::vector<uint32_t> next_ys(mem.count());
    for (size_t i = 0; i < mem.count(); i++)
    {
        next_ys.at(i) = static_cast<uint32_t>(indexes.at(i)) - prev_y.at(i) * 24725;
    }
    _next_ys.push_back(next_ys);
}

void helper::beam_search::add_results(const cldnn::memory & attn, const cldnn::memory & arg_max_output, const cldnn::memory & best_scores)
{
    _attns.push_back(std::move(memory_to_vector(attn)));
    _scores.push_back(std::move(memory_to_vector(best_scores)));
    add_prev_ys_next_ys(arg_max_output);
}


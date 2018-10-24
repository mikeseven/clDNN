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

#include "lstm_utils.h"
#include <fstream>
#include <string>
#include <regex>
#include "helpers.h"

void nmt_utils::pre_process(const std::vector<std::string>& input_files)
{
    auto input_file = input_files.at(0);
    load_input_sequence(input_file);
}

std::string nmt_utils::get_predictions()
{
    std::string ret;
    for (auto const& mb : _mini_batches)
    {
        ret += mb->get_translation() + " \n";
    }
    return ret;
}

std::vector<float> nmt_utils::get_input_to_iteration(const uint32_t seq_len, const size_t batch_nr)
{
    std::vector<float> ret;
    auto& current_beam_last_ys = _packed_mini_batches.at(seq_len).get_mini_batches().at(batch_nr)->get_last_next_ys();
    for (size_t i = 0; i < current_beam_last_ys.size(); i++)
    {
        ret.push_back(static_cast<float>(current_beam_last_ys.at(i)));
    }
    return ret;
}

std::vector<std::string> nmt_utils::load_vocabulary_file(const std::string & vocab_file)
{
    std::ifstream label_file(vocab_file.c_str());
    std::string line;
    std::vector<std::string> ret;
    while (std::getline(label_file, line))
    {
        ret.push_back(line);
    }
    return ret;
}

void nmt_utils::load_input_sequence(const std::string & file)
{
    std::string sentence_end_characters = ".!?"; //also we can add here other characters "<>,:"
    std::ifstream inp_file;
    inp_file.open(file);
    std::string word;
    std::string mini_batch;
    while (std::getline(inp_file, mini_batch))
    {
        //[0] To lowercases.
        std::transform(mini_batch.begin(), mini_batch.end(), mini_batch.begin(), ::tolower);  
        //[1] Split sentences by delimeters.
        std::regex regex_rule("([.,;-]|[^.,;-]+)");
        std::regex_iterator<std::string::iterator> it_start(mini_batch.begin(), mini_batch.end(), regex_rule);
        std::regex_iterator<std::string::iterator> it_end;
        std::string transfomed_mini_batch;
        while (it_start != it_end)
        {
            transfomed_mini_batch += it_start->str() + " ";
            ++it_start;
        }
        //[2] Create and add mini batch object.
        _mini_batches.push_back(std::make_shared<helper::mini_batch>(transfomed_mini_batch, _src_vocab, _tgt_vocab, _pad, _sos));
    }

    //Create packed mini btaches - they are packed by sequnce length (we are creating batches here).
    for (auto const& mb : _mini_batches)
    {
        if (_packed_mini_batches.find(mb->size()) != _packed_mini_batches.end())
        {
            _packed_mini_batches.at(mb->size()).push_mini_batch(mb);
        }
        else
        {
            _packed_mini_batches.insert({ mb->size(), helper::batch(static_cast<uint32_t>(mb->size()),{ mb }) });
        }
    }
    _max_seq_len = (--_packed_mini_batches.end())->first; //get max seq len
}

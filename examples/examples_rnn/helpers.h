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


#pragma once
#include <string>
#include <vector>
#include <deque>
#include <random>
#include <api/CPP/memory.hpp>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iterator>


namespace helper
{

    class beam_search
    {
    private:
        size_t _size;
        std::vector<uint32_t> _beam_vector;
        std::vector<std::vector<float>> _attns;
        std::vector<std::vector<float>> _scores;
        std::vector<std::vector<uint32_t>> _prev_ys;
        std::vector<std::vector<uint32_t>> _next_ys;
        uint32_t _beam_size;
        uint32_t _word_prob_size;

        std::vector<float> memory_to_vector(const cldnn::memory& mem);
        void add_prev_ys_next_ys(const cldnn::memory& mem);

    public:
        beam_search(const size_t s = 0, const uint32_t beg = 0, const uint32_t pad = 0) : _size(s) //add size of tgt_vocab = 24725
        {
            _beam_vector.push_back(beg);
            for (size_t i = 1; i < s; i++)
            {
                _beam_vector.push_back(pad);
            }
            _next_ys.push_back(_beam_vector);
        }

        beam_search& operator()(const beam_search& rhs) //after refactor remove this
        {
            _size = rhs._size;
            _beam_vector = rhs._beam_vector;
            _next_ys = _next_ys;
        }

        std::vector<uint32_t> get_beam() const { return _beam_vector; }

        std::vector<std::vector<uint32_t>> get_next_ys() const { return _next_ys; }

        std::vector<std::vector<uint32_t>> get_prev_ys() const { return _prev_ys; }

        std::vector<std::vector<float>> get_scores() const { return _scores; }

        std::vector<std::vector<float>> get_attns() const { return _attns; }

        void add_results(const cldnn::memory& attn, const cldnn::memory& arg_max_output, const cldnn::memory& best_scores);

    };

    class mini_batch
    {
    public:
        mini_batch(const std::string& sentence, const std::vector<std::string>& src_vocab, const std::vector<std::string>& tgt_vocab,
            const std::pair<uint32_t, std::string>& unk, const std::pair<uint32_t, std::string>& pad,
            const std::pair<uint32_t, std::string>& sos, const std::pair<uint32_t, std::string>& eos)
            : _beam(5, sos.first, pad.first)
            , _tgt_vocab(tgt_vocab)
        {
            std::istringstream iss(sentence);
            _words = {
                std::istream_iterator<std::string>{iss},
                std::istream_iterator<std::string>()
            };
            set_values(src_vocab);
        }

        void pad(size_t size);

        size_t size() const { return _words.size(); }
        auto get_values() const { return _values; }
        auto get_value(size_t idx) { return _values.at(idx); }
        auto get_words() const { return _words; }
        auto get_word(size_t idx) const { return _words.at(idx); }

        void set_result(const std::vector<uint32_t>& result) { _result = result; }
        auto get_result() { return _result; }

        void append_to_beam(const cldnn::memory& attention_output, const cldnn::memory& arg_max_output, const cldnn::memory& best_scores);

        std::string get_translation() { return _prediction_str; }
        bool eos_top() { return _is_eos_top; }
        auto get_last_next_ys() { return _beam.get_next_ys().back(); }
        auto get_last_prev_ys() { return _beam.get_prev_ys().back(); }
    private:
        std::vector<std::string> _words;
        std::vector<uint32_t> _values;
        std::pair<uint32_t, std::string> _unk; //unknown character
        std::pair<uint32_t, std::string> _pad; //padding character
        std::pair<uint32_t, std::string> _sos; //start of sentence
        std::pair<uint32_t, std::string> _eos; //end of sentence
        std::string _prediction_str;
        std::vector<uint32_t> _predictions;
        std::vector<std::vector<float>> _attn;
        std::vector<uint32_t> _result;
        beam_search _beam; //every sentence (minibatch) has its own beam search
        std::vector<std::pair<float, std::pair<size_t, size_t>>> _finished;
        std::vector<std::string> _tgt_vocab;
        bool _is_eos_top = false; //do we found the eos token?

        void set_values(const std::vector<std::string>& vocab);
        void search_for_eos_token();
        void calc_prediction();
        void from_beam();

    };

    class batch
    {
    public:
        batch(uint32_t seq_len, const std::vector<std::shared_ptr<mini_batch>>& init_mb)
            : _seq_length(seq_len)
            , _mini_batches(init_mb)
        {}

        auto get_batch_size() const { return _mini_batches.size(); }

        void push_mini_batch(std::shared_ptr<mini_batch> mb) { _mini_batches.push_back(mb); }

        auto get_mini_batches() const { return _mini_batches; }

        auto get_input_values() const
        {
            std::vector<uint32_t> ret;

            for (auto const& mb : _mini_batches)
            {
                auto& values = mb->get_values();
                ret.insert(ret.end(), values.begin(), values.end());
            }
            return ret;
        }

    private:
        uint32_t _seq_length;
        std::vector<std::shared_ptr<mini_batch>> _mini_batches;
    };
}
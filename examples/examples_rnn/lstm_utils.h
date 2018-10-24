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


#pragma once
#include <string>
#include <vector>
#include <deque>
#include <random>
#include <api/CPP/memory.hpp>
#include <api/CPP/network.hpp>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iterator>

#include "helpers.h"

//itnerface for all RNN-type models
class rnn_base
{
public:
    virtual void pre_process(const std::vector<std::string>& input_files) = 0;
    virtual std::string get_predictions() = 0;
    virtual std::vector<std::pair<size_t, size_t>> batches_and_seq_lens() = 0;
    virtual size_t get_batch_size() = 0;
    virtual void fill_memory(const cldnn::memory& mem, size_t offset = 0) //no tempalte so we can override
    {
        if (mem.get_layout().data_type == cldnn::data_types::f16)
            fill_mem_impl<half_t>(mem, offset);        
        if (mem.get_layout().data_type == cldnn::data_types::f32)
            fill_mem_impl<float>(mem, offset);
    }

protected:
    std::vector<float> _input_vector; //this is actual input_data

private:

    template<typename T>
    void fill_mem_impl(const cldnn::memory& mem, size_t offset = 0)
    {
        check_memory<T>(mem);
        auto ptr = mem.pointer<T>();
        auto it = ptr.begin();
        std::vector<T> input_values(_input_vector.begin() + offset, _input_vector.begin() + offset + mem.get_layout().size.batch[0] * mem.get_layout().size.spatial[0]);

        for (auto const& x : input_values)
        {
            *it++ = (T)x;
        }
    }

    template<typename MemElemTy>
    void check_memory(const cldnn::memory& mem)
    {
        auto memory_layout = mem.get_layout();
        if (memory_layout.format != cldnn::format::bfyx) throw std::runtime_error("Only bfyx format is supported as input for text files.");

        if (!cldnn::data_type_match<MemElemTy>(memory_layout.data_type))
            throw std::runtime_error("Memory format expects different type of elements than specified");
    }
};

class char_rnn_utils : public rnn_base
{
public:
    char_rnn_utils(uint32_t seq_len, uint32_t loop, float temperature, const std::string& vocab_file, bool random_seed = false)
        : _seq_length(seq_len)
        , _loop(loop)
        , _temperature(temperature)
        , _random_engine(std::mt19937(random_seed ? (unsigned int)(std::chrono::high_resolution_clock::now().time_since_epoch().count()) : 42)) //42 as a magic seed
    {
        load_vocabulary_file(vocab_file);
    }
    void pre_process(const std::vector<std::string>& input_files) override;
    size_t get_batch_size() override { return _batch; }
    std::string get_predictions() override;
    void post_process_output(const cldnn::memory& memory);
    std::vector<std::pair<size_t, size_t>> batches_and_seq_lens() override { return { {_batch, _seq_length} }; };
private:
    uint32_t _seq_length;
    uint32_t _loop;
    size_t _batch;
    float _temperature;
    std::mt19937 _random_engine;
    std::vector<char> _int_to_char;
    std::vector<char> _correct_length_sequence; // proper sized input sequence
    std::vector<char> _input_sequence; //sequence from input file
    std::vector<std::vector<std::pair<char, float>>> _predictions;
    std::vector<std::vector<char>> _seed;
    std::deque<int> _input; //represent current processed input sequence
    std::vector<std::vector<float>> _output_vector;

    void load_input_sequence(const std::vector<std::string>& input_file);
    void prepare_processing_input_vector();
    void load_vocabulary_file(const std::string& vocab_file);
    void prepare_input_vector_for_new_prediction();
    std::pair<size_t, float> get_predicted_character_and_its_value(size_t batch_index);
    char random_char();

    template<typename MemElemTy = float>
    void prepare_output_vector(const cldnn::memory& memory)
    {
        auto ptr = memory.pointer<MemElemTy>();
        _output_vector.clear();
        _output_vector.resize(_batch);
        for (size_t i = 0; i < ptr.size(); i++)
        {
            size_t idx = i % _batch;
            _output_vector.at(idx).push_back(ptr[i]);
        }
    }
};

class nmt_utils : public rnn_base
{

public:
    nmt_utils(const std::string& src_vocab_file, const std::string& tgt_vocab_file, int32_t beam_size, int32_t embedding_size)
        : _src_vocab(load_vocabulary_file(src_vocab_file))
        , _tgt_vocab(load_vocabulary_file(tgt_vocab_file))
        , _beam_size(beam_size)
        , _embedding_size(embedding_size)
    {
        _unk = { 0, _src_vocab.at(0) };
        _pad = { 1, _src_vocab.at(1) };
        _sos = { 2, _tgt_vocab.at(2) };
        _eos = { 3, _tgt_vocab.at(3) };
    }

    std::string get_predictions() override;

    void pre_process(const std::vector<std::string>& input_files) override;
    
    std::vector<std::pair<size_t, size_t>> batches_and_seq_lens() override 
    {
        std::vector<std::pair<size_t, size_t>> ret;
        for (const auto& mb : _packed_mini_batches)
        {
            ret.emplace_back(mb.second.get_batch_size(), mb.first);
        }
        return ret;
    }

    size_t get_batch_size() override { 
        size_t ret = 0;
        for (const auto& mb : _packed_mini_batches)
        {
            ret += mb.second.get_batch_size();
        }
        return ret;
    }

    size_t get_max_seq_len() const { return _max_seq_len; }

    std::vector<float> get_input_to_iteration(const uint32_t seq_len, const size_t batch_nr); //needs to convert to float, sine it will be input to next interation

    bool found_eos_token(const uint32_t seq_len, const size_t batch_nr)
    {
        return _packed_mini_batches.at(seq_len).get_mini_batches().at(batch_nr)->eos_top();
    }

    void add_attention_output(const uint32_t seq_len, const size_t batch_nr, const cldnn::memory& attention_output, const cldnn::memory& arg_max_output, const cldnn::memory& best_scores)
    {
        _packed_mini_batches.at(seq_len).get_mini_batches().at(batch_nr)->append_to_beam(attention_output, arg_max_output, best_scores);
    }

    auto get_batches() const { return _packed_mini_batches; }

    auto get_batch_sizes() const
    {
        std::vector<size_t> ret;
        for (auto const& b : _packed_mini_batches)
        {
            ret.push_back(b.second.get_batch_size());
        }
        return ret;
    }

    auto get_seq_and_batch()  const
    {
        std::vector<std::pair<int32_t, int32_t>> ret;
        for (auto const& b : _packed_mini_batches)
        {
            ret.push_back({ static_cast<int32_t>(b.first), static_cast<int32_t>(b.second.get_batch_size()) });
        }
        return ret;
    }

    void fill_memory(const cldnn::memory& mem, size_t seq_len = 0) override
    {
        if (mem.get_layout().data_type == cldnn::data_types::f16)
            fill_mem_impl<half_t>(mem, seq_len);
        if (mem.get_layout().data_type == cldnn::data_types::f32)
            fill_mem_impl<float>(mem, seq_len);
    }

    int32_t get_beam_size() { return _beam_size; }

    int32_t get_embedding_size() { return _embedding_size; }

private:
    std::vector<std::string> _src_vocab;
    std::vector<std::string> _tgt_vocab;
    int32_t _beam_size = 0;
    int32_t _embedding_size = 0;
    std::vector<std::shared_ptr<helper::mini_batch>> _mini_batches; //original ordered sentences
    std::map<size_t, helper::batch> _packed_mini_batches; //packed mini_batches by seq_len
    std::pair<uint32_t, std::string> _unk; //unknown characterl
    std::pair<uint32_t, std::string> _pad; //padding character
    std::pair<uint32_t, std::string> _sos; //start of sentence
    std::pair<uint32_t, std::string> _eos; //end of sentence
    size_t _max_seq_len = 0;

    std::vector<std::string> load_vocabulary_file(const std::string& vocab_file);

    void load_input_sequence(const std::string& file);

    template<typename T>
    void fill_mem_impl(const cldnn::memory& mem, size_t seq_len = 0)
    {
        auto ptr = mem.pointer<T>();
        auto it = ptr.begin();
        auto const& input_values = _packed_mini_batches.at(seq_len).get_input_values();
        for (auto const& x : input_values)
        {
            *it++ = (T)x;
        }
    }

};
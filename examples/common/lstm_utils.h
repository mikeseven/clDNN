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
#include <chrono>

class lstm_utils
{
public:
    lstm_utils(uint32_t seq_len, uint32_t batch, uint32_t predictions, float temperature, bool random_seed = false)
        : _seq_length(seq_len)
        , _batch(batch)
        , _predictions_count(predictions)
        , _temperature(temperature)
        , random_engine(std::mt19937( random_seed ? (unsigned int)(std::chrono::high_resolution_clock::now().time_since_epoch().count()) : 42)) //42 as a magic seed
    {}

    lstm_utils()
        : _seq_length(0)
        , _batch(0)
        , _predictions_count(0)
        , _temperature(0)
        , random_engine (std::mt19937(0))
    {}

    void pre_processing(std::vector<std::string> input_files, const std::string& vocab_file);
    std::string get_predictions(size_t batch_index, bool validation_print_type=false)
    { 
        std::string seed = "";
        if (batch_index < _seed.size())
        {
            seed = std::string(_seed.at(batch_index).begin(), _seed.at(batch_index).end());
        }
        std::string predicted_text;
        std::string seperator = validation_print_type ? "\n" : "";
        std::for_each(_predictions.at(batch_index).begin(), _predictions.at(batch_index).end(), [&](std::pair<char, float> predi)
            { 
                if (!validation_print_type)
                {
                    predicted_text.push_back(predi.first);
                }
                else
                {
                    predicted_text.append(std::to_string(predi.first) + " " + std::to_string(predi.second) + " \n");
                }
            });
        return seed + seperator + predicted_text;
    }

    void post_process_output(const cldnn::memory& memory)
    {
        if (memory.get_layout().data_type == cldnn::data_types::f32)
        {
            prepare_output_vector<float>(memory);
        }
        else  if (memory.get_layout().data_type == cldnn::data_types::f16)
        {
            prepare_output_vector<half_t>(memory);
        }
        for (size_t b = 0; b < _batch; ++b)
        {
            auto& predct_vector_batch = _predictions.at(b);
            auto prediction = get_predicted_character_and_its_value(b);
            size_t start_idx_in_input = b*_seq_length;
            size_t end_idx_in_input = start_idx_in_input + _seq_length;
            std::rotate(_input.begin() + start_idx_in_input, _input.begin() + start_idx_in_input + 1, _input.begin() + end_idx_in_input);
            _input.at(end_idx_in_input - 1) = static_cast<int32_t>(prediction.first);
            predct_vector_batch.push_back({_int_to_char[prediction.first], prediction.second});
        }
        prepare_input_vector_for_new_prediction();
    }

    template<typename T>
    void fill_memory(const cldnn::memory& mem)
    {
        check_memory<T>(mem);
        auto ptr =mem.pointer<T>();
        auto it = ptr.begin();

        for (auto x : _input_vector)
        {
            *it++ = (T)x;
        }
    }

private:
    uint32_t _seq_length;
    uint32_t _batch;
    uint32_t _predictions_count;
    float    _temperature;
    std::mt19937 random_engine;
    std::vector<char> _int_to_char;
    std::vector<float> _clip; //used in lstm 
    std::vector<char> _correct_length_sequence; // proper sized input sequence
    std::deque<int> _input; //represent current processed input sequence
    std::vector<char> _input_sequence; //sequence from input file
    std::vector<float> _input_vector; //this is actual input_data
    std::vector<std::vector<float>> _output_vector;
    std::vector<std::vector<char>> _seed;
    std::vector<std::vector<std::pair<char, float>>> _predictions;

    void load_input_sequence(std::vector<std::string> input_files);
    void load_vocabulary_file(const std::string& vocab_file);
    void prepare_processing_input_vector();
    void prepare_input_vector_for_new_prediction();
    std::pair<size_t, float> get_predicted_character_and_its_value(size_t batch_index);
    char random_char();
    template<typename MemElemTy = float>
    void check_memory(const cldnn::memory& mem)
    {
        auto memory_layout = mem.get_layout();
        if (memory_layout.format != cldnn::format::bfyx) throw std::runtime_error("Only bfyx format is supported as input for text files.");

        if (!cldnn::data_type_match<MemElemTy>(memory_layout.data_type))
            throw std::runtime_error("Memory format expects different type of elements than specified");
    }
    template<typename MemElemTy>
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



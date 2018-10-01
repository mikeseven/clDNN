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

void lstm_utils::pre_processing(std::vector<std::string> input_files, const std::string & vocab_file)
{
    load_vocabulary_file(vocab_file);
    load_input_sequence(input_files);
    prepare_processing_input_vector();
    prepare_input_vector_for_new_prediction();
    _predictions.resize(_batch);
}

void lstm_utils::prepare_input_vector_for_new_prediction()
{
    _input_vector = std::vector<float>(_seq_length * _batch, 0.0f);
    for (uint32_t j = 0; j < _seq_length * _batch; ++j)
    {
        _input_vector.at(j) = static_cast<float>(_input.at(j));
    }
}

std::pair<unsigned, float> lstm_utils::get_predicted_character_and_its_value(size_t batch_index)
{
    auto& out_vector_batch = _output_vector.at(batch_index);
    auto max_element = std::max_element(out_vector_batch.begin(), out_vector_batch.end()); //iterator
    if (_temperature == 0.0f)
    {
        auto result = std::distance(out_vector_batch.begin(), max_element);
        return{ result, *max_element };
    } 
	else //else if temparture (0.0f, 1.0f>
	{
        std::vector<float> proba(out_vector_batch.size());
        std::vector<float> accumulated_proba(out_vector_batch.size());
        bool found_predicted_char = false;
		for (int i = 0; i < out_vector_batch.size(); ++i)
		{
			proba[i] = exp((out_vector_batch[i] - *max_element) / _temperature);
		}

		float expo_sum = std::accumulate(proba.begin(), proba.end(), 0.0f);

		proba[0] /= expo_sum;
		accumulated_proba[0] = proba[0];
		float random_number = std::uniform_real_distribution<float>(0.0f, 1.0f)(random_engine);

		for (int i = 1; i < proba.size(); ++i)
		{
			//We shouuld return the first number, which probability is higher then random number. 
			if (accumulated_proba[i - 1] > random_number)
			{
                return{ (i - 1), accumulated_proba[i - 1] };
            }
			proba[i] /= expo_sum;
			accumulated_proba[i] = accumulated_proba[i - 1] + proba[i];
		}
        //Not found number? Return last number.
        return{ (proba.size() - 1), accumulated_proba.back() };
	}
}

char lstm_utils::random_char() { return _int_to_char[std::uniform_int_distribution<int>(0, _int_to_char.size() - 1)(random_engine)]; }

void lstm_utils::load_input_sequence(std::vector<std::string> files)
{
    if (files.size() != 1)
    {
       // throw std::runtime_error("Currently only 1 input file is supported.");
    }
    _seed.resize(files.size());
	auto inp_seq_size = _seq_length;
	for (auto i = 0; i < files.size(); i++) {
		auto input = files.at(i);
		std::ifstream input_file(input.c_str(), std::ios::binary | std::ios::ate);
		int file_size = (int)input_file.tellg();
		input_file.seekg(0, std::ios::beg);
		auto currentSequenceLength = 0;
		char currentChar;
        int dummy_spaces = inp_seq_size - file_size;

        for (currentSequenceLength; currentSequenceLength < dummy_spaces; currentSequenceLength++)
        {
            // we should insert random char here, but for testing purposes leave blank space
            // with random char every time topology generates new output (new stories etc. etc.)
            // with blank space we get same output every time
            _input_sequence.push_back(' '); //_input_sequence.push_back(random_char());
        }

        while ((currentChar = input_file.get()) != EOF)
        {
            _input_sequence.push_back(currentChar);
            _seed.at(i).push_back(currentChar);
            if (_input_sequence.size() - (_seq_length * i)  == _seq_length)
            {
                break;
            }
        }
	}
	if (files.size() < _batch)
	{
		auto dummyDataSize = (_batch - files.size()) * inp_seq_size;
		for (int i = 0; i < dummyDataSize; i++)
		{
			_input_sequence.push_back(' ');
		}
	}

}

void lstm_utils::load_vocabulary_file(const std::string& vocab_file)
{
    std::ifstream label_file(vocab_file.c_str());
    char currentChar;
    while ((currentChar = label_file.get()) != EOF)
    {
        _int_to_char.push_back(currentChar);
    }
}

void lstm_utils::prepare_processing_input_vector()
{
    for (int i = 0; i < _input_sequence.size(); ++i)
    {
        for (int j = 0; j < _int_to_char.size(); ++j)
        {
            if (_input_sequence.at(i) == _int_to_char.at(j))
            {
                _input.push_back(j);
                break;
            }
            if (j == _int_to_char.size()-1)
            {
                std::cerr << "Error, the input sequence contains at least one unknown character: " << _input_sequence.at(j) << std::endl;
                throw;
            }
        }
    }
}
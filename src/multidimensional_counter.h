#pragma once
#include <vector>

size_t calculate_offset(const std::vector<uint32_t> &size, const std::vector<uint32_t> &position);
size_t calculate_offset(const std::vector<int32_t>  &size, const std::vector<int32_t>  &position);
bool counter_finished(const std::vector<uint32_t> &size, const std::vector<uint32_t> &counter);
bool counter_finished(const std::vector<int32_t>  &size, const std::vector<int32_t>  &counter);
void counter_increase(const std::vector<uint32_t> &size, std::vector<uint32_t> &counter);
void counter_increase(const std::vector<int32_t>  &size, std::vector<int32_t>  &counter);
void counter_increase(const std::vector<int32_t>  &size, std::vector<int32_t>  &counter, std::vector<int32_t> &val);
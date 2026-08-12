// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "onnxruntime_c_api.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "ep_utils.h"
#include "parse_string.h"
#include "provider_options.h"

enum class TensorrtRtxWeightStreamingBudgetMode
{
    Disabled,
    Automatic,
    MinimumVram,
    Bytes,
    Percent,
};

struct TensorrtRtxWeightStreamingBudget
{
    TensorrtRtxWeightStreamingBudgetMode mode{TensorrtRtxWeightStreamingBudgetMode::Disabled};
    int64_t bytes{0};
    double percent{0.0};
    std::string requested_value{"0"};

    bool IsEnabled() const
    {
        return mode != TensorrtRtxWeightStreamingBudgetMode::Disabled;
    }
};

constexpr std::string_view kValidWeightStreamingBudgetUnitSuffixes = "B, K, M, G";

inline bool IsAsciiAlpha(char c)
{
    return std::isalpha(static_cast<unsigned char>(c)) != 0;
}

inline OrtStatus* ParseWeightStreamingBudget(const std::string& value_str, TensorrtRtxWeightStreamingBudget& budget)
{
    RETURN_IF_NOT(!value_str.empty(), "Invalid nv_weight_streaming_budget: value must not be empty.");

    const char last_char = value_str.back();
    if (last_char == '%')
    {
        const std::string percent_value = value_str.substr(0, value_str.size() - 1);
        double percent{};
        RETURN_IF_NOT(TryParseStringWithClassicLocale(percent_value, percent) && std::isfinite(percent),
                      "Invalid nv_weight_streaming_budget: ", value_str,
                      ". The weight streaming percentage must be between 0 and 100.");
        RETURN_IF_NOT(percent >= 0.0 && percent <= 100.0, "Invalid nv_weight_streaming_budget: ", value_str,
                      ". The weight streaming percentage must be between 0 and 100.");

        budget.mode = TensorrtRtxWeightStreamingBudgetMode::Percent;
        budget.percent = percent;
        budget.bytes = 0;
        budget.requested_value = value_str;
        return nullptr;
    }

    if (IsAsciiAlpha(last_char))
    {
        RETURN_IF_NOT(value_str.size() < 2 || !IsAsciiAlpha(value_str[value_str.size() - 2]),
                      "Invalid nv_weight_streaming_budget: ", value_str,
                      ". Invalid unit specifier. Valid base-2 unit suffixes include: ",
                      kValidWeightStreamingBudgetUnitSuffixes, ".");

        const char unit = static_cast<char>(std::toupper(static_cast<unsigned char>(last_char)));
        int64_t multiplier = 0;
        switch (unit)
        {
        case 'B':
            multiplier = 1LL;
            break;
        case 'K':
            multiplier = 1LL << 10;
            break;
        case 'M':
            multiplier = 1LL << 20;
            break;
        case 'G':
            multiplier = 1LL << 30;
            break;
        default:
            RETURN_IF_NOT(false, "Invalid nv_weight_streaming_budget: ", value_str, ". Invalid unit specifier '", unit,
                          "'. Valid base-2 unit suffixes include: ", kValidWeightStreamingBudgetUnitSuffixes, ".");
        }

        const std::string byte_value = value_str.substr(0, value_str.size() - 1);
        double unit_count{};
        RETURN_IF_NOT(TryParseStringWithClassicLocale(byte_value, unit_count) && std::isfinite(unit_count),
                      "Invalid nv_weight_streaming_budget: ", value_str,
                      ". Unit budgets must start with a non-negative number.");
        RETURN_IF_NOT(unit_count >= 0.0, "Invalid nv_weight_streaming_budget: ", value_str,
                      ". Unit budgets must be non-negative.");

        const double byte_budget = unit_count * static_cast<double>(multiplier);
        const double max_int64_exclusive = std::ldexp(1.0, 63);
        RETURN_IF_NOT(byte_budget < max_int64_exclusive, "Invalid nv_weight_streaming_budget: ", value_str,
                      ". Budget exceeds int64 byte range.");

        budget.mode = TensorrtRtxWeightStreamingBudgetMode::Bytes;
        budget.bytes = static_cast<int64_t>(byte_budget);
        budget.percent = 0.0;
        budget.requested_value = value_str;
        return nullptr;
    }

    int64_t legacy_budget{};
    RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, legacy_budget));
    RETURN_IF_NOT(legacy_budget >= -1, "Invalid nv_weight_streaming_budget: ", legacy_budget,
                  ". Valid values are 0 (disabled), -1 (automatic), "
                  "1 (minimum VRAM), an explicit byte budget greater than 1, "
                  "a byte budget with one of the base-2 suffixes ",
                  kValidWeightStreamingBudgetUnitSuffixes, ", or a percentage budget from 0% to 100%.");

    budget.requested_value = value_str;
    budget.percent = 0.0;
    budget.bytes = 0;
    if (legacy_budget == 0)
    {
        budget.mode = TensorrtRtxWeightStreamingBudgetMode::Disabled;
    }
    else if (legacy_budget == -1)
    {
        budget.mode = TensorrtRtxWeightStreamingBudgetMode::Automatic;
    }
    else if (legacy_budget == 1)
    {
        budget.mode = TensorrtRtxWeightStreamingBudgetMode::MinimumVram;
    }
    else
    {
        budget.mode = TensorrtRtxWeightStreamingBudgetMode::Bytes;
        budget.bytes = legacy_budget;
    }

    return nullptr;
}

template <typename TEnum>
using EnumNameMapping = std::vector<std::pair<TEnum, std::string>>;

/**
 * Given a mapping and an enumeration value, gets the corresponding name.
 */
template <typename TEnum>
OrtStatus* EnumToName(const EnumNameMapping<TEnum>& mapping, TEnum value, std::string& name)
{
    const auto it = std::find_if(mapping.begin(), mapping.end(),
                                 [&value](const std::pair<TEnum, std::string>& entry)
                                 {
                                     return entry.first == value;
                                 });
    RETURN_IF(it == mapping.end(),
              "Failed to map enum value to name: ", static_cast<typename std::underlying_type<TEnum>::type>(value));
    name = it->second;
    return nullptr;
}

template <typename TEnum>
std::string EnumToName(const EnumNameMapping<TEnum>& mapping, TEnum value)
{
    std::string name;
    THROW_IF_ERROR(EnumToName(mapping, value, name));
    return name;
}

/**
 * Given a mapping and a name, gets the corresponding enumeration value.
 */
template <typename TEnum>
OrtStatus* NameToEnum(const EnumNameMapping<TEnum>& mapping, const std::string& name, TEnum& value)
{
    const auto it = std::find_if(mapping.begin(), mapping.end(),
                                 [&name](const std::pair<TEnum, std::string>& entry)
                                 {
                                     return entry.second == name;
                                 });
    RETURN_IF(it == mapping.end(), "Failed to map enum name to value: ", name);
    value = it->first;
    return nullptr;
}

template <typename TEnum>
TEnum NameToEnum(const EnumNameMapping<TEnum>& mapping, const std::string& name)
{
    TEnum value;
    THROW_IF_ERROR(NameToEnum(mapping, name, value));
    return value;
}

class ProviderOptionsParser
{
public:
    /**
     * Adds a parser for a particular provider option value.
     *
     * @param name The provider option name.
     * @param value_parser An object that parses the option value.
     *        It should be callable with the following signature and return
     *        whether the parsing was successful:
     *            Status value_parser(const std::string&)
     *
     * @return The current ProviderOptionsParser instance.
     */
    template <typename ValueParserType>
    ProviderOptionsParser& AddValueParser(const std::string& name, ValueParserType value_parser)
    {
        ENFORCE(value_parsers_.emplace(name, ValueParser{value_parser}).second, "Provider option \"", name,
                "\" already has a value parser.");
        return *this;
    }

    /**
     * Adds a parser for a particular provider option value which converts a
     * value to the right type and assigns it to the given reference.
     *
     * IMPORTANT: This function stores a reference to the destination variable.
     * The caller must ensure that the reference is valid when Parse() is called!
     *
     * @param name The provider option name.
     * @param dest The destination variable reference.
     *
     * @return The current ProviderOptionsParser instance.
     */
    template <typename ValueType>
    ProviderOptionsParser& AddAssignmentToReference(const std::string& name, ValueType& dest)
    {
        return AddValueParser(name,
                              [&dest](const std::string& value_str) -> OrtStatus*
                              {
                                  return ParseStringWithClassicLocale(value_str, dest);
                              });
    }

    /**
     * Adds a parser for a particular provider option value which maps an
     * enumeration name to a value and assigns it to the given reference.
     *
     * IMPORTANT: This function stores references to the mapping and destination
     * variables. The caller must ensure that the references are valid when
     * Parse() is called!
     *
     * @param name The provider option name.
     * @param mapping The enumeration value to name mapping.
     * @param dest The destination variable reference.
     *
     * @return The current ProviderOptionsParser instance.
     */
    template <typename EnumType>
    ProviderOptionsParser& AddAssignmentToEnumReference(const std::string& name,
                                                        const EnumNameMapping<EnumType>& mapping, EnumType& dest)
    {
        return AddValueParser(name,
                              [&mapping, &dest](const std::string& value_str) -> OrtStatus*
                              {
                                  return NameToEnum(mapping, value_str, dest);
                              });
    }

    /**
     * Parses the given provider options.
     */
    OrtStatus* Parse(const ProviderOptions& options) const
    {
        for (const auto& option : options)
        {
            const auto& name = option.first;
            const auto& value_str = option.second;
            const auto value_parser_it = value_parsers_.find(name);
            RETURN_IF(value_parser_it == value_parsers_.end(), "Unknown provider option: \"", name, "\".");

            const auto parse_status = value_parser_it->second(value_str);
            if (parse_status != nullptr)
            {
                const std::string parse_message = Ort::GetApi().GetErrorMessage(parse_status);
                Ort::GetApi().ReleaseStatus(parse_status);
                RETURN_IF(true, "Failed to parse provider option \"", name, "\": ", parse_message);
            }
        }

        return nullptr;
    }

private:
    using ValueParser = std::function<OrtStatus*(const std::string&)>;
    std::unordered_map<std::string, ValueParser> value_parsers_;
};

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

namespace trt_rtx_ep::profiler_metadata
{

inline bool DecodeJsonString(std::string_view encoded, std::string& decoded)
{
    decoded.clear();
    decoded.reserve(encoded.size());

    for (size_t i = 0; i < encoded.size(); ++i)
    {
        const char ch = encoded[i];
        if (ch != '\\')
        {
            decoded += ch;
            continue;
        }

        if (++i == encoded.size())
        {
            return false;
        }

        switch (encoded[i])
        {
        case '"':
            decoded += '"';
            break;
        case '\\':
            decoded += '\\';
            break;
        case '/':
            decoded += '/';
            break;
        case 'b':
            decoded += '\b';
            break;
        case 'f':
            decoded += '\f';
            break;
        case 'n':
            decoded += '\n';
            break;
        case 'r':
            decoded += '\r';
            break;
        case 't':
            decoded += '\t';
            break;
        case 'u':
        {
            if (i + 4 >= encoded.size())
            {
                return false;
            }

            unsigned value = 0;
            for (size_t digit = 0; digit != 4; ++digit)
            {
                const char hex = encoded[i + 1 + digit];
                value <<= 4;
                if (hex >= '0' && hex <= '9')
                {
                    value += static_cast<unsigned>(hex - '0');
                }
                else if (hex >= 'a' && hex <= 'f')
                {
                    value += static_cast<unsigned>(hex - 'a' + 10);
                }
                else if (hex >= 'A' && hex <= 'F')
                {
                    value += static_cast<unsigned>(hex - 'A' + 10);
                }
                else
                {
                    return false;
                }
            }
            i += 4;

            if (value >= 0xd800 && value <= 0xdbff)
            {
                // High surrogate: must be immediately followed by a \uDC00-\uDFFF low surrogate.
                if (i + 6 >= encoded.size() || encoded[i + 1] != '\\' || encoded[i + 2] != 'u')
                {
                    return false;
                }

                unsigned low = 0;
                for (size_t digit = 0; digit != 4; ++digit)
                {
                    const char hex = encoded[i + 3 + digit];
                    low <<= 4;
                    if (hex >= '0' && hex <= '9')
                    {
                        low += static_cast<unsigned>(hex - '0');
                    }
                    else if (hex >= 'a' && hex <= 'f')
                    {
                        low += static_cast<unsigned>(hex - 'a' + 10);
                    }
                    else if (hex >= 'A' && hex <= 'F')
                    {
                        low += static_cast<unsigned>(hex - 'A' + 10);
                    }
                    else
                    {
                        return false;
                    }
                }
                if (low < 0xdc00 || low > 0xdfff)
                {
                    return false;
                }
                i += 6;

                const unsigned codepoint = 0x10000 + ((value - 0xd800) << 10) + (low - 0xdc00);
                decoded += static_cast<char>(0xf0 | (codepoint >> 18));
                decoded += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
                decoded += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
                decoded += static_cast<char>(0x80 | (codepoint & 0x3f));
            }
            else if (value >= 0xdc00 && value <= 0xdfff)
            {
                // Lone low surrogate with no preceding high surrogate.
                return false;
            }
            else if (value <= 0x7f)
            {
                decoded += static_cast<char>(value);
            }
            else if (value <= 0x7ff)
            {
                decoded += static_cast<char>(0xc0 | (value >> 6));
                decoded += static_cast<char>(0x80 | (value & 0x3f));
            }
            else
            {
                decoded += static_cast<char>(0xe0 | (value >> 12));
                decoded += static_cast<char>(0x80 | ((value >> 6) & 0x3f));
                decoded += static_cast<char>(0x80 | (value & 0x3f));
            }
            break;
        }
        default:
            return false;
        }
    }

    return true;
}

inline bool GetJsonStringField(std::string_view json, std::string_view field, std::string& value)
{
    const std::string key = "\"" + std::string(field) + "\"";
    const size_t key_pos = json.find(key);
    if (key_pos == std::string_view::npos)
    {
        return false;
    }

    size_t value_start = key_pos + key.size();
    while (value_start < json.size() &&
           (json[value_start] == ' ' || json[value_start] == '\t' || json[value_start] == '\n' ||
            json[value_start] == '\r' || json[value_start] == ':'))
    {
        ++value_start;
    }
    if (value_start == json.size() || json[value_start] != '"')
    {
        return false;
    }

    ++value_start;
    size_t value_end = value_start;
    bool escaped = false;
    for (; value_end < json.size(); ++value_end)
    {
        if (!escaped && json[value_end] == '"')
        {
            return DecodeJsonString(json.substr(value_start, value_end - value_start), value);
        }
        escaped = !escaped && json[value_end] == '\\';
    }

    return false;
}

inline std::string ExtractOnnxNodeNames(std::string_view metadata)
{
    constexpr std::string_view prefix = "[ONNX Layer: ";
    std::string onnx_nodes;
    size_t pos = 0;
    while (pos < metadata.size())
    {
        size_t end = metadata.find_first_of("\x1e\x1f", pos);
        if (end == std::string_view::npos)
        {
            end = metadata.size();
        }

        const std::string_view token = metadata.substr(pos, end - pos);
        if (token.size() > prefix.size() + 1 && token.starts_with(prefix) && token.back() == ']')
        {
            if (!onnx_nodes.empty())
            {
                onnx_nodes += ", ";
            }
            onnx_nodes += token.substr(prefix.size(), token.size() - prefix.size() - 1);
        }
        pos = end == metadata.size() ? end : end + 1;
    }
    return onnx_nodes;
}

}  // namespace trt_rtx_ep::profiler_metadata

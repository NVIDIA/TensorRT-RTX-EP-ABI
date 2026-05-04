#pragma once

#include <optional>
#include <string>

namespace trt_rtx_ep
{

// Decodes the original ORT node id that OrtGraphToProto stores in
// NodeProto::doc_string. Returns nullopt for synthesized helper nodes or any
// foreign / legacy doc_string content that does not parse as a numeric ORT id.
inline std::optional<size_t> TryParseNodeId(const std::string& doc_string)
{
    if (doc_string.empty())
    {
        return std::nullopt;
    }

    try
    {
        return static_cast<size_t>(std::stoull(doc_string));
    }
    catch (...)
    {
        return std::nullopt;
    }
}

}  // namespace trt_rtx_ep

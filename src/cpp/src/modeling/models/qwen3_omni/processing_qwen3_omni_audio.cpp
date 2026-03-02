// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_omni/processing_qwen3_omni_audio.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <openvino/core/except.hpp>

namespace {

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::filesystem::path make_temp_path(const std::string& stem) {
    auto tmp_dir = std::filesystem::temp_directory_path();
    auto name = stem + "_" + std::to_string(std::rand()) + ".json";
    return tmp_dir / name;
}

std::string shell_quote(const std::string& value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3OmniAudioInputType Qwen3OmniAudioProcess::classify_source(const nlohmann::json& value) {
    if (value.is_array()) {
        return Qwen3OmniAudioInputType::kNumpyArray;
    }
    if (!value.is_string()) {
        return Qwen3OmniAudioInputType::kUnknown;
    }

    const std::string source = value.get<std::string>();
    if (starts_with(source, "data:audio")) {
        return Qwen3OmniAudioInputType::kDataUri;
    }
    if (starts_with(source, "http://") || starts_with(source, "https://")) {
        return Qwen3OmniAudioInputType::kHttpUrl;
    }
    if (starts_with(source, "file://")) {
        return Qwen3OmniAudioInputType::kFileUrl;
    }
    if (!source.empty()) {
        return Qwen3OmniAudioInputType::kLocalPath;
    }

    return Qwen3OmniAudioInputType::kUnknown;
}

std::vector<Qwen3OmniAudioEntry> Qwen3OmniAudioProcess::extract_audio_entries(
    const nlohmann::json& conversations,
    bool use_audio_in_video) {
    std::vector<Qwen3OmniAudioEntry> out;

    auto process_conversation = [&out, use_audio_in_video](const nlohmann::json& conversation) {
        if (!conversation.is_array()) {
            return;
        }

        for (const auto& message : conversation) {
            if (!message.contains("content") || !message.at("content").is_array()) {
                continue;
            }

            for (const auto& element : message.at("content")) {
                const std::string type = element.value("type", "");
                if (type == "audio" && element.contains("audio")) {
                    const auto& value = element.at("audio");
                    Qwen3OmniAudioEntry entry;
                    entry.type = classify_source(value);
                    entry.source = value.is_string() ? value.get<std::string>() : "";
                    entry.from_video = false;
                    out.push_back(std::move(entry));
                }

                if (use_audio_in_video && type == "video" && element.contains("video")) {
                    const auto& value = element.at("video");
                    Qwen3OmniAudioEntry entry;
                    entry.type = classify_source(value);
                    entry.source = value.is_string() ? value.get<std::string>() : "";
                    entry.from_video = true;
                    out.push_back(std::move(entry));
                }
            }
        }
    };

    if (conversations.is_array() && !conversations.empty() && conversations.front().is_object()) {
        process_conversation(conversations);
        return out;
    }

    if (conversations.is_array()) {
        for (const auto& conversation : conversations) {
            process_conversation(conversation);
        }
    }

    return out;
}

nlohmann::json Qwen3OmniAudioProcess::process_audio_info_via_python(
    const nlohmann::json& conversations,
    bool use_audio_in_video,
    const std::string& python_executable) {
    const auto bridge_script = std::filesystem::path(__FILE__).parent_path() / "processing_qwen3_omni_bridge.py";
    if (!std::filesystem::exists(bridge_script)) {
        OPENVINO_THROW("Bridge script not found: ", bridge_script.string());
    }

    const auto input_path = make_temp_path("qwen3_omni_audio_in");
    const auto output_path = make_temp_path("qwen3_omni_audio_out");

    {
        std::ofstream input_file(input_path);
        if (!input_file.is_open()) {
            OPENVINO_THROW("Failed to create temp input file: ", input_path.string());
        }
        input_file << conversations.dump();
    }

    std::string cmd = shell_quote(python_executable) + " " +
                      shell_quote(bridge_script.string()) +
                      " --mode audio --input " + shell_quote(input_path.string()) +
                      " --output " + shell_quote(output_path.string());
    if (use_audio_in_video) {
        cmd += " --use-audio-in-video";
    }

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::filesystem::remove(input_path);
        std::filesystem::remove(output_path);
        OPENVINO_THROW("Python bridge audio process failed with code: ", ret);
    }

    nlohmann::json result;
    {
        std::ifstream output_file(output_path);
        if (!output_file.is_open()) {
            std::filesystem::remove(input_path);
            std::filesystem::remove(output_path);
            OPENVINO_THROW("Failed to read bridge output file: ", output_path.string());
        }
        output_file >> result;
    }

    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);
    return result;
}

nlohmann::json Qwen3OmniAudioProcess::process_audio_features_via_python(
    const nlohmann::json& conversations,
    const std::string& model_dir,
    bool use_audio_in_video,
    const std::string& python_executable) {
    const auto bridge_script = std::filesystem::path(__FILE__).parent_path() / "processing_qwen3_omni_bridge.py";
    if (!std::filesystem::exists(bridge_script)) {
        OPENVINO_THROW("Bridge script not found: ", bridge_script.string());
    }

    const auto input_path = make_temp_path("qwen3_omni_audio_features_in");
    const auto output_path = make_temp_path("qwen3_omni_audio_features_out");

    {
        std::ofstream input_file(input_path);
        if (!input_file.is_open()) {
            OPENVINO_THROW("Failed to create temp input file: ", input_path.string());
        }
        input_file << conversations.dump();
    }

    std::string cmd = shell_quote(python_executable) + " " +
                      shell_quote(bridge_script.string()) +
                      " --mode audio-features --input " + shell_quote(input_path.string()) +
                      " --output " + shell_quote(output_path.string());
    if (use_audio_in_video) {
        cmd += " --use-audio-in-video";
    }
    if (!model_dir.empty()) {
        cmd += " --model-dir " + shell_quote(model_dir);
    }

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::filesystem::remove(input_path);
        std::filesystem::remove(output_path);
        OPENVINO_THROW("Python bridge audio feature process failed with code: ", ret);
    }

    nlohmann::json result;
    {
        std::ifstream output_file(output_path);
        if (!output_file.is_open()) {
            std::filesystem::remove(input_path);
            std::filesystem::remove(output_path);
            OPENVINO_THROW("Failed to read bridge output file: ", output_path.string());
        }
        output_file >> result;
    }

    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);
    return result;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

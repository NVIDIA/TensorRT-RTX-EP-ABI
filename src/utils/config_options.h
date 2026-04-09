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

#include <string>
#include <unordered_map>

// Configuration options class for session configuration
class ConfigOptions {
 public:
  ConfigOptions() = default;
  ConfigOptions(const std::unordered_map<std::string, std::string>& config) : config_(config) {}

  // Get a configuration value or return the default if not found
  std::string GetConfigOrDefault(const std::string& key, const std::string& default_value) const {
    auto it = config_.find(key);
    if (it != config_.end()) {
      return it->second;
    }
    return default_value;
  }

  // Set a configuration value
  void SetConfig(const std::string& key, const std::string& value) {
    config_[key] = value;
  }

  // Check if a configuration key exists
  bool HasConfig(const std::string& key) const {
    return config_.find(key) != config_.end();
  }

  // Get the underlying map
  const std::unordered_map<std::string, std::string>& GetConfigs() const {
    return config_;
  }

 private:
  std::unordered_map<std::string, std::string> config_;
};
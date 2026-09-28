// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/barney-common.h"

namespace BARNEY_NS {
  namespace native {
    
    struct FromEnv {
      static bool enabled(const std::string &key);
      
      /*! allows for querying whether a value _was_ set _and_ set to
        false. E.g, 'denoising=0' will return true for
        explicitDisabled("denosing"); "denoising=1' would return false
        (because it's _en_abled, not disabled), and 'denoising' not
        set at all would return false (because it hasn't even been
        set, and thus not explicitly disabled */
      static bool explicitlyDisabled(const std::string &key);

      /*! returns the integer value of a 'key=value' BARNEY_CONFIG
          entry, or 'defaultValue' if the key was not set or its value
          is not an integer (a key without value counts as 1) */
      static int intValue(const std::string &key, int defaultValue);
      
      static void init();
      
      static bool logQueues;
      static bool skipDenoising;
      static bool logConfig;
      static bool logBackend;
      static bool logTopo;
    };

  }
}

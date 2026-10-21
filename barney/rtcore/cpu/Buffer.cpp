// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/cpu/Buffer.h"

namespace BARNEY_NS {
  namespace rtc {

    /*! embree reads the last element of a (shared) vertex buffer with a
        16 byte load, so every buffer gets that much padding; without it
        a buffer which ends at a page boundary (common for large
        allocations on macOS) crashes in rtcSetGeometryBuffer() */
    static const size_t embreePadding = 16;

    Buffer::Buffer(Device *device,
                   size_t numBytes,
                   const void *initMem)
    {
      mem = numBytes?malloc(numBytes+embreePadding):nullptr;
      if (initMem)
        memcpy(mem,initMem,numBytes);
    }
    
    Buffer::~Buffer()
    {
      if (mem) free(mem);
    }

    void Buffer::resize(size_t numBytes)
    {
      if (mem) free(mem);
      mem = numBytes?malloc(numBytes+embreePadding):nullptr;
    }
    
    void *Buffer::getDD() const
    {
      return mem;
    }

    void Buffer::upload(const void *hostPtr,
                        size_t numBytes,
                        size_t ofs)
    {
      memcpy(((uint8_t*)mem) + ofs,hostPtr,numBytes);
    }
      
  }
}

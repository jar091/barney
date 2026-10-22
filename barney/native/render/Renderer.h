// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"
#include "native/common/Texture.h"

namespace BARNEY_NS {
  namespace native {
    
    struct Device;
  
    /*! the base class for _any_ other type of object/actor in the
      barney class hierarchy */
    struct Renderer : public Object {
      typedef std::shared_ptr<Renderer> SP;

      struct DD {
        vec4f              bgColor;
        rtc::TextureObject bgTexture;
        float              ambientRadiance;
        int                pathsPerPixel;
        vec4f              cutPlane;
#if BARNEY_USE_MULTI_SCATTERING
        int                maxVolumeBounces;
        int                volumeMultiScatter;
#endif
        /*! path length limits with the meaning of cycles: number of
          bounces after the camera hit, in total and per kind of
          bounce; -1 means 'not set' (the built-in limits apply). The
          bounce that exceeds a limit is still traced, but only
          collects emission and environment light */
        int                maxBounce;
        int                maxDiffuseBounce;
        int                maxGlossyBounce;
        int                maxTransmissionBounce;
        /*! limits of R+G+B of one light contribution to a path, for
          light arriving at the camera hit (direct) and at later hits
          (indirect); 0 means 'no clamp' */
        float              clampDirectSum;
        float              clampIndirectSum;
      };
    
      Renderer(Context *context);
      virtual ~Renderer() {}

      /*! pretty-printer for printf-debugging */
      std::string toString() const override;

      static SP create(Context *context);

      DD getDD(Device *device) const;
    
      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      bool setObject(const std::string &member,
                     const std::shared_ptr<Object> &value) override;
      bool set1i(const std::string &member, const int &value) override;
      bool set1f(const std::string &member, const float &value) override;
      bool set4f(const std::string &member, const vec4f &value) override;
      /*! @} */
      // ------------------------------------------------------------------

      struct { /* no default values here, they all get set on host anyway */
        vec4f       bgColor;
        vec4f       cutPlane;
        Texture::SP bgTexture;
        int         pathsPerPixel;
        float       ambientRadiance;
        int         crosshairs;
#if BARNEY_USE_MULTI_SCATTERING
        int         maxVolumeBounces = 8;
        int         volumeMultiScatter = 0;
#endif
        int         sampleLimit;
        int         maxBounce = -1;
        int         maxDiffuseBounce = -1;
        int         maxGlossyBounce = -1;
        int         maxTransmissionBounce = -1;
        float       clampDirect = 0.f;
        float       clampIndirect = 0.f;
      } staged;
      vec4f       bgColor         = vec4f(0,0,0,1.f);
      Texture::SP bgTexture       = 0;
      int         pathsPerPixel   = 1;
      float       ambientRadiance = 1.f;
      int         crosshairs      = 0;
      vec4f       cutPlane        = vec4f(0,0,0,-1e30f);
#if BARNEY_USE_MULTI_SCATTERING
      int         maxVolumeBounces = 8;
      int         volumeMultiScatter = 0;
#endif
      // maximum number of samples accumulated:
      int         sampleLimit = 1024;
      /* path length limits and clamps, see DD; -1 and 0: not set */
      int         maxBounce = -1;
      int         maxDiffuseBounce = -1;
      int         maxGlossyBounce = -1;
      int         maxTransmissionBounce = -1;
      float       clampDirect = 0.f;
      float       clampIndirect = 0.f;
    };

  }
}

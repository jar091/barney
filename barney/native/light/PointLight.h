// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/math.h"
#include "native/light/Light.h"

namespace BARNEY_NS {
  namespace native {
    
    struct PointLight : public Light {
      struct DD : public Light::DD {
        inline __rtc_device vec3f radianceTowards(vec3f P) const
        {
          float spot = 1.f;
          if (cosOuter > -1.f) {
            /* spot light cone (ANARI 'spot' light) */
            const float c = dot(normalize(P - position), direction);
            if (c <= cosOuter)
              return vec3f(0.f);
            if (c < cosInner) {
              const float t = (c - cosOuter) / (cosInner - cosOuter);
              spot = t * t * (3.f - 2.f * t);
            }
          }
          if (isnan(intensity))
            return color * power * ONE_OVER_FOUR_PI * spot;
          else
            return color * intensity * spot;
        }
        vec3f position;
        float intensity;
        float power;
        /*! spot light: normalized axis and cosines of the outer and
            inner cone half angles; cosOuter <= -1 means no cone */
        vec3f direction;
        float cosOuter;
        float cosInner;
      };
    
      typedef std::shared_ptr<PointLight> SP;
      PointLight(Context *context,
                 const DevGroup::SP &devices)
        : Light(context,devices)
      {}
    
      DD getDD(const affine3f &instanceXfm) const;
    
      std::string toString() const override { return "PointLight"; }
    
      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      bool set1f(const std::string &member, const float &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      /*! @} */
      // ------------------------------------------------------------------

      vec3f position = vec3f(0.f,0.f,0.f);
      float power = 1.f;
      float intensity = NAN;
      vec3f direction = vec3f(0.f,0.f,-1.f);
      float cosOuter = -2.f;
      float cosInner = -2.f;
    };

  }
}

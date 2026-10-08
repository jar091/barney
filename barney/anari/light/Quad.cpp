// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/light/Quad.h"

#include <cmath>

namespace BARNEY_NS {
  namespace anari {
  
    QuadLight::QuadLight(BarneyGlobalState *s) : Light(s) {}

    void QuadLight::commitParameters()
    {
      Light::commitParameters();
      m_position = getParam<math::float3>("position", math::float3(0.f, 0.f, 0.f));
      m_edge1 = getParam<math::float3>("edge1", math::float3(1.f, 0.f, 0.f));
      m_edge2 = getParam<math::float3>("edge2", math::float3(0.f, 1.f, 0.f));
      m_intensity = getParam<float>("intensity", NAN);
      m_power = getParam<float>("power", NAN);
      m_radiance = getParam<float>("radiance", NAN);
    }

    const char *QuadLight::bnSubtype() const
    {
      return "quad";
    }

    void QuadLight::setBarneyParameters()
    {
      if (!m_bnLight)
        return;
      bnSetVec(m_bnLight, "corner", m_position);
      bnSetVec(m_bnLight, "edge0", m_edge1);
      bnSetVec(m_bnLight, "edge1", m_edge2);
      /* SPEC: 'radiance' is in W/sr/m^2, 'intensity' is the light emitted
         in the direction of the normal in W/sr, and 'power' the total
         emitted power in W of the one-sided Lambertian emitter. Barney's
         emission is radiance. */
      const float area = length(cross(m_edge1, m_edge2));
      float radiance = 1.f;
      if (!std::isnan(m_radiance))
        radiance = m_radiance;
      else if (!std::isnan(m_intensity))
        radiance = (area > 0.f) ? m_intensity / area : 0.f;
      else if (!std::isnan(m_power))
        radiance = (area > 0.f) ? m_power / (float(M_PI) * area) : 0.f;
      bnSetVec(m_bnLight, "emission", m_color*radiance);
      bnCommit(m_bnLight);
    }

  }
} // ::barney_device

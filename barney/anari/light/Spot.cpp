// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/light/Spot.h"

#include <algorithm>
#include <cmath>

namespace BARNEY_NS {
  namespace anari {

    SpotLight::SpotLight(BarneyGlobalState *s) : Light(s) {}

    void SpotLight::commitParameters()
    {
      Light::commitParameters();
      m_position = getParam<math::float3>("position", math::float3(0.f, 0.f, 0.f));
      m_direction = getParam<math::float3>("direction", math::float3(0.f, 0.f, -1.f));
      m_openingAngle = getParam<float>("openingAngle", float(M_PI));
      m_falloffAngle = getParam<float>("falloffAngle", 0.1f);
      m_intensity = getParam<float>("intensity", NAN);
      m_power = getParam<float>("power", 1.f);
    }

    const char *SpotLight::bnSubtype() const
    {
      /* implemented by the native point light with a cone */
      return "point";
    }

    void SpotLight::setBarneyParameters()
    {
      if (!m_bnLight)
        return;
      const float halfAngle = std::clamp(0.5f * m_openingAngle, 0.f, float(M_PI));
      const float innerAngle = std::max(halfAngle - std::max(m_falloffAngle, 0.f), 0.f);
      const float cosOuter = std::cos(halfAngle);
      const float cosInner = std::max(std::cos(innerAngle), cosOuter + 1e-6f);

      /* SPEC: 'power' is spread over the solid angle of the cone */
      float intensity = m_intensity;
      if (std::isnan(intensity)) {
        const float solidAngle = 2.f * float(M_PI) * (1.f - cosOuter);
        intensity = solidAngle > 0.f ? m_power / solidAngle : 0.f;
      }

      bnSetVec(m_bnLight, "position", m_position);
      bnSetVec(m_bnLight, "direction", m_direction);
      bnSetVec(m_bnLight, "color", m_color);
      bnSet1f(m_bnLight, "intensity", intensity);
      bnSet1f(m_bnLight, "power", m_power);
      bnSet1f(m_bnLight, "cosOuter", cosOuter);
      bnSet1f(m_bnLight, "cosInner", cosInner);
      bnCommit(m_bnLight);
    }

  }
} // ::barney_device

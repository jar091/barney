// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "anari/light/Light.h"

namespace BARNEY_NS {
  namespace anari {

    /*! ANARI 'spot' light, a point light with an angular cone */
    struct SpotLight : public Light
    {
      SpotLight(BarneyGlobalState *s);

      void commitParameters() override;

    private:
      const char *bnSubtype() const override;
      void setBarneyParameters() override;

      math::float3 m_position{0.f, 0.f, 0.f};
      math::float3 m_direction{0.f, 0.f, -1.f};

      /*! SPEC: full opening angle of the cone, in radians */
      float m_openingAngle = float(M_PI);
      /*! SPEC: size of the angular falloff region inside the cone */
      float m_falloffAngle = 0.1f;

      /*! SPEC: light emitted along the axis in W/sr */
      float m_intensity = NAN;
      /*! SPEC: overall emitted power in W */
      float m_power = 1.f;
    };

  }
} // ::barney_device

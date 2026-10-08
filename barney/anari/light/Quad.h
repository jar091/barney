// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "anari/light/Light.h"

namespace BARNEY_NS {
  namespace anari {

    struct QuadLight : public Light
    {
      QuadLight(BarneyGlobalState *s);

      void commitParameters() override;

    private:
      const char *bnSubtype() const override;
      void setBarneyParameters() override;

      /*! reference corner of the quad geometry */
      math::float3 m_position{0.f, 0.f, 0.f};

      /*! edge point to the "right" */
      math::float3 m_edge1{1.f, 0.f, 0.f};

      /*! edge point "up" */
      math::float3 m_edge2{0.f, 1.f, 0.f};

      /*! SPEC: the overall amount of light emitted by the light in a
        direction, in W/sr */
      float m_intensity = NAN;
      /*! SPEC: the overall amount of light energy emitted, in W */
      float m_power = NAN;
      /*! SPEC: the amount of light emitted by a point on the light
        source in a direction, in W/sr/m^2 */
      float m_radiance = NAN;
    };

  }
} // ::barney_device

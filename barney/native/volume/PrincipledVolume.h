// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/native/common/barney-common.h"
#include "native/volume/TransferFunction.h"

namespace BARNEY_NS {
  namespace native {
    
    struct Device;

    /*! Direct two-transfer-function volume model. The regular transfer
      function supplies scattering tint and extinction; the emission transfer
      function supplies emission color and mask. */
    struct PrincipledVolumeParams {
      int     enabled = 0;
<<<<<<< HEAD
      /*! Blender ColorRamp mode: the transfer function drives everything.
        sigma = density * tf.w * (scatter+absorption) (tf.w already holds
        1/unitDistance), Le = emissionStrength * emissionColor * tf.rgb *
        tf.w; the scalar itself does not scale the density. */
      int     colorRamp = 0;
      float   density = 1.f;
      vec3f   scatterColor = vec3f(0.8f);
      vec3f   absorptionColor = vec3f(0.f);
      vec3f   emissionColor = vec3f(1.f);
      float   densityThreshold = 0.f;
      float   densityScale = 1.f;
      range1f valueRange = { 0.f, 1.f };
=======
>>>>>>> 4d9b843ff2ffd7e78b8fcd9423c6ea90fea90309
      float   emissionStrength = 0.f;

      struct DD {
        int     enabled;
<<<<<<< HEAD
        int     colorRamp;
        float   density;
        vec3f   scatterColor;
        vec3f   absorptionColor;
        vec3f   emissionColor;
        float   densityThreshold;
        float   densityScale;
        range1f valueRange;
=======
>>>>>>> 4d9b843ff2ffd7e78b8fcd9423c6ea90fea90309
        float   emissionStrength;
      };

      inline DD getDD(Device *device) const
      {
        DD dd;
        dd.enabled = enabled;
<<<<<<< HEAD
        dd.colorRamp = colorRamp;
        dd.density = density;
        dd.scatterColor = scatterColor;
        dd.absorptionColor = absorptionColor;
        dd.emissionColor = emissionColor;
        dd.densityThreshold = densityThreshold;
        dd.densityScale = densityScale;
        dd.valueRange = valueRange;
=======
>>>>>>> 4d9b843ff2ffd7e78b8fcd9423c6ea90fea90309
        dd.emissionStrength = emissionStrength;
        (void)device;
        return dd;
      }
    };

    /*! Radiance added at a real collision. The emission transfer function
        independently controls emission color and mask. */
    inline __rtc_device
    vec3f principledCollisionEmission(float scalar,
                                      const TransferFunction::DD &emissionXf,
                                      int emissionXfActive,
                                      const PrincipledVolumeParams::DD &p)
    {
      if (p.emissionStrength <= 0.f || !emissionXfActive)
        return vec3f(0.f);

<<<<<<< HEAD
    inline __rtc_device
    float principledDensityFactor(float scalar,
                                  const PrincipledVolumeParams::DD &p)
    {
      if (p.colorRamp)
        return p.density;
      float s = clamp(scalar, p.valueRange.lower, p.valueRange.upper);
      if (p.densityThreshold > 0.f && s < p.densityThreshold)
        return 0.f;
      return p.density * max(s, 0.f) * p.densityScale;
    }

    inline __rtc_device
    vec3f principledSigmaS(float scalar,
                           const PrincipledVolumeParams::DD &p)
    {
      return principledDensityFactor(scalar, p) * p.scatterColor;
    }

    inline __rtc_device
    vec3f principledSigmaA(float scalar,
                           const PrincipledVolumeParams::DD &p)
    {
      return principledDensityFactor(scalar, p) * p.absorptionColor;
    }

    inline __rtc_device
    float principledSigmaT(float scalar,
                           const PrincipledVolumeParams::DD &p)
    {
      vec3f sigma_t = principledSigmaS(scalar, p) + principledSigmaA(scalar, p);
      return reduce_max(sigma_t);
    }

    inline __rtc_device
    vec4f principledSampleMap(float scalar,
                              const PrincipledVolumeParams::DD &p)
    {
      vec3f sigma_s = principledSigmaS(scalar, p);
      vec3f sigma_a = principledSigmaA(scalar, p);
      vec3f sigma_t = sigma_s + sigma_a;
      float sigma_t_scalar = reduce_max(sigma_t);
      vec3f tint = vec3f(1.f);
      if (sigma_t_scalar > 0.f) {
        float ms = reduce_max(sigma_s);
        tint = ms > 0.f ? sigma_s / ms : vec3f(1.f);
      }
      return vec4f(tint, sigma_t_scalar);
    }

    inline __rtc_device
    vec3f principledEmission(float scalar,
                             const PrincipledVolumeParams::DD &p)
    {
      float d = principledDensityFactor(scalar, p);
      vec3f Le = p.emissionStrength * p.emissionColor * d;
      if (p.blackbodyIntensity > 0.f && p.temperature > 0.f)
        Le += p.blackbodyIntensity * p.blackbodyTint
          * principledBlackbodyColor(p.temperature) * d;
      return Le;
    }

    /* iw - note this doesn't guarantee that the computed value is
       truly a majorant; it samples the range and returns the maximum
       of these samples; this is maybe/porbably? a godo guess but no
       guarantee that this is truly the majorant - should actually use
       interval aritmethic or similar */
    inline __rtc_device
    float principledMajorant(range1f r,
                             const PrincipledVolumeParams::DD &p)
    {
      if (r.lower > r.upper)
        return 0.f;
      float m = 0.f;
      const int steps = 8;
      for (int i = 0; i <= steps; ++i) {
        float t = float(i) / float(steps);
        float s = r.lower + t * r.span();
        m = max(m, principledSigmaT(s, p));
      }
      return m;
=======
      const vec4f e = emissionXf.map(scalar);
      if (e.w <= 0.f)
        return vec3f(0.f);
      return p.emissionStrength * e.w * getPos(e);
>>>>>>> 4d9b843ff2ffd7e78b8fcd9423c6ea90fea90309
    }

  }
}
